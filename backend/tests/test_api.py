"""Tests for FastAPI endpoints."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.predictions as predictions_api
from app.main import app
from app.ml.evaluate import score_prediction
from app.models.base import Base, get_db
from app.models.market_odds import MarketOdds
from app.models.match import Match
from app.models.prediction import Prediction

# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSession = sessionmaker(bind=engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def seed_data():
    """Seed test database with sample matches and predictions."""
    db = TestSession()

    # Add finished match
    match = Match(
        api_id=12345,
        season="2025",
        matchday=28,
        utc_date=datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc),
        status="FINISHED",
        home_team="Arsenal FC",
        away_team="Chelsea FC",
        home_goals=2,
        away_goals=1,
    )
    db.add(match)

    # Add upcoming match
    upcoming = Match(
        api_id=12346,
        season="2025",
        matchday=29,
        utc_date=datetime(2026, 3, 15, 15, 0, tzinfo=timezone.utc),
        status="SCHEDULED",
        home_team="Liverpool FC",
        away_team="Manchester City FC",
    )
    db.add(upcoming)

    # Add prediction for upcoming match
    pred = Prediction(
        match_api_id=12346,
        home_team="Liverpool FC",
        away_team="Manchester City FC",
        predicted_home_goals=1.65,
        predicted_away_goals=1.32,
        raw_home_win_prob=0.40,
        raw_draw_prob=0.27,
        raw_away_win_prob=0.33,
        home_win_prob=0.42,
        draw_prob=0.28,
        away_win_prob=0.30,
        over25_prob=0.58,
        btts_prob=0.55,
        most_likely_score="1-1",
        outcome_score="1-1",
        confidence="medium",
        model_name="challenger",
        model_version="challenger",
        calibration_version="ovr-isotonic-v1",
    )
    db.add(pred)

    finished_pred = Prediction(
        match_api_id=12345,
        home_team="Arsenal FC",
        away_team="Chelsea FC",
        predicted_home_goals=1.85,
        predicted_away_goals=0.92,
        raw_home_win_prob=0.57,
        raw_draw_prob=0.24,
        raw_away_win_prob=0.19,
        home_win_prob=0.54,
        draw_prob=0.25,
        away_win_prob=0.21,
        over25_prob=0.51,
        btts_prob=0.44,
        most_likely_score="2-1",
        outcome_score="2-1",
        confidence="medium",
        model_name="challenger",
        model_version="challenger",
        calibration_version="ovr-isotonic-v1",
    )
    db.add(finished_pred)

    odds = MarketOdds(
        match_api_id=12345,
        source="sports-betting",
        home_win_odds=1.8,
        draw_odds=3.6,
        away_win_odds=4.2,
    )
    db.add(odds)
    db.commit()
    db.close()


class TestRootEndpoint:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "PredictEPL"
        assert data["status"] == "running"

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestFixturesEndpoints:
    def test_upcoming_empty(self, client):
        response = client.get("/api/fixtures/upcoming")
        assert response.status_code == 200
        assert response.json()["count"] == 0

    def test_upcoming_with_data(self, client, seed_data):
        response = client.get("/api/fixtures/upcoming")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        fixture = data["fixtures"][0]
        assert fixture["home"] == "Liverpool FC"
        assert fixture["away"] == "Manchester City FC"
        assert fixture["prediction"] is not None
        assert fixture["prediction"]["confidence"] == "medium"

    def test_standings_empty(self, client):
        response = client.get("/api/standings")
        assert response.status_code == 200
        assert response.json()["standings"] == []

    def test_standings_with_data(self, client, seed_data):
        response = client.get("/api/standings")
        assert response.status_code == 200
        data = response.json()
        assert len(data["standings"]) == 2
        # Arsenal won, so should be first
        assert data["standings"][0]["team"] == "Arsenal FC"
        assert data["standings"][0]["points"] == 3


class TestPredictionEndpoints:
    def test_prediction_not_found(self, client):
        response = client.get("/api/predictions/99999")
        assert response.status_code == 404

    def test_prediction_found(self, client, seed_data):
        response = client.get("/api/predictions/12346")
        assert response.status_code == 200
        data = response.json()
        assert data["match"]["home"] == "Liverpool FC"
        assert data["predictions"]["outcome"]["home_win"] == 0.42
        assert data["raw_predictions"]["outcome"]["home_win"] == 0.4
        assert data["confidence"] == "medium"
        assert data["model"]["name"] == "challenger"

    def test_accuracy_no_data(self, client):
        response = client.get("/api/accuracy")
        assert response.status_code == 200
        assert response.json()["total_evaluated"] == 0

    def test_accuracy_with_dashboard_data(self, client, seed_data):
        response = client.get("/api/accuracy")
        assert response.status_code == 200
        data = response.json()

        assert data["total_evaluated"] == 1
        assert data["outcome_accuracy"] == 1.0
        assert data["summary"]["active_model"] == "challenger"
        assert data["summary"]["calibrated"] is True
        assert "brier_score" in data["summary"]
        assert data["calibration"]["target"] == "predicted_outcome"
        assert isinstance(data["calibration"]["buckets"], list)
        assert isinstance(data["segments"], list)
        assert data["benchmarks"]["model"]["available"] is True
        assert data["benchmarks"]["naive"]["available"] is True
        assert data["benchmarks"]["bookmaker"]["available"] is True

    def test_accuracy_falls_back_to_snapshot_metrics(self, client, monkeypatch):
        db = TestSession()
        db.add(
            Match(
                api_id=20001,
                season="2025",
                matchday=20,
                utc_date=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc),
                status="FINISHED",
                home_team="Arsenal FC",
                away_team="Chelsea FC",
                home_goals=2,
                away_goals=1,
            )
        )
        db.add(
            Match(
                api_id=20002,
                season="2025",
                matchday=21,
                utc_date=datetime(2026, 1, 17, 15, 0, tzinfo=timezone.utc),
                status="FINISHED",
                home_team="Liverpool FC",
                away_team="Manchester City FC",
                home_goals=1,
                away_goals=1,
            )
        )
        db.commit()
        db.close()

        monkeypatch.setattr(
            predictions_api,
            "build_recent_snapshot_predictions",
            lambda *args, **kwargs: [
                score_prediction(
                    predicted_probs=(0.56, 0.24, 0.20),
                    actual_outcome="home",
                    predicted_score="2-1",
                    actual_score="2-1",
                    baseline_probs=(0.34, 0.33, 0.33),
                    bookmaker_probs=(0.49, 0.27, 0.24),
                )
            ],
        )
        monkeypatch.setattr(predictions_api.PredictionService, "load_model", lambda self: None)

        response = client.get("/api/accuracy")
        assert response.status_code == 200
        data = response.json()

        assert data["total_evaluated"] == 1
        assert data["summary"]["evaluation_source"] == "model_snapshot"
        assert "saved model" in data["message"]
        assert data["benchmarks"]["model"]["available"] is True
