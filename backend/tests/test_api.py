"""Tests for FastAPI endpoints."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.models.base import Base, get_db
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
        home_win_prob=0.42,
        draw_prob=0.28,
        away_win_prob=0.30,
        over25_prob=0.58,
        btts_prob=0.55,
        most_likely_score="1-1",
        confidence="medium",
    )
    db.add(pred)
    db.commit()
    db.close()


class TestRootEndpoint:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "PredictEPL"
        assert data["status"] == "running"


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
        assert data["confidence"] == "medium"

    def test_accuracy_no_data(self, client):
        response = client.get("/api/accuracy")
        assert response.status_code == 200
        assert response.json()["total_evaluated"] == 0
