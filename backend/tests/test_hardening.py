"""Regression tests for security, season rollover, and honest evaluation."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import app.seasons as seasons
import app.api.fixtures as fixtures_api
import app.api.predictions as predictions_api
from app.main import app
from app.ml import evaluate
from app.ml.artifacts import load_bundle
from app.ml.dixon_coles import DixonColesModel, MatchData
from app.models.base import Base, get_db
from app.models.match import Match
from app.models.prediction import Prediction
from app.services.prediction_history import eligible_prediction_ids
from app.services.predictor import PredictionService
from app.services.tracking import SafeTracking


@pytest.fixture
def db(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    now = datetime(2026, 9, 6)
    monkeypatch.setattr(seasons, "utc_now", lambda: now)
    monkeypatch.setattr(fixtures_api, "utc_now", lambda: now)
    with Session(engine) as session:
        app.dependency_overrides[get_db] = lambda: session
        yield session
    app.dependency_overrides.clear()
    engine.dispose()


def match(api_id=1, **kwargs):
    data = dict(api_id=api_id, season="2026", utc_date=datetime(2026, 9, 1),
                status="FINISHED", home_team="A", away_team="B", home_goals=2, away_goals=1)
    return Match(**(data | kwargs))


def prediction(**kwargs):
    data = dict(match_api_id=1, home_team="A", away_team="B", created_at=datetime(2026, 8, 31),
                predicted_home_goals=1.5, predicted_away_goals=1.0,
                home_win_prob=.6, draw_prob=.25, away_win_prob=.15,
                over25_prob=.5, btts_prob=.5, most_likely_score="2-1", model_name="dixon_coles")
    return Prediction(**(data | kwargs))


@pytest.mark.parametrize("date,year", [(datetime(2026, 6, 30), 2025), (datetime(2026, 7, 1), 2026),
                                      (datetime(2026, 9, 6), 2026), (datetime(2027, 9, 1), 2027)])
def test_season_rollover(date, year):
    assert seasons.current_season_year(date) == year


def test_current_standings_include_unplayed_teams_not_previous_season(db):
    db.add_all([match(season="2025"), match(2, status="TIMED", home_team="C", away_team="D", home_goals=None, away_goals=None)])
    db.commit()
    result = TestClient(app).get("/api/standings").json()
    assert result["season"] == "2026"
    assert {r["team"] for r in result["standings"]} == {"C", "D"}
    assert all(r["played"] == 0 for r in result["standings"])


def test_incomplete_results_are_not_scored(db):
    db.add_all([match(away_goals=None), prediction()])
    db.commit()
    client = TestClient(app)
    assert client.get("/api/accuracy").json()["total_evaluated"] == 0
    assert all(r["played"] == 0 for r in client.get("/api/standings").json()["standings"])


def test_stale_scheduled_matches_are_not_upcoming(db):
    db.add_all([match(status="TIMED"), match(2, status="TIMED", utc_date=datetime(2026, 9, 10))])
    db.commit()
    result = TestClient(app).get("/api/fixtures/upcoming").json()
    assert [r["match_id"] for r in result["fixtures"]] == [2]


def test_accuracy_uses_latest_eligible_not_latest_inserted(db):
    db.add(match())
    db.add(prediction())
    db.add(prediction(created_at=datetime(2026, 9, 2), home_win_prob=.1, away_win_prob=.65))
    db.add(prediction(created_at=datetime(2026, 8, 20), home_win_prob=.1, away_win_prob=.65))
    db.commit()
    result = TestClient(app).get("/api/accuracy").json()
    assert result["total_evaluated"] == 1
    assert result["outcome_accuracy"] == 1


def test_calibration_filters_model_before_deduplication(db):
    db.add_all([match(), prediction(), prediction(model_name="challenger", created_at=datetime(2026, 8, 31, 12))])
    db.commit()
    ids = eligible_prediction_ids(db, "dixon_coles")
    selected = db.query(Prediction).join(ids, Prediction.id == ids.c.latest_id).one()
    assert selected.model_name == "dixon_coles"


def test_calibrator_does_not_train_on_post_kickoff_rows(db):
    db.add_all([match(), prediction(created_at=datetime(2026, 9, 2))])
    db.commit()
    service = PredictionService(db)
    service._fit_outcome_calibrator()
    assert service.calibrator is None


def test_probability_buckets_do_not_duplicate_low_confidence_rows():
    rows = [evaluate.score_prediction(predicted_probs=p, actual_outcome="home") for p in [(.6, .2, .2), (.95, .03, .02), (1, 0, 0)]]
    buckets = evaluate.build_calibration_buckets(rows)
    assert sum(b.count for b in buckets) == 3
    assert buckets[-1].count == 2


@pytest.mark.parametrize("probs", [(float("nan"), .2, .3), (float("inf"), .2, .3), (-.1, .5, .6), (0, 0, 0), (.5, .5)])
def test_bad_probabilities_rejected(probs):
    with pytest.raises(ValueError):
        evaluate.normalize_probs(probs)


def test_model_market_comparison_uses_same_cohort():
    rows = [evaluate.score_prediction(predicted_probs=(.7, .2, .1), actual_outcome="home", bookmaker_probs=(.6, .2, .2)),
            evaluate.score_prediction(predicted_probs=(.7, .2, .1), actual_outcome="away")]
    result = evaluate.compare_benchmarks(rows)
    assert result["model"].total_matches == 2
    assert result["model_on_market"].total_matches == result["bookmaker"].total_matches == 1
    assert result["model_on_market"].outcome_accuracy == 1


def test_simultaneous_kickoffs_have_identical_prior():
    rows = [match(), match(2, home_goals=0, away_goals=1)]
    priors = predictions_api._league_priors(rows)
    assert priors[1] == priors[2]


def test_backtest_cache_invalidates_result_correction(monkeypatch):
    evaluate._cached_backtest.cache_clear()
    compute = MagicMock(return_value=[])
    monkeypatch.setattr(evaluate, "_build_recent_backtest_predictions", compute)
    row = match()
    evaluate.build_recent_backtest_predictions([row])
    evaluate.build_recent_backtest_predictions([row])
    assert compute.call_count == 1
    row.away_goals = 3
    evaluate.build_recent_backtest_predictions([row])
    assert compute.call_count == 2
    evaluate._cached_backtest.cache_clear()


def test_legacy_pickle_is_never_loaded(tmp_path):
    (tmp_path / "trained_model.pkl").write_bytes(b"untrusted legacy artifact")
    with pytest.raises(FileNotFoundError, match="legacy pickle"):
        load_bundle(tmp_path / "prediction_bundle.skops")


def test_tracking_failure_does_not_mask_application_errors():
    client = MagicMock()
    client.start_run.side_effect = ConnectionError("offline")
    client.log_metric.side_effect = ConnectionError("offline")
    tracking = SafeTracking(client)
    with tracking.start_run():
        tracking.log_metric("brier", .5)
    with pytest.raises(ValueError, match="application"):
        with tracking.start_run():
            raise ValueError("application")


def test_fallback_model_does_not_use_challenger_calibration(db):
    service = PredictionService(db)
    service.active_model = "challenger"
    service.calibrator = MagicMock(is_fitted=True)
    model = DixonColesModel()
    model.fit([MatchData("A", "B", 2, 1), MatchData("B", "A", 1, 0)])
    pred = model.predict_match("A", "B")
    service._apply_outcome_calibration(pred, "dixon_coles")
    service.calibrator.transform.assert_not_called()


def test_security_headers_and_disallowed_origin(db):
    result = TestClient(app).get("/health", headers={"Origin": "https://untrusted.example"})
    assert result.headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" not in result.headers


def test_data_status_distinguishes_unsynced_season_from_historical_data(db):
    db.add(match(season="2025"))
    db.commit()
    client = TestClient(app)
    result = client.get("/api/data-status")
    assert result.status_code == 200
    assert result.headers["cache-control"] == "no-store"
    assert result.json()["current_season"] == "2026"
    assert result.json()["latest_available_season"] == "2025"
    assert result.json()["current_season_matches"] == 0
    assert result.json()["latest_prediction_at"] is None
    db.add_all([match(2), prediction(match_api_id=2)])
    db.commit()
    refreshed = client.get("/api/data-status").json()
    assert refreshed["current_season_matches"] == 1
    assert refreshed["current_season_finished"] == 1
    assert refreshed["latest_prediction_at"].endswith("+00:00")


@pytest.mark.parametrize("isotonic", [False, True])
def test_fitted_bundle_roundtrip(tmp_path, monkeypatch, isotonic):
    from app.services import predictor as module
    from app.ml.calibration import OutcomeCalibrator
    from app.ml.elo import EloSystem
    monkeypatch.setattr(module, "BUNDLE_PATH", tmp_path / "bundle.skops")
    service = PredictionService(MagicMock())
    service.dc_model.fit([MatchData("A", "B", 2, 1), MatchData("B", "A", 0, 0)])
    service.challenger.dixon_coles = service.dc_model
    x = np.random.default_rng(42).normal(size=(60, 19))
    service.challenger.classifier.set_params(n_estimators=2)
    service.challenger.classifier.fit(x, np.arange(60) % 3)
    service.challenger.is_fitted = True
    service.elo_system = EloSystem()
    service.active_model = "challenger"
    service.calibrator = OutcomeCalibrator(min_samples=3, min_class_examples=1, isotonic_min_samples=3 if isotonic else 80)
    service.calibrator.fit([(.8, .1, .1), (.1, .8, .1), (.1, .1, .8)] * 10, ["home", "draw", "away"] * 10)
    service.save_model()
    restored = PredictionService(MagicMock())
    restored.load_model()
    np.testing.assert_allclose(restored.challenger.classifier.predict_proba(x), service.challenger.classifier.predict_proba(x))
    np.testing.assert_allclose(restored.calibrator.transform((.5, .3, .2)), service.calibrator.transform((.5, .3, .2)))


def test_prediction_refresh_preserves_history_and_skips_past_kickoff(db, monkeypatch):
    import app.services.predictor as module
    monkeypatch.setattr(module, "utc_now", lambda: datetime(2026, 9, 6))
    db.add_all([match(status="TIMED", utc_date=datetime(2026, 9, 10)), prediction(), match(2, status="TIMED")])
    db.commit()
    service = PredictionService(db)
    service.dc_model.fit([MatchData("A", "B", 2, 1), MatchData("B", "A", 0, 0)])
    assert len(service.predict_upcoming()) == 1
    assert db.query(Prediction).filter_by(match_api_id=1).count() == 2
    assert db.query(Prediction).filter_by(match_api_id=2).count() == 0
