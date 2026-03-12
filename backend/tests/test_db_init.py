"""Tests for lightweight database schema reconciliation."""

from sqlalchemy import inspect, text

from app.config import settings
from app.models.base import _make_engine, _make_session_local, ensure_database_ready, get_engine


LEGACY_PREDICTIONS_DDL = """
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    match_api_id INTEGER NOT NULL,
    home_team VARCHAR NOT NULL,
    away_team VARCHAR NOT NULL,
    predicted_home_goals FLOAT NOT NULL,
    predicted_away_goals FLOAT NOT NULL,
    home_win_prob FLOAT NOT NULL,
    draw_prob FLOAT NOT NULL,
    away_win_prob FLOAT NOT NULL,
    over25_prob FLOAT NOT NULL,
    btts_prob FLOAT NOT NULL,
    most_likely_score VARCHAR,
    outcome_score VARCHAR,
    confidence VARCHAR,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""


def _reset_database_caches() -> None:
    ensure_database_ready.cache_clear()
    _make_session_local.cache_clear()
    _make_engine.cache_clear()


def test_ensure_database_ready_adds_missing_prediction_columns(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.sqlite"
    database_url = f"sqlite:///{db_path.as_posix()}"
    original_database_url = settings.database_url

    monkeypatch.setattr(settings, "database_url", database_url)
    _reset_database_caches()

    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text(LEGACY_PREDICTIONS_DDL))

        ensure_database_ready()

        columns = {column["name"] for column in inspect(get_engine()).get_columns("predictions")}
        assert "raw_home_win_prob" in columns
        assert "raw_draw_prob" in columns
        assert "raw_away_win_prob" in columns
        assert "model_name" in columns
        assert "model_version" in columns
        assert "calibration_version" in columns
    finally:
        monkeypatch.setattr(settings, "database_url", original_database_url)
        _reset_database_caches()
