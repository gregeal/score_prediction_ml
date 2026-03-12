"""Regression tests for the fixture sync script."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
from app.models.match import Match
from scripts.fetch_data import sync_parsed_matches


TEST_DATABASE_URL = "sqlite:///./test_fetch_data.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSession = sessionmaker(bind=engine)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def _match_payload(**overrides):
    payload = {
        "api_id": 1001,
        "season": "2025",
        "matchday": 10,
        "utc_date": datetime(2026, 3, 20, 15, 0),
        "status": "SCHEDULED",
        "home_team": "Arsenal FC",
        "away_team": "Chelsea FC",
        "home_goals": None,
        "away_goals": None,
    }
    payload.update(overrides)
    return payload


def test_sync_updates_rescheduled_fixture_metadata():
    db = TestSession()
    try:
        db.add(Match(**_match_payload()))
        db.commit()

        added, updated = sync_parsed_matches(
            db,
            [
                _match_payload(
                    matchday=11,
                    utc_date=datetime(2026, 3, 21, 17, 30),
                    status="TIMED",
                )
            ],
        )
        db.commit()

        match = db.query(Match).filter_by(api_id=1001).one()
        assert (added, updated) == (0, 1)
        assert match.matchday == 11
        assert match.utc_date == datetime(2026, 3, 21, 17, 30)
        assert match.status == "TIMED"
    finally:
        db.close()


def test_sync_updates_corrected_scores_when_only_away_goals_change():
    db = TestSession()
    try:
        db.add(
            Match(
                **_match_payload(
                    status="FINISHED",
                    home_goals=2,
                    away_goals=0,
                )
            )
        )
        db.commit()

        added, updated = sync_parsed_matches(
            db,
            [
                _match_payload(
                    status="FINISHED",
                    home_goals=2,
                    away_goals=1,
                )
            ],
        )
        db.commit()

        match = db.query(Match).filter_by(api_id=1001).one()
        assert (added, updated) == (0, 1)
        assert match.home_goals == 2
        assert match.away_goals == 1
    finally:
        db.close()


def test_sync_clears_scores_when_upstream_removes_result():
    db = TestSession()
    try:
        db.add(
            Match(
                **_match_payload(
                    status="FINISHED",
                    home_goals=1,
                    away_goals=1,
                )
            )
        )
        db.commit()

        added, updated = sync_parsed_matches(
            db,
            [
                _match_payload(
                    status="POSTPONED",
                    home_goals=None,
                    away_goals=None,
                )
            ],
        )
        db.commit()

        match = db.query(Match).filter_by(api_id=1001).one()
        assert (added, updated) == (0, 1)
        assert match.status == "POSTPONED"
        assert match.home_goals is None
        assert match.away_goals is None
    finally:
        db.close()
