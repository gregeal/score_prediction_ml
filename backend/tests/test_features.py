"""Tests for feature engineering functions."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.ml.elo import EloSystem
from app.ml.features import (
    MatchFeatures,
    compute_home_form,
    compute_away_form,
    compute_rest_days,
    compute_h2h,
    is_newly_promoted,
    build_match_features,
)


def _make_match(home, away, hg, ag, date_day, status="FINISHED"):
    m = MagicMock()
    m.home_team = home
    m.away_team = away
    m.home_goals = hg
    m.away_goals = ag
    m.status = status
    m.utc_date = datetime(2026, 1, date_day, tzinfo=timezone.utc)
    return m


@pytest.fixture
def matches():
    """10 matches, most recent first (descending date)."""
    return sorted([
        _make_match("Arsenal", "Chelsea", 2, 1, 10),
        _make_match("Chelsea", "Arsenal", 0, 1, 8),
        _make_match("Arsenal", "Liverpool", 1, 1, 6),
        _make_match("Liverpool", "Arsenal", 3, 0, 4),
        _make_match("Arsenal", "ManUtd", 2, 0, 2),
        _make_match("Chelsea", "Liverpool", 1, 2, 9),
        _make_match("Liverpool", "Chelsea", 0, 0, 7),
        _make_match("ManUtd", "Chelsea", 1, 1, 5),
        _make_match("ManUtd", "Liverpool", 0, 2, 3),
        _make_match("Liverpool", "ManUtd", 1, 0, 1),
    ], key=lambda m: m.utc_date, reverse=True)


class TestHomeForm:
    def test_home_form_counts_only_home_games(self, matches):
        form = compute_home_form(matches, "Arsenal", last_n=5)
        # Arsenal home games: vs Chelsea (2-1 W), vs Liverpool (1-1 D), vs ManUtd (2-0 W)
        assert form.last_n == 3
        assert form.wins == 2
        assert form.draws == 1
        assert form.points == 7

    def test_home_form_respects_last_n(self, matches):
        form = compute_home_form(matches, "Arsenal", last_n=2)
        assert form.last_n == 2


class TestAwayForm:
    def test_away_form_counts_only_away_games(self, matches):
        form = compute_away_form(matches, "Arsenal", last_n=5)
        # Arsenal away: at Chelsea (0-1, Arsenal wins), at Liverpool (3-0, Arsenal loses)
        assert form.last_n == 2
        assert form.wins == 1
        assert form.losses == 1


class TestRestDays:
    def test_rest_days_from_most_recent_match(self, matches):
        ref = datetime(2026, 1, 12, tzinfo=timezone.utc)
        days = compute_rest_days(matches, "Arsenal", ref)
        assert days == 2  # Most recent is day 10

    def test_rest_days_no_matches(self, matches):
        ref = datetime(2026, 1, 12, tzinfo=timezone.utc)
        days = compute_rest_days(matches, "UnknownFC", ref)
        assert days == 30


class TestH2H:
    def test_h2h_counts_correctly(self, matches):
        h_wins, draws, a_wins = compute_h2h(matches, "Arsenal", "Chelsea", last_n=3)
        # Arsenal vs Chelsea: Arsenal home 2-1 (H win), Chelsea home 0-1 (A win for Arsenal)
        assert h_wins + draws + a_wins == 2
        assert h_wins == 2  # Arsenal won both meetings
        assert a_wins == 0  # Chelsea won none


class TestNewlyPromoted:
    def test_team_with_few_matches(self, matches):
        assert is_newly_promoted(matches, "UnknownFC", min_matches=10) is True

    def test_team_with_enough_matches(self, matches):
        assert is_newly_promoted(matches, "Arsenal", min_matches=3) is False


class TestBuildMatchFeatures:
    def test_returns_match_features(self, matches):
        elo = EloSystem.from_matches(matches)
        ref = datetime(2026, 1, 12, tzinfo=timezone.utc)
        features = build_match_features(
            matches=matches, home_team="Arsenal", away_team="Chelsea",
            elo_system=elo, dc_home_xg=1.5, dc_away_xg=1.2, reference_date=ref,
        )
        assert isinstance(features, MatchFeatures)
        assert features.dc_home_xg == 1.5
        assert features.elo_diff == features.home_elo - features.away_elo
        assert features.rest_diff == features.home_rest_days - features.away_rest_days

    def test_feature_vector_length(self, matches):
        elo = EloSystem.from_matches(matches)
        ref = datetime(2026, 1, 12, tzinfo=timezone.utc)
        features = build_match_features(
            matches=matches, home_team="Arsenal", away_team="Chelsea",
            elo_system=elo, dc_home_xg=1.5, dc_away_xg=1.2, reference_date=ref,
        )
        vec = features.to_vector()
        assert len(vec) == 19
        assert all(isinstance(v, (int, float)) for v in vec)
