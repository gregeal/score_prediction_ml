"""Tests for the gradient-boosted challenger model."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.ml.challenger_model import ChallengerModel
from app.ml.elo import EloSystem


def _make_match(home, away, hg, ag, day, status="FINISHED"):
    m = MagicMock()
    m.home_team = home
    m.away_team = away
    m.home_goals = hg
    m.away_goals = ag
    m.status = status
    m.utc_date = datetime(2025, 1, min(day, 28), tzinfo=timezone.utc)
    m.api_id = day
    return m


def _make_league_matches():
    """Create 84 matches for a 4-team league (enough for GBM training)."""
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    score_map = {
        ("TeamA", "TeamB"): (2, 1), ("TeamA", "TeamC"): (3, 0), ("TeamA", "TeamD"): (4, 0),
        ("TeamB", "TeamA"): (1, 2), ("TeamB", "TeamC"): (2, 0), ("TeamB", "TeamD"): (3, 1),
        ("TeamC", "TeamA"): (0, 2), ("TeamC", "TeamB"): (1, 1), ("TeamC", "TeamD"): (2, 1),
        ("TeamD", "TeamA"): (0, 3), ("TeamD", "TeamB"): (0, 2), ("TeamD", "TeamC"): (1, 2),
    }
    matches = []
    day = 1
    for _round in range(7):
        for (h, a), (hg, ag) in score_map.items():
            matches.append(_make_match(h, a, hg, ag, day))
            day += 1
    return sorted(matches, key=lambda m: m.utc_date)


@pytest.fixture
def league_matches():
    return _make_league_matches()


class TestChallengerModel:
    def test_fit_runs_without_error(self, league_matches):
        model = ChallengerModel()
        elo = EloSystem.from_matches(league_matches)
        model.fit(league_matches, elo)
        assert model.is_fitted

    def test_predict_returns_valid_probabilities(self, league_matches):
        model = ChallengerModel()
        elo = EloSystem.from_matches(league_matches)
        model.fit(league_matches, elo)

        sorted_desc = sorted(league_matches, key=lambda m: m.utc_date, reverse=True)
        pred = model.predict_match("TeamA", "TeamD", elo, sorted_desc)

        total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
        assert abs(total - 1.0) < 0.01
        assert 0 <= pred.home_win_prob <= 1
        assert 0 <= pred.draw_prob <= 1
        assert 0 <= pred.away_win_prob <= 1

    def test_strong_team_favored(self, league_matches):
        model = ChallengerModel()
        elo = EloSystem.from_matches(league_matches)
        model.fit(league_matches, elo)

        sorted_desc = sorted(league_matches, key=lambda m: m.utc_date, reverse=True)
        pred = model.predict_match("TeamA", "TeamD", elo, sorted_desc)
        assert pred.home_win_prob > pred.away_win_prob

    def test_score_matrix_shape(self, league_matches):
        model = ChallengerModel()
        elo = EloSystem.from_matches(league_matches)
        model.fit(league_matches, elo)

        sorted_desc = sorted(league_matches, key=lambda m: m.utc_date, reverse=True)
        pred = model.predict_match("TeamA", "TeamB", elo, sorted_desc)
        assert pred.score_matrix.shape == (10, 10)
        assert abs(pred.score_matrix.sum() - 1.0) < 0.01

    def test_not_fitted_raises(self):
        model = ChallengerModel()
        elo = EloSystem()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict_match("TeamA", "TeamB", elo, [])
