"""Tests for the Dixon-Coles model."""

import numpy as np
import pytest

from app.ml.dixon_coles import DixonColesModel, MatchData, MatchPrediction


def _make_fake_matches() -> list[MatchData]:
    """Create fake match data for testing.

    Simulates a simple 4-team league with clear strength differences:
    - TeamA: strong attack, decent defense
    - TeamB: decent attack, strong defense
    - TeamC: average all-round
    - TeamD: weak team
    """
    matches = []
    results = [
        # TeamA dominates
        ("TeamA", "TeamB", 2, 1),
        ("TeamA", "TeamC", 3, 0),
        ("TeamA", "TeamD", 4, 1),
        ("TeamB", "TeamA", 1, 2),
        # TeamB is solid
        ("TeamB", "TeamC", 2, 0),
        ("TeamB", "TeamD", 3, 1),
        ("TeamC", "TeamB", 0, 1),
        # TeamC is mid-table
        ("TeamC", "TeamD", 2, 1),
        ("TeamD", "TeamC", 1, 2),
        # TeamD struggles
        ("TeamD", "TeamA", 0, 3),
        ("TeamD", "TeamB", 0, 2),
        ("TeamC", "TeamA", 1, 2),
        # Second round
        ("TeamA", "TeamB", 1, 0),
        ("TeamA", "TeamC", 2, 1),
        ("TeamB", "TeamC", 1, 1),
        ("TeamB", "TeamD", 2, 0),
        ("TeamC", "TeamD", 1, 0),
        ("TeamD", "TeamA", 1, 4),
        ("TeamD", "TeamB", 0, 1),
        ("TeamD", "TeamC", 1, 1),
    ]
    for home, away, hg, ag in results:
        matches.append(MatchData(home_team=home, away_team=away, home_goals=hg, away_goals=ag))
    return matches


class TestDixonColesModel:
    def setup_method(self):
        self.model = DixonColesModel()
        self.matches = _make_fake_matches()
        self.model.fit(self.matches)

    def test_fit_produces_params(self):
        assert self.model.params is not None
        assert len(self.model.params.teams) == 4
        assert "TeamA" in self.model.params.attack

    def test_attack_strengths_order(self):
        """TeamA should have highest attack, TeamD lowest."""
        attack = self.model.params.attack
        assert attack["TeamA"] > attack["TeamD"]

    def test_home_advantage_positive(self):
        """Home advantage should be positive."""
        assert self.model.params.home_advantage > 0

    def test_predict_returns_valid_probabilities(self):
        pred = self.model.predict_match("TeamA", "TeamD")

        # Probabilities should sum to ~1
        total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
        assert abs(total - 1.0) < 0.01

        # All probabilities should be between 0 and 1
        assert 0 <= pred.home_win_prob <= 1
        assert 0 <= pred.draw_prob <= 1
        assert 0 <= pred.away_win_prob <= 1
        assert 0 <= pred.over25_prob <= 1
        assert 0 <= pred.btts_prob <= 1

    def test_strong_team_favored_at_home(self):
        """TeamA at home vs TeamD should strongly favor TeamA."""
        pred = self.model.predict_match("TeamA", "TeamD")
        assert pred.home_win_prob > pred.away_win_prob
        assert pred.home_win_prob > 0.5

    def test_score_matrix_shape(self):
        pred = self.model.predict_match("TeamA", "TeamB")
        assert pred.score_matrix.shape == (10, 10)

    def test_score_matrix_sums_to_one(self):
        pred = self.model.predict_match("TeamA", "TeamB")
        assert abs(pred.score_matrix.sum() - 1.0) < 0.001

    def test_top_scores_returned(self):
        pred = self.model.predict_match("TeamA", "TeamB")
        assert len(pred.top_scores) == 5
        # Probabilities should be descending
        probs = [p for _, p in pred.top_scores]
        assert probs == sorted(probs, reverse=True)

    def test_confidence_rating(self):
        pred = self.model.predict_match("TeamA", "TeamD")
        assert pred.confidence in ("high", "medium", "low")

    def test_unknown_team_raises(self):
        with pytest.raises(ValueError, match="Unknown team"):
            self.model.predict_match("TeamA", "NonExistentFC")

    def test_unfitted_model_raises(self):
        fresh_model = DixonColesModel()
        with pytest.raises(ValueError, match="not fitted"):
            fresh_model.predict_match("TeamA", "TeamB")

    def test_most_likely_score_format(self):
        pred = self.model.predict_match("TeamB", "TeamC")
        parts = pred.most_likely_score.split("-")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)
