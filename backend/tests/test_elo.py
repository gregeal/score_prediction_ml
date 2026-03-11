"""Tests for the Elo rating system."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from app.ml.elo import EloSystem


class TestEloSystem:
    def setup_method(self):
        self.elo = EloSystem(k_factor=20, home_advantage=100, default_rating=1500)

    def test_default_rating(self):
        assert self.elo.get_rating("Unknown FC") == 1500

    def test_home_win_increases_home_rating(self):
        before_home = self.elo.get_rating("TeamA")
        before_away = self.elo.get_rating("TeamB")
        self.elo.update("TeamA", "TeamB", 2, 0)
        assert self.elo.get_rating("TeamA") > before_home
        assert self.elo.get_rating("TeamB") < before_away

    def test_away_win_increases_away_rating(self):
        self.elo.update("TeamA", "TeamB", 0, 3)
        assert self.elo.get_rating("TeamB") > self.elo.get_rating("TeamA")

    def test_draw_moves_ratings_toward_each_other(self):
        self.elo.update("TeamA", "TeamB", 3, 0)
        gap_before = self.elo.get_rating("TeamA") - self.elo.get_rating("TeamB")
        self.elo.update("TeamA", "TeamB", 1, 1)
        gap_after = self.elo.get_rating("TeamA") - self.elo.get_rating("TeamB")
        assert gap_after < gap_before

    def test_ratings_are_zero_sum(self):
        self.elo.update("TeamA", "TeamB", 2, 1)
        self.elo.update("TeamC", "TeamD", 0, 0)
        total = sum(self.elo.ratings.values())
        expected = 1500 * len(self.elo.ratings)
        assert abs(total - expected) < 0.01

    def test_expected_score_home_advantage(self):
        home_exp, away_exp = self.elo.expected_score("TeamA", "TeamA")
        assert home_exp > away_exp

    def test_expected_scores_sum_to_one(self):
        home_exp, away_exp = self.elo.expected_score("TeamA", "TeamB")
        assert abs(home_exp + away_exp - 1.0) < 0.001

    def test_from_matches(self):
        matches = []
        for i, (h, a, hg, ag) in enumerate([
            ("TeamA", "TeamB", 2, 0),
            ("TeamC", "TeamD", 1, 1),
            ("TeamA", "TeamC", 3, 1),
        ]):
            m = MagicMock()
            m.home_team = h
            m.away_team = a
            m.home_goals = hg
            m.away_goals = ag
            m.status = "FINISHED"
            m.utc_date = datetime(2025, 1, 1 + i, tzinfo=timezone.utc)
            matches.append(m)

        elo = EloSystem.from_matches(matches)
        assert elo.get_rating("TeamA") > elo.get_rating("TeamB")
        assert elo.get_rating("TeamA") > elo.get_rating("TeamC")
