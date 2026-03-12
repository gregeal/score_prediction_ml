# Phase 1: Feature Expansion + Challenger Model — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add richer pre-match features (home/away form, Elo, rest days, H2H), an Elo rating system, and a gradient-boosted challenger model that competes with Dixon-Coles on Brier score.

**Architecture:** Dixon-Coles stays as baseline. New features feed a scikit-learn GBM classifier. The GBM's 1X2 probabilities reweight the Dixon-Coles score matrix. The predictor trains both, picks the winner by Brier score.

**Tech Stack:** Python 3.12, scikit-learn (GradientBoostingClassifier), scipy, numpy, pytest

**Test command:** `cd backend && python -m pytest tests/ -v` (from venv, or `docker compose exec backend python -m pytest tests/ -v`)

**Design doc:** `docs/plans/2026-03-11-phase1-feature-expansion-design.md`

---

### Task 1: Elo Rating System

**Files:**
- Create: `backend/app/ml/elo.py`
- Create: `backend/tests/test_elo.py`

**Step 1: Write the failing tests**

Create `backend/tests/test_elo.py`:

```python
"""Tests for the Elo rating system."""

import pytest
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
        # Give TeamA a higher rating first
        self.elo.update("TeamA", "TeamB", 3, 0)
        high = self.elo.get_rating("TeamA")
        low = self.elo.get_rating("TeamB")
        gap_before = high - low
        # Now draw
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
        # Same-rated teams: home should have higher expected score due to home advantage
        assert home_exp > away_exp

    def test_expected_scores_sum_to_one(self):
        home_exp, away_exp = self.elo.expected_score("TeamA", "TeamB")
        assert abs(home_exp + away_exp - 1.0) < 0.001

    def test_from_matches(self):
        """Test building Elo from a list of Match-like objects."""
        from unittest.mock import MagicMock
        from datetime import datetime, timezone

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
        # TeamA won twice, should have highest rating
        assert elo.get_rating("TeamA") > elo.get_rating("TeamB")
        assert elo.get_rating("TeamA") > elo.get_rating("TeamC")
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_elo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ml.elo'`

**Step 3: Implement Elo system**

Create `backend/app/ml/elo.py`:

```python
"""Elo rating system for football teams."""

from __future__ import annotations


class EloSystem:
    """Standard Elo rating system with home advantage."""

    def __init__(
        self,
        k_factor: float = 20,
        home_advantage: float = 100,
        default_rating: float = 1500,
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.default_rating = default_rating
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.default_rating)

    def expected_score(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected score (0-1) for each team, accounting for home advantage."""
        home_r = self.get_rating(home_team) + self.home_advantage
        away_r = self.get_rating(away_team)
        exp_home = 1.0 / (1.0 + 10.0 ** ((away_r - home_r) / 400.0))
        return exp_home, 1.0 - exp_home

    def update(
        self, home_team: str, away_team: str, home_goals: int, away_goals: int
    ) -> None:
        """Update ratings after a match result."""
        # Ensure teams exist
        if home_team not in self.ratings:
            self.ratings[home_team] = self.default_rating
        if away_team not in self.ratings:
            self.ratings[away_team] = self.default_rating

        exp_home, exp_away = self.expected_score(home_team, away_team)

        # Actual score: 1 for win, 0.5 for draw, 0 for loss
        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals == away_goals:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        self.ratings[home_team] += self.k_factor * (actual_home - exp_home)
        self.ratings[away_team] += self.k_factor * (actual_away - exp_away)

    @classmethod
    def from_matches(cls, matches: list, **kwargs) -> "EloSystem":
        """Build Elo ratings from chronological match history.

        Args:
            matches: Match objects sorted by utc_date ascending.
                     Must have: home_team, away_team, home_goals, away_goals, status.
        """
        elo = cls(**kwargs)
        sorted_matches = sorted(
            [m for m in matches if m.status == "FINISHED" and m.home_goals is not None],
            key=lambda m: m.utc_date,
        )
        for match in sorted_matches:
            elo.update(match.home_team, match.away_team, match.home_goals, match.away_goals)
        return elo
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_elo.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add backend/app/ml/elo.py backend/tests/test_elo.py
git commit -m "feat: add Elo rating system for team strength tracking"
```

---

### Task 2: Expanded Feature Engineering

**Files:**
- Modify: `backend/app/ml/features.py`
- Create: `backend/tests/test_features.py`

**Step 1: Write the failing tests**

Create `backend/tests/test_features.py`:

```python
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
        assert form.last_n == 3  # Only 3 home games exist
        assert form.wins == 2
        assert form.draws == 1
        assert form.points == 7

    def test_home_form_respects_last_n(self, matches):
        form = compute_home_form(matches, "Arsenal", last_n=2)
        assert form.last_n == 2


class TestAwayForm:
    def test_away_form_counts_only_away_games(self, matches):
        form = compute_away_form(matches, "Arsenal", last_n=5)
        # Arsenal away games: at Chelsea (0-1 W for Arsenal), at Liverpool (3-0 L)
        assert form.last_n == 2
        assert form.wins == 1
        assert form.losses == 1


class TestRestDays:
    def test_rest_days_from_most_recent_match(self, matches):
        ref = datetime(2026, 1, 12, tzinfo=timezone.utc)
        days = compute_rest_days(matches, "Arsenal", ref)
        # Arsenal's most recent match is day 10
        assert days == 2

    def test_rest_days_no_matches(self, matches):
        ref = datetime(2026, 1, 12, tzinfo=timezone.utc)
        days = compute_rest_days(matches, "UnknownFC", ref)
        assert days == 30  # default for unknown


class TestH2H:
    def test_h2h_counts_correctly(self, matches):
        h_wins, draws, a_wins = compute_h2h(matches, "Arsenal", "Chelsea", last_n=3)
        # Arsenal vs Chelsea: Arsenal home 2-1 (H win), Chelsea home 0-1 (A win for Arsenal)
        assert h_wins + draws + a_wins == 2  # only 2 meetings exist
        assert h_wins == 1  # Arsenal home win
        assert a_wins == 1  # Arsenal away win (Chelsea home loss)


class TestNewlyPromoted:
    def test_team_with_few_matches(self, matches):
        assert is_newly_promoted(matches, "UnknownFC", min_matches=10) is True

    def test_team_with_enough_matches(self, matches):
        assert is_newly_promoted(matches, "Arsenal", min_matches=3) is False


class TestBuildMatchFeatures:
    def test_returns_match_features(self, matches):
        elo = EloSystem.from_matches(matches)
        dc_home_xg, dc_away_xg = 1.5, 1.2
        ref = datetime(2026, 1, 12, tzinfo=timezone.utc)
        features = build_match_features(
            matches=matches,
            home_team="Arsenal",
            away_team="Chelsea",
            elo_system=elo,
            dc_home_xg=dc_home_xg,
            dc_away_xg=dc_away_xg,
            reference_date=ref,
        )
        assert isinstance(features, MatchFeatures)
        assert features.dc_home_xg == 1.5
        assert features.elo_diff == features.home_elo - features.away_elo
        assert features.rest_diff == features.home_rest_days - features.away_rest_days

    def test_feature_vector_length(self, matches):
        elo = EloSystem.from_matches(matches)
        ref = datetime(2026, 1, 12, tzinfo=timezone.utc)
        features = build_match_features(
            matches=matches,
            home_team="Arsenal",
            away_team="Chelsea",
            elo_system=elo,
            dc_home_xg=1.5,
            dc_away_xg=1.2,
            reference_date=ref,
        )
        vec = features.to_vector()
        assert len(vec) == 19  # All numeric features
        assert all(isinstance(v, (int, float)) for v in vec)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL — `ImportError: cannot import name 'MatchFeatures'`

**Step 3: Implement expanded features**

Add to `backend/app/ml/features.py` (append after existing code):

```python
@dataclass
class MatchFeatures:
    """Feature vector for a single upcoming match."""
    # Elo
    home_elo: float
    away_elo: float
    elo_diff: float

    # Dixon-Coles expected goals
    dc_home_xg: float
    dc_away_xg: float

    # Home team form (last 5 HOME games)
    home_form_ppg: float
    home_form_gf_pg: float
    home_form_ga_pg: float

    # Away team form (last 5 AWAY games)
    away_form_ppg: float
    away_form_gf_pg: float
    away_form_ga_pg: float

    # Rest days
    home_rest_days: int
    away_rest_days: int
    rest_diff: int

    # H2H (last 3 meetings)
    h2h_home_wins: int
    h2h_draws: int
    h2h_away_wins: int

    # Newly promoted
    home_is_promoted: bool
    away_is_promoted: bool

    def to_vector(self) -> list[float]:
        """Convert to numeric feature vector for ML models."""
        return [
            self.home_elo, self.away_elo, self.elo_diff,
            self.dc_home_xg, self.dc_away_xg,
            self.home_form_ppg, self.home_form_gf_pg, self.home_form_ga_pg,
            self.away_form_ppg, self.away_form_gf_pg, self.away_form_ga_pg,
            float(self.home_rest_days), float(self.away_rest_days), float(self.rest_diff),
            float(self.h2h_home_wins), float(self.h2h_draws), float(self.h2h_away_wins),
            float(self.home_is_promoted), float(self.away_is_promoted),
        ]


def compute_home_form(matches: list, team: str, last_n: int = 5) -> TeamForm:
    """Compute form from HOME games only (matches sorted by date descending)."""
    form = TeamForm(team=team, last_n=0)
    for match in matches:
        if match.status != "FINISHED" or match.home_goals is None:
            continue
        if match.home_team != team:
            continue
        if form.last_n >= last_n:
            break
        form.last_n += 1
        gf, ga = match.home_goals, match.away_goals
        form.goals_scored += gf
        form.goals_conceded += ga
        if ga == 0:
            form.clean_sheets += 1
        if gf == 0:
            form.failed_to_score += 1
        if gf > ga:
            form.wins += 1
            form.points += 3
        elif gf == ga:
            form.draws += 1
            form.points += 1
        else:
            form.losses += 1
    return form


def compute_away_form(matches: list, team: str, last_n: int = 5) -> TeamForm:
    """Compute form from AWAY games only (matches sorted by date descending)."""
    form = TeamForm(team=team, last_n=0)
    for match in matches:
        if match.status != "FINISHED" or match.away_goals is None:
            continue
        if match.away_team != team:
            continue
        if form.last_n >= last_n:
            break
        form.last_n += 1
        gf, ga = match.away_goals, match.home_goals
        form.goals_scored += gf
        form.goals_conceded += ga
        if ga == 0:
            form.clean_sheets += 1
        if gf == 0:
            form.failed_to_score += 1
        if gf > ga:
            form.wins += 1
            form.points += 3
        elif gf == ga:
            form.draws += 1
            form.points += 1
        else:
            form.losses += 1
    return form


def compute_rest_days(
    matches: list, team: str, reference_date: datetime, default: int = 30
) -> int:
    """Days since the team's most recent match."""
    for match in matches:
        if match.status != "FINISHED":
            continue
        if match.home_team == team or match.away_team == team:
            match_date = match.utc_date
            if match_date.tzinfo is None:
                match_date = match_date.replace(tzinfo=timezone.utc)
            return (reference_date - match_date).days
    return default


def compute_h2h(
    matches: list, home_team: str, away_team: str, last_n: int = 3
) -> tuple[int, int, int]:
    """Head-to-head record from recent meetings (home perspective).

    Returns:
        (home_team_wins, draws, away_team_wins) across their last_n meetings.
    """
    h_wins, draws, a_wins = 0, 0, 0
    count = 0
    for match in matches:
        if match.status != "FINISHED" or match.home_goals is None:
            continue
        teams = {match.home_team, match.away_team}
        if teams != {home_team, away_team}:
            continue
        if count >= last_n:
            break
        count += 1
        if match.home_team == home_team:
            hg, ag = match.home_goals, match.away_goals
        else:
            hg, ag = match.away_goals, match.home_goals
        if hg > ag:
            h_wins += 1
        elif hg == ag:
            draws += 1
        else:
            a_wins += 1
    return h_wins, draws, a_wins


def is_newly_promoted(matches: list, team: str, min_matches: int = 10) -> bool:
    """Check if a team has too few finished matches in the dataset."""
    count = 0
    for match in matches:
        if match.status != "FINISHED":
            continue
        if match.home_team == team or match.away_team == team:
            count += 1
            if count >= min_matches:
                return False
    return True


def build_match_features(
    matches: list,
    home_team: str,
    away_team: str,
    elo_system,
    dc_home_xg: float,
    dc_away_xg: float,
    reference_date: datetime | None = None,
) -> MatchFeatures:
    """Assemble all features for a single fixture.

    Args:
        matches: All matches sorted by date descending (most recent first).
        home_team: Home team name.
        away_team: Away team name.
        elo_system: EloSystem instance with current ratings.
        dc_home_xg: Dixon-Coles expected home goals.
        dc_away_xg: Dixon-Coles expected away goals.
        reference_date: Date of the fixture (for rest day calculation).
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    home_elo = elo_system.get_rating(home_team)
    away_elo = elo_system.get_rating(away_team)

    hf = compute_home_form(matches, home_team)
    af = compute_away_form(matches, away_team)

    h_rest = compute_rest_days(matches, home_team, reference_date)
    a_rest = compute_rest_days(matches, away_team, reference_date)

    h2h_h, h2h_d, h2h_a = compute_h2h(matches, home_team, away_team)

    return MatchFeatures(
        home_elo=home_elo,
        away_elo=away_elo,
        elo_diff=home_elo - away_elo,
        dc_home_xg=dc_home_xg,
        dc_away_xg=dc_away_xg,
        home_form_ppg=hf.points_per_game,
        home_form_gf_pg=hf.goals_scored_per_game,
        home_form_ga_pg=hf.goals_conceded_per_game,
        away_form_ppg=af.points_per_game,
        away_form_gf_pg=af.goals_scored_per_game,
        away_form_ga_pg=af.goals_conceded_per_game,
        home_rest_days=h_rest,
        away_rest_days=a_rest,
        rest_diff=h_rest - a_rest,
        h2h_home_wins=h2h_h,
        h2h_draws=h2h_d,
        h2h_away_wins=h2h_a,
        home_is_promoted=is_newly_promoted(matches, home_team),
        away_is_promoted=is_newly_promoted(matches, away_team),
    )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features.py -v`
Expected: All tests PASS

**Step 5: Run full suite to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (20 existing + new feature tests)

**Step 6: Commit**

```bash
git add backend/app/ml/features.py backend/tests/test_features.py
git commit -m "feat: add home/away form, rest days, H2H, promoted flags, MatchFeatures"
```

---

### Task 3: Challenger Model (GBM)

**Files:**
- Create: `backend/app/ml/challenger_model.py`
- Create: `backend/tests/test_challenger.py`

**Step 1: Write the failing tests**

Create `backend/tests/test_challenger.py`:

```python
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
    m.utc_date = datetime(2025, 1, day, tzinfo=timezone.utc)
    m.api_id = day
    return m


def _make_league_matches():
    """Create 80+ matches for a 4-team league (enough for GBM training)."""
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    # Strength order: A > B > C > D
    score_map = {
        ("TeamA", "TeamB"): (2, 1), ("TeamA", "TeamC"): (3, 0), ("TeamA", "TeamD"): (4, 0),
        ("TeamB", "TeamA"): (1, 2), ("TeamB", "TeamC"): (2, 0), ("TeamB", "TeamD"): (3, 1),
        ("TeamC", "TeamA"): (0, 2), ("TeamC", "TeamB"): (1, 1), ("TeamC", "TeamD"): (2, 1),
        ("TeamD", "TeamA"): (0, 3), ("TeamD", "TeamB"): (0, 2), ("TeamD", "TeamC"): (1, 2),
    }
    matches = []
    day = 1
    for _round in range(7):  # 7 rounds = 84 matches
        for (h, a), (hg, ag) in score_map.items():
            # Add some noise to scores
            matches.append(_make_match(h, a, hg, ag, min(day, 365)))
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_challenger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ml.challenger_model'`

**Step 3: Implement challenger model**

Create `backend/app/ml/challenger_model.py`:

```python
"""Gradient-boosted challenger model using features + Dixon-Coles base."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from app.ml.dixon_coles import DixonColesModel, MatchData, MatchPrediction
from app.ml.elo import EloSystem
from app.ml.features import (
    MatchFeatures,
    build_match_features,
    matches_to_training_data,
)

logger = logging.getLogger(__name__)


class ChallengerModel:
    """GBM classifier that reweights Dixon-Coles score matrix with richer features."""

    def __init__(self, time_decay_days: int = 365):
        self.classifier = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.dixon_coles = DixonColesModel(time_decay_days=time_decay_days)
        self.is_fitted = False
        self.time_decay_days = time_decay_days

    def fit(self, matches: list, elo_system: EloSystem) -> None:
        """Train the GBM on features extracted from historical matches.

        Args:
            matches: Match objects sorted by utc_date ascending.
            elo_system: Pre-built EloSystem with ratings from these matches.
        """
        # First train Dixon-Coles as our base model
        training_data = matches_to_training_data(matches, use_form_weighting=False)
        if len(training_data) < 50:
            raise ValueError("Need at least 50 matches to train challenger model")
        self.dixon_coles.fit(training_data)

        # Build feature vectors and labels for each finished match
        # We need enough context, so skip early matches
        finished = [m for m in matches if m.status == "FINISHED" and m.home_goals is not None]
        sorted_by_date = sorted(finished, key=lambda m: m.utc_date)

        X, y = [], []
        # Start from match 20+ so we have form history
        for i in range(20, len(sorted_by_date)):
            match = sorted_by_date[i]
            # Use only matches before this one for features
            context = sorted(sorted_by_date[:i], key=lambda m: m.utc_date, reverse=True)

            try:
                dc_home_xg = np.exp(
                    self.dixon_coles.params.attack[match.home_team]
                    + self.dixon_coles.params.defense[match.away_team]
                    + self.dixon_coles.params.home_advantage
                )
                dc_away_xg = np.exp(
                    self.dixon_coles.params.attack[match.away_team]
                    + self.dixon_coles.params.defense[match.home_team]
                )
            except KeyError:
                continue  # Unknown team

            match_date = match.utc_date
            if match_date.tzinfo is None:
                match_date = match_date.replace(tzinfo=timezone.utc)

            features = build_match_features(
                matches=context,
                home_team=match.home_team,
                away_team=match.away_team,
                elo_system=elo_system,
                dc_home_xg=float(dc_home_xg),
                dc_away_xg=float(dc_away_xg),
                reference_date=match_date,
            )
            X.append(features.to_vector())

            # Label: 0=home, 1=draw, 2=away
            if match.home_goals > match.away_goals:
                y.append(0)
            elif match.home_goals == match.away_goals:
                y.append(1)
            else:
                y.append(2)

        if len(X) < 50:
            raise ValueError(f"Only {len(X)} feature samples, need at least 50")

        self.classifier.fit(X, y)
        self.is_fitted = True
        logger.info(f"Challenger model trained on {len(X)} samples")

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        elo_system: EloSystem,
        matches: list,
        reference_date: datetime | None = None,
    ) -> MatchPrediction:
        """Predict using GBM probabilities reweighting Dixon-Coles score matrix.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            elo_system: Current Elo ratings.
            matches: Recent matches sorted by date descending (for feature computation).
            reference_date: Date of the fixture.
        """
        if not self.is_fitted:
            raise ValueError("Challenger model not fitted. Call fit() first.")

        if reference_date is None:
            reference_date = datetime.now(timezone.utc)

        # Get Dixon-Coles base prediction
        dc_pred = self.dixon_coles.predict_match(home_team, away_team)

        # Build features
        features = build_match_features(
            matches=matches,
            home_team=home_team,
            away_team=away_team,
            elo_system=elo_system,
            dc_home_xg=dc_pred.predicted_home_goals,
            dc_away_xg=dc_pred.predicted_away_goals,
            reference_date=reference_date,
        )

        # GBM probabilities [home, draw, away]
        gbm_probs = self.classifier.predict_proba([features.to_vector()])[0]

        # Reweight Dixon-Coles score matrix
        dc_home = dc_pred.home_win_prob
        dc_draw = dc_pred.draw_prob
        dc_away = dc_pred.away_win_prob

        # Avoid division by zero
        eps = 1e-10
        home_ratio = gbm_probs[0] / max(dc_home, eps)
        draw_ratio = gbm_probs[1] / max(dc_draw, eps)
        away_ratio = gbm_probs[2] / max(dc_away, eps)

        matrix = dc_pred.score_matrix.copy()
        n = matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i > j:
                    matrix[i, j] *= home_ratio
                elif i == j:
                    matrix[i, j] *= draw_ratio
                else:
                    matrix[i, j] *= away_ratio

        # Renormalize
        matrix /= matrix.sum()

        # Build prediction from adjusted matrix
        prediction = MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            score_matrix=matrix,
            predicted_home_goals=dc_pred.predicted_home_goals,
            predicted_away_goals=dc_pred.predicted_away_goals,
        )
        DixonColesModel._derive_predictions(prediction, matrix)
        return prediction
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_challenger.py -v`
Expected: All 5 tests PASS

**Step 5: Run full suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add backend/app/ml/challenger_model.py backend/tests/test_challenger.py
git commit -m "feat: add gradient-boosted challenger model with feature reweighting"
```

---

### Task 4: Predictor Orchestration — Train Both Models, Auto-Select

**Files:**
- Modify: `backend/app/services/predictor.py`
- Modify: `backend/app/ml/evaluate.py`

**Step 1: Update evaluate.py to support any model with a predict_match interface**

Add a `backtest_callable` function to `backend/app/ml/evaluate.py`:

```python
def backtest_callable(
    predict_fn,
    test_matches: list[MatchData],
) -> EvaluationResult:
    """Evaluate any prediction function on test matches.

    Args:
        predict_fn: Callable(home_team, away_team) -> MatchPrediction.
        test_matches: Test data.

    Returns:
        EvaluationResult with accuracy metrics.
    """
    # Same logic as backtest() but uses predict_fn instead of model.predict_match
    correct_outcome = 0
    correct_exact = 0
    correct_over25 = 0
    correct_btts = 0
    brier_scores = []
    log_losses = []
    total = 0

    for match in test_matches:
        try:
            pred = predict_fn(match.home_team, match.away_team)
        except (ValueError, KeyError):
            continue

        total += 1
        # ... (same scoring logic as existing backtest)

    if total == 0:
        raise ValueError("No test matches could be evaluated")

    return EvaluationResult(
        total_matches=total,
        outcome_accuracy=round(correct_outcome / total, 4),
        exact_score_accuracy=round(correct_exact / total, 4),
        over25_accuracy=round(correct_over25 / total, 4),
        btts_accuracy=round(correct_btts / total, 4),
        brier_score=round(float(np.mean(brier_scores)), 4),
        avg_log_loss=round(float(np.mean(log_losses)), 4),
    )
```

**Step 2: Update predictor.py to train both models**

Key changes to `backend/app/services/predictor.py`:

- Add imports for `ChallengerModel`, `EloSystem`
- `train_model()`: Train Dixon-Coles, build Elo, train Challenger, compare Brier scores
- `_evaluate_and_log()`: Backtest both models, log metrics prefixed `dc_` and `gbm_`
- `predict_upcoming()`: Use whichever model had better Brier score
- Store `active_model_type` as attribute (either `"dixon_coles"` or `"challenger"`)
- Fallback: use Dixon-Coles if challenger has <200 training samples

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All existing + new tests PASS

**Step 4: Run Docker pipeline end-to-end**

```bash
docker compose exec backend python scripts/fetch_data.py
docker compose exec backend python scripts/train_model.py
```

Expected: Both models train, Brier scores logged to MLflow, predictions generated.

**Step 5: Commit**

```bash
git add backend/app/ml/evaluate.py backend/app/services/predictor.py
git commit -m "feat: train both Dixon-Coles and challenger, auto-select by Brier score"
```

---

## Verification Checklist

- [ ] `python -m pytest tests/ -v` — All tests pass
- [ ] Elo ratings computed for all teams
- [ ] Home/away form computed correctly
- [ ] Challenger model trains without error
- [ ] Both models logged to MLflow
- [ ] Brier scores compared, better model selected
- [ ] Docker pipeline works end-to-end
- [ ] Frontend shows predictions (unchanged API contract)
