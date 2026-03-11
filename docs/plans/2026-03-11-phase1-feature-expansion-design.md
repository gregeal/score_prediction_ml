# Phase 1 Design: Feature Expansion + Challenger Model

**Date:** 2026-03-11
**Status:** Approved

## Context

The current Dixon-Coles model uses only team attack/defense strengths with time-weighted historical results. It has no explicit form tracking by venue, no rest-day awareness, no opponent-quality adjustment, and no alternative model to compare against. Phase 1 adds richer features from existing data and a challenger model to improve prediction quality.

## Decisions

- **xG data deferred** to Phase 4 — all features computed from existing match results only.
- **Gradient boosting via scikit-learn** — no new dependencies (already in requirements).
- **Dixon-Coles remains baseline** — challenger adjusts 1X2 probabilities; exact scores/O/U/BTTS still derived from Dixon-Coles Poisson matrix.
- **Auto model selection** by Brier score on held-out data.

## Design

### 1A. Feature Expansion — `backend/app/ml/features.py`

New `MatchFeatures` dataclass for each upcoming match:

```python
@dataclass
class MatchFeatures:
    # Elo
    home_elo: float
    away_elo: float
    elo_diff: float  # home - away

    # Dixon-Coles expected goals
    dc_home_xg: float
    dc_away_xg: float

    # Home team form (last 5 HOME games)
    home_form_ppg: float
    home_form_gf_pg: float  # goals for per game
    home_form_ga_pg: float  # goals against per game

    # Away team form (last 5 AWAY games)
    away_form_ppg: float
    away_form_gf_pg: float
    away_form_ga_pg: float

    # Rest days
    home_rest_days: int
    away_rest_days: int
    rest_diff: int  # home - away

    # H2H (last 3 meetings)
    h2h_home_wins: int
    h2h_draws: int
    h2h_away_wins: int

    # Newly promoted flag
    home_is_promoted: bool
    away_is_promoted: bool
```

New functions:
- `compute_home_form(matches, team, last_n=5)` — Form from home games only.
- `compute_away_form(matches, team, last_n=5)` — Form from away games only.
- `compute_rest_days(matches, team, reference_date)` — Days since last match.
- `compute_h2h(matches, home_team, away_team, last_n=3)` — H2H record.
- `is_newly_promoted(matches, team, min_matches=10)` — Flag for insufficient history.
- `build_match_features(...)` — Assembles a MatchFeatures for a given fixture.

### 1B. Elo Rating System — `backend/app/ml/elo.py` (new)

```python
class EloSystem:
    def __init__(self, k_factor=20, home_advantage=100, default_rating=1500):
        self.ratings: dict[str, float]

    def update(self, home_team, away_team, home_goals, away_goals) -> None
    def get_rating(self, team) -> float
    def expected_score(self, home_team, away_team) -> tuple[float, float]

    @classmethod
    def from_matches(cls, matches: list[Match]) -> "EloSystem":
        """Build Elo from chronological match history."""
```

- K-factor: 20 (standard for football).
- Home advantage: +100 Elo points for home team in expected score calculation.
- Default: 1500 for unknown/newly promoted teams.
- Built once from all finished matches sorted by date.

### 1C. Challenger Model — `backend/app/ml/challenger_model.py` (new)

```python
class ChallengerModel:
    def __init__(self):
        self.classifier = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1
        )
        self.dixon_coles: DixonColesModel  # baseline for xG + score matrix

    def fit(self, matches, elo_system) -> None:
        """Train GBM on features extracted from historical matches."""
        # For each match: build MatchFeatures, label = {0: home, 1: draw, 2: away}
        # Train classifier on feature vectors

    def predict_match(self, home_team, away_team, elo_system, matches) -> MatchPrediction:
        """Predict using GBM probabilities + Dixon-Coles score matrix."""
        # 1. Get Dixon-Coles expected goals + score matrix
        # 2. Build MatchFeatures for this fixture
        # 3. Get GBM 1X2 probabilities
        # 4. Reweight Dixon-Coles score matrix by GBM 1X2 ratios
        # 5. Derive all prediction types from adjusted matrix
```

**Score matrix reweighting:** The GBM gives better 1X2 probabilities. The Dixon-Coles score matrix gives the goal distribution. We rescale:
- Multiply all home-win cells by `(gbm_home_prob / dc_home_prob)`
- Multiply all draw cells by `(gbm_draw_prob / dc_draw_prob)`
- Multiply all away-win cells by `(gbm_away_prob / dc_away_prob)`
- Renormalize to sum to 1

This preserves the Poisson goal structure while improving outcome probabilities.

### 1D. Predictor Updates — `backend/app/services/predictor.py`

- `train_model()` trains both Dixon-Coles and Challenger.
- Both logged to MLflow (separate metrics prefixed `dc_` and `gbm_`).
- `_evaluate_and_log()` runs backtest for both, compares Brier scores.
- `predict_upcoming()` uses whichever model had better eval Brier score.
- Fallback: if challenger has <200 training matches, use Dixon-Coles.

## Files Changed

| File | Change |
|------|--------|
| `backend/app/ml/features.py` | Add home/away form, rest days, H2H, promoted flag, MatchFeatures |
| `backend/app/ml/elo.py` | New: Elo rating system |
| `backend/app/ml/challenger_model.py` | New: GBM challenger model |
| `backend/app/ml/evaluate.py` | Add backtest support for challenger model |
| `backend/app/services/predictor.py` | Train both models, auto-select best |
| `backend/tests/test_model.py` | Add tests for Elo, features, challenger |

## Verification

1. All existing tests continue to pass (20/20).
2. New tests for: Elo updates, home/away form computation, rest days, H2H, challenger model predictions.
3. Both models logged to MLflow with comparable metrics.
4. Challenger Brier score <= Dixon-Coles Brier score on backtest.
5. Docker pipeline: fetch -> train -> predict still works end-to-end.
