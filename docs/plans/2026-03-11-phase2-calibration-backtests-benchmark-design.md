# Phase 2: Probability Calibration, Rolling Backtests, and Benchmarking

## Status

Approved assumptions captured. Ready for implementation.

## Goal

Improve trust in the product by making probabilities better calibrated, making evaluation more temporally realistic, and showing whether the model adds value over simpler baselines.

## Current State

- `backend/app/ml/evaluate.py` supports a single holdout backtest and returns aggregate metrics only.
- `backend/app/services/predictor.py` can train and select between Dixon-Coles and the challenger model, but it does not persist calibration artifacts or rolling evaluation outputs.
- `backend/app/models/prediction.py` stores only the current predicted probabilities and does not distinguish raw vs calibrated probabilities.
- `backend/app/api/predictions.py` exposes a simple `/api/accuracy` summary with outcome/exact-score/O-U/BTTS accuracy.
- `frontend/app/accuracy/page.tsx` renders summary cards only.
- There is no bookmaker odds model or benchmark ingestion path in the current codebase.

## Confirmed Decisions

These Phase 2 decisions are now locked in:

1. Bookmaker benchmarking should use the `sports-betting` Python package as the primary integration candidate for historical and fixtures odds data, rather than a manual CSV-only path.
2. Calibration should focus on `1X2` probabilities first. `Over/Under 2.5` and `BTTS` will still be evaluated, but not post-hoc calibrated in the first Phase 2 implementation.
3. Rolling backtests should be exposed in both the backend/API layer and the frontend accuracy dashboard in this phase.

## Rejected Options

- `OpenML-CC18` is useful for standardized ML benchmarks across generic tabular datasets, but it is not designed to benchmark bookmaker odds against an EPL-specific forecasting product.
- `MLBench` is a distributed machine learning benchmark framework, which is also outside the scope of product-facing sports probability evaluation.

These can still be useful in a separate research workflow later, but they are not part of the implementation plan for this repo.

## Scope

### In Scope

- Post-hoc calibration for `home/draw/away` probabilities
- Walk-forward rolling backtests with no future leakage
- Benchmarking against a naive baseline and bookmaker odds when available from the `sports-betting` integration path
- A richer `/api/accuracy` response
- Accuracy dashboard updates for calibration, rolling trends, and benchmarks

### Out of Scope

- Live bookmaker API integration
- xG or shots integration
- Injury or lineup data
- Full odds-driven betting features or value-bet recommendations
- Calibration of exact-score distributions

## Proposed Design

### 1. Calibration Layer

Create a new module: `backend/app/ml/calibration.py`

Responsibilities:

- Fit a multiclass outcome calibrator using out-of-sample predictions only
- Transform `home/draw/away` probabilities while preserving a valid probability simplex
- Persist and reload the calibration artifact alongside model artifacts

Proposed approach:

- Use one-vs-rest calibration for each outcome bucket (`home`, `draw`, `away`)
- Prefer isotonic regression when there are enough historical samples
- Fall back to sigmoid-style calibration if the sample size is too small
- Renormalize calibrated outputs so they sum to `1.0`

Why this approach:

- It works for both Dixon-Coles and challenger outputs
- It does not require changing the core model internals
- It can be trained from walk-forward predictions, which keeps the calibration set temporally honest

Artifact:

- `backend/outcome_calibrator.pkl`

Predictor changes:

- `PredictionService.train_model()` should fit the calibrator after active-model selection using out-of-sample walk-forward predictions from the chosen model
- `PredictionService.load_model()` should restore the calibrator if present
- `PredictionService.predict_upcoming()` should:
  - generate raw probabilities from the active model
  - apply calibration if a fitted calibrator exists
  - persist both raw and served probabilities

### 2. Rolling Backtest Engine

Refactor `backend/app/ml/evaluate.py` from a single holdout helper into a reusable evaluation engine.

Add data structures:

- `EvaluatedPrediction`
- `CalibrationBucket`
- `SegmentMetrics`
- `RollingWindowMetrics`
- `BenchmarkMetrics`
- `AccuracyDashboardResult`

Add functions:

- `score_prediction(...)`
- `evaluate_predictions(...)`
- `build_calibration_buckets(...)`
- `build_segment_metrics(...)`
- `walk_forward_backtest(...)`
- `compare_benchmarks(...)`

Walk-forward rules:

- Sort all finished matches chronologically by `utc_date`
- Use an expanding training window
- Default minimum training size: `200` finished matches
- Default evaluation window size: `20` matches
- Default step size: `20` matches
- Retrain fresh model state for each window
- Rebuild Elo from training-window matches only
- Never use test-window matches for feature context, calibration fitting, or benchmark priors

Outputs:

- Per-match evaluated rows for aggregate analysis
- Per-window metrics for trend charts
- Aggregate summary metrics for the dashboard and MLflow

### 3. Benchmarking

Add a benchmark abstraction with three levels:

1. `Model`: active model probabilities, with raw and calibrated metrics available
2. `Naive baseline`: training-period league prior for `home/draw/away`
3. `Bookmaker`: optional implied probabilities derived from imported odds

Why not "home team always wins":

- It is useful as a sanity check, but it is weaker and less stable than a league-prior baseline
- We can still include a simple heuristic row later if wanted, but the league-prior baseline is the better default benchmark

Bookmaker design:

Create a new model: `backend/app/models/market_odds.py`

Suggested fields:

- `id`
- `match_api_id`
- `source`
- `captured_at`
- `home_win_odds`
- `draw_odds`
- `away_win_odds`
- `over25_odds`
- `under25_odds`
- `btts_yes_odds`
- `btts_no_odds`

Supporting code:

- `backend/app/services/odds_provider.py`
- `backend/scripts/fetch_market_odds.py`
- `backend/app/ml/benchmarks.py` or benchmark helpers inside `evaluate.py`

Odds provider design:

- Wrap `sports-betting` behind a small adapter interface so the rest of the app is not coupled directly to package-specific objects
- Persist normalized odds rows into `market_odds`
- Match odds to fixtures using `match_api_id` when possible, and fall back to `(utc_date, home_team, away_team)` reconciliation if the upstream package does not expose the football-data.org IDs
- Record `source`, `captured_at`, and normalization status for auditability

Why an adapter layer matters:

- `sports-betting` is a separate ecosystem with its own data-loading conventions and identifiers
- An adapter keeps the ingestion boundary narrow and lets us swap the source later without rewriting evaluation or API code

Bookmaker logic:

- Convert decimal odds into implied probabilities
- Remove overround by normalizing probabilities to sum to `1.0`
- Evaluate bookmaker probabilities with the same Brier/log-loss functions used for the model
- If no bookmaker odds are available for a given evaluation slice, return `bookmaker_available: false` instead of failing the dashboard
- Keep the naive baseline as a first-class benchmark even when bookmaker odds are available

### 4. Prediction Storage Changes

Extend `backend/app/models/prediction.py`

Add fields:

- `raw_home_win_prob`
- `raw_draw_prob`
- `raw_away_win_prob`
- `model_name`
- `model_version`
- `calibration_version`

Column behavior:

- Existing `home_win_prob`, `draw_prob`, `away_win_prob` become the served probabilities
- When a calibrator is available, these are the calibrated outputs
- Raw model outputs are preserved in the new `raw_*` columns for audit and benchmark comparison

Notes:

- The repo currently does not appear to have a checked-in Alembic migration workflow even though Alembic is installed. For this phase, schema changes can still be implemented and tested through the SQLAlchemy models, but we should either add a migration script or document the required table rebuild before production deployment.

### 5. Accuracy API

Extend `GET /api/accuracy` in `backend/app/api/predictions.py`

Keep the current top-level summary fields for compatibility, then add nested sections:

```json
{
  "total_evaluated": 214,
  "outcome_accuracy": 0.57,
  "exact_score_accuracy": 0.11,
  "over_under_accuracy": 0.63,
  "btts_accuracy": 0.6,
  "summary": {
    "active_model": "challenger",
    "calibrated": true,
    "brier_score": 0.612,
    "avg_log_loss": 0.941,
    "model_version": "challenger-2026-03-11",
    "calibration_version": "ovr-isotonic-v1"
  },
  "calibration": {
    "target": "predicted_outcome",
    "buckets": [
      {
        "label": "50-60%",
        "range_start": 0.5,
        "range_end": 0.6,
        "avg_confidence": 0.552,
        "actual_rate": 0.531,
        "count": 32
      }
    ]
  },
  "segments": [
    {
      "name": "home_favorites",
      "count": 61,
      "outcome_accuracy": 0.64,
      "brier_score": 0.56
    }
  ],
  "rolling_backtest": {
    "window_size": 20,
    "step_size": 20,
    "windows": [
      {
        "label": "2025-10-01 to 2025-10-29",
        "match_count": 20,
        "outcome_accuracy": 0.6,
        "brier_score": 0.59,
        "avg_log_loss": 0.89
      }
    ]
  },
  "benchmarks": {
    "model": {
      "brier_score": 0.612,
      "avg_log_loss": 0.941
    },
    "naive": {
      "brier_score": 0.667,
      "avg_log_loss": 1.031
    },
    "bookmaker": {
      "available": false
    }
  }
}
```

Segment defaults:

- `home_favorites`
- `away_favorites`
- `predicted_draws`
- `high_confidence` (`>= 0.60`)
- `very_high_confidence` (`>= 0.70`)

### 6. Frontend Accuracy Dashboard

Update `frontend/app/accuracy/page.tsx`

Design direction:

- Keep the current summary cards, but add trust-oriented metrics
- Use simple inline SVG or CSS-based charts instead of adding a charting library in the first pass
- Show calibration and benchmark information without overwhelming the user

New sections:

1. Summary row
   - outcome accuracy
   - Brier score
   - log loss
   - benchmark delta vs naive
2. Calibration panel
   - bucket chart or bar grid
   - explanatory copy: "When we predict 60%, it has landed about 57% of the time"
3. Confidence segments
   - high-confidence pick accuracy with counts
4. Rolling trend panel
   - recent window Brier/outcome trend
5. Benchmark table
   - model vs naive vs bookmaker (if available)

Compatibility:

- The page should gracefully handle older API payloads by showing the existing summary-only view when new sections are absent

## MLflow Logging

Phase 2 should log richer evaluation artifacts:

- aggregate Brier score and log loss
- raw vs calibrated Brier score
- rolling-window Brier score and outcome accuracy
- calibration error by bucket
- benchmark deltas vs naive and bookmaker

If artifact logging is helpful, write compact JSON summaries to disk and log them as MLflow artifacts.

## Testing Plan

### Backend Unit Tests

- calibrator fits and preserves probability normalization
- calibrator improves or at least transforms known synthetic probability inputs correctly
- bookmaker implied probability normalization removes overround correctly
- rolling backtest windows are chronological and non-leaky
- benchmark metrics use only available odds rows

### Backend Service/API Tests

- `PredictionService` persists and reloads calibrator artifacts
- predictions store raw and calibrated probabilities correctly
- `/api/accuracy` returns legacy summary plus new nested sections
- `/api/accuracy` works with:
  - no predictions
  - predictions but no odds
  - predictions plus bookmaker odds

### Frontend Verification

- `npm run build` passes
- accuracy page renders:
  - empty state
  - summary-only API payload
  - full Phase 2 payload

## Implementation Slices

### Slice 1: Evaluation Foundations

- Expand `evaluate.py`
- Add benchmark math helpers
- Add unit tests for metrics, buckets, and rolling windows

### Slice 2: Calibration Infrastructure

- Add `calibration.py`
- Persist/load calibrator in `PredictionService`
- Store raw vs calibrated probabilities
- Add predictor tests

### Slice 3: Benchmark Data Path

- Add `MarketOdds` model
- Add manual CSV import script
- Add bookmaker benchmark calculations
- Add API tests with odds fixtures

### Slice 4: Accuracy API and MLflow

- Enrich `/api/accuracy`
- Log rolling and calibration metrics
- Keep legacy fields stable

### Slice 5: Frontend Dashboard

- Update `accuracy/page.tsx`
- Add calibration, rolling trend, and benchmark views
- Preserve graceful fallback for sparse data

## Risks and Mitigations

### Risk: Calibration overfits on small samples

Mitigation:

- require a minimum sample threshold before enabling calibration
- fall back to uncalibrated probabilities when thresholds are not met
- log whether calibration is active

### Risk: Walk-forward backtests become slow

Mitigation:

- start with `20`-match windows and `20`-match step size
- cache intermediate evaluated rows if needed
- keep the first implementation offline during training, not on request

### Risk: `sports-betting` data does not align cleanly with our fixture IDs or team names

Mitigation:

- introduce an adapter layer instead of coupling core code to the package
- normalize team names at ingestion time
- match rows conservatively and skip ambiguous odds instead of attaching them to the wrong fixture
- keep naive benchmarking first-class so Phase 2 still delivers value if odds coverage is incomplete

### Risk: Schema changes break existing local databases

Mitigation:

- update tests to recreate tables cleanly
- document DB reset steps in the implementation PR
- optionally add a lightweight Alembic setup if we want a cleaner deployment story

## Recommended Next Step

Implementation should start with Slice 1 and Slice 2 together so the calibration pipeline is built on top of the new rolling evaluation primitives instead of being bolted on afterward. After that, the odds adapter and benchmark plumbing can be layered in before the frontend dashboard is finalized.
