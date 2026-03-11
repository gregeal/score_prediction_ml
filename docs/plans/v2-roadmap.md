# PredictEPL v2 Roadmap

## Guiding Principle

The biggest improvement now is not "more ML complexity" by itself, but making the product better at three things: **richer inputs**, **better probability quality**, and **more user trust**.

---

## Phase 1: Feature Expansion + Challenger Model

### 1A. Richer Pre-Match Features
**File:** `backend/app/ml/features.py`

The model is currently team-strength driven only. Add these features:

- **Home/away split form** — Separate last-5 form for home and away matches. Home form is a stronger predictor of home performance than overall form.
- **Rolling xG / shots** — Expected goals are more stable than actual goals. Source: Understat (free scraping) or API-Football (paid).
- **Rest days** — Days since each team's last match. Congested schedules reduce performance.
- **Opponent strength adjustment** — Weight recent results by the quality of opponent faced (Elo-weighted form).
- **Red cards / disciplinary** — Teams with recent red card issues may be weakened.
- **Newly promoted teams** — Flag teams with no prior-season data; use league-average priors instead of missing parameters.
- **Head-to-head** — Recent H2H record between the two teams (secondary feature, lower predictive power than form).

### 1B. Challenger Model: Dixon-Coles + Elo + Gradient Boosting
**Files:** `backend/app/ml/challenger_model.py` (new), `backend/app/services/predictor.py`

- Keep Dixon-Coles as the baseline model.
- Add a second model: gradient-boosted classifier (XGBoost or LightGBM) that takes Dixon-Coles expected goals + all features from 1A as inputs.
- Use Dixon-Coles Poisson output as a prior, then adjust with the feature-rich model.
- Compare both models in MLflow on every training run.
- Serve whichever model has better recent Brier score.

### 1C. Elo Rating System
**File:** `backend/app/ml/elo.py` (new)

- Implement Elo ratings updated after each match.
- Feed Elo difference as a feature to the challenger model.
- Log Elo ratings per team to MLflow for tracking over time.

---

## Phase 2: Calibration + Richer Accuracy Dashboard

### 2A. Probability Calibration
**File:** `backend/app/ml/evaluate.py`

Calibrated probabilities matter more than flashy scoreline predictions.

- Track **Brier score**, **log loss**, and **calibration by confidence bucket** (not just hit rate).
- Add calibration curves: "when we say 60%, it should happen ~60% of the time."
- Segment accuracy by: home favorites, away favorites, draws, high-confidence picks.
- Add rolling backtests by month/season (not just one 80/20 split).

### 2B. Benchmark Table
**File:** `backend/app/ml/evaluate.py`, `backend/app/api/predictions.py`

- Compare: **our model** vs **bookmaker implied probabilities** vs **naive baseline** (home team always wins / league averages).
- Show this on the accuracy dashboard so users can see the model adds value over simple heuristics.

### 2C. Confidence Accuracy Display
**File:** `frontend/app/accuracy/page.tsx`

- Replace `high/medium/low` labels with actual historical accuracy: "High confidence picks have been correct 62% of the time."
- Show calibration chart (predicted probability vs actual outcome rate).

---

## Phase 3: Match Explanation Page + Data Freshness

### 3A. Match Detail Page
**File:** `frontend/app/match/[id]/page.tsx` (new)

Users trust predictions more when they understand the reasoning. Show:

- **Team strength gap** — Visual comparison of attack/defense ratings.
- **Recent form** — Last 5 results for each team with W/D/L indicators.
- **Expected goals** — Predicted xG for each team.
- **Top 5 scorelines** — Full probability breakdown, not just the most likely.
- **Key factors** — Why the model favors one side (e.g., "Arsenal's attack rating is 2nd highest in the league, Burnley's defense is 19th").
- **Historical accuracy** — "For matches with this confidence level, the model has been correct X% of the time."

### 3B. Data Freshness + Model Version in UI
**Files:** `frontend/app/page.tsx`, `frontend/app/accuracy/page.tsx`, `backend/app/api/predictions.py`

- Show "Trained on data up to [date]" on prediction pages.
- Show "Model version: Dixon-Coles v1.2 (trained 2026-03-11)" in footer or accuracy page.
- Add a `/api/status` endpoint returning: last data fetch time, model version, number of matches in DB, latest prediction date.

### 3C. Health/Status Endpoint
**File:** `backend/app/main.py`

```
GET /api/status
{
  "db_connected": true,
  "model_loaded": true,
  "total_matches": 1140,
  "total_predictions": 88,
  "latest_data": "2026-03-11",
  "model_trained_at": "2026-03-11T19:20:00Z",
  "model_type": "dixon_coles_v1"
}
```

---

## Phase 4: Data Expansion

### 4A. Richer Data Ingestion
**File:** `backend/app/services/data_fetcher.py`

- **Standings context** — Current league position, points gap to relegation/title.
- **Injuries/lineups** — If available from API-Football (paid) or free sources.
- **Bookmaker odds** — As both a benchmark and optional feature. Closing odds are the market's best prediction.

### 4B. xG Data Integration
**File:** `backend/app/services/xg_fetcher.py` (new)

- Source: Understat (free, scrapable) or StatsBomb/API-Football (paid).
- Rolling xG per team (last 5/10 games).
- xG vs actual goals differential (luck factor).

### 4C. Player Availability
**File:** `backend/app/services/injury_fetcher.py` (new)

- Key injuries can swing predictions 10-15%.
- Source: API-Football injuries endpoint or football-data.org (if available on paid tier).
- Weight by player importance (starters vs squad players).

---

## Phase 5: Product Features

### 5A. Team Watchlists + Notifications
- User accounts (simple auth).
- "Follow" teams to get prediction summaries.
- Daily email/push notification with predictions for followed teams.

### 5B. Daily Prediction Summaries
- Auto-generated "Today's Picks" page with highest-confidence predictions.
- "Banker of the Day" — single highest-confidence pick.
- Weekly roundup comparing predictions vs actual results.

### 5C. Odds Calibration View
- Side-by-side: our prediction vs bookmaker odds.
- Highlight value bets where our model disagrees significantly with the market.

---

## Phase 6: Engineering

### 6A. Scheduled Daily Jobs
- Automated: fetch data -> retrain -> predict -> alert if model is stale.
- Celery + Redis, or simple cron in Docker.
- Alert if no new data fetched in 48 hours.

### 6B. Model Registry
- "Active model", "candidate model", "last promoted at".
- Auto-promote candidate if it beats active on rolling 20-match Brier score.
- MLflow model registry integration.

### 6C. Test Coverage Expansion
- Feature generation edge cases (newly promoted teams, missing data).
- Retraining with partial data.
- Upsert idempotency tests.
- API accuracy calculation with duplicate/missing predictions.
- End-to-end: fetch -> train -> predict -> API response.

---

## Implementation Priority

If choosing the next three things to build:

1. **Feature expansion + challenger model** (Phase 1) — Highest accuracy ROI
2. **Calibration and richer accuracy dashboard** (Phase 2) — Builds user trust
3. **Match explanation page + data freshness/status** (Phase 3) — Turns demo into product

---

## v2 Feature Summary by File

| File | Changes |
|------|---------|
| `backend/app/ml/features.py` | Home/away form, rest days, opponent strength, xG features |
| `backend/app/ml/challenger_model.py` | New gradient-boosted model |
| `backend/app/ml/elo.py` | Elo rating system |
| `backend/app/ml/evaluate.py` | Calibration curves, rolling backtests, benchmark table |
| `backend/app/services/predictor.py` | Multi-model support, model selection |
| `backend/app/services/data_fetcher.py` | Standings, injuries, odds ingestion |
| `backend/app/api/predictions.py` | Status endpoint, model version, benchmark data |
| `frontend/app/match/[id]/page.tsx` | Match explanation page |
| `frontend/app/accuracy/page.tsx` | Calibration chart, confidence accuracy, benchmarks |
| `frontend/app/page.tsx` | Data freshness indicator |
