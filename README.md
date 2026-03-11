# PredictEPL

AI-powered English Premier League score predictions for the Nigerian market.

Uses a **Dixon-Coles Poisson model** to predict match outcomes, exact scores, over/under 2.5 goals, and both-teams-to-score (BTTS) probabilities.

## Tech Stack

- **ML Model:** Python, SciPy, scikit-learn (Dixon-Coles Poisson regression)
- **Experiment Tracking:** MLflow
- **Backend API:** FastAPI
- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **Database:** PostgreSQL (via Docker)
- **Data Source:** football-data.org API
- **Infrastructure:** Docker Compose

## Project Structure

```
score_prediction_ml/
├── docker-compose.yml           # PostgreSQL + MLflow + Backend
├── backend/
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── config.py            # Settings & env vars
│   │   ├── models/              # SQLAlchemy database models
│   │   ├── api/                 # API route handlers
│   │   ├── services/            # Data fetcher + predictor
│   │   └── ml/                  # Dixon-Coles ML model
│   ├── scripts/
│   │   ├── fetch_data.py        # Fetch historical EPL data
│   │   └── train_model.py       # Train model & generate predictions
│   ├── tests/
│   └── requirements.txt
├── frontend/                    # Next.js web app
│   ├── app/
│   │   ├── page.tsx             # Home - upcoming matches
│   │   ├── accuracy/page.tsx    # Model accuracy dashboard
│   │   └── standings/page.tsx   # EPL table
│   └── package.json
└── .env.example
```

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Node.js 18+
- A free API key from [football-data.org](https://www.football-data.org/client/register)

### Quick Start (Docker)

This is the recommended way to run the full stack.

```bash
# 1. Clone and configure
git clone <your-repo-url>
cd score_prediction_ml
cp .env.example .env
# Edit .env and add your football-data.org API key

# 2. Start PostgreSQL, MLflow, and the backend API
docker compose up -d

# 3. Fetch EPL data (run inside the backend container)
docker compose exec backend python scripts/fetch_data.py

# 4. Train model & generate predictions
docker compose exec backend python scripts/train_model.py

# 5. Start the frontend (in a separate terminal)
cd frontend
npm install
npm run dev
```

After this:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000 (Swagger docs at /docs)
- **MLflow UI:** http://localhost:5000

### Local Development (without Docker)

If you prefer running without Docker:

```bash
# 1. Set up Python environment
uv venv venv --python 3.12
source venv/Scripts/activate  # Windows Git Bash
# source venv/bin/activate    # macOS/Linux
uv pip install --python venv/Scripts/python.exe -r backend/requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env:
#   - Add your football-data.org API key
#   - Change DATABASE_URL to sqlite:///./predictepl.db if you don't have PostgreSQL

# 3. Fetch data, train, and run
cd backend
python scripts/fetch_data.py
python scripts/train_model.py
uvicorn app.main:app --reload --port 8000

# 4. Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `db` | 5432 | PostgreSQL 16 database |
| `mlflow` | 5000 | MLflow tracking server (UI + API) |
| `backend` | 8000 | FastAPI prediction API |

```bash
# View logs
docker compose logs -f backend

# Stop everything
docker compose down

# Stop and remove all data (fresh start)
docker compose down -v
```

## MLflow Experiment Tracking

Every training run logs to MLflow:

- **Parameters:** model type, time decay, number of training matches, team count
- **Metrics:** home advantage, rho, per-team attack/defense strengths
- **Evaluation metrics:** outcome accuracy, exact score accuracy, O/U 2.5 accuracy, BTTS accuracy, Brier score, log loss
- **Artifacts:** trained model pickle file

Access the MLflow UI at http://localhost:5000 to compare runs, view metrics, and track model evolution over time.

## Running Tests

```bash
# With Docker
docker compose exec backend python -m pytest tests/ -v

# Without Docker (from backend/ directory)
python -m pytest tests/ -v
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/fixtures/upcoming` | Upcoming matches with predictions |
| `GET /api/predictions/{match_id}` | Detailed prediction for a match |
| `GET /api/accuracy` | Model accuracy statistics |
| `GET /api/standings` | Current EPL table |
| `GET /docs` | Interactive Swagger docs |

## How It Works

### The Dixon-Coles Model

1. **Team Ratings:** Estimates attack and defense strength for each EPL team using maximum likelihood estimation on historical results.
2. **Home Advantage:** A parameter capturing the statistical edge of playing at home.
3. **Low-Score Correction:** The Dixon-Coles rho factor adjusts probabilities for 0-0, 1-0, 0-1, and 1-1 scorelines.
4. **Time Weighting:** Exponential decay (half-life = 1 season) so recent form matters more.
5. **Score Matrix:** A 10x10 probability matrix for every possible scoreline.
6. **Derived Predictions:**
   - **1X2:** Home win / Draw / Away win probabilities
   - **Exact Score:** Top 5 most likely scorelines
   - **Over/Under 2.5:** Probability of total goals > 2.5
   - **BTTS:** Probability that both teams score

### Daily Pipeline

Run data fetch + retrain daily to keep predictions current:

```bash
# With Docker
docker compose exec backend python scripts/fetch_data.py
docker compose exec backend python scripts/train_model.py
```

## Roadmap

See [docs/plans/v2-roadmap.md](docs/plans/v2-roadmap.md) for the full implementation plan.

**Next 3 priorities:**
1. Feature expansion + challenger model (home/away form, xG, Elo, gradient boosting)
2. Calibration and richer accuracy dashboard (Brier score by bucket, benchmark vs bookmakers)
3. Match explanation page + data freshness/status

**Full roadmap:**
- [ ] Phase 1: Richer features + Dixon-Coles + Elo + gradient boosting challenger model
- [ ] Phase 2: Probability calibration, rolling backtests, benchmark table
- [ ] Phase 3: Match explanation page, data freshness in UI, `/api/status` endpoint
- [ ] Phase 4: xG data, injuries, bookmaker odds integration
- [ ] Phase 5: Team watchlists, notifications, daily prediction summaries
- [ ] Phase 6: Scheduled jobs, model registry, expanded test coverage
- [ ] Android app (React Native or Flutter)
- [ ] Betting platform integration (Bet9ja, BetKing affiliate links)
- [ ] Telegram bot for prediction delivery
