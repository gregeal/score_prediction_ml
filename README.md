# PredictEPL

AI-powered English Premier League score predictions for the Nigerian market.

Uses a **Dixon-Coles Poisson model** to predict match outcomes, exact scores, over/under 2.5 goals, and both-teams-to-score (BTTS) probabilities.

## Tech Stack

- **ML Model:** Python, SciPy, scikit-learn (Dixon-Coles Poisson regression)
- **Backend API:** FastAPI
- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **Database:** SQLite (dev) / PostgreSQL (production)
- **Data Source:** football-data.org API

## Project Structure

```
score_prediction_ml/
├── backend/
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

- Python 3.12+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A free API key from [football-data.org](https://www.football-data.org/client/register)

### 1. Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd score_prediction_ml

# Create Python virtual environment
uv venv venv --python 3.12

# Activate it
# On Windows (Git Bash):
source venv/Scripts/activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
uv pip install --python venv/Scripts/python.exe -r backend/requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your football-data.org API key
```

The `.env` file should contain:
```
FOOTBALL_DATA_API_KEY=your_api_key_here
DATABASE_URL=sqlite:///./predictepl.db
```

### 3. Fetch Historical Data

This fetches EPL match data for the 2022/23 through 2025/26 seasons:

```bash
cd backend
python scripts/fetch_data.py
```

> Note: The free API tier allows 10 requests/minute. The script handles rate limiting automatically, so fetching 4 seasons takes a few minutes.

### 4. Train Model & Generate Predictions

```bash
python scripts/train_model.py
```

This will:
1. Train the Dixon-Coles model on all historical match data
2. Generate predictions for all upcoming (scheduled) fixtures
3. Save the trained model to `backend/trained_model.pkl`
4. Print a summary of predictions to the console

### 5. Start the Backend API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Key endpoints:
- `GET /api/fixtures/upcoming` — Upcoming matches with predictions
- `GET /api/predictions/{match_id}` — Detailed prediction for a match
- `GET /api/accuracy` — Model accuracy statistics
- `GET /api/standings` — Current EPL table
- `GET /docs` — Interactive API documentation (Swagger UI)

### 6. Start the Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

## How It Works

### The Dixon-Coles Model

1. **Team Ratings:** The model estimates attack strength and defense strength for each EPL team using maximum likelihood estimation on historical match results.

2. **Home Advantage:** A home advantage parameter captures the statistical edge of playing at home.

3. **Low-Score Correction:** The Dixon-Coles correction factor (rho) adjusts probabilities for 0-0, 1-0, 0-1, and 1-1 scorelines, which basic Poisson models get wrong.

4. **Time Weighting:** Recent matches are weighted more heavily using exponential decay (half-life = 1 season), so the model adapts to current form.

5. **Score Matrix:** For each match, the model produces a 10x10 probability matrix for every possible scoreline (0-0 through 9-9).

6. **Derived Predictions:** From the score matrix, we calculate:
   - **1X2:** Home win / Draw / Away win probabilities
   - **Exact Score:** Top 5 most likely scorelines
   - **Over/Under 2.5:** Probability of total goals > 2.5
   - **BTTS:** Probability that both teams score

### Daily Pipeline

For production use, run the pipeline daily:
```bash
# Fetch latest results + retrain + predict upcoming
cd backend
python scripts/fetch_data.py && python scripts/train_model.py
```

This can be scheduled as a cron job:
```cron
0 6 * * * cd /path/to/score_prediction_ml/backend && python scripts/fetch_data.py && python scripts/train_model.py
```

## API Response Examples

### Upcoming Fixtures
```json
{
  "fixtures": [
    {
      "match_id": 12345,
      "home": "Arsenal FC",
      "away": "Chelsea FC",
      "date": "2026-03-14T15:00:00+00:00",
      "matchday": 28,
      "prediction": {
        "outcome": {"home_win": 0.52, "draw": 0.24, "away_win": 0.24},
        "most_likely_score": "2-1",
        "over_under_25": 0.58,
        "btts": 0.55,
        "confidence": "medium"
      }
    }
  ],
  "count": 1
}
```

## Roadmap

- [ ] Upgrade to hybrid model (xG + Elo + gradient boosting)
- [ ] Android app
- [ ] Betting platform integration (Bet9ja, BetKing)
- [ ] User accounts & saved predictions
- [ ] Push notifications
- [ ] Telegram bot
