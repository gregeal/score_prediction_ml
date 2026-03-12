"""Train the model and generate predictions for upcoming fixtures.

Run this daily as a cron job or manually:
    python scripts/train_model.py
"""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.base import ensure_database_ready, get_session_local
from app.services.predictor import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    ensure_database_ready()

    db = get_session_local()()
    try:
        service = PredictionService(db)

        # Train model on all historical data
        service.train_model()

        # Generate predictions for upcoming matches
        predictions = service.predict_upcoming()

        # Print summary
        for pred in predictions:
            logger.info(
                f"{pred.home_team} vs {pred.away_team}: "
                f"H={pred.home_win_prob:.0%} D={pred.draw_prob:.0%} A={pred.away_win_prob:.0%} | "
                f"Score: {pred.most_likely_score} | "
                f"O/U 2.5: {pred.over25_prob:.0%} | "
                f"BTTS: {pred.btts_prob:.0%} | "
                f"Confidence: {pred.confidence}"
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
