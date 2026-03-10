"""Prediction service: orchestrates model training and prediction generation."""

import logging
import pickle
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.match import Match
from app.models.prediction import Prediction
from app.ml.dixon_coles import DixonColesModel, MatchPrediction
from app.ml.features import matches_to_training_data

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "trained_model.pkl"


class PredictionService:
    """Orchestrates model training and prediction generation."""

    def __init__(self, db: Session):
        self.db = db
        self.model = DixonColesModel(time_decay_days=365)

    def train_model(self) -> None:
        """Train the model on all finished matches in the database."""
        matches = (
            self.db.query(Match)
            .filter(Match.status == "FINISHED")
            .all()
        )
        if not matches:
            raise ValueError("No finished matches in database to train on")

        training_data = matches_to_training_data(matches)
        logger.info(f"Training model on {len(training_data)} matches...")

        params = self.model.fit(training_data)
        logger.info(
            f"Model trained. Home advantage: {params.home_advantage:.3f}, "
            f"Rho: {params.rho:.3f}"
        )

        # Save model to disk
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {MODEL_PATH}")

    def load_model(self) -> None:
        """Load a previously trained model from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run train_model() first."
            )
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded from disk")

    def predict_upcoming(self) -> list[MatchPrediction]:
        """Generate predictions for all upcoming matches."""
        upcoming = (
            self.db.query(Match)
            .filter(Match.status.in_(["SCHEDULED", "TIMED"]))
            .order_by(Match.utc_date)
            .all()
        )

        predictions = []
        for match in upcoming:
            try:
                pred = self.model.predict_match(match.home_team, match.away_team)
                predictions.append(pred)

                # Store prediction in database
                db_pred = Prediction(
                    match_api_id=match.api_id,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    predicted_home_goals=pred.predicted_home_goals,
                    predicted_away_goals=pred.predicted_away_goals,
                    home_win_prob=pred.home_win_prob,
                    draw_prob=pred.draw_prob,
                    away_win_prob=pred.away_win_prob,
                    over25_prob=pred.over25_prob,
                    btts_prob=pred.btts_prob,
                    most_likely_score=pred.most_likely_score,
                    confidence=pred.confidence,
                )
                self.db.add(db_pred)
            except ValueError as e:
                logger.warning(f"Could not predict {match.home_team} vs {match.away_team}: {e}")

        self.db.commit()
        logger.info(f"Generated {len(predictions)} predictions for upcoming matches")
        return predictions
