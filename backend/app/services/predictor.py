"""Prediction service: orchestrates model training and prediction generation."""

import logging
import pickle
from pathlib import Path

import mlflow
from sqlalchemy.orm import Session

from app.config import settings
from app.models.match import Match
from app.models.prediction import Prediction
from app.ml.dixon_coles import DixonColesModel, MatchPrediction
from app.ml.features import matches_to_training_data
from app.ml.evaluate import backtest

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "trained_model.pkl"
MLFLOW_EXPERIMENT = "predictepl-dixon-coles"


class PredictionService:
    """Orchestrates model training and prediction generation."""

    def __init__(self, db: Session):
        self.db = db
        self.model = DixonColesModel(time_decay_days=365)
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    def train_model(self, run_evaluation: bool = True) -> None:
        """Train the model on all finished matches in the database.

        Args:
            run_evaluation: If True, hold out the latest season for backtesting
                and log metrics to MLflow.
        """
        matches = (
            self.db.query(Match)
            .filter(Match.status == "FINISHED")
            .order_by(Match.utc_date)
            .all()
        )
        if not matches:
            raise ValueError("No finished matches in database to train on")

        training_data = matches_to_training_data(matches)
        logger.info(f"Training model on {len(training_data)} matches...")

        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="train"):
            # Log parameters
            mlflow.log_param("model_type", "dixon_coles")
            mlflow.log_param("time_decay_days", self.model.time_decay_days)
            mlflow.log_param("num_training_matches", len(training_data))
            mlflow.log_param("num_teams", len(set(
                m.home_team for m in training_data
            ) | set(m.away_team for m in training_data)))

            # Train
            params = self.model.fit(training_data)

            # Log model parameters
            mlflow.log_metric("home_advantage", params.home_advantage)
            mlflow.log_metric("rho", params.rho)

            # Log attack/defense strengths for each team
            for team in params.teams:
                safe_name = "".join(c if c.isalnum() or c in "_-. /" else "" for c in team).replace(" ", "_")
                mlflow.log_metric(f"attack_{safe_name}", params.attack[team])
                mlflow.log_metric(f"defense_{safe_name}", params.defense[team])

            # Run evaluation if requested (pass raw matches to avoid form-weight leakage)
            if run_evaluation and len(training_data) > 100:
                self._evaluate_and_log(matches)

            # Save model artifact
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(self.model, f)
            mlflow.log_artifact(str(MODEL_PATH))

            logger.info(
                f"Model trained. Home advantage: {params.home_advantage:.3f}, "
                f"Rho: {params.rho:.3f}"
            )

    def _evaluate_and_log(self, raw_matches: list[Match]) -> None:
        """Split data, evaluate, and log metrics to MLflow.

        Builds train/test splits from raw Match objects without form weighting
        to prevent data leakage (form weights use future match info).
        """
        # Convert without form weighting to get clean splits
        all_data = matches_to_training_data(raw_matches, use_form_weighting=False)

        # Use last 20% of matches as test set
        split_idx = int(len(all_data) * 0.8)
        train_split = all_data[:split_idx]
        test_split = all_data[split_idx:]

        if len(test_split) < 10:
            logger.warning("Not enough test matches for evaluation, skipping")
            return

        eval_model = DixonColesModel(time_decay_days=self.model.time_decay_days)
        try:
            result = backtest(eval_model, train_split, test_split)

            mlflow.log_metric("eval_outcome_accuracy", result.outcome_accuracy)
            mlflow.log_metric("eval_exact_score_accuracy", result.exact_score_accuracy)
            mlflow.log_metric("eval_over25_accuracy", result.over25_accuracy)
            mlflow.log_metric("eval_btts_accuracy", result.btts_accuracy)
            mlflow.log_metric("eval_brier_score", result.brier_score)
            mlflow.log_metric("eval_log_loss", result.avg_log_loss)
            mlflow.log_metric("eval_test_matches", result.total_matches)

            logger.info(
                f"Evaluation: outcome={result.outcome_accuracy:.1%}, "
                f"exact={result.exact_score_accuracy:.1%}, "
                f"O/U={result.over25_accuracy:.1%}, "
                f"BTTS={result.btts_accuracy:.1%}, "
                f"brier={result.brier_score:.4f}"
            )
        except ValueError as e:
            logger.warning(f"Evaluation failed: {e}")

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
        """Generate predictions for all upcoming matches.

        Replaces any existing predictions for each match so there is
        at most one prediction row per match_api_id.
        """
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

                # Delete any previous prediction for this match
                self.db.query(Prediction).filter(
                    Prediction.match_api_id == match.api_id
                ).delete()

                # Store prediction in database (convert np.float64 to float for PostgreSQL)
                db_pred = Prediction(
                    match_api_id=match.api_id,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    predicted_home_goals=float(pred.predicted_home_goals),
                    predicted_away_goals=float(pred.predicted_away_goals),
                    home_win_prob=float(pred.home_win_prob),
                    draw_prob=float(pred.draw_prob),
                    away_win_prob=float(pred.away_win_prob),
                    over25_prob=float(pred.over25_prob),
                    btts_prob=float(pred.btts_prob),
                    most_likely_score=pred.most_likely_score,
                    outcome_score=pred.outcome_score,
                    confidence=pred.confidence,
                )
                self.db.add(db_pred)
            except ValueError as e:
                logger.warning(f"Could not predict {match.home_team} vs {match.away_team}: {e}")

        self.db.commit()
        logger.info(f"Generated {len(predictions)} predictions for upcoming matches")
        return predictions
