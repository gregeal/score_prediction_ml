"""Prediction service: orchestrates model training and prediction generation."""

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.models.match import Match
from app.models.prediction import Prediction
from app.ml.dixon_coles import DixonColesModel, MatchPrediction
from app.ml.challenger_model import ChallengerModel
from app.ml.elo import EloSystem
from app.ml.features import build_match_features, matches_to_training_data
from app.ml.evaluate import backtest

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent
DC_MODEL_PATH = MODEL_DIR / "trained_model.pkl"
CHALLENGER_MODEL_PATH = MODEL_DIR / "challenger_model.pkl"
ELO_PATH = MODEL_DIR / "elo_system.pkl"
ACTIVE_MODEL_PATH = MODEL_DIR / "active_model.txt"
MLFLOW_EXPERIMENT = "predictepl"


class PredictionService:
    """Orchestrates model training and prediction generation."""

    def __init__(self, db: Session):
        self.db = db
        self.dc_model = DixonColesModel(time_decay_days=365)
        self.challenger = ChallengerModel(time_decay_days=365)
        self.elo_system: EloSystem | None = None
        self.active_model = "dixon_coles"  # or "challenger"
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    @property
    def model(self):
        """Backward compat: return the active Dixon-Coles model."""
        return self.dc_model

    def train_model(self, run_evaluation: bool = True) -> None:
        """Train both models on all finished matches in the database."""
        matches = (
            self.db.query(Match)
            .filter(Match.status == "FINISHED")
            .order_by(Match.utc_date)
            .all()
        )
        if not matches:
            raise ValueError("No finished matches in database to train on")

        # 1. Train Dixon-Coles
        training_data = matches_to_training_data(matches)
        logger.info(f"Training Dixon-Coles on {len(training_data)} matches...")
        dc_params = self.dc_model.fit(training_data)

        # 2. Build Elo ratings
        self.elo_system = EloSystem.from_matches(matches)
        logger.info(f"Elo ratings computed for {len(self.elo_system.ratings)} teams")

        # 3. Train Challenger (needs enough data — 200+ matches for reliable GBM)
        challenger_trained = False
        if len(training_data) >= 200:
            try:
                self.challenger.fit(matches, self.elo_system)
                challenger_trained = True
                logger.info("Challenger model trained successfully")
            except ValueError as e:
                logger.warning(f"Challenger model skipped: {e}")
        else:
            logger.info(f"Challenger skipped: only {len(training_data)} matches (need 200+)")

        # 4. Log to MLflow
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="train"):
            mlflow.log_param("dc_time_decay_days", self.dc_model.time_decay_days)
            mlflow.log_param("num_training_matches", len(training_data))
            num_teams = len(set(m.home_team for m in training_data) | set(m.away_team for m in training_data))
            mlflow.log_param("num_teams", num_teams)
            mlflow.log_param("challenger_trained", challenger_trained)

            mlflow.log_metric("dc_home_advantage", dc_params.home_advantage)
            mlflow.log_metric("dc_rho", dc_params.rho)

            # Log Elo ratings for top/bottom teams
            if self.elo_system:
                sorted_elo = sorted(self.elo_system.ratings.items(), key=lambda x: x[1], reverse=True)
                for team, rating in sorted_elo[:5]:
                    safe = "".join(c if c.isalnum() or c in "_-." else "" for c in team).replace(" ", "_")
                    mlflow.log_metric(f"elo_{safe}", rating)

            # 5. Evaluate and pick best model
            if run_evaluation and len(training_data) > 100:
                self._evaluate_and_log(matches, challenger_trained)

            # 6. Save models + active model choice
            with open(DC_MODEL_PATH, "wb") as f:
                pickle.dump(self.dc_model, f)
            if challenger_trained:
                with open(CHALLENGER_MODEL_PATH, "wb") as f:
                    pickle.dump(self.challenger, f)
            with open(ELO_PATH, "wb") as f:
                pickle.dump(self.elo_system, f)
            ACTIVE_MODEL_PATH.write_text(self.active_model)

            mlflow.log_param("active_model", self.active_model)
            mlflow.log_artifact(str(DC_MODEL_PATH))

            logger.info(
                f"Training complete. Active model: {self.active_model}. "
                f"DC home_adv={dc_params.home_advantage:.3f}, rho={dc_params.rho:.3f}"
            )

    def _evaluate_and_log(self, raw_matches: list[Match], challenger_trained: bool) -> None:
        """Backtest both models, log metrics, pick the winner by Brier score."""
        all_data = matches_to_training_data(raw_matches, use_form_weighting=False)

        split_idx = int(len(all_data) * 0.8)
        train_split = all_data[:split_idx]
        test_split = all_data[split_idx:]

        if len(test_split) < 10:
            logger.warning("Not enough test matches for evaluation, skipping")
            return

        # Evaluate Dixon-Coles
        dc_eval = DixonColesModel(time_decay_days=self.dc_model.time_decay_days)
        try:
            dc_result = backtest(dc_eval, train_split, test_split)
            mlflow.log_metric("dc_outcome_accuracy", dc_result.outcome_accuracy)
            mlflow.log_metric("dc_brier_score", dc_result.brier_score)
            mlflow.log_metric("dc_log_loss", dc_result.avg_log_loss)
            mlflow.log_metric("dc_over25_accuracy", dc_result.over25_accuracy)
            mlflow.log_metric("dc_btts_accuracy", dc_result.btts_accuracy)
            mlflow.log_metric("dc_test_matches", dc_result.total_matches)
            logger.info(
                f"DC eval: outcome={dc_result.outcome_accuracy:.1%}, "
                f"brier={dc_result.brier_score:.4f}"
            )
        except ValueError as e:
            logger.warning(f"DC evaluation failed: {e}")
            return

        # Evaluate Challenger (if trained)
        if not challenger_trained or self.elo_system is None:
            self.active_model = "dixon_coles"
            return

        # Build a FRESH challenger trained only on the training split to avoid
        # leaking test-set info through Dixon-Coles params or Elo ratings.
        finished_sorted = sorted(
            [m for m in raw_matches if m.status == "FINISHED"],
            key=lambda m: m.utc_date,
        )
        train_matches_raw = finished_sorted[:split_idx]
        train_elo = EloSystem.from_matches(train_matches_raw)

        eval_challenger = ChallengerModel(time_decay_days=self.challenger.time_decay_days)
        try:
            eval_challenger.fit(train_matches_raw, train_elo)
        except ValueError as e:
            logger.warning(f"Challenger eval skipped (not enough train data): {e}")
            self.active_model = "dixon_coles"
            return

        # Context for feature computation: only training-period matches
        train_context_desc = sorted(train_matches_raw, key=lambda m: m.utc_date, reverse=True)

        challenger_correct = 0
        challenger_brier = []
        challenger_total = 0

        for match_data in test_split:
            try:
                pred = eval_challenger.predict_match(
                    match_data.home_team,
                    match_data.away_team,
                    train_elo,
                    train_context_desc,
                )
            except (ValueError, KeyError):
                continue

            challenger_total += 1

            # Actual outcome
            if match_data.home_goals > match_data.away_goals:
                actual_vec = [1, 0, 0]
                actual = "home"
            elif match_data.home_goals == match_data.away_goals:
                actual_vec = [0, 1, 0]
                actual = "draw"
            else:
                actual_vec = [0, 0, 1]
                actual = "away"

            pred_vec = [pred.home_win_prob, pred.draw_prob, pred.away_win_prob]
            predicted = ["home", "draw", "away"][np.argmax(pred_vec)]
            if predicted == actual:
                challenger_correct += 1

            brier = sum((p - a) ** 2 for p, a in zip(pred_vec, actual_vec))
            challenger_brier.append(brier)

        if challenger_total > 0:
            gbm_accuracy = round(challenger_correct / challenger_total, 4)
            gbm_brier = round(float(np.mean(challenger_brier)), 4)
            mlflow.log_metric("gbm_outcome_accuracy", gbm_accuracy)
            mlflow.log_metric("gbm_brier_score", gbm_brier)
            mlflow.log_metric("gbm_test_matches", challenger_total)
            logger.info(f"GBM eval: outcome={gbm_accuracy:.1%}, brier={gbm_brier:.4f}")

            # Pick winner
            if gbm_brier < dc_result.brier_score:
                self.active_model = "challenger"
                logger.info(f"Challenger wins: brier {gbm_brier:.4f} < {dc_result.brier_score:.4f}")
            else:
                self.active_model = "dixon_coles"
                logger.info(f"Dixon-Coles wins: brier {dc_result.brier_score:.4f} <= {gbm_brier:.4f}")
        else:
            self.active_model = "dixon_coles"

    def load_model(self) -> None:
        """Load previously trained models from disk."""
        if not DC_MODEL_PATH.exists():
            raise FileNotFoundError(f"No trained model at {DC_MODEL_PATH}")
        with open(DC_MODEL_PATH, "rb") as f:
            self.dc_model = pickle.load(f)
        if CHALLENGER_MODEL_PATH.exists():
            with open(CHALLENGER_MODEL_PATH, "rb") as f:
                self.challenger = pickle.load(f)
        if ELO_PATH.exists():
            with open(ELO_PATH, "rb") as f:
                self.elo_system = pickle.load(f)
        if ACTIVE_MODEL_PATH.exists():
            self.active_model = ACTIVE_MODEL_PATH.read_text().strip()
        logger.info(f"Models loaded. Active: {self.active_model}")

    def predict_upcoming(self) -> list[MatchPrediction]:
        """Generate predictions for all upcoming matches using the active model."""
        upcoming = (
            self.db.query(Match)
            .filter(Match.status.in_(["SCHEDULED", "TIMED"]))
            .order_by(Match.utc_date)
            .all()
        )

        # Get match context for challenger model
        sorted_desc = None
        if self.active_model == "challenger" and self.elo_system:
            finished = (
                self.db.query(Match)
                .filter(Match.status == "FINISHED")
                .order_by(Match.utc_date.desc())
                .all()
            )
            sorted_desc = finished

        predictions = []
        for match in upcoming:
            try:
                if self.active_model == "challenger" and self.challenger.is_fitted and sorted_desc:
                    match_date = match.utc_date
                    if match_date.tzinfo is None:
                        match_date = match_date.replace(tzinfo=timezone.utc)
                    pred = self.challenger.predict_match(
                        match.home_team, match.away_team,
                        self.elo_system, sorted_desc,
                        reference_date=match_date,
                    )
                else:
                    pred = self.dc_model.predict_match(match.home_team, match.away_team)

                predictions.append(pred)

                # Delete any previous prediction for this match
                self.db.query(Prediction).filter(
                    Prediction.match_api_id == match.api_id
                ).delete()

                # Store prediction (convert np.float64 to float for PostgreSQL)
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
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not predict {match.home_team} vs {match.away_team}: {e}")

        self.db.commit()
        logger.info(f"Generated {len(predictions)} predictions (model: {self.active_model})")
        return predictions
