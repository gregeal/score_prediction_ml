"""Prediction service: orchestrates model training and prediction generation."""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

# Fail fast when the MLflow tracking server is unreachable instead of
# stalling training for minutes in exponential-backoff retries. Users can
# still override these through their environment.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")


class _NoOpRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _NoOpMlflow:
    def set_tracking_uri(self, *args, **kwargs):
        pass

    def set_experiment(self, *args, **kwargs):
        pass

    def start_run(self, *args, **kwargs):
        return _NoOpRun()

    def log_param(self, *args, **kwargs):
        pass

    def log_metric(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass


try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ModuleNotFoundError:
    MLFLOW_AVAILABLE = False
    mlflow = _NoOpMlflow()

from app.config import settings
from app.seasons import current_season_year, utc_now
from app.ml.artifacts import load_bundle, save_bundle
from app.services.prediction_history import eligible_prediction_ids
from app.services.tracking import SafeTracking
from app.models.match import Match
from app.models.prediction import Prediction
from app.ml.calibration import OutcomeCalibrator
from app.ml.dixon_coles import DixonColesModel, MatchPrediction
from app.ml.challenger_model import ChallengerModel
from app.ml.elo import EloSystem
from app.ml.evaluate import OUTCOMES, backtest
from app.ml.features import build_match_features, matches_to_training_data

logger = logging.getLogger(__name__)
if not MLFLOW_AVAILABLE:
    logger.warning("MLflow is not installed; training will continue without experiment logging.")

MODEL_DIR = Path(__file__).parent.parent.parent
DC_MODEL_PATH = MODEL_DIR / "trained_model.pkl"
CHALLENGER_MODEL_PATH = MODEL_DIR / "challenger_model.pkl"
ELO_PATH = MODEL_DIR / "elo_system.pkl"
ACTIVE_MODEL_PATH = MODEL_DIR / "active_model.txt"
CALIBRATOR_PATH = MODEL_DIR / "outcome_calibrator.pkl"
BUNDLE_PATH = MODEL_DIR / "prediction_bundle.skops"
MLFLOW_EXPERIMENT = "predictepl"


class PredictionService:
    """Orchestrates model training and prediction generation."""

    def __init__(self, db: Session):
        self.db = db
        self._mlflow_override = None  # set to a no-op when tracking is unreachable
        # Defaults tuned by scripts/benchmark_model.py (walk-forward Brier)
        self.dc_model = DixonColesModel()
        self.challenger = ChallengerModel(time_decay_days=540)
        self.elo_system: EloSystem | None = None
        self.calibrator: OutcomeCalibrator | None = None
        self.active_model = "dixon_coles"  # or "challenger"
        self.model_version = "untrained"
        if settings.mlflow_tracking_uri:
            SafeTracking(mlflow).set_tracking_uri(settings.mlflow_tracking_uri)

    @property
    def model(self):
        """Backward compat: return the active Dixon-Coles model."""
        return self.dc_model

    @property
    def _mlflow(self):
        """MLflow logger, resolved lazily so an unreachable tracking server
        can be swapped for a no-op and tests can patch the module global."""
        if not settings.mlflow_tracking_uri:
            return _NoOpMlflow()
        return SafeTracking(self._mlflow_override if self._mlflow_override is not None else mlflow)

    def train_model(self, run_evaluation: bool = True) -> None:
        """Train both models on all finished matches in the database."""
        matches = (
            self.db.query(Match)
            .filter(Match.status == "FINISHED", Match.home_goals.isnot(None),
                    Match.away_goals.isnot(None), Match.utc_date < utc_now())
            .order_by(Match.utc_date)
            .all()
        )
        if not matches:
            raise ValueError("No finished matches in database to train on")

        # 1. Train Dixon-Coles
        self.active_model = "dixon_coles"
        self.challenger = ChallengerModel(time_decay_days=self.dc_model.time_decay_days)
        training_data = matches_to_training_data(matches, time_decay_days=self.dc_model.time_decay_days)
        logger.info(f"Training Dixon-Coles on {len(training_data)} matches...")
        dc_params = self.dc_model.fit(training_data)

        # 2. Build Elo ratings
        self.elo_system = EloSystem.from_matches(matches)
        logger.info(f"Elo ratings computed for {len(self.elo_system.ratings)} teams")

        # 3. Train Challenger (needs enough data - 200+ matches for reliable GBM)
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

        # 4. Log to MLflow. An unreachable tracking server must never abort
        # training - degrade to no-op logging instead.
        self._mlflow_override = None
        try:
            self._mlflow.set_experiment(MLFLOW_EXPERIMENT)
        except Exception as exc:
            logger.warning(
                "MLflow tracking unavailable (%s); continuing without experiment logging",
                exc,
            )
            self._mlflow_override = _NoOpMlflow()
        mlflow_log = self._mlflow
        with mlflow_log.start_run(run_name="train"):
            mlflow_log.log_param("dc_time_decay_days", self.dc_model.time_decay_days)
            mlflow_log.log_param("num_training_matches", len(training_data))
            num_teams = len(set(m.home_team for m in training_data) | set(m.away_team for m in training_data))
            mlflow_log.log_param("num_teams", num_teams)
            mlflow_log.log_param("challenger_trained", challenger_trained)

            mlflow_log.log_metric("dc_home_advantage", dc_params.home_advantage)
            mlflow_log.log_metric("dc_rho", dc_params.rho)

            # Log Elo ratings for top/bottom teams
            if self.elo_system:
                sorted_elo = sorted(self.elo_system.ratings.items(), key=lambda x: x[1], reverse=True)
                for team, rating in sorted_elo[:5]:
                    safe = "".join(c if c.isalnum() or c in "_-." else "" for c in team).replace(" ", "_")
                    mlflow_log.log_metric(f"elo_{safe}", rating)

            # 5. Evaluate and pick best model
            if run_evaluation and len(training_data) > 100:
                self._evaluate_and_log(matches, challenger_trained)

            # 6. Fit calibrator from real historical predictions generated before matches kicked off
            self._fit_outcome_calibrator()

            # 7. Save models + active model choice
            self.model_version = datetime.now(timezone.utc).isoformat()
            self.save_model()

            mlflow_log.log_param("active_model", self.active_model)
            mlflow_log.log_param("calibration_enabled", bool(self.calibrator and self.calibrator.is_fitted))
            if self.calibrator and self.calibrator.is_fitted:
                mlflow_log.log_param("calibration_version", self.calibrator.version)
            mlflow_log.log_artifact(str(BUNDLE_PATH))

            logger.info(
                f"Training complete. Active model: {self.active_model}. "
                f"DC home_adv={dc_params.home_advantage:.3f}, rho={dc_params.rho:.3f}"
            )

    def _evaluate_and_log(self, raw_matches: list[Match], challenger_trained: bool) -> None:
        """Backtest both models, log metrics, pick the winner by Brier score."""
        finished_sorted = sorted(
            [m for m in raw_matches if m.status == "FINISHED" and m.home_goals is not None and m.away_goals is not None],
            key=lambda m: m.utc_date,
        )
        if len(finished_sorted) < 50:
            return
        cutoff = finished_sorted[int(len(finished_sorted) * 0.8)].utc_date
        train_matches_raw = [m for m in finished_sorted if m.utc_date < cutoff]
        test_matches_raw = [m for m in finished_sorted if m.utc_date >= cutoff]
        reference_date = cutoff.replace(tzinfo=timezone.utc) if cutoff.tzinfo is None else cutoff
        train_split = matches_to_training_data(train_matches_raw, time_decay_days=self.dc_model.time_decay_days, reference_date=reference_date, use_form_weighting=False)
        test_split = matches_to_training_data(test_matches_raw, reference_date=reference_date, use_form_weighting=False)

        if len(test_split) < 10:
            logger.warning("Not enough test matches for evaluation, skipping")
            return

        # Evaluate Dixon-Coles
        dc_eval = DixonColesModel(time_decay_days=self.dc_model.time_decay_days)
        try:
            dc_result = backtest(dc_eval, train_split, test_split)
            self._mlflow.log_metric("dc_outcome_accuracy", dc_result.outcome_accuracy)
            self._mlflow.log_metric("dc_brier_score", dc_result.brier_score)
            self._mlflow.log_metric("dc_log_loss", dc_result.avg_log_loss)
            self._mlflow.log_metric("dc_over25_accuracy", dc_result.over25_accuracy)
            self._mlflow.log_metric("dc_btts_accuracy", dc_result.btts_accuracy)
            self._mlflow.log_metric("dc_test_matches", dc_result.total_matches)
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
        train_elo = EloSystem.from_matches(train_matches_raw)

        eval_challenger = ChallengerModel(time_decay_days=self.challenger.time_decay_days)
        try:
            eval_challenger.fit(train_matches_raw, train_elo, reference_date=reference_date)
        except ValueError as e:
            logger.warning(f"Challenger eval skipped (not enough train data): {e}")
            self.active_model = "dixon_coles"
            return

        # Context for feature computation: only training-period matches
        train_context_desc = sorted(train_matches_raw, key=lambda m: m.utc_date, reverse=True)

        challenger_correct = 0
        challenger_brier = []
        challenger_total = 0

        for raw_match, match_data in zip(test_matches_raw, test_split):
            try:
                pred = eval_challenger.predict_match(
                    match_data.home_team,
                    match_data.away_team,
                    train_elo,
                    train_context_desc,
                    reference_date=raw_match.utc_date.replace(tzinfo=timezone.utc),
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
            self._mlflow.log_metric("gbm_outcome_accuracy", gbm_accuracy)
            self._mlflow.log_metric("gbm_brier_score", gbm_brier)
            self._mlflow.log_metric("gbm_test_matches", challenger_total)
            logger.info(f"GBM eval: outcome={gbm_accuracy:.1%}, brier={gbm_brier:.4f}")

            # Pick winner
            if challenger_total == dc_result.total_matches and gbm_brier < dc_result.brier_score:
                self.active_model = "challenger"
                logger.info(f"Challenger wins: brier {gbm_brier:.4f} < {dc_result.brier_score:.4f}")
            else:
                self.active_model = "dixon_coles"
                logger.info(f"Dixon-Coles wins: brier {dc_result.brier_score:.4f} <= {gbm_brier:.4f}")
        else:
            self.active_model = "dixon_coles"

    def _fit_outcome_calibrator(self) -> None:
        """Fit a calibrator from historical finished matches with stored predictions."""

        latest_prediction_ids = eligible_prediction_ids(self.db, self.active_model)

        rows = (
            self.db.query(Prediction, Match)
            .join(latest_prediction_ids, Prediction.id == latest_prediction_ids.c.latest_id)
            .join(Match, Match.api_id == Prediction.match_api_id)
            .filter(Match.status == "FINISHED", Match.home_goals.isnot(None))
            .order_by(Match.utc_date)
            .all()
        )

        probabilities: list[tuple[float, float, float]] = []
        labels: list[str] = []
        for prediction, match in rows:
            if prediction.model_name not in (None, self.active_model):
                continue

            raw_home = prediction.raw_home_win_prob if prediction.raw_home_win_prob is not None else prediction.home_win_prob
            raw_draw = prediction.raw_draw_prob if prediction.raw_draw_prob is not None else prediction.draw_prob
            raw_away = prediction.raw_away_win_prob if prediction.raw_away_win_prob is not None else prediction.away_win_prob
            probabilities.append((float(raw_home), float(raw_draw), float(raw_away)))

            if match.home_goals > match.away_goals:
                labels.append("home")
            elif match.home_goals == match.away_goals:
                labels.append("draw")
            else:
                labels.append("away")

        if not probabilities:
            self.calibrator = None
            return

        calibrator = OutcomeCalibrator()
        try:
            calibrator.fit(probabilities, labels)
        except ValueError as exc:
            logger.info(f"Outcome calibrator skipped: {exc}")
            self.calibrator = None
            return

        self.calibrator = calibrator
        logger.info(
            "Outcome calibrator fitted on %s historical predictions (%s)",
            len(probabilities),
            calibrator.version,
        )

    @staticmethod
    def _confidence_label(max_probability: float) -> str:
        if max_probability >= 0.60:
            return "high"
        if max_probability >= 0.45:
            return "medium"
        return "low"

    @staticmethod
    def _outcome_score_for_matrix(score_matrix: np.ndarray, predicted_outcome: str, fallback_score: str) -> str:
        """Pick the most likely scoreline consistent with the served outcome probabilities."""

        candidates = []
        size = score_matrix.shape[0]
        for home_goals in range(size):
            for away_goals in range(size):
                if predicted_outcome == "home" and home_goals <= away_goals:
                    continue
                if predicted_outcome == "draw" and home_goals != away_goals:
                    continue
                if predicted_outcome == "away" and home_goals >= away_goals:
                    continue
                candidates.append((score_matrix[home_goals, away_goals], f"{home_goals}-{away_goals}"))

        if not candidates:
            return fallback_score
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _apply_outcome_calibration(self, prediction: MatchPrediction, model_used: str | None = None) -> tuple[float, float, float]:
        """Apply 1X2 calibration to a match prediction if a calibrator is available."""

        raw_probs = (
            float(prediction.home_win_prob),
            float(prediction.draw_prob),
            float(prediction.away_win_prob),
        )
        served_probs = raw_probs

        if self.calibrator and self.calibrator.is_fitted and (model_used is None or model_used == self.active_model):
            served_probs = self.calibrator.transform(raw_probs)

        prediction.home_win_prob = round(float(served_probs[0]), 4)
        prediction.draw_prob = round(float(served_probs[1]), 4)
        prediction.away_win_prob = round(float(served_probs[2]), 4)
        prediction.confidence = self._confidence_label(max(served_probs))

        served_outcome = OUTCOMES[int(np.argmax(served_probs))]
        prediction.outcome_score = self._outcome_score_for_matrix(
            prediction.score_matrix,
            served_outcome,
            prediction.most_likely_score,
        )
        return raw_probs

    def load_model(self) -> None:
        """Load a complete generation without deserializing executable pickle."""
        bundle = load_bundle(BUNDLE_PATH)
        self.dc_model = bundle["dc_model"]
        self.challenger = bundle["challenger"]
        self.elo_system = bundle["elo_system"]
        self.calibrator = bundle["calibrator"]
        self.active_model = bundle["active_model"]
        self.model_version = bundle["model_version"]
        logger.info(f"Models loaded. Active: {self.active_model}")

    def save_model(self) -> None:
        save_bundle(BUNDLE_PATH, {
            "format_version": 1, "dc_model": self.dc_model,
            "challenger": self.challenger, "elo_system": self.elo_system,
            "calibrator": self.calibrator, "active_model": self.active_model,
            "model_version": self.model_version,
        })

    def predict_upcoming(self) -> list[MatchPrediction]:
        """Generate predictions for all upcoming matches using the active model."""
        upcoming = (
            self.db.query(Match)
            .filter(Match.status.in_(["SCHEDULED", "TIMED"]),
                    Match.utc_date > utc_now(), Match.season == str(current_season_year()))
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
                model_used = "dixon_coles"
                pred = None
                if self.active_model == "challenger" and self.challenger.is_fitted and sorted_desc:
                    match_date = match.utc_date
                    if match_date.tzinfo is None:
                        match_date = match_date.replace(tzinfo=timezone.utc)
                    try:
                        pred = self.challenger.predict_match(
                            match.home_team, match.away_team,
                            self.elo_system, sorted_desc,
                            reference_date=match_date,
                        )
                        model_used = "challenger"
                    except (ValueError, KeyError) as e:
                        logger.warning(
                            f"Challenger failed for {match.home_team} vs {match.away_team}, "
                            f"falling back to Dixon-Coles: {e}"
                        )
                if pred is None:
                    pred = self.dc_model.predict_match(match.home_team, match.away_team)

                # Training can run past a kickoff. Do not create a post-match forecast.
                if match.utc_date.replace(tzinfo=None) <= utc_now():
                    continue
                raw_probs = self._apply_outcome_calibration(pred, model_used)
                predictions.append(pred)

                # Keep the forecast history for honest pre-kickoff evaluation.

                # Store prediction (convert np.float64 to float for PostgreSQL)
                db_pred = Prediction(
                    match_api_id=match.api_id,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    predicted_home_goals=float(pred.predicted_home_goals),
                    predicted_away_goals=float(pred.predicted_away_goals),
                    raw_home_win_prob=float(raw_probs[0]),
                    raw_draw_prob=float(raw_probs[1]),
                    raw_away_win_prob=float(raw_probs[2]),
                    home_win_prob=float(pred.home_win_prob),
                    draw_prob=float(pred.draw_prob),
                    away_win_prob=float(pred.away_win_prob),
                    over25_prob=float(pred.over25_prob),
                    btts_prob=float(pred.btts_prob),
                    most_likely_score=pred.most_likely_score,
                    outcome_score=pred.outcome_score,
                    confidence=pred.confidence,
                    model_name=model_used,
                    model_version=self.model_version,
                    created_at=utc_now(),
                    calibration_version=self.calibrator.version if model_used == self.active_model and self.calibrator and self.calibrator.is_fitted else None,
                )
                self.db.add(db_pred)
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not predict {match.home_team} vs {match.away_team}: {e}")

        self.db.commit()
        logger.info(f"Generated {len(predictions)} predictions (model: {self.active_model})")
        return predictions
