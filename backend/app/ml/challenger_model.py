"""Gradient-boosted challenger model using features + Dixon-Coles base."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from app.ml.dixon_coles import DixonColesModel, MatchData, MatchPrediction
from app.ml.elo import EloSystem
from app.ml.features import (
    MatchFeatures,
    build_match_features,
    matches_to_training_data,
)

logger = logging.getLogger(__name__)


class ChallengerModel:
    """GBM classifier that reweights Dixon-Coles score matrix with richer features.

    Training features are built strictly walk-forward so they match what is
    available at prediction time: Elo ratings are replayed chronologically and
    snapshotted before each match, and the Dixon-Coles xG features come from
    models refit periodically on only the matches played before each row.
    """

    # Skip the first rows so every training example has some form history.
    MIN_HISTORY = 20
    # Refit the walk-forward Dixon-Coles model every N matches during training.
    DC_REFIT_EVERY = 40

    def __init__(self, time_decay_days: int = 365):
        self.classifier = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.dixon_coles = DixonColesModel(time_decay_days=time_decay_days)
        self.is_fitted = False
        self.time_decay_days = time_decay_days

    def fit(self, matches: list, elo_system: EloSystem | None = None) -> None:
        """Train the GBM on features extracted from historical matches.

        Args:
            matches: Match objects sorted by utc_date ascending.
            elo_system: Unused for training rows (kept for API compatibility).
                Ratings are replayed chronologically inside this method so each
                training row sees only pre-match Elo, exactly as at serving time.
        """
        # Train Dixon-Coles on the full history: this is the base model used
        # at prediction time (never for historical feature rows).
        training_data = matches_to_training_data(matches, use_form_weighting=False)
        if len(training_data) < 50:
            raise ValueError("Need at least 50 matches to train challenger model")
        self.dixon_coles.fit(training_data)

        finished = [m for m in matches if m.status == "FINISHED" and m.home_goals is not None]
        sorted_by_date = sorted(finished, key=lambda m: m.utc_date)

        rolling_elo = EloSystem()
        walk_forward_dc: DixonColesModel | None = None
        last_refit_index = -1

        X, y = [], []
        for i, match in enumerate(sorted_by_date):
            if i >= self.MIN_HISTORY:
                match_date = match.utc_date
                if match_date.tzinfo is None:
                    match_date = match_date.replace(tzinfo=timezone.utc)

                # Refit the walk-forward DC model every DC_REFIT_EVERY matches
                # on strictly prior matches, so xG features for row i never see
                # match i or anything after it.
                if walk_forward_dc is None or i - last_refit_index >= self.DC_REFIT_EVERY:
                    prior_data = matches_to_training_data(
                        sorted_by_date[:i],
                        time_decay_days=self.time_decay_days,
                        reference_date=match_date,
                        use_form_weighting=False,
                    )
                    try:
                        candidate = DixonColesModel(time_decay_days=self.time_decay_days)
                        candidate.fit(prior_data)
                        walk_forward_dc = candidate
                        last_refit_index = i
                    except ValueError:
                        walk_forward_dc = None

                if walk_forward_dc is not None:
                    try:
                        dc_pred = walk_forward_dc.predict_match(match.home_team, match.away_team)
                    except ValueError:
                        dc_pred = None

                    if dc_pred is not None:
                        # Feature context: only matches before this one (no leakage)
                        context = sorted_by_date[i - 1 :: -1] if i > 0 else []

                        features = build_match_features(
                            matches=context,
                            home_team=match.home_team,
                            away_team=match.away_team,
                            elo_system=rolling_elo,  # pre-match snapshot
                            dc_home_xg=float(dc_pred.predicted_home_goals),
                            dc_away_xg=float(dc_pred.predicted_away_goals),
                            reference_date=match_date,
                            promotion_context=sorted_by_date,
                        )
                        X.append(features.to_vector())

                        # Label: 0=home, 1=draw, 2=away
                        if match.home_goals > match.away_goals:
                            y.append(0)
                        elif match.home_goals == match.away_goals:
                            y.append(1)
                        else:
                            y.append(2)

            # Update rolling Elo AFTER extracting features for this match
            rolling_elo.update(match.home_team, match.away_team, match.home_goals, match.away_goals)

        if len(X) < 50:
            raise ValueError(f"Only {len(X)} feature samples, need at least 50")

        self.classifier.fit(X, y)
        self.is_fitted = True
        logger.info(f"Challenger model trained on {len(X)} samples")

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        elo_system: EloSystem,
        matches: list,
        reference_date: datetime | None = None,
    ) -> MatchPrediction:
        """Predict using GBM probabilities reweighting Dixon-Coles score matrix."""
        if not self.is_fitted:
            raise ValueError("Challenger model not fitted. Call fit() first.")

        if reference_date is None:
            reference_date = datetime.now(timezone.utc)

        # Get Dixon-Coles base prediction
        dc_pred = self.dixon_coles.predict_match(home_team, away_team)

        # Build features
        features = build_match_features(
            matches=matches,
            home_team=home_team,
            away_team=away_team,
            elo_system=elo_system,
            dc_home_xg=dc_pred.predicted_home_goals,
            dc_away_xg=dc_pred.predicted_away_goals,
            reference_date=reference_date,
        )

        # GBM probabilities [home, draw, away]
        gbm_probs = self.classifier.predict_proba([features.to_vector()])[0]

        # Reweight Dixon-Coles score matrix
        dc_home = dc_pred.home_win_prob
        dc_draw = dc_pred.draw_prob
        dc_away = dc_pred.away_win_prob

        eps = 1e-10
        home_ratio = gbm_probs[0] / max(dc_home, eps)
        draw_ratio = gbm_probs[1] / max(dc_draw, eps)
        away_ratio = gbm_probs[2] / max(dc_away, eps)

        matrix = dc_pred.score_matrix.copy()
        n = matrix.shape[0]
        home_idx, away_idx = np.indices((n, n))
        matrix[home_idx > away_idx] *= home_ratio
        matrix[home_idx == away_idx] *= draw_ratio
        matrix[home_idx < away_idx] *= away_ratio

        # Clip any negative values (can arise from Dixon-Coles rho correction)
        matrix = np.clip(matrix, 0, None)

        # Renormalize
        matrix /= matrix.sum()

        # Build prediction from adjusted matrix
        prediction = MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            score_matrix=matrix,
            predicted_home_goals=dc_pred.predicted_home_goals,
            predicted_away_goals=dc_pred.predicted_away_goals,
        )
        DixonColesModel._derive_predictions(prediction, matrix)
        return prediction
