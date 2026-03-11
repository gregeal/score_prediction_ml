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
    """GBM classifier that reweights Dixon-Coles score matrix with richer features."""

    def __init__(self, time_decay_days: int = 365):
        self.classifier = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.dixon_coles = DixonColesModel(time_decay_days=time_decay_days)
        self.is_fitted = False
        self.time_decay_days = time_decay_days

    def fit(self, matches: list, elo_system: EloSystem) -> None:
        """Train the GBM on features extracted from historical matches.

        Args:
            matches: Match objects sorted by utc_date ascending.
            elo_system: Pre-built EloSystem with ratings from these matches.
        """
        # First train Dixon-Coles as our base model
        training_data = matches_to_training_data(matches, use_form_weighting=False)
        if len(training_data) < 50:
            raise ValueError("Need at least 50 matches to train challenger model")
        self.dixon_coles.fit(training_data)

        # Build feature vectors and labels for each finished match
        finished = [m for m in matches if m.status == "FINISHED" and m.home_goals is not None]
        sorted_by_date = sorted(finished, key=lambda m: m.utc_date)

        X, y = [], []
        # Start from match 20+ so we have form history
        for i in range(20, len(sorted_by_date)):
            match = sorted_by_date[i]
            # Use only matches before this one for features (no leakage)
            context = sorted(sorted_by_date[:i], key=lambda m: m.utc_date, reverse=True)

            try:
                dc_home_xg = float(np.exp(
                    self.dixon_coles.params.attack[match.home_team]
                    + self.dixon_coles.params.defense[match.away_team]
                    + self.dixon_coles.params.home_advantage
                ))
                dc_away_xg = float(np.exp(
                    self.dixon_coles.params.attack[match.away_team]
                    + self.dixon_coles.params.defense[match.home_team]
                ))
            except KeyError:
                continue  # Unknown team

            match_date = match.utc_date
            if match_date.tzinfo is None:
                match_date = match_date.replace(tzinfo=timezone.utc)

            features = build_match_features(
                matches=context,
                home_team=match.home_team,
                away_team=match.away_team,
                elo_system=elo_system,
                dc_home_xg=dc_home_xg,
                dc_away_xg=dc_away_xg,
                reference_date=match_date,
            )
            X.append(features.to_vector())

            # Label: 0=home, 1=draw, 2=away
            if match.home_goals > match.away_goals:
                y.append(0)
            elif match.home_goals == match.away_goals:
                y.append(1)
            else:
                y.append(2)

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
        for i in range(n):
            for j in range(n):
                if i > j:
                    matrix[i, j] *= home_ratio
                elif i == j:
                    matrix[i, j] *= draw_ratio
                else:
                    matrix[i, j] *= away_ratio

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
