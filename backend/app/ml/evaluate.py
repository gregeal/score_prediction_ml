"""Model evaluation and accuracy tracking."""

import logging
from dataclasses import dataclass

import numpy as np

from app.ml.dixon_coles import DixonColesModel, MatchData

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from backtesting the model."""
    total_matches: int
    outcome_accuracy: float  # % of 1X2 outcomes predicted correctly
    exact_score_accuracy: float  # % of exact scores predicted correctly
    over25_accuracy: float  # % of O/U 2.5 predicted correctly
    btts_accuracy: float  # % of BTTS predicted correctly
    brier_score: float  # Brier score for 1X2 probabilities (lower = better)
    avg_log_loss: float  # Average log loss (lower = better)


def backtest(
    model: DixonColesModel,
    train_matches: list[MatchData],
    test_matches: list[MatchData],
) -> EvaluationResult:
    """Evaluate model accuracy on held-out test matches.

    Fits the model on training data and evaluates predictions against
    actual results in the test set.

    Args:
        model: DixonColesModel instance.
        train_matches: Training data (earlier matches).
        test_matches: Test data (later matches to predict).

    Returns:
        EvaluationResult with accuracy metrics.
    """
    model.fit(train_matches)

    correct_outcome = 0
    correct_exact = 0
    correct_over25 = 0
    correct_btts = 0
    brier_scores = []
    log_losses = []
    total = 0

    for match in test_matches:
        try:
            pred = model.predict_match(match.home_team, match.away_team)
        except ValueError:
            # Unknown team (e.g., newly promoted)
            continue

        total += 1

        # Actual outcome
        if match.home_goals > match.away_goals:
            actual_outcome = "home"
            actual_vec = [1, 0, 0]
        elif match.home_goals == match.away_goals:
            actual_outcome = "draw"
            actual_vec = [0, 1, 0]
        else:
            actual_outcome = "away"
            actual_vec = [0, 0, 1]

        # Predicted outcome
        probs = [pred.home_win_prob, pred.draw_prob, pred.away_win_prob]
        predicted_outcome = ["home", "draw", "away"][np.argmax(probs)]

        if predicted_outcome == actual_outcome:
            correct_outcome += 1

        # Exact score
        actual_score = f"{match.home_goals}-{match.away_goals}"
        if pred.most_likely_score == actual_score:
            correct_exact += 1

        # Over/Under 2.5
        actual_total = match.home_goals + match.away_goals
        predicted_over = pred.over25_prob > 0.5
        actual_over = actual_total > 2
        if predicted_over == actual_over:
            correct_over25 += 1

        # BTTS
        predicted_btts = pred.btts_prob > 0.5
        actual_btts = match.home_goals > 0 and match.away_goals > 0
        if predicted_btts == actual_btts:
            correct_btts += 1

        # Brier score: mean squared error of probability estimates
        pred_vec = [pred.home_win_prob, pred.draw_prob, pred.away_win_prob]
        brier = sum((p - a) ** 2 for p, a in zip(pred_vec, actual_vec))
        brier_scores.append(brier)

        # Log loss
        epsilon = 1e-15
        prob_of_actual = max(pred_vec[actual_vec.index(1)], epsilon)
        log_losses.append(-np.log(prob_of_actual))

    if total == 0:
        raise ValueError("No test matches could be evaluated")

    return EvaluationResult(
        total_matches=total,
        outcome_accuracy=round(correct_outcome / total, 4),
        exact_score_accuracy=round(correct_exact / total, 4),
        over25_accuracy=round(correct_over25 / total, 4),
        btts_accuracy=round(correct_btts / total, 4),
        brier_score=round(float(np.mean(brier_scores)), 4),
        avg_log_loss=round(float(np.mean(log_losses)), 4),
    )
