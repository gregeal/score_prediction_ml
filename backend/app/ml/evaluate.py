"""Model evaluation, calibration buckets, and benchmark helpers."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from math import log

import numpy as np

from app.ml.dixon_coles import DixonColesModel, MatchData

logger = logging.getLogger(__name__)

OUTCOMES = ("home", "draw", "away")
EPSILON = 1e-15


@dataclass
class EvaluationResult:
    """Aggregate accuracy metrics for a set of predictions."""

    total_matches: int
    outcome_accuracy: float
    exact_score_accuracy: float
    over25_accuracy: float
    btts_accuracy: float
    brier_score: float
    avg_log_loss: float


@dataclass
class EvaluatedPrediction:
    """A scored prediction with optional benchmark probabilities attached."""

    predicted_probs: tuple[float, float, float]
    actual_outcome: str
    match_date: datetime | None = None
    match_api_id: int | None = None
    predicted_score: str | None = None
    actual_score: str | None = None
    over25_prob: float | None = None
    btts_prob: float | None = None
    baseline_probs: tuple[float, float, float] | None = None
    bookmaker_probs: tuple[float, float, float] | None = None

    def probs_for(self, source: str = "model") -> tuple[float, float, float] | None:
        if source == "model":
            return self.predicted_probs
        if source == "naive":
            return self.baseline_probs
        if source == "bookmaker":
            return self.bookmaker_probs
        raise ValueError(f"Unknown probability source: {source}")

    def predicted_outcome_for(self, source: str = "model") -> str | None:
        probs = self.probs_for(source)
        if probs is None:
            return None
        return OUTCOMES[int(np.argmax(probs))]

    def confidence_for(self, source: str = "model") -> float | None:
        probs = self.probs_for(source)
        if probs is None:
            return None
        return float(max(probs))

    def is_correct_for(self, source: str = "model") -> bool | None:
        predicted = self.predicted_outcome_for(source)
        if predicted is None:
            return None
        return predicted == self.actual_outcome

    def brier_for(self, source: str = "model") -> float | None:
        probs = self.probs_for(source)
        if probs is None:
            return None
        actual = outcome_vector(self.actual_outcome)
        return float(sum((prob - truth) ** 2 for prob, truth in zip(probs, actual)))

    def log_loss_for(self, source: str = "model") -> float | None:
        probs = self.probs_for(source)
        if probs is None:
            return None
        actual_index = OUTCOMES.index(self.actual_outcome)
        actual_prob = max(float(probs[actual_index]), EPSILON)
        return float(-log(actual_prob))

    @property
    def exact_score_correct(self) -> bool:
        return self.predicted_score is not None and self.predicted_score == self.actual_score

    @property
    def over25_correct(self) -> bool | None:
        if self.over25_prob is None or self.actual_score is None:
            return None
        home_goals, away_goals = parse_score(self.actual_score)
        actual_over = home_goals + away_goals > 2
        return (self.over25_prob > 0.5) == actual_over

    @property
    def btts_correct(self) -> bool | None:
        if self.btts_prob is None or self.actual_score is None:
            return None
        home_goals, away_goals = parse_score(self.actual_score)
        actual_btts = home_goals > 0 and away_goals > 0
        return (self.btts_prob > 0.5) == actual_btts


@dataclass
class CalibrationBucket:
    """Calibration stats for a confidence range."""

    label: str
    range_start: float
    range_end: float
    avg_confidence: float
    actual_rate: float
    count: int


@dataclass
class SegmentMetrics:
    """Accuracy metrics for a subset of predictions."""

    name: str
    count: int
    outcome_accuracy: float
    brier_score: float
    avg_log_loss: float


@dataclass
class RollingWindowMetrics:
    """Metrics for a rolling evaluation window."""

    label: str
    match_count: int
    outcome_accuracy: float
    brier_score: float
    avg_log_loss: float


@dataclass
class BenchmarkMetrics:
    """Comparable metrics for a single benchmark source."""

    available: bool
    total_matches: int
    outcome_accuracy: float | None = None
    brier_score: float | None = None
    avg_log_loss: float | None = None


@dataclass
class AccuracyDashboardResult:
    """Structured result used by the richer accuracy dashboard."""

    total_evaluated: int
    outcome_accuracy: float
    exact_score_accuracy: float
    over25_accuracy: float
    btts_accuracy: float
    brier_score: float
    avg_log_loss: float
    calibration_buckets: list[CalibrationBucket]
    segments: list[SegmentMetrics]
    rolling_windows: list[RollingWindowMetrics]
    benchmarks: dict[str, BenchmarkMetrics]

    def to_dict(self) -> dict:
        return {
            "total_evaluated": self.total_evaluated,
            "outcome_accuracy": self.outcome_accuracy,
            "exact_score_accuracy": self.exact_score_accuracy,
            "over25_accuracy": self.over25_accuracy,
            "btts_accuracy": self.btts_accuracy,
            "brier_score": self.brier_score,
            "avg_log_loss": self.avg_log_loss,
            "calibration_buckets": [asdict(bucket) for bucket in self.calibration_buckets],
            "segments": [asdict(segment) for segment in self.segments],
            "rolling_windows": [asdict(window) for window in self.rolling_windows],
            "benchmarks": {name: asdict(metrics) for name, metrics in self.benchmarks.items()},
        }


def outcome_vector(outcome: str) -> tuple[int, int, int]:
    """Return one-hot encoding for a match outcome."""

    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown outcome: {outcome}")
    return tuple(1 if candidate == outcome else 0 for candidate in OUTCOMES)


def parse_score(score: str) -> tuple[int, int]:
    """Parse a score string like '2-1'."""

    home, away = score.split("-")
    return int(home), int(away)


def score_prediction(
    *,
    predicted_probs: tuple[float, float, float],
    actual_outcome: str,
    match_date: datetime | None = None,
    match_api_id: int | None = None,
    predicted_score: str | None = None,
    actual_score: str | None = None,
    over25_prob: float | None = None,
    btts_prob: float | None = None,
    baseline_probs: tuple[float, float, float] | None = None,
    bookmaker_probs: tuple[float, float, float] | None = None,
) -> EvaluatedPrediction:
    """Create a scored prediction row."""

    return EvaluatedPrediction(
        predicted_probs=normalize_probs(predicted_probs),
        actual_outcome=actual_outcome,
        match_date=match_date,
        match_api_id=match_api_id,
        predicted_score=predicted_score,
        actual_score=actual_score,
        over25_prob=over25_prob,
        btts_prob=btts_prob,
        baseline_probs=normalize_probs(baseline_probs) if baseline_probs else None,
        bookmaker_probs=normalize_probs(bookmaker_probs) if bookmaker_probs else None,
    )


def normalize_probs(probs: tuple[float, float, float] | list[float] | np.ndarray) -> tuple[float, float, float]:
    """Normalize probabilities so they sum to 1.0."""

    array = np.asarray(probs, dtype=float)
    total = float(array.sum())
    if total <= 0:
        raise ValueError("Probability vector must sum to a positive number")
    normalized = array / total
    return tuple(float(value) for value in normalized)


def evaluate_predictions(
    predictions: list[EvaluatedPrediction],
    probability_source: str = "model",
) -> EvaluationResult:
    """Compute aggregate metrics for a list of evaluated predictions."""

    available = [prediction for prediction in predictions if prediction.probs_for(probability_source) is not None]
    if not available:
        raise ValueError("No predictions could be evaluated")

    outcome_correct = []
    exact_correct = []
    over25_correct = []
    btts_correct = []
    brier_scores = []
    log_losses = []

    for prediction in available:
        outcome = prediction.is_correct_for(probability_source)
        if outcome is not None:
            outcome_correct.append(float(outcome))
        exact_correct.append(float(prediction.exact_score_correct))
        if prediction.over25_correct is not None:
            over25_correct.append(float(prediction.over25_correct))
        if prediction.btts_correct is not None:
            btts_correct.append(float(prediction.btts_correct))
        brier_scores.append(prediction.brier_for(probability_source))
        log_losses.append(prediction.log_loss_for(probability_source))

    return EvaluationResult(
        total_matches=len(available),
        outcome_accuracy=round(float(np.mean(outcome_correct)), 4),
        exact_score_accuracy=round(float(np.mean(exact_correct)), 4),
        over25_accuracy=round(float(np.mean(over25_correct)), 4) if over25_correct else 0.0,
        btts_accuracy=round(float(np.mean(btts_correct)), 4) if btts_correct else 0.0,
        brier_score=round(float(np.mean(brier_scores)), 4),
        avg_log_loss=round(float(np.mean(log_losses)), 4),
    )


def build_calibration_buckets(
    predictions: list[EvaluatedPrediction],
    probability_source: str = "model",
    bucket_size: float = 0.1,
) -> list[CalibrationBucket]:
    """Bucket predictions by confidence and compare predicted vs realized rates."""

    buckets: list[CalibrationBucket] = []
    if not predictions:
        return buckets

    start = 0.0
    while start < 1.0:
        end = min(start + bucket_size, 1.0)
        bucket_predictions = []
        for prediction in predictions:
            confidence = prediction.confidence_for(probability_source)
            if confidence is None:
                continue
            if start <= confidence < end or (end == 1.0 and confidence <= end):
                bucket_predictions.append(prediction)

        if bucket_predictions:
            avg_confidence = float(np.mean([prediction.confidence_for(probability_source) for prediction in bucket_predictions]))
            actual_rate = float(np.mean([prediction.is_correct_for(probability_source) for prediction in bucket_predictions]))
            buckets.append(
                CalibrationBucket(
                    label=f"{int(start * 100)}-{int(end * 100)}%",
                    range_start=round(start, 2),
                    range_end=round(end, 2),
                    avg_confidence=round(avg_confidence, 4),
                    actual_rate=round(actual_rate, 4),
                    count=len(bucket_predictions),
                )
            )
        start = round(start + bucket_size, 10)

    return buckets


def build_segment_metrics(
    predictions: list[EvaluatedPrediction],
    probability_source: str = "model",
) -> list[SegmentMetrics]:
    """Compute metrics for common confidence and outcome segments."""

    selectors = {
        "home_favorites": lambda prediction: prediction.predicted_outcome_for(probability_source) == "home",
        "away_favorites": lambda prediction: prediction.predicted_outcome_for(probability_source) == "away",
        "predicted_draws": lambda prediction: prediction.predicted_outcome_for(probability_source) == "draw",
        "high_confidence": lambda prediction: (prediction.confidence_for(probability_source) or 0.0) >= 0.60,
        "very_high_confidence": lambda prediction: (prediction.confidence_for(probability_source) or 0.0) >= 0.70,
    }

    segments: list[SegmentMetrics] = []
    for name, selector in selectors.items():
        subset = [prediction for prediction in predictions if selector(prediction)]
        if not subset:
            continue
        summary = evaluate_predictions(subset, probability_source=probability_source)
        segments.append(
            SegmentMetrics(
                name=name,
                count=len(subset),
                outcome_accuracy=summary.outcome_accuracy,
                brier_score=summary.brier_score,
                avg_log_loss=summary.avg_log_loss,
            )
        )
    return segments


def build_rolling_window_metrics(
    predictions: list[EvaluatedPrediction],
    probability_source: str = "model",
    window_size: int = 20,
    step_size: int = 20,
) -> list[RollingWindowMetrics]:
    """Compute rolling metrics across chronologically ordered predictions."""

    if not predictions:
        return []

    ordered = sorted(
        predictions,
        key=lambda prediction: prediction.match_date.isoformat() if prediction.match_date else "",
    )

    windows: list[RollingWindowMetrics] = []
    limit = len(ordered)
    for start in range(0, limit, step_size):
        window = ordered[start : start + window_size]
        if len(window) < window_size and windows:
            break
        summary = evaluate_predictions(window, probability_source=probability_source)
        first_date = window[0].match_date
        last_date = window[-1].match_date
        if first_date and last_date:
            label = f"{first_date.date().isoformat()} to {last_date.date().isoformat()}"
        else:
            label = f"Matches {start + 1}-{start + len(window)}"
        windows.append(
            RollingWindowMetrics(
                label=label,
                match_count=len(window),
                outcome_accuracy=summary.outcome_accuracy,
                brier_score=summary.brier_score,
                avg_log_loss=summary.avg_log_loss,
            )
        )
        if len(window) < window_size:
            break
    return windows


def compare_benchmarks(predictions: list[EvaluatedPrediction]) -> dict[str, BenchmarkMetrics]:
    """Compare model performance against available benchmark sources."""

    benchmarks: dict[str, BenchmarkMetrics] = {}
    for source in ("model", "naive", "bookmaker"):
        available = [prediction for prediction in predictions if prediction.probs_for(source) is not None]
        if not available:
            benchmarks[source] = BenchmarkMetrics(available=False, total_matches=0)
            continue

        summary = evaluate_predictions(available, probability_source=source)
        benchmarks[source] = BenchmarkMetrics(
            available=True,
            total_matches=summary.total_matches,
            outcome_accuracy=summary.outcome_accuracy,
            brier_score=summary.brier_score,
            avg_log_loss=summary.avg_log_loss,
        )

    return benchmarks


def build_dashboard_result(predictions: list[EvaluatedPrediction]) -> AccuracyDashboardResult:
    """Build the full structured payload used by the accuracy dashboard."""

    summary = evaluate_predictions(predictions)
    return AccuracyDashboardResult(
        total_evaluated=summary.total_matches,
        outcome_accuracy=summary.outcome_accuracy,
        exact_score_accuracy=summary.exact_score_accuracy,
        over25_accuracy=summary.over25_accuracy,
        btts_accuracy=summary.btts_accuracy,
        brier_score=summary.brier_score,
        avg_log_loss=summary.avg_log_loss,
        calibration_buckets=build_calibration_buckets(predictions),
        segments=build_segment_metrics(predictions),
        rolling_windows=build_rolling_window_metrics(predictions),
        benchmarks=compare_benchmarks(predictions),
    )


def backtest(
    model: DixonColesModel,
    train_matches: list[MatchData],
    test_matches: list[MatchData],
) -> EvaluationResult:
    """Evaluate model accuracy on held-out test matches."""

    model.fit(train_matches)
    evaluated_predictions: list[EvaluatedPrediction] = []

    for match in test_matches:
        try:
            prediction = model.predict_match(match.home_team, match.away_team)
        except ValueError:
            continue

        if match.home_goals > match.away_goals:
            actual_outcome = "home"
        elif match.home_goals == match.away_goals:
            actual_outcome = "draw"
        else:
            actual_outcome = "away"

        evaluated_predictions.append(
            score_prediction(
                predicted_probs=(
                    prediction.home_win_prob,
                    prediction.draw_prob,
                    prediction.away_win_prob,
                ),
                actual_outcome=actual_outcome,
                predicted_score=prediction.most_likely_score,
                actual_score=f"{match.home_goals}-{match.away_goals}",
                over25_prob=prediction.over25_prob,
                btts_prob=prediction.btts_prob,
            )
        )

    if not evaluated_predictions:
        raise ValueError("No test matches could be evaluated")

    return evaluate_predictions(evaluated_predictions)
