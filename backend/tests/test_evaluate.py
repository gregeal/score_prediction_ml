"""Tests for evaluation, calibration buckets, and benchmarks."""

from datetime import datetime, timedelta, timezone

from app.ml.evaluate import (
    build_calibration_buckets,
    build_dashboard_result,
    build_rolling_window_metrics,
    compare_benchmarks,
    evaluate_predictions,
    score_prediction,
)


def _prediction(day: int, predicted_probs, actual_outcome, baseline_probs=None, bookmaker_probs=None):
    date = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    actual_score = "2-1" if actual_outcome == "home" else "1-1" if actual_outcome == "draw" else "0-1"
    predicted_score = "2-1" if predicted_probs[0] >= max(predicted_probs[1], predicted_probs[2]) else "1-1"
    return score_prediction(
        predicted_probs=predicted_probs,
        actual_outcome=actual_outcome,
        match_date=date,
        predicted_score=predicted_score,
        actual_score=actual_score,
        over25_prob=0.55,
        btts_prob=0.52,
        baseline_probs=baseline_probs,
        bookmaker_probs=bookmaker_probs,
    )


class TestEvaluate:
    def test_evaluate_predictions_returns_expected_metrics(self):
        predictions = [
            _prediction(1, (0.62, 0.2, 0.18), "home"),
            _prediction(2, (0.2, 0.58, 0.22), "draw"),
            _prediction(3, (0.25, 0.25, 0.5), "away"),
        ]

        result = evaluate_predictions(predictions)

        assert result.total_matches == 3
        assert result.outcome_accuracy == 1.0
        assert 0.0 <= result.brier_score <= 2.0
        assert 0.0 <= result.avg_log_loss <= 5.0

    def test_build_calibration_buckets_groups_by_confidence(self):
        predictions = [
            _prediction(1, (0.62, 0.2, 0.18), "home"),
            _prediction(2, (0.66, 0.18, 0.16), "away"),
            _prediction(3, (0.51, 0.26, 0.23), "home"),
        ]

        buckets = build_calibration_buckets(predictions)

        assert buckets
        assert any(bucket.count > 0 for bucket in buckets)
        assert all(0.0 <= bucket.avg_confidence <= 1.0 for bucket in buckets)

    def test_build_rolling_window_metrics_uses_chronological_windows(self):
        predictions = [
            _prediction(day, (0.62, 0.2, 0.18), "home")
            for day in range(25)
        ]

        windows = build_rolling_window_metrics(predictions, window_size=10, step_size=10)

        assert len(windows) == 2
        assert windows[0].match_count == 10
        assert windows[0].label.startswith("2026-01-01")

    def test_compare_benchmarks_handles_optional_sources(self):
        predictions = [
            _prediction(
                1,
                (0.62, 0.2, 0.18),
                "home",
                baseline_probs=(0.45, 0.28, 0.27),
                bookmaker_probs=(0.58, 0.22, 0.20),
            ),
            _prediction(
                2,
                (0.2, 0.58, 0.22),
                "draw",
                baseline_probs=(0.45, 0.28, 0.27),
            ),
        ]

        benchmarks = compare_benchmarks(predictions)

        assert benchmarks["model"].available is True
        assert benchmarks["naive"].available is True
        assert benchmarks["bookmaker"].available is True
        assert benchmarks["bookmaker"].total_matches == 1

    def test_build_dashboard_result_includes_nested_metrics(self):
        predictions = [
            _prediction(1, (0.62, 0.2, 0.18), "home", baseline_probs=(0.45, 0.28, 0.27)),
            _prediction(2, (0.2, 0.58, 0.22), "draw", baseline_probs=(0.45, 0.28, 0.27)),
            _prediction(3, (0.24, 0.23, 0.53), "away", baseline_probs=(0.45, 0.28, 0.27)),
        ]

        dashboard = build_dashboard_result(predictions)

        assert dashboard.total_evaluated == 3
        assert dashboard.calibration_buckets
        assert dashboard.segments
        assert dashboard.rolling_windows
        assert dashboard.benchmarks["naive"].available is True
