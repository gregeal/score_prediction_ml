"""Tests for 1X2 probability calibration."""

import pytest

from app.ml.calibration import OutcomeCalibrator


def _sample_rows(count: int) -> tuple[list[tuple[float, float, float]], list[str]]:
    probabilities = []
    labels = []
    base = [
        ((0.7, 0.2, 0.1), "home"),
        ((0.2, 0.55, 0.25), "draw"),
        ((0.15, 0.2, 0.65), "away"),
    ]
    for index in range(count):
        probs, label = base[index % len(base)]
        probabilities.append(probs)
        labels.append(label)
    return probabilities, labels


class TestOutcomeCalibrator:
    def test_fit_uses_sigmoid_for_smaller_samples(self):
        probabilities, labels = _sample_rows(45)
        calibrator = OutcomeCalibrator(min_samples=30, isotonic_min_samples=80)

        calibrator.fit(probabilities, labels)

        assert calibrator.is_fitted is True
        assert calibrator.mode == "sigmoid"
        assert calibrator.version == "ovr-sigmoid-v1"

    def test_fit_uses_isotonic_when_history_is_large_enough(self):
        probabilities, labels = _sample_rows(90)
        calibrator = OutcomeCalibrator(min_samples=30, isotonic_min_samples=80)

        calibrator.fit(probabilities, labels)

        assert calibrator.is_fitted is True
        assert calibrator.mode == "isotonic"
        assert calibrator.version == "ovr-isotonic-v1"

    def test_transform_returns_normalized_vector(self):
        probabilities, labels = _sample_rows(90)
        calibrator = OutcomeCalibrator(min_samples=30, isotonic_min_samples=80)
        calibrator.fit(probabilities, labels)

        calibrated = calibrator.transform((0.62, 0.21, 0.17))

        assert len(calibrated) == 3
        assert pytest.approx(sum(calibrated), rel=1e-6) == 1.0
        assert all(0.0 <= value <= 1.0 for value in calibrated)

    def test_fit_requires_minimum_samples(self):
        probabilities, labels = _sample_rows(10)
        calibrator = OutcomeCalibrator(min_samples=30)

        with pytest.raises(ValueError, match="Need at least 30 samples"):
            calibrator.fit(probabilities, labels)
