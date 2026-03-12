"""Post-hoc calibration for multiclass football outcome probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from app.ml.evaluate import OUTCOMES, normalize_probs


@dataclass
class BinaryCalibrator:
    """Calibrate one outcome vs the rest."""

    mode: str
    model: object

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.mode == "isotonic":
            return np.asarray(self.model.predict(scores), dtype=float)
        return np.asarray(self.model.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=float)


class OutcomeCalibrator:
    """One-vs-rest calibrator for 1X2 probabilities."""

    def __init__(
        self,
        min_samples: int = 30,
        isotonic_min_samples: int = 80,
        min_class_examples: int = 5,
    ):
        self.min_samples = min_samples
        self.isotonic_min_samples = isotonic_min_samples
        self.min_class_examples = min_class_examples
        self.calibrators: dict[str, BinaryCalibrator] = {}
        self.mode: str | None = None
        self.is_fitted = False
        self.version = "uncalibrated"

    def fit(self, probabilities: list[tuple[float, float, float]] | np.ndarray, labels: list[str]) -> None:
        """Fit calibrators for the three 1X2 outcomes."""

        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 2 or probs.shape[1] != 3:
            raise ValueError("Probabilities must be a 2D array with shape (n_samples, 3)")
        if len(probs) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} samples to fit calibrator")
        if len(labels) != len(probs):
            raise ValueError("Labels and probabilities must have the same length")

        use_isotonic = len(probs) >= self.isotonic_min_samples
        self.mode = "isotonic" if use_isotonic else "sigmoid"
        fitted: dict[str, BinaryCalibrator] = {}

        for index, outcome in enumerate(OUTCOMES):
            binary_labels = np.asarray([1 if label == outcome else 0 for label in labels], dtype=int)
            positives = int(binary_labels.sum())
            negatives = int(len(binary_labels) - positives)
            if positives < self.min_class_examples or negatives < self.min_class_examples:
                raise ValueError(f"Not enough class examples to calibrate {outcome}")

            scores = probs[:, index]
            if use_isotonic:
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(scores, binary_labels)
            else:
                model = LogisticRegression(random_state=42)
                model.fit(scores.reshape(-1, 1), binary_labels)
            fitted[outcome] = BinaryCalibrator(mode=self.mode, model=model)

        self.calibrators = fitted
        self.is_fitted = True
        self.version = f"ovr-{self.mode}-v1"

    def transform(self, probabilities: tuple[float, float, float] | list[float] | np.ndarray) -> tuple[float, float, float]:
        """Calibrate a single probability vector."""

        if not self.is_fitted:
            raise ValueError("Calibrator is not fitted")

        probs = np.asarray(probabilities, dtype=float).reshape(1, 3)
        calibrated = []
        for index, outcome in enumerate(OUTCOMES):
            calibrated_value = self.calibrators[outcome].predict(probs[:, index])[0]
            calibrated.append(float(calibrated_value))
        return normalize_probs(calibrated)
