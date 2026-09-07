"""Atomic, explicitly type-checked model storage. Never load legacy pickle files."""

import os
import tempfile
from pathlib import Path

import skops.io as sio

# Do not replace this with the untrusted types returned by an input file.
TRUSTED_TYPES = [
    "app.ml.dixon_coles.DixonColesModel", "app.ml.dixon_coles.ModelParams",
    "app.ml.challenger_model.ChallengerModel", "app.ml.elo.EloSystem",
    "app.ml.calibration.OutcomeCalibrator", "app.ml.calibration.BinaryCalibrator",
    "sklearn._loss.loss.HalfMultinomialLoss", "sklearn._loss.link.MultinomialLogit",
    "sklearn._loss._loss.CyHalfMultinomialLoss",
]


def save_bundle(path: Path, bundle: dict) -> None:
    data = sio.dumps(bundle)
    # Validate round-trip before replacing the last usable generation.
    sio.loads(data, trusted=TRUSTED_TYPES)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError("No safe model bundle. Run scripts/train_model.py to retrain; legacy pickle files are not loaded.")
    data = sio.loads(path.read_bytes(), trusted=TRUSTED_TYPES)
    if not isinstance(data, dict) or data.get("format_version") != 1:
        raise ValueError("Unsupported model bundle")
    if data.get("active_model") not in ("dixon_coles", "challenger"):
        raise ValueError("Invalid active model")
    return data
