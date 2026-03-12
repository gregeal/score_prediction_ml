"""Run the full data-refresh pipeline in one command."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_STEPS = (
    "fetch_data.py",
    "fetch_market_odds.py",
    "train_model.py",
)


def main() -> None:
    for script_name in PIPELINE_STEPS:
        script_path = SCRIPT_DIR / script_name
        logger.info("Running %s", script_name)
        subprocess.run([sys.executable, str(script_path)], check=True)

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
