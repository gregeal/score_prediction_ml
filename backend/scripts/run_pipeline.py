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

# Bookmaker odds only feed the benchmark dashboard; a broken upstream odds
# source must not prevent fetching results and retraining the model.
OPTIONAL_STEPS = frozenset({"fetch_market_odds.py"})


def main() -> None:
    for script_name in PIPELINE_STEPS:
        script_path = SCRIPT_DIR / script_name
        logger.info("Running %s", script_name)
        if script_name in OPTIONAL_STEPS:
            result = subprocess.run([sys.executable, str(script_path)])
            if result.returncode != 0:
                logger.warning(
                    "Optional step %s failed with exit code %s; continuing",
                    script_name,
                    result.returncode,
                )
        else:
            subprocess.run([sys.executable, str(script_path)], check=True)

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
