"""Fetch historical EPL data and store in database."""

import sys
import os
import logging

import requests

# Add backend to path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.base import ensure_database_ready, get_session_local
from app.models.match import Match
from app.services.data_fetcher import FootballDataFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Seasons to fetch (starting year). Free API tier may only support recent seasons.
SEASONS = [2022, 2023, 2024, 2025]
SYNC_FIELDS = (
    "season",
    "matchday",
    "utc_date",
    "status",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)


def sync_parsed_matches(db, parsed_matches: list[dict]) -> tuple[int, int]:
    """Insert new matches and update mutable fields on existing ones."""
    added = 0
    updated = 0

    for match_data in parsed_matches:
        existing = db.query(Match).filter_by(api_id=match_data["api_id"]).first()
        if existing:
            changed = False
            for field in SYNC_FIELDS:
                new_value = match_data[field]
                if getattr(existing, field) != new_value:
                    setattr(existing, field, new_value)
                    changed = True
            if changed:
                updated += 1
        else:
            db.add(Match(**match_data))
            added += 1

    return added, updated


def main():
    # Create tables and backfill any newly added nullable columns.
    ensure_database_ready()
    logger.info("Database tables created")

    fetcher = FootballDataFetcher()
    db = get_session_local()()

    try:
        total_added = 0
        for season in SEASONS:
            logger.info(f"Fetching {season}/{season + 1} season...")
            try:
                parsed_matches = fetcher.fetch_and_parse_season(season)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 403:
                    logger.warning(
                        f"Season {season}/{season + 1}: 403 Forbidden - "
                        f"free API tier may not support this season, skipping"
                    )
                    continue
                raise

            season_added, season_updated = sync_parsed_matches(db, parsed_matches)
            total_added += season_added

            db.commit()
            logger.info(
                f"Season {season}/{season + 1}: "
                f"added {season_added} new, updated {season_updated} existing"
            )

        logger.info(f"Done. Total new matches added: {total_added}. "
                     f"Total in database: {db.query(Match).count()}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
