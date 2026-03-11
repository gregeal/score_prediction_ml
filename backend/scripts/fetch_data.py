"""Fetch historical EPL data and store in database."""

import sys
import os
import logging

import requests

# Add backend to path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.base import Base, engine, SessionLocal
from app.models.match import Match
from app.services.data_fetcher import FootballDataFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Seasons to fetch (starting year). Free API tier may only support recent seasons.
SEASONS = [2022, 2023, 2024, 2025]


def main():
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")

    fetcher = FootballDataFetcher()
    db = SessionLocal()

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

            season_added = 0
            for match_data in parsed_matches:
                # Skip if already exists
                existing = db.query(Match).filter_by(api_id=match_data["api_id"]).first()
                if existing:
                    continue

                match = Match(**match_data)
                db.add(match)
                season_added += 1
                total_added += 1

            db.commit()
            logger.info(f"Season {season}/{season + 1}: added {season_added} new matches")

        logger.info(f"Done. Total new matches added: {total_added}. "
                     f"Total in database: {db.query(Match).count()}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
