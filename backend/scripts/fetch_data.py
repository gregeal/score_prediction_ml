"""Fetch historical EPL data and store in database."""

import sys
import os
import logging

# Add backend to path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.base import Base, engine, SessionLocal
from app.models.match import Match
from app.services.data_fetcher import FootballDataFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Seasons to fetch (starting year)
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
            parsed_matches = fetcher.fetch_and_parse_season(season)

            for match_data in parsed_matches:
                # Skip if already exists
                existing = db.query(Match).filter_by(api_id=match_data["api_id"]).first()
                if existing:
                    continue

                match = Match(**match_data)
                db.add(match)
                total_added += 1

            db.commit()
            logger.info(f"Season {season}/{season + 1}: added {total_added} new matches")

        logger.info(f"Done. Total matches in database: {db.query(Match).count()}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
