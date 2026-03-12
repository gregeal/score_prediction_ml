"""Fetch sports-betting market odds and sync them into the database."""

from __future__ import annotations

import logging
import os
import sys

# Add backend to path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.base import ensure_database_ready, get_session_local
from app.models.market_odds import MarketOdds
from app.models.match import Match
from app.services.odds_provider import SportsBettingOddsFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEASONS = [2022, 2023, 2024, 2025]


def sync_market_odds(db, odds_rows: list[dict]) -> tuple[int, int]:
    """Insert new odds rows and update existing ones for the same source."""

    added = 0
    updated = 0

    for row in odds_rows:
        existing = (
            db.query(MarketOdds)
            .filter(
                MarketOdds.match_api_id == row["match_api_id"],
                MarketOdds.source == row["source"],
            )
            .first()
        )
        if existing:
            changed = False
            for field in (
                "captured_at",
                "home_win_odds",
                "draw_odds",
                "away_win_odds",
                "over25_odds",
                "under25_odds",
                "btts_yes_odds",
                "btts_no_odds",
            ):
                new_value = row[field]
                if getattr(existing, field) != new_value:
                    setattr(existing, field, new_value)
                    changed = True
            if changed:
                updated += 1
        else:
            db.add(MarketOdds(**row))
            added += 1

    return added, updated


def main():
    ensure_database_ready()
    logger.info("Database tables created")

    db = get_session_local()()
    fetcher = SportsBettingOddsFetcher(seasons=SEASONS)

    try:
        matches = db.query(Match).all()
        odds_rows, unmatched = fetcher.build_market_odds_rows(matches, include_fixtures=True)
        added, updated = sync_market_odds(db, odds_rows)
        db.commit()

        logger.info(
            "Sports-betting odds sync complete: added=%s updated=%s unmatched=%s total_rows=%s",
            added,
            updated,
            unmatched,
            db.query(MarketOdds).count(),
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
