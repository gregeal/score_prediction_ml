"""Fetch EPL match data from football-data.org API."""

import time
import logging
from datetime import datetime

import requests

from app.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"
EPL_COMPETITION_CODE = "PL"

# Free tier: 10 requests per minute
RATE_LIMIT_DELAY = 6.5  # seconds between requests


class FootballDataFetcher:
    """Client for the football-data.org API (free tier)."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.football_data_api_key
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": self.api_key})
        self._last_request_time = 0.0

    def _rate_limited_get(self, url: str, params: dict | None = None) -> dict:
        """Make a GET request with rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)

        response = self.session.get(url, params=params)
        self._last_request_time = time.time()

        if response.status_code == 429:
            logger.warning("Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            return self._rate_limited_get(url, params)

        response.raise_for_status()
        return response.json()

    def fetch_matches(self, season: int) -> list[dict]:
        """Fetch all EPL matches for a season.

        Args:
            season: The starting year of the season (e.g., 2025 for 2025/26).

        Returns:
            List of match dicts from the API.
        """
        url = f"{BASE_URL}/competitions/{EPL_COMPETITION_CODE}/matches"
        data = self._rate_limited_get(url, params={"season": season})
        matches = data.get("matches", [])
        logger.info(f"Fetched {len(matches)} matches for {season}/{season + 1} season")
        return matches

    def fetch_upcoming_fixtures(self) -> list[dict]:
        """Fetch upcoming (scheduled/timed) EPL fixtures.

        Returns:
            List of upcoming match dicts.
        """
        url = f"{BASE_URL}/competitions/{EPL_COMPETITION_CODE}/matches"
        data = self._rate_limited_get(
            url, params={"status": "SCHEDULED,TIMED"}
        )
        matches = data.get("matches", [])
        logger.info(f"Fetched {len(matches)} upcoming fixtures")
        return matches

    def fetch_standings(self) -> dict:
        """Fetch current EPL standings.

        Returns:
            Standings data dict.
        """
        url = f"{BASE_URL}/competitions/{EPL_COMPETITION_CODE}/standings"
        data = self._rate_limited_get(url)
        logger.info("Fetched current standings")
        return data

    def fetch_teams(self) -> list[dict]:
        """Fetch all EPL teams for the current season.

        Returns:
            List of team dicts.
        """
        url = f"{BASE_URL}/competitions/{EPL_COMPETITION_CODE}/teams"
        data = self._rate_limited_get(url)
        teams = data.get("teams", [])
        logger.info(f"Fetched {len(teams)} teams")
        return teams

    @staticmethod
    def parse_match(raw: dict) -> dict:
        """Parse a raw API match dict into our internal format.

        Args:
            raw: Raw match dict from football-data.org API.

        Returns:
            Cleaned match dict matching our Match model fields.
        """
        score = raw.get("score", {})
        full_time = score.get("fullTime", {})

        return {
            "api_id": raw["id"],
            "season": raw.get("season", {}).get("startDate", "")[:4],
            "matchday": raw.get("matchday"),
            "utc_date": datetime.fromisoformat(
                raw["utcDate"].replace("Z", "+00:00")
            ),
            "status": raw.get("status", "SCHEDULED"),
            "home_team": raw.get("homeTeam", {}).get("name", "Unknown"),
            "away_team": raw.get("awayTeam", {}).get("name", "Unknown"),
            "home_goals": full_time.get("home"),
            "away_goals": full_time.get("away"),
        }

    def fetch_and_parse_season(self, season: int) -> list[dict]:
        """Fetch and parse all matches for a season.

        Args:
            season: Starting year of the season.

        Returns:
            List of parsed match dicts.
        """
        raw_matches = self.fetch_matches(season)
        return [self.parse_match(m) for m in raw_matches]
