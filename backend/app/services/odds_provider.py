"""Odds ingestion from the sports-betting soccer modelling data source."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import pandas as pd

from app.models.match import Match

logger = logging.getLogger(__name__)

SPORTSBET_SOURCE = "sports-betting"
SPORTSBET_EPL_URL = (
    "https://raw.githubusercontent.com/georgedouzas/sports-betting/"
    "data/data/soccer/modelling/England_1_{year}.csv"
)
SPORTSBET_FIXTURES_URL = (
    "https://raw.githubusercontent.com/georgedouzas/sports-betting/"
    "data/data/soccer/modelling/fixtures.csv"
)

HOME_WIN_COL = "odds__market_maximum__home_win__full_time_goals"
DRAW_COL = "odds__market_maximum__draw__full_time_goals"
AWAY_WIN_COL = "odds__market_maximum__away_win__full_time_goals"
OVER25_COL = "odds__market_maximum__over_2.5__full_time_goals"
UNDER25_COL = "odds__market_maximum__under_2.5__full_time_goals"

TEAM_ALIASES = {
    "arsenal": "arsenal",
    "aston villa": "aston villa",
    "bournemouth": "bournemouth",
    "brentford": "brentford",
    "brighton": "brighton hove albion",
    "brighton hove albion": "brighton hove albion",
    "burnley": "burnley",
    "cardiff": "cardiff city",
    "chelsea": "chelsea",
    "crystal palace": "crystal palace",
    "everton": "everton",
    "fulham": "fulham",
    "huddersfield": "huddersfield town",
    "hull": "hull city",
    "ipswich": "ipswich town",
    "leeds": "leeds united",
    "leicester": "leicester city",
    "liverpool": "liverpool",
    "luton": "luton town",
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "middlesbrough": "middlesbrough",
    "newcastle": "newcastle united",
    "norwich": "norwich city",
    "nott m forest": "nottingham forest",
    "nottm forest": "nottingham forest",
    "nottingham forest": "nottingham forest",
    "sheff utd": "sheffield united",
    "southampton": "southampton",
    "stoke": "stoke city",
    "sunderland": "sunderland",
    "swansea": "swansea city",
    "tottenham": "tottenham hotspur",
    "west brom": "west bromwich albion",
    "west bromwich": "west bromwich albion",
    "west ham": "west ham united",
    "wolves": "wolverhampton wanderers",
}


def normalize_team_name(name: str) -> str:
    """Collapse upstream team-name variants into a canonical comparable form."""

    cleaned = name.lower()
    for token in (" football club", " fc", " afc", " cf"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return TEAM_ALIASES.get(cleaned, cleaned)


class SportsBettingOddsFetcher:
    """Fetch historical and fixture odds from the sports-betting EPL data source."""

    def __init__(self, seasons: list[int] | None = None):
        self.seasons = seasons or []

    @staticmethod
    def _read_csv(url: str) -> pd.DataFrame:
        return pd.read_csv(url)

    def _load_historical(self) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        for year in self.seasons:
            url = SPORTSBET_EPL_URL.format(year=year)
            try:
                frames.append(self._read_csv(url))
            except Exception as exc:  # pragma: no cover - defensive network handling
                logger.warning("Could not load sports-betting EPL file for %s: %s", year, exc)
        return frames

    def _load_fixtures(self) -> pd.DataFrame:
        return self._read_csv(SPORTSBET_FIXTURES_URL)

    def load_epl_rows(self, include_fixtures: bool = True) -> pd.DataFrame:
        """Load raw EPL rows from sports-betting's published modelling dataset."""

        frames = self._load_historical()
        if include_fixtures:
            frames.append(self._load_fixtures())
        if not frames:
            return pd.DataFrame()

        data = pd.concat(frames, ignore_index=True)
        data = data[(data["league"] == "England") & (data["division"] == 1)].copy()
        data["match_date"] = pd.to_datetime(data["date"], format="mixed", utc=True).dt.date
        return data

    @staticmethod
    def build_match_index(matches: list[Match]) -> dict[tuple[str, str, str], Match]:
        """Index DB matches by date + normalized team names."""

        index: dict[tuple[str, str, str], Match] = {}
        for match in matches:
            key = (
                match.utc_date.date().isoformat(),
                normalize_team_name(match.home_team),
                normalize_team_name(match.away_team),
            )
            index[key] = match
        return index

    def build_market_odds_rows(self, matches: list[Match], include_fixtures: bool = True) -> tuple[list[dict], int]:
        """Map sports-betting rows onto local matches and produce DB-ready odds payloads."""

        raw_rows = self.load_epl_rows(include_fixtures=include_fixtures)
        if raw_rows.empty:
            return [], 0

        index = self.build_match_index(matches)
        snapshots: list[dict] = []
        unmatched = 0

        for row in raw_rows.to_dict(orient="records"):
            key = (
                row["match_date"].isoformat(),
                normalize_team_name(row["home_team"]),
                normalize_team_name(row["away_team"]),
            )
            match = index.get(key)
            if not match:
                unmatched += 1
                continue

            snapshots.append(
                {
                    "match_api_id": match.api_id,
                    "source": SPORTSBET_SOURCE,
                    "captured_at": datetime.now(timezone.utc),
                    "home_win_odds": _float_or_none(row.get(HOME_WIN_COL)),
                    "draw_odds": _float_or_none(row.get(DRAW_COL)),
                    "away_win_odds": _float_or_none(row.get(AWAY_WIN_COL)),
                    "over25_odds": _float_or_none(row.get(OVER25_COL)),
                    "under25_odds": _float_or_none(row.get(UNDER25_COL)),
                    "btts_yes_odds": None,
                    "btts_no_odds": None,
                }
            )

        return snapshots, unmatched


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
