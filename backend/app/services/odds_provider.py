"""Bookmaker odds ingestion from football-data.co.uk.

football-data.co.uk publishes one CSV per league season (E0 = Premier League)
with closing odds for finished matches, plus a rolling fixtures.csv with
current odds for upcoming matches. The URLs and column layout have been
stable for two decades, unlike the previously used sports-betting GitHub
data branch, which restructured and broke ingestion.
"""

from __future__ import annotations

import io
import logging
import re
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from app.models.match import Match

logger = logging.getLogger(__name__)

ODDS_SOURCE = "football-data.co.uk"
EPL_DIVISION = "E0"
SEASON_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# (home, draw, away) column preference: market maximum first, Bet365 fallback
MATCH_ODDS_COLUMNS = (
    ("MaxH", "MaxD", "MaxA"),
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
)
OVER_UNDER_COLUMNS = (
    ("Max>2.5", "Max<2.5"),
    ("B365>2.5", "B365<2.5"),
    ("Avg>2.5", "Avg<2.5"),
)

TEAM_ALIASES = {
    "afc bournemouth": "bournemouth",
    "arsenal": "arsenal",
    "aston villa": "aston villa",
    "birmingham": "birmingham city",
    "blackburn": "blackburn rovers",
    "bournemouth": "bournemouth",
    "brentford": "brentford",
    "brighton": "brighton hove albion",
    "brighton and hove albion": "brighton hove albion",
    "brighton hove albion": "brighton hove albion",
    "burnley": "burnley",
    "cardiff": "cardiff city",
    "chelsea": "chelsea",
    "coventry": "coventry city",
    "crystal palace": "crystal palace",
    "derby": "derby county",
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
    "preston": "preston north end",
    "qpr": "queens park rangers",
    "sheff utd": "sheffield united",
    "sheff wed": "sheffield wednesday",
    "sheffield united": "sheffield united",
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
    """Collapse team-name variants into a canonical comparable form."""

    cleaned = name.lower()
    for token in (" football club", " fc", " afc", " cf"):
        cleaned = cleaned.replace(token, "")
    # Punctuation (including '&') collapses to whitespace, so
    # "Brighton & Hove Albion FC" and "Brighton" both normalize to the
    # same canonical form via the alias table.
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return TEAM_ALIASES.get(cleaned, cleaned)


def season_code(start_year: int) -> str:
    """football-data.co.uk season code: 2025 (2025-26 season) -> '2526'."""

    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _first_available(row: dict, column_groups) -> list[float | None]:
    """Pick the first odds column group with usable values in this row."""

    for group in column_groups:
        values = [_float_or_none(row.get(column)) for column in group]
        if all(value is not None and value > 0 for value in values):
            return values
    return [None] * len(column_groups[0])


class FootballDataOddsFetcher:
    """Fetch historical and fixture odds from football-data.co.uk."""

    def __init__(self, seasons: list[int] | None = None):
        self.seasons = seasons or []

    @staticmethod
    def _read_csv(url: str) -> pd.DataFrame:
        request = urllib.request.Request(url, headers={"User-Agent": "predictepl/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        try:
            return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", on_bad_lines="skip")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")

    def _load_historical(self) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        for year in self.seasons:
            url = SEASON_URL.format(code=season_code(year))
            try:
                frames.append(self._read_csv(url))
            except Exception as exc:  # pragma: no cover - defensive network handling
                logger.warning("Could not load football-data.co.uk file for %s: %s", year, exc)
        return frames

    def _load_fixtures(self) -> pd.DataFrame | None:
        try:
            fixtures = self._read_csv(FIXTURES_URL)
        except Exception as exc:  # pragma: no cover - defensive network handling
            logger.warning("Could not load football-data.co.uk fixtures: %s", exc)
            return None
        if "Div" not in fixtures.columns:
            return None
        return fixtures[fixtures["Div"] == EPL_DIVISION]

    def load_epl_rows(self, include_fixtures: bool = True) -> pd.DataFrame:
        """Load raw EPL odds rows (historical seasons + optional fixtures)."""

        frames = self._load_historical()
        if include_fixtures:
            fixtures = self._load_fixtures()
            if fixtures is not None and not fixtures.empty:
                frames.append(fixtures)
        if not frames:
            return pd.DataFrame()

        data = pd.concat(frames, ignore_index=True)
        data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
        data["match_date"] = pd.to_datetime(
            data["Date"], dayfirst=True, format="mixed", errors="coerce"
        ).dt.date
        data = data.dropna(subset=["match_date"])
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
        """Map football-data.co.uk rows onto local matches and produce DB-ready payloads."""

        raw_rows = self.load_epl_rows(include_fixtures=include_fixtures)
        if raw_rows.empty:
            return [], 0

        index = self.build_match_index(matches)
        snapshots: list[dict] = []
        unmatched = 0

        for row in raw_rows.to_dict(orient="records"):
            key = (
                row["match_date"].isoformat(),
                normalize_team_name(str(row["HomeTeam"])),
                normalize_team_name(str(row["AwayTeam"])),
            )
            match = index.get(key)
            if not match:
                unmatched += 1
                continue

            home_odds, draw_odds, away_odds = _first_available(row, MATCH_ODDS_COLUMNS)
            over25_odds, under25_odds = _first_available(row, OVER_UNDER_COLUMNS)
            if home_odds is None and over25_odds is None:
                continue  # No usable odds in this row

            snapshots.append(
                {
                    "match_api_id": match.api_id,
                    "source": ODDS_SOURCE,
                    "captured_at": datetime.now(timezone.utc),
                    "home_win_odds": home_odds,
                    "draw_odds": draw_odds,
                    "away_win_odds": away_odds,
                    "over25_odds": over25_odds,
                    "under25_odds": under25_odds,
                    "btts_yes_odds": None,
                    "btts_no_odds": None,
                }
            )

        return snapshots, unmatched


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Backward-compatible alias for the previous provider name.
SportsBettingOddsFetcher = FootballDataOddsFetcher
SPORTSBET_SOURCE = ODDS_SOURCE
