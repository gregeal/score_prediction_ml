"""Tests for football-data.co.uk odds ingestion helpers."""

from datetime import datetime, timezone

import pandas as pd

from app.services.odds_provider import (
    FootballDataOddsFetcher,
    normalize_team_name,
    season_code,
)


def _match(api_id: int, home: str, away: str, date_str: str):
    class MatchStub:
        pass

    match = MatchStub()
    match.api_id = api_id
    match.home_team = home
    match.away_team = away
    match.utc_date = datetime.fromisoformat(f"{date_str}T15:00:00+00:00").astimezone(timezone.utc)
    return match


class FakeOddsFetcher(FootballDataOddsFetcher):
    def __init__(self, data: pd.DataFrame):
        super().__init__(seasons=[2024])
        self._data = data

    def load_epl_rows(self, include_fixtures: bool = True) -> pd.DataFrame:
        return self._data.copy()


class TestOddsProvider:
    def test_normalize_team_name_handles_common_aliases(self):
        assert normalize_team_name("Manchester City FC") == "manchester city"
        assert normalize_team_name("Man City") == "manchester city"
        assert normalize_team_name("Nott'm Forest") == "nottingham forest"
        assert normalize_team_name("Brighton") == "brighton hove albion"
        assert normalize_team_name("Wolves") == "wolverhampton wanderers"

    def test_normalize_handles_new_season_promoted_clubs(self):
        assert normalize_team_name("Coventry") == "coventry city"
        assert normalize_team_name("Coventry City FC") == "coventry city"
        assert normalize_team_name("Hull") == "hull city"
        assert normalize_team_name("Hull City AFC") == "hull city"

    def test_bournemouth_db_and_upstream_names_agree(self):
        # DB stores "AFC Bournemouth"; football-data.co.uk uses "Bournemouth".
        assert normalize_team_name("AFC Bournemouth") == normalize_team_name("Bournemouth")

    def test_brighton_ampersand_db_name_matches_upstream(self):
        # DB stores "Brighton & Hove Albion FC"; the upstream CSV says "Brighton".
        # The old '&' -> ' and ' rewrite silently unmatched every Brighton game.
        assert normalize_team_name("Brighton & Hove Albion FC") == normalize_team_name("Brighton")

    def test_season_code(self):
        assert season_code(2025) == "2526"
        assert season_code(2026) == "2627"
        assert season_code(1999) == "9900"

    def test_build_market_odds_rows_matches_rows_to_local_fixtures(self):
        data = pd.DataFrame(
            [
                {
                    "match_date": pd.Timestamp("2024-08-18").date(),
                    "HomeTeam": "Man City",
                    "AwayTeam": "Arsenal",
                    "MaxH": 1.9,
                    "MaxD": 3.7,
                    "MaxA": 4.1,
                    "Max>2.5": 1.85,
                    "Max<2.5": 2.0,
                },
                {
                    "match_date": pd.Timestamp("2024-08-19").date(),
                    "HomeTeam": "Leeds",
                    "AwayTeam": "Leicester",
                    "MaxH": 2.2,
                    "MaxD": 3.1,
                    "MaxA": 3.5,
                    "Max>2.5": 1.95,
                    "Max<2.5": 1.95,
                },
            ]
        )
        fetcher = FakeOddsFetcher(data)
        matches = [
            _match(1001, "Manchester City FC", "Arsenal FC", "2024-08-18"),
        ]

        odds_rows, unmatched = fetcher.build_market_odds_rows(matches)

        assert unmatched == 1
        assert len(odds_rows) == 1
        row = odds_rows[0]
        assert row["match_api_id"] == 1001
        assert row["home_win_odds"] == 1.9
        assert row["draw_odds"] == 3.7
        assert row["away_win_odds"] == 4.1
        assert row["over25_odds"] == 1.85
        assert row["under25_odds"] == 2.0

    def test_falls_back_to_b365_columns_when_max_missing(self):
        data = pd.DataFrame(
            [
                {
                    "match_date": pd.Timestamp("2024-08-18").date(),
                    "HomeTeam": "Man City",
                    "AwayTeam": "Arsenal",
                    "B365H": 2.0,
                    "B365D": 3.4,
                    "B365A": 3.9,
                    "B365>2.5": 1.8,
                    "B365<2.5": 2.05,
                },
            ]
        )
        fetcher = FakeOddsFetcher(data)
        matches = [_match(1001, "Manchester City FC", "Arsenal FC", "2024-08-18")]

        odds_rows, _ = fetcher.build_market_odds_rows(matches)

        assert len(odds_rows) == 1
        assert odds_rows[0]["home_win_odds"] == 2.0
        assert odds_rows[0]["over25_odds"] == 1.8

    def test_rows_without_usable_odds_are_skipped(self):
        data = pd.DataFrame(
            [
                {
                    "match_date": pd.Timestamp("2024-08-18").date(),
                    "HomeTeam": "Man City",
                    "AwayTeam": "Arsenal",
                },
            ]
        )
        fetcher = FakeOddsFetcher(data)
        matches = [_match(1001, "Manchester City FC", "Arsenal FC", "2024-08-18")]

        odds_rows, unmatched = fetcher.build_market_odds_rows(matches)

        assert odds_rows == []
        assert unmatched == 0

    def test_empty_load_returns_no_rows(self):
        fetcher = FakeOddsFetcher(pd.DataFrame())
        odds_rows, unmatched = fetcher.build_market_odds_rows([])
        assert odds_rows == []
        assert unmatched == 0
