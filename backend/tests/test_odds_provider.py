"""Tests for sports-betting odds ingestion helpers."""

from datetime import datetime, timezone

import pandas as pd

from app.services.odds_provider import SportsBettingOddsFetcher, normalize_team_name


def _match(api_id: int, home: str, away: str, date_str: str):
    class MatchStub:
        pass

    match = MatchStub()
    match.api_id = api_id
    match.home_team = home
    match.away_team = away
    match.utc_date = datetime.fromisoformat(f"{date_str}T15:00:00+00:00").astimezone(timezone.utc)
    return match


class FakeOddsFetcher(SportsBettingOddsFetcher):
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

    def test_build_market_odds_rows_matches_rows_to_local_fixtures(self):
        data = pd.DataFrame(
            [
                {
                    "match_date": pd.Timestamp("2024-08-18").date(),
                    "home_team": "Man City",
                    "away_team": "Arsenal",
                    "odds__market_maximum__home_win__full_time_goals": 1.9,
                    "odds__market_maximum__draw__full_time_goals": 3.7,
                    "odds__market_maximum__away_win__full_time_goals": 4.1,
                    "odds__market_maximum__over_2.5__full_time_goals": 1.85,
                    "odds__market_maximum__under_2.5__full_time_goals": 2.0,
                },
                {
                    "match_date": pd.Timestamp("2024-08-19").date(),
                    "home_team": "Leeds",
                    "away_team": "Leicester",
                    "odds__market_maximum__home_win__full_time_goals": 2.2,
                    "odds__market_maximum__draw__full_time_goals": 3.1,
                    "odds__market_maximum__away_win__full_time_goals": 3.5,
                    "odds__market_maximum__over_2.5__full_time_goals": 1.95,
                    "odds__market_maximum__under_2.5__full_time_goals": 1.95,
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
