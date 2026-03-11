"""Elo rating system for football teams."""

from __future__ import annotations


class EloSystem:
    """Standard Elo rating system with home advantage."""

    def __init__(
        self,
        k_factor: float = 20,
        home_advantage: float = 100,
        default_rating: float = 1500,
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.default_rating = default_rating
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.default_rating)

    def expected_score(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected score (0-1) for each team, accounting for home advantage."""
        home_r = self.get_rating(home_team) + self.home_advantage
        away_r = self.get_rating(away_team)
        exp_home = 1.0 / (1.0 + 10.0 ** ((away_r - home_r) / 400.0))
        return exp_home, 1.0 - exp_home

    def update(
        self, home_team: str, away_team: str, home_goals: int, away_goals: int
    ) -> None:
        """Update ratings after a match result."""
        if home_team not in self.ratings:
            self.ratings[home_team] = self.default_rating
        if away_team not in self.ratings:
            self.ratings[away_team] = self.default_rating

        exp_home, exp_away = self.expected_score(home_team, away_team)

        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals == away_goals:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        self.ratings[home_team] += self.k_factor * (actual_home - exp_home)
        self.ratings[away_team] += self.k_factor * (actual_away - exp_away)

    @classmethod
    def from_matches(cls, matches: list, **kwargs) -> "EloSystem":
        """Build Elo ratings from chronological match history."""
        elo = cls(**kwargs)
        sorted_matches = sorted(
            [m for m in matches if m.status == "FINISHED" and m.home_goals is not None],
            key=lambda m: m.utc_date,
        )
        for match in sorted_matches:
            elo.update(match.home_team, match.away_team, match.home_goals, match.away_goals)
        return elo
