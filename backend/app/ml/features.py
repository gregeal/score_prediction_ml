"""Feature engineering: convert raw match data to model inputs."""

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from app.models.match import Match
from app.ml.dixon_coles import MatchData


@dataclass
class TeamForm:
    """Recent form stats for a team."""
    team: str
    last_n: int  # Number of recent matches considered
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    points: int = 0
    clean_sheets: int = 0
    failed_to_score: int = 0

    @property
    def points_per_game(self) -> float:
        return self.points / self.last_n if self.last_n > 0 else 0.0

    @property
    def goals_scored_per_game(self) -> float:
        return self.goals_scored / self.last_n if self.last_n > 0 else 0.0

    @property
    def goals_conceded_per_game(self) -> float:
        return self.goals_conceded / self.last_n if self.last_n > 0 else 0.0

    @property
    def form_string(self) -> str:
        """Return W/D/L string for recent results (most recent first)."""
        return ""  # Populated externally


def compute_form(matches: list[Match], team: str, last_n: int = 5) -> TeamForm:
    """Compute recent form for a team from finished matches.

    Args:
        matches: List of Match objects sorted by date (most recent first).
        team: Team name to compute form for.
        last_n: Number of recent matches to consider.

    Returns:
        TeamForm with aggregated stats.
    """
    form = TeamForm(team=team, last_n=0)
    results = []

    for match in matches:
        if match.status != "FINISHED" or match.home_goals is None:
            continue
        if match.home_team != team and match.away_team != team:
            continue

        if form.last_n >= last_n:
            break

        form.last_n += 1
        is_home = match.home_team == team
        gf = match.home_goals if is_home else match.away_goals
        ga = match.away_goals if is_home else match.home_goals

        form.goals_scored += gf
        form.goals_conceded += ga
        if ga == 0:
            form.clean_sheets += 1
        if gf == 0:
            form.failed_to_score += 1

        if gf > ga:
            form.wins += 1
            form.points += 3
            results.append("W")
        elif gf == ga:
            form.draws += 1
            form.points += 1
            results.append("D")
        else:
            form.losses += 1
            results.append("L")

    return form


def compute_all_team_forms(
    matches: list[Match], last_n: int = 5
) -> dict[str, TeamForm]:
    """Compute form for all teams.

    Args:
        matches: Matches sorted by date descending (most recent first).
        last_n: Number of recent matches per team.

    Returns:
        Dict mapping team name to TeamForm.
    """
    teams = set()
    for m in matches:
        if m.status == "FINISHED":
            teams.add(m.home_team)
            teams.add(m.away_team)

    return {team: compute_form(matches, team, last_n) for team in teams}


def apply_form_weight(
    training_data: list[MatchData],
    matches: list[Match],
    form_boost: float = 0.15,
    last_n: int = 5,
) -> list[MatchData]:
    """Boost weights of training matches based on current team form.

    Teams on a strong run get slightly boosted attack weights,
    teams on a poor run get slightly reduced weights. This helps
    the model capture current momentum beyond just time decay.

    Args:
        training_data: Existing MatchData with time decay weights.
        matches: All matches sorted by date descending.
        form_boost: Maximum weight adjustment factor (0.15 = +/-15%).
        last_n: Number of recent matches to compute form from.

    Returns:
        Modified training data with form-adjusted weights.
    """
    forms = compute_all_team_forms(matches, last_n)

    # Compute average PPG across all teams for normalization
    all_ppg = [f.points_per_game for f in forms.values() if f.last_n > 0]
    if not all_ppg:
        return training_data

    avg_ppg = sum(all_ppg) / len(all_ppg)
    max_deviation = max(abs(ppg - avg_ppg) for ppg in all_ppg) or 1.0

    for match_data in training_data:
        # Boost recent matches involving in-form teams
        home_form = forms.get(match_data.home_team)
        away_form = forms.get(match_data.away_team)

        if home_form and home_form.last_n > 0:
            home_deviation = (home_form.points_per_game - avg_ppg) / max_deviation
            match_data.weight *= (1.0 + form_boost * home_deviation)

        if away_form and away_form.last_n > 0:
            away_deviation = (away_form.points_per_game - avg_ppg) / max_deviation
            match_data.weight *= (1.0 + form_boost * away_deviation)

    return training_data


def matches_to_training_data(
    matches: list[Match],
    time_decay_days: int = 365,
    reference_date: datetime | None = None,
    use_form_weighting: bool = True,
) -> list[MatchData]:
    """Convert database Match objects to MatchData for model training.

    Only includes finished matches with valid scores. Applies exponential
    time decay weighting so recent matches have more influence. Optionally
    applies form-based weight adjustments.

    Args:
        matches: List of Match ORM objects.
        time_decay_days: Half-life for time weighting in days.
        reference_date: Date to calculate weights from (defaults to now).
        use_form_weighting: Whether to apply form-based weight adjustments.

    Returns:
        List of MatchData objects ready for model.fit().
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    training_data = []
    for match in matches:
        # Only use finished matches with scores
        if match.status != "FINISHED" or match.home_goals is None or match.away_goals is None:
            continue

        # Calculate time decay weight
        match_date = match.utc_date
        if match_date.tzinfo is None:
            match_date = match_date.replace(tzinfo=timezone.utc)

        days_ago = (reference_date - match_date).days
        weight = math.exp(-math.log(2) * days_ago / time_decay_days)

        training_data.append(
            MatchData(
                home_team=match.home_team,
                away_team=match.away_team,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
                weight=weight,
            )
        )

    # Apply form-based weight adjustments
    if use_form_weighting and training_data:
        sorted_matches = sorted(
            [m for m in matches if m.status == "FINISHED"],
            key=lambda m: m.utc_date,
            reverse=True,
        )
        training_data = apply_form_weight(training_data, sorted_matches)

    return training_data
