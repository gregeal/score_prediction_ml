"""Feature engineering: convert raw match data to model inputs."""

import math
from datetime import datetime, timezone

from app.models.match import Match
from app.ml.dixon_coles import MatchData


def matches_to_training_data(
    matches: list[Match],
    time_decay_days: int = 365,
    reference_date: datetime | None = None,
) -> list[MatchData]:
    """Convert database Match objects to MatchData for model training.

    Only includes finished matches with valid scores. Applies exponential
    time decay weighting so recent matches have more influence.

    Args:
        matches: List of Match ORM objects.
        time_decay_days: Half-life for time weighting in days.
        reference_date: Date to calculate weights from (defaults to now).

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

    return training_data
