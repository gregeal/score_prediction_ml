"""Fixtures API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.base import get_db
from app.models.match import Match
from app.models.prediction import Prediction

router = APIRouter(tags=["fixtures"])


@router.get("/fixtures/upcoming")
def get_upcoming_fixtures(db: Session = Depends(get_db)):
    """Get upcoming fixtures with their predictions."""
    upcoming = (
        db.query(Match)
        .filter(Match.status.in_(["SCHEDULED", "TIMED"]))
        .order_by(Match.utc_date)
        .limit(20)
        .all()
    )

    results = []
    for match in upcoming:
        pred = (
            db.query(Prediction)
            .filter(Prediction.match_api_id == match.api_id)
            .order_by(Prediction.created_at.desc())
            .first()
        )

        fixture = {
            "match_id": match.api_id,
            "home": match.home_team,
            "away": match.away_team,
            "date": match.utc_date.isoformat(),
            "matchday": match.matchday,
        }

        if pred:
            fixture["prediction"] = {
                "outcome": {
                    "home_win": pred.home_win_prob,
                    "draw": pred.draw_prob,
                    "away_win": pred.away_win_prob,
                },
                "most_likely_score": pred.most_likely_score,
                "over_under_25": pred.over25_prob,
                "btts": pred.btts_prob,
                "confidence": pred.confidence,
            }
        else:
            fixture["prediction"] = None

        results.append(fixture)

    return {"fixtures": results, "count": len(results)}


@router.get("/standings")
def get_standings(db: Session = Depends(get_db)):
    """Get current EPL standings derived from match results."""
    finished = (
        db.query(Match)
        .filter(Match.status == "FINISHED")
        .order_by(Match.utc_date.desc())
        .all()
    )

    if not finished:
        return {"standings": [], "season": None}

    # Get current season
    current_season = finished[0].season

    # Filter to current season only
    season_matches = [m for m in finished if m.season == current_season]

    # Build standings table
    table: dict[str, dict] = {}
    for match in season_matches:
        for team in [match.home_team, match.away_team]:
            if team not in table:
                table[team] = {
                    "team": team, "played": 0, "won": 0, "drawn": 0,
                    "lost": 0, "goals_for": 0, "goals_against": 0,
                    "goal_difference": 0, "points": 0,
                }

        if match.home_goals is None:
            continue

        ht, at = match.home_team, match.away_team
        hg, ag = match.home_goals, match.away_goals

        table[ht]["played"] += 1
        table[at]["played"] += 1
        table[ht]["goals_for"] += hg
        table[ht]["goals_against"] += ag
        table[at]["goals_for"] += ag
        table[at]["goals_against"] += hg

        if hg > ag:
            table[ht]["won"] += 1
            table[ht]["points"] += 3
            table[at]["lost"] += 1
        elif hg == ag:
            table[ht]["drawn"] += 1
            table[ht]["points"] += 1
            table[at]["drawn"] += 1
            table[at]["points"] += 1
        else:
            table[at]["won"] += 1
            table[at]["points"] += 3
            table[ht]["lost"] += 1

    for entry in table.values():
        entry["goal_difference"] = entry["goals_for"] - entry["goals_against"]

    standings = sorted(
        table.values(),
        key=lambda x: (x["points"], x["goal_difference"], x["goals_for"]),
        reverse=True,
    )

    for i, entry in enumerate(standings, 1):
        entry["position"] = i

    return {"standings": standings, "season": current_season}
