"""Select genuine pre-kickoff forecasts before deduplicating history."""

from sqlalchemy import func

from app.models.match import Match
from app.models.prediction import Prediction


def eligible_prediction_ids(db, model_name: str | None = None):
    query = db.query(
        Prediction.id.label("prediction_id"),
        func.row_number().over(
            partition_by=Prediction.match_api_id,
            order_by=(Prediction.created_at.desc(), Prediction.id.desc()),
        ).label("rank"),
    ).join(Match, Match.api_id == Prediction.match_api_id).filter(
        Prediction.created_at < Match.utc_date,
        Prediction.home_team == Match.home_team,
        Prediction.away_team == Match.away_team,
        Match.status == "FINISHED",
        Match.home_goals.isnot(None),
        Match.away_goals.isnot(None),
    )
    if model_name is not None:
        query = query.filter(Prediction.model_name == model_name)
    ranked = query.subquery()
    return db.query(ranked.c.prediction_id.label("latest_id")).filter(ranked.c.rank == 1).subquery()
