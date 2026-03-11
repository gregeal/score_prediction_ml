"""Prediction API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.base import get_db
from app.models.match import Match
from app.models.prediction import Prediction

router = APIRouter(tags=["predictions"])


@router.get("/predictions/{match_api_id}")
def get_prediction(match_api_id: int, db: Session = Depends(get_db)):
    """Get the prediction for a specific match by its API ID."""
    prediction = (
        db.query(Prediction)
        .filter(Prediction.match_api_id == match_api_id)
        .order_by(Prediction.created_at.desc())
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found for this match")

    match = db.query(Match).filter(Match.api_id == match_api_id).first()

    return {
        "match": {
            "home": prediction.home_team,
            "away": prediction.away_team,
            "date": match.utc_date.isoformat() if match else None,
            "matchday": match.matchday if match else None,
        },
        "predictions": {
            "outcome": {
                "home_win": prediction.home_win_prob,
                "draw": prediction.draw_prob,
                "away_win": prediction.away_win_prob,
            },
            "most_likely_score": prediction.most_likely_score,
            "over_under_25": {"over": prediction.over25_prob, "under": round(1 - prediction.over25_prob, 4)},
            "btts": {"yes": prediction.btts_prob, "no": round(1 - prediction.btts_prob, 4)},
        },
        "expected_goals": {
            "home": prediction.predicted_home_goals,
            "away": prediction.predicted_away_goals,
        },
        "confidence": prediction.confidence,
    }


@router.get("/accuracy")
def get_accuracy(db: Session = Depends(get_db)):
    """Get model accuracy stats by comparing predictions to actual results.

    Uses only the latest prediction per match to avoid counting duplicates.
    """
    # Get finished matches that have predictions
    from sqlalchemy import func as sa_func

    # Subquery: latest prediction id per match
    latest_pred = (
        db.query(
            Prediction.match_api_id,
            sa_func.max(Prediction.id).label("latest_id"),
        )
        .group_by(Prediction.match_api_id)
        .subquery()
    )

    predictions = (
        db.query(Prediction)
        .join(latest_pred, Prediction.id == latest_pred.c.latest_id)
        .all()
    )

    if not predictions:
        return {"total_evaluated": 0, "message": "No predictions to evaluate yet"}

    correct_outcome = 0
    correct_exact = 0
    correct_over25 = 0
    correct_btts = 0
    total = 0

    for pred in predictions:
        match = db.query(Match).filter(
            Match.api_id == pred.match_api_id,
            Match.status == "FINISHED",
        ).first()
        if not match or match.home_goals is None:
            continue

        total += 1

        # Check outcome
        probs = [pred.home_win_prob, pred.draw_prob, pred.away_win_prob]
        predicted = ["home", "draw", "away"][probs.index(max(probs))]
        if match.home_goals > match.away_goals:
            actual = "home"
        elif match.home_goals == match.away_goals:
            actual = "draw"
        else:
            actual = "away"
        if predicted == actual:
            correct_outcome += 1

        # Check exact score
        actual_score = f"{match.home_goals}-{match.away_goals}"
        if pred.most_likely_score == actual_score:
            correct_exact += 1

        # Check O/U 2.5
        if (pred.over25_prob > 0.5) == (match.home_goals + match.away_goals > 2):
            correct_over25 += 1

        # Check BTTS
        if (pred.btts_prob > 0.5) == (match.home_goals > 0 and match.away_goals > 0):
            correct_btts += 1

    if total == 0:
        return {"total_evaluated": 0, "message": "No finished matches with predictions yet"}

    return {
        "total_evaluated": total,
        "outcome_accuracy": round(correct_outcome / total, 4),
        "exact_score_accuracy": round(correct_exact / total, 4),
        "over_under_accuracy": round(correct_over25 / total, 4),
        "btts_accuracy": round(correct_btts / total, 4),
    }
