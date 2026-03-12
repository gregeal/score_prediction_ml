"""Prediction API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func as sa_func
from sqlalchemy.orm import Session

from app.ml.evaluate import build_dashboard_result, build_recent_snapshot_predictions, score_prediction
from app.models.base import get_db
from app.models.market_odds import MarketOdds
from app.models.match import Match
from app.models.prediction import Prediction
from app.services.predictor import PredictionService

router = APIRouter(tags=["predictions"])


def _match_outcome(match: Match) -> str:
    if match.home_goals > match.away_goals:
        return "home"
    if match.home_goals == match.away_goals:
        return "draw"
    return "away"


def _latest_market_odds_map(db: Session) -> dict[int, MarketOdds]:
    latest_odds = (
        db.query(
            MarketOdds.match_api_id,
            sa_func.max(MarketOdds.id).label("latest_id"),
        )
        .group_by(MarketOdds.match_api_id)
        .subquery()
    )

    rows = (
        db.query(MarketOdds)
        .join(latest_odds, MarketOdds.id == latest_odds.c.latest_id)
        .all()
    )
    return {row.match_api_id: row for row in rows}


def _implied_probs(odds: MarketOdds | None) -> tuple[float, float, float] | None:
    if odds is None:
        return None
    values = [odds.home_win_odds, odds.draw_odds, odds.away_win_odds]
    if any(value is None or value <= 0 for value in values):
        return None

    inverted = [1.0 / float(value) for value in values]
    total = sum(inverted)
    if total <= 0:
        return None
    return tuple(round(value / total, 6) for value in inverted)


def _league_priors(matches: list[Match]) -> dict[int, tuple[float, float, float]]:
    counts = {"home": 1, "draw": 1, "away": 1}
    priors: dict[int, tuple[float, float, float]] = {}

    for match in matches:
        total = counts["home"] + counts["draw"] + counts["away"]
        priors[match.api_id] = (
            counts["home"] / total,
            counts["draw"] / total,
            counts["away"] / total,
        )
        counts[_match_outcome(match)] += 1

    return priors


@router.get("/predictions/{match_api_id}")
def get_prediction(match_api_id: int, db: Session = Depends(get_db)):
    """Get the prediction for a specific match by its API ID."""

    prediction = (
        db.query(Prediction)
        .filter(Prediction.match_api_id == match_api_id)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found for this match")

    match = db.query(Match).filter(Match.api_id == match_api_id).first()

    response = {
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
            "over_under_25": {
                "over": prediction.over25_prob,
                "under": round(1 - prediction.over25_prob, 4),
            },
            "btts": {
                "yes": prediction.btts_prob,
                "no": round(1 - prediction.btts_prob, 4),
            },
        },
        "expected_goals": {
            "home": prediction.predicted_home_goals,
            "away": prediction.predicted_away_goals,
        },
        "confidence": prediction.confidence,
        "model": {
            "name": prediction.model_name,
            "version": prediction.model_version,
            "calibration_version": prediction.calibration_version,
        },
    }

    if prediction.raw_home_win_prob is not None:
        response["raw_predictions"] = {
            "outcome": {
                "home_win": prediction.raw_home_win_prob,
                "draw": prediction.raw_draw_prob,
                "away_win": prediction.raw_away_win_prob,
            }
        }

    return response


@router.get("/accuracy")
def get_accuracy(db: Session = Depends(get_db)):
    """Get model accuracy stats plus calibration, trend, and benchmark views."""

    latest_pred = (
        db.query(
            Prediction.match_api_id,
            sa_func.max(Prediction.id).label("latest_id"),
        )
        .group_by(Prediction.match_api_id)
        .subquery()
    )

    finished_matches = (
        db.query(Match)
        .filter(Match.status == "FINISHED", Match.home_goals.isnot(None))
        .order_by(Match.utc_date)
        .all()
    )
    priors = _league_priors(finished_matches)
    odds_map = _latest_market_odds_map(db)
    bookmaker_probs_by_match = {
        match_api_id: _implied_probs(odds)
        for match_api_id, odds in odds_map.items()
    }

    prediction_rows = (
        db.query(Prediction, Match)
        .join(latest_pred, Prediction.id == latest_pred.c.latest_id)
        .join(Match, Match.api_id == Prediction.match_api_id)
        .filter(Match.status == "FINISHED", Match.home_goals.isnot(None))
        .order_by(Match.utc_date)
        .all()
    )

    evaluated = []
    message = None
    evaluation_source = "stored_predictions"

    if prediction_rows:
        for prediction, match in prediction_rows:
            evaluated.append(
                score_prediction(
                    predicted_probs=(
                        float(prediction.home_win_prob),
                        float(prediction.draw_prob),
                        float(prediction.away_win_prob),
                    ),
                    actual_outcome=_match_outcome(match),
                    match_date=match.utc_date,
                    match_api_id=match.api_id,
                    predicted_score=prediction.outcome_score or prediction.most_likely_score,
                    actual_score=f"{match.home_goals}-{match.away_goals}",
                    over25_prob=float(prediction.over25_prob),
                    btts_prob=float(prediction.btts_prob),
                    baseline_probs=priors.get(match.api_id),
                    bookmaker_probs=bookmaker_probs_by_match.get(match.api_id),
                )
            )
    else:
        try:
            service = PredictionService(db)
            service.load_model()
            finished_desc = sorted(finished_matches, key=lambda match: match.utc_date, reverse=True)

            def predict_finished_match(match: Match):
                try:
                    if (
                        service.active_model == "challenger"
                        and service.challenger.is_fitted
                        and service.elo_system is not None
                    ):
                        pred = service.challenger.predict_match(
                            match.home_team,
                            match.away_team,
                            service.elo_system,
                            finished_desc,
                            reference_date=match.utc_date,
                        )
                    else:
                        pred = service.dc_model.predict_match(match.home_team, match.away_team)
                    service._apply_outcome_calibration(pred)
                    return pred
                except (ValueError, KeyError):
                    return None

            evaluated = build_recent_snapshot_predictions(
                finished_matches,
                predict_match=predict_finished_match,
                baseline_probs_by_match=priors,
                bookmaker_probs_by_match=bookmaker_probs_by_match,
            )
            evaluation_source = "model_snapshot"
            if evaluated:
                message = (
                    "Showing a recent snapshot benchmark from the saved model until enough predicted "
                    "fixtures have finished for live evaluation."
                )
            else:
                return {"total_evaluated": 0, "message": "No finished matches with predictions or snapshot data yet"}
        except FileNotFoundError:
            return {"total_evaluated": 0, "message": "No finished matches with predictions or saved model artifacts yet"}

    dashboard = build_dashboard_result(evaluated)
    latest_prediction = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .first()
    )

    benchmark_delta = None
    naive_benchmark = dashboard.benchmarks.get("naive")
    model_benchmark = dashboard.benchmarks.get("model")
    if naive_benchmark and naive_benchmark.available and model_benchmark and model_benchmark.available:
        benchmark_delta = round(naive_benchmark.brier_score - model_benchmark.brier_score, 4)

    return {
        "total_evaluated": dashboard.total_evaluated,
        "outcome_accuracy": dashboard.outcome_accuracy,
        "exact_score_accuracy": dashboard.exact_score_accuracy,
        "over_under_accuracy": dashboard.over25_accuracy,
        "btts_accuracy": dashboard.btts_accuracy,
        "summary": {
            "active_model": latest_prediction.model_name if latest_prediction else None,
            "calibrated": bool(latest_prediction and latest_prediction.calibration_version),
            "brier_score": dashboard.brier_score,
            "avg_log_loss": dashboard.avg_log_loss,
            "model_version": latest_prediction.model_version if latest_prediction else None,
            "calibration_version": latest_prediction.calibration_version if latest_prediction else None,
            "benchmark_delta_vs_naive": benchmark_delta,
            "evaluation_source": evaluation_source,
        },
        "message": message,
        "calibration": {
            "target": "predicted_outcome",
            "buckets": [bucket.__dict__ for bucket in dashboard.calibration_buckets],
        },
        "segments": [segment.__dict__ for segment in dashboard.segments],
        "rolling_backtest": {
            "window_size": 20,
            "step_size": 20,
            "windows": [window.__dict__ for window in dashboard.rolling_windows],
        },
        "benchmarks": {
            name: metrics.__dict__ for name, metrics in dashboard.benchmarks.items()
        },
    }
