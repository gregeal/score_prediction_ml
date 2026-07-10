"""Walk-forward benchmark for prediction model configurations.

Evaluates Dixon-Coles configurations (time-decay half-life, L2 shrinkage,
form weighting) and the challenger model on the most recent matches using a
strict walk-forward protocol: each chunk of test matches is predicted by a
model trained only on matches played before that chunk.

Usage:
    python scripts/benchmark_model.py                 # default config sweep
    python scripts/benchmark_model.py --test-size 380 --challenger
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.base import ensure_database_ready, get_session_local
from app.models.match import Match
from app.ml.challenger_model import ChallengerModel
from app.ml.dixon_coles import DixonColesModel
from app.ml.elo import EloSystem
from app.ml.evaluate import evaluate_predictions, score_prediction
from app.ml.features import matches_to_training_data

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _match_outcome(match: Match) -> str:
    if match.home_goals > match.away_goals:
        return "home"
    if match.home_goals == match.away_goals:
        return "draw"
    return "away"


def _aware(dt):
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def load_finished_matches() -> list[Match]:
    ensure_database_ready()
    db = get_session_local()()
    try:
        return (
            db.query(Match)
            .filter(Match.status == "FINISHED", Match.home_goals.isnot(None))
            .order_by(Match.utc_date)
            .all()
        )
    finally:
        db.close()


def walk_forward_dixon_coles(
    finished: list[Match],
    *,
    test_size: int,
    chunk_size: int,
    half_life: int,
    l2_reg: float,
    form_weighting: bool,
):
    """Walk-forward evaluation of a Dixon-Coles configuration."""
    evaluated = []
    start = max(60, len(finished) - test_size)

    for chunk_start in range(start, len(finished), chunk_size):
        test_chunk = finished[chunk_start : chunk_start + chunk_size]
        if not test_chunk:
            break
        reference_date = _aware(test_chunk[0].utc_date)
        train = finished[:chunk_start]

        data = matches_to_training_data(
            train,
            time_decay_days=half_life,
            reference_date=reference_date,
            use_form_weighting=form_weighting,
        )
        model = DixonColesModel(time_decay_days=half_life, l2_reg=l2_reg)
        model.fit(data)

        for match in test_chunk:
            pred = model.predict_match(match.home_team, match.away_team)
            evaluated.append(
                score_prediction(
                    predicted_probs=(pred.home_win_prob, pred.draw_prob, pred.away_win_prob),
                    actual_outcome=_match_outcome(match),
                    match_date=match.utc_date,
                    match_api_id=match.api_id,
                    predicted_score=pred.outcome_score or pred.most_likely_score,
                    actual_score=f"{match.home_goals}-{match.away_goals}",
                    over25_prob=pred.over25_prob,
                    btts_prob=pred.btts_prob,
                )
            )

    return evaluate_predictions(evaluated)


def walk_forward_naive(finished: list[Match], *, test_size: int):
    """League-average baseline: predicts historical home/draw/away rates."""
    evaluated = []
    start = max(60, len(finished) - test_size)
    for i in range(start, len(finished)):
        train = finished[:i]
        counts = {"home": 1, "draw": 1, "away": 1}
        for m in train:
            counts[_match_outcome(m)] += 1
        total = sum(counts.values())
        probs = (counts["home"] / total, counts["draw"] / total, counts["away"] / total)
        match = finished[i]
        evaluated.append(
            score_prediction(
                predicted_probs=probs,
                actual_outcome=_match_outcome(match),
                match_date=match.utc_date,
            )
        )
    return evaluate_predictions(evaluated)


def walk_forward_challenger(finished: list[Match], *, test_size: int, time_decay_days: int):
    """Walk-forward challenger eval: GBM fit once on pre-test history, Elo rolled forward."""
    split = max(60, len(finished) - test_size)
    train, test = finished[:split], finished[split:]

    rolling_elo = EloSystem.from_matches(train)
    challenger = ChallengerModel(time_decay_days=time_decay_days)
    challenger.fit(train, rolling_elo)

    evaluated = []
    for i, match in enumerate(test):
        context = sorted(finished[: split + i], key=lambda m: m.utc_date, reverse=True)
        try:
            pred = challenger.predict_match(
                match.home_team,
                match.away_team,
                rolling_elo,
                context,
                reference_date=_aware(match.utc_date),
            )
        except (ValueError, KeyError) as exc:
            logger.warning("Challenger skipped %s vs %s: %s", match.home_team, match.away_team, exc)
            continue
        evaluated.append(
            score_prediction(
                predicted_probs=(pred.home_win_prob, pred.draw_prob, pred.away_win_prob),
                actual_outcome=_match_outcome(match),
                match_date=match.utc_date,
                predicted_score=pred.outcome_score or pred.most_likely_score,
                actual_score=f"{match.home_goals}-{match.away_goals}",
                over25_prob=pred.over25_prob,
                btts_prob=pred.btts_prob,
            )
        )
        rolling_elo.update(match.home_team, match.away_team, match.home_goals, match.away_goals)

    return evaluate_predictions(evaluated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=int, default=380, help="Matches in the walk-forward test window")
    parser.add_argument("--chunk-size", type=int, default=20, help="Matches per retrain chunk")
    parser.add_argument("--half-lives", type=int, nargs="*", default=[90, 120, 180, 270, 365, 540])
    parser.add_argument("--l2-regs", type=float, nargs="*", default=[0.0, 1.0, 2.0, 5.0])
    parser.add_argument("--challenger", action="store_true", help="Also evaluate the challenger model")
    args = parser.parse_args()

    finished = load_finished_matches()
    print(f"Finished matches available: {len(finished)}")
    print(f"Walk-forward test window: last {min(args.test_size, len(finished) - 60)} matches, "
          f"retrain every {args.chunk_size}\n")

    header = f"{'config':<45} {'accuracy':>9} {'brier':>8} {'logloss':>8} {'o/u2.5':>7} {'btts':>7}"
    print(header)
    print("-" * len(header))

    def show(name: str, result) -> None:
        print(
            f"{name:<45} {result.outcome_accuracy:>9.4f} {result.brier_score:>8.4f} "
            f"{result.avg_log_loss:>8.4f} {result.over25_accuracy:>7.4f} {result.btts_accuracy:>7.4f}"
        )

    naive = walk_forward_naive(finished, test_size=args.test_size)
    show("naive league-average baseline", naive)

    # Legacy production configuration (pre-optimization defaults)
    legacy = walk_forward_dixon_coles(
        finished,
        test_size=args.test_size,
        chunk_size=args.chunk_size,
        half_life=365,
        l2_reg=0.0,
        form_weighting=True,
    )
    show("LEGACY dc hl=365 l2=0 form=on", legacy)

    results = {}
    for half_life in args.half_lives:
        for l2_reg in args.l2_regs:
            start_time = time.time()
            result = walk_forward_dixon_coles(
                finished,
                test_size=args.test_size,
                chunk_size=args.chunk_size,
                half_life=half_life,
                l2_reg=l2_reg,
                form_weighting=False,
            )
            results[(half_life, l2_reg)] = result
            show(f"dc hl={half_life} l2={l2_reg} form=off ({time.time() - start_time:.0f}s)", result)

    best_key = min(results, key=lambda key: results[key].brier_score)
    best = results[best_key]
    print(f"\nBest Dixon-Coles config by Brier: half_life={best_key[0]}, l2_reg={best_key[1]} "
          f"(brier {best.brier_score:.4f} vs legacy {legacy.brier_score:.4f}, "
          f"naive {naive.brier_score:.4f})")

    if args.challenger:
        print("\nEvaluating challenger model (walk-forward)...")
        challenger_result = walk_forward_challenger(
            finished, test_size=args.test_size, time_decay_days=best_key[0]
        )
        show(f"challenger (dc hl={best_key[0]})", challenger_result)


if __name__ == "__main__":
    main()
