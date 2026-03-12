"""Tests for PredictionService persistence and model selection."""

from datetime import datetime, timedelta, timezone
import pickle
from unittest.mock import MagicMock, patch

import pytest

from app.ml.calibration import OutcomeCalibrator
from app.ml.challenger_model import ChallengerModel
from app.ml.dixon_coles import DixonColesModel, MatchData
from app.ml.elo import EloSystem
from app.services import predictor as predictor_module
from app.services.predictor import PredictionService


def _make_match(home, away, hg, ag, day, status="FINISHED"):
    match = MagicMock()
    match.home_team = home
    match.away_team = away
    match.home_goals = hg
    match.away_goals = ag
    match.status = status
    match.utc_date = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1)
    match.api_id = day
    return match


def _fit_dummy_dc_model() -> DixonColesModel:
    model = DixonColesModel()
    model.fit(
        [
            MatchData("A", "B", 2, 0),
            MatchData("B", "A", 1, 1),
            MatchData("A", "B", 1, 0),
            MatchData("B", "A", 0, 2),
        ]
    )
    return model


def _make_home_win_matches(rounds: int = 50) -> list:
    fixtures = [
        ("TeamA", "TeamB"),
        ("TeamC", "TeamD"),
        ("TeamA", "TeamC"),
        ("TeamB", "TeamD"),
    ]
    matches = []
    day = 1
    for _ in range(rounds):
        for home, away in fixtures:
            matches.append(_make_match(home, away, 2, 0, day))
            day += 1
    return matches


def _fake_dc_result(brier_score: float) -> MagicMock:
    result = MagicMock()
    result.outcome_accuracy = 0.5
    result.brier_score = brier_score
    result.avg_log_loss = 1.0
    result.over25_accuracy = 0.5
    result.btts_accuracy = 0.5
    result.total_matches = 40
    return result


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Redirect all predictor artifacts to a temp directory."""
    with (
        patch.object(predictor_module, "DC_MODEL_PATH", tmp_path / "dc.pkl"),
        patch.object(predictor_module, "CHALLENGER_MODEL_PATH", tmp_path / "challenger.pkl"),
        patch.object(predictor_module, "ELO_PATH", tmp_path / "elo.pkl"),
        patch.object(predictor_module, "ACTIVE_MODEL_PATH", tmp_path / "active_model.txt"),
        patch.object(predictor_module, "CALIBRATOR_PATH", tmp_path / "calibrator.pkl"),
    ):
        yield tmp_path


class TestLoadModelPersistence:
    def test_active_model_persisted_and_restored(self, tmp_model_dir):
        dc_path = tmp_model_dir / "dc.pkl"
        challenger_path = tmp_model_dir / "challenger.pkl"
        elo_path = tmp_model_dir / "elo.pkl"
        active_path = tmp_model_dir / "active_model.txt"
        calibrator_path = tmp_model_dir / "calibrator.pkl"

        with open(dc_path, "wb") as fh:
            pickle.dump(_fit_dummy_dc_model(), fh)

        challenger = ChallengerModel()
        challenger.is_fitted = True
        with open(challenger_path, "wb") as fh:
            pickle.dump(challenger, fh)

        elo = EloSystem()
        elo.update("A", "B", 3, 0)
        with open(elo_path, "wb") as fh:
            pickle.dump(elo, fh)

        calibrator = OutcomeCalibrator(min_samples=3, isotonic_min_samples=50, min_class_examples=1)
        calibrator.fit(
            [(0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7)],
            ["home", "draw", "away"],
        )
        with open(calibrator_path, "wb") as fh:
            pickle.dump(calibrator, fh)

        active_path.write_text("challenger")

        service = PredictionService(MagicMock())
        assert service.active_model == "dixon_coles"

        service.load_model()

        assert service.active_model == "challenger"
        assert service.challenger.is_fitted is True
        assert service.elo_system is not None
        assert service.calibrator is not None
        assert service.calibrator.is_fitted is True
        assert service.elo_system.get_rating("A") > service.elo_system.get_rating("B")

    def test_load_without_active_file_defaults_to_dc(self, tmp_model_dir):
        dc_path = tmp_model_dir / "dc.pkl"
        with open(dc_path, "wb") as fh:
            pickle.dump(_fit_dummy_dc_model(), fh)

        service = PredictionService(MagicMock())
        service.load_model()

        assert service.active_model == "dixon_coles"


class TestModelSelection:
    def test_fallback_to_dc_when_challenger_not_trained(self):
        matches = _make_home_win_matches()
        service = PredictionService(MagicMock())
        service.elo_system = EloSystem.from_matches(matches)

        with (
            patch.object(predictor_module, "backtest", return_value=_fake_dc_result(0.6)),
            patch.object(predictor_module, "mlflow", MagicMock()),
        ):
            service._evaluate_and_log(matches, challenger_trained=False)

        assert service.active_model == "dixon_coles"

    def test_challenger_selected_when_better_brier(self):
        matches = _make_home_win_matches()
        service = PredictionService(MagicMock())
        service.elo_system = EloSystem.from_matches(matches)
        strong_home_pred = MagicMock(
            home_win_prob=0.95,
            draw_prob=0.03,
            away_win_prob=0.02,
        )

        with (
            patch.object(predictor_module, "backtest", return_value=_fake_dc_result(0.2)),
            patch.object(ChallengerModel, "fit", return_value=None) as fit_mock,
            patch.object(ChallengerModel, "predict_match", return_value=strong_home_pred) as predict_mock,
            patch.object(predictor_module, "mlflow", MagicMock()),
        ):
            service._evaluate_and_log(matches, challenger_trained=True)

        fit_mock.assert_called_once()
        assert predict_mock.called
        assert service.active_model == "challenger"

    def test_dc_selected_when_equal_brier(self):
        matches = _make_home_win_matches()
        service = PredictionService(MagicMock())
        service.elo_system = EloSystem.from_matches(matches)
        equal_pred = MagicMock(
            home_win_prob=0.95,
            draw_prob=0.03,
            away_win_prob=0.02,
        )

        with (
            patch.object(predictor_module, "backtest", return_value=_fake_dc_result(0.0038)),
            patch.object(ChallengerModel, "fit", return_value=None),
            patch.object(ChallengerModel, "predict_match", return_value=equal_pred),
            patch.object(predictor_module, "mlflow", MagicMock()),
        ):
            service._evaluate_and_log(matches, challenger_trained=True)

        assert service.active_model == "dixon_coles"
