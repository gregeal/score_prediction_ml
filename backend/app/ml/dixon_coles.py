"""Dixon-Coles model for predicting football match scores.

The Dixon-Coles model extends the independent Poisson model by:
1. Estimating attack and defense strength parameters for each team
2. Including a home advantage parameter
3. Applying a correction factor (rho) for low-scoring outcomes
4. Using time-weighted maximum likelihood estimation

This implementation fits by fully vectorized penalized maximum likelihood:
the log-likelihood is computed with numpy over all matches at once and
optimized with L-BFGS-B, which is several orders of magnitude faster than
a per-match Python loop under SLSQP. A small L2 penalty shrinks attack and
defense strengths toward the league average, which keeps parameters sane
for teams with very few observed matches (e.g. newly promoted sides early
in a season).

Reference: Dixon, M.J. & Coles, S.G. (1997) "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market"
"""

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

logger = logging.getLogger(__name__)

MAX_GOALS = 10  # Max goals to consider in score matrix

# tau factors are clipped to this floor inside the log-likelihood so that
# parameter regions where the Dixon-Coles correction becomes invalid
# (tau <= 0) are heavily penalized but never produce log(<=0).
_TAU_FLOOR = 1e-10


@dataclass
class MatchData:
    """A single match result for model training."""
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    weight: float = 1.0  # Time decay weight


@dataclass
class MatchPrediction:
    """Full prediction output for a single match."""
    home_team: str
    away_team: str
    score_matrix: np.ndarray  # (MAX_GOALS x MAX_GOALS) probability matrix
    home_win_prob: float = 0.0
    draw_prob: float = 0.0
    away_win_prob: float = 0.0
    over25_prob: float = 0.0
    btts_prob: float = 0.0
    top_scores: list = field(default_factory=list)  # [(score_str, prob), ...]
    most_likely_score: str = ""  # Overall most likely scoreline
    outcome_score: str = ""  # Most likely score consistent with predicted outcome
    confidence: str = "medium"
    predicted_home_goals: float = 0.0
    predicted_away_goals: float = 0.0


@dataclass
class ModelParams:
    """Fitted model parameters."""
    teams: list[str]
    attack: dict[str, float]  # Team -> attack strength
    defense: dict[str, float]  # Team -> defense strength
    home_advantage: float
    rho: float  # Dixon-Coles low-score correction
    # Fallback strengths served for teams absent from training data
    # (e.g. newly promoted sides before their first finished match).
    default_attack: float = 0.0
    default_defense: float = 0.0


def _tau(x: int, y: int, lambda_: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring outcomes.

    Adjusts probabilities for 0-0, 1-0, 0-1, and 1-1 scorelines which
    the independent Poisson model gets wrong due to correlation between
    home and away goals in low-scoring games.
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_ * mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_ * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


class DixonColesModel:
    """Dixon-Coles model for EPL score prediction."""

    def __init__(self, time_decay_days: int = 540, l2_reg: float = 5.0):
        """Initialize the model.

        Args:
            time_decay_days: Half-life for time weighting in days.
                Matches older than this get half the weight.
            l2_reg: L2 penalty strength on attack/defense parameters.
                Shrinks strengths toward the league average; mainly
                stabilizes teams with few observed matches.

        Defaults were selected by walk-forward backtest over the most recent
        380 finished matches (see scripts/benchmark_model.py): half_life=540
        with l2_reg=5.0 gave the best Brier score and log loss.
        """
        self.time_decay_days = time_decay_days
        self.l2_reg = l2_reg
        self.params: ModelParams | None = None

    def fit(self, matches: list[MatchData]) -> ModelParams:
        """Fit the model on historical match data.

        Args:
            matches: List of MatchData with results and time weights.

        Returns:
            Fitted ModelParams.
        """
        if not matches:
            raise ValueError("No matches provided to fit()")

        teams = sorted(set(
            [m.home_team for m in matches] + [m.away_team for m in matches]
        ))
        n = len(teams)
        team_idx = {team: i for i, team in enumerate(teams)}

        # Precompute vectorized match arrays
        home_idx = np.array([team_idx[m.home_team] for m in matches], dtype=np.intp)
        away_idx = np.array([team_idx[m.away_team] for m in matches], dtype=np.intp)
        home_goals = np.array([m.home_goals for m in matches], dtype=np.float64)
        away_goals = np.array([m.away_goals for m in matches], dtype=np.float64)
        weights = np.array([m.weight for m in matches], dtype=np.float64)

        # Constant terms of the Poisson log-pmf
        log_factorials = gammaln(home_goals + 1.0) + gammaln(away_goals + 1.0)

        # Masks for the four scorelines the tau correction touches
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_11 = (home_goals == 1) & (away_goals == 1)

        l2_reg = self.l2_reg

        def neg_log_likelihood(params: np.ndarray) -> float:
            attack = params[:n]
            defense = params[n:2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]

            log_lambda = attack[home_idx] + defense[away_idx] + home_adv
            log_mu = attack[away_idx] + defense[home_idx]
            lambda_ = np.exp(log_lambda)
            mu = np.exp(log_mu)

            log_lik = (
                home_goals * log_lambda - lambda_
                + away_goals * log_mu - mu
                - log_factorials
            )

            tau = np.ones_like(lambda_)
            tau[mask_00] = 1.0 - lambda_[mask_00] * mu[mask_00] * rho
            tau[mask_01] = 1.0 + lambda_[mask_01] * rho
            tau[mask_10] = 1.0 + mu[mask_10] * rho
            tau[mask_11] = 1.0 - rho
            log_lik += np.log(np.clip(tau, _TAU_FLOOR, None))

            penalty = l2_reg * (np.dot(attack, attack) + np.dot(defense, defense))
            return -np.dot(weights, log_lik) + penalty

        # Initial params: zero attack/defense, small home advantage, small rho
        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25
        x0[2 * n + 1] = -0.05

        # Attack/defense bounded generously (exp(3) ~ 20 goals); rho kept in a
        # range where the tau correction stays a valid probability adjustment
        # for realistic scoring rates.
        bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 1.0), (-0.5, 0.5)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500},
            )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        # The likelihood is invariant under attack += c, defense -= c.
        # Fix the gauge so attack strengths average to zero (identifiability),
        # matching the constraint the original SLSQP formulation imposed.
        attack_arr = result.x[:n].copy()
        defense_arr = result.x[n:2 * n].copy()
        shift = attack_arr.mean()
        attack_arr -= shift
        defense_arr += shift

        attack = {team: float(attack_arr[i]) for i, team in enumerate(teams)}
        defense = {team: float(defense_arr[i]) for i, team in enumerate(teams)}

        # Fallback strengths for unknown (e.g. newly promoted) teams: the
        # average of the three weakest teams by net strength. Promoted sides
        # historically perform like the bottom of the league.
        net_strength = attack_arr - defense_arr
        weakest = np.argsort(net_strength)[:min(3, n)]
        default_attack = float(attack_arr[weakest].mean())
        default_defense = float(defense_arr[weakest].mean())

        self.params = ModelParams(
            teams=teams,
            attack=attack,
            defense=defense,
            home_advantage=float(result.x[2 * n]),
            rho=float(result.x[2 * n + 1]),
            default_attack=default_attack,
            default_defense=default_defense,
        )
        return self.params

    def is_known_team(self, team: str) -> bool:
        """Whether the team was present in the training data."""
        return self.params is not None and team in self.params.attack

    def _team_strengths(self, team: str) -> tuple[float, float]:
        """Attack/defense for a team, falling back to promoted-team defaults."""
        if team in self.params.attack:
            return self.params.attack[team], self.params.defense[team]
        warned = getattr(self, "_warned_unknown_teams", None)
        if warned is None:
            warned = self._warned_unknown_teams = set()
        if team not in warned:
            warned.add(team)
            logger.warning(
                "Team %r not in training data; using promoted-team default strengths",
                team,
            )
        return self.params.default_attack, self.params.default_defense

    def predict_match(self, home_team: str, away_team: str) -> MatchPrediction:
        """Predict the outcome of a match.

        Teams unseen in training (e.g. newly promoted sides) are assigned
        the fallback strengths estimated in fit() rather than raising.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.

        Returns:
            MatchPrediction with all derived prediction types.

        Raises:
            ValueError: If model is not fitted.
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        home_attack, home_defense = self._team_strengths(home_team)
        away_attack, away_defense = self._team_strengths(away_team)

        # Expected goals
        lambda_ = np.exp(home_attack + away_defense + self.params.home_advantage)
        mu = np.exp(away_attack + home_defense)

        # Build score probability matrix
        score_matrix = self._calculate_score_matrix(lambda_, mu)

        # Derive all prediction types
        prediction = MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            score_matrix=score_matrix,
            predicted_home_goals=round(float(lambda_), 2),
            predicted_away_goals=round(float(mu), 2),
        )

        self._derive_predictions(prediction, score_matrix)
        return prediction

    def _calculate_score_matrix(self, lambda_: float, mu: float) -> np.ndarray:
        """Calculate the score probability matrix with Dixon-Coles correction.

        Args:
            lambda_: Expected home goals.
            mu: Expected away goals.

        Returns:
            (MAX_GOALS x MAX_GOALS) numpy array of score probabilities.
        """
        rho = self.params.rho if self.params else 0.0

        home_probs = poisson.pmf(np.arange(MAX_GOALS), lambda_)
        away_probs = poisson.pmf(np.arange(MAX_GOALS), mu)
        matrix = np.outer(home_probs, away_probs)

        matrix[0, 0] *= 1.0 - lambda_ * mu * rho
        matrix[0, 1] *= 1.0 + lambda_ * rho
        matrix[1, 0] *= 1.0 + mu * rho
        matrix[1, 1] *= 1.0 - rho

        # The tau correction can push individual cells negative when rho is
        # outside its validity range for extreme lambda/mu; clip before
        # normalizing so the matrix is a valid probability distribution.
        matrix = np.clip(matrix, 0.0, None)
        matrix /= matrix.sum()
        return matrix

    @staticmethod
    def _derive_predictions(prediction: MatchPrediction, matrix: np.ndarray) -> None:
        """Derive all prediction types from the score matrix.

        Modifies prediction in place.
        """
        n = matrix.shape[0]
        home_idx, away_idx = np.indices((n, n))

        home_win = float(matrix[home_idx > away_idx].sum())
        draw = float(matrix[home_idx == away_idx].sum())
        away_win = float(matrix[home_idx < away_idx].sum())

        prediction.home_win_prob = round(home_win, 4)
        prediction.draw_prob = round(draw, 4)
        prediction.away_win_prob = round(away_win, 4)

        # Over/Under 2.5 goals
        prediction.over25_prob = round(float(matrix[home_idx + away_idx > 2].sum()), 4)

        # Both Teams To Score (BTTS)
        prediction.btts_prob = round(float(matrix[1:, 1:].sum()), 4)

        # Top 5 most likely exact scores
        scores = []
        for i in range(n):
            for j in range(n):
                scores.append((f"{i}-{j}", matrix[i, j], i, j))
        scores.sort(key=lambda x: x[1], reverse=True)
        prediction.top_scores = [(s, round(float(p), 4)) for s, p, _, _ in scores[:5]]
        prediction.most_likely_score = scores[0][0]

        # Most likely score consistent with the predicted outcome
        # This avoids the confusing case where team A is favored but predicted score is a draw
        predicted_outcome = "home" if home_win >= away_win and home_win >= draw else (
            "away" if away_win >= home_win and away_win >= draw else "draw"
        )
        for score_str, prob, i, j in scores:
            if predicted_outcome == "home" and i > j:
                prediction.outcome_score = score_str
                break
            elif predicted_outcome == "away" and j > i:
                prediction.outcome_score = score_str
                break
            elif predicted_outcome == "draw" and i == j:
                prediction.outcome_score = score_str
                break
        if not prediction.outcome_score:
            prediction.outcome_score = prediction.most_likely_score

        # Confidence rating
        max_outcome = max(home_win, draw, away_win)
        if max_outcome >= 0.60:
            prediction.confidence = "high"
        elif max_outcome >= 0.45:
            prediction.confidence = "medium"
        else:
            prediction.confidence = "low"
