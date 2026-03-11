"""Dixon-Coles model for predicting football match scores.

The Dixon-Coles model extends the independent Poisson model by:
1. Estimating attack and defense strength parameters for each team
2. Including a home advantage parameter
3. Applying a correction factor (rho) for low-scoring outcomes
4. Using time-weighted maximum likelihood estimation

Reference: Dixon, M.J. & Coles, S.G. (1997) "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market"
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


MAX_GOALS = 10  # Max goals to consider in score matrix


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


def _match_log_likelihood(
    home_goals: int,
    away_goals: int,
    home_attack: float,
    home_defense: float,
    away_attack: float,
    away_defense: float,
    home_adv: float,
    rho: float,
    weight: float = 1.0,
) -> float:
    """Compute log-likelihood of a single match result."""
    lambda_ = np.exp(home_attack + away_defense + home_adv)  # Expected home goals
    mu = np.exp(away_attack + home_defense)  # Expected away goals

    # Poisson probabilities
    home_prob = poisson.pmf(home_goals, lambda_)
    away_prob = poisson.pmf(away_goals, mu)

    # Dixon-Coles correction
    tau = _tau(home_goals, away_goals, lambda_, mu, rho)

    prob = tau * home_prob * away_prob
    if prob <= 0:
        return -30.0 * weight  # Avoid log(0)

    return weight * np.log(prob)


def _neg_log_likelihood(params: np.ndarray, matches: list[MatchData], teams: list[str]) -> float:
    """Negative log-likelihood for all matches (to minimize).

    Parameter layout:
      params[0:n]     = attack strengths for each team
      params[n:2n]    = defense strengths for each team
      params[2n]      = home advantage
      params[2n+1]    = rho (Dixon-Coles correction)
    """
    n = len(teams)
    team_idx = {team: i for i, team in enumerate(teams)}

    attack = params[:n]
    defense = params[n : 2 * n]
    home_adv = params[2 * n]
    rho = params[2 * n + 1]

    log_lik = 0.0
    for match in matches:
        hi = team_idx[match.home_team]
        ai = team_idx[match.away_team]
        log_lik += _match_log_likelihood(
            match.home_goals,
            match.away_goals,
            attack[hi],
            defense[hi],
            attack[ai],
            defense[ai],
            home_adv,
            rho,
            match.weight,
        )

    return -log_lik


class DixonColesModel:
    """Dixon-Coles model for EPL score prediction."""

    def __init__(self, time_decay_days: int = 365):
        """Initialize the model.

        Args:
            time_decay_days: Half-life for time weighting in days.
                Matches older than this get half the weight.
        """
        self.time_decay_days = time_decay_days
        self.params: ModelParams | None = None

    def fit(self, matches: list[MatchData]) -> ModelParams:
        """Fit the model on historical match data.

        Args:
            matches: List of MatchData with results and time weights.

        Returns:
            Fitted ModelParams.
        """
        teams = sorted(set(
            [m.home_team for m in matches] + [m.away_team for m in matches]
        ))
        n = len(teams)

        # Initial params: zero attack/defense, small home advantage, zero rho
        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25  # Initial home advantage
        x0[2 * n + 1] = -0.05  # Initial rho

        # Constraint: sum of attack strengths = 0 (identifiability)
        constraints = [
            {"type": "eq", "fun": lambda p, n=n: np.sum(p[:n])}
        ]

        # Bounds: rho between -1 and 1
        bounds = [(None, None)] * (2 * n)  # attack + defense: unbounded
        bounds.append((None, None))  # home advantage: unbounded
        bounds.append((-1.0, 1.0))  # rho: bounded

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                _neg_log_likelihood,
                x0,
                args=(matches, teams),
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-8},
            )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        # Extract parameters
        attack = {team: result.x[i] for i, team in enumerate(teams)}
        defense = {team: result.x[n + i] for i, team in enumerate(teams)}
        home_adv = result.x[2 * n]
        rho = result.x[2 * n + 1]

        self.params = ModelParams(
            teams=teams,
            attack=attack,
            defense=defense,
            home_advantage=home_adv,
            rho=rho,
        )
        return self.params

    def predict_match(self, home_team: str, away_team: str) -> MatchPrediction:
        """Predict the outcome of a match.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.

        Returns:
            MatchPrediction with all derived prediction types.

        Raises:
            ValueError: If model is not fitted or team is unknown.
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if home_team not in self.params.attack:
            raise ValueError(f"Unknown team: {home_team}")
        if away_team not in self.params.attack:
            raise ValueError(f"Unknown team: {away_team}")

        # Expected goals
        lambda_ = np.exp(
            self.params.attack[home_team]
            + self.params.defense[away_team]
            + self.params.home_advantage
        )
        mu = np.exp(
            self.params.attack[away_team]
            + self.params.defense[home_team]
        )

        # Build score probability matrix
        score_matrix = self._calculate_score_matrix(lambda_, mu)

        # Derive all prediction types
        prediction = MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            score_matrix=score_matrix,
            predicted_home_goals=round(lambda_, 2),
            predicted_away_goals=round(mu, 2),
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
        matrix = np.zeros((MAX_GOALS, MAX_GOALS))

        for i in range(MAX_GOALS):
            for j in range(MAX_GOALS):
                base_prob = poisson.pmf(i, lambda_) * poisson.pmf(j, mu)
                tau = _tau(i, j, lambda_, mu, rho)
                matrix[i, j] = base_prob * tau

        # Normalize to ensure probabilities sum to 1
        matrix /= matrix.sum()
        return matrix

    @staticmethod
    def _derive_predictions(prediction: MatchPrediction, matrix: np.ndarray) -> None:
        """Derive all prediction types from the score matrix.

        Modifies prediction in place.
        """
        n = matrix.shape[0]

        # 1X2 outcome probabilities
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        for i in range(n):
            for j in range(n):
                if i > j:
                    home_win += matrix[i, j]
                elif i == j:
                    draw += matrix[i, j]
                else:
                    away_win += matrix[i, j]

        prediction.home_win_prob = round(home_win, 4)
        prediction.draw_prob = round(draw, 4)
        prediction.away_win_prob = round(away_win, 4)

        # Over/Under 2.5 goals
        over25 = 0.0
        for i in range(n):
            for j in range(n):
                if i + j > 2:
                    over25 += matrix[i, j]
        prediction.over25_prob = round(over25, 4)

        # Both Teams To Score (BTTS)
        btts = 0.0
        for i in range(1, n):
            for j in range(1, n):
                btts += matrix[i, j]
        prediction.btts_prob = round(btts, 4)

        # Top 5 most likely exact scores
        scores = []
        for i in range(n):
            for j in range(n):
                scores.append((f"{i}-{j}", matrix[i, j], i, j))
        scores.sort(key=lambda x: x[1], reverse=True)
        prediction.top_scores = [(s, round(p, 4)) for s, p, _, _ in scores[:5]]
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
