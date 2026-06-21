"""Small Bayesian models for personalization without GPU infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from .ml_utils import outcome_reward
from .models import jsonable


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _reward_from_feedback(event: dict[str, Any]) -> float | None:
    return outcome_reward(event)


@dataclass
class BetaPosterior:
    name: str
    alpha: float
    beta: float
    mean: float
    ci_low: float
    ci_high: float
    n_observations: int

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class NormalPosterior:
    name: str
    mean: float
    ci_low: float
    ci_high: float
    n_observations: int
    sample_std: float

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class BetaBernoulliModel:
    """Conjugate model for adherence probability."""

    def __init__(self, name: str, prior_alpha: float = 2.0, prior_beta: float = 2.0) -> None:
        self.name = name
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.successes = 0
        self.failures = 0

    def update(self, success: bool) -> None:
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def posterior(self) -> BetaPosterior:
        alpha = self.prior_alpha + self.successes
        beta = self.prior_beta + self.failures
        total = alpha + beta
        mean = alpha / total
        variance = (alpha * beta) / ((total * total) * (total + 1.0))
        width = 1.96 * sqrt(variance)
        return BetaPosterior(
            name=self.name,
            alpha=round(alpha, 3),
            beta=round(beta, 3),
            mean=round(mean, 3),
            ci_low=round(_clamp(mean - width), 3),
            ci_high=round(_clamp(mean + width), 3),
            n_observations=self.successes + self.failures,
        )


class EmpiricalNormalRewardModel:
    """Simple posterior-style summary for bounded rewards."""

    def __init__(self, name: str, prior_mean: float = 0.5, prior_weight: float = 2.0) -> None:
        self.name = name
        self.prior_mean = prior_mean
        self.prior_weight = prior_weight
        self.values: list[float] = []

    def update(self, reward: float) -> None:
        self.values.append(_clamp(float(reward)))

    def posterior(self) -> NormalPosterior:
        n = len(self.values)
        if n == 0:
            return NormalPosterior(
                name=self.name,
                mean=round(self.prior_mean, 3),
                ci_low=0.20,
                ci_high=0.80,
                n_observations=0,
                sample_std=0.0,
            )
        weighted_total = self.prior_mean * self.prior_weight + sum(self.values)
        weighted_n = self.prior_weight + n
        posterior_mean = weighted_total / weighted_n
        sample_mean = sum(self.values) / n
        sample_variance = sum((value - sample_mean) ** 2 for value in self.values) / max(n - 1, 1)
        sample_std = sqrt(sample_variance)
        standard_error = max(sample_std / sqrt(n), 0.08 / sqrt(n))
        width = 1.96 * standard_error
        return NormalPosterior(
            name=self.name,
            mean=round(posterior_mean, 3),
            ci_low=round(_clamp(posterior_mean - width), 3),
            ci_high=round(_clamp(posterior_mean + width), 3),
            n_observations=n,
            sample_std=round(sample_std, 3),
        )


@dataclass
class ActionBelief:
    action_id: str
    adherence: BetaPosterior
    reward: NormalPosterior
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class BayesianPersonalizer:
    """Maintains per-action beliefs from adherence and outcome feedback."""

    def __init__(self) -> None:
        self._adherence: dict[str, BetaBernoulliModel] = {}
        self._reward: dict[str, EmpiricalNormalRewardModel] = {}

    def _ensure_action(self, action_id: str) -> None:
        self._adherence.setdefault(action_id, BetaBernoulliModel(f"{action_id}.adherence"))
        self._reward.setdefault(action_id, EmpiricalNormalRewardModel(f"{action_id}.reward"))

    def update_from_feedback(self, feedback_events: list[dict[str, Any]]) -> None:
        for event in feedback_events:
            action_id = event.get("action_id")
            if not action_id:
                continue
            self._ensure_action(str(action_id))
            if event.get("adhered") is not None:
                self._adherence[str(action_id)].update(bool(event["adhered"]))
            reward = _reward_from_feedback(event)
            if reward is not None:
                self._reward[str(action_id)].update(reward)

    def summarize(self, action_ids: list[str] | None = None) -> dict[str, dict[str, Any]]:
        ids = sorted(set(action_ids or []) | set(self._adherence) | set(self._reward))
        output: dict[str, dict[str, Any]] = {}
        for action_id in ids:
            self._ensure_action(action_id)
            adherence = self._adherence[action_id].posterior()
            reward = self._reward[action_id].posterior()
            notes = []
            if adherence.n_observations < 5:
                notes.append("low_data_prior_dominates")
            if reward.n_observations < 5:
                notes.append("reward_uncertainty_high")
            output[action_id] = ActionBelief(
                action_id=action_id,
                adherence=adherence,
                reward=reward,
                notes=notes,
            ).to_dict()
        return output
