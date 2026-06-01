"""Runtime causal summaries for low-data personal experiments.

These estimators are intentionally conservative. They are useful for showing the
app whether it has enough evidence to personalize a recommendation, while
keeping observational uncertainty visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from .ml_utils import outcome_reward
from .models import PosteriorDistribution, jsonable


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return sum((value - avg) ** 2 for value in values) / (len(values) - 1)


@dataclass
class NOf1CausalSummary:
    action_id: str
    outcome: str
    status: str
    posterior: PosteriorDistribution
    treated_n: int
    control_n: int
    assumptions: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class NOf1CausalRuntime:
    """Small-sample causal summaries over logged intervention feedback."""

    def estimate_action_effect(
        self,
        feedback_events: list[dict[str, Any]],
        action_id: str,
        outcome: str = "utility_reward",
        minimum_group_size: int = 3,
    ) -> NOf1CausalSummary:
        treated: list[float] = []
        control: list[float] = []
        for event in feedback_events:
            reward = self._outcome_value(event, outcome)
            if reward is None:
                continue
            if event.get("action_id") == action_id:
                treated.append(reward)
            else:
                control.append(reward)

        assumptions = [
            "same-user comparison",
            "feedback reward is a proxy for next-day health utility",
            "no unmeasured time-varying confounding strong enough to dominate the signal",
        ]
        warnings = []
        status = "estimable"
        if len(treated) < minimum_group_size or len(control) < minimum_group_size:
            status = "insufficient_data"
            warnings.append("Need more treated and non-treated days before trusting this effect.")
        if len(feedback_events) < 14:
            warnings.append("Less than two weeks of feedback; posterior should be treated as exploratory.")

        effect = (_mean(treated) - _mean(control)) if treated and control else 0.0
        treated_var = _variance(treated) / max(len(treated), 1)
        control_var = _variance(control) / max(len(control), 1)
        standard_error = sqrt(treated_var + control_var) if treated and control else 0.22
        standard_error = max(standard_error, 0.08)
        width = 1.96 * standard_error
        probability = _clamp(0.5 + effect / max(2.0 * width, 0.16))
        posterior = PosteriorDistribution(
            mean=round(effect, 3),
            ci_low=round(effect - width, 3),
            ci_high=round(effect + width, 3),
            probability_above_threshold=round(probability, 3),
            method="n_of_1_difference_in_means",
            identification_uncertainty=0.55 if status == "insufficient_data" else 0.35,
            notes=[
                "This is an observational N-of-1 summary, not proof of causality.",
                "Use randomized or alternating experiments before making strong claims.",
            ],
        )
        return NOf1CausalSummary(
            action_id=action_id,
            outcome=outcome,
            status=status,
            posterior=posterior,
            treated_n=len(treated),
            control_n=len(control),
            assumptions=assumptions,
            warnings=warnings,
        )

    def estimate_all_actions(
        self,
        feedback_events: list[dict[str, Any]],
        action_ids: list[str],
        outcome: str = "utility_reward",
    ) -> dict[str, dict[str, Any]]:
        return {
            action_id: self.estimate_action_effect(
                feedback_events=feedback_events,
                action_id=action_id,
                outcome=outcome,
            ).to_dict()
            for action_id in sorted(set(action_ids))
        }

    def _outcome_value(self, event: dict[str, Any], outcome: str) -> float | None:
        if outcome == "utility_reward":
            return outcome_reward(event)
        outcomes = event.get("outcomes") or {}
        if outcomes.get(outcome) is None:
            return None
        value = float(outcomes[outcome])
        if value > 1.0 and outcome.endswith("_1_to_10"):
            value /= 10.0
        if value > 1.0 and "readiness" in outcome:
            value /= 100.0
        return _clamp(value)
