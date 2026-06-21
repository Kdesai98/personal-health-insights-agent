"""Contextual bandit personalization for low-risk health actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from .ml_utils import CORE_FEATURE_NAMES, clamp, dot, normalized_feature_vector, outcome_reward
from .models import jsonable, now_iso


@dataclass
class BanditArm:
    action_id: str
    weights: list[float]
    intercept: float
    n: int
    mean_reward: float
    uncertainty: float


@dataclass
class BanditPolicyArtifact:
    policy_id: str
    feature_names: list[str]
    arms: dict[str, BanditArm]
    trained_at: str = field(default_factory=now_iso)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class BanditDecision:
    selected_action_id: str
    scores: dict[str, dict[str, float]]
    policy_id: str
    decision_rule: str

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class ContextualBanditTrainer:
    """Trains per-action linear reward heads with uncertainty bonuses."""

    def __init__(
        self,
        action_ids: list[str],
        feature_names: list[str] | None = None,
        learning_rate: float = 0.04,
        l2: float = 0.02,
        epochs: int = 80,
    ) -> None:
        self.action_ids = sorted(set(action_ids))
        self.feature_names = feature_names or CORE_FEATURE_NAMES
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs

    def fit(
        self,
        feature_bundles: list[dict[str, Any]],
        feedback_events: list[dict[str, Any]],
    ) -> BanditPolicyArtifact:
        bundle_by_date = {
            str(bundle.get("as_of_date")): bundle
            for bundle in feature_bundles
        }
        by_action: dict[str, list[tuple[list[float], float]]] = {
            action_id: [] for action_id in self.action_ids
        }
        for event in feedback_events:
            action_id = str(event.get("action_id"))
            if action_id not in by_action:
                continue
            reward = outcome_reward(event)
            bundle = bundle_by_date.get(str(event.get("date")))
            if reward is None or bundle is None:
                continue
            by_action[action_id].append(
                (normalized_feature_vector(bundle, self.feature_names), reward)
            )

        arms: dict[str, BanditArm] = {}
        notes = []
        for action_id, examples in by_action.items():
            if len(examples) < 3:
                arms[action_id] = BanditArm(
                    action_id=action_id,
                    weights=[0.0 for _ in self.feature_names],
                    intercept=0.50,
                    n=len(examples),
                    mean_reward=0.50,
                    uncertainty=0.35,
                )
                notes.append(f"{action_id}: low_data_prior_used")
                continue

            w = [0.0 for _ in self.feature_names]
            b = sum(reward for _, reward in examples) / len(examples)
            for _ in range(self.epochs):
                for x, reward in examples:
                    pred = clamp(b + dot(w, x))
                    err = pred - reward
                    b -= self.learning_rate * err
                    for idx, value in enumerate(x):
                        w[idx] -= self.learning_rate * (err * value + self.l2 * w[idx])
            residuals = [reward - clamp(b + dot(w, x)) for x, reward in examples]
            variance = sum(value * value for value in residuals) / max(len(residuals), 1)
            arms[action_id] = BanditArm(
                action_id=action_id,
                weights=[round(value, 6) for value in w],
                intercept=round(b, 6),
                n=len(examples),
                mean_reward=round(sum(reward for _, reward in examples) / len(examples), 3),
                uncertainty=round(max(0.05, sqrt(variance) + 0.20 / sqrt(len(examples))), 3),
            )

        return BanditPolicyArtifact(
            policy_id=f"contextual_bandit_{now_iso()}",
            feature_names=self.feature_names,
            arms=arms,
            notes=notes,
        )


class ContextualBanditPolicy:
    """Selects actions by expected reward plus an exploration bonus."""

    def __init__(self, artifact: BanditPolicyArtifact, exploration_weight: float = 0.25) -> None:
        self.artifact = artifact
        self.exploration_weight = exploration_weight

    def choose(
        self,
        feature_bundle: dict[str, Any],
        allowed_actions: list[str] | None = None,
    ) -> BanditDecision:
        x = normalized_feature_vector(feature_bundle, self.artifact.feature_names)
        allowed = set(allowed_actions or self.artifact.arms.keys())
        scores = {}
        for action_id, arm in self.artifact.arms.items():
            if action_id not in allowed:
                continue
            expected_reward = clamp(arm.intercept + dot(arm.weights, x))
            exploration_bonus = self.exploration_weight * arm.uncertainty
            scores[action_id] = {
                "expected_reward": round(expected_reward, 3),
                "uncertainty": round(arm.uncertainty, 3),
                "exploration_bonus": round(exploration_bonus, 3),
                "score": round(expected_reward + exploration_bonus, 3),
                "n": float(arm.n),
            }
        if not scores:
            raise ValueError("No allowed bandit actions were available")
        selected = max(scores.items(), key=lambda item: item[1]["score"])[0]
        return BanditDecision(
            selected_action_id=selected,
            scores=scores,
            policy_id=self.artifact.policy_id,
            decision_rule="linear_ucb",
        )
