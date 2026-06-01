"""Trainable state encoder artifacts for local, low-data personalization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from .ml_utils import CORE_FEATURE_NAMES, clamp, dot, normalized_feature_vector, outcome_reward, sigmoid
from .models import ScoreWithUncertainty, jsonable, now_iso


TARGET_NAMES = [
    "utility_reward",
    "adherence_probability",
    "next_day_energy",
    "next_day_readiness",
]


def _target_values(event: dict[str, Any]) -> dict[str, float]:
    outcomes = event.get("outcomes") or {}
    values: dict[str, float] = {}
    reward = outcome_reward(event)
    if reward is not None:
        values["utility_reward"] = reward
    if event.get("adhered") is not None:
        values["adherence_probability"] = 1.0 if event["adhered"] else 0.0
    if outcomes.get("energy_next_day_1_to_10") is not None:
        values["next_day_energy"] = clamp(float(outcomes["energy_next_day_1_to_10"]) / 10.0)
    if outcomes.get("readiness_next_day") is not None:
        readiness = float(outcomes["readiness_next_day"])
        values["next_day_readiness"] = clamp(readiness / 100.0 if readiness > 1.0 else readiness)
    return values


@dataclass
class StateModelArtifact:
    model_id: str
    feature_names: list[str]
    target_names: list[str]
    weights: dict[str, list[float]]
    intercepts: dict[str, float]
    residual_std: dict[str, float]
    n_examples: dict[str, int]
    trained_at: str = field(default_factory=now_iso)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StateModelArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


class StateModelTrainer:
    """Fits small linear heads on top of the longitudinal feature bundle."""

    def __init__(
        self,
        feature_names: list[str] | None = None,
        target_names: list[str] | None = None,
        learning_rate: float = 0.05,
        l2: float = 0.01,
        epochs: int = 120,
    ) -> None:
        self.feature_names = feature_names or CORE_FEATURE_NAMES
        self.target_names = target_names or TARGET_NAMES
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs

    def fit(
        self,
        feature_bundles: list[dict[str, Any]],
        feedback_events: list[dict[str, Any]],
    ) -> StateModelArtifact:
        bundle_by_date = {
            str(bundle.get("as_of_date")): bundle
            for bundle in feature_bundles
        }
        examples: dict[str, list[tuple[list[float], float]]] = {
            target: [] for target in self.target_names
        }
        for event in feedback_events:
            bundle = bundle_by_date.get(str(event.get("date")))
            if not bundle:
                continue
            x = normalized_feature_vector(bundle, self.feature_names)
            for target, value in _target_values(event).items():
                if target in examples:
                    examples[target].append((x, value))

        weights: dict[str, list[float]] = {}
        intercepts: dict[str, float] = {}
        residual_std: dict[str, float] = {}
        n_examples: dict[str, int] = {}
        notes = []
        for target in self.target_names:
            target_examples = examples[target]
            n_examples[target] = len(target_examples)
            if len(target_examples) < 3:
                weights[target] = [0.0 for _ in self.feature_names]
                intercepts[target] = self._prior_for_target(target)
                residual_std[target] = 0.30
                notes.append(f"{target}: low_data_prior_used")
                continue
            w = [0.0 for _ in self.feature_names]
            b = mean(value for _, value in target_examples)
            for _ in range(self.epochs):
                for x, y in target_examples:
                    pred = clamp(b + dot(w, x))
                    err = pred - y
                    b -= self.learning_rate * err
                    for idx, value in enumerate(x):
                        w[idx] -= self.learning_rate * (err * value + self.l2 * w[idx])
            residuals = [y - clamp(b + dot(w, x)) for x, y in target_examples]
            residual_std[target] = max(
                0.06,
                (sum(residual * residual for residual in residuals) / len(residuals)) ** 0.5,
            )
            weights[target] = [round(value, 6) for value in w]
            intercepts[target] = round(b, 6)

        return StateModelArtifact(
            model_id=f"state_model_{now_iso()}",
            feature_names=self.feature_names,
            target_names=self.target_names,
            weights=weights,
            intercepts=intercepts,
            residual_std={key: round(value, 4) for key, value in residual_std.items()},
            n_examples=n_examples,
            notes=notes,
        )

    def _prior_for_target(self, target: str) -> float:
        priors = {
            "utility_reward": 0.50,
            "adherence_probability": 0.55,
            "next_day_energy": 0.65,
            "next_day_readiness": 0.75,
        }
        return priors.get(target, 0.50)


class TrainableStateEncoder:
    """Predicts outcome-oriented state scores from a trained local artifact."""

    def __init__(self, artifact: StateModelArtifact) -> None:
        self.artifact = artifact

    def predict(self, feature_bundle: dict[str, Any]) -> dict[str, ScoreWithUncertainty]:
        x = normalized_feature_vector(feature_bundle, self.artifact.feature_names)
        scores = {}
        for target in self.artifact.target_names:
            raw = self.artifact.intercepts[target] + dot(self.artifact.weights[target], x)
            mean_value = sigmoid(raw) if raw < 0.0 or raw > 1.0 else clamp(raw)
            n = self.artifact.n_examples.get(target, 0)
            residual = self.artifact.residual_std.get(target, 0.30)
            epistemic = min(0.30, 0.24 / max(n, 1) ** 0.5)
            width = residual + epistemic
            scores[target] = ScoreWithUncertainty(
                mean=round(mean_value, 3),
                ci_low=round(clamp(mean_value - width), 3),
                ci_high=round(clamp(mean_value + width), 3),
                method="local_linear_state_model",
                calibration_subgroup="low_data" if n < 20 else "personal_history",
                epistemic_uncertainty=round(epistemic, 3),
                aleatoric_uncertainty=round(residual, 3),
            )
        return scores
