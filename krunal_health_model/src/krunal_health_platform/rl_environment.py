"""Offline RL-style environment for health policy simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .features import extract_daily_features
from .ml_utils import CORE_FEATURE_NAMES, clamp
from .models import jsonable, now_iso


REWARD_WEIGHTS = {
    "readiness_score": 0.25,
    "energy_1_to_10": 0.18,
    "focus_1_to_10": 0.14,
    "digestion_1_to_10": 0.12,
    "sleep_debt_minutes": -0.16,
    "recovery_pressure": -0.15,
}


ACTION_PRIOR_DELTAS = {
    "strength_progression": {
        "training_load_score": 35.0,
        "soreness_1_to_10": 0.7,
        "energy_1_to_10": -0.2,
        "readiness_score": -1.0,
    },
    "recovery_row": {
        "training_load_score": 18.0,
        "soreness_1_to_10": -0.2,
        "energy_1_to_10": 0.2,
        "recovery_pressure": -0.03,
    },
    "bedtime_protection": {
        "sleep_debt_minutes": -20.0,
        "readiness_score": 1.8,
        "energy_1_to_10": 0.2,
    },
    "protein_portion_check": {
        "protein_target_gap_g": -18.0,
        "nutrition_confidence": 0.08,
    },
    "travel_context_check": {
        "nutrition_confidence": 0.03,
    },
}


@dataclass
class TransitionModelArtifact:
    action_deltas: dict[str, dict[str, float]]
    n_transitions: dict[str, int]
    trained_at: str = field(default_factory=now_iso)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class HealthTransitionModel:
    """Learns average next-state deltas by action from daily states and traces."""

    def fit(
        self,
        daily_states: list[dict[str, Any]],
        intervention_traces: list[dict[str, Any]],
    ) -> TransitionModelArtifact:
        states_by_date = {
            str(state.get("date")): extract_daily_features(state).metrics
            for state in daily_states
        }
        actions_by_date = {}
        for trace in intervention_traces:
            date = trace.get("date")
            if not date:
                created_at = str(trace.get("created_at", ""))
                date = created_at[:10] if created_at else None
            action_id = trace.get("selected_action_id")
            if date and action_id:
                actions_by_date[str(date)] = action_id

        ordered_dates = sorted(states_by_date)
        deltas: dict[str, list[dict[str, float]]] = {}
        for idx, date in enumerate(ordered_dates[:-1]):
            action_id = actions_by_date.get(date)
            if not action_id:
                continue
            current = states_by_date[date]
            nxt = states_by_date[ordered_dates[idx + 1]]
            delta = {}
            for feature in CORE_FEATURE_NAMES:
                if isinstance(current.get(feature), (int, float)) and isinstance(nxt.get(feature), (int, float)):
                    delta[feature] = float(nxt[feature]) - float(current[feature])
            if delta:
                deltas.setdefault(str(action_id), []).append(delta)

        action_deltas: dict[str, dict[str, float]] = {}
        n_transitions = {}
        notes = []
        for action_id, prior in ACTION_PRIOR_DELTAS.items():
            observed = deltas.get(action_id, [])
            n_transitions[action_id] = len(observed)
            if not observed:
                action_deltas[action_id] = prior
                notes.append(f"{action_id}: prior_transition_used")
                continue
            keys = sorted({key for item in observed for key in item})
            action_deltas[action_id] = {
                key: round(sum(item.get(key, 0.0) for item in observed) / len(observed), 3)
                for key in keys
            }
        return TransitionModelArtifact(
            action_deltas=action_deltas,
            n_transitions=n_transitions,
            notes=notes,
        )


class HealthRLEnvironment:
    """A small gym-like environment for offline policy simulation."""

    def __init__(self, transition_model: TransitionModelArtifact, horizon: int = 7) -> None:
        self.transition_model = transition_model
        self.horizon = horizon
        self.state: dict[str, float | bool | None] = {}
        self.t = 0

    def reset(self, feature_bundle: dict[str, Any]) -> dict[str, float | bool | None]:
        self.state = dict(feature_bundle.get("current", feature_bundle))
        self.t = 0
        return dict(self.state)

    def step(self, action_id: str) -> tuple[dict[str, float | bool | None], float, bool, dict[str, Any]]:
        if not self.state:
            raise RuntimeError("Call reset before step")
        deltas = self.transition_model.action_deltas.get(action_id, {})
        next_state = dict(self.state)
        for key, delta in deltas.items():
            current = next_state.get(key)
            if isinstance(current, bool):
                continue
            if current is None:
                current = 0.0
            next_state[key] = self._bounded_metric(key, float(current) + float(delta))
        self.t += 1
        self.state = next_state
        reward = self._reward(next_state)
        done = self.t >= self.horizon
        info = {
            "action_id": action_id,
            "transition_source": "observed" if self.transition_model.n_transitions.get(action_id, 0) else "prior",
        }
        return dict(next_state), reward, done, info

    def _bounded_metric(self, key: str, value: float) -> float:
        if key.endswith("_1_to_10"):
            return round(clamp(value, 1.0, 10.0), 3)
        if key.endswith("_score") or key in {"readiness_score", "sleep_score"}:
            return round(clamp(value, 0.0, 100.0), 3)
        if key in {"recovery_pressure", "nutrition_confidence"}:
            return round(clamp(value), 3)
        if key.endswith("_minutes") or key.endswith("_g"):
            return round(max(0.0, value), 3)
        return round(value, 3)

    def _reward(self, state: dict[str, Any]) -> float:
        score = 0.50
        for key, weight in REWARD_WEIGHTS.items():
            value = state.get(key)
            if value is None:
                continue
            normalized = float(value)
            if key.endswith("_1_to_10"):
                normalized /= 10.0
            elif key.endswith("_score"):
                normalized /= 100.0
            elif key == "sleep_debt_minutes":
                normalized = min(1.0, normalized / 180.0)
            score += weight * normalized
        return round(clamp(score), 3)
