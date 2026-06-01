"""Small ML utilities used by local personalization modules."""

from __future__ import annotations

from math import exp, sqrt
from typing import Any


CORE_FEATURE_NAMES = [
    "sleep_debt_minutes",
    "readiness_score",
    "hrv_delta_percent",
    "resting_hr_delta_bpm",
    "training_load_score",
    "protein_target_gap_g",
    "nutrition_confidence",
    "energy_1_to_10",
    "focus_1_to_10",
    "stress_1_to_10",
    "digestion_1_to_10",
    "soreness_1_to_10",
    "recovery_pressure",
    "traveling",
]


FEATURE_DEFAULTS = {
    "sleep_debt_minutes": 45.0,
    "readiness_score": 75.0,
    "hrv_delta_percent": 0.0,
    "resting_hr_delta_bpm": 0.0,
    "training_load_score": 70.0,
    "protein_target_gap_g": 40.0,
    "nutrition_confidence": 0.5,
    "energy_1_to_10": 6.0,
    "focus_1_to_10": 6.0,
    "stress_1_to_10": 4.0,
    "digestion_1_to_10": 7.0,
    "soreness_1_to_10": 4.0,
    "recovery_pressure": 0.35,
    "traveling": 0.0,
}


FEATURE_SCALES = {
    "sleep_debt_minutes": 180.0,
    "readiness_score": 100.0,
    "hrv_delta_percent": 30.0,
    "resting_hr_delta_bpm": 8.0,
    "training_load_score": 180.0,
    "protein_target_gap_g": 100.0,
    "nutrition_confidence": 1.0,
    "energy_1_to_10": 10.0,
    "focus_1_to_10": 10.0,
    "stress_1_to_10": 10.0,
    "digestion_1_to_10": 10.0,
    "soreness_1_to_10": 10.0,
    "recovery_pressure": 1.0,
    "traveling": 1.0,
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def vector_norm(values: list[float]) -> float:
    return sqrt(sum(value * value for value in values))


def normalized_feature_vector(
    feature_bundle: dict[str, Any],
    feature_names: list[str] | None = None,
) -> list[float]:
    names = feature_names or CORE_FEATURE_NAMES
    current = feature_bundle.get("current", feature_bundle)
    vector = []
    for name in names:
        raw = current.get(name, FEATURE_DEFAULTS.get(name, 0.0))
        if isinstance(raw, bool):
            raw_value = 1.0 if raw else 0.0
        elif raw is None:
            raw_value = FEATURE_DEFAULTS.get(name, 0.0)
        else:
            raw_value = float(raw)
        scale = FEATURE_SCALES.get(name, 1.0) or 1.0
        vector.append(raw_value / scale)
    return vector


def outcome_reward(event: dict[str, Any]) -> float | None:
    if event.get("reward") is not None:
        return clamp(float(event["reward"]))

    outcomes = event.get("outcomes") or {}
    terms = []
    for key in ("energy_next_day_1_to_10", "mood_next_day_1_to_10", "digestion_next_day_1_to_10"):
        if outcomes.get(key) is not None:
            terms.append(clamp(float(outcomes[key]) / 10.0))
    if outcomes.get("readiness_next_day") is not None:
        readiness = float(outcomes["readiness_next_day"])
        terms.append(clamp(readiness / 100.0 if readiness > 1.0 else readiness))
    if event.get("adhered") is True:
        terms.append(0.65)
    elif event.get("adhered") is False:
        terms.append(0.25)
    return round(sum(terms) / len(terms), 3) if terms else None
