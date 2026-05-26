"""Longitudinal feature materialization for the health intelligence service.

The goal is not to create a big offline feature platform yet. This module gives
the app a stable, local feature bundle that downstream state encoders, causal
estimators, and policy learners can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any

from .models import jsonable, now_iso


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


def _std(values: list[float]) -> float | None:
    return round(pstdev(values), 3) if len(values) > 1 else 0.0 if values else None


def _trend(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return round((values[-1] - values[0]) / max(len(values) - 1, 1), 3)


@dataclass
class DailyFeatureVector:
    date: str
    metrics: dict[str, float | bool | None]
    source_coverage: dict[str, float]


@dataclass
class LongitudinalFeatureBundle:
    user_id_hash: str
    as_of_date: str
    lookback_days: int
    current: dict[str, float | bool | None]
    baselines: dict[str, dict[str, float | None]]
    deltas_from_baseline: dict[str, float | None]
    trends: dict[str, float | None]
    data_quality: dict[str, Any]
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


def extract_daily_features(daily_state: dict[str, Any]) -> DailyFeatureVector:
    wearable = daily_state.get("wearable", {})
    sleep = wearable.get("sleep", {})
    readiness = wearable.get("readiness", {})
    activity = wearable.get("activity", {})
    subjective = daily_state.get("subjective", {})
    nutrition = daily_state.get("nutrition_summary", {})
    exercise = daily_state.get("exercise_summary", {})
    environment = daily_state.get("environment", {})

    sleep_minutes = _num(sleep.get("duration_minutes"))
    protein_g = _num(nutrition.get("protein_g"))
    training_load = _num(exercise.get("training_load_score"))
    readiness_score = _num(readiness.get("readiness_score"))
    sleep_score = _num(sleep.get("sleep_score"))
    stress = _num(subjective.get("stress_1_to_10"))
    soreness = _num(subjective.get("soreness_1_to_10"))
    energy = _num(subjective.get("energy_1_to_10"))
    rhr_delta = _num(readiness.get("resting_heart_rate_delta_from_baseline_bpm"))
    hrv_delta = _num(readiness.get("hrv_delta_from_baseline_percent"))

    sleep_debt_minutes = None
    if sleep_minutes is not None:
        sleep_debt_minutes = max(0.0, 480.0 - sleep_minutes)

    recovery_pressure = None
    recovery_terms = []
    if sleep_debt_minutes is not None:
        recovery_terms.append(min(1.0, sleep_debt_minutes / 180.0))
    if rhr_delta is not None:
        recovery_terms.append(min(1.0, max(0.0, rhr_delta) / 8.0))
    if hrv_delta is not None:
        recovery_terms.append(min(1.0, max(0.0, -hrv_delta) / 25.0))
    if stress is not None:
        recovery_terms.append(stress / 10.0)
    if soreness is not None:
        recovery_terms.append(soreness / 10.0)
    if recovery_terms:
        recovery_pressure = round(sum(recovery_terms) / len(recovery_terms), 3)

    protein_target_gap_g = None
    if protein_g is not None:
        protein_target_gap_g = round(max(0.0, 130.0 - protein_g), 1)

    metrics: dict[str, float | bool | None] = {
        "sleep_duration_minutes": sleep_minutes,
        "sleep_score": sleep_score,
        "sleep_debt_minutes": sleep_debt_minutes,
        "readiness_score": readiness_score,
        "hrv_rmssd_ms": _num(readiness.get("hrv_rmssd_ms")),
        "hrv_delta_percent": hrv_delta,
        "resting_hr_bpm": _num(readiness.get("resting_heart_rate_bpm")),
        "resting_hr_delta_bpm": rhr_delta,
        "steps": _num(activity.get("steps")),
        "calories_burned": _num(activity.get("calories_burned")),
        "active_calories": _num(activity.get("active_calories")),
        "training_load_score": training_load,
        "protein_g": protein_g,
        "protein_target_gap_g": protein_target_gap_g,
        "nutrition_confidence": _num(nutrition.get("confidence")),
        "energy_1_to_10": energy,
        "mood_1_to_10": _num(subjective.get("mood_1_to_10")),
        "focus_1_to_10": _num(subjective.get("focus_1_to_10")),
        "stress_1_to_10": stress,
        "digestion_1_to_10": _num(subjective.get("digestion_1_to_10")),
        "soreness_1_to_10": soreness,
        "recovery_pressure": recovery_pressure,
        "traveling": bool(environment.get("traveling", False)),
    }

    source_coverage = {
        "wearable": 1.0 if wearable else 0.0,
        "sleep": 1.0 if sleep_minutes is not None else 0.0,
        "readiness": 1.0 if readiness_score is not None else 0.0,
        "activity": 1.0 if activity else 0.0,
        "nutrition": _num(nutrition.get("confidence")) or 0.0,
        "exercise": 1.0 if training_load is not None else 0.0,
        "subjective": round(
            sum(1 for key in subjective if key.endswith("_1_to_10")) / 6.0,
            2,
        ),
    }
    return DailyFeatureVector(
        date=str(daily_state.get("date")),
        metrics=metrics,
        source_coverage=source_coverage,
    )


def materialize_longitudinal_features(
    user_id_hash: str,
    daily_states: list[dict[str, Any]],
    lookback_days: int = 28,
) -> LongitudinalFeatureBundle:
    if not daily_states:
        raise ValueError("daily_states must contain at least one daily health state")

    ordered = sorted(daily_states, key=lambda item: str(item.get("date")))
    window = ordered[-lookback_days:]
    vectors = [extract_daily_features(item) for item in window]
    current = vectors[-1]

    numeric_keys = sorted(
        key
        for key, value in current.metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    baselines: dict[str, dict[str, float | None]] = {}
    deltas: dict[str, float | None] = {}
    trends: dict[str, float | None] = {}
    for key in numeric_keys:
        values = [
            float(vector.metrics[key])
            for vector in vectors
            if isinstance(vector.metrics.get(key), (int, float))
            and not isinstance(vector.metrics.get(key), bool)
        ]
        baseline_mean = _mean(values[:-1] or values)
        baselines[key] = {
            "mean": baseline_mean,
            "std": _std(values[:-1] or values),
            "n": float(len(values)),
        }
        current_value = current.metrics.get(key)
        if current_value is None or baseline_mean is None:
            deltas[key] = None
        else:
            deltas[key] = round(float(current_value) - baseline_mean, 3)
        trends[key] = _trend(values)

    avg_coverage = {
        key: round(
            sum(vector.source_coverage.get(key, 0.0) for vector in vectors) / len(vectors),
            3,
        )
        for key in sorted({key for vector in vectors for key in vector.source_coverage})
    }
    missing_current = [
        key for key, value in current.metrics.items() if value is None
    ]

    return LongitudinalFeatureBundle(
        user_id_hash=user_id_hash,
        as_of_date=current.date,
        lookback_days=len(vectors),
        current=current.metrics,
        baselines=baselines,
        deltas_from_baseline=deltas,
        trends=trends,
        data_quality={
            "average_source_coverage": avg_coverage,
            "missing_current_metrics": missing_current,
            "history_days_available": len(vectors),
        },
    )
