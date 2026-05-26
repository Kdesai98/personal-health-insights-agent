"""Translate the existing daily state into the Bloom-style state encoder contract."""

from __future__ import annotations

from typing import Any

from .models import StateEncoderRequest


def _subjective_event(name: str, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"name": name, "value": value, "source": "self_report"}


def daily_state_to_state_encoder_request(
    daily_state: dict[str, Any],
    user_id_hash: str = "krunal_demo_user",
) -> StateEncoderRequest:
    wearable = daily_state.get("wearable", {})
    sleep = wearable.get("sleep", {})
    readiness = wearable.get("readiness", {})
    activity = wearable.get("activity", {})
    subjective = daily_state.get("subjective", {})
    manual_logs = daily_state.get("manual_logs", {})

    self_report_events = [
        _subjective_event("energy_1_to_10", subjective.get("energy_1_to_10")),
        _subjective_event("mood_1_to_10", subjective.get("mood_1_to_10")),
        _subjective_event("focus_1_to_10", subjective.get("focus_1_to_10")),
        _subjective_event("stress_1_to_10", subjective.get("stress_1_to_10")),
        _subjective_event("digestion_1_to_10", subjective.get("digestion_1_to_10")),
        _subjective_event("soreness_1_to_10", subjective.get("soreness_1_to_10")),
    ]
    self_report_events = [event for event in self_report_events if event is not None]

    food_logs = manual_logs.get("food", [])
    exercise_logs = manual_logs.get("exercise", [])
    food_confidences = [log.get("confidence") for log in food_logs if log.get("confidence") is not None]

    per_modality_coverage = {
        "sleep_episodes": 1.0 if sleep.get("duration_minutes") is not None else 0.0,
        "wearable_hr_series": 1.0 if readiness.get("resting_heart_rate_bpm") is not None else 0.0,
        "wearable_hrv_series": 1.0 if readiness.get("hrv_rmssd_ms") is not None else 0.0,
        "workout_sessions": min(1.0, len(exercise_logs) / 2.0),
        "self_report_events": round(len(self_report_events) / 6.0, 2),
        "journal_text": 1.0 if subjective.get("notes") else 0.0,
        "voice_notes": min(1.0, (len(food_logs) + len(exercise_logs)) / 4.0),
        "meal_photos": 0.0,
        "nutrition": round(sum(food_confidences) / len(food_confidences), 2) if food_confidences else 0.0,
    }

    anomalies = []
    if readiness.get("hrv_rmssd_ms") is None:
        anomalies.append("missing_hrv")
    if not exercise_logs:
        anomalies.append("missing_manual_exercise")
    if not food_logs:
        anomalies.append("missing_food_logs")

    return StateEncoderRequest(
        user_id_hash=user_id_hash,
        window={
            "start_ts": f"{daily_state['date']}T00:00:00-04:00",
            "end_ts": f"{daily_state['date']}T23:59:59-04:00",
        },
        modalities={
            "wearable_hr_series": {
                "resting_heart_rate_bpm": readiness.get("resting_heart_rate_bpm"),
                "resting_heart_rate_delta_from_baseline_bpm": readiness.get(
                    "resting_heart_rate_delta_from_baseline_bpm"
                ),
                "steps": activity.get("steps"),
            },
            "wearable_hrv_series": {
                "hrv_rmssd_ms": readiness.get("hrv_rmssd_ms"),
                "hrv_delta_from_baseline_percent": readiness.get("hrv_delta_from_baseline_percent"),
            },
            "sleep_episodes": [
                {
                    "duration_minutes": sleep.get("duration_minutes"),
                    "sleep_score": sleep.get("sleep_score"),
                    "efficiency_percent": sleep.get("efficiency_percent"),
                    "bedtime_start": sleep.get("bedtime_start"),
                    "bedtime_end": sleep.get("bedtime_end"),
                }
            ],
            "workout_sessions": [
                activity
                for log in exercise_logs
                for activity in log.get("activities", [])
            ],
            "self_report_events": self_report_events,
            "journal_text": subjective.get("notes"),
            "meal_photos": [],
            "voice_notes": [
                {"kind": "food", "text": log.get("raw_text")} for log in food_logs
            ]
            + [
                {"kind": "exercise", "text": log.get("raw_text")} for log in exercise_logs
            ],
        },
        context={
            "calendar_load": {"load": "unknown"},
            "location_history": {
                "traveling": daily_state.get("environment", {}).get("traveling", False),
                "location_context": daily_state.get("environment", {}).get("location_context"),
            },
            "weather": {"air_quality_context": daily_state.get("environment", {}).get("air_quality_context")},
            "social_graph_summary": {"neighbors": 0, "opt_in": False},
        },
        data_quality_flags={
            "per_modality_coverage": per_modality_coverage,
            "detected_anomalies": anomalies,
            "firmware_versions": {"oura": "unknown"},
        },
        encoder_version_pin="reference_v0_bloom_scaffold",
    )

