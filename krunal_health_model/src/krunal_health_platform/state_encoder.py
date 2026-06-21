"""Reference Bloom-style state encoder implementation."""

from __future__ import annotations

import time
from typing import Protocol

from .models import ScoreWithUncertainty, StateEncoderRequest, StateEncoderResponse, now_iso


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class StateEncoder(Protocol):
    def encode(self, request: StateEncoderRequest) -> StateEncoderResponse:
        """Encode multimodal user state into uncertainty-aware scores."""


class ReferenceStateEncoder:
    """A fast v0 ensemble-style encoder behind the stable API contract."""

    version = "reference_v0_bloom_scaffold"

    def encode(self, request: StateEncoderRequest) -> StateEncoderResponse:
        start = time.perf_counter()

        coverage = request.data_quality_flags.get("per_modality_coverage", {})
        sleep_episodes = request.modalities.get("sleep_episodes", [])
        sleep = sleep_episodes[0] if sleep_episodes else {}
        hr_series = request.modalities.get("wearable_hr_series", {})
        hrv_series = request.modalities.get("wearable_hrv_series", {})
        self_report_events = {
            event["name"]: event["value"]
            for event in request.modalities.get("self_report_events", [])
        }
        workouts = request.modalities.get("workout_sessions", [])

        sleep_minutes = _safe_float(sleep.get("duration_minutes"))
        sleep_score = _safe_float(sleep.get("sleep_score")) / 100.0
        hrv_ms = _safe_float(hrv_series.get("hrv_rmssd_ms"))
        hrv_delta = _safe_float(hrv_series.get("hrv_delta_from_baseline_percent"))
        rhr_delta = _safe_float(hr_series.get("resting_heart_rate_delta_from_baseline_bpm"))
        energy = _safe_float(self_report_events.get("energy_1_to_10"), 5.0) / 10.0
        stress = _safe_float(self_report_events.get("stress_1_to_10"), 5.0) / 10.0
        soreness = _safe_float(self_report_events.get("soreness_1_to_10"), 5.0) / 10.0
        focus = _safe_float(self_report_events.get("focus_1_to_10"), 5.0) / 10.0
        digestion = _safe_float(self_report_events.get("digestion_1_to_10"), 5.0) / 10.0

        workout_minutes = 0.0
        for session in workouts:
            duration = _safe_float(session.get("duration_minutes"))
            if not duration and session.get("activity") == "rowing":
                duration = _safe_float(session.get("distance_miles")) * 10.0
            workout_minutes += duration

        sleep_debt = _clamp((480.0 - sleep_minutes) / 480.0 if sleep_minutes else 0.35)
        normalized_hrv = _clamp(hrv_ms / 100.0 if hrv_ms else 0.4)
        normalized_rhr_load = _clamp(max(rhr_delta, 0.0) / 8.0)
        training_load = _clamp(workout_minutes / 120.0)
        readiness = _clamp(
            0.42 * sleep_score
            + 0.20 * normalized_hrv
            + 0.18 * energy
            + 0.10 * (1.0 - soreness)
            + 0.10 * (1.0 - normalized_rhr_load)
        )
        fatigue = _clamp(
            0.36 * sleep_debt
            + 0.18 * training_load
            + 0.16 * soreness
            + 0.15 * (1.0 - energy)
            + 0.15 * normalized_rhr_load
        )
        stress_score = _clamp(0.72 * stress + 0.28 * normalized_rhr_load)

        travel_context = request.context.get("location_history", {}).get("traveling", False)
        ood_score = 0.42 if travel_context else 0.12
        if request.context.get("weather", {}).get("air_quality_context") == "low_air_quality":
            ood_score = max(ood_score, 0.35)

        avg_support = (
            coverage.get("sleep_episodes", 0.0)
            + coverage.get("wearable_hr_series", 0.0)
            + coverage.get("wearable_hrv_series", 0.0)
            + coverage.get("self_report_events", 0.0)
        ) / 4.0
        abstain = avg_support < 0.2
        abstain_reasons = []
        if abstain:
            abstain_reasons.append("insufficient_multimodal_coverage")
        if coverage.get("wearable_hrv_series", 0.0) < 0.5:
            abstain_reasons.append("low_hrv_support")

        def scored(mean: float, support_keys: list[str]) -> ScoreWithUncertainty:
            support = sum(coverage.get(key, 0.0) for key in support_keys) / max(len(support_keys), 1)
            epistemic = 0.08 + (1.0 - support) * 0.20 + ood_score * 0.12
            aleatoric = 0.05 + (0.10 if support < 0.6 else 0.04)
            width = epistemic + aleatoric
            calibration_subgroup = "low_coverage" if support < 0.7 else "high_coverage"
            return ScoreWithUncertainty(
                mean=round(mean, 3),
                ci_low=round(_clamp(mean - width), 3),
                ci_high=round(_clamp(mean + width), 3),
                method="ensemble",
                calibration_subgroup=calibration_subgroup,
                epistemic_uncertainty=round(epistemic, 3),
                aleatoric_uncertainty=round(aleatoric, 3),
            )

        adherence_likelihood = {
            "strength_progression": scored(_clamp(0.55 * readiness + 0.25 * focus + 0.20 * (1.0 - fatigue)), ["sleep_episodes", "wearable_hrv_series", "self_report_events"]),
            "recovery_row": scored(_clamp(0.45 + 0.35 * fatigue + 0.20 * digestion), ["sleep_episodes", "wearable_hr_series", "self_report_events"]),
            "bedtime_protection": scored(_clamp(0.50 + 0.30 * sleep_debt + 0.20 * stress_score), ["sleep_episodes", "self_report_events"]),
            "protein_portion_check": scored(_clamp(0.48 + 0.25 * focus + 0.27 * energy), ["self_report_events"]),
        }

        short_term_embedding = [
            round(value, 3)
            for value in [
                readiness,
                fatigue,
                stress_score,
                sleep_debt,
                training_load,
                digestion,
                energy,
                focus,
            ]
        ]
        long_term_embedding = [
            round(value, 3)
            for value in [
                (sleep_score + readiness) / 2.0,
                (energy + focus) / 2.0,
                (1.0 - stress_score + digestion) / 2.0,
                1.0 - ood_score,
                coverage.get("self_report_events", 0.0),
                coverage.get("nutrition", 0.0),
                coverage.get("workout_sessions", 0.0),
                0.0 if travel_context else 1.0,
            ]
        ]

        state_scores = {
            "fatigue": scored(fatigue, ["sleep_episodes", "wearable_hrv_series", "self_report_events"]),
            "readiness": scored(readiness, ["sleep_episodes", "wearable_hr_series", "wearable_hrv_series"]),
            "stress": scored(stress_score, ["wearable_hr_series", "self_report_events"]),
            "sleep_debt": scored(sleep_debt, ["sleep_episodes"]),
            "adherence_likelihood": adherence_likelihood,
        }

        return StateEncoderResponse(
            short_term_embedding=short_term_embedding,
            long_term_embedding=long_term_embedding,
            state_scores=state_scores,
            modality_support={key: round(float(value), 2) for key, value in coverage.items()},
            ood_score=round(ood_score, 3),
            abstain=abstain,
            abstain_reasons=abstain_reasons,
            encoder_version=self.version,
            computed_at=now_iso(),
            computation_cost_ms=int((time.perf_counter() - start) * 1000),
        )

