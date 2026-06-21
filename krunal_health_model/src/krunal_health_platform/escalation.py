"""Clinical escalation workflow for safety-sensitive user states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import jsonable, now_iso


EMERGENCY_TERMS = [
    "chest pain",
    "fainted",
    "fainting",
    "can't breathe",
    "cannot breathe",
    "severe shortness of breath",
    "stroke",
    "suicidal",
    "kill myself",
    "self harm",
    "blood in stool",
    "black stool",
]


CLINICIAN_TERMS = [
    "heart palpitations",
    "dizzy",
    "dizziness",
    "persistent pain",
    "injury",
    "panic attack",
    "depressed",
    "insomnia",
]


@dataclass
class EscalationDecision:
    level: str
    reasons: list[str]
    user_message: str
    recommended_destination: str
    evaluated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class ClinicalEscalationWorkflow:
    """Separates routine wellness coaching from clinician-level issues."""

    def evaluate(
        self,
        daily_state: dict[str, Any] | None = None,
        free_text: str | None = None,
    ) -> EscalationDecision:
        state = daily_state or {}
        subjective = state.get("subjective", {})
        notes = " ".join(
            str(item)
            for item in [
                free_text,
                subjective.get("notes"),
            ]
            if item
        ).lower()

        emergency_hits = [term for term in EMERGENCY_TERMS if term in notes]
        if emergency_hits:
            return EscalationDecision(
                level="emergency",
                reasons=[f"matched emergency term: {term}" for term in emergency_hits],
                user_message="This is outside wellness coaching. Seek urgent medical help now.",
                recommended_destination="emergency_services_or_local_crisis_line",
            )

        clinician_hits = [term for term in CLINICIAN_TERMS if term in notes]
        numeric_reasons = self._numeric_reasons(state)
        if clinician_hits or numeric_reasons:
            return EscalationDecision(
                level="clinician_discussion",
                reasons=[f"matched clinician term: {term}" for term in clinician_hits] + numeric_reasons,
                user_message="This should be discussed with a qualified clinician before changing training, supplements, or treatment.",
                recommended_destination="primary_care_or_relevant_specialist",
            )

        return EscalationDecision(
            level="routine_wellness",
            reasons=["no escalation triggers detected"],
            user_message="Routine wellness coaching is allowed within the app boundaries.",
            recommended_destination="app_coaching",
        )

    def _numeric_reasons(self, state: dict[str, Any]) -> list[str]:
        reasons = []
        readiness = state.get("wearable", {}).get("readiness", {})
        subjective = state.get("subjective", {})
        rhr_delta = readiness.get("resting_heart_rate_delta_from_baseline_bpm")
        hrv_delta = readiness.get("hrv_delta_from_baseline_percent")
        soreness = subjective.get("soreness_1_to_10")
        stress = subjective.get("stress_1_to_10")
        if isinstance(rhr_delta, (int, float)) and rhr_delta >= 10:
            reasons.append("resting heart rate materially above baseline")
        if isinstance(hrv_delta, (int, float)) and hrv_delta <= -35:
            reasons.append("HRV materially below baseline")
        if isinstance(soreness, (int, float)) and soreness >= 9:
            reasons.append("very high soreness or pain report")
        if isinstance(stress, (int, float)) and stress >= 9:
            reasons.append("very high stress report")
        return reasons
