"""Safety cascade and lightweight evidence verification."""

from __future__ import annotations

from typing import Any

from .models import InterventionCandidate, SafetyDecision, SafetyStatus, StateEncoderResponse


HIGH_RISK_TOKENS = [
    "chest pain",
    "suicidal",
    "self harm",
    "panic attack",
    "fainted",
]


class SafetyCascade:
    """Cheap rules -> cheap screening -> heavy precision check."""

    def evaluate_candidate(
        self,
        state: StateEncoderResponse,
        candidate: InterventionCandidate,
        subjective_notes: str | None,
    ) -> list[SafetyDecision]:
        notes = (subjective_notes or "").lower()
        results: list[SafetyDecision] = []

        if any(token in notes for token in HIGH_RISK_TOKENS):
            return [
                SafetyDecision(
                    stage="stage_0_rules",
                    status=SafetyStatus.BLOCK,
                    reasons=["High-risk symptom language detected. Redirect to clinician or emergency pathway."],
                )
            ]

        fatigue = state.state_scores["fatigue"].mean
        readiness = state.state_scores["readiness"].mean
        sleep_debt = state.state_scores["sleep_debt"].mean

        stage0_reasons = []
        if "high_intensity" in candidate.tags and fatigue > 0.70:
            stage0_reasons.append("High-intensity recommendation conflicts with elevated fatigue.")
        if "training" in candidate.tags and sleep_debt > 0.35 and readiness < 0.55:
            stage0_reasons.append("Training recommendation conflicts with high sleep debt and low readiness.")

        results.append(
            SafetyDecision(
                stage="stage_0_rules",
                status=SafetyStatus.BLOCK if stage0_reasons else SafetyStatus.PASS,
                reasons=stage0_reasons or ["Passed rule-based safety checks."],
            )
        )
        if stage0_reasons:
            return results

        risk_score = candidate.risk_cost + 0.40 * fatigue + 0.20 * sleep_debt + 0.15 * state.ood_score
        results.append(
            SafetyDecision(
                stage="stage_1_screen",
                status=SafetyStatus.ESCALATE if risk_score > 0.65 else SafetyStatus.PASS,
                reasons=["Escalated to precision check due to aggregate risk score."] if risk_score > 0.65 else ["Passed cheap screening model."],
                score=round(risk_score, 3),
            )
        )

        precision_reasons = []
        if "strength" in candidate.tags and readiness < 0.50:
            precision_reasons.append("Precision check rejected strength-focused action under low readiness.")
        if candidate.category == "question" and candidate.burden_cost > 0.6:
            precision_reasons.append("Question is too burdensome for routine interruption.")
        results.append(
            SafetyDecision(
                stage="stage_2_precision",
                status=SafetyStatus.BLOCK if precision_reasons else SafetyStatus.PASS,
                reasons=precision_reasons or ["Passed precision check."],
            )
        )
        return results


class SimpleEvidenceVerifier:
    """Heuristic stand-in for the NLI verifier described in the PDF."""

    def verify(
        self,
        explanation: str,
        evidence_refs: list[str],
        evidence_base: dict[str, str],
    ) -> dict[str, Any]:
        missing = [ref for ref in evidence_refs if ref not in evidence_base]
        if missing:
            return {
                "status": "fail",
                "reasons": [f"Missing evidence refs: {', '.join(missing)}"],
            }
        if not explanation.strip():
            return {"status": "fail", "reasons": ["Explanation draft was empty."]}
        return {
            "status": "pass",
            "reasons": ["Evidence refs resolved in the vetted evidence base."],
        }

