"""App-facing orchestration over the platform layers."""

from __future__ import annotations

from typing import Any

from .causal import CausalStudyRegistry, ReferenceCausalEstimator
from .models import (
    AppPlan,
    InterventionTrace,
    PurposeOfUse,
    SafetyStatus,
    StateEncoderResponse,
    UserBurdenBudget,
    generate_id,
    jsonable,
    now_iso,
)
from .policy import PolicyPlanner, build_default_candidates
from .safety import SafetyCascade, SimpleEvidenceVerifier
from .state_encoder import ReferenceStateEncoder


DEFAULT_EVIDENCE_BASE = {
    "GUIDE_RECOVERY_001": "Recovery-biased days are preferable when fatigue and sleep debt are elevated.",
    "GUIDE_STRENGTH_001": "Progressive strength work is most useful when readiness is strong and soreness is manageable.",
    "GUIDE_ZONE2_001": "Easy aerobic work can preserve consistency without overreaching.",
    "GUIDE_SLEEP_001": "Stable bedtime protects next-day readiness and long-term adherence.",
    "GUIDE_PROTEIN_001": "Protein distribution and sufficient intake matter for muscle gain, especially on plant-forward diets.",
    "GUIDE_VOI_001": "When uncertainty is high, ask the shortest question that changes the downstream decision.",
}


class DataAnalystAgent:
    def summarize(self, daily_state: dict[str, Any], state: StateEncoderResponse) -> list[str]:
        summary = [
            f"Readiness mean {state.state_scores['readiness'].mean:.2f}",
            f"Fatigue mean {state.state_scores['fatigue'].mean:.2f}",
            f"OOD score {state.ood_score:.2f}",
        ]
        protein = daily_state.get("nutrition_summary", {}).get("protein_g")
        if protein is not None:
            summary.append(f"Estimated protein {protein:.0f}g")
        return summary


class DomainKnowledgeAgent:
    def __init__(self, evidence_base: dict[str, str]) -> None:
        self.evidence_base = evidence_base

    def retrieve(self, evidence_refs: list[str]) -> dict[str, str]:
        return {ref: self.evidence_base[ref] for ref in evidence_refs if ref in self.evidence_base}


class CoachAgent:
    def draft(
        self,
        daily_state: dict[str, Any],
        state: StateEncoderResponse,
        decision_type: str,
        selected_candidate: dict[str, Any] | None,
        evidence: dict[str, str],
    ) -> tuple[str, str, str, str, str, str, str]:
        readiness = state.state_scores["readiness"]
        fatigue = state.state_scores["fatigue"]
        sleep_debt = state.state_scores["sleep_debt"]
        protein = daily_state.get("nutrition_summary", {}).get("protein_g")

        day_type = "maintenance-biased"
        if fatigue.mean > readiness.mean + 0.10:
            day_type = "recovery-biased"
        elif readiness.mean > fatigue.mean + 0.10:
            day_type = "training-biased"

        headline = "Act quietly, but with evidence." if decision_type == "act" else "Ask before acting."
        summary = (
            f"Readiness is {readiness.mean:.2f} with CI [{readiness.ci_low:.2f}, {readiness.ci_high:.2f}], "
            f"fatigue is {fatigue.mean:.2f}, and sleep debt is {sleep_debt.mean:.2f}."
        )

        if selected_candidate and selected_candidate["action_id"] == "strength_progression":
            workout = "Use today for strength progression and keep any rowing easy enough that it does not dilute the lift."
        elif selected_candidate and selected_candidate["action_id"] == "recovery_row":
            workout = "Bias the day toward easy rowing, walking, and mobility. Preserve consistency without chasing intensity."
        else:
            workout = "Keep training moderate unless new information raises or lowers confidence."

        if protein is None:
            nutrition = "Protein intake is still uncertain. Log the next meal precisely enough that the system can estimate protein."
        elif protein < 110:
            nutrition = f"Current logged protein is about {protein:.0f}g. Bias the next meal toward dense vegan-vegetarian protein."
        else:
            nutrition = f"Logged protein is about {protein:.0f}g. Keep distribution even across meals so the total is actually usable."

        sleep = "Protect bedtime tonight. The system treats sleep regularity as a first-class lever, not an afterthought."
        focus = "Use your strongest-focus window for demanding work and keep interruptions sparse unless the expected value is high."
        question = selected_candidate.get("question_text") if selected_candidate else None
        return day_type, headline, summary, workout, nutrition, sleep, focus, question


class HealthPlatformOrchestrator:
    def __init__(self) -> None:
        self.state_encoder = ReferenceStateEncoder()
        self.registry = CausalStudyRegistry()
        self.causal_estimator = ReferenceCausalEstimator()
        self.safety = SafetyCascade()
        self.verifier = SimpleEvidenceVerifier()
        self.analyst = DataAnalystAgent()
        self.knowledge = DomainKnowledgeAgent(DEFAULT_EVIDENCE_BASE)
        self.coach = CoachAgent()
        self.policy = PolicyPlanner()

    def run_daily_plan(
        self,
        user_id_hash: str,
        daily_state: dict[str, Any],
        encoder_request: Any,
        zone_lineage: dict[str, str],
        burden_budget: UserBurdenBudget,
    ) -> AppPlan:
        state = self.state_encoder.encode(encoder_request)
        candidates = build_default_candidates(daily_state, state)
        estimates = self.causal_estimator.estimate_candidates(state, candidates, self.registry)

        subjective_notes = daily_state.get("subjective", {}).get("notes")
        safety_by_action = {
            candidate.action_id: [
                jsonable(decision)
                for decision in self.safety.evaluate_candidate(state, candidate, subjective_notes)
            ]
            for candidate in candidates
        }

        planning = self.policy.plan(
            daily_state=daily_state,
            state=state,
            estimates=estimates,
            candidates=candidates,
            safety_by_action=safety_by_action,
            budget=burden_budget,
        )

        selected_candidate = jsonable(planning.candidate) if planning.candidate else None
        evidence_refs = planning.candidate.evidence_refs if planning.candidate else ["GUIDE_VOI_001"]
        evidence = self.knowledge.retrieve(evidence_refs)
        day_type, headline, summary, workout, nutrition, sleep, focus, question = self.coach.draft(
            daily_state=daily_state,
            state=state,
            decision_type=planning.voi_decision.decision_type,
            selected_candidate=selected_candidate,
            evidence=evidence,
        )

        explanation = (
            f"{headline} {summary} "
            f"Chosen path: {selected_candidate['title'] if selected_candidate else 'wait and observe'}."
        )
        verifier_outcome = self.verifier.verify(explanation, evidence_refs, DEFAULT_EVIDENCE_BASE)

        if verifier_outcome["status"] != "pass":
            explanation = (
                "The system could not verify every explanatory claim against the evidence base, "
                "so it is defaulting to a conservative summary."
            )
            question = planning.voi_decision.clarifying_question

        candidate_posteriors = {
            estimate.action_id: jsonable(
                {
                    "study_id": estimate.study_id,
                    "population_effect": estimate.population_effect,
                    "personal_effect": estimate.personal_effect,
                    "rationale": estimate.rationale,
                }
            )
            for estimate in estimates
        }
        voi_scores = {
            item.candidate.action_id: jsonable(
                {
                    "acquisition_function": item.acquisition_function,
                    "expected_information_gain": item.expected_information_gain,
                    "expected_utility_improvement": item.expected_utility_improvement,
                    "decision_change_prob": item.downstream_decision_change_prob,
                    "combined_score": round(item.combined_score, 3),
                    "notes": item.notes,
                }
            )
            for item in planning.scored_candidates
        }

        trace = InterventionTrace(
            trace_id=generate_id("trace"),
            user_id_hash=user_id_hash,
            purpose_of_use=PurposeOfUse.CONSUMER_COACHING,
            zone_lineage=zone_lineage,
            state_encoder_version=state.encoder_version,
            candidate_posteriors=candidate_posteriors,
            voi_scores=voi_scores,
            safety_decisions=safety_by_action,
            selected_action_id=planning.voi_decision.selected_candidate_id,
            evidence_refs=evidence_refs,
            generated_explanation=explanation,
            verifier_outcome=verifier_outcome,
            created_at=now_iso(),
        )

        provenance = self.analyst.summarize(daily_state, state)
        uncertainty = []
        if state.ood_score > 0.3:
            uncertainty.append("Current context looks meaningfully out of distribution relative to the reference cohort.")
        if daily_state.get("nutrition_summary", {}).get("confidence", 1.0) < 0.8:
            uncertainty.append("Nutrition estimates are still rough enough that one clarification could change the recommendation.")
        if state.state_scores["fatigue"].width > 0.25:
            uncertainty.append("Fatigue uncertainty is still wide because modality coverage is incomplete.")

        safety_notes = [
            "No diagnosis, medication changes, or unverified supplements.",
            "Escalate to a clinician for high-risk symptoms or anything outside routine wellness coaching.",
        ]
        if any(
            decision["status"] == SafetyStatus.ESCALATE.value
            for decisions in safety_by_action.values()
            for decision in decisions
        ):
            safety_notes.append("At least one candidate required deeper safety screening before selection.")

        return AppPlan(
            date=daily_state["date"],
            day_type=day_type,
            headline=headline,
            summary=explanation,
            workout=workout,
            nutrition=nutrition,
            sleep=sleep,
            focus=focus,
            question=question,
            uncertainty=uncertainty,
            provenance=provenance,
            safety_notes=safety_notes,
            trace=jsonable(trace),
        )

