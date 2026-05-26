"""Candidate generation, VoI scoring, and policy planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import (
    CausalEstimate,
    InterventionCandidate,
    SafetyStatus,
    StateEncoderResponse,
    UserBurdenBudget,
    VoICandidate,
    VoIDecision,
)


@dataclass
class PlanningDecision:
    candidate: InterventionCandidate | None
    voi_decision: VoIDecision
    surviving_candidates: list[InterventionCandidate]
    scored_candidates: list[VoICandidate]


def build_default_candidates(
    daily_state: dict[str, Any],
    state: StateEncoderResponse,
) -> list[InterventionCandidate]:
    nutrition = daily_state.get("nutrition_summary", {})
    candidates = [
        InterventionCandidate(
            action_id="strength_progression",
            category="action",
            title="Progress strength work",
            description="Use today's stronger recovery window for progressive strength training.",
            channel="app",
            burden_cost=0.18,
            risk_cost=0.30,
            evidence_refs=["GUIDE_RECOVERY_001", "GUIDE_STRENGTH_001"],
            tags=["training", "strength", "high_intensity"],
        ),
        InterventionCandidate(
            action_id="recovery_row",
            category="action",
            title="Recovery-biased cardio",
            description="Bias the day toward easier aerobic work and mobility.",
            channel="app",
            burden_cost=0.12,
            risk_cost=0.10,
            evidence_refs=["GUIDE_RECOVERY_001", "GUIDE_ZONE2_001"],
            tags=["recovery", "training", "low_intensity"],
        ),
        InterventionCandidate(
            action_id="bedtime_protection",
            category="action",
            title="Protect bedtime",
            description="Anchor bedtime and avoid letting training or work push it later.",
            channel="push",
            burden_cost=0.10,
            risk_cost=0.04,
            evidence_refs=["GUIDE_SLEEP_001"],
            tags=["sleep"],
        ),
    ]

    if nutrition.get("confidence") is None or nutrition.get("confidence", 1.0) < 0.8:
        candidates.append(
            InterventionCandidate(
                action_id="protein_portion_check",
                category="question",
                title="Clarify protein portion",
                description="Ask one short question to reduce protein-estimate uncertainty.",
                channel="chat",
                burden_cost=0.15,
                risk_cost=0.02,
                evidence_refs=["GUIDE_PROTEIN_001", "GUIDE_VOI_001"],
                tags=["question", "nutrition"],
                question_text="Was your highest-protein meal today smaller, typical, or larger than usual?",
            )
        )

    if state.ood_score > 0.35:
        candidates.append(
            InterventionCandidate(
                action_id="travel_context_check",
                category="question",
                title="Clarify travel context",
                description="Ask one question to disambiguate out-of-distribution context.",
                channel="chat",
                burden_cost=0.12,
                risk_cost=0.02,
                evidence_refs=["GUIDE_VOI_001"],
                tags=["question", "context"],
                question_text="Is today unusual for you because of travel, time-zone shift, or an atypical schedule?",
            )
        )
    return candidates


class ValueOfInformationEngine:
    """VoI-style scorer for ask vs act vs wait."""

    def score_candidates(
        self,
        daily_state: dict[str, Any],
        state: StateEncoderResponse,
        estimates: list[CausalEstimate],
        candidates: list[InterventionCandidate],
        budget: UserBurdenBudget,
    ) -> list[VoICandidate]:
        estimate_by_action = {estimate.action_id: estimate for estimate in estimates}
        nutrition_confidence = daily_state.get("nutrition_summary", {}).get("confidence")
        sleep_debt = state.state_scores["sleep_debt"].mean
        fatigue_width = state.state_scores["fatigue"].width

        scored: list[VoICandidate] = []
        for candidate in candidates:
            estimate = estimate_by_action[candidate.action_id]
            utility = max(estimate.personal_effect.mean, 0.0)
            decision_change_prob = estimate.personal_effect.probability_above_threshold
            if candidate.category == "question":
                info_gain = 0.35 + 0.40 * fatigue_width
                if candidate.action_id == "protein_portion_check":
                    info_gain += (1.0 - (nutrition_confidence or 0.4)) * 0.45
                if candidate.action_id == "travel_context_check":
                    info_gain += state.ood_score * 0.55
                acquisition = "BALD"
            elif candidate.action_id == "bedtime_protection":
                info_gain = 0.20 + 0.35 * sleep_debt
                acquisition = "expected_improvement"
            else:
                info_gain = 0.18 + 0.20 * fatigue_width
                acquisition = "thompson_sampling"

            if budget.used_today + candidate.burden_cost > budget.daily_cap:
                burden = 1.0
                note = "Rejected by daily burden budget."
            elif candidate.burden_cost > budget.remaining_for_channel(candidate.channel):
                burden = 1.0
                note = "Rejected by channel-specific burden budget."
            else:
                burden = candidate.burden_cost
                note = "Within user burden budget."

            risk_penalty = candidate.risk_cost * (1.0 + state.ood_score)
            scored.append(
                VoICandidate(
                    candidate=candidate,
                    acquisition_function=acquisition,
                    expected_information_gain=round(info_gain, 3),
                    expected_utility_improvement=round(utility, 3),
                    downstream_decision_change_prob=round(decision_change_prob, 3),
                    burden_cost=round(burden, 3),
                    risk_penalty=round(risk_penalty, 3),
                    notes=[note, f"Posterior from {estimate.study_id}."],
                )
            )
        return scored


class PolicyPlanner:
    """Combine safety, causal posteriors, and VoI into one decision."""

    def __init__(self) -> None:
        self.voi = ValueOfInformationEngine()

    def plan(
        self,
        daily_state: dict[str, Any],
        state: StateEncoderResponse,
        estimates: list[CausalEstimate],
        candidates: list[InterventionCandidate],
        safety_by_action: dict[str, list[dict[str, Any]]],
        budget: UserBurdenBudget,
    ) -> PlanningDecision:
        safe_candidates = []
        for candidate in candidates:
            safety_trace = safety_by_action[candidate.action_id]
            final_status = safety_trace[-1]["status"]
            if final_status != SafetyStatus.BLOCK.value:
                safe_candidates.append(candidate)

        scored_candidates = self.voi.score_candidates(
            daily_state=daily_state,
            state=state,
            estimates=estimates,
            candidates=safe_candidates,
            budget=budget,
        )
        scored_candidates.sort(key=lambda item: item.combined_score, reverse=True)

        if state.abstain:
            question_candidate = next(
                (item for item in scored_candidates if item.candidate.category == "question"),
                None,
            )
            if question_candidate:
                return PlanningDecision(
                    candidate=question_candidate.candidate,
                    surviving_candidates=safe_candidates,
                    scored_candidates=scored_candidates,
                    voi_decision=VoIDecision(
                        decision_type="ask",
                        selected_candidate_id=question_candidate.candidate.action_id,
                        score=round(question_candidate.combined_score, 3),
                        rationale=["State encoder abstained; asking the highest-VoI clarifying question."],
                        clarifying_question=question_candidate.candidate.question_text,
                        abstain=True,
                    ),
                )
            return PlanningDecision(
                candidate=None,
                surviving_candidates=safe_candidates,
                scored_candidates=scored_candidates,
                voi_decision=VoIDecision(
                    decision_type="wait",
                    selected_candidate_id=None,
                    score=0.0,
                    rationale=["State encoder abstained and no safe clarifying question was available."],
                    abstain=True,
                ),
            )

        top = scored_candidates[0] if scored_candidates else None
        if top is None or top.combined_score <= 0.05:
            return PlanningDecision(
                candidate=None,
                surviving_candidates=safe_candidates,
                scored_candidates=scored_candidates,
                voi_decision=VoIDecision(
                    decision_type="wait",
                    selected_candidate_id=None,
                    score=round(top.combined_score if top else 0.0, 3),
                    rationale=["No candidate cleared the VoI threshold after burden and risk penalties."],
                ),
            )

        second = scored_candidates[1] if len(scored_candidates) > 1 else None
        if (
            top.candidate.category == "action"
            and second is not None
            and second.candidate.category == "question"
            and abs(top.combined_score - second.combined_score) < 0.08
            and state.state_scores["fatigue"].width > 0.25
        ):
            return PlanningDecision(
                candidate=second.candidate,
                surviving_candidates=safe_candidates,
                scored_candidates=scored_candidates,
                voi_decision=VoIDecision(
                    decision_type="ask",
                    selected_candidate_id=second.candidate.action_id,
                    score=round(second.combined_score, 3),
                    rationale=["Top action and clarifying question were close; wide uncertainty favors asking first."],
                    clarifying_question=second.candidate.question_text,
                ),
            )

        decision_type = "ask" if top.candidate.category == "question" else "act"
        return PlanningDecision(
            candidate=top.candidate,
            surviving_candidates=safe_candidates,
            scored_candidates=scored_candidates,
            voi_decision=VoIDecision(
                decision_type=decision_type,
                selected_candidate_id=top.candidate.action_id,
                score=round(top.combined_score, 3),
                rationale=top.notes + [f"Acquisition function: {top.acquisition_function}"],
                clarifying_question=top.candidate.question_text,
            ),
        )

