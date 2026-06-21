"""Experiment registry and reference causal estimator."""

from __future__ import annotations

from typing import Any

from .models import (
    CausalEstimate,
    CausalStudy,
    Experiment,
    InterventionCandidate,
    PosteriorDistribution,
    StateEncoderResponse,
    generate_id,
    now_iso,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _posterior(mean: float, width: float, identification_uncertainty: float, notes: list[str]) -> PosteriorDistribution:
    threshold = 0.08
    probability = _clamp(0.5 + ((mean - threshold) / max(width * 2.0, 0.12)))
    return PosteriorDistribution(
        mean=round(mean, 3),
        ci_low=round(mean - width, 3),
        ci_high=round(mean + width, 3),
        probability_above_threshold=round(probability, 3),
        method="bayesian_reference",
        identification_uncertainty=round(identification_uncertainty, 3),
        notes=notes,
    )


class CausalStudyRegistry:
    """Pre-registered study registry mirroring the PDF's discipline."""

    def __init__(self) -> None:
        self.experiments: dict[str, Experiment] = {}
        self.studies: dict[str, CausalStudy] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        default_specs = [
            {
                "name": "training_load_adjustment",
                "tags": ["training", "recovery"],
                "hypothesis": "Personalized training-load adjustments improve weekly adherence and readiness.",
                "design_type": "RCT",
                "estimator": "DML",
                "identification_strategy": "randomized_feature_flag_with_long_term_holdout",
            },
            {
                "name": "protein_portion_prompt",
                "tags": ["nutrition", "question"],
                "hypothesis": "Low-burden protein clarifications improve nutrition estimate accuracy and adherence.",
                "design_type": "RCT",
                "estimator": "AIPW",
                "identification_strategy": "micro-randomized_question_prompt",
            },
            {
                "name": "bedtime_protection_nudge",
                "tags": ["sleep"],
                "hypothesis": "Protecting bedtime after higher training load improves next-day readiness.",
                "design_type": "ITS",
                "estimator": "mixed_effects",
                "identification_strategy": "interrupted_time_series_with_per_user_baselines",
            },
        ]

        for spec in default_specs:
            experiment_id = generate_id("experiment")
            study_id = generate_id("study")
            experiment = Experiment(
                experiment_id=experiment_id,
                owner="krunal_health_platform",
                hypothesis=spec["hypothesis"],
                design_type=spec["design_type"],
                unit_type="user",
                randomization={"method": "hash", "salt": experiment_id},
                eligibility_filter="user_has_required_modality_support == true",
                treatment_arms=[
                    {"arm_id": "control", "description": "No intervention"},
                    {"arm_id": "treatment", "description": spec["name"]},
                ],
                primary_outcome={"metric_id": "effective_engagement.plan_adherence"},
                secondary_outcomes=[
                    {"metric_id": "objective.readiness_next_day"},
                    {"metric_id": "subjective.energy_next_day"},
                ],
                minimum_practical_effect=0.08,
                power_calc={"alpha": 0.05, "power": 0.8, "mde": 0.08, "n_required": 400},
                decision_rules={
                    "ship_if": "P(effect > 0.08) >= 0.85",
                    "kill_if": "P(effect < -0.05) >= 0.85",
                    "iterate_otherwise": "true",
                },
                stop_rules={
                    "max_duration_days": 28,
                    "early_kill": True,
                    "early_ship": True,
                    "anytime_valid": True,
                },
                analysis_plan={
                    "estimator": spec["estimator"],
                    "covariates": ["baseline_readiness", "sleep_score", "training_load"],
                    "subgroups": ["low_data_user", "traveling_context"],
                    "multiple_comparison_correction": "BH",
                },
                registered_at=now_iso(),
                long_term_holdout_flag=True,
            )
            study = CausalStudy(
                study_id=study_id,
                name=spec["name"],
                experiments=[experiment_id],
                dag={
                    "nodes": ["intervention", "baseline_readiness", "sleep", "outcome"],
                    "edges": [["baseline_readiness", "intervention"], ["baseline_readiness", "outcome"], ["intervention", "outcome"]],
                },
                identification_strategy=spec["identification_strategy"],
                estimators=[{"name": spec["estimator"]}],
                results={},
                decision="pending",
                rationale="Pre-registered study scaffold from the Building Bloom adaptation.",
                reviewed_by=["krunal_health_platform"],
                primary_tags=spec["tags"],
                registered_at=now_iso(),
            )
            self.experiments[experiment_id] = experiment
            self.studies[study_id] = study

    def resolve_for_candidate(self, candidate: InterventionCandidate) -> CausalStudy:
        for study in self.studies.values():
            if any(tag in study.primary_tags for tag in candidate.tags):
                return study
        return next(iter(self.studies.values()))


class ReferenceCausalEstimator:
    """Reference posterior generator for candidate interventions."""

    def estimate_candidates(
        self,
        state: StateEncoderResponse,
        candidates: list[InterventionCandidate],
        registry: CausalStudyRegistry,
    ) -> list[CausalEstimate]:
        readiness = state.state_scores["readiness"].mean
        fatigue = state.state_scores["fatigue"].mean
        stress = state.state_scores["stress"].mean
        sleep_debt = state.state_scores["sleep_debt"].mean

        estimates: list[CausalEstimate] = []
        for candidate in candidates:
            study = registry.resolve_for_candidate(candidate)
            if candidate.action_id == "strength_progression":
                personal_mean = 0.22 + 0.25 * readiness - 0.18 * fatigue
                notes = ["Strength progression benefits when readiness dominates fatigue."]
            elif candidate.action_id == "recovery_row":
                personal_mean = 0.16 + 0.22 * fatigue + 0.10 * sleep_debt - 0.08 * stress
                notes = ["Recovery-biased movement helps more when fatigue and sleep debt are elevated."]
            elif candidate.action_id == "bedtime_protection":
                personal_mean = 0.12 + 0.28 * sleep_debt + 0.14 * stress
                notes = ["Sleep-protection nudges matter most when sleep debt and stress are non-trivial."]
            elif candidate.action_id == "protein_portion_check":
                personal_mean = 0.11 + 0.14 * fatigue + 0.12 * (1.0 - readiness)
                notes = ["Low-burden protein clarification mainly improves downstream recommendation quality."]
            else:
                personal_mean = 0.04
                notes = ["Fallback prior for uncategorized interventions."]

            personal_mean = _clamp(personal_mean, -0.1, 0.6)
            uncertainty = 0.08 + state.ood_score * 0.12 + state.state_scores["readiness"].width * 0.15
            identification_uncertainty = 0.18 if study.identification_strategy.startswith("randomized") else 0.28
            personal_effect = _posterior(personal_mean, uncertainty, identification_uncertainty, notes)
            population_effect = _posterior(personal_mean * 0.82, uncertainty + 0.03, identification_uncertainty, ["Population effect is shrunk relative to personal effect."])

            estimates.append(
                CausalEstimate(
                    action_id=candidate.action_id,
                    study_id=study.study_id,
                    population_effect=population_effect,
                    personal_effect=personal_effect,
                    rationale=notes + [f"Study: {study.name}", f"Design: {study.identification_strategy}"],
                )
            )
        return estimates

