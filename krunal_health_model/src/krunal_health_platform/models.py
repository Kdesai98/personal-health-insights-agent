"""Domain models for the Bloom-style health platform scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class DataZone(str, Enum):
    ZONE1_INGESTION = "zone1_ingestion"
    ZONE2_CANONICAL = "zone2_canonical"
    ZONE3_FEATURE_EXPERIMENT = "zone3_feature_experiment"
    ZONE4_INSIGHT = "zone4_insight"


class PurposeOfUse(str, Enum):
    CONSUMER_COACHING = "consumer_coaching"
    RESEARCH = "research"
    CLINICAL_EXPORT = "clinical_export"


class SafetyStatus(str, Enum):
    PASS = "pass"
    ESCALATE = "escalate"
    BLOCK = "block"


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


@dataclass
class ScoreWithUncertainty:
    mean: float
    ci_low: float
    ci_high: float
    method: str
    calibration_subgroup: str | None = None
    epistemic_uncertainty: float | None = None
    aleatoric_uncertainty: float | None = None

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low


@dataclass
class PosteriorDistribution:
    mean: float
    ci_low: float
    ci_high: float
    probability_above_threshold: float
    method: str
    identification_uncertainty: float
    notes: list[str] = field(default_factory=list)


@dataclass
class StateEncoderRequest:
    user_id_hash: str
    window: dict[str, str]
    modalities: dict[str, Any]
    context: dict[str, Any]
    data_quality_flags: dict[str, Any]
    encoder_version_pin: str | None = None


@dataclass
class StateEncoderResponse:
    short_term_embedding: list[float]
    long_term_embedding: list[float]
    state_scores: dict[str, ScoreWithUncertainty]
    modality_support: dict[str, float]
    ood_score: float
    abstain: bool
    abstain_reasons: list[str]
    encoder_version: str
    computed_at: str
    computation_cost_ms: int


@dataclass
class InterventionCandidate:
    action_id: str
    category: str
    title: str
    description: str
    channel: str
    burden_cost: float
    risk_cost: float
    evidence_refs: list[str]
    tags: list[str]
    question_text: str | None = None
    action_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEstimate:
    action_id: str
    study_id: str
    population_effect: PosteriorDistribution
    personal_effect: PosteriorDistribution
    rationale: list[str]


@dataclass
class UserBurdenBudget:
    daily_cap: float
    weekly_cap: float
    per_channel_caps: dict[str, float]
    quiet_hours: list[str]
    used_today: float = 0.0
    used_this_week: float = 0.0

    def remaining_for_channel(self, channel: str) -> float:
        return self.per_channel_caps.get(channel, self.daily_cap)


@dataclass
class VoICandidate:
    candidate: InterventionCandidate
    acquisition_function: str
    expected_information_gain: float
    expected_utility_improvement: float
    downstream_decision_change_prob: float
    burden_cost: float
    risk_penalty: float
    notes: list[str] = field(default_factory=list)

    @property
    def combined_score(self) -> float:
        return (
            self.expected_utility_improvement
            + 0.7 * self.expected_information_gain * self.downstream_decision_change_prob
            - self.burden_cost
            - self.risk_penalty
        )


@dataclass
class VoIDecision:
    decision_type: str
    selected_candidate_id: str | None
    score: float
    rationale: list[str]
    clarifying_question: str | None = None
    abstain: bool = False


@dataclass
class SafetyDecision:
    stage: str
    status: SafetyStatus
    reasons: list[str]
    score: float | None = None


@dataclass
class Experiment:
    experiment_id: str
    owner: str
    hypothesis: str
    design_type: str
    unit_type: str
    randomization: dict[str, Any]
    eligibility_filter: str
    treatment_arms: list[dict[str, Any]]
    primary_outcome: dict[str, Any]
    secondary_outcomes: list[dict[str, Any]]
    minimum_practical_effect: float
    power_calc: dict[str, Any]
    decision_rules: dict[str, str]
    stop_rules: dict[str, Any]
    analysis_plan: dict[str, Any]
    registered_at: str
    long_term_holdout_flag: bool


@dataclass
class CausalStudy:
    study_id: str
    name: str
    experiments: list[str]
    dag: dict[str, Any]
    identification_strategy: str
    estimators: list[dict[str, Any]]
    results: dict[str, Any]
    decision: str
    rationale: str
    reviewed_by: list[str]
    primary_tags: list[str]
    registered_at: str


@dataclass
class MetricDefinition:
    metric_id: str
    definition: str
    tier: str
    ownership: str
    alarm_thresholds: dict[str, Any]
    consumers: list[str]


@dataclass
class MetricSnapshot:
    metric_id: str
    cohort: str
    window: str
    value: float
    computed_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionTrace:
    trace_id: str
    user_id_hash: str
    purpose_of_use: PurposeOfUse
    zone_lineage: dict[str, str]
    state_encoder_version: str
    candidate_posteriors: dict[str, dict[str, Any]]
    voi_scores: dict[str, dict[str, Any]]
    safety_decisions: dict[str, list[dict[str, Any]]]
    selected_action_id: str | None
    evidence_refs: list[str]
    generated_explanation: str
    verifier_outcome: dict[str, Any]
    created_at: str


@dataclass
class AppPlan:
    date: str
    day_type: str
    headline: str
    summary: str
    workout: str
    nutrition: str
    sleep: str
    focus: str
    question: str | None
    uncertainty: list[str]
    provenance: list[str]
    safety_notes: list[str]
    trace: dict[str, Any]

