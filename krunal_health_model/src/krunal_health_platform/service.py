"""App-facing service boundary for the health intelligence platform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bandit import ContextualBanditPolicy, ContextualBanditTrainer
from .bayesian import BayesianPersonalizer
from .calibration import ProbabilityCalibrator
from .causal_runtime import NOf1CausalRuntime
from .escalation import ClinicalEscalationWorkflow
from .features import materialize_longitudinal_features
from .labs import LabIngestionBoundary
from .models import UserBurdenBudget, jsonable
from .network_experiments import NetworkExperimentDesigner, SocialEdge
from .nutrition import VoiceNutritionParser
from .orchestration import HealthPlatformOrchestrator
from .rl_environment import HealthRLEnvironment, HealthTransitionModel
from .storage import SQLiteHealthStore
from .state_training import StateModelTrainer, TrainableStateEncoder
from .translators import daily_state_to_state_encoder_request


DEFAULT_ACTION_IDS = [
    "strength_progression",
    "recovery_row",
    "bedtime_protection",
    "protein_portion_check",
    "travel_context_check",
]


@dataclass
class DailyCycleResult:
    plan: dict[str, Any]
    feature_bundle: dict[str, Any]
    learning_summary: dict[str, Any]
    trace_id: str

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class HealthIntelligenceService:
    """Coordinates storage, features, inference, policy, and learning."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.store = SQLiteHealthStore(db_path)
        self.orchestrator = HealthPlatformOrchestrator()
        self.personalizer = BayesianPersonalizer()
        self.causal_runtime = NOf1CausalRuntime()
        self.nutrition_parser = VoiceNutritionParser()
        self.lab_boundary = LabIngestionBoundary()
        self.escalation = ClinicalEscalationWorkflow()
        self.network_designer = NetworkExperimentDesigner()

    def close(self) -> None:
        self.store.close()

    def run_daily_cycle(
        self,
        user_id_hash: str,
        daily_state: dict[str, Any],
        burden_budget: UserBurdenBudget | None = None,
    ) -> DailyCycleResult:
        self.store.upsert_daily_state(user_id_hash, daily_state)
        history = self.store.list_daily_states(user_id_hash, limit=28)
        feature_bundle = materialize_longitudinal_features(
            user_id_hash=user_id_hash,
            daily_states=history,
            lookback_days=28,
        ).to_dict()
        self.store.save_feature_bundle(user_id_hash, feature_bundle)

        encoder_request = daily_state_to_state_encoder_request(
            daily_state,
            user_id_hash=user_id_hash,
        )
        plan = self.orchestrator.run_daily_plan(
            user_id_hash=user_id_hash,
            daily_state=daily_state,
            encoder_request=encoder_request,
            zone_lineage={
                "daily_state": f"{user_id_hash}:{daily_state.get('date')}",
                "feature_bundle": f"{user_id_hash}:{feature_bundle['as_of_date']}",
            },
            burden_budget=burden_budget
            or UserBurdenBudget(
                daily_cap=1.0,
                weekly_cap=5.0,
                per_channel_caps={"app": 0.5, "push": 0.3, "chat": 0.4},
                quiet_hours=["22:00-07:00"],
            ),
        )
        plan_dict = jsonable(plan)
        trace_id = self.store.save_intervention_trace(
            user_id_hash=user_id_hash,
            date=str(daily_state.get("date")),
            trace=plan_dict["trace"],
        )
        learning_summary = self.build_learning_summary(user_id_hash)
        return DailyCycleResult(
            plan=plan_dict,
            feature_bundle=feature_bundle,
            learning_summary=learning_summary,
            trace_id=trace_id,
        )

    def record_feedback(
        self,
        user_id_hash: str,
        date: str,
        action_id: str,
        adhered: bool | None = None,
        outcomes: dict[str, Any] | None = None,
        reward: float | None = None,
        notes: str | None = None,
    ) -> str:
        feedback = {
            "date": date,
            "action_id": action_id,
            "adhered": adhered,
            "outcomes": outcomes or {},
            "reward": reward,
            "notes": notes,
        }
        return self.store.append_feedback_event(user_id_hash, feedback)

    def build_learning_summary(
        self,
        user_id_hash: str,
        action_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        actions = action_ids or DEFAULT_ACTION_IDS
        feedback_events = self.store.list_feedback_events(user_id_hash)
        personalizer = BayesianPersonalizer()
        personalizer.update_from_feedback(feedback_events)
        beliefs = personalizer.summarize(actions)
        causal_summaries = self.causal_runtime.estimate_all_actions(
            feedback_events=feedback_events,
            action_ids=actions,
        )
        feature_bundles = self.store.list_feature_bundles(user_id_hash)
        advanced = self.build_advanced_learning_summary(
            user_id_hash=user_id_hash,
            action_ids=actions,
            feature_bundles=feature_bundles,
            feedback_events=feedback_events,
        )
        return {
            "feedback_events": len(feedback_events),
            "action_beliefs": beliefs,
            "causal_summaries": causal_summaries,
            "advanced_learning": advanced,
        }

    def parse_food_voice_log(self, raw_text: str) -> dict[str, Any]:
        return self.nutrition_parser.parse(raw_text).to_dict()

    def parse_lab_text(self, text: str) -> dict[str, Any]:
        return self.lab_boundary.parse_text(text).to_dict()

    def evaluate_escalation(
        self,
        daily_state: dict[str, Any] | None = None,
        free_text: str | None = None,
    ) -> dict[str, Any]:
        return self.escalation.evaluate(daily_state=daily_state, free_text=free_text).to_dict()

    def design_network_experiment(
        self,
        experiment_id: str,
        user_ids: list[str],
        edges: list[dict[str, Any]] | None = None,
        treatment_arms: list[str] | None = None,
    ) -> dict[str, Any]:
        social_edges = [SocialEdge(**edge) for edge in (edges or [])]
        return self.network_designer.design_cluster_randomized_plan(
            experiment_id=experiment_id,
            user_ids=user_ids,
            edges=social_edges,
            treatment_arms=treatment_arms or ["control", "treatment"],
        ).to_dict()

    def build_advanced_learning_summary(
        self,
        user_id_hash: str,
        action_ids: list[str] | None = None,
        feature_bundles: list[dict[str, Any]] | None = None,
        feedback_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        actions = action_ids or DEFAULT_ACTION_IDS
        bundles = feature_bundles if feature_bundles is not None else self.store.list_feature_bundles(user_id_hash)
        feedback = feedback_events if feedback_events is not None else self.store.list_feedback_events(user_id_hash)
        if not bundles:
            return {
                "status": "no_feature_history",
                "state_model": None,
                "bandit": None,
                "rl_environment": None,
            }

        state_artifact = StateModelTrainer().fit(bundles, feedback)
        state_scores = TrainableStateEncoder(state_artifact).predict(bundles[-1])
        bandit_artifact = ContextualBanditTrainer(actions).fit(bundles, feedback)
        bandit_decision = ContextualBanditPolicy(bandit_artifact).choose(bundles[-1], allowed_actions=actions)
        transition_artifact = HealthTransitionModel().fit(
            daily_states=self.store.list_daily_states(user_id_hash),
            intervention_traces=self.store.list_intervention_traces(user_id_hash),
        )
        env = HealthRLEnvironment(transition_artifact, horizon=3)
        env.reset(bundles[-1])
        _, simulated_reward, _, transition_info = env.step(bandit_decision.selected_action_id)

        calibration_report = None
        if len(feedback) >= 5:
            predictions = []
            outcomes = []
            bundle_by_date = {str(bundle.get("as_of_date")): bundle for bundle in bundles}
            encoder = TrainableStateEncoder(state_artifact)
            for event in feedback:
                bundle = bundle_by_date.get(str(event.get("date")))
                if not bundle:
                    continue
                reward = event.get("reward")
                if reward is None:
                    continue
                predictions.append(encoder.predict(bundle)["utility_reward"].mean)
                outcomes.append(float(reward))
            if predictions:
                calibration_report = ProbabilityCalibrator(
                    model_name=state_artifact.model_id,
                    target_name="utility_reward",
                ).fit(predictions, outcomes).to_dict()

        return {
            "status": "ready_low_data" if len(feedback) < 20 else "ready_personalized",
            "state_model": {
                "artifact": state_artifact.to_dict(),
                "latest_scores": {key: jsonable(value) for key, value in state_scores.items()},
            },
            "bandit": {
                "artifact": bandit_artifact.to_dict(),
                "latest_decision": bandit_decision.to_dict(),
            },
            "rl_environment": {
                "transition_model": transition_artifact.to_dict(),
                "one_step_policy_reward": simulated_reward,
                "transition_info": transition_info,
            },
            "calibration": calibration_report,
        }
