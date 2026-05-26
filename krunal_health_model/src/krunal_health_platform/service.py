"""App-facing service boundary for the health intelligence platform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bayesian import BayesianPersonalizer
from .causal_runtime import NOf1CausalRuntime
from .features import materialize_longitudinal_features
from .models import UserBurdenBudget, jsonable
from .orchestration import HealthPlatformOrchestrator
from .storage import SQLiteHealthStore
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
        return {
            "feedback_events": len(feedback_events),
            "action_beliefs": beliefs,
            "causal_summaries": causal_summaries,
        }
