"""Run the Bloom-style scaffold on the existing sample inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from krunal_health_engine.daily_health_engine import build_daily_state

from .data_platform import FourZoneDataPlatform, MemoryArchitecture
from .metrics import AlignmentMonitor, MetricsRegistry
from .models import UserBurdenBudget, jsonable
from .orchestration import HealthPlatformOrchestrator
from .translators import daily_state_to_state_encoder_request


def render_text(plan: dict[str, object]) -> str:
    lines = [
        f"Date: {plan['date']}",
        f"Day type: {plan['day_type']}",
        f"Headline: {plan['headline']}",
        "",
        f"Summary: {plan['summary']}",
        "",
        f"Workout: {plan['workout']}",
        "",
        f"Nutrition: {plan['nutrition']}",
        "",
        f"Sleep: {plan['sleep']}",
        "",
        f"Focus: {plan['focus']}",
    ]
    if plan.get("question"):
        lines.extend(["", f"Clarifying question: {plan['question']}"])
    uncertainty = plan.get("uncertainty") or []
    if uncertainty:
        lines.append("")
        lines.append("Uncertainty:")
        lines.extend(f"- {item}" for item in uncertainty)
    lines.append("")
    lines.append("Provenance:")
    lines.extend(f"- {item}" for item in plan.get("provenance", []))
    lines.append("")
    lines.append("Safety:")
    lines.extend(f"- {item}" for item in plan.get("safety_notes", []))
    return "\n".join(lines)


def run_reference_system(input_dir: Path) -> dict[str, object]:
    user_id_hash = "krunal_demo_user"
    platform = FourZoneDataPlatform()
    memory = MemoryArchitecture()
    metrics = MetricsRegistry()
    alignment_monitor = AlignmentMonitor()
    orchestrator = HealthPlatformOrchestrator()

    raw_files = {
        "oura": input_dir / "oura_daily.json",
        "food_voice": input_dir / "food_voice_logs.json",
        "exercise_voice": input_dir / "exercise_voice_logs.json",
        "subjective": input_dir / "subjective_checkin.json",
    }

    ingestion_records = []
    for vendor, path in raw_files.items():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        record = platform.ingest_raw_vendor_payload(user_id_hash, vendor, payload)
        ingestion_records.append(record.record_id)

    daily_state = build_daily_state(input_dir)
    canonical_record = platform.write_canonical_timeline(
        user_id_hash=user_id_hash,
        canonical_timeline=daily_state,
        source_record_ids=ingestion_records,
    )
    encoder_request = daily_state_to_state_encoder_request(daily_state, user_id_hash=user_id_hash)
    feature_record = platform.write_feature_bundle(
        user_id_hash=user_id_hash,
        feature_bundle=encoder_request,
        canonical_record_id=canonical_record.record_id,
    )

    memory.remember_fact(user_id_hash, "diet_pattern", daily_state.get("baseline_behaviors", {}).get("diet_pattern"))
    memory.remember_fact(user_id_hash, "meditation_rhythm", "morning_and_night")
    memory.set_working_memory(
        user_id_hash,
        goal="optimize training, sleep, and nutrition with low maintenance",
        plan={"phase": "building_bloom_architecture_scaffold"},
    )

    plan = orchestrator.run_daily_plan(
        user_id_hash=user_id_hash,
        daily_state=daily_state,
        encoder_request=encoder_request,
        zone_lineage={
            "zone1": ingestion_records[0],
            "zone2": canonical_record.record_id,
            "zone3": feature_record.record_id,
        },
        burden_budget=UserBurdenBudget(
            daily_cap=1.0,
            weekly_cap=5.0,
            per_channel_caps={"app": 0.5, "push": 0.3, "chat": 0.4},
            quiet_hours=["22:00-07:00"],
        ),
    )
    insight_record = platform.publish_insight(
        user_id_hash=user_id_hash,
        insight=plan,
        feature_record_id=feature_record.record_id,
    )
    memory.cache_insight(user_id_hash, jsonable(plan))

    alignment_snapshot = alignment_monitor.evaluate(
        engagement_value=0.62,
        outcome_value=0.58,
    )

    return {
        "plan": jsonable(plan),
        "zone_records": {
            "ingestion": ingestion_records,
            "canonical": canonical_record.record_id,
            "feature": feature_record.record_id,
            "insight": insight_record.record_id,
        },
        "audit_log_count": len(platform.audit_log),
        "memory_context": memory.get_context(user_id_hash),
        "metrics_registered": sorted(metrics.metrics.keys()),
        "alignment_snapshot": jsonable(alignment_snapshot),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Bloom-style health platform scaffold.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing sample input JSON files.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    result = run_reference_system(args.input_dir)
    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(render_text(result["plan"]))


if __name__ == "__main__":
    main()

