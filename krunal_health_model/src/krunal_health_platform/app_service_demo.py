"""Run the app-facing health service on local sample data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from krunal_health_engine.daily_health_engine import build_daily_state

from .service import HealthIntelligenceService


def render_text(result: dict[str, object]) -> str:
    plan = result["plan"]
    learning = result["learning_summary"]
    feature_bundle = result["feature_bundle"]
    lines = [
        f"Date: {plan['date']}",
        f"Day type: {plan['day_type']}",
        f"Selected action: {plan['trace'].get('selected_action_id')}",
        "",
        f"Summary: {plan['summary']}",
        "",
        f"Workout: {plan['workout']}",
        f"Nutrition: {plan['nutrition']}",
        f"Sleep: {plan['sleep']}",
        f"Focus: {plan['focus']}",
    ]
    if plan.get("question"):
        lines.extend(["", f"Question: {plan['question']}"])
    lines.extend(
        [
            "",
            "Feature snapshot:",
            f"- sleep debt minutes: {feature_bundle['current'].get('sleep_debt_minutes')}",
            f"- recovery pressure: {feature_bundle['current'].get('recovery_pressure')}",
            f"- protein target gap: {feature_bundle['current'].get('protein_target_gap_g')}",
            "",
            "Learning state:",
            f"- feedback events: {learning['feedback_events']}",
            "- personal posteriors are initialized until real adherence/outcome feedback is logged",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the app-facing health intelligence service.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing sample input JSON files.")
    parser.add_argument("--db-path", type=Path, default=None, help="Optional SQLite DB path. Defaults to in-memory.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    service = HealthIntelligenceService(args.db_path or ":memory:")
    try:
        daily_state = build_daily_state(args.input_dir)
        result = service.run_daily_cycle(
            user_id_hash="krunal_demo_user",
            daily_state=daily_state,
        ).to_dict()
    finally:
        service.close()

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(render_text(result))


if __name__ == "__main__":
    main()
