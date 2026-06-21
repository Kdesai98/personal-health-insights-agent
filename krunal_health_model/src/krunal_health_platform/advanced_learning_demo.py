"""Exercise trainable state, bandit, RL, nutrition, labs, and escalation code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from krunal_health_engine.daily_health_engine import build_daily_state

from .service import HealthIntelligenceService


SAMPLE_FEEDBACK = [
    ("protein_portion_check", True, 0.74),
    ("strength_progression", True, 0.68),
    ("recovery_row", True, 0.71),
    ("bedtime_protection", False, 0.52),
    ("protein_portion_check", True, 0.77),
    ("strength_progression", False, 0.45),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run advanced local ML components.")
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()

    service = HealthIntelligenceService(":memory:")
    try:
        daily_state = build_daily_state(args.input_dir)
        service.run_daily_cycle("krunal_demo_user", daily_state)
        for action_id, adhered, reward in SAMPLE_FEEDBACK:
            service.record_feedback(
                user_id_hash="krunal_demo_user",
                date=str(daily_state["date"]),
                action_id=action_id,
                adhered=adhered,
                reward=reward,
                outcomes={
                    "energy_next_day_1_to_10": 8 if reward >= 0.70 else 6,
                    "digestion_next_day_1_to_10": 7,
                    "readiness_next_day": 84 if reward >= 0.70 else 72,
                },
            )
        output = {
            "learning_summary": service.build_learning_summary("krunal_demo_user"),
            "nutrition_parse": service.parse_food_voice_log("I had two servings of tofu, lentils, rice, and blueberries."),
            "lab_parse": service.parse_lab_text("ApoB 88 mg/dL\nHbA1c 5.4 %\nVitamin D 28 ng/mL"),
            "escalation": service.evaluate_escalation(free_text="Felt normal today, just sore after rowing."),
            "network_experiment": service.design_network_experiment(
                experiment_id="demo_social_challenge",
                user_ids=["krunal_demo_user"],
            ),
        }
    finally:
        service.close()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
