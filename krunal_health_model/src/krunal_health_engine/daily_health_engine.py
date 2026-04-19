"""Build a daily health state and recommendation from Oura-style data and voice logs.

This is intentionally rule-guided. It gives the repo a working first loop while
keeping medical risk low. RL-style personalization should be layered on after
enough longitudinal data and feedback exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIMARY_GOALS = [
    "gain muscle",
    "reduce body fat",
    "improve sleep",
    "improve mood",
    "improve digestion",
    "improve focus",
    "maintain strong energy",
    "improve athletic performance",
    "support healthy testosterone",
    "increase peace and stability of mind",
]

HARD_CONSTRAINTS = [
    "no diagnosis",
    "no medication changes",
    "no unverified supplement recommendations",
    "no extreme cutting",
    "vegan-vegetarian compatible",
]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_nutrition(food_logs: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "estimated_calories": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
        "fiber_g": 0.0,
    }
    confidence_values = []

    for log in food_logs:
        nutrition = log.get("estimated_nutrition") or {}
        totals["estimated_calories"] += number(nutrition.get("calories"))
        totals["protein_g"] += number(nutrition.get("protein_g"))
        totals["carbs_g"] += number(nutrition.get("carbs_g"))
        totals["fat_g"] += number(nutrition.get("fat_g"))
        totals["fiber_g"] += number(nutrition.get("fiber_g"))
        confidence_values.append(number(log.get("confidence"), default=0.5))

    if not food_logs:
        return {
            "estimated_calories": None,
            "protein_g": None,
            "carbs_g": None,
            "fat_g": None,
            "fiber_g": None,
            "confidence": None,
        }

    confidence = sum(confidence_values) / len(confidence_values)
    return {**totals, "confidence": round(confidence, 2)}


def estimate_training_load(exercise_logs: list[dict[str, Any]]) -> dict[str, Any]:
    intensity_points = {
        "easy": 1.0,
        "moderate": 2.0,
        "hard": 3.0,
        "max": 4.0,
        "unknown": 1.5,
        None: 1.5,
    }
    load = 0.0
    activities = []

    for log in exercise_logs:
        for activity in log.get("activities", []):
            activity_type = activity.get("activity", "other")
            minutes = number(activity.get("duration_minutes"))
            distance = number(activity.get("distance_miles"))
            intensity = activity.get("intensity")

            if activity_type == "rowing" and minutes == 0 and distance > 0:
                minutes = distance * 10
            if activity_type == "strength_training" and minutes == 0:
                minutes = 45

            load += minutes * intensity_points.get(intensity, 1.5)
            activities.append(activity_type)

    if load >= 160:
        category = "high"
    elif load >= 70:
        category = "moderate"
    elif load > 0:
        category = "light"
    else:
        category = "none"

    return {
        "training_load_score": round(load, 1),
        "training_load_category": category,
        "logged_activity_types": sorted(set(activities)),
    }


def infer_day_type(
    wearable: dict[str, Any],
    subjective: dict[str, Any],
    exercise_summary: dict[str, Any],
) -> dict[str, Any]:
    sleep = wearable.get("sleep", {})
    readiness = wearable.get("readiness", {})

    sleep_minutes = number(sleep.get("duration_minutes"))
    sleep_score = number(sleep.get("sleep_score"))
    readiness_score = number(readiness.get("readiness_score"))
    resting_hr_delta = number(readiness.get("resting_heart_rate_delta_from_baseline_bpm"))
    hrv_delta_pct = number(readiness.get("hrv_delta_from_baseline_percent"))
    soreness = number(subjective.get("soreness_1_to_10"))
    energy = number(subjective.get("energy_1_to_10"))
    load_category = exercise_summary.get("training_load_category")

    recovery_flags = []
    if 0 < sleep_minutes < 390:
        recovery_flags.append("sleep under 6.5 hours")
    if 0 < sleep_score < 70:
        recovery_flags.append("sleep score below 70")
    if 0 < readiness_score < 70:
        recovery_flags.append("readiness below 70")
    if resting_hr_delta >= 5:
        recovery_flags.append("resting heart rate above baseline")
    if hrv_delta_pct <= -12:
        recovery_flags.append("HRV below baseline")
    if soreness >= 7:
        recovery_flags.append("high soreness")
    if 0 < energy <= 4:
        recovery_flags.append("low energy")

    if len(recovery_flags) >= 2:
        day_type = "recovery-biased"
        confidence = 0.78
    elif load_category == "high":
        day_type = "maintenance-biased"
        confidence = 0.68
    elif readiness_score >= 78 and sleep_score >= 78 and soreness <= 5:
        day_type = "training-biased"
        confidence = 0.72
    else:
        day_type = "maintenance-biased"
        confidence = 0.58

    return {
        "name": "day_type",
        "value": day_type,
        "confidence": confidence,
        "source": "rule_v0_sleep_readiness_subjective_training_load",
        "supporting_flags": recovery_flags,
    }


def build_daily_state(input_dir: Path) -> dict[str, Any]:
    oura = load_json(input_dir / "oura_daily.json", default={})
    food_logs = load_json(input_dir / "food_voice_logs.json", default=[])
    exercise_logs = load_json(input_dir / "exercise_voice_logs.json", default=[])
    subjective = load_json(input_dir / "subjective_checkin.json", default={})

    nutrition_summary = summarize_nutrition(food_logs)
    exercise_summary = estimate_training_load(exercise_logs)

    wearable = {
        "sleep": oura.get("sleep", {}),
        "readiness": oura.get("readiness", {}),
        "activity": oura.get("activity", {}),
    }

    day_type = infer_day_type(wearable, subjective, exercise_summary)

    return {
        "date": oura.get("date"),
        "sources": {
            "wearable_primary": "oura",
            "manual_food": "wispr_flow",
            "manual_exercise": "wispr_flow",
        },
        "wearable": wearable,
        "manual_logs": {
            "food": food_logs,
            "exercise": exercise_logs,
        },
        "nutrition_summary": nutrition_summary,
        "exercise_summary": exercise_summary,
        "baseline_behaviors": {
            "diet_pattern": "vegan-vegetarian",
            "caffeine": "none",
            "meditation_morning": True,
            "meditation_night": True,
            "supplements_taken": [
                "vegan multivitamin",
                "saffron",
                "ashwagandha",
            ],
        },
        "subjective": subjective,
        "environment": {
            "traveling": False,
            "location_context": "normal home environment",
            "air_quality_context": "high_air_quality",
        },
        "labs": {},
        "inferences": [day_type],
        "recommendation_context": {
            "primary_goals": PRIMARY_GOALS,
            "hard_constraints": HARD_CONSTRAINTS,
            "allowed_recommendation_types": [
                "workouts",
                "meals",
                "bedtime",
                "recovery days",
                "lab tests to discuss with clinician",
                "low-risk experiments",
            ],
        },
    }


def generate_recommendation(state: dict[str, Any]) -> dict[str, Any]:
    day_type = state["inferences"][0]["value"]
    flags = state["inferences"][0].get("supporting_flags", [])
    nutrition = state.get("nutrition_summary", {})
    protein = nutrition.get("protein_g")
    exercise_summary = state.get("exercise_summary", {})

    if day_type == "recovery-biased":
        workout = (
            "Keep training light: easy rowing, mobility, or walking. Avoid heavy lifting "
            "unless soreness and energy improve later in the day."
        )
        sleep = "Protect bedtime and keep the evening low-stimulation. Recovery is the main target today."
    elif day_type == "training-biased":
        workout = (
            "Lift today. Rowing is fine as easy or moderate conditioning, but keep enough "
            "energy for progressive strength work."
        )
        sleep = "Keep bedtime stable so training does not borrow from tomorrow's recovery."
    else:
        workout = (
            "Use a moderate session: lifting or rowing is fine, but avoid stacking hard "
            "rowing and heavy lifting unless subjective energy is high."
        )
        sleep = "Keep the sleep routine stable and watch whether training load affects tomorrow's readiness."

    if protein is None:
        meal = (
            "Log meals with enough detail to estimate protein. For muscle gain while cutting fat, "
            "protein is the first nutrition signal to stabilize."
        )
    elif protein < 110:
        meal = (
            f"Current logged protein is about {protein:.0f}g. Bias the next meal toward "
            "vegan-vegetarian protein such as tofu, tempeh, seitan, lentils, Greek yogurt if allowed, "
            "or a protein-forward bowl."
        )
    else:
        meal = (
            f"Logged protein is about {protein:.0f}g. Keep protein distributed across meals and avoid "
            "accidentally under-eating after training."
        )

    experiment = (
        "Track digestion after the highest-protein meal today. This helps find protein sources "
        "that support muscle gain without hurting digestion."
    )

    uncertainties = []
    if nutrition.get("confidence") is not None and nutrition["confidence"] < 0.75:
        uncertainties.append("Nutrition estimates are still rough from voice logs.")
    if not state.get("labs"):
        uncertainties.append("No lab data is connected yet.")
    if exercise_summary.get("training_load_category") == "none":
        uncertainties.append("No manual exercise was logged today.")

    return {
        "day_type": day_type,
        "why": flags or ["sleep, readiness, training load, and subjective check-in do not show strong recovery risk"],
        "workout": workout,
        "meal": meal,
        "sleep": sleep,
        "focus": "Use your strongest-focus window for demanding work. No caffeine recommendations because baseline is no caffeine.",
        "recovery": "Keep morning and night Vipassana protected. Do not add new supplements.",
        "experiment": experiment,
        "uncertainties": uncertainties,
        "safety": [
            "This is not medical advice.",
            "No diagnosis, medication changes, or unverified supplements.",
            "Discuss lab interpretation or concerning symptoms with a clinician.",
        ],
    }


def render_text(state: dict[str, Any], recommendation: dict[str, Any]) -> str:
    lines = [
        f"Date: {state.get('date')}",
        f"Day type: {recommendation['day_type']}",
        "",
        "Why:",
    ]
    lines.extend(f"- {item}" for item in recommendation["why"])
    lines.extend(
        [
            "",
            f"Workout: {recommendation['workout']}",
            "",
            f"Food: {recommendation['meal']}",
            "",
            f"Sleep: {recommendation['sleep']}",
            "",
            f"Focus: {recommendation['focus']}",
            "",
            f"Recovery: {recommendation['recovery']}",
            "",
            f"Experiment: {recommendation['experiment']}",
        ]
    )
    if recommendation["uncertainties"]:
        lines.append("")
        lines.append("Uncertainty:")
        lines.extend(f"- {item}" for item in recommendation["uncertainties"])
    lines.append("")
    lines.append("Safety:")
    lines.extend(f"- {item}" for item in recommendation["safety"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Krunal's daily health engine v0.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing sample input JSON files.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    state = build_daily_state(args.input_dir)
    recommendation = generate_recommendation(state)

    if args.format == "json":
        print(json.dumps({"state": state, "recommendation": recommendation}, indent=2))
    else:
        print(render_text(state, recommendation))


if __name__ == "__main__":
    main()

