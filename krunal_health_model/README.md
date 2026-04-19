# Krunal Health Model

This folder adds a Krunal-specific personal health layer on top of the PHIA wearable-insights repo.

The goal is to build a low-maintenance health thought partner that combines Oura data, voice-logged food, voice-logged exercise, labs, and daily feedback into practical recommendations for training, meals, sleep, recovery, focus, digestion, mood, and long-term health.

This is not a doctor replacement. It can reason, plan, track, and suggest low-risk experiments, but it must not diagnose, prescribe medication, change medication, handle emergencies, or override licensed clinicians.

## Current State

The first working loop is:

```text
sample Oura-style data + voice logs -> daily health state -> daily recommendation
```

Run it locally from the repo root:

```bash
PYTHONPATH=krunal_health_model/src python -m krunal_health_engine.daily_health_engine --input-dir krunal_health_model/data/sample
```

To emit JSON instead of a readable text plan:

```bash
PYTHONPATH=krunal_health_model/src python -m krunal_health_engine.daily_health_engine --input-dir krunal_health_model/data/sample --format json
```

## Files

- `docs/krunal-health-data-contract.md`: what the system knows, where it comes from, and what is off-limits.
- `docs/personal-objectives.md`: Krunal's current goal stack.
- `docs/rl-environment-v0.md`: the first RL-style environment design.
- `docs/voice-logging-flow.md`: how Wispr Flow-style food and exercise logs should become structured data.
- `schemas/*.schema.json`: canonical data contracts for daily state and voice logs.
- `data/sample/*.json`: sample inputs for the local prototype.
- `src/krunal_health_engine/daily_health_engine.py`: first local recommendation engine.

## Build Order

1. Keep the data contract stable.
2. Wire Oura API data into the daily state.
3. Pipe Wispr Flow transcripts into food and exercise logs.
4. Add nutrition parsing with uncertainty and clarification questions.
5. Track adherence and next-day outcomes.
6. Add personalization through N-of-1 experiments and contextual bandits.
7. Add constrained RL-style planning only after enough clean longitudinal data exists.

