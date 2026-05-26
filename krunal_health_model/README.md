# Krunal Health Model

This folder adds a Krunal-specific personal health layer on top of the PHIA wearable-insights repo.

The goal is to build a low-maintenance health thought partner that combines Oura data, voice-logged food, voice-logged exercise, labs, and daily feedback into practical recommendations for training, meals, sleep, recovery, focus, digestion, mood, and long-term health.

This is not a doctor replacement. It can reason, plan, track, and suggest low-risk experiments, but it must not diagnose, prescribe medication, change medication, handle emergencies, or override licensed clinicians.

## Current State

The first working loop is:

```text
sample Oura-style data + voice logs -> daily health state -> daily recommendation
```

The repo now also includes a broader Bloom-style architecture scaffold that treats the app as a layered platform:

```text
zone 1 ingestion -> zone 2 canonical timeline -> zone 3 features/experiments
-> state encoder -> causal layer -> safety cascade -> VoI planner
-> app-facing plan -> zone 4 insight cache
```

Run it locally from the repo root:

```bash
PYTHONPATH=krunal_health_model/src python -m krunal_health_engine.daily_health_engine --input-dir krunal_health_model/data/sample
```

To emit JSON instead of a readable text plan:

```bash
PYTHONPATH=krunal_health_model/src python -m krunal_health_engine.daily_health_engine --input-dir krunal_health_model/data/sample --format json
```

Run the Bloom-style architecture scaffold:

```bash
PYTHONPATH=krunal_health_model/src python -m krunal_health_platform.reference_system --input-dir krunal_health_model/data/sample
```

Run the app-facing backend/ML service boundary:

```bash
PYTHONPATH=krunal_health_model/src python -m krunal_health_platform.app_service_demo --input-dir krunal_health_model/data/sample
```

Run the local HTTP API:

```bash
PYTHONPATH=krunal_health_model/src python -m krunal_health_platform.local_api --port 8765
```

## Files

- `docs/krunal-health-data-contract.md`: what the system knows, where it comes from, and what is off-limits.
- `docs/personal-objectives.md`: Krunal's current goal stack.
- `docs/rl-environment-v0.md`: the first RL-style environment design.
- `docs/voice-logging-flow.md`: how Wispr Flow-style food and exercise logs should become structured data.
- `docs/architecture/building-bloom-adaptation.md`: how the PDF architecture was translated into this repo.
- `docs/architecture/implementation-roadmap.md`: implementation phases from scaffold to production.
- `docs/architecture/local-ml-backend-framework.md`: what useful local ML/backend code exists before external APIs are connected.
- `schemas/*.schema.json`: canonical data contracts for daily state and voice logs.
- `schemas/bloom/*.schema.json`: Bloom-style architecture contracts for the state encoder, studies, traces, and metrics.
- `data/sample/*.json`: sample inputs for the local prototype.
- `src/krunal_health_engine/daily_health_engine.py`: first local recommendation engine.
- `src/krunal_health_platform/`: layered platform scaffold based on the Building Bloom document, including storage, feature materialization, Bayesian personalization, causal summaries, safety, VoI planning, and app orchestration.

## Local API

The local API intentionally uses only the Python standard library. It gives the future app a concrete backend contract before we choose hosting infrastructure.

- `GET /health`
- `GET /learning-summary?user_id_hash=krunal_demo_user`
- `POST /daily-cycle` with either a `daily_state` object or an `input_dir`
- `POST /feedback` with `date`, `action_id`, optional `adhered`, `reward`, and `outcomes`

## Build Order

1. Keep the data contract stable.
2. Wire Oura API data into the daily state.
3. Pipe Wispr Flow transcripts into food and exercise logs.
4. Add nutrition parsing with uncertainty and clarification questions.
5. Track adherence and next-day outcomes.
6. Add personalization through N-of-1 experiments and contextual bandits.
7. Add constrained RL-style planning only after enough clean longitudinal data exists.
