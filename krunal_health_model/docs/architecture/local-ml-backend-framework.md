# Local ML and Backend Framework

This is the practical code framework for the personal health app before Oura, Wispr Flow, labs, or hosted infrastructure are connected.

The system does not train a large model and does not need GPUs. It runs as a local inference, learning, and orchestration layer around longitudinal health data.

## What Exists Now

```text
daily health state
-> SQLite app store
-> longitudinal feature bundle
-> state encoder
-> causal posterior scaffold
-> safety cascade
-> Value of Information policy
-> app plan
-> trace persistence
-> feedback/adherence events
-> Bayesian personalization summaries
-> N-of-1 causal summaries
```

## App-Useful Modules

- `storage.py`: local SQLite persistence for daily states, feature bundles, intervention traces, and feedback.
- `features.py`: extracts daily metrics and rolling baselines that future models can use.
- `bayesian.py`: lightweight personal posteriors for adherence and reward by action.
- `causal_runtime.py`: conservative N-of-1 summaries over logged feedback.
- `service.py`: app-facing service boundary that runs a daily cycle and records feedback.
- `local_api.py`: standard-library HTTP API for local app development.
- `app_service_demo.py`: runnable local example of the app service.

## Why This Is The Right Amount Of ML For Now

The app needs reliable longitudinal infrastructure before it needs deep RL. The first useful learning system is:

1. Store what happened.
2. Store what was recommended.
3. Store whether Krunal followed it.
4. Store what happened next.
5. Update personal beliefs about which actions work.
6. Ask only when the expected value is high.

This matches the Building Bloom document's structure without pretending we already have enough data for a trained model.

## What Comes Later

- Oura ingestion populates the daily health state automatically.
- Wispr Flow transcripts populate food and exercise logs.
- Labs add slow-moving clinical context.
- Randomized or alternating N-of-1 experiments replace purely observational summaries.
- Contextual bandits can use the Bayesian posteriors once feedback accumulates.
- Offline RL only makes sense after enough longitudinal state-action-outcome data exists.
