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
- `state_training.py`: trainable local state model heads for reward, adherence, energy, and readiness prediction.
- `calibration.py`: reliability and calibration reports for bounded predictions.
- `bayesian.py`: lightweight personal posteriors for adherence and reward by action.
- `bandit.py`: contextual bandit policy using feature bundles and feedback rewards.
- `rl_environment.py`: offline RL-style environment for policy simulation with learned or prior transitions.
- `causal_runtime.py`: conservative N-of-1 summaries over logged feedback.
- `nutrition.py`: deterministic voice-food parser with uncertainty and clarifying questions.
- `labs.py`: lab parser with clinician-discussion boundaries, not diagnosis.
- `escalation.py`: routine vs clinician-discussion vs emergency safety routing.
- `network_experiments.py`: cluster-randomized social experiment plans for future social features.
- `privacy.py`: consent, PHI redaction, user hashing, and bearer-token auth primitives.
- `service.py`: app-facing service boundary that runs a daily cycle and records feedback.
- `local_api.py`: standard-library HTTP API for local app development.
- `app_service_demo.py`: runnable local example of the app service.
- `advanced_learning_demo.py`: exercises trainable state, bandit, RL, nutrition, labs, escalation, and network experiment code.

## Why This Is The Right Amount Of ML For Now

The app needs reliable longitudinal infrastructure before it needs deep RL. The first useful learning system is:

1. Store what happened.
2. Store what was recommended.
3. Store whether Krunal followed it.
4. Store what happened next.
5. Update personal beliefs about which actions work.
6. Ask only when the expected value is high.

This matches the Building Bloom document's structure without pretending we already have enough data for a trained model.

## What "Built Before Data" Means

The code can train and run today, but the artifacts are low-data until Oura, voice logs, labs, and feedback accumulate. That is intentional:

- A trainable state model exists, but its useful weights need Krunal's history.
- Contextual bandit code exists, but it starts from priors until enough action feedback exists.
- The RL environment exists for offline policy simulation, but its transition model uses priors until daily transitions are observed.
- Nutrition and lab ingestion boundaries exist, but production-grade parsing can later swap in model calls behind the same contracts.
- The escalation workflow exists now because safety boundaries should not wait for personalization.

## What Comes Later

- Oura ingestion populates the daily health state automatically.
- Wispr Flow transcripts populate food and exercise logs.
- Labs add slow-moving clinical context.
- Randomized or alternating N-of-1 experiments replace purely observational summaries.
- Contextual bandits can use the Bayesian posteriors once feedback accumulates.
- Offline RL only makes sense after enough longitudinal state-action-outcome data exists.
