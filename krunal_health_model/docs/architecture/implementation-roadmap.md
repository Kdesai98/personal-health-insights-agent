# Implementation Roadmap

## Phase 0: Architecture Scaffold

Current deliverable:

- Four-zone data-platform abstraction.
- Memory separation between raw log, working memory, long-term facts, and insight cache.
- Stable state-encoder API.
- Reference uncertainty-aware encoder.
- Reference causal-study registry and posterior generator.
- VoI planner with burden budgets.
- Safety cascade and explanation verification.
- App-facing orchestrator and trace output.

This phase exists so all future work lands against stable contracts rather than improvising architecture feature by feature.

## Phase 1: Real Data Plane

- Replace sample JSON ingestion with Oura API sync.
- Add Wispr Flow ingestion bridge for food and exercise transcripts.
- Normalize raw payloads into Zone-2 canonical event types.
- Add a persistent audit trail and replayable transformations.
- Add structured long-term memory facts and a user-editable memory explorer.

## Phase 2: Stronger Inference

- Replace the heuristic state encoder with a trainable v0 ensemble.
- Add calibration jobs, subgroup calibration reporting, and OOD monitoring.
- Replace heuristic causal posteriors with pre-registered experiment estimators.
- Add off-policy evaluation for policy changes.
- Add real burden-budget learning from user behavior.

## Phase 3: Productized Personalization

- Add per-user adapter learning and longitudinal preference updates.
- Add network-aware social experimentation if social features exist.
- Add lab ingestion and FHIR-aligned clinical export.
- Add evidence retrieval from a vetted corpus and NLI-style claim verification.
- Add model promotion gates tied to the experiment registry and metric store.

## Phase 4: Research Extensions

- Streaming variational inference or another Bayesian update path.
- Better multivariate posterior approximations for joint outcome modeling.
- Contextual bandits for routine nudges.
- Conservative offline RL for multi-day plans.
- Stronger causal representation learning behind the same state-encoder API.

