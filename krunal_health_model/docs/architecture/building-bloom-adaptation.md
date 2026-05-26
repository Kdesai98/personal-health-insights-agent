# Building Bloom Adaptation

This folder translates the system design in `/Users/krunal_desai/Downloads/Building Bloom - Max Balandat.pdf` into a concrete scaffold for Krunal's health app.

The document is not being copied verbatim into product code. Instead, its main contracts and architectural commitments are being encoded into the repo so they can guide the actual implementation.

## What Was Carried Over

The scaffold keeps the core architectural decisions from the document:

1. A four-zone data platform with explicit transformations between raw ingestion, canonical timeline, feature/experiment workloads, and app-facing insight artifacts.
2. A state encoder API that is foundation-model-agnostic and emits interpretable scores with uncertainty, modality support, OOD flags, and abstention.
3. A causal layer that treats every effect claim as belonging to a registered `CausalStudy`.
4. A policy layer that uses Value of Information to decide ask vs act vs wait under explicit burden budgets.
5. A cross-cutting safety cascade that blocks unsafe actions early and verifies generated explanations against a vetted evidence base.
6. A separate app orchestration layer that uses the lower layers instead of replacing them with a single LLM.
7. A metrics layer that treats alignment between engagement and outcomes as a first-class meta-metric.

## Repo Mapping

The Bloom-style adaptation lives under:

- `krunal_health_model/src/krunal_health_platform`
- `krunal_health_model/schemas/bloom`
- `krunal_health_model/docs/architecture`

Module mapping:

- `models.py`: domain contracts for scores, state encoder requests/responses, experiments, causal studies, traces, metrics, plans.
- `data_platform.py`: four-zone storage, audit log, purpose guard, memory architecture.
- `translators.py`: converts the existing daily state into the state encoder contract.
- `state_encoder.py`: reference v0 encoder with uncertainty propagation.
- `causal.py`: study registry and reference posterior generator.
- `policy.py`: candidate generation, VoI scoring, and final planning decision.
- `safety.py`: rules -> screening -> precision -> explanation verification.
- `metrics.py`: metric registry and engagement-outcome alignment monitor.
- `orchestration.py`: app-facing orchestration over the lower layers.
- `reference_system.py`: runnable end-to-end demonstration.

## What The Reference System Actually Does

The runnable scaffold is not pretending to be a production-grade ML system. It does four real things:

1. Ingests the current sample Oura-style and manual inputs into the four-zone data platform.
2. Builds a canonical daily timeline and a state-encoder request.
3. Runs a reference state encoder, causal estimator, safety cascade, and VoI planner.
4. Produces an app-facing plan plus an auditable trace.

That gives the project a durable system shape:

```text
raw inputs
-> zone 1 ingestion
-> zone 2 canonical timeline
-> zone 3 feature/experiment bundle
-> state encoder
-> causal posteriors
-> safety cascade
-> VoI planner
-> app plan + trace
-> zone 4 insight cache
```

## What Is Still A Scaffold

The following are intentionally reference implementations rather than finished product code:

- Uncertainty estimates are heuristic and architecture-preserving, not fully Bayesian.
- The causal layer emits structured posteriors but does not train causal forests, TMLE, or DML yet.
- The NLI verifier is a stand-in that checks evidence resolution, not a real entailment model.
- The evidence base is a local vetted store, not a clinical guideline retrieval system.
- The four-zone platform is in-memory, not backed by cloud infrastructure.
- The VoI engine uses explicit formulas and burden budgets, not streaming variational inference.

This is deliberate. The document's value is the system architecture. The scaffold preserves that architecture so the real components can replace the reference ones without breaking the contract.

## Immediate Next Build Steps

1. Real Oura connector into Zone 1 and Zone 2.
2. Voice-log ingestion path from Wispr Flow into canonical self-report and nutrition/exercise events.
3. Real nutrition lookup path using USDA / packaged-food sources.
4. Persistent storage for the four zones and audit trail.
5. Stronger state encoder trained on the canonical timeline.
6. Registered experiment workflow persisted in a real study registry.
7. Real claim verification and guideline retrieval.

