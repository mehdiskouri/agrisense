# AgriSense Architecture Decision Record (ADR)

- Status: Accepted
- Date: 2026-03-02
- Owners: Core platform maintainers

## 1) Decision Scope

This record defines the system architecture, operational model, quality gates, and implementation boundaries for AgriSense. It is the single source of truth for technical decisions that govern backend behavior, data modeling, compute strategy, and release reliability.

## 2) Context

AgriSense is an API-first platform for agricultural intelligence that must:
- model farms as cross-layer systems rather than isolated sensor tables,
- support heterogeneous farm configurations (open-field, greenhouse, hybrid),
- provide reproducible analytics and recommendations,
- expose natural-language answers grounded in observable data,
- run in both CPU-only and GPU-capable environments,
- remain deployable and verifiable with deterministic CI.

The platform also needs to preserve strict separations of responsibility across HTTP/API orchestration, persistence, numerical compute, and model execution.

## 3) Core Decisions

### D1 — Layered hypergraph is the canonical domain model

**Decision**
- Represent farms as layered hypergraphs where vertices capture entities (zones, sensors, valves, crop beds, weather stations, cameras, controllers) and hyperedges capture typed many-to-many relationships.
- Preserve explicit layer semantics (soil, irrigation, lighting/solar, weather, crop requirements, nutrients, vision).

**Rationale**
- Enables first-class cross-layer reasoning (e.g., irrigation requires soil + weather + crop demand + actuation capacity).
- Avoids loss of structure from flattening interactions into independent relational rows.

**Consequence**
- APIs and analytics are designed around relationship-aware queries and derived state, not just raw metric retrieval.

### D2 — Farm profile drives topology and capabilities

**Decision**
- Farm type is declared at creation (`open_field`, `greenhouse`, `hybrid`) and determines active layers, supported entities, and model execution paths.

**Rationale**
- Prevents invalid topology combinations and keeps behavior deterministic by configuration.

**Consequence**
- Ingestion validation and analytics routing are profile-aware.
- Hybrid farms may mix zone-level modes while sharing infrastructure.

### D3 — Deliberate Python/Julia boundary

**Decision**
- Python owns API, auth, validation, persistence, orchestration, jobs, and external integrations.
- Julia owns hypergraph compute, synthetic data generation, and numerical models.
- Bridge contract is narrow and explicit (Python invokes Julia via `juliacall`; structured payloads cross the boundary).

**Rationale**
- Keeps web concerns and compute concerns decoupled.
- Allows independent evolution of compute kernels and API surface.

**Consequence**
- Boundary stability is treated as an interface contract.
- Runtime initialization and readiness checks include bridge health.

### D4 — PostgreSQL + Redis persistence strategy

**Decision**
- PostgreSQL is the durable system of record for topology and time-series events.
- Redis is used for transient acceleration concerns (cache, coordination, rate limiting, pub/sub, job state).

**Rationale**
- Balances durable relational guarantees with low-latency operational primitives.

**Consequence**
- Startup/readiness requires both data store health and runtime bridge health.
- Database migrations are mandatory before serving traffic.

### D5 — API-first product surface with auditable answers

**Decision**
- Expose typed REST endpoints for topology, ingestion, analytics, and jobs.
- Expose a natural-language query endpoint that returns grounded answers with source context.
- Expose a real-time channel for live updates and alerts.

**Rationale**
- Supports both machine clients and non-technical operators.
- Preserves answer traceability.

**Consequence**
- Language responses must remain data-grounded and attributable.
- LLM output is framed as interface output, not source-of-truth state.

### D6 — Synthetic data is mandatory for reproducible demos

**Decision**
- Seed workflows generate realistic, correlated, multi-layer historical data with controlled stochastic behavior (seasonality, coupling, anomalies, dropouts).

**Rationale**
- Enables safe demonstrations and deterministic testing without private production data.

**Consequence**
- Data generation quality is part of platform correctness, not an optional script.

### D7 — Build once, run CPU or GPU

**Decision**
- Maintain runtime targets for CPU and GPU variants from a shared runtime base.
- Keep compute fallback behavior explicit so non-GPU environments remain functional.

**Rationale**
- Supports broad development and CI portability while preserving accelerated production paths.

**Consequence**
- Release validation includes image build checks for both runtime targets.
- Optional dedicated GPU validation lane remains separately triggerable.

### D8 — Reliability gates are strict and blocking

**Decision**
- CI gates for linting, formatting, typing, tests, and image build are all blocking.
- Container verification in CI must be deterministic and not dependent on flaky multi-service timing.

**Rationale**
- Prevents drift between local development and merge quality.
- Reduces false negatives from environmental nondeterminism.

**Consequence**
- Workflow changes prioritize reproducibility over brittle “live-like” smoke choreography.
- CI remains the source of truth when local Docker daemon access is unavailable.

### D9 — Startup orchestration is dependency-safe

**Decision**
- Runtime orchestration enforces dependency readiness and migration completion prior to API serving.
- Seeding operations are gated on migration and API readiness.

**Rationale**
- Avoids schema/race failures at boot and stabilizes environment bring-up.

**Consequence**
- Local and CI startup behavior follow the same dependency-order guarantees.

## 4) Non-Functional Targets

- Maintain low-latency API behavior for operational endpoints.
- Keep startup and rebuild paths bounded for demo-scale datasets.
- Preserve typed contracts and strict static analysis across service code.
- Provide operational health/readiness endpoints with meaningful dependency checks.

## 5) Security and Governance Constraints

- Authentication and authorization are enforced for protected operations.
- Rate limiting and API-key pathways support machine-to-machine integrations safely.
- Public/demo datasets remain synthetic; private client data is excluded.

## 6) Accepted Tradeoffs

- Additional architectural complexity is accepted to preserve cross-layer semantics and compute fidelity.
- Separate compute runtime (Julia) introduces boundary overhead but improves numerical extensibility and performance portability.
- Deterministic CI checks may be less production-like than full end-to-end orchestration, but provide higher reliability for merge gating.

## 7) Evolution Rules

Future changes should conform to these rules:
- Preserve the Python/Julia ownership boundary.
- Keep the bridge contract explicit and typed.
- Do not weaken blocking quality gates.
- Do not introduce runtime paths that bypass migration/readiness sequencing.
- Keep demo data synthetic and reproducible.

## 8) Supersession Policy

This file supersedes prior planning notes for architecture and delivery decisions. Any future deviation requires updating this ADR with explicit rationale, impact, and migration plan.
