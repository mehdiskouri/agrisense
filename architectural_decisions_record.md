# AgriSense — Architectural Decision Record

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-03-19 |
| **Owners** | Core platform maintainers |
| **Supersedes** | `ADR.md` (2026-03-02) |

---

## 1) Purpose

This record is the single source of truth for all technical decisions governing AgriSense's architecture, data modeling, compute strategy, operational model, and feature design. Every decision includes its context, rationale, consequences (benefits and costs), rejected alternatives with specific reasons, and file-level implementation evidence.

---

## 2) System Context

AgriSense is an API-first agricultural intelligence platform that must:

- Model farms as cross-layer systems rather than isolated sensor tables.
- Support heterogeneous farm configurations (`open_field`, `greenhouse`, `hybrid`).
- Provide reproducible analytics, recommendations, and uncertainty-quantified forecasts.
- Expose natural-language answers grounded in observable data via a tool-calling LLM agent.
- Detect anomalies with configurable sensitivity and dispatch alerts to external systems.
- Generate downloadable reports for offline stakeholder review.
- Offer interactive visualization of hypergraph topology.
- Run in both CPU-only and GPU-capable environments with identical correctness.
- Remain deployable and verifiable with deterministic CI.

The platform preserves strict separation of responsibility: HTTP/API orchestration (Python), persistence (PostgreSQL + Redis), numerical compute and modeling (Julia), and LLM orchestration (LangChain + Anthropic).

---

## 3) Decision Matrix

| ID | Category | Decision | Key Driver |
|----|----------|----------|------------|
| D1 | Data Model | Layered hypergraph as canonical domain model | Cross-layer reasoning fidelity |
| D2 | Configuration | Farm profile drives topology and capabilities | Deterministic behavior by declaration |
| D3 | Architecture | Deliberate Python/Julia boundary | Compute decoupling + performance |
| D4 | Persistence | PostgreSQL + Redis split strategy | Durability + low-latency operations |
| D5 | API Surface | REST + NL + WS product surface | Multi-client support + traceability |
| D6 | Testing | Mandatory synthetic data for reproducible demos | Safe demonstrations + deterministic CI |
| D7 | Runtime | Build once, run CPU or GPU | Development and CI portability |
| D8 | CI/CD | Strict blocking reliability gates | Merge quality assurance |
| D9 | Operations | Dependency-safe startup orchestration | Boot reliability |
| D10 | Analytics | Ensemble yield forecasting over single-model | Forecast robustness + uncertainty quantification |
| D11 | NL/AI | LangChain agent over bare Anthropic API | Structured tool calling + memory abstraction |
| D12 | Visualization | D3 force-directed dashboard over server-side rendering | Interactive exploration without build pipeline |
| D13 | Reporting | Programmatic openpyxl + ReportLab generation | Full formatting control without templates |
| D14 | Anomaly | Severity classification enum on anomaly events | Prioritized alerting + filtered queries |
| D15 | Integration | Webhook subscriptions with HMAC-SHA256 dispatch | Secure external notification |
| D16 | Memory | Redis conversation memory over database persistence | Fast ephemeral multi-turn dialogue |
| D17 | Anomaly | Per-sensor-type configurable thresholds | Domain-specific sensitivity tuning |

---

## 4) Core Decisions

### D1 — Layered hypergraph as canonical domain model

**Context**
Agricultural systems are inherently multi-dimensional: an irrigation decision depends on soil moisture, weather forecasts, crop growth stage, and valve capacity simultaneously. Representing these interactions as isolated relational tables loses structural information that is required for correct cross-layer reasoning.

**Decision**
Represent farms as layered hypergraphs `H = (V, E)` where vertices capture physical entities (sensors, valves, crop beds, weather stations, cameras, light fixtures, climate controllers) and hyperedges capture typed many-to-many relationships across 7 layers: `soil`, `irrigation`, `lighting`, `weather`, `crop_requirements`, `npk`, `vision`. Sparse incidence matrices encode layer structure; cross-layer queries are sparse matrix multiplications (`B_a' * B_b`) via CUSPARSE on GPU.

**Consequences**

_Benefits:_
- First-class cross-layer reasoning without lossy joins.
- Sparse matrix operations scale to large farm topologies on GPU.
- Layer semantics are preserved through the entire pipeline (ingestion → storage → compute → analytics → API).

_Costs:_
- Higher conceptual complexity than flat relational modeling.
- Incidence matrix construction adds overhead on graph build/update.
- Developers must understand hypergraph semantics to extend the system.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Flat relational model (one table per sensor type) | Loses cross-layer structure; multi-table joins for irrigation decisions become ad-hoc and fragile |
| Property graph (Neo4j / JanusGraph) | Adds operational complexity (separate database process); sparse matrix operations for GPU-accelerated cross-layer queries are not native to property graph engines |
| Simple adjacency graph | Cannot express many-to-many relationships across typed layers; hyperedges connecting arbitrary vertex subsets are required |

**Implementation Evidence**
- `core/AgriSenseCore/src/types.jl` — `FarmProfile` struct (fields: `farm_id`, `farm_type`, `active_layers::Set{Symbol}`, `zones`, `models`), `HyperGraphLayer` struct (fields: `incidence`, `vertex_features`, `feature_history`, `history_head`)
- `core/AgriSenseCore/src/hypergraph.jl` — `build_hypergraph(profile, vertices, edges)`, `make_sparse_incidence(row_inds, col_inds, nv, ne)` (GPU/CPU dispatch), `LAYER_FEATURE_DIMS` constant (soil→4, irrigation→3, weather→5, npk→3, lighting→3, vision→4, crop_requirements→5)
- `app/models/farm.py` — `Farm`, `Zone`, `Vertex`, `HyperEdge` ORM models
- `app/models/enums.py` — `HyperEdgeLayerEnum` (soil, irrigation, lighting, weather, crop_requirements, npk, vision), `VertexTypeEnum` (sensor, valve, crop_bed, weather_station, camera, light_fixture, climate_controller)

---

### D2 — Farm profile drives topology and capabilities

**Context**
Different farm types (open-field, greenhouse, hybrid) activate different sensor layers and support different entity types. Without a profile-driven model, invalid topology combinations are possible and analytics behavior becomes non-deterministic.

**Decision**
Farm type is declared at creation (`open_field`, `greenhouse`, `hybrid`) and determines active layers, supported vertex types, and model execution paths. Ingestion validation and analytics routing are profile-aware.

**Consequences**

_Benefits:_
- Invalid topology combinations are prevented at ingestion time.
- Analytics routing is deterministic by configuration, not runtime inference.
- Hybrid farms can mix zone-level modes (greenhouse zones + open-field zones) while sharing infrastructure.

_Costs:_
- Adding a new farm type requires updating the profile-to-layer mapping in both Python and Julia.
- Farm type is immutable after creation — changing it requires recreating the farm.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Unconstrained topology (accept any sensor on any farm) | Leads to invalid layer combinations, non-deterministic model behavior, and harder debugging |
| Runtime inference from connected sensors | Fragile — adding a single sensor could change analytics routing; behavior would not be reproducible |

**Implementation Evidence**
- `app/models/enums.py` — `FarmTypeEnum` (open_field, greenhouse, hybrid), `ZoneTypeEnum` (open_field, greenhouse)
- `app/services/farm_service.py` — `FarmService` (layer activation logic by farm type)
- `app/services/ingest_service.py` — `IngestService` (validates layer activity and vertex type compatibility per farm profile)
- `core/AgriSenseCore/src/types.jl` — `FarmProfile.active_layers::Set{Symbol}` determines which layers are built in the hypergraph

---

### D3 — Deliberate Python/Julia boundary

**Context**
The system requires both high-level web orchestration (routing, auth, persistence, external integrations) and high-performance numerical compute (sparse matrix operations, GPU kernels, statistical models). Coupling these in a single language forces compromises in either web ergonomics or compute performance.

**Decision**
- **Python** owns: API, auth, validation, persistence, orchestration, jobs, LLM integration, webhook dispatch, report generation.
- **Julia** owns: hypergraph construction, sparse matrix operations, GPU kernels, predictive models (irrigation, nutrients, yield, anomaly), synthetic data generation.
- Bridge uses `juliacall` (in-process FFI). Contract is narrow: Python invokes Julia functions with `Dict`/`Vector` arguments; Julia returns `Dict`/`Vector` results. GPU arrays never cross the boundary.

**Consequences**

_Benefits:_
- Web concerns and compute concerns evolve independently.
- Julia's multiple dispatch and JIT compilation deliver C-level performance for numerical kernels.
- In-process bridge eliminates network overhead (vs. microservice).
- Boundary is explicitly typed via `julia_contracts.py` TypedDict definitions.

_Costs:_
- `juliacall` adds startup latency (Julia JIT warm-up + precompilation).
- Developers need familiarity with both ecosystems.
- Bridge health must be included in readiness checks.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Pure Python (NumPy/SciPy) | Performance ceiling for sparse GPU operations; no native CUDA sparse matrix support; GIL constrains parallel compute |
| Julia gRPC microservice | Network serialization overhead for in-process-viable workloads; adds container orchestration complexity; latency budget is tight for interactive analytics |
| Rust + PyO3 | Lacks Julia's mathematical ecosystem (CUDA.jl, KernelAbstractions.jl, sparse matrix primitives); higher development cost for equivalent functionality |

**Implementation Evidence**
- `app/services/julia_bridge.py` — `initialize_julia()`, `build_graph()`, `yield_forecast_ensemble()`, `backtest_yield()`, `detect_anomalies()`, `cross_layer_query()`, `update_features()`, `generate_synthetic()` — all use `_bridge_timing()` telemetry and raise `JuliaBridgeError` on failure
- `app/services/julia_contracts.py` — TypedDict payload contracts for boundary type safety
- `app/services/julia_validators.py` — Runtime validators for dynamic Julia return payloads
- `core/AgriSenseCore/src/bridge.jl` — Python-facing API: `build_graph`, `query_farm_status`, `irrigation_schedule`, `nutrient_report`, `yield_forecast`, `yield_forecast_ensemble`, `backtest_yield` — all accept/return plain Dict/Vector
- `app/main.py` — Lifespan startup: `julia_bridge.initialize_julia()` + warm-up with `generate_synthetic(farm_type=..., days=1, seed=1)`; readiness check includes Julia bridge health

---

### D4 — PostgreSQL + Redis persistence strategy

**Context**
The system needs durable relational storage for farm topology, time-series sensor data, and audit records, plus low-latency primitives for caching, coordination, rate limiting, pub/sub, job state, and ephemeral conversation memory.

**Decision**
- **PostgreSQL 16 + PostGIS** is the durable system of record for topology, time-series events, auth, jobs, anomaly events, thresholds, and webhook subscriptions (18 tables total).
- **Redis 7** is used exclusively for transient acceleration across 7 lanes: graph cache, rate limiting, pub/sub (live feed), job state, report cache, conversation memory, and webhook dispatch queue.

**Consequences**

_Benefits:_
- Relational guarantees (ACID, foreign keys, constraints) protect data integrity.
- PostGIS enables future spatial queries without schema changes.
- Redis provides sub-millisecond latency for all transient operations.
- Clean separation: Redis loss causes degraded (not broken) behavior; PostgreSQL is the recovery source.

_Costs:_
- Two data stores to operate and monitor.
- Startup/readiness requires both stores to be healthy.
- Redis data (conversation memory, report cache) is volatile — lost on restart.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| TimescaleDB | Adds extension complexity on top of PostgreSQL; PostGIS is already required for spatial; hypertable management overhead for a system that also needs relational topology tables |
| Redis-only persistence (with RDB/AOF) | No durable relational guarantees; foreign key enforcement and complex queries (anomaly history with filters) are impractical in Redis |
| MongoDB | Document model is a poor fit for the relational topology (farms → zones → vertices → hyperedges); cross-document joins are expensive |

**Implementation Evidence**
- `docker-compose.yml` — `postgres` service (`postgis/postgis:16-3.4`, port 5432, healthcheck: `pg_isready`), `redis` service (`redis:7-alpine`, port 6379, healthcheck: `redis-cli ping`)
- `app/database.py` — Async SQLAlchemy engine with `asyncpg` driver
- `app/config.py` — `database_url` (PostgreSQL), `redis_url` (Redis)
- `app/main.py` — Lifespan: validates DB connectivity (`SELECT 1`), initializes Redis client, readiness endpoint checks both
- `app/middleware/rate_limit.py` — `RateLimitMiddleware`: Redis atomic counters with key pattern `ratelimit:farm:{id}:{identity}:{minute_bucket}`, 65s TTL
- `alembic/versions/` — 4 migrations managing 18 tables across the PostgreSQL schema

---

### D5 — API-first product surface with auditable answers

**Context**
The platform serves diverse clients: machine-to-machine IoT controllers, web dashboards, mobile apps, and non-technical agronomists. Answers from the NL interface must be traceable to data, not treated as ground truth.

**Decision**
- Typed REST endpoints (`/api/v1`) for topology, ingestion, analytics, anomalies, jobs, and reports.
- Natural-language query endpoint (`/api/v1/ask/{farm_id}`) returns grounded answers with tool attribution and telemetry.
- Real-time WebSocket channel (`/ws/{farm_id}/live`) for live sensor updates via Redis pub/sub.
- Interactive documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc).

**Consequences**

_Benefits:_
- Machine clients get typed, predictable responses.
- NL answers include `tools_called`, `intent`, and `telemetry` (token counts, cost) for auditability.
- WebSocket enables real-time dashboards without polling.
- OpenAPI spec is auto-generated from Pydantic schemas.

_Costs:_
- NL endpoint adds LLM latency and cost to the request path.
- WebSocket requires persistent connections and Redis pub/sub subscription management.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| GraphQL | Hypergraph structure doesn't map cleanly to GraphQL's schema model; REST is simpler for the predominantly CRUD + analytics query pattern; auto-generated docs via OpenAPI are better supported in FastAPI |
| gRPC | Not browser-friendly without a proxy; Swagger/ReDoc documentation is more accessible for agronomist users |
| NL-only interface | Loses type safety for machine clients; makes integration with IoT controllers impractical |

**Implementation Evidence**
- `app/routes/` — 7 router modules: `farms.py` (7 endpoints), `ingest.py` (6), `analytics.py` (16), `anomalies.py` (9), `ask.py` (3), `jobs.py` (2), `ws.py` (1 WebSocket)
- `app/main.py` — Router registration with `prefix="/api/v1"` for all REST routers; WebSocket router without prefix; `docs_url="/docs"`, `redoc_url="/redoc"`
- `app/schemas/` — Pydantic v2 request/response models across 7 schema modules
- `app/services/llm_service.py` — `LLMService.ask()` returns `AskResponse` including `tools_called`, `intent`, `telemetry` (token counts + cost)

---

### D6 — Mandatory synthetic data for reproducible demos

**Context**
Production data is under NDA. Development, CI, and demo workflows require realistic, deterministic test data that covers all 7 layers with proper cross-layer correlations, temporal patterns, anomalies, and dropouts.

**Decision**
The seed workflow generates correlated multi-layer historical data with controlled stochastic behavior. Synthetic generation supports configurable seasonality, cross-layer coupling, anomaly injection, and NaN dropout patterns. GPU acceleration is used when available (CURAND + CUBLAS).

**Consequences**

_Benefits:_
- Safe demonstrations without private data exposure.
- Deterministic seeding (via explicit `seed` parameter) enables reproducible CI.
- Synthetic data quality is treated as a correctness concern, not an afterthought.

_Costs:_
- Generator complexity (7 sub-generators with cross-layer correlation modeling).
- GPU synthetic generation adds Julia startup latency to seed workflow.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Static fixture files (JSON/CSV) | Cannot model temporal correlations, anomalies, or dropouts; brittle to schema changes |
| Production data subset (anonymized) | NDA restrictions prevent even anonymized production data in the repository; anonymization may not be sufficient for agricultural domain data |
| Third-party synthetic data service | External dependency; no control over agricultural-domain-specific correlation patterns |

**Implementation Evidence**
- `core/AgriSenseCore/src/synthetic/generator.jl` — Orchestrator for all synthetic layers
- `core/AgriSenseCore/src/synthetic/` — 7 sub-generators: `soil.jl`, `weather.jl`, `lighting.jl`, `npk.jl`, `vision.jl`, `correlations.jl` (cross-layer coupling)
- `scripts/seed_db.py` — Seed script invoking Julia synthetic generation
- `app/services/julia_bridge.py` — `generate_synthetic(farm_type, days, seed)` function

---

### D7 — Build once, run CPU or GPU

**Context**
Development machines and CI runners rarely have GPUs. Production may have NVIDIA hardware. The same codebase and test suite must produce correct results on both.

**Decision**
Julia compute uses `KernelAbstractions.jl` for hardware abstraction: `@kernel` macros compile to CUDA or CPU backends at runtime. Docker multi-stage build produces `runtime-cpu` and `runtime-gpu` targets from a shared `runtime-base`. GPU availability is detected automatically — no code changes or configuration flags required.

**Consequences**

_Benefits:_
- Single codebase, single test suite, two runtime targets.
- CI runs all tests on CPU without GPU hardware.
- Production can upgrade to GPU without code changes.

_Costs:_
- KernelAbstractions overhead on CPU (small constant factor vs. hand-written loops).
- Must maintain and test both Docker targets.
- Optional GPU CI lane requires dedicated runner infrastructure.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| CUDA-only implementation | Blocks development and CI on GPU hardware availability; makes the project inaccessible to contributors without GPUs |
| Separate CPU and GPU codebases | Doubles maintenance; divergence risk between implementations; test coverage gaps |
| Python-only GPU (CuPy/Numba) | Lacks Julia's sparse matrix ecosystem; GIL limits parallel compute; doesn't integrate with existing Julia models |

**Implementation Evidence**
- `core/AgriSenseCore/src/models/yield.jl` — `fao_yield_kernel!`, `nutrient_stress_kernel!`, `weather_stress_kernel!` all use `@kernel` macro from KernelAbstractions
- `core/AgriSenseCore/src/models/anomaly.jl` — `rolling_stats_kernel!`, `western_electric_kernel!` use `@kernel`
- `core/AgriSenseCore/src/hypergraph.jl` — `make_sparse_incidence()` dispatches to CuSparseMatrixCSC or standard SparseMatrixCSC based on CUDA availability
- `Dockerfile` — 4 stages: `julia-deps` (Julia 1.12), `python-deps` (Python 3.13), `runtime-base` (shared), `runtime-cpu` (env: `AGRISENSE_RUNTIME_VARIANT=cpu`), `runtime-gpu` (env: `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`)
- `.github/workflows/ci.yml` — `julia-gpu-test` job: manual `workflow_dispatch` with `enable_gpu_ci` input, runs on self-hosted GPU runner

---

### D8 — Strict blocking reliability gates

**Context**
Drift between local development and merge quality leads to regressions. Flaky CI jobs that pass inconsistently erode trust in the pipeline and allow defective code to merge.

**Decision**
All CI gates — linting (`ruff`), type checking (`mypy`), Python tests (`pytest`), Julia tests (`Pkg.test`), and Docker image builds — are blocking. Container verification is deterministic and does not depend on multi-service timing. A separate optional GPU test lane is manually triggerable.

**Consequences**

_Benefits:_
- Merge quality is guaranteed by CI — no "works on my machine" regressions.
- Deterministic checks (no timing-dependent multi-service orchestration in CI).
- GPU tests are isolated to a dedicated lane, preventing CI flakiness from GPU runner availability.

_Costs:_
- Stricter gates slow merge velocity for trivial changes.
- CI is less "production-like" than full end-to-end orchestration (accepted tradeoff for reliability).

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Non-blocking CI with warnings | Allows quality drift; warnings are systematically ignored over time |
| Full docker-compose integration test in CI | Timing-dependent service orchestration is inherently flaky in CI; healthcheck race conditions cause false negatives |
| GPU tests in every CI run | GPU runners are scarce and expensive; blocking on GPU availability would stall all PRs |

**Implementation Evidence**
- `.github/workflows/ci.yml` — 5 jobs: `lint` (ruff check + ruff format --check), `python-test` (pytest with PostgreSQL + Redis services), `julia-test` (Pkg.test on CPU), `docker-build` (build both `runtime-cpu` and `runtime-gpu` targets), `julia-gpu-test` (manual dispatch, `enable_gpu_ci=true`)
- `Makefile` — `lint`, `format`, `typecheck`, `test`, `test-julia` targets

---

### D9 — Dependency-safe startup orchestration

**Context**
The API server depends on PostgreSQL (schema), Redis (cache), and Julia (compute core). Starting the API before migrations complete or before dependencies are healthy causes schema errors, cache misses, and bridge failures.

**Decision**
Docker Compose enforces dependency ordering via healthchecks and `depends_on` conditions: `postgres` healthy → `migrate` (one-shot: `alembic upgrade head`) → `api` starts → `seed` executes. The API lifespan validates database connectivity, initializes Redis, starts the webhook dispatch worker, initializes Julia, warms the bridge with synthetic generation, and optionally bootstraps graph cache.

**Consequences**

_Benefits:_
- Deterministic boot sequence — no schema/race failures.
- Same dependency ordering in local development, CI, and production.
- Readiness endpoint (`/health/ready`) checks all three subsystems before accepting traffic.

_Costs:_
- Startup latency includes Julia JIT warm-up + optional graph cache bootstrap.
- One-shot `migrate` container adds a transient Docker resource.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Retry loops in application code | Obscures boot failures; harder to debug; non-deterministic startup behavior |
| Combined migrate + serve in single container | Migration failures would bring down the API; separation allows independent debugging |
| External migration runner (CI-only) | Local development would lack migration enforcement; divergence between local and production boot sequences |

**Implementation Evidence**
- `docker-compose.yml` — `migrate` service: `depends_on: postgres: condition: service_healthy`, command: `alembic upgrade head`, restart: `"no"`; `api` service: `depends_on: postgres: service_healthy, redis: service_healthy, migrate: service_completed_successfully`; `seed` service: `depends_on: api: service_healthy, migrate: service_completed_successfully`
- `app/main.py` — Lifespan startup sequence: (1) `configure_structured_logging()`, (2) DB connectivity check (`SELECT 1`), (3) Redis init (`Redis.from_url`), (4) webhook worker task start (`run_dispatch_queue_worker`), (5) Julia init + warm-up, (6) optional graph cache bootstrap (`_bootstrap_graph_cache`)
- `app/main.py` — Lifespan shutdown: signal webhook worker stop → await/cancel, close Redis, dispose SQLAlchemy engine

---

### D10 — Ensemble yield forecasting over single-model prediction

**Context**
The original `YieldForecaster` uses a single FAO thermal-time model with optional ridge-regression residual correction. While effective for point estimates, it lacks uncertainty quantification — stakeholders cannot assess confidence in the forecast. A single model's structural assumptions also create blind spots when those assumptions are violated.

**Decision**
Implement a 3-member ensemble combining structurally diverse models:

| Member | Model | Strength |
|---|---|---|
| FAO thermal-time | Physiological stress model (soil Ks × nutrient Kn × light Kl × weather Kw) | Captures biological growth ceilings |
| Exponential smoothing | Time-series trend + seasonality decomposition | Adapts to recent trajectory shifts |
| Quantile regression | Non-parametric distributional estimation | Robust to outliers, produces native quantiles |

Final forecast is a **weighted quantile combination** — weights are derived from inverse-MAE performance in expanding-window backtest. Backtest uses temporal split: train 75% → validation 16.7% → test 8.3%, with configurable `n_folds` and `min_history`.

**Consequences**

_Benefits:_
- Uncertainty bounds (confidence intervals) on every forecast.
- Model diversity reduces structural blind spots.
- Backtestable — ensemble weights are empirically justified, not hand-tuned.
- Async backtest jobs prevent long-running backtest from blocking request threads.

_Costs:_
- ~3× compute cost per forecast (three model evaluations + weight combination).
- Additional infrastructure: `backtest_jobs` table, async job endpoints, Redis job state.
- Grid search over hyperparameters (α, β, λ) adds computational expense to backtest.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Single model + bootstrap confidence intervals | Bootstrap CIs lack structural diversity — they quantify sampling uncertainty of one model's assumptions, not model uncertainty |
| Bayesian model averaging (BMA) | Implementation complexity disproportionate to gain; requires prior specification and MCMC/variational inference machinery |
| Deep learning ensemble (LSTM, Transformer) | Data volume at farm scale is insufficient for deep models; overfitting risk is high; interpretability is poor for agronomist stakeholders |

**Implementation Evidence**
- `core/AgriSenseCore/src/models/yield.jl` — `compute_yield_forecast()` (single-model), `compute_stress_coefficients()` (4 GPU-accelerated stress vectors), `fao_yield_kernel!` / `nutrient_stress_kernel!` / `weather_stress_kernel!` (`@kernel` GPU kernels), `fit_residual_model()` (ridge regression)
- `core/AgriSenseCore/src/models/backtesting.jl` — `backtest_yield_ensemble()` (full temporal backtest), `optimize_ensemble_hyperparams!()` (grid search), `_collect_member_predictions()` (runs FAO-single, exp-smoothing, quantile-regression), `_ensemble_from_members()` (weighted combination), `_inverse_mae_weights()` (weight derivation), `_fold_train_windows()` (expanding-window fold boundaries)
- `app/services/julia_bridge.py` — `yield_forecast_ensemble(farm_id, include_members)`, `backtest_yield(farm_id, n_folds, min_history)` — Python→Julia bridge functions
- `core/AgriSenseCore/src/bridge.jl` — `yield_forecast_ensemble(graph_state; include_members)`, `backtest_yield(graph_state; n_folds, min_history)` — Julia entry points
- `app/models/jobs.py` — `BacktestJob` ORM model (table: `backtest_jobs`, columns: `farm_id`, `status`, `n_folds`, `min_history`, `result` JSONB)
- `app/routes/analytics.py` — `get_ensemble_yield_forecast()`, `run_yield_backtest()`, `enqueue_yield_backtest()`, `get_yield_backtest_job_status()`
- `alembic/versions/c3a1b9d2e4f6_add_backtest_jobs.py` — Migration adding `backtest_jobs` table

---

### D11 — LangChain agent over bare Anthropic API

**Context**
The `/ask` endpoint must translate natural-language questions into structured analytics queries, invoke the correct tools with correct arguments, maintain multi-turn conversation context, and stream token-by-token responses. Implementing this from scratch against the raw Anthropic API requires hand-coding tool dispatch, message formatting, retry logic, memory management, and streaming protocols.

**Decision**
Use **LangChain** (`langchain` + `langchain-anthropic` + `langchain-community`) to build a tool-calling agent backed by **Anthropic Claude 3.5 Sonnet** (`ChatAnthropic`). 7 analytics tools are bound to the agent via `build_tools()` factory. Conversation memory is Redis-backed with per-user-per-farm isolation, 20-message sliding window, and 1-hour TTL. Two tools (`get_zone_detail`, `run_yield_backtest`) are feature-flagged.

**Consequences**

_Benefits:_
- Declarative tool binding — tools are Python functions with type hints; LangChain handles schema extraction and invocation dispatch.
- Built-in conversation memory abstraction (`RedisChatMessageHistory`) with TTL support.
- Streaming support via LangChain's `astream_events` API.
- Feature-flagged tools allow gradual capability rollout.
- Cost tracking integrated into telemetry (`AskTelemetry` with input/output token counts and USD estimate).

_Costs:_
- LangChain dependency surface — transitive dependencies, version coupling, abstraction overhead.
- LangChain's abstractions can obscure debugging (agent decision-making is less transparent than hand-rolled dispatch).
- Anthropic model changes may require LangChain version updates.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Bare `anthropic` Python SDK | Requires manual tool dispatch loop, message formatting, memory management, and streaming implementation; significant boilerplate for equivalent functionality |
| LlamaIndex | Heavier framework oriented toward RAG/indexing; less suited to the tool-calling agent pattern used here |
| Custom agent loop | Lower-level control but higher maintenance burden; tool schema extraction, retry logic, and memory windowing must all be implemented manually |
| OpenAI API (GPT-4) | Anthropic's tool-calling protocol is cleaner for structured analytics tools; Claude 3.5 Sonnet offers strong reasoning at competitive cost |

**Implementation Evidence**
- `app/services/llm_service.py` — `LLMService` class: `ask()` (full agent invocation), `ask_stream()` (SSE streaming via `astream_events`), `_build_telemetry()` (cost estimation from `anthropic_input_cost_usd_per_million`/`anthropic_output_cost_usd_per_million`), `_fallback_response()` (static response when no API key), `clear_conversation()`
- `app/services/agent_tools.py` — `build_tools(farm_id, analytics, settings)` factory function returning 7 `BaseTool` instances: `get_farm_status`, `get_irrigation_schedule`, `get_nutrient_report`, `get_yield_forecast`, `get_active_alerts`, `get_zone_detail` (gated: `ask_enable_zone_detail_tool`), `run_yield_backtest` (gated: `ask_enable_backtest_tool`)
- `app/services/conversation_memory.py` — `build_memory()` (returns `RedisChatMessageHistory` or `InMemoryChatMessageHistory` fallback), `get_window_messages()` (bounded window), `refresh_ttl()`, `clear_conversation()`
- `app/routes/ask.py` — `ask_farm()` (POST), `stream_ask_farm()` (POST, SSE), `clear_farm_conversation()` (DELETE)
- `app/config.py` — Settings: `anthropic_api_key`, `anthropic_model` (default: `claude-3-5-sonnet-20241022`), `langchain_max_tokens` (1024), `langchain_max_iterations` (6), `langchain_conversation_ttl_seconds` (3600), `langchain_max_context_messages` (20), `langchain_verbose`, `ask_enable_zone_detail_tool` (True), `ask_enable_backtest_tool` (False)

---

### D12 — D3 force-directed dashboard over server-side rendering

**Context**
The hypergraph structure (vertices, hyperedges, layer memberships) needs a visual representation for operators to explore farm topology, identify connectivity patterns, and inspect vertex metadata. The visualization must be interactive (zoom, click, filter) and must not require a JavaScript build pipeline.

**Decision**
Serve a **static HTML page** at `/static/dashboard.html` that loads **D3.js v7 from CDN** and fetches graph data from `GET /api/v1/farms/{farm_id}/visualization`. The visualization uses a force-directed layout with layer-colored nodes, edge bundling by hyperedge membership, and click-to-inspect detail panels.

**Consequences**

_Benefits:_
- Zero server-side rendering cost — all computation happens in the browser.
- No build step — single HTML file with inline JavaScript, served via FastAPI's `StaticFiles` mount.
- Interactive: zoom, pan, click-to-inspect vertex metadata, layer filtering.
- D3 v7 is loaded from CDN — no bundling or npm required.

_Costs:_
- Requires a JavaScript-capable browser (no terminal/CLI access to visualization).
- No offline graph export (PDF/PNG) — strictly browser-rendered.
- Large graphs (hundreds of vertices) may have layout performance issues in the browser.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Server-side matplotlib/graphviz PNG | No interactivity — static images cannot support zoom, click-to-inspect, or layer filtering |
| Cytoscape.js | Heavier library (~300KB min); less ecosystem tooling than D3; compound graph support not needed for hypergraph visualization |
| Embedded React/Vue SPA | Requires a build pipeline (webpack/vite), `node_modules`, and ongoing maintenance; disproportionate infrastructure for a single dashboard page |
| Plotly Dash | Python-rendered HTML adds server-side complexity; less layout control than D3 for force-directed graphs |

**Implementation Evidence**
- `app/static/dashboard.html` — D3 v7 (CDN), force-directed layout, layer-colored nodes (CSS variables: `--soil`, `--irrigation`, `--lighting`, `--weather`, `--crop_requirements`, `--npk`, `--vision`), click-to-inspect panels; fonts: Space Grotesk + IBM Plex Mono
- `app/routes/farms.py` — `get_visualization(farm_id)` endpoint returning D3-compatible JSON (nodes + links with layer metadata)
- `app/services/analytics_service.py` — Visualization payload assembly
- `app/main.py` — `app.mount("/static", StaticFiles(directory="app/static"), name="static")`

---

### D13 — Programmatic openpyxl + ReportLab generation over templates

**Context**
Stakeholders (agronomists, farm managers) need downloadable reports aggregating farm analytics (status, irrigation schedule, nutrient diagnostics, yield forecasts, alerts) into formatted documents suitable for offline review and sharing.

**Decision**
Generate reports programmatically using **openpyxl** (XLSX) and **ReportLab** (PDF). XLSX includes multi-sheet workbooks with styled headers, conditional formatting on anomaly severity, embedded charts for history series. Reports support both synchronous streaming (with 15-minute Redis cache) and asynchronous job workflow (with 6-hour Redis artifact TTL).

**Consequences**

_Benefits:_
- Full programmatic control over formatting, conditional styling, and chart embedding.
- No template files to maintain — report structure is defined in code alongside the data.
- Dual delivery modes: sync (immediate download) and async (background generation for large reports).
- Redis caching prevents redundant generation for repeated requests.

_Costs:_
- openpyxl API is verbose for complex formatting operations.
- PDF quality is limited by ReportLab's rendering (not pixel-identical to XLSX).
- Report generation is CPU-intensive — async mode prevents blocking request threads but still consumes worker resources.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Jinja2 HTML templates → wkhtmltopdf | Poor table and chart fidelity; HTML-to-PDF conversion introduces layout inconsistencies; no native XLSX output |
| XlsxWriter | Write-only (cannot read existing workbooks); fewer chart options than openpyxl; less maintained |
| Google Sheets API | External dependency; requires Google Cloud credentials; adds network latency and availability risk |
| LaTeX → PDF | High-quality typesetting but steep learning curve; no XLSX output; requires TeX distribution in container |

**Implementation Evidence**
- `app/services/report_service.py` — `ReportService` class: `generate_report_artifact()` (sync generation), `create_report_job()` / `execute_report_job()` (async lifecycle), `_sync_cache_key()` (SHA256-based Redis cache key), constants: `SYNC_CACHE_TTL_SECONDS=900`, `JOB_TTL_SECONDS=21600`, `FILE_TTL_SECONDS=21600`
- `app/routes/analytics.py` — `generate_spreadsheet_report()` (sync streaming), `enqueue_spreadsheet_report()` (async), `get_spreadsheet_report_job_status()`, `download_spreadsheet_report()`
- `app/schemas/reports.py` — `ReportRequest` (fields: `irrigation_horizon_days`, `include_members`, `include_history_charts`)
- `pyproject.toml` — Dependencies: `openpyxl>=3.1.0`, `reportlab>=4.2.0`

---

### D14 — Severity classification enum for anomaly events

**Context**
The original anomaly detector treated all anomalies equally — a marginal sigma-1 deviation received the same visibility as a critical multi-layer-confirmed outlier. Operators need priority triage: critical alerts must surface immediately, while informational anomalies are logged for trend analysis.

**Decision**
Introduce `AnomalySeverityEnum` with three levels: `info`, `warning`, `critical`. Every `AnomalyEvent` record includes a `severity` column. Severity drives: filtered history queries, webhook dispatch criteria, and dashboard alerting priority.

**Consequences**

_Benefits:_
- Enables filtered queries (`GET .../anomalies/history?severity=critical`).
- Webhook subscriptions can filter by severity via `event_types`.
- Operators can focus on critical alerts and batch-review informational ones.
- Clean enumeration — no ambiguity about severity levels.

_Costs:_
- Requires threshold-to-severity mapping logic (what sigma deviation maps to what severity).
- Additional enum migration and schema change.
- Severity assignment is deterministic but may need tuning per deployment.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Numeric score (0–100) | Harder to filter and alert on; no clear cutoff semantics; requires arbitrary binning for display |
| Binary (normal / anomaly) | Insufficient granularity — treats a sensor slightly out of range the same as a multi-layer critical failure |
| Five-level scale (trace/debug/info/warning/critical) | Unnecessary granularity for agricultural anomalies; three levels map cleanly to operator workflows (ignore/investigate/act) |

**Implementation Evidence**
- `app/models/enums.py` — `AnomalySeverityEnum` (StrEnum): `info`, `warning`, `critical`
- `app/models/anomalies.py` — `AnomalyEvent` table (`anomaly_events`): columns include `severity` (Enum), `layer`, `anomaly_type`, `current_value`, `rolling_mean`, `rolling_std`, `sigma_deviation`, `cross_layer_confirmed`, `webhook_notified`; indexes: `ix_anomaly_events_farm_severity`
- `app/schemas/anomalies.py` — `AnomalyHistoryQuery` with `severity` filter parameter
- `app/routes/analytics.py` — `get_anomaly_history()` with severity/layer/vertex/type/time filters
- `alembic/versions/d91a6e7b4c20_add_anomaly_detection_enhancements.py` — Migration creating `anomaly_events` table with severity enum

---

### D15 — Webhook subscriptions with HMAC-SHA256 dispatch

**Context**
Anomaly alerts need to reach external systems (Slack, PagerDuty, custom monitoring dashboards) in real time. Push-based notification (webhooks) is preferred over polling because external systems should not need to poll AgriSense continuously.

**Decision**
Full webhook subscription CRUD with per-subscription secrets. Payloads are signed with **HMAC-SHA256** so receivers can verify authenticity. Dispatch is asynchronous via a Redis queue (`anomaly:webhook:dispatch`) consumed by a background worker task started in the API lifespan. Failed deliveries are retried up to `retry_max` (default 3) per subscription. A test endpoint fires a synthetic delivery for connectivity verification.

**Consequences**

_Benefits:_
- Secure: HMAC-SHA256 signing with per-subscription secrets prevents payload forgery.
- Non-blocking: async dispatch queue decouples anomaly detection from delivery latency.
- Self-service: operators manage their own webhook subscriptions via CRUD endpoints.
- Testable: `/test` endpoint verifies connectivity before enabling production alerts.

_Costs:_
- Background worker adds operational complexity (must be started/stopped in lifespan).
- HMAC secret management is per-subscription (operators must store their secrets securely).
- Retry logic can cause delayed delivery on transient failures.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Email notifications | Requires SMTP infrastructure; email is too slow for real-time anomaly alerting; delivery reliability is poor |
| Server-Sent Events (SSE) only | Push-based but requires persistent connection from external system; no push to systems that expose an HTTP endpoint (Slack, PagerDuty) |
| AWS SNS/SQS | Cloud vendor lock-in; adds AWS credential management; AgriSense is designed to be self-hosted |
| Unsigned webhooks | Receivers cannot verify payload authenticity; opens risk of spoofed alerts |

**Implementation Evidence**
- `app/services/webhook_service.py` — `WebhookService` class: `create_subscription()`, `list_subscriptions()`, `update_subscription()`, `delete_subscription()`, `test_subscription()`; module-level: `run_dispatch_queue_worker(redis, stop_event)` (background task), `DISPATCH_QUEUE_KEY = "anomaly:webhook:dispatch"`, HMAC signing via `hmac` + `hashlib` with per-subscription `secret`
- `app/routes/anomalies.py` — 5 webhook endpoints: list, create, update, delete, test
- `app/models/anomalies.py` — `WebhookSubscription` table (`webhook_subscriptions`): columns `url`, `secret` (String(256)), `event_types` (JSONB), `is_active`, `retry_max` (default 3), `failure_count`, `last_triggered_at`, `last_status_code`; index: `ix_webhook_subscriptions_farm_active`
- `app/main.py` — Lifespan: starts `run_dispatch_queue_worker` as `asyncio.create_task`; shutdown signals stop event and awaits/cancels worker

---

### D16 — Redis conversation memory over database persistence

**Context**
The LangChain agent needs multi-turn conversation context to maintain coherent dialogue across questions. Conversation history is accessed on every request (read recent messages), appended to on every response (write new messages), and expires naturally (users don't return after a session). The access pattern is high-frequency read/write with natural expiry — a poor fit for durable database storage.

**Decision**
Store conversation history in **Redis** using `RedisChatMessageHistory` from `langchain-community`. Keys are scoped per-user-per-farm (`conversation:{farm_id}:{user_id}`). Window is bounded to 20 messages (`langchain_max_context_messages`). TTL is 1 hour (`langchain_conversation_ttl_seconds`). Falls back to `InMemoryChatMessageHistory` when Redis is unavailable. Explicit clear endpoint: `DELETE /api/v1/ask/{farm_id}/conversation`.

**Consequences**

_Benefits:_
- Sub-millisecond read/write for conversation operations.
- Automatic expiry via Redis TTL — no garbage collection needed.
- Per-farm isolation prevents conversation cross-contamination.
- No database migration required for conversation storage.
- Graceful degradation: in-memory fallback when Redis is unavailable.

_Costs:_
- Volatile — conversation history is lost on Redis restart.
- TTL tuning may be needed per deployment (1 hour may be too short or too long).
- Memory pressure at scale if many concurrent users maintain active conversations.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| PostgreSQL conversation table | Slower for the frequent read/write pattern (every request reads history, every response appends); requires migration; unnecessary durability for ephemeral chat |
| In-memory Python dict | Lost on API restart; no shared state across multiple workers; doesn't scale beyond single process |
| SQLite sidecar | No shared state in multi-container deployments; file locking issues with async; adds filesystem dependency |
| No conversation memory | Each question is stateless; users cannot ask follow-up questions that reference prior context; poor UX for exploratory data analysis |

**Implementation Evidence**
- `app/services/conversation_memory.py` — `resolve_conversation_id(farm_id, user_id)` (key pattern: `conversation:{farm_id}:{user_id}`), `build_memory()` (returns `RedisChatMessageHistory` with TTL or `InMemoryChatMessageHistory` fallback), `get_window_messages(chat_history, max_messages)` (bounded window), `refresh_ttl()`, `clear_conversation()`; constant: `REDIS_HISTORY_KEY_PREFIX = "message_store:"`
- `app/config.py` — `langchain_conversation_ttl_seconds` (default 3600), `langchain_max_context_messages` (default 20)
- `app/routes/ask.py` — `clear_farm_conversation()` (DELETE endpoint)
- `app/services/llm_service.py` — `LLMService.ask()` and `ask_stream()` integrate conversation memory; `clear_conversation()` delegates to `conversation_memory.clear_conversation()`

---

### D17 — Per-sensor-type configurable thresholds

**Context**
The original anomaly detector used global sigma thresholds for all sensor types. A soil moisture sensor and a temperature sensor have fundamentally different normal ranges and variability profiles — a 2-sigma deviation in soil EC is routine, while a 2-sigma deviation in greenhouse temperature may be critical. Global thresholds either over-alert on noisy sensors or under-alert on stable ones.

**Decision**
Introduce `anomaly_thresholds` table with per-farm, per-vertex-type, per-layer threshold configuration. Operators configure sigma levels (`sigma1`, `sigma2`, `sigma3`), minimum history length (`min_history`), NaN-run outage detection (`min_nan_run_outage`), vision anomaly score threshold, and rule suppression (`suppress_rule3_only`). Thresholds are managed via CRUD endpoints and consumed by the anomaly detection pipeline. Unique constraint enforces one threshold config per `(farm_id, vertex_type, layer)` tuple.

**Consequences**

_Benefits:_
- Domain-specific sensitivity — soil sensors can have different thresholds than weather stations.
- Operator-tunable without code changes or redeployment.
- Per-farm customization — different farms can have different threshold profiles.
- Unique constraint prevents conflicting threshold configurations.

_Costs:_
- Additional CRUD surface (4 endpoints) to maintain.
- Default thresholds must be sensible when no custom config exists — if defaults are too permissive, anomalies are missed.
- Threshold proliferation at scale (many farms × many vertex types × many layers) may become hard to manage.

**Rejected Alternatives**

| Alternative | Reason for Rejection |
|---|---|
| Global config file thresholds | No per-farm tuning; requires redeployment to change; cannot customize per vertex type |
| ML-learned thresholds (auto-tuning) | Cold-start problem with limited history; interpretability concerns — operators cannot understand or override model-determined thresholds |
| Percentile-based auto-thresholds | Unstable with limited history; seasonal variation causes threshold drift; no operator control |
| Single sigma parameter per farm | Insufficient granularity — different sensor types have different noise profiles |

**Implementation Evidence**
- `app/models/anomalies.py` — `AnomalyThreshold` table (`anomaly_thresholds`): columns `farm_id`, `vertex_type` (VertexTypeEnum), `layer` (HyperEdgeLayerEnum, nullable), `sigma1` (default 1.0), `sigma2` (default 2.0), `sigma3` (default 3.0), `min_history` (default 8), `min_nan_run_outage` (default 4), `vision_anomaly_score_threshold` (default 0.7), `suppress_rule3_only` (default True), `enabled` (default True); unique constraint: `uq_anomaly_thresholds_farm_vertex_layer`
- `app/services/anomaly_service.py` — `AnomalyService`: `get_thresholds()`, `create_threshold()`, `update_threshold()`, `delete_threshold()`
- `app/routes/anomalies.py` — 4 threshold endpoints: list, create, update, delete
- `app/services/julia_bridge.py` — `detect_anomalies(farm_id, thresholds)` — passes threshold configuration to Julia for sigma-map construction
- `core/AgriSenseCore/src/models/anomaly.jl` — `western_electric_kernel!` uses configurable `sigma1_map`, `sigma2_map`, `sigma3_map` arrays constructed from threshold configuration
- `alembic/versions/d91a6e7b4c20_add_anomaly_detection_enhancements.py` — Migration creating `anomaly_thresholds` table with unique constraint

---

## 5) Non-Functional Targets

- **Latency**: Sub-second API responses for standard analytics endpoints; NL query bounded by LLM response time (configurable `anthropic_timeout_seconds: 15s`).
- **Caching**: Sync report cache at 15-minute TTL; graph cache refreshed on ingest; conversation memory at 1-hour TTL.
- **Reliability**: Webhook dispatch with configurable retry (default 3 attempts); async job state persisted in Redis with 6-hour TTL.
- **Startup**: Julia warm-up + optional graph cache bootstrap; bounded by farm count at demo scale.
- **Type Safety**: Pydantic v2 for all request/response schemas; TypedDict contracts for Julia boundary; `mypy` strict mode in CI.
- **Health**: Liveness (`/health`) and readiness (`/health/ready`) with dependency checks for PostgreSQL, Redis, and Julia bridge.

---

## 6) Security and Governance Constraints

- **Authentication**: Dual-mode — JWT bearer tokens (RBAC: `admin`, `agronomist`, `field_operator`, `readonly`) and API keys (scopes: `ingest`, `jobs`). API key validation via SHA-256 digest (fast path) or bcrypt (legacy fallback).
- **Authorization**: Role-based endpoint restrictions (e.g., report generation requires `admin` or `agronomist`). Machine-scope enforcement for ingest and job endpoints.
- **Rate Limiting**: Per-user (100/min) and per-API-key (1000/min) via Redis atomic counters with sliding window.
- **Webhook Security**: HMAC-SHA256 payload signing with per-subscription secrets. Minimum secret length enforced (16 characters) at creation.
- **Data Governance**: Public/demo datasets are synthetic only. Private client data is excluded from the repository and CI.
- **CORS**: Configurable allowed origins; credentials only enabled when origins are not wildcard.

---

## 7) Accepted Tradeoffs

| Tradeoff | Accepted Because |
|---|---|
| Julia runtime adds startup latency (JIT warm-up) | In-process bridge eliminates per-request network overhead; one-time startup cost is acceptable for long-running API process |
| LangChain dependency surface is large | Declarative tool binding + memory abstraction + streaming support justify the transitive dependency cost; alternatives require more custom code |
| Redis conversation memory is volatile | Conversation history is ephemeral by nature; 1-hour TTL aligns with session duration; loss on restart is acceptable — users restart conversations naturally |
| openpyxl API verbosity for report formatting | Full programmatic control over styling, conditional formatting, and charts; no template files to maintain; verbosity is contained in a single service module |
| CI tests run on CPU only (GPU lane is manual) | GPU runners are scarce and expensive; KernelAbstractions guarantees identical correctness on CPU; GPU lane validates performance, not correctness |
| Separate compute runtime (Julia) requires two-ecosystem expertise | Julia's sparse matrix, GPU, and mathematical ecosystem is unmatched for the workload; the bridge contract is narrow and explicitly typed |
| Anomaly threshold defaults may miss domain-specific patterns | Operators can tune thresholds per sensor type; conservative defaults (sigma3=3.0) minimize false positives; domain adaptation is an operational concern, not an architecture concern |

---

## 8) Evolution Rules

Future changes must conform to these constraints:

1. **Preserve the Python/Julia ownership boundary.** Python owns orchestration; Julia owns compute. Do not introduce numerical compute in Python or web routing in Julia.
2. **Keep the bridge contract explicit and typed.** All new Julia functions exposed to Python must have corresponding TypedDict contracts in `julia_contracts.py` and validators in `julia_validators.py`.
3. **Do not weaken blocking quality gates.** New CI jobs may be added, but existing gates (lint, type check, test, build) must remain blocking.
4. **Do not bypass migration/readiness sequencing.** New dependencies must be included in readiness checks. New tables require Alembic migrations.
5. **Keep demo data synthetic and reproducible.** Seed data must be deterministically generated from explicit seeds. Never commit real client data.
6. **Maintain webhook dispatch idempotency.** Dispatch worker must handle duplicate queue entries safely. Subscribers must be able to deduplicate by event ID.
7. **Keep LangChain tool definitions synchronized with analytics capabilities.** If a new analytics endpoint is added, evaluate whether it should be exposed as an agent tool. Feature flags control gradual rollout.
8. **Preserve anomaly severity enum backward compatibility.** New severity levels may be appended but existing levels (`info`, `warning`, `critical`) must not be renamed or removed — webhook subscribers depend on stable string values.
9. **Redis key namespacing must be maintained.** New Redis usage lanes must use distinct key prefixes to prevent collisions (e.g., `ratelimit:`, `message_store:`, `anomaly:webhook:`, `report:`, `graph:`).

---

## 9) Supersession Policy

This document supersedes the prior `ADR.md` (dated 2026-03-02). Any future architectural deviation requires updating this record with explicit rationale, impact analysis, rejected alternatives, and a migration plan. Changes to this document must be reviewed by core platform maintainers.
