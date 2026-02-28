

   

Read [](file:///workspace/agrisense/AGENT/PRD.md#1-1), lines 1 to 200

Read [](file:///workspace/agrisense/AGENT/ARCHITECTURE.md#1-1), lines 1 to 200

Read [](file:///workspace/agrisense/AGENT/PRD.md#200-200), lines 200 to 289

Read [](file:///workspace/agrisense/AGENT/ARCHITECTURE.md#200-200), lines 200 to 400

Read [](file:///workspace/agrisense/AGENT/ARCHITECTURE.md#400-400), lines 400 to 559

Now let me check the current state of the workspace to understand what already exists. 

Read [](file:///workspace/agrisense)

I've completed a thorough review of both documents. Before drafting the roadmap, I need to resolve a key architectural tension. 

Asked 3 questions (GPU Runtime, GPU Deploy, Repo Host)

Research Julia GPU ecosystem

Read [](file:///workspace/agrisense/AGENT/ARCHITECTURE.md#1-1), lines 1 to 200

Read [](file:///workspace/agrisense/AGENT/PRD.md#1-1), lines 1 to 200

Read [](file:///workspace/agrisense/AGENT/PRD.md#200-200), lines 200 to 289

Read [](file:///workspace/agrisense/AGENT/ARCHITECTURE.md#200-200), lines 200 to 559

Read [](file:///root/.vscode-server/data/User/workspaceStorage/f75f4313f80b21fab0607c05ba789a1a/GitHub.copilot-chat/chat-session-resources/32ddf05d-bd24-45d4-a6a7-7cb6590f4108/toolu_bdrk_012VF79okUQbNCbzwcpRmpmw__vscode-1772282945402/content.txt#1-1), lines 1 to 200

Read [](file:///root/.vscode-server/data/User/workspaceStorage/f75f4313f80b21fab0607c05ba789a1a/GitHub.copilot-chat/chat-session-resources/32ddf05d-bd24-45d4-a6a7-7cb6590f4108/toolu_bdrk_012VF79okUQbNCbzwcpRmpmw__vscode-1772282945402/content.txt#200-200), lines 200 to 623

I now have comprehensive context from both documents and the GPU ecosystem research. Here is the draft implementation roadmap.

---

## Plan: AgriSense GPU-Centric Implementation Roadmap

**TL;DR:** Build a FastAPI backend with a Julia GPU-accelerated computational core, modeling farms as layered hypergraphs via sparse incidence matrices on CUDA. The roadmap spans 10 phases — from repo scaffolding and CI to deployment — prioritizing the GPU-native hypergraph engine early, then layering on ingestion, analytics, LLM, and observability. The core architectural bet: represent all 7 hypergraph layers as sparse matrices and express cross-layer queries as SpMV/SpMM, making GPU acceleration natural via CUSPARSE/CUBLAS. KernelAbstractions.jl provides CPU fallback for CI and non-GPU hosts.

---

### Phase 0 — Repository Setup & Workflow Scaffolding

1. Initialize the GitHub repo with a `main` branch, branch protection (require PR + CI pass), and conventional commit enforcement
2. Create the full directory skeleton matching ARCHITECTURE.md:
   - `app/` (Python: routes, services, models, schemas, auth, middleware)
   - `core/AgriSenseCore/` (Julia package: `Project.toml`, `src/`, `test/`)
   - `alembic/`, `scripts/`, `tests/`, `.github/workflows/`
3. Bootstrap ARCHITECTURE.md with all env vars (`DATABASE_URL`, `REDIS_URL`, `ANTHROPIC_API_KEY`, `JWT_SECRET`, `LOG_LEVEL`, `FARM_DEFAULT_TYPE`)
4. Create `pyproject.toml` with dependencies: `fastapi`, `uvicorn`, `sqlalchemy[asyncio]`, `asyncpg`, `alembic`, `pydantic-settings`, `redis`, `httpx`, `python-jose[cryptography]`, `juliacall`, `ruff`, `mypy`, `pytest`, `pytest-asyncio`
5. Create `core/AgriSenseCore/Project.toml` with the GPU dependency set:
   - **Core:** `CUDA.jl`, `KernelAbstractions.jl`, `Adapt.jl`
   - **Data:** `SparseArrays` (stdlib), `LinearAlgebra` (stdlib), `StaticArrays.jl`, `StructArrays.jl`
   - **Optional:** `Tullio.jl` (batched tensor ops)
   - **Compat:** `julia = "1.10"`, `CUDA = "5"`, `KernelAbstractions = "0.9"`
6. Set up `.github/workflows/ci.yml`:
   - **Lint job:** `ruff check` + `mypy --strict app/`
   - **Python test job:** `pytest tests/ -x --asyncio-mode=auto` (uses test PostgreSQL via Docker service container)
   - **Julia test job:** `julia --project=core/AgriSenseCore -e 'using Pkg; Pkg.test()'` on CPU backend (no GPU in CI)
   - **Docker build job:** build the multi-stage `Dockerfile`, validate it starts
7. Create `Dockerfile` (multi-stage): base `nvidia/cuda:12.x-runtime-ubuntu22.04`, install Python 3.12 + Julia 1.11, precompile Julia packages at build time (separate layer for cache), copy app code, set entrypoint to `uvicorn`
8. Create `docker-compose.yml` with 4 services per ARCHITECTURE.md: `api`, `postgres` (16 + PostGIS), `redis` (7), `seed` (one-shot)
9. Add a `Makefile` or `justfile` with common commands: `dev`, `test`, `lint`, `seed`, `migrate`, `docker-up`

---

### Phase 1 — Database Schema & Migrations

1. Define SQLAlchemy 2.0 async ORM models in ARCHITECTURE.md matching the schema from ARCHITECTURE.md:
   - `Farm` (UUID pk, name, `farm_type` enum, PostGIS geography point, timezone)
   - `Zone` (UUID pk, FK→Farm, name, `zone_type` enum, PostGIS polygon boundary, area_m2, soil_type, JSONB metadata)
   - `Vertex` (UUID pk, FK→Farm, FK→Zone nullable, `vertex_type` enum, JSONB config, installed_at, last_seen_at)
   - `HyperEdge` (UUID pk, FK→Farm, `layer` enum, UUID[] vertex_ids, JSONB metadata)
2. Define time-series models in ARCHITECTURE.md matching ARCHITECTURE.md:
   - `SoilReading`, `WeatherReading`, `IrrigationEvent`, `NpkSample`, `VisionEvent`, `LightingReading`
   - All indexed on `(sensor_id/station_id, timestamp DESC)`
3. Define `CropProfile` in ARCHITECTURE.md matching ARCHITECTURE.md
4. Define auth models (`User`, `APIKey`) in ARCHITECTURE.md
5. Configure Alembic for async PostgreSQL. Generate initial migration covering all tables + PostGIS extension creation + enum types
6. Write a Docker init script for PostgreSQL that creates the database and enables `postgis` and `uuid-ossp` extensions

---

### Phase 2 — Julia GPU Core: Hypergraph Engine

This is the centrepiece. All hypergraph state is sparse-matrix-based, GPU-portable.

1. Implement `core/AgriSenseCore/src/types.jl` — define the core types:
   - `FarmProfile` struct (farm_type symbol, active_layers set, zones vector, model config) — as in ARCHITECTURE.md
   - `HyperGraphLayer` struct holding: sparse incidence matrix (`AbstractSparseMatrix{Float32}`), vertex feature matrix (`AbstractMatrix{Float32}`), edge metadata vector
   - `LayeredHyperGraph` struct: `n_vertices::Int`, `layers::Dict{Symbol, HyperGraphLayer}`, per-farm metadata
   - Register all structs with `Adapt.@adapt_structure` so `adapt(CuArray, graph)` moves the entire structure to GPU (custom `adapt` rule for sparse → `CuSparseMatrixCSR` conversion)
2. Implement `core/AgriSenseCore/src/hypergraph.jl` — build, query, update:
   - `build_graph(farm_config::Dict)` — construct per-layer incidence matrices from vertex/edge definitions, store as `SparseMatrixCSC` on CPU
   - `to_gpu(graph::LayeredHyperGraph)` — uses Adapt.jl to move to GPU; returns GPU-resident version with `CuSparseMatrixCSR` incidence and `CuMatrix` features
   - `query_layer(graph, layer::Symbol, zone_id)` — extract subgraph for a zone via sparse column slicing
   - `cross_layer_query(graph, layer_a, layer_b)` — compute `B_a' * B_b` (SpMM) to get inter-layer connectivity matrix, then extract connected edges
   - `update_vertex_features!(graph, layer, vertex_id, new_data)` — incremental update of a single vertex's feature row
   - `add_hyperedge!(graph, layer, vertex_ids)` — add a column to the incidence matrix (rebuild sparse structure)
3. Implement backend selection in a utility module:
   - `get_backend()` returns `CUDABackend()` if `CUDA.functional()`, else `CPU()`
   - `get_array_type()` returns `CuArray` or `Array` accordingly
   - Conditional CUDA import: `const HAS_CUDA = try; using CUDA; CUDA.functional(); catch; false; end`
4. Write Julia tests in `core/AgriSenseCore/test/test_hypergraph.jl`:
   - Test build from config dict → correct incidence matrix shapes
   - Test cross-layer query returns correct vertex overlap
   - Test incremental update modifies features correctly
   - All tests run on CPU backend (CI-compatible)

---

### Phase 3 — Julia GPU Core: Predictive Models

1. Implement `core/AgriSenseCore/src/models/irrigation.jl` — **Water Balance Scheduler**:
   - KernelAbstractions `@kernel` for the water balance equation: `moisture_tomorrow = current - ET₀ × crop_coefficient + rainfall_forecast + irrigation_applied`
   - Batch across all zones simultaneously (matrix operation on GPU)
   - Threshold triggers: if projected moisture < wilting point → recommend irrigation
   - Output: per-zone `Dict` with timing, volume (liters), priority
   - Uses `StaticArrays.SVector{4, Float32}` inside kernels for the 4-variable balance
2. Implement `core/AgriSenseCore/src/models/nutrients.jl` — **NPK Deficit Scoring**:
   - KernelAbstractions `@kernel` comparing current NPK `SVector{3}` against crop stage requirements
   - Severity scoring: deficit magnitude × growth stage sensitivity weight
   - Visual confirmation boost: if CV layer has `nutrient_deficiency` flag → multiply priority by 2.0
   - Output: per-zone alert `Dict` with deficit magnitudes, suggested amendment, urgency
3. Implement `core/AgriSenseCore/src/models/yield.jl` — **Yield Regression Forecaster**:
   - Feature assembly on GPU: cumulative DLI, soil health score (rolling mean moisture stability), growth stage progress fraction, canopy coverage (from CV layer)
   - Dense least-squares regression on GPU: `X \ y` dispatches to CUBLAS on `CuArray`
   - Train on synthetic historical data; predict per crop_bed with confidence interval (residual std)
   - Output: per-bed yield estimate + CI bounds
4. Implement `core/AgriSenseCore/src/models/anomaly.jl` — **Statistical Process Control**:
   - Rolling mean and standard deviation on GPU via `cumsum` + broadcasts
   - Western Electric / Nelson rules approximation: detect consecutive points beyond 2σ, trending runs
   - Applied to soil moisture, weather readings, irrigation flow rates
   - Output: list of anomaly alerts with sensor_id, metric, severity, timestamp range
5. Write Julia tests for each model in `core/AgriSenseCore/test/`:
   - `test_irrigation.jl` — known input → expected schedule
   - `test_nutrients.jl` — deficit calculations match manual
   - `test_yield.jl` — regression on synthetic data produces reasonable R²
   - `test_anomaly.jl` — injected anomalies detected, clean data produces no false alerts

---

### Phase 4 — Julia GPU Core: Synthetic Data Generator

1. Implement `core/AgriSenseCore/src/synthetic/generator.jl` — master dispatcher:
   - Takes `farm_type`, `days`, `seed` → dispatches to per-layer generators
   - Returns a `Dict` with all layer data as CPU arrays (for bridge crossing to Python)
   - Uses `StructArrays` for SoA layout of sensor batches during generation
2. Implement per-layer generators, all GPU-accelerated:
   - ARCHITECTURE.md — Exponential moisture decay between irrigation events, sharp rises on irrigation/rainfall, diurnal temperature coupling via sinusoidal broadcast on `CuArray` time grid, ~3% random dropout (`CUDA.rand .< 0.03`)
   - ARCHITECTURE.md — Sinusoidal diurnal temperature + normal perturbation, seasonal rainfall probability via Bernoulli (`CUDA.rand`), correlated humidity via Cholesky factor × independent samples (CUBLAS matmul)
   - ARCHITECTURE.md — Slow linear drift + periodic fertilization step-changes via masking
   - ARCHITECTURE.md — Stochastic anomaly events: pest at ~2%, disease at ~0.5%, spatial clustering via sparse neighbor multiply on bed adjacency matrix
   - ARCHITECTURE.md — Cross-layer correlation injection: build correlation matrix between all sensor channels, Cholesky factorize on GPU, multiply with independent samples
3. Implement `core/AgriSenseCore/src/bridge.jl` — the Python-facing contract:
   - `build_graph(farm_config::Dict)::Dict` — build hypergraph, serialize to CPU dicts
   - `query_farm_status(graph_state::Dict, zone_id::String)::Dict` — deserialize → GPU → compute → CPU → Dict
   - `irrigation_schedule(graph_state::Dict, horizon_days::Int)::Vector{Dict}`
   - `nutrient_report(graph_state::Dict)::Vector{Dict}`
   - `yield_forecast(graph_state::Dict)::Vector{Dict}`
   - `detect_anomalies(graph_state::Dict)::Vector{Dict}`
   - `generate_synthetic(; farm_type::String, days::Int, seed::Int)::Dict`
   - **Rule:** all inputs/outputs are plain `Dict`, `Vector`, `Array` — never `CuArray`; GPU is internal
4. Write tests in `test/test_synthetic.jl`:
   - Verify 90-day generation produces correct row counts per layer
   - Verify moisture values are in [0, 1] range
   - Verify reproducibility: same seed → same output
   - Verify dropout: ~3% `NaN` or missing values

---

### Phase 5 — Python Layer: Configuration, Bridge & Core CRUD

1. Implement ARCHITECTURE.md — pydantic-settings loading from env vars
2. Implement ARCHITECTURE.md — Python side of the Julia bridge:
   - Initialize `juliacall` at startup via FastAPI lifespan handler (`@asynccontextmanager`)
   - `jl.seval("using AgriSenseCore")` once; cache the module reference
   - Wrapper functions mapping to each `bridge.jl` export; handle serialization of UUIDs (→ strings), datetimes (→ ISO strings), numpy arrays (→ lists) before crossing
   - Error handling: catch Julia exceptions, wrap as Python `RuntimeError` with structured logging
3. Implement ARCHITECTURE.md — FastAPI app with lifespan:
   - Initialize Julia runtime + precompile GPU kernels (first dummy call to warm up)
   - Initialize async SQLAlchemy engine + Redis connection pool
   - Build in-memory hypergraph state from DB for each farm (or lazily on first access)
   - Register all routers, middleware
4. Implement farm CRUD service + routes:
   - ARCHITECTURE.md — create farm (validates `farm_type` enum → sets active layers), get farm, list farms, add zone, register vertex (sensor/valve/crop_bed/camera)
   - ARCHITECTURE.md — Pydantic request/response models for all farm objects
   - ARCHITECTURE.md — routes matching PRD.md
5. Write Python tests in `tests/test_farms.py`:
   - CRUD operations, validation errors, farm_type propagation
   - Use `httpx.AsyncClient` with test database

---

### Phase 6 — Data Ingestion Pipeline

1. Implement ARCHITECTURE.md — Pydantic schemas for each ingestion payload type (soil batch, weather batch, irrigation event, NPK sample, CV inference result, bulk multi-layer)
2. Implement ARCHITECTURE.md:
   - Validate incoming data against schema + farm topology (sensor must belong to farm, layer must be active)
   - Batch insert into appropriate time-series table via SQLAlchemy async
   - Update in-memory hypergraph vertex features via Julia bridge call (`update_vertex_features!`)
   - Publish event to Redis pub/sub for WebSocket live feed
   - Return ingestion receipt (count inserted, any validation warnings)
3. Implement ARCHITECTURE.md — all 6 ingestion endpoints from PRD.md
4. Write tests in `tests/test_ingest.py`:
   - Valid batch ingestion for each layer
   - Validation rejection (wrong sensor type, inactive layer for farm type)
   - Bulk ingestion spanning multiple layers

---

### Phase 7 — Analytics, Predictions & Alerts

1. Implement ARCHITECTURE.md:
   - `get_farm_status(farm_id)` — calls Julia `query_farm_status` for each zone, aggregates
   - `get_zone_detail(farm_id, zone_id)` — cross-layer view via Julia `cross_layer_query`
   - `get_irrigation_schedule(farm_id)` — calls Julia `irrigation_schedule`, caches result in Redis (TTL 15 min)
   - `get_nutrient_report(farm_id)` — calls Julia `nutrient_report`
   - `get_yield_forecast(farm_id)` — calls Julia `yield_forecast`
   - `get_active_alerts(farm_id)` — calls Julia `detect_anomalies`, merges with NPK alerts and CV alerts
2. Implement ARCHITECTURE.md — response schemas for each analytics endpoint
3. Implement ARCHITECTURE.md — all 6 analytics endpoints from PRD.md
4. Implement background job endpoints:
   - `POST /api/v1/jobs/{farm_id}/recompute` — triggers full hypergraph rebuild in background task
   - `GET /api/v1/jobs/{job_id}/status` — reads job status from Redis
5. Write tests in `tests/test_analytics.py`:
   - Seed DB with known data → verify analytics responses match expected
   - Mock Julia bridge for unit tests (integration tests call real bridge)

---

### Phase 8 — Auth, LLM Interface & WebSocket

1. Implement auth in ARCHITECTURE.md:
   - `jwt.py` — JWT access + refresh token creation/validation using `python-jose`
   - `dependencies.py` — `get_current_user` dependency (decode JWT, load user), `require_role` dependency (RBAC check against `admin`, `agronomist`, `field_operator`, `readonly`)
   - API key auth for machine-to-machine (irrigation controllers)
   - Rate limiting middleware in ARCHITECTURE.md — Redis-backed, 100 req/min standard / 1000 req/min API keys
2. Implement LLM service in ARCHITECTURE.md:
   - **Intent classifier** — rule-based first pass (keyword matching for irrigation/nutrient/yield/status), LLM fallback for ambiguous queries
   - **Context assembler** — given intent → determine relevant layers + target zones → call Julia bridge for current state → format as structured context block for LLM
   - **LLM call** — system prompt enforcing grounded responses (no hallucination, respond in requested language, simple language), user message = question + context. Use `httpx` async to Anthropic Claude API
   - **Response parser** — extract answer text, action recommendation, source data references
   - Multilingual support: Arabic, French, English — language parameter in request
3. Implement ARCHITECTURE.md — `POST /api/v1/ask/{farm_id}` per PRD.md
4. Implement WebSocket in ARCHITECTURE.md:
   - `WS /ws/{farm_id}/live` — authenticates via token query param, subscribes to Redis pub/sub channel for farm, forwards sensor updates + alerts as JSON frames
5. Write tests:
   - `tests/test_auth.py` — JWT flow, RBAC enforcement, API key auth, rate limit hit
   - `tests/test_ask.py` — mock LLM responses, verify context assembly, verify source attribution
   - `tests/test_websocket.py` — connect, receive published event, disconnect

---

### Phase 9 — Seed Script, Observability & Documentation

1. Implement ARCHITECTURE.md:
   - Call Julia `generate_synthetic(farm_type="greenhouse", days=90, seed=42)` for the demo farm (2 greenhouses + 4 open field zones)
   - Insert farm topology first (farm, zones, vertices, hyperedges)
   - Bulk insert all time-series data (90 days × appropriate frequencies)
   - Insert crop profiles from agronomic reference tables
   - Target: completes in < 30 seconds per PRD.md
2. Implement ARCHITECTURE.md — curl examples demonstrating all major endpoints
3. Add structured JSON logging in ARCHITECTURE.md:
   - Request ID propagation (UUID per request in header + log context)
   - Request/response timing
   - Julia bridge call timing (track GPU vs CPU path)
4. Add health check endpoint at `/health` — verifies DB connectivity, Redis connectivity, Julia runtime alive
5. Write `README.md`:
   - One-paragraph description mentioning hypergraph modeling, GPU-accelerated Julia core, real-world agricultural context
   - Mermaid architecture diagram showing layered hypergraph + data flow
   - Synthetic data disclaimer per PRD.md
   - Quickstart: `docker-compose up` → seeded and running
   - Example API calls from `demo_queries.sh`
   - GPU setup notes (NVIDIA driver requirements, CPU fallback behavior)
   - Link to live deployed instance

---

### Phase 10 — Docker, Deployment & Final Polish

1. Finalize `Dockerfile`:
   - Multi-stage build: Julia precompilation stage (caches compiled packages), Python dependency stage, final runtime stage
   - GPU support: `nvidia/cuda:12.x` base for GPU, with `--gpus all` Docker runtime flag
   - CPU fallback: same image works without GPU (KernelAbstractions detects and falls back)
   - Precompile Julia on build: `julia --project=core/AgriSenseCore -e 'using Pkg; Pkg.precompile()'`
2. Finalize `docker-compose.yml`:
   - `api` service with `deploy.resources.reservations.devices` for GPU pass-through (optional)
   - `postgres` with health check and init script
   - `redis` with health check
   - `seed` with `depends_on: [api, postgres]`, runs once then exits
3. Deploy to GPU-enabled host (RunPod Serverless, Lambda Cloud, or GPU-enabled Fly.io):
   - Configure env vars
   - Verify GPU detection after deploy (`/health` endpoint reports GPU status)
   - Verify seed data loads correctly
   - Test all endpoints against live deployment
4. Final CI pipeline validation:
   - All 3 jobs pass (lint, Python tests on CPU, Julia tests on CPU)
   - Docker build succeeds
   - Add badge to README
5. Create a `CLAUDE.md` or contributing guide documenting:
   - GPU development setup (NVIDIA driver, CUDA toolkit)
   - How the Julia bridge works
   - How to add a new hypergraph layer
   - How to add a new predictive model

---

### Verification

- **Unit:** `pytest tests/ -x` — all Python tests pass (mocked Julia bridge for fast feedback)
- **Integration:** `pytest tests/ --integration` — real Julia bridge calls on CPU backend
- **Julia:** `julia --project=core/AgriSenseCore -e 'using Pkg; Pkg.test()'` — all hypergraph + model + synthetic tests pass
- **Docker:** `docker-compose up --build` → seed completes → `curl localhost:8000/health` returns OK with GPU status
- **Smoke:** run `scripts/demo_queries.sh` against local instance — all endpoints return valid responses
- **GPU verification** (if GPU available): Julia logs show `CUDABackend()` selected, bridge calls complete faster than CPU baseline
- **Lint/Type:** `ruff check app/` clean, `mypy --strict app/` clean

### Decisions

- **Julia + CUDA.jl over Python GPU stack:** keeps the Architecture doc's language boundary intact; Julia's multiple dispatch makes the CPU/GPU portability cleaner than Python CuPy/PyTorch; CUSPARSE is first-class in Julia
- **Custom sparse-matrix hypergraph over SimpleHypergraphs.jl:** existing Julia hypergraph libraries have no GPU support and use set-based representations incompatible with GPU; sparse incidence matrices enable SpMV/SpMM cross-layer queries that map directly to CUSPARSE
- **KernelAbstractions.jl as primary kernel interface:** ensures all custom compute works on both CPU and GPU with zero code branching; CUDA.jl is just the backend provider
- **Bridge always returns CPU data:** `CuArray` cannot cross the juliacall boundary cleanly; GPU stays internal to Julia, serialization boundary is always plain dicts/arrays
- **Incidence matrix representation (`B ∈ {0,1}^{|V|×|E|}`):** cross-layer queries become `B_a' * B_b` (one SpMM call), vertex feature aggregation is `B' * X` — both are the most optimized operations on GPU
- **StructArrays for sensor batches:** SoA layout gives coalesced GPU memory access; column-major `(n_timesteps, n_sensors)` matrices keep each sensor's time series contiguous