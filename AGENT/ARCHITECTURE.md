

---

## AgriSense — Architecture Document

**Version:** 1.0
**Date:** February 28, 2026

---

### 1. System Context

AgriSense is a backend API service that models agricultural environments as layered hypergraphs for intelligent farm management. The system supports two deployment configurations:

**Open Field Mode:** Layers 1–6 active (soil, irrigation, solar, weather, crop requirements, NPK). No computer vision layer. Weather station is primary environmental data source. Zones are GPS-bounded polygons.

**Greenhouse Mode:** All 7 layers active including computer vision. Artificial lighting layer replaces/supplements solar. Climate is partially controlled (ambient humidity, temperature can be actuated, not just observed). Camera vertices are added to the graph. Zones are structurally bounded (bay, bench, row).

The mode is set per-farm at creation time via a `farm_type` enum (`open_field` | `greenhouse` | `hybrid`) and determines which layers are instantiated, which endpoints are available, and which predictive models run.

---

### 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                               │
│  (curl, frontend, irrigation controller, mobile app)         │
└──────────────┬──────────────────────┬───────────────────────┘
               │ HTTPS/REST           │ WebSocket
               ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Python API Layer                           │
│                      (FastAPI)                                │
│                                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐   │
│  │  Auth    │ │  CRUD    │ │ Ingest   │ │  Analytics    │   │
│  │  (JWT,   │ │  (farms, │ │ (soil,   │ │  (status,     │   │
│  │  RBAC,   │ │  zones,  │ │  weather,│ │  schedule,    │   │
│  │  API key)│ │  sensors)│ │  NPK,    │ │  forecast,    │   │
│  │          │ │          │ │  vision) │ │  alerts)      │   │
│  └──────────┘ └──────────┘ └──────────┘ └───────┬───────┘   │
│                                                   │           │
│  ┌──────────────┐  ┌─────────────────────────────┐│           │
│  │  /ask (NL)   │  │  WebSocket Live Feed        ││           │
│  │  LLM Router  │  │  (sensor updates + alerts)  ││           │
│  └──────┬───────┘  └─────────────────────────────┘│           │
│         │                                         │           │
│         ▼                                         ▼           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Service Layer (Python)                      │ │
│  │                                                          │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │ │
│  │  │ Farm        │  │ Ingestion    │  │ LLM           │  │ │
│  │  │ Service     │  │ Service      │  │ Service       │  │ │
│  │  │ (CRUD,      │  │ (validate,   │  │ (context      │  │ │
│  │  │  topology)  │  │  store,      │  │  assembly,    │  │ │
│  │  │             │  │  notify      │  │  prompt,      │  │ │
│  │  │             │  │  graph)      │  │  parse,       │  │ │
│  │  │             │  │              │  │  translate)   │  │ │
│  │  └─────────────┘  └──────────────┘  └───────────────┘  │ │
│  │                         │                               │ │
│  │                         ▼                               │ │
│  │  ┌──────────────────────────────────────────────────┐   │ │
│  │  │            Julia Computational Core               │   │ │
│  │  │              (via juliacall)                       │   │ │
│  │  │                                                    │   │ │
│  │  │  ┌──────────────┐  ┌────────────────────────┐    │   │ │
│  │  │  │ Hypergraph   │  │ Predictive Models      │    │   │ │
│  │  │  │ Engine       │  │                        │    │   │ │
│  │  │  │              │  │ • IrrigationScheduler  │    │   │ │
│  │  │  │ • build()    │  │ • NutrientAlertEngine  │    │   │ │
│  │  │  │ • query()    │  │ • YieldForecaster      │    │   │ │
│  │  │  │ • update()   │  │ • AnomalyDetector      │    │   │ │
│  │  │  │ • cross_     │  │                        │    │   │ │
│  │  │  │   layer()    │  │ (water balance,        │    │   │ │
│  │  │  │              │  │  threshold + scoring,  │    │   │ │
│  │  │  │              │  │  regression,           │    │   │ │
│  │  │  │              │  │  statistical process   │    │   │ │
│  │  │  │              │  │  control)              │    │   │ │
│  │  │  └──────────────┘  └────────────────────────┘    │   │ │
│  │  │                                                    │   │ │
│  │  │  ┌──────────────────────────────────────────┐     │   │ │
│  │  │  │ Synthetic Data Generator                  │     │   │ │
│  │  │  │ (realistic sensor patterns, correlations, │     │   │ │
│  │  │  │  missing data, anomaly injection)         │     │   │ │
│  │  │  └──────────────────────────────────────────┘     │   │ │
│  │  └────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                         │              │                      │
└─────────────────────────┼──────────────┼──────────────────────┘
                          ▼              ▼
              ┌───────────────┐  ┌──────────────┐
              │  PostgreSQL   │  │    Redis      │
              │               │  │               │
              │ • vertices    │  │ • graph cache │
              │ • edges       │  │ • rate limits │
              │ • time-series │  │ • pub/sub for │
              │ • sensor data │  │   WebSocket   │
              │ • farm config │  │ • job status  │
              └───────────────┘  └──────────────┘
```

---

### 3. Language Boundary

The boundary between Python and Julia is clean and deliberate.

**Python owns:**
- HTTP layer (FastAPI, routing, middleware, auth)
- Request/response validation (Pydantic schemas)
- Database operations (SQLAlchemy, Alembic migrations)
- LLM API calls (httpx to Anthropic/OpenAI)
- WebSocket management
- Docker, CI, deployment configuration
- Test harness (pytest)

**Julia owns:**
- Hypergraph data structure (construction, traversal, cross-layer queries)
- All numerical computation (water balance, nutrient scoring, yield regression)
- Synthetic data generation (correlated time-series with realistic patterns)
- Any future physics-informed models (this is where TSR-derived techniques could slot in without being exposed)

**The bridge:** Python calls Julia via `juliacall`. The Julia core exposes a small set of functions that the Python service layer calls:

```python
# Python service layer calls into Julia
from juliacall import Main as jl

# Initialize hypergraph for a farm
jl.seval("using AgriSenseCore")
graph = jl.AgriSenseCore.build_graph(farm_config_dict)

# Query cross-layer state
state = jl.AgriSenseCore.query_farm_status(graph, zone_id)

# Run irrigation model
schedule = jl.AgriSenseCore.irrigation_schedule(graph, farm_id, horizon_days=7)

# Generate synthetic data
data = jl.AgriSenseCore.generate_synthetic(farm_type="greenhouse", days=90, seed=42)
```

The Julia code lives in a `core/` directory as a proper Julia package with its own `Project.toml`. This means it can also be tested independently with Julia's native test framework.

---

### 4. Farm Type Configuration

Farm type determines the active system topology at creation time.

**Open Field:**
```
Active layers: soil, irrigation, solar, weather, crop_requirements, npk
Vertex types:  zone (GPS polygon), sensor, valve, crop_bed, weather_station
CV layer:      disabled
Lighting:      solar only (observed, not actuated)
Climate:       fully external (weather API + on-site station)
Alerts:        irrigation, nutrient deficit, frost warning, heat stress
```

**Greenhouse:**
```
Active layers: soil, irrigation, lighting, weather, crop_requirements, npk, vision
Vertex types:  zone (structural), sensor, valve, crop_bed, weather_station,
               camera, light_fixture, climate_controller
CV layer:      enabled (camera → crop_bed hyperedges)
Lighting:      artificial (actuated) + supplemental solar
Climate:       partially controlled (HVAC vertices, humidity control)
Alerts:        all open field alerts + CV anomaly (pest, disease, wilting),
               light schedule deviation, climate control failure
```

**Hybrid:**
```
Both configurations coexist within one farm. Each zone declares its own type.
The hypergraph connects them through shared weather and irrigation infrastructure.
```

The configuration propagates through the system via a `FarmProfile` object that the Julia core uses to determine which layers to instantiate and which model parameters to load:

```julia
struct FarmProfile
    farm_type::Symbol           # :open_field, :greenhouse, :hybrid
    active_layers::Set{Symbol}  # {:soil, :irrigation, :weather, ...}
    zones::Vector{ZoneConfig}   # per-zone type override for hybrid
    models::ModelConfig         # which predictive models to run
end
```

---

### 5. Database Schema

#### 5.1 Core Tables

```
farms
├── id (UUID, PK)
├── name (VARCHAR)
├── farm_type (ENUM: open_field, greenhouse, hybrid)
├── location (GEOGRAPHY — PostGIS point)
├── timezone (VARCHAR)
├── created_at, updated_at

zones
├── id (UUID, PK)
├── farm_id (FK → farms)
├── name (VARCHAR)
├── zone_type (ENUM: open_field, greenhouse)
├── boundary (GEOGRAPHY — PostGIS polygon, nullable for greenhouse)
├── area_m2 (FLOAT)
├── soil_type (VARCHAR)
├── metadata (JSONB)

vertices
├── id (UUID, PK)
├── farm_id (FK → farms)
├── zone_id (FK → zones, nullable — weather stations may be farm-level)
├── vertex_type (ENUM: sensor, valve, crop_bed, weather_station, camera, light_fixture, climate_controller)
├── config (JSONB — type-specific attributes)
├── installed_at, last_seen_at

hyperedges
├── id (UUID, PK)
├── farm_id (FK → farms)
├── layer (ENUM: soil, irrigation, lighting, weather, crop_requirements, npk, vision)
├── vertex_ids (UUID[] — the vertices this hyperedge connects)
├── metadata (JSONB)
```

#### 5.2 Time-Series Tables

```
soil_readings
├── id (BIGSERIAL, PK)
├── sensor_id (FK → vertices)
├── timestamp (TIMESTAMPTZ)
├── moisture (FLOAT)
├── temperature (FLOAT)
├── conductivity (FLOAT, nullable)
├── ph (FLOAT, nullable)
INDEX: (sensor_id, timestamp DESC)

weather_readings
├── id (BIGSERIAL, PK)
├── station_id (FK → vertices)
├── timestamp (TIMESTAMPTZ)
├── temperature (FLOAT)
├── humidity (FLOAT)
├── precipitation_mm (FLOAT)
├── wind_speed (FLOAT, nullable)
├── wind_direction (FLOAT, nullable)
├── pressure_hpa (FLOAT, nullable)
├── et0 (FLOAT, nullable — calculated)
INDEX: (station_id, timestamp DESC)

irrigation_events
├── id (BIGSERIAL, PK)
├── valve_id (FK → vertices)
├── timestamp_start (TIMESTAMPTZ)
├── timestamp_end (TIMESTAMPTZ, nullable)
├── volume_liters (FLOAT, nullable)
├── trigger (ENUM: manual, scheduled, auto, emergency)

npk_samples
├── id (BIGSERIAL, PK)
├── zone_id (FK → zones)
├── timestamp (TIMESTAMPTZ)
├── nitrogen_mg_kg (FLOAT)
├── phosphorus_mg_kg (FLOAT)
├── potassium_mg_kg (FLOAT)
├── organic_matter_pct (FLOAT, nullable)
├── source (ENUM: lab, inline_sensor)

vision_events
├── id (BIGSERIAL, PK)
├── camera_id (FK → vertices)
├── crop_bed_id (FK → vertices)
├── timestamp (TIMESTAMPTZ)
├── anomaly_type (ENUM: none, pest, disease, nutrient_deficiency, wilting, other)
├── confidence (FLOAT)
├── canopy_coverage_pct (FLOAT, nullable)
├── metadata (JSONB — bounding boxes, class labels, etc.)

lighting_readings (greenhouse only)
├── id (BIGSERIAL, PK)
├── fixture_id (FK → vertices)
├── timestamp (TIMESTAMPTZ)
├── par_umol (FLOAT)
├── dli_cumulative (FLOAT)
├── duty_cycle_pct (FLOAT)
├── spectrum_profile (JSONB, nullable)
```

#### 5.3 Crop Reference Table

```
crop_profiles
├── id (UUID, PK)
├── crop_type (VARCHAR — e.g., "tomato", "wheat", "strawberry")
├── growth_stages (JSONB — array of stage objects)
│   └── each: { name, duration_days, optimal_moisture_range,
│                optimal_temp_range, water_demand_mm_day,
│                light_requirement_dli, npk_demand }
├── source (VARCHAR — "FAO", "field_calibrated", etc.)
```

---

### 6. Julia Core — Module Structure

```
core/AgriSenseCore/
├── Project.toml
├── src/
│   ├── AgriSenseCore.jl          # module entry, exports
│   ├── types.jl                   # FarmProfile, ZoneConfig, Vertex, HyperEdge
│   ├── hypergraph.jl              # build, query, update, cross_layer_query
│   ├── models/
│   │   ├── irrigation.jl          # water balance scheduler
│   │   ├── nutrients.jl           # NPK deficit scoring
│   │   ├── yield.jl               # regression forecaster
│   │   └── anomaly.jl             # statistical process control
│   ├── synthetic/
│   │   ├── generator.jl           # master generator (dispatches on farm_type)
│   │   ├── soil.jl                # moisture decay, temperature coupling
│   │   ├── weather.jl             # diurnal cycle, seasonal rainfall
│   │   ├── npk.jl                 # slow drift + fertilization steps
│   │   ├── vision.jl              # stochastic anomaly events
│   │   └── correlations.jl        # cross-layer correlation injection
│   └── bridge.jl                  # functions exposed to Python via juliacall
├── test/
│   ├── runtests.jl
│   ├── test_hypergraph.jl
│   ├── test_irrigation.jl
│   └── test_synthetic.jl
```

The `bridge.jl` file is the contract between Python and Julia. It exposes only plain data types (dicts, arrays, scalars) — no Julia-specific types cross the boundary:

```julia
# bridge.jl — Python-facing API
function build_graph(farm_config::Dict)::Dict
    profile = FarmProfile(farm_config)
    graph = HyperGraph(profile)
    return serialize(graph)  # returns Dict for Python consumption
end

function query_farm_status(graph_state::Dict, zone_id::String)::Dict
    graph = deserialize(graph_state)
    return Dict(status_report(graph, zone_id))
end

function irrigation_schedule(graph_state::Dict, horizon_days::Int)::Vector{Dict}
    graph = deserialize(graph_state)
    return [Dict(rec) for rec in compute_schedule(graph, horizon_days)]
end

function generate_synthetic(; farm_type::String, days::Int, seed::Int)::Dict
    return Dict(generate(Symbol(farm_type), days, seed))
end
```

---

### 7. LLM Integration Architecture

The `/ask` endpoint follows a retrieval-augmented generation pattern grounded in the hypergraph state.

```
User question ("When should I irrigate field 3?", language="ar")
                    │
                    ▼
        ┌───────────────────────┐
        │   Intent Classifier    │
        │   (rule-based or LLM)  │
        │                        │
        │   Determines:          │
        │   • relevant layers    │
        │   • target zone(s)     │
        │   • query type         │
        │     (irrigation,       │
        │      nutrient, yield,  │
        │      general status)   │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Context Assembler    │
        │                        │
        │   Queries hypergraph   │
        │   for relevant layers  │
        │   of target zone(s).   │
        │   Pulls latest sensor  │
        │   values, model        │
        │   outputs, active      │
        │   alerts.              │
        │                        │
        │   Formats as           │
        │   structured context   │
        │   block for the LLM.   │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   LLM Call             │
        │                        │
        │   System prompt:       │
        │   "You are an          │
        │   agricultural         │
        │   advisor. Answer      │
        │   ONLY based on the    │
        │   provided sensor      │
        │   data. Respond in     │
        │   {language}. Use      │
        │   simple language      │
        │   accessible to        │
        │   someone with no      │
        │   formal education."   │
        │                        │
        │   User message:        │
        │   question + context   │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Response Parser      │
        │                        │
        │   Extracts:            │
        │   • answer text        │
        │   • action if any      │
        │   • confidence         │
        │   Attaches source      │
        │   data references      │
        │   for auditability.    │
        └───────────────────────┘
```

The system prompt enforces grounding — the LLM cannot hallucinate data because it only sees what the context assembler provides. If the hypergraph doesn't have data for a requested zone, the LLM is instructed to say so rather than guess.

---

### 8. Deployment Architecture

```
docker-compose.yml
├── api        (Python FastAPI + Julia runtime)
│              Dockerfile installs Python 3.12 + Julia 1.11
│              Julia packages precompiled at build time
│              Exposes port 8000
│
├── postgres   (PostgreSQL 16 + PostGIS)
│              Volume-mounted for persistence
│              Init script creates DB + extensions
│              Exposes port 5432 (internal only)
│
├── redis      (Redis 7)
│              Exposes port 6379 (internal only)
│
└── seed       (one-shot container)
               Runs seed_db.py on first startup
               Calls Julia synthetic generator
               Populates DB with 90-day demo data
               Exits after completion
```

**Production deployment (portfolio demo):** Single instance on Railway or Fly.io. The Julia precompilation adds ~2 minutes to cold start but subsequent requests hit the compiled cache. For a portfolio demo this is acceptable.

**Environment variables:**
```
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://...
ANTHROPIC_API_KEY=sk-...        (for /ask endpoint)
JWT_SECRET=...
FARM_DEFAULT_TYPE=greenhouse    (for demo)
LOG_LEVEL=info
```

---

### 9. Test Strategy

```
tests/
├── conftest.py                  # async test client, test DB, fixtures
├── test_auth.py                 # JWT flow, RBAC enforcement, API key auth
├── test_farms.py                # CRUD operations, farm_type validation
├── test_ingest.py               # all ingestion endpoints, validation errors
├── test_analytics.py            # status, schedule, forecast, alerts
├── test_ask.py                  # NL endpoint with mocked LLM responses
├── test_websocket.py            # live feed subscription
└── test_julia_bridge.py         # bridge functions return expected shapes
```

Julia core has its own test suite run via `julia --project=core/AgriSenseCore -e 'using Pkg; Pkg.test()'`. The GitHub Actions CI runs both Python and Julia tests.

---

### 10. File Tree — Complete

```
agrisense/
├── README.md
├── LICENSE
├── docker-compose.yml
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml               # lint + type check + pytest + Julia tests + Docker build
├── .env.example
├── alembic/
│   ├── alembic.ini
│   └── versions/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, lifespan (init Julia, build graphs)
│   ├── config.py                # pydantic-settings, env loading
│   ├── auth/
│   │   ├── dependencies.py      # get_current_user, require_role
│   │   ├── jwt.py               # token creation/validation
│   │   └── models.py            # User, APIKey
│   ├── models/
│   │   ├── farm.py              # Farm, Zone, Vertex, HyperEdge ORM models
│   │   ├── sensors.py           # SoilReading, WeatherReading, etc.
│   │   └── crops.py             # CropProfile
│   ├── schemas/
│   │   ├── farm.py              # Pydantic request/response schemas
│   │   ├── ingest.py
│   │   ├── analytics.py
│   │   └── ask.py
│   ├── routes/
│   │   ├── farms.py
│   │   ├── ingest.py
│   │   ├── analytics.py
│   │   ├── ask.py
│   │   └── ws.py
│   ├── services/
│   │   ├── farm_service.py
│   │   ├── ingest_service.py
│   │   ├── analytics_service.py
│   │   ├── llm_service.py
│   │   └── julia_bridge.py      # Python side of the bridge
│   └── middleware/
│       ├── rate_limit.py
│       └── logging.py
├── core/
│   └── AgriSenseCore/           # Julia package (entire module structure from §6)
├── scripts/
│   ├── seed_db.py               # calls Julia generator, inserts into DB
│   └── demo_queries.sh          # curl examples for README
└── tests/                       # (structure from §9)
```

---

