## AgriSense API — Product Requirements Document

**Version:** 1.0
**Author:** Mehdi Skouri
**Date:** February 28, 2026
**Status:** Draft

---

### 1. Overview

AgriSense is a FastAPI-based backend service that models agricultural fields as layered hypergraphs, where each data layer (soil, irrigation, climate, crop, vision) forms a hyperedge substrate connecting relevant sensor nodes. The system ingests multimodal farm data, maintains a live hypergraph state per field, runs predictive models for irrigation scheduling, nutrient management, and yield estimation, and exposes an LLM-powered natural language query interface that lets non-technical users ask questions about their fields in plain language.

**Core differentiator:** Unlike flat relational schemas that treat each sensor reading independently, AgriSense models the farm as a structured hypergraph where cross-layer interactions (e.g., soil moisture × ambient humidity × crop transpiration rate) are first-class objects. This enables predictions that account for systemic coupling between variables rather than treating them in isolation.

---

### 2. Business Context

Field Partner SARL deploys agricultural AI solutions for farms across Morocco, many operated by staff with no formal education. The production system uses proprietary client data under NDA. AgriSense is a portfolio-grade reconstruction that demonstrates the architectural and modeling approach using synthetic data modeled on realistic agricultural patterns.

**Target users in the portfolio framing:**
- Farm managers querying field status via natural language
- Agronomists reviewing cross-layer analytics dashboards
- Irrigation system controllers consuming scheduled recommendations via API

---

### 3. Data Architecture — Layered Hypergraph Model

The farm is modeled as a hypergraph `H = (V, E)` where vertices `V` represent physical entities (sensors, field zones, irrigation valves, crop beds) and hyperedges `E` represent typed relationships that can connect arbitrary subsets of vertices.

#### 3.1 Vertex Types

| Vertex Type | Description | Example Attributes |
|---|---|---|
| `zone` | A bounded area of the field | area_m2, gps_polygon, elevation, soil_type |
| `sensor` | A physical or virtual sensor node | sensor_type, location, calibration_date |
| `valve` | Irrigation control point | flow_rate_max, zone_assignment |
| `crop_bed` | A planting unit within a zone | crop_type, planting_date, growth_stage |
| `weather_station` | Ambient condition source | location, data_source (local sensor vs API) |

#### 3.2 Hyperedge Layers

Each layer is a set of hyperedges connecting relevant vertices. Layers interact through shared vertices (a zone vertex appears in multiple layers simultaneously).

**Layer 1 — Soil Substrate**
- Hyperedges connect zone vertices to soil sensor vertices
- Time-series attributes: moisture (volumetric %), temperature (°C), electrical conductivity (dS/m), pH
- Spatial interpolation between sensor points within a zone
- Sampling frequency: every 15 minutes

**Layer 2 — Irrigation System**
- Hyperedges connect valve vertices to zone vertices they service (one valve may serve multiple zones — this is why hyperedges matter, not simple edges)
- Attributes: flow rate (L/min), cumulative volume delivered, schedule state (on/off/scheduled), pressure (bar)
- Event-driven: logs state changes + periodic polling

**Layer 3 — Solar/Lighting**
- Hyperedges connect zones to light source vertices (sun for open fields, artificial lighting arrays for greenhouses)
- Attributes: PAR (photosynthetically active radiation, µmol/m²/s), daily light integral (DLI), photoperiod hours
- For greenhouses: includes light fixture vertices with spectrum profile, duty cycle, height

**Layer 4 — Rainfall & Weather**
- Hyperedges connect weather station vertices to all zones within their coverage radius
- Attributes: precipitation (mm/hr and cumulative), ambient temperature (°C), relative humidity (%), wind speed (m/s), wind direction, atmospheric pressure (hPa), ET₀ (reference evapotranspiration, calculated)
- Source: on-site station + external weather API fallback

**Layer 5 — Crop Requirements (Metric Layer)**
- Hyperedges connect crop_bed vertices to their requirement profile
- This layer is largely static/slowly-changing — it encodes the crop's needs rather than sensor readings
- Attributes per growth stage: optimal soil moisture range, optimal temperature range, water demand (mm/day), light requirement (DLI), days to next stage transition, expected yield per m² at current stage
- Source: agronomic reference tables + field-calibrated adjustments

**Layer 6 — NPK & Nutrient**
- Hyperedges connect zone vertices to nutrient sensor/sample vertices
- Attributes: nitrogen (mg/kg), phosphorus (mg/kg), potassium (mg/kg), organic matter (%), CEC (meq/100g)
- Sampling frequency: lower cadence (weekly lab results or inline sensor readings where available)
- Derived attributes: deficit/surplus relative to Layer 5 crop requirements

**Layer 7 — Computer Vision (Greenhouse)**
- Hyperedges connect camera vertices to the crop_bed vertices in their field of view (one camera may cover multiple beds — hyperedge, not edge)
- Attributes: latest frame timestamp, detected anomalies (pest, disease, nutrient deficiency, wilting), growth metric estimates (canopy coverage %, estimated height, leaf color index), fruit count/ripeness stage where applicable
- Processing: frames processed by a CV model (out of scope for this API — the API receives inference results, not raw frames)
- Frequency: every 30 minutes or on-demand trigger

#### 3.3 Cross-Layer Interactions

The power of the hypergraph model is in cross-layer queries. Examples:

- **Irrigation decision:** Requires Layer 1 (current soil moisture) × Layer 4 (upcoming rainfall forecast) × Layer 5 (crop water demand at current growth stage) × Layer 2 (valve capacity and schedule)
- **Nutrient alert:** Layer 6 (current NPK) × Layer 5 (crop NPK requirements) × Layer 7 (visual deficiency symptoms detected by CV)
- **Yield prediction:** Layer 3 (cumulative DLI) × Layer 1 (soil health trend) × Layer 5 (expected yield at stage) × Layer 7 (canopy coverage as proxy for actual biomass)

#### 3.4 Implementation

The hypergraph is stored as:
- **PostgreSQL** for persistent vertex/edge storage, time-series sensor data, and relational metadata
- **In-memory adjacency structure** (Python dict-of-sets or a lightweight hypergraph library) rebuilt on startup and updated incrementally as data arrives
- Hyperedge queries are resolved in the service layer, not in SQL — the database stores the raw data, the hypergraph logic lives in Python

---

### 4. Synthetic Data Generation

A `scripts/seed_db.py` script generates realistic data for a demonstration farm:

**Farm layout:** 2 greenhouses + 4 open field zones, 18 sensors, 6 irrigation valves, 3 weather stations, 4 cameras (greenhouse only), 8 crop beds

**Temporal range:** 90 days of historical data at appropriate frequencies per layer

**Realism patterns:**
- Soil moisture: exponential decay between irrigation events, sharp rises on irrigation/rainfall, diurnal temperature coupling
- Weather: sinusoidal diurnal temperature cycle + random perturbation, seasonal rainfall probability, correlated humidity
- NPK: slow drift with periodic fertilization step-changes
- CV: stochastic anomaly events (pest detection at ~2% of frames, disease at ~0.5%) with spatial clustering (if one bed has pests, adjacent beds have elevated probability)
- Crop growth: deterministic stage transitions with noise on timing
- Missing values: ~3% random sensor dropout, occasional full-day outages on individual sensors

---

### 5. API Design

#### 5.1 Authentication & Authorization

- JWT-based auth with access + refresh tokens
- Role-based access: `admin`, `agronomist`, `field_operator`, `readonly`
- API key alternative for machine-to-machine integrations (irrigation controllers)
- Rate limiting: 100 req/min for standard, 1000 req/min for API keys

#### 5.2 Core Endpoints

**Farm & Topology**
```
POST   /api/v1/farms                    Create a farm
GET    /api/v1/farms/{farm_id}          Get farm metadata + topology summary
GET    /api/v1/farms/{farm_id}/graph    Get full hypergraph structure (vertices + edges by layer)
POST   /api/v1/farms/{farm_id}/zones    Add a zone
POST   /api/v1/farms/{farm_id}/sensors  Register a sensor to a zone
```

**Data Ingestion**
```
POST   /api/v1/ingest/soil              Batch ingest soil readings
POST   /api/v1/ingest/weather           Batch ingest weather data
POST   /api/v1/ingest/irrigation        Log irrigation event
POST   /api/v1/ingest/npk              Ingest nutrient sample results
POST   /api/v1/ingest/vision            Post CV inference results
POST   /api/v1/ingest/bulk              Multi-layer batch ingest (for backfill/sync)
```

**Analytics & Predictions**
```
GET    /api/v1/analytics/{farm_id}/status              Current farm state across all layers
GET    /api/v1/analytics/{farm_id}/zones/{zone_id}     Zone deep-dive with cross-layer view
GET    /api/v1/analytics/{farm_id}/irrigation/schedule  Recommended irrigation schedule (next 7 days)
GET    /api/v1/analytics/{farm_id}/nutrients/report     NPK status vs crop requirements
GET    /api/v1/analytics/{farm_id}/yield/forecast       Yield prediction per crop bed
GET    /api/v1/analytics/{farm_id}/alerts               Active alerts (irrigation needed, nutrient deficit, CV anomaly)
```

**Natural Language Interface**
```
POST   /api/v1/ask/{farm_id}
Body: { "question": "When should I irrigate field 3?", "language": "ar" }
Response: {
  "answer": "...",
  "language": "ar",
  "sources": [
    { "layer": "soil", "zone": "field_3", "metric": "moisture", "value": 0.18, "timestamp": "..." },
    { "layer": "weather", "metric": "rainfall_forecast_48h", "value": 0.0 },
    { "layer": "crop", "zone": "field_3", "metric": "water_demand_mm_day", "value": 5.2 }
  ],
  "recommendation": { "action": "irrigate", "suggested_time": "...", "volume_liters": 2400 }
}
```

The `/ask` endpoint:
1. Parses the natural language question
2. Determines which hypergraph layers are relevant
3. Queries current state from those layers
4. Constructs a context payload with real data
5. Sends to LLM with a system prompt that enforces grounded, data-backed responses
6. Returns the answer in the requested language (Arabic, French, or English)
7. Includes the source data so the answer is auditable

**Background Jobs**
```
POST   /api/v1/jobs/{farm_id}/recompute    Trigger full hypergraph recomputation
GET    /api/v1/jobs/{job_id}/status         Check job status
```

#### 5.3 WebSocket

```
WS     /ws/{farm_id}/live                  Real-time feed of sensor updates + alerts
```

---

### 6. Predictive Models

Lightweight models that run inside the API process (no separate ML serving infrastructure for the portfolio version).

**Irrigation Scheduler:**
- Input: current soil moisture (Layer 1), ET₀ and rainfall forecast (Layer 4), crop water demand (Layer 5), valve capacity (Layer 2)
- Method: water balance equation with threshold triggers — `soil_moisture_tomorrow = current - ET₀_crop + rainfall_forecast + irrigation_applied`
- Output: per-zone irrigation recommendation with timing and volume

**Nutrient Alert Engine:**
- Input: current NPK (Layer 6), crop requirements at growth stage (Layer 5), CV deficiency flags (Layer 7)
- Method: threshold comparison with severity scoring — visual confirmation from CV elevates alert priority
- Output: alert with deficit magnitude, suggested amendment, and urgency level

**Yield Forecaster:**
- Input: cumulative DLI (Layer 3), soil health trajectory (Layer 1), growth stage progression (Layer 5), canopy coverage trend (Layer 7)
- Method: linear regression on historical yield data indexed by these features (trained on synthetic historical data)
- Output: per-bed yield estimate with confidence interval

---

### 7. Technical Stack

| Component | Choice | Rationale |
|---|---|---|
| Framework | FastAPI | Async-native, auto-generated OpenAPI docs, Pydantic validation |
| Database | PostgreSQL 16 | Time-series data, JSONB for flexible sensor payloads, PostGIS for spatial |
| ORM | SQLAlchemy 2.0 + Alembic | Async support, migration management |
| Caching | Redis | Hypergraph state cache, rate limiting, job queue |
| LLM | Anthropic Claude API (or OpenAI) | Natural language interface |
| Task Queue | FastAPI BackgroundTasks (simple) or Celery (if needed) | Recomputation jobs |
| Testing | pytest + httpx (async test client) | Full endpoint coverage |
| CI | GitHub Actions | Lint (ruff), type check (mypy), test, build Docker image |
| Containerization | Docker + docker-compose | PostgreSQL + Redis + API in one `docker-compose up` |
| Deployment | Railway or Fly.io | Live demo URL |

---

### 8. Non-Functional Requirements

- **Startup:** Hypergraph rebuilds from DB in under 5 seconds for the demo farm size
- **Latency:** Sensor data endpoints < 50ms p95, analytics endpoints < 200ms p95, `/ask` endpoint < 3s (LLM-bound)
- **Seeding:** `seed_db.py` populates the full 90-day demo dataset in under 30 seconds
- **Documentation:** Swagger UI auto-served at `/docs`, README with architecture diagram, setup guide, and example curl commands
- **Observability:** Structured JSON logging, request ID propagation, basic health check endpoint at `/health`

---

### 9. README Must-Haves

- One-paragraph project description mentioning hypergraph modeling and real-world agricultural context
- Architecture diagram (Mermaid) showing the layered hypergraph and data flow
- "Uses synthetic sensor data modeled on real-world agricultural patterns. Production deployment at Field Partner SARL uses proprietary client data under NDA."
- Quickstart: `docker-compose up` → seeded and running
- Example API calls with curl or httpx
- Screenshot of Swagger UI
- Link to live deployed instance

---

### 10. Scope Boundaries

**In scope:** Everything above. The API, the data model, the hypergraph logic, the predictive models, the LLM interface, synthetic data generation, tests, Docker, CI, deployment.

**Out of scope:** Frontend/dashboard (API-only project), real CV model inference (API receives results, doesn't process frames), mobile app, real-time hardware integration, production-grade monitoring/alerting infrastructure.

---

### 11. Estimated Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Database schema + models + migrations | 4 hours | Working PostgreSQL with all vertex/edge tables |
| Seed script | 3 hours | 90-day realistic synthetic dataset |
| Core CRUD endpoints (farm, zones, sensors, ingestion) | 4 hours | All ingestion and topology endpoints working |
| Hypergraph service layer | 4 hours | In-memory graph construction, cross-layer queries |
| Predictive models (irrigation, nutrients, yield) | 3 hours | Three working models with test coverage |
| Analytics endpoints | 3 hours | Status, schedule, report, forecast, alerts |
| LLM natural language endpoint | 3 hours | `/ask` with multilingual support and source attribution |
| Auth (JWT + API keys + RBAC) | 2 hours | Full auth flow |
| WebSocket live feed | 1 hour | Real-time sensor stream |
| Tests | 3 hours | pytest suite covering critical paths |
| Docker + compose + CI | 2 hours | One-command startup, GitHub Actions pipeline |
| README + architecture diagram | 2 hours | Portfolio-ready documentation |
| Deploy to Railway/Fly.io | 1 hour | Live URL |
| **Total** | **~35 hours** | **~4 focused days** |

---

