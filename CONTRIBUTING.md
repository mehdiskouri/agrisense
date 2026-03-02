# Contributing

## Development setup
- Use Python `3.13+` and Julia `1.12+`.
- Install dependencies with `make install`.
- Run migrations with `make migrate`.
- Run local API with `make dev`.

## GPU development
- NVIDIA drivers + `nvidia-container-toolkit` are required for Docker GPU execution.
- CPU fallback is always supported through `KernelAbstractions.CPU()`.
- Docker targets:
  - CPU image: `docker build --target runtime-cpu -t agrisense:cpu .`
  - GPU image: `docker build --target runtime-gpu -t agrisense:gpu .`

## Julia bridge architecture
- Python service layer calls Julia via `app/services/julia_bridge.py`.
- Julia boundary contract is defined in `core/AgriSenseCore/src/bridge.jl`.
- Bridge I/O must remain plain Dict/Vector/Array values.
- GPU arrays stay internal to Julia and must not cross into Python.

## Add a new hypergraph layer
1. Add feature dimensionality in `core/AgriSenseCore/src/hypergraph.jl` (`LAYER_FEATURE_DIMS`).
2. Ensure layer is represented in farm topology/build config (`app/services/farm_service.py`).
3. Add ingestion schema + route + service mapping (`app/schemas/ingest.py`, `app/routes/ingest.py`, `app/services/ingest_service.py`).
4. Extend serialization/deserialization if layer-specific state is introduced (`core/AgriSenseCore/src/bridge.jl`).
5. Add layer tests in Julia (`core/AgriSenseCore/test/`) and Python (`tests/`).

## Add a new predictive model
1. Implement Julia model in `core/AgriSenseCore/src/models/`.
2. Export the model in `core/AgriSenseCore/src/AgriSenseCore.jl`.
3. Expose a bridge wrapper in `core/AgriSenseCore/src/bridge.jl`.
4. Add Python wrapper in `app/services/julia_bridge.py`.
5. Wire endpoint/service response in `app/services/analytics_service.py` and relevant route/schema files.
6. Add deterministic model tests in Julia + API contract tests in Python.

## CI expectations
- `ruff`, `mypy`, Python tests, Julia CPU tests, and Docker smoke checks are required.
- GPU tests run in a dedicated GPU CI lane via manual workflow dispatch (`enable_gpu_ci=true`).
