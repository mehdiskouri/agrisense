"""Python-side bridge to Julia AgriSenseCore via juliacall."""

from __future__ import annotations

import importlib
import logging
import os
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.julia_validators import ensure_record, ensure_record_list

_lock = threading.Lock()
_initialized = False
_jl_main: Any | None = None
_agrisense_module: Any | None = None
_logger = logging.getLogger("agrisense.julia_bridge")


class JuliaBridgeError(RuntimeError):
    """Raised when a Julia bridge call fails."""


def _project_path() -> Path:
    settings = get_settings()
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / settings.julia_project).resolve()


def _to_plain(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for (k, v) in value.items()}
    if isinstance(value, set):
        return [_to_plain(v) for v in sorted(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_plain(v) for v in value]
    return value


def _from_julia(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _from_julia(v) for (k, v) in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_from_julia(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def initialize_julia() -> None:
    """Initialize Julia runtime and load AgriSenseCore once per process."""
    global _initialized, _jl_main, _agrisense_module

    if _initialized:
        return

    with _lock:
        if _initialized:
            return

        settings = get_settings()
        os.environ.setdefault("JULIA_NUM_THREADS", settings.julia_num_threads)

        try:
            jl_module = importlib.import_module("juliacall")
            jl_main = jl_module.Main

            project = _project_path().as_posix()
            jl_main.seval(f'import Pkg; Pkg.activate(raw"{project}")')
            jl_main.seval("using AgriSenseCore")

            _jl_main = jl_main
            _agrisense_module = jl_main.AgriSenseCore
            _initialized = True
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise JuliaBridgeError(f"Failed to initialize Julia runtime: {exc}") from exc


def _require_module() -> Any:
    initialize_julia()
    if _agrisense_module is None:
        raise JuliaBridgeError("Julia bridge is not initialized")
    return _agrisense_module


def _bridge_timing(op: str, start: float, ok: bool, error: str | None = None) -> None:
    duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
    extra = {
        "operation": op,
        "duration_ms": duration_ms,
        "ok": ok,
        "error": error,
    }
    if ok:
        _logger.info("julia_bridge_call", extra=extra)
    else:
        _logger.error("julia_bridge_call_failed", extra=extra)


def build_graph(farm_config: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    module = _require_module()
    payload = _to_plain(farm_config)
    try:
        result = module.build_graph(payload)
        parsed = ensure_record(_from_julia(result), context="build_graph")
        _bridge_timing("build_graph", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("build_graph", start, False, str(exc))
        raise JuliaBridgeError(f"build_graph failed: {exc}") from exc


def query_farm_status(farm_id: str, zone_id: str) -> dict[str, Any]:
    start = time.perf_counter()
    module = _require_module()
    try:
        result = module.query_farm_status(
            {"farm_id": str(farm_id)},
            str(zone_id),
        )
        parsed = ensure_record(_from_julia(result), context="query_farm_status")
        _bridge_timing("query_farm_status", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("query_farm_status", start, False, str(exc))
        raise JuliaBridgeError(f"query_farm_status failed: {exc}") from exc


def irrigation_schedule(
    farm_id: str,
    horizon_days: int,
    weather_forecast: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    module = _require_module()
    try:
        forecast = _to_plain(weather_forecast or {})
        result = module.irrigation_schedule(
            {"farm_id": str(farm_id)},
            int(horizon_days),
            forecast,
        )
        parsed = ensure_record_list(_from_julia(result), context="irrigation_schedule")
        _bridge_timing("irrigation_schedule", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("irrigation_schedule", start, False, str(exc))
        raise JuliaBridgeError(f"irrigation_schedule failed: {exc}") from exc


def nutrient_report(farm_id: str) -> list[dict[str, Any]]:
    start = time.perf_counter()
    module = _require_module()
    try:
        result = module.nutrient_report({"farm_id": str(farm_id)})
        parsed = ensure_record_list(_from_julia(result), context="nutrient_report")
        _bridge_timing("nutrient_report", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("nutrient_report", start, False, str(exc))
        raise JuliaBridgeError(f"nutrient_report failed: {exc}") from exc


def yield_forecast(farm_id: str) -> list[dict[str, Any]]:
    start = time.perf_counter()
    module = _require_module()
    try:
        result = module.yield_forecast({"farm_id": str(farm_id)})
        parsed = ensure_record_list(_from_julia(result), context="yield_forecast")
        _bridge_timing("yield_forecast", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("yield_forecast", start, False, str(exc))
        raise JuliaBridgeError(f"yield_forecast failed: {exc}") from exc


def yield_forecast_ensemble(
    farm_id: str,
    include_members: bool = False,
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    module = _require_module()
    try:
        result = module.yield_forecast_ensemble(
            {"farm_id": str(farm_id)},
            include_members=bool(include_members),
        )
        parsed = ensure_record_list(_from_julia(result), context="yield_forecast_ensemble")
        _bridge_timing("yield_forecast_ensemble", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("yield_forecast_ensemble", start, False, str(exc))
        raise JuliaBridgeError(f"yield_forecast_ensemble failed: {exc}") from exc


def backtest_yield(
    farm_id: str,
    n_folds: int = 5,
    min_history: int = 24,
) -> dict[str, Any]:
    start = time.perf_counter()
    module = _require_module()
    try:
        result = module.backtest_yield(
            {"farm_id": str(farm_id)},
            n_folds=int(n_folds),
            min_history=int(min_history),
        )
        parsed = ensure_record(_from_julia(result), context="backtest_yield")
        _bridge_timing("backtest_yield", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("backtest_yield", start, False, str(exc))
        raise JuliaBridgeError(f"backtest_yield failed: {exc}") from exc


def detect_anomalies(farm_id: str) -> list[dict[str, Any]]:
    start = time.perf_counter()
    module = _require_module()
    try:
        result = module.detect_anomalies({"farm_id": str(farm_id)})
        parsed = ensure_record_list(_from_julia(result), context="detect_anomalies")
        _bridge_timing("detect_anomalies", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("detect_anomalies", start, False, str(exc))
        raise JuliaBridgeError(f"detect_anomalies failed: {exc}") from exc


def cross_layer_query(
    farm_id: str,
    layer_a: str,
    layer_b: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    module = _require_module()
    if _jl_main is None:
        raise JuliaBridgeError("cross_layer_query failed: Julia runtime is not initialized")
    try:
        graph = module.get_cached_graph(str(farm_id))
        if graph is None:
            raise JuliaBridgeError(
                f"cross_layer_query failed: graph for farm '{farm_id}' not cached"
            )
        layer_a_sym = _jl_main.Symbol(str(layer_a))
        layer_b_sym = _jl_main.Symbol(str(layer_b))
        result = module.cross_layer_query(graph, layer_a_sym, layer_b_sym)
        parsed = ensure_record(_from_julia(result), context="cross_layer_query")
        _bridge_timing("cross_layer_query", start, True)
        return parsed
    except JuliaBridgeError:
        raise
    except Exception as exc:
        _bridge_timing("cross_layer_query", start, False, str(exc))
        raise JuliaBridgeError(f"cross_layer_query failed: {exc}") from exc


def update_features(
    farm_id: str,
    layer: str,
    vertex_id: str,
    features: list[float],
) -> dict[str, Any]:
    """Push features for a single vertex. Returns lightweight ack."""
    start = time.perf_counter()
    module = _require_module()
    try:
        result = module.update_features(
            {"farm_id": str(farm_id)},
            str(layer),
            str(vertex_id),
            _to_plain(features),
        )
        parsed = ensure_record(_from_julia(result), context="update_features")
        _bridge_timing("update_features", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("update_features", start, False, str(exc))
        raise JuliaBridgeError(f"update_features failed: {exc}") from exc


def batch_update_features(
    farm_id: str,
    updates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Batch push features for multiple vertices. Returns lightweight ack."""
    start = time.perf_counter()
    module = _require_module()
    try:
        result = module.batch_update_features(
            str(farm_id),
            _to_plain(updates),
        )
        parsed = ensure_record(_from_julia(result), context="batch_update_features")
        _bridge_timing("batch_update_features", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("batch_update_features", start, False, str(exc))
        raise JuliaBridgeError(f"batch_update_features failed: {exc}") from exc


def ensure_graph_cached(farm_id: str) -> dict[str, Any]:
    """Check if graph is cached in Julia. Returns ack or raises."""
    start = time.perf_counter()
    module = _require_module()
    try:
        result = module.ensure_graph(str(farm_id))
        parsed = ensure_record(_from_julia(result), context="ensure_graph")
        _bridge_timing("ensure_graph", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("ensure_graph", start, False, str(exc))
        raise JuliaBridgeError(f"ensure_graph failed: {exc}") from exc


def train_yield_residual(
    farm_id: str,
    outcomes: dict[str, float],
) -> dict[str, Any]:
    start = time.perf_counter()
    module = _require_module()
    try:
        result = module.train_yield_residual(
            {"farm_id": str(farm_id)},
            _to_plain(outcomes),
        )
        parsed = ensure_record(_from_julia(result), context="train_yield_residual")
        _bridge_timing("train_yield_residual", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("train_yield_residual", start, False, str(exc))
        raise JuliaBridgeError(f"train_yield_residual failed: {exc}") from exc


def generate_synthetic(
    farm_type: str = "greenhouse",
    days: int = 90,
    seed: int = 42,
) -> dict[str, Any]:
    start = time.perf_counter()
    module = _require_module()
    try:
        result = module.generate_synthetic(
            farm_type=str(farm_type),
            days=int(days),
            seed=int(seed),
        )
        parsed = ensure_record(_from_julia(result), context="generate_synthetic")
        _bridge_timing("generate_synthetic", start, True)
        return parsed
    except Exception as exc:
        _bridge_timing("generate_synthetic", start, False, str(exc))
        raise JuliaBridgeError(f"generate_synthetic failed: {exc}") from exc
