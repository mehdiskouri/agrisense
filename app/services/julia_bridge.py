"""Python-side bridge to Julia AgriSenseCore via juliacall."""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Any

from app.config import get_settings

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
			from juliacall import Main as jl_main  # type: ignore[import-untyped]

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
		parsed = _from_julia(result)
		_bridge_timing("build_graph", start, True)
		return parsed
	except Exception as exc:
		_bridge_timing("build_graph", start, False, str(exc))
		raise JuliaBridgeError(f"build_graph failed: {exc}") from exc


def query_farm_status(graph_state: dict[str, Any], zone_id: str) -> dict[str, Any]:
	start = time.perf_counter()
	module = _require_module()
	try:
		result = module.query_farm_status(_to_plain(graph_state), str(zone_id))
		parsed = _from_julia(result)
		_bridge_timing("query_farm_status", start, True)
		return parsed
	except Exception as exc:
		_bridge_timing("query_farm_status", start, False, str(exc))
		raise JuliaBridgeError(f"query_farm_status failed: {exc}") from exc


def irrigation_schedule(
	graph_state: dict[str, Any],
	horizon_days: int,
	weather_forecast: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
	start = time.perf_counter()
	module = _require_module()
	try:
		forecast = _to_plain(weather_forecast or {})
		result = module.irrigation_schedule(
			_to_plain(graph_state),
			int(horizon_days),
			forecast,
		)
		parsed = _from_julia(result)
		_bridge_timing("irrigation_schedule", start, True)
		return [dict(item) for item in parsed]
	except Exception as exc:
		_bridge_timing("irrigation_schedule", start, False, str(exc))
		raise JuliaBridgeError(f"irrigation_schedule failed: {exc}") from exc


def nutrient_report(graph_state: dict[str, Any]) -> list[dict[str, Any]]:
	start = time.perf_counter()
	module = _require_module()
	try:
		result = module.nutrient_report(_to_plain(graph_state))
		parsed = _from_julia(result)
		_bridge_timing("nutrient_report", start, True)
		return [dict(item) for item in parsed]
	except Exception as exc:
		_bridge_timing("nutrient_report", start, False, str(exc))
		raise JuliaBridgeError(f"nutrient_report failed: {exc}") from exc


def yield_forecast(graph_state: dict[str, Any]) -> list[dict[str, Any]]:
	start = time.perf_counter()
	module = _require_module()
	try:
		result = module.yield_forecast(_to_plain(graph_state))
		parsed = _from_julia(result)
		_bridge_timing("yield_forecast", start, True)
		return [dict(item) for item in parsed]
	except Exception as exc:
		_bridge_timing("yield_forecast", start, False, str(exc))
		raise JuliaBridgeError(f"yield_forecast failed: {exc}") from exc


def detect_anomalies(graph_state: dict[str, Any]) -> list[dict[str, Any]]:
	start = time.perf_counter()
	module = _require_module()
	try:
		result = module.detect_anomalies(_to_plain(graph_state))
		parsed = _from_julia(result)
		_bridge_timing("detect_anomalies", start, True)
		return [dict(item) for item in parsed]
	except Exception as exc:
		_bridge_timing("detect_anomalies", start, False, str(exc))
		raise JuliaBridgeError(f"detect_anomalies failed: {exc}") from exc


def cross_layer_query(
	graph_state: dict[str, Any],
	layer_a: str,
	layer_b: str,
) -> dict[str, Any]:
	start = time.perf_counter()
	module = _require_module()
	if _jl_main is None:
		raise JuliaBridgeError("cross_layer_query failed: Julia runtime is not initialized")
	try:
		graph = module.deserialize_graph(_to_plain(graph_state))
		layer_a_sym = _jl_main.Symbol(str(layer_a))
		layer_b_sym = _jl_main.Symbol(str(layer_b))
		result = module.cross_layer_query(graph, layer_a_sym, layer_b_sym)
		parsed = _from_julia(result)
		_bridge_timing("cross_layer_query", start, True)
		return parsed
	except Exception as exc:
		_bridge_timing("cross_layer_query", start, False, str(exc))
		raise JuliaBridgeError(f"cross_layer_query failed: {exc}") from exc


def update_features(
	graph_state: dict[str, Any],
	layer: str,
	vertex_id: str,
	features: list[float],
) -> dict[str, Any]:
	start = time.perf_counter()
	module = _require_module()
	try:
		result = module.update_features(
			_to_plain(graph_state),
			str(layer),
			str(vertex_id),
			_to_plain(features),
		)
		parsed = _from_julia(result)
		_bridge_timing("update_features", start, True)
		return parsed
	except Exception as exc:
		_bridge_timing("update_features", start, False, str(exc))
		raise JuliaBridgeError(f"update_features failed: {exc}") from exc


def train_yield_residual(
	graph_state: dict[str, Any],
	outcomes: dict[str, float],
) -> dict[str, Any]:
	start = time.perf_counter()
	module = _require_module()
	try:
		result = module.train_yield_residual(_to_plain(graph_state), _to_plain(outcomes))
		parsed = _from_julia(result)
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
		parsed = _from_julia(result)
		_bridge_timing("generate_synthetic", start, True)
		return parsed
	except Exception as exc:
		_bridge_timing("generate_synthetic", start, False, str(exc))
		raise JuliaBridgeError(f"generate_synthetic failed: {exc}") from exc
