"""Python-side bridge to Julia AgriSenseCore via juliacall."""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from app.config import get_settings

_lock = threading.Lock()
_initialized = False
_jl_main: Any | None = None
_agrisense_module: Any | None = None


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


def build_graph(farm_config: dict[str, Any]) -> dict[str, Any]:
	module = _require_module()
	payload = _to_plain(farm_config)
	try:
		result = module.build_graph(payload)
		return _from_julia(result)
	except Exception as exc:
		raise JuliaBridgeError(f"build_graph failed: {exc}") from exc


def query_farm_status(graph_state: dict[str, Any], zone_id: str) -> dict[str, Any]:
	module = _require_module()
	try:
		result = module.query_farm_status(_to_plain(graph_state), str(zone_id))
		return _from_julia(result)
	except Exception as exc:
		raise JuliaBridgeError(f"query_farm_status failed: {exc}") from exc


def irrigation_schedule(
	graph_state: dict[str, Any],
	horizon_days: int,
	weather_forecast: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
	module = _require_module()
	try:
		forecast = _to_plain(weather_forecast or {})
		result = module.irrigation_schedule(
			_to_plain(graph_state),
			int(horizon_days),
			forecast,
		)
		parsed = _from_julia(result)
		return [dict(item) for item in parsed]
	except Exception as exc:
		raise JuliaBridgeError(f"irrigation_schedule failed: {exc}") from exc


def nutrient_report(graph_state: dict[str, Any]) -> list[dict[str, Any]]:
	module = _require_module()
	try:
		result = module.nutrient_report(_to_plain(graph_state))
		parsed = _from_julia(result)
		return [dict(item) for item in parsed]
	except Exception as exc:
		raise JuliaBridgeError(f"nutrient_report failed: {exc}") from exc


def yield_forecast(graph_state: dict[str, Any]) -> list[dict[str, Any]]:
	module = _require_module()
	try:
		result = module.yield_forecast(_to_plain(graph_state))
		parsed = _from_julia(result)
		return [dict(item) for item in parsed]
	except Exception as exc:
		raise JuliaBridgeError(f"yield_forecast failed: {exc}") from exc


def detect_anomalies(graph_state: dict[str, Any]) -> list[dict[str, Any]]:
	module = _require_module()
	try:
		result = module.detect_anomalies(_to_plain(graph_state))
		parsed = _from_julia(result)
		return [dict(item) for item in parsed]
	except Exception as exc:
		raise JuliaBridgeError(f"detect_anomalies failed: {exc}") from exc


def cross_layer_query(
	graph_state: dict[str, Any],
	layer_a: str,
	layer_b: str,
) -> dict[str, Any]:
	module = _require_module()
	try:
		graph = module.deserialize_graph(_to_plain(graph_state))
		result = module.cross_layer_query(graph, str(layer_a), str(layer_b))
		return _from_julia(result)
	except Exception as exc:
		raise JuliaBridgeError(f"cross_layer_query failed: {exc}") from exc


def update_features(
	graph_state: dict[str, Any],
	layer: str,
	vertex_id: str,
	features: list[float],
) -> dict[str, Any]:
	module = _require_module()
	try:
		result = module.update_features(
			_to_plain(graph_state),
			str(layer),
			str(vertex_id),
			_to_plain(features),
		)
		return _from_julia(result)
	except Exception as exc:
		raise JuliaBridgeError(f"update_features failed: {exc}") from exc


def train_yield_residual(
	graph_state: dict[str, Any],
	outcomes: dict[str, float],
) -> dict[str, Any]:
	module = _require_module()
	try:
		result = module.train_yield_residual(_to_plain(graph_state), _to_plain(outcomes))
		return _from_julia(result)
	except Exception as exc:
		raise JuliaBridgeError(f"train_yield_residual failed: {exc}") from exc


def generate_synthetic(
	farm_type: str = "greenhouse",
	days: int = 90,
	seed: int = 42,
) -> dict[str, Any]:
	module = _require_module()
	try:
		result = module.generate_synthetic(
			farm_type=str(farm_type),
			days=int(days),
			seed=int(seed),
		)
		return _from_julia(result)
	except Exception as exc:
		raise JuliaBridgeError(f"generate_synthetic failed: {exc}") from exc
