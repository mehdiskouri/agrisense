"""Typed contracts for Julia bridge ingress/egress payloads."""

from __future__ import annotations

from typing import Any, TypedDict

JuliaRecord = dict[str, Any]
JuliaRecordList = list[JuliaRecord]
JuliaGraphState = dict[str, Any]


class FarmGraphResponse(TypedDict):
    """Serialized graph payload returned by Julia graph builder."""

    farm_id: str
    n_vertices: int
    vertex_index: dict[str, int]
    layers: dict[str, Any]


class CrossLayerResult(TypedDict, total=False):
    """Cross-layer query response payload."""

    score: float
    links: list[dict[str, Any]]
    attributes: dict[str, Any]


class UpdateFeaturesResponse(TypedDict, total=False):
    """Lightweight ack returned by update_features / batch_update_features."""

    ok: bool
    farm_id: str
    version: int
    layer: str
    vertex_id: str
    n_updated: int


class TrainYieldResidualResponse(TypedDict, total=False):
    """Training summary from residual model fit."""

    status: str
    n_observations: int
    n_coefficients: int


class EnsembleMember(TypedDict, total=False):
    """One member forecast within an ensemble prediction."""

    model_name: str
    yield_estimate: float
    lower: float
    upper: float
    weight: float


class EnsembleYieldForecastItem(TypedDict, total=False):
    """Per-bed ensemble yield forecast item."""

    crop_bed_id: str
    yield_estimate_kg_m2: float
    yield_lower: float
    yield_upper: float
    confidence: float
    model_layer: str
    stress_factors: dict[str, Any]
    ensemble_weights: dict[str, float]
    hyperparameters: dict[str, float]
    ensemble_members: list[EnsembleMember]


class BacktestResult(TypedDict, total=False):
    """Backtesting summary for ensemble model calibration."""

    farm_id: str
    n_folds: int
    per_fold_metrics: list[dict[str, Any]]
    aggregate_metrics: dict[str, Any]
    weights: dict[str, float]
    hyperparameters: dict[str, float]
    temporal_split: dict[str, dict[str, int]]
    oracle_provenance: dict[str, Any]
    status: str


class SyntheticGenerationResponse(TypedDict, total=False):
    """Synthetic-data generator top-level output."""

    farm: dict[str, Any]
    zones: list[dict[str, Any]]
    vertices: list[dict[str, Any]]
    readings: dict[str, list[dict[str, Any]]]
