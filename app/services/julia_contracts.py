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
    """Result shape for post-update graph state mutation."""

    ok: bool
    message: str
    graph_state: dict[str, Any]


class TrainYieldResidualResponse(TypedDict, total=False):
    """Training summary from residual model fit."""

    ok: bool
    metrics: dict[str, float]
    model: dict[str, Any]


class SyntheticGenerationResponse(TypedDict, total=False):
    """Synthetic-data generator top-level output."""

    farm: dict[str, Any]
    zones: list[dict[str, Any]]
    vertices: list[dict[str, Any]]
    readings: dict[str, list[dict[str, Any]]]
