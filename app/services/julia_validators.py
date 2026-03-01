"""Runtime validators for Julia bridge payload boundaries.

These helpers convert dynamic Julia-returned objects into typed Python shapes
while failing fast with clear error messages when shape contracts are violated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def ensure_record(value: Any, *, context: str) -> dict[str, Any]:
    """Validate and normalize a mapping-like payload into a plain dict."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected mapping payload, got {type(value).__name__}")
    return {str(key): _normalize_dynamic(item) for key, item in value.items()}


def ensure_record_list(value: Any, *, context: str) -> list[dict[str, Any]]:
    """Validate and normalize a sequence of mapping payloads."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{context}: expected sequence payload, got {type(value).__name__}")

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(value):
        records.append(ensure_record(item, context=f"{context}[{idx}]"))
    return records


def ensure_typed_record(value: Any, *, context: str) -> dict[str, Any]:
    """Validate a mapping and return a normalized plain dict payload."""

    return ensure_record(value, context=context)


def _normalize_dynamic(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_dynamic(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_dynamic(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
