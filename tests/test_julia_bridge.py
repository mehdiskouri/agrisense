from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.services import julia_bridge
from app.services.julia_bridge import JuliaBridgeError


class _FakeModule:
    def __init__(self) -> None:
        self.last_payload: dict[str, object] | None = None

    def build_graph(self, payload: dict[str, object]) -> dict[str, object]:
        self.last_payload = payload
        return {"status": "ok", "echo": payload}


class _FailingModule:
    def build_graph(self, payload: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("boom")


def test_build_graph_serializes_uuid_and_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeModule()
    monkeypatch.setattr(julia_bridge, "_initialized", True)
    monkeypatch.setattr(julia_bridge, "_agrisense_module", fake)

    farm_id = uuid4()
    now = datetime(2026, 2, 28, 12, 0, tzinfo=UTC)
    payload = {
        "farm_id": farm_id,
        "created_at": now,
        "zones": [{"id": uuid4(), "name": "z1"}],
    }

    result = julia_bridge.build_graph(payload)

    assert result["status"] == "ok"
    assert fake.last_payload is not None
    assert fake.last_payload["farm_id"] == str(farm_id)
    assert fake.last_payload["created_at"] == now.isoformat()
    zones = fake.last_payload["zones"]
    assert isinstance(zones, list)
    first_zone = zones[0]
    assert isinstance(first_zone, dict)
    zone_id = first_zone["id"]
    assert isinstance(zone_id, str)
    assert UUID(zone_id)


def test_build_graph_wraps_julia_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(julia_bridge, "_initialized", True)
    monkeypatch.setattr(julia_bridge, "_agrisense_module", _FailingModule())

    with pytest.raises(JuliaBridgeError, match="build_graph failed"):
        julia_bridge.build_graph({"farm_id": str(uuid4())})
