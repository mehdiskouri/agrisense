from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.services import julia_bridge
from app.services.julia_bridge import JuliaBridgeError


class _FakeModule:
    def __init__(self) -> None:
        self.last_payload: dict[str, object] | None = None
        self.last_call: tuple[str, tuple[object, ...], dict[str, object]] | None = None

    def build_graph(self, payload: dict[str, object]) -> dict[str, object]:
        self.last_payload = payload
        return {"status": "ok", "echo": payload}

    def query_farm_status(self, graph_state: dict[str, object], zone_id: str) -> dict[str, object]:
        self.last_call = ("query_farm_status", (graph_state, zone_id), {})
        return {"zone": zone_id}

    def irrigation_schedule(
        self,
        graph_state: dict[str, object],
        horizon_days: int,
        weather_forecast: dict[str, object],
    ) -> list[dict[str, object]]:
        self.last_call = (
            "irrigation_schedule",
            (graph_state, horizon_days, weather_forecast),
            {},
        )
        return [{"ok": True}]

    def nutrient_report(self, graph_state: dict[str, object]) -> list[dict[str, object]]:
        self.last_call = ("nutrient_report", (graph_state,), {})
        return [{"nutrient": "ok"}]

    def yield_forecast(self, graph_state: dict[str, object]) -> list[dict[str, object]]:
        self.last_call = ("yield_forecast", (graph_state,), {})
        return [{"yield": "ok"}]

    def detect_anomalies(self, graph_state: dict[str, object]) -> list[dict[str, object]]:
        self.last_call = ("detect_anomalies", (graph_state,), {})
        return [{"anomaly": "ok"}]

    def deserialize_graph(self, graph_state: dict[str, object]) -> dict[str, object]:
        self.last_call = ("deserialize_graph", (graph_state,), {})
        return graph_state

    def cross_layer_query(
        self,
        graph_state: dict[str, object],
        layer_a: str,
        layer_b: str,
    ) -> dict[str, object]:
        self.last_call = ("cross_layer_query", (graph_state, layer_a, layer_b), {})
        return {"layer_a": layer_a, "layer_b": layer_b, "connected": 1}

    def update_features(
        self,
        graph_state: dict[str, object],
        layer: str,
        vertex_id: str,
        features: list[float],
    ) -> dict[str, object]:
        self.last_call = ("update_features", (graph_state, layer, vertex_id, features), {})
        return {"updated": True}

    def train_yield_residual(
        self,
        graph_state: dict[str, object],
        outcomes: dict[str, float],
    ) -> dict[str, object]:
        self.last_call = ("train_yield_residual", (graph_state, outcomes), {})
        return {"status": "trained"}

    def generate_synthetic(self, **kwargs: object) -> dict[str, object]:
        self.last_call = ("generate_synthetic", (), kwargs)
        return {"status": "ok", "seed": kwargs.get("seed")}


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


def test_initialize_julia_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = _FakeModule()

    class _FakeMain:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.AgriSenseCore = fake_module

        def seval(self, code: str) -> None:
            self.calls.append(code)

    fake_main = _FakeMain()
    fake_juliacall = ModuleType("juliacall")
    fake_juliacall.Main = fake_main  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "juliacall", fake_juliacall)
    monkeypatch.setattr(julia_bridge, "_initialized", False)
    monkeypatch.setattr(julia_bridge, "_agrisense_module", None)
    monkeypatch.setattr(
        julia_bridge,
        "get_settings",
        lambda: SimpleNamespace(julia_num_threads="1", julia_project="core/AgriSenseCore"),
    )

    julia_bridge.initialize_julia()

    assert julia_bridge._initialized is True
    assert julia_bridge._agrisense_module is fake_module
    assert any("Pkg.activate" in call for call in fake_main.calls)
    assert any("using AgriSenseCore" in call for call in fake_main.calls)


def test_wrapper_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeModule()
    monkeypatch.setattr(julia_bridge, "_initialized", True)
    monkeypatch.setattr(julia_bridge, "_agrisense_module", fake)

    graph_state = {"farm_id": str(uuid4())}

    status = julia_bridge.query_farm_status(graph_state, "zone-1")
    assert status["zone"] == "zone-1"

    schedule = julia_bridge.irrigation_schedule(graph_state, 7, {"rain": 0.0})
    assert schedule[0]["ok"] is True

    assert julia_bridge.nutrient_report(graph_state)[0]["nutrient"] == "ok"
    assert julia_bridge.yield_forecast(graph_state)[0]["yield"] == "ok"
    assert julia_bridge.detect_anomalies(graph_state)[0]["anomaly"] == "ok"

    cross = julia_bridge.cross_layer_query(graph_state, "soil", "weather")
    assert cross["layer_a"] == "soil"
    assert cross["layer_b"] == "weather"

    updated = julia_bridge.update_features(graph_state, "soil", "vertex-1", [0.1, 0.2])
    assert updated["updated"] is True

    trained = julia_bridge.train_yield_residual(graph_state, {"vertex-1": 1.2})
    assert trained["status"] == "trained"

    synthetic = julia_bridge.generate_synthetic("greenhouse", 10, 42)
    assert synthetic["status"] == "ok"
    assert synthetic["seed"] == 42
