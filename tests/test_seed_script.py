from __future__ import annotations

from scripts.seed_db import _build_topology, _is_missing


def _synthetic_stub() -> dict[str, object]:
    return {
        "topology": {
            "zones": [
                {"zone_id": "zone_1", "zone_type": "greenhouse"},
                {"zone_id": "zone_2", "zone_type": "greenhouse"},
                {"zone_id": "zone_3", "zone_type": "open_field"},
                {"zone_id": "zone_4", "zone_type": "open_field"},
                {"zone_id": "zone_5", "zone_type": "open_field"},
                {"zone_id": "zone_6", "zone_type": "open_field"},
            ],
            "soil_sensors": {
                "sensor_id": [f"soil_sensor_{idx}" for idx in range(1, 7)],
                "zone_id": [f"zone_{idx}" for idx in range(1, 7)],
            },
            "weather_stations": {"station_id": ["weather_station_1", "weather_station_2"]},
        },
        "layers": {
            "lighting": {"n_sensors": 4},
            "vision": {
                "n_beds": 12,
                "n_cameras": 4,
                "camera_for_bed": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            },
        },
    }


def test_build_topology_hybrid_contract() -> None:
    artifacts = _build_topology(_synthetic_stub())
    assert len(artifacts.zone_ids) == 6
    assert len(artifacts.valve_ids) == 6
    assert len(artifacts.camera_ids) == 4
    assert len(artifacts.bed_ids) == 12
    assert len(artifacts.fixture_ids) == 4


def test_is_missing_helper() -> None:
    assert _is_missing(None) is True
    assert _is_missing(float("nan")) is True
    assert _is_missing(1.23) is False
