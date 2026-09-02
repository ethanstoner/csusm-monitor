"""Smoke test: app starts with mocked FrigateListener, APIs respond, data flows to DB."""
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.config import TIMEZONE


def make_msg(topic, payload):
    msg = MagicMock()
    msg.topic = topic
    msg.payload = payload.encode()
    return msg


@pytest.fixture
def live_client(tmp_path, monkeypatch):
    """Client with FrigateListener running (MQTT connection mocked)."""
    import backend.config as config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setenv("MQTT_HOST", "localhost")

    # This test exercises the FrigateListener path, but START_WORKERS also has
    # to stay True for the listener to start — which incidentally spun up real
    # YOLO workers against the live CSUSM streams, wrote into data/snapshots,
    # and added ~20s to teardown. Stub them out.
    #
    # Same for the collectors: each one's first cycle runs immediately, so the
    # suite was calling out to open-meteo, parkingstatus.csusm.edu, NCTD's GTFS
    # feed and m.csusm.edu. Keep the real objects (the transit API route probes
    # them by attribute) but make the network cycle a no-op.
    for name in ("WeatherCollector", "ParkingCollector", "AirQualityCollector",
                 "TransitCollector", "EventsCollector"):
        monkeypatch.setattr(f"backend.collectors.{name}.collect", lambda self: None)

    with patch("backend.frigate_listener.mqtt.Client") as MockMqttClient, \
         patch("backend.detector.DetectionWorker") as MockWorker:
        MockMqttClient.return_value = MagicMock()
        MockWorker.return_value = MagicMock()
        from backend.main import app
        with TestClient(app) as c:
            # Access the module attribute directly (not a value copy) to get
            # the post-lifespan value that was assigned during app startup
            import backend.main as main_mod
            if main_mod._frigate_listener is not None:
                main_mod._frigate_listener._on_message(None, None, make_msg("frigate/starbucks/person", "3"))
                main_mod._frigate_listener._on_message(None, None, make_msg("frigate/coffeecart/person", "1"))
            yield c


def test_full_pipeline(live_client):
    """Detection counts flow through FrigateListener into DB and are returned by API."""
    resp = live_client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["cameras"]) == 2
    for cam in data["cameras"]:
        assert cam["count"] >= 0
        assert "healthy" in cam

    resp = live_client.get("/api/cameras")
    assert resp.status_code == 200
    assert len(resp.json()["cameras"]) == 2

    # Detections are stored as naive Pacific (database.insert_detection), so the
    # date to ask for is today *there*, not on whatever clock the runner keeps.
    # CI runs in UTC, where every run between midnight and ~08:00 UTC is still
    # the previous day in California and this query came back empty.
    today = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d")
    resp = live_client.get(f"/api/history/timeline?camera=starbucks&date={today}")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) > 0
