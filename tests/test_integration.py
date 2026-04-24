"""Smoke test: app starts with mocked FrigateListener, APIs respond, data flows to DB."""
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


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

    with patch("backend.frigate_listener.mqtt.Client") as MockMqttClient:
        MockMqttClient.return_value = MagicMock()
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

    today = datetime.now().strftime("%Y-%m-%d")
    resp = live_client.get(f"/api/history/timeline?camera=starbucks&date={today}")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) > 0
