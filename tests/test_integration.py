"""Smoke test: start the app, verify APIs respond, verify detection worker lifecycle."""
import time
from datetime import datetime
from unittest.mock import patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def live_client(tmp_path, monkeypatch):
    """Client with detection workers running (mocked ffmpeg + YOLO)."""
    import backend.config as config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "DETECTION_INTERVAL", 1)  # Fast for testing

    # Mock capture_frame to return a blank image
    fake_frame = np.full((480, 640, 3), 128, dtype=np.uint8)  # gray, not black
    with patch("backend.detector.capture_frame", return_value=fake_frame):
        with patch("backend.detector.detect_people", return_value=(3, [])):
            from backend.main import app
            with TestClient(app) as c:
                time.sleep(3)  # Let workers run a few cycles
                yield c

def test_full_pipeline(live_client):
    # Status should have detections
    resp = live_client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    for cam in data["cameras"]:
        assert cam["count"] >= 0
        assert cam["healthy"] is True

    # Cameras endpoint
    resp = live_client.get("/api/cameras")
    assert resp.status_code == 200
    assert len(resp.json()["cameras"]) == 2

    # Timeline should have data for today
    today = datetime.now().strftime("%Y-%m-%d")
    resp = live_client.get(f"/api/history/timeline?camera=starbucks&date={today}")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) > 0
