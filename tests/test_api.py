import pytest
from datetime import datetime
from fastapi.testclient import TestClient

@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test client with a temporary database."""
    import backend.config as config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("backend.main.START_WORKERS", False)
    from backend.main import app
    with TestClient(app) as c:
        yield c

@pytest.fixture
def seeded_client(tmp_path, monkeypatch):
    """Test client with some detection data seeded."""
    import backend.config as config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("backend.main.START_WORKERS", False)
    from backend.main import app, get_db
    with TestClient(app) as c:
        conn = get_db()
        from backend.database import insert_detection
        insert_detection(conn, "starbucks", 5, datetime(2026, 4, 12, 10, 0, 0))
        insert_detection(conn, "starbucks", 3, datetime(2026, 4, 12, 10, 0, 30))
        insert_detection(conn, "coffeecart", 2, datetime(2026, 4, 12, 10, 0, 15))
        yield c

def test_get_status(seeded_client):
    resp = seeded_client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "cameras" in data
    assert len(data["cameras"]) == 2
    starbucks = [c for c in data["cameras"] if c["id"] == "starbucks"][0]
    assert starbucks["count"] == 3
    assert "healthy" in starbucks

def test_get_cameras(client):
    resp = client.get("/api/cameras")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["cameras"]) == 2

def test_get_heatmap(seeded_client):
    resp = seeded_client.get("/api/history/heatmap?camera=starbucks&days=30")
    assert resp.status_code == 200
    data = resp.json()
    assert data["camera"] == "starbucks"
    assert "data" in data

def test_get_timeline(seeded_client):
    resp = seeded_client.get("/api/history/timeline?camera=starbucks&date=2026-04-12")
    assert resp.status_code == 200
    data = resp.json()
    assert data["camera"] == "starbucks"
    assert data["date"] == "2026-04-12"
    assert "data" in data

def test_root_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_detection_log_proxies_frigate(client):
    """GET /api/detection-log proxies to Frigate events API."""
    import respx
    import httpx
    frigate_events = [
        {
            "id": "abc123",
            "camera": "starbucks",
            "start_time": 1745000000.0,
            "label": "person",
            "has_snapshot": True,
            "top_score": 0.92,
        }
    ]
    with respx.mock:
        respx.get("http://localhost:5000/api/events").mock(
            return_value=httpx.Response(200, json=frigate_events)
        )
        resp = client.get("/api/detection-log?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["detections"]) == 1
    assert data["detections"][0]["camera"] == "starbucks"
    assert data["detections"][0]["filename"] == "abc123"
    assert "/api/detection-log/image/abc123" in data["detections"][0]["url"]


def test_detection_log_image_proxies_frigate(client):
    """GET /api/detection-log/image/{id} proxies to Frigate snapshot."""
    import respx
    import httpx
    with respx.mock:
        respx.get("http://localhost:5000/api/events/abc123/snapshot.jpg").mock(
            return_value=httpx.Response(200, content=b"fakejpeg", headers={"content-type": "image/jpeg"})
        )
        resp = client.get("/api/detection-log/image/abc123")
    assert resp.status_code == 200
    assert resp.content == b"fakejpeg"


def test_snapshot_endpoint_removed(client):
    """The old /api/snapshot/{id} endpoint no longer exists."""
    resp = client.get("/api/snapshot/starbucks")
    assert resp.status_code == 404
