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
