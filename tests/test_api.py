import time

import pytest
from datetime import datetime
from fastapi.testclient import TestClient

@pytest.fixture(autouse=True)
def _clean_camera_health():
    """Camera health is module-level state shared by every worker thread.

    Left over between tests it makes ordering matter, so clear it around each.
    """
    from backend.detector import camera_health, _health_lock
    with _health_lock:
        camera_health.clear()
    yield
    with _health_lock:
        camera_health.clear()


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


def test_camera_hours(client):
    """GET /api/cameras/{id}/hours returns the analytics window."""
    resp = client.get("/api/cameras/coffeecart/hours")
    assert resp.status_code == 200
    assert resp.json()["open_hours"] == [7, 17]


def test_camera_hours_unknown_camera(client):
    resp = client.get("/api/cameras/nope/hours")
    assert resp.status_code == 404


def test_snapshot_endpoint_removed(client):
    """The old /api/snapshot/{id} endpoint no longer exists."""
    resp = client.get("/api/snapshot/starbucks")
    assert resp.status_code == 404


def test_proxy_serves_a_live_manifest(client):
    """A manifest with segments passes through, trimmed to the live window."""
    import respx
    import httpx
    manifest = (
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:5\n"
        "#EXT-X-MEDIA-SEQUENCE:100\n"
        + "".join(f"#EXTINF:4.0,\nseg{i}.ts\n" for i in range(10))
    )
    with respx.mock:
        respx.get("https://stream.csusm.edu/coffeecart.m3u8").mock(
            return_value=httpx.Response(200, content=manifest.encode())
        )
        resp = client.get("/api/stream/coffeecart/coffeecart.m3u8")
    assert resp.status_code == 200
    assert resp.text.count("#EXTINF") == 6


def test_proxy_reports_offline_for_empty_manifest(client):
    """A reachable-but-empty manifest is a dead camera, not a 200.

    CSUSM serves a zero-byte .m3u8 for a camera that has been taken down. The
    proxy used to forward that as 200 with a body of b'\\n', so hls.js hung on
    a manifest it could never parse and the card span forever with no way for
    the dashboard to know why.
    """
    import respx
    import httpx
    with respx.mock:
        respx.get("https://stream.csusm.edu/starbucks.m3u8").mock(
            return_value=httpx.Response(200, content=b"")
        )
        resp = client.get("/api/stream/starbucks/starbucks.m3u8")
    assert resp.status_code == 502
    assert resp.headers["x-stream-status"] == "offline"


def test_proxy_reports_offline_when_upstream_errors(client):
    """An upstream 404/500 is surfaced as offline rather than a broken manifest."""
    import respx
    import httpx
    with respx.mock:
        respx.get("https://stream.csusm.edu/starbucks.m3u8").mock(
            return_value=httpx.Response(404, content=b"not found")
        )
        resp = client.get("/api/stream/starbucks/starbucks.m3u8")
    assert resp.status_code == 502
    assert resp.headers["x-stream-status"] == "offline"


def test_proxy_survives_upstream_connection_failure(client):
    """A network error must not become a 500 from our own API."""
    import respx
    import httpx
    with respx.mock:
        respx.get("https://stream.csusm.edu/starbucks.m3u8").mock(
            side_effect=httpx.ConnectError("no route")
        )
        resp = client.get("/api/stream/starbucks/starbucks.m3u8")
    assert resp.status_code == 502
    assert resp.headers["x-stream-status"] == "offline"


def test_proxy_survives_a_segment_fetch_failing(client):
    """A .ts that vanishes mid-playback is an outage, not a fault in this API."""
    import respx
    import httpx
    with respx.mock:
        respx.get("https://stream.csusm.edu/coffeecart.000.ts").mock(
            side_effect=httpx.ConnectError("no route")
        )
        resp = client.get("/api/stream/coffeecart/coffeecart.000.ts")
    assert resp.status_code == 502
    assert resp.headers["x-stream-status"] == "offline"


def test_proxy_rejects_an_unknown_camera(client):
    """The proxy builds an upstream URL from config; unknown ids never get there."""
    resp = client.get("/api/stream/not-a-camera/x.m3u8")
    assert resp.status_code == 404


def test_cameras_report_offline_after_sustained_capture_failure(client):
    """A camera whose stream has failed repeatedly is labelled offline.

    The dashboard needs this to render 'offline' instead of mounting a video
    element for a stream that will never produce a frame.
    """
    from backend.detector import OFFLINE_AFTER_FAILURES, record_camera_health

    record_camera_health("starbucks", ok=False, error="ffmpeg returned no data")
    for _ in range(OFFLINE_AFTER_FAILURES):
        record_camera_health("starbucks", ok=False, error="ffmpeg returned no data")
    record_camera_health("coffeecart", ok=True)

    cams = {c["id"]: c for c in client.get("/api/cameras").json()["cameras"]}
    assert cams["starbucks"]["stream_status"] == "offline"
    assert cams["starbucks"]["last_error"] == "ffmpeg returned no data"
    assert cams["coffeecart"]["stream_status"] == "live"


def test_camera_health_clears_on_first_good_frame(client):
    """One successful capture takes a camera back out of the offline state."""
    from backend.detector import OFFLINE_AFTER_FAILURES, record_camera_health

    for _ in range(OFFLINE_AFTER_FAILURES + 1):
        record_camera_health("starbucks", ok=False, error="ffmpeg returned no data")
    record_camera_health("starbucks", ok=True)

    cams = {c["id"]: c for c in client.get("/api/cameras").json()["cameras"]}
    assert cams["starbucks"]["stream_status"] == "live"
    assert cams["starbucks"]["last_error"] is None


def test_offline_camera_reports_no_count(seeded_client):
    """An offline camera must not keep publishing its last known count.

    get_latest_counts falls back to the newest row in the table, which for a
    camera that has been down for months is a number from months ago. Served as
    `count` it read as current occupancy and was summed into the campus total.
    """
    from backend.detector import OFFLINE_AFTER_FAILURES, record_camera_health

    for _ in range(OFFLINE_AFTER_FAILURES + 1):
        record_camera_health("starbucks", ok=False, error="dead")

    cams = {c["id"]: c for c in seeded_client.get("/api/status").json()["cameras"]}
    assert cams["starbucks"]["count"] is None, "stale count served as live occupancy"
    assert cams["coffeecart"]["count"] == 2, "a live camera still reports its count"


def test_status_reports_stream_status_too(client):
    """/api/status carries the same offline signal the camera list does."""
    from backend.detector import OFFLINE_AFTER_FAILURES, record_camera_health

    for _ in range(OFFLINE_AFTER_FAILURES + 1):
        record_camera_health("starbucks", ok=False, error="dead")

    cams = {c["id"]: c for c in client.get("/api/status").json()["cameras"]}
    assert cams["starbucks"]["stream_status"] == "offline"


def test_observations_endpoint_is_empty_without_a_grounding_service(client):
    """The open-vocabulary panel degrades to 'nothing yet', not an error."""
    resp = client.get("/api/observations")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": False, "observations": []}


def test_observations_endpoint_returns_the_latest_per_label(client):
    from datetime import datetime as dt
    from backend.database import insert_observation
    from backend.main import get_db

    conn = get_db()
    insert_observation(conn, "coffeecart", "bicycle", 3, dt(2026, 4, 12, 10, 0, 0), latency_ms=900)
    insert_observation(conn, "coffeecart", "bicycle", 7, dt(2026, 4, 12, 10, 5, 0), latency_ms=880)
    insert_observation(conn, "coffeecart", "person", 2, dt(2026, 4, 12, 10, 5, 5), latency_ms=910)

    data = client.get("/api/observations").json()["observations"]
    by_label = {o["label"]: o for o in data}
    assert by_label["bicycle"]["count"] == 7
    assert by_label["person"]["count"] == 2
    assert by_label["bicycle"]["latency_ms"] == 880


def test_observations_endpoint_filters_by_camera(client):
    from datetime import datetime as dt
    from backend.database import insert_observation
    from backend.main import get_db

    conn = get_db()
    insert_observation(conn, "coffeecart", "bicycle", 3, dt(2026, 4, 12, 10, 0, 0))
    insert_observation(conn, "starbucks", "bicycle", 8, dt(2026, 4, 12, 10, 0, 0))

    data = client.get("/api/observations?camera=starbucks").json()["observations"]
    assert [o["count"] for o in data] == [8]


def test_observation_history_endpoint(client):
    from datetime import datetime as dt, timedelta as td
    from zoneinfo import ZoneInfo
    from backend.database import insert_observation
    from backend.main import get_db

    conn = get_db()
    base = (dt.now(ZoneInfo("America/Los_Angeles")) - td(days=1)).replace(
        hour=11, minute=0, second=0, microsecond=0)
    for i in range(3):
        insert_observation(conn, "coffeecart", "bicycle", 5, base.replace(second=i))

    data = client.get("/api/observations/history?camera=coffeecart&label=bicycle").json()
    assert data["label"] == "bicycle"
    assert data["data"] == [{"hour": 11, "avg_count": 5.0, "samples": 3}]


def test_cleanup_thread_does_not_outlive_shutdown(tmp_path, monkeypatch):
    """The retention thread holds _db_conn — it must die before the app closes it."""
    import threading
    import backend.config as config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("backend.main.START_WORKERS", False)
    from backend.main import app

    def cleanup_threads():
        return [t for t in threading.enumerate() if t.name == "daily-cleanup" and t.is_alive()]

    assert not cleanup_threads()
    with TestClient(app):
        assert len(cleanup_threads()) == 1, "cleanup thread should be running while the app is up"
    assert not cleanup_threads(), "cleanup thread survived shutdown holding the DB connection"


def test_cleanup_thread_still_sweeps(tmp_path, monkeypatch):
    """Shortening the interval proves the loop actually runs its sweep."""
    import backend.config as config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("backend.main.START_WORKERS", False)
    monkeypatch.setattr("backend.main.CLEANUP_INTERVAL", 0.1)
    calls = []
    monkeypatch.setattr("backend.main.cleanup_old_data", lambda *a: calls.append(a) or 0)

    from backend.main import app
    with TestClient(app):
        deadline = time.time() + 5
        while len(calls) < 2 and time.time() < deadline:
            time.sleep(0.05)
    assert len(calls) >= 2, "cleanup loop never ran a periodic sweep"
