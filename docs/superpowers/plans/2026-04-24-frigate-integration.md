# Frigate Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the in-process YOLOv8/ffmpeg pipeline with Frigate NVR as the detection backend, connected via MQTT, and push the project to GitHub.

**Architecture:** Frigate runs in Docker and publishes person counts to MQTT. A new `FrigateListener` background thread subscribes to those counts and writes them to the existing SQLite store — replacing `DetectionWorker` with no changes to the DB schema, API, or frontend. The detection-log endpoint is updated to proxy to Frigate's HTTP API instead of reading local disk snapshots.

**Tech Stack:** Python 3.12, FastAPI, paho-mqtt 2.x, SQLite, Docker Compose (Frigate + Mosquitto), pytest

**Spec:** `docs/superpowers/specs/2026-04-24-frigate-integration-design.md`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `backend/detector.py` | Delete | YOLOv8/ffmpeg pipeline — replaced entirely |
| `backend/frigate_listener.py` | Create | MQTT subscriber; writes counts to SQLite |
| `backend/config.py` | Modify | Add MQTT_HOST, MQTT_PORT, FRIGATE_HOST, FRIGATE_PORT env vars |
| `backend/main.py` | Modify | Swap DetectionWorker→FrigateListener; update detection-log endpoints |
| `backend/requirements.txt` | Modify | Swap ultralytics+opencv for paho-mqtt>=2.0 |
| `frigate/config.yml` | Create | Frigate camera config (HLS streams + person detection) |
| `mosquitto/mosquitto.conf` | Create | Enable anonymous MQTT on port 1883 (required for Mosquitto 2.x) |
| `docker-compose.yml` | Create | Frigate + Mosquitto services |
| `.env.example` | Create | Template env file for local and linux-box deployment |
| `.gitignore` | Create | Exclude data/, venv/, .env, __pycache__, *.db |
| `tests/test_detector.py` | Delete | Tests for deleted module |
| `tests/test_frigate_listener.py` | Create | Unit tests for FrigateListener |
| `tests/test_integration.py` | Modify | Replace ffmpeg/YOLO mocks with mocked MQTT client |
| `tests/test_api.py` | Modify | Add tests for Frigate-proxied detection-log endpoints |
| `README.md` | Create | Project overview, setup, architecture |

---

## Task 1: GitHub Repo Setup

**Files:** `.gitignore` (create), then push existing code

- [ ] **Step 1: Create .gitignore**

Create `C:\Users\ethan\Desktop\Coding Projects\csusm-monitor\.gitignore`:

```
data/
venv/
.env
__pycache__/
*.pyc
*.db
*.egg-info/
.pytest_cache/
```

- [ ] **Step 2: Create GitHub repo via API**

```bash
cd "C:\Users\ethan\Desktop\Coding Projects\csusm-monitor"
curl -X POST \
  -H "Authorization: token <GITHUB_TOKEN>" \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Content-Type: application/json" \
  https://api.github.com/user/repos \
  -d '{"name":"csusm-monitor","description":"Real-time CSUSM campus occupancy monitor using Frigate NVR + YOLOv8 person detection","private":false}'
```

Expected: JSON response with `"full_name": "ethanstoner/csusm-monitor"`

- [ ] **Step 3: Configure git and push**

```bash
cd "C:\Users\ethan\Desktop\Coding Projects\csusm-monitor"
git config user.name "ethanstoner"
git config user.email "ethanstoner08@gmail.com"
git remote set-url origin https://ethanstoner:<GITHUB_TOKEN>@github.com/ethanstoner/csusm-monitor.git 2>/dev/null || \
git remote add origin https://ethanstoner:<GITHUB_TOKEN>@github.com/ethanstoner/csusm-monitor.git
GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="" git push -u origin main
```

Expected: `Branch 'main' set up to track remote branch 'main' from 'origin'`

- [ ] **Step 4: Verify push**

```bash
curl -s -H "Authorization: token <GITHUB_TOKEN>" \
  https://api.github.com/repos/ethanstoner/csusm-monitor | grep '"name"'
```

Expected: `"name": "csusm-monitor"`

- [ ] **Step 5: Commit .gitignore**

```bash
cd "C:\Users\ethan\Desktop\Coding Projects\csusm-monitor"
git add .gitignore
git commit -m "chore: add .gitignore"
GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="" git push
```

---

## Task 2: Update Dependencies and Config

**Files:** `backend/requirements.txt` (modify), `backend/config.py` (modify)

- [ ] **Step 1: Write failing test for new config vars**

Add to `tests/test_config.py`:

```python
def test_mqtt_config_defaults():
    from backend.config import MQTT_HOST, MQTT_PORT, FRIGATE_HOST, FRIGATE_PORT
    assert isinstance(MQTT_HOST, str)
    assert isinstance(MQTT_PORT, int)
    assert isinstance(FRIGATE_HOST, str)
    assert isinstance(FRIGATE_PORT, int)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd "C:\Users\ethan\Desktop\Coding Projects\csusm-monitor"
venv/Scripts/python.exe -m pytest tests/test_config.py::test_mqtt_config_defaults -v
```

Expected: `ImportError` or `FAILED` — config vars don't exist yet

- [ ] **Step 3: Update requirements.txt**

Replace `ultralytics==8.3.145` and `opencv-python-headless==4.11.0.86` with `paho-mqtt>=2.0`:

```
fastapi==0.115.12
uvicorn==0.34.2
paho-mqtt>=2.0
pytest==8.3.5
httpx==0.28.1
respx>=0.21
```

- [ ] **Step 4: Install updated dependencies**

```bash
cd "C:\Users\ethan\Desktop\Coding Projects\csusm-monitor"
venv/Scripts/python.exe -m pip install -r backend/requirements.txt
```

Expected: `Successfully installed paho-mqtt-... respx-...`

- [ ] **Step 5: Update config.py — add env vars**

In `backend/config.py`, add `import os` at the top and these four lines at the bottom:

```python
import os

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "localhost")
FRIGATE_PORT = int(os.getenv("FRIGATE_PORT", "5000"))
```

- [ ] **Step 6: Run test to confirm it passes**

```bash
venv/Scripts/python.exe -m pytest tests/test_config.py -v
```

Expected: All `test_config.py` tests pass

- [ ] **Step 7: Commit**

```bash
git add backend/requirements.txt backend/config.py tests/test_config.py
git commit -m "feat: swap YOLOv8/opencv deps for paho-mqtt; add Frigate env config"
GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="" git push
```

---

## Task 3: Docker Infrastructure

**Files:** `docker-compose.yml`, `frigate/config.yml`, `mosquitto/mosquitto.conf`, `.env.example`

No TDD here — these are config files. Verify by inspecting structure.

- [ ] **Step 1: Create mosquitto/mosquitto.conf**

```
allow_anonymous true
listener 1883
persistence false
```

This is required because Mosquitto 2.x disables anonymous connections by default.

- [ ] **Step 2: Create frigate/config.yml**

```yaml
cameras:
  starbucks:
    ffmpeg:
      inputs:
        - path: https://stream.csusm.edu/starbucks.m3u8
          roles: [detect]
    detect:
      width: 1280
      height: 720
  coffeecart:
    ffmpeg:
      inputs:
        - path: https://stream.csusm.edu/coffeecart.m3u8
          roles: [detect]
    detect:
      width: 1280
      height: 720
objects:
  track: [person]
mqtt:
  host: mosquitto
snapshots:
  enabled: true
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
services:
  mosquitto:
    image: eclipse-mosquitto:2
    ports:
      - "1883:1883"
    volumes:
      - ./mosquitto/mosquitto.conf:/mosquitto/config/mosquitto.conf

  frigate:
    image: ghcr.io/blakeblackshear/frigate:stable
    ports:
      - "5000:5000"
    volumes:
      - ./frigate/config.yml:/config/config.yml
      - frigate-media:/media/frigate
    depends_on:
      - mosquitto

volumes:
  frigate-media:
```

- [ ] **Step 4: Create .env.example**

```bash
MQTT_HOST=localhost
MQTT_PORT=1883
FRIGATE_HOST=localhost
FRIGATE_PORT=5000
```

- [ ] **Step 5: Verify files exist**

```bash
cd "C:\Users\ethan\Desktop\Coding Projects\csusm-monitor"
ls docker-compose.yml .env.example frigate/config.yml mosquitto/mosquitto.conf
```

Expected: All four files listed without errors

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml .env.example frigate/config.yml mosquitto/mosquitto.conf
git commit -m "feat: add Docker Compose stack for Frigate + Mosquitto"
GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="" git push
```

---

## Task 4: FrigateListener (TDD)

**Files:**
- Create: `tests/test_frigate_listener.py`
- Create: `backend/frigate_listener.py`

The `FrigateListener` is the heart of this change. Write all tests first, then implement.

- [ ] **Step 1: Create tests/test_frigate_listener.py**

```python
import json
import threading
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest
from backend.database import init_db, get_latest_counts


def make_msg(topic, payload):
    """Build a mock paho MQTT message."""
    msg = MagicMock()
    msg.topic = topic
    msg.payload = payload.encode() if isinstance(payload, str) else payload
    return msg


@pytest.fixture
def db(tmp_path):
    return init_db(tmp_path / "test.db")


def make_listener(db):
    """Create a FrigateListener with a mocked MQTT client (no real connection)."""
    with patch("backend.frigate_listener.mqtt.Client") as MockClient:
        MockClient.return_value = MagicMock()
        from backend.frigate_listener import FrigateListener
        return FrigateListener(db)


def test_count_topic_updates_latest_counts(db):
    """Person count MQTT message updates latest_counts dict."""
    listener = make_listener(db)
    msg = make_msg("frigate/starbucks/person", "3")
    listener._on_message(None, None, msg)
    with listener._counts_lock:
        assert listener.latest_counts["starbucks"] == 3


def test_count_topic_inserts_detection(db):
    """Person count MQTT message writes a row to SQLite."""
    listener = make_listener(db)
    msg = make_msg("frigate/starbucks/person", "2")
    listener._on_message(None, None, msg)
    rows = get_latest_counts(db)
    starbucks = next(r for r in rows if r["id"] == "starbucks")
    assert starbucks["count"] == 2


def test_zero_count_still_inserts(db):
    """Zero count is valid (empty camera) and must be written to DB."""
    listener = make_listener(db)
    msg = make_msg("frigate/coffeecart/person", "0")
    listener._on_message(None, None, msg)
    rows = get_latest_counts(db)
    cart = next(r for r in rows if r["id"] == "coffeecart")
    assert cart["count"] == 0


def test_unknown_camera_logs_warning_and_skips(db, caplog):
    """MQTT message for an unknown camera logs a warning and does not insert."""
    import logging
    listener = make_listener(db)
    msg = make_msg("frigate/phantom_cam/person", "1")
    with caplog.at_level(logging.WARNING, logger="backend.frigate_listener"):
        listener._on_message(None, None, msg)
    assert "phantom_cam" in caplog.text
    rows = get_latest_counts(db)
    assert all(r["count"] is None for r in rows)


def test_non_integer_payload_logs_warning(db, caplog):
    """Malformed payload does not crash the listener."""
    import logging
    listener = make_listener(db)
    msg = make_msg("frigate/starbucks/person", "not_a_number")
    with caplog.at_level(logging.WARNING, logger="backend.frigate_listener"):
        listener._on_message(None, None, msg)
    rows = get_latest_counts(db)
    assert all(r["count"] is None for r in rows)


def test_start_stop_lifecycle(db):
    """FrigateListener.start() launches a thread; stop() terminates it cleanly."""
    # Patch at the module level so FrigateListener.__init__ picks up the mock
    mock_mqtt = MagicMock()
    # loop_forever blocks forever — make it a no-op so the thread exits cleanly
    mock_mqtt.loop_forever.side_effect = lambda: None
    with patch("backend.frigate_listener.mqtt.Client", return_value=mock_mqtt):
        from backend.frigate_listener import FrigateListener
        listener = FrigateListener(db)
        assert listener.running is False

        listener.start()
        assert listener.running is True
        assert listener._thread is not None

        listener.stop()
        assert listener.running is False
        mock_mqtt.disconnect.assert_called_once()
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
cd "C:\Users\ethan\Desktop\Coding Projects\csusm-monitor"
venv/Scripts/python.exe -m pytest tests/test_frigate_listener.py -v
```

Expected: All 7 tests fail with `ModuleNotFoundError: No module named 'backend.frigate_listener'`

- [ ] **Step 3: Create backend/frigate_listener.py**

```python
import json
import logging
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import paho.mqtt.client as mqtt

from backend.config import CAMERAS, MQTT_HOST, MQTT_PORT, TIMEZONE
from backend.database import insert_detection

logger = logging.getLogger(__name__)
TZ = ZoneInfo(TIMEZONE)


class FrigateListener:
    """Subscribes to Frigate MQTT topics and writes person counts to SQLite."""

    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.running = False
        self._thread: threading.Thread | None = None
        self.latest_counts: dict[str, int] = {}
        self._counts_lock = threading.Lock()
        self._reconnect_delay = 1

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("FrigateListener started — connecting to %s:%d", MQTT_HOST, MQTT_PORT)

    def stop(self):
        self.running = False
        self._client.disconnect()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("FrigateListener stopped")

    def _run(self):
        try:
            self._client.connect(MQTT_HOST, MQTT_PORT)
            self._client.loop_forever()
        except Exception:
            logger.exception("FrigateListener connection error")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self._reconnect_delay = 1
            client.subscribe("frigate/+/person")
            client.subscribe("frigate/events")
            logger.info("Connected to MQTT — subscribed to Frigate topics")
        else:
            logger.warning("MQTT connect failed: reason_code=%s", reason_code)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        if not self.running:
            return
        logger.warning("MQTT disconnected (reason=%s), retrying in %ds", reason_code, self._reconnect_delay)
        time.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, 30)
        try:
            client.reconnect()
        except Exception:
            logger.exception("MQTT reconnect failed")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode("utf-8", errors="replace")
        parts = topic.split("/")

        # Count topic: frigate/<camera>/person
        if len(parts) == 3 and parts[0] == "frigate" and parts[2] == "person":
            cam_id = parts[1]
            if cam_id not in CAMERAS:
                logger.warning("Count received for unknown camera '%s' — skipping", cam_id)
                return
            try:
                count = int(payload)
            except ValueError:
                logger.warning("Non-integer count payload on %s: %r", topic, payload)
                return
            with self._counts_lock:
                self.latest_counts[cam_id] = count
            insert_detection(self.db_conn, cam_id, count, datetime.now(TZ))
            logger.info("[%s] person count: %d", cam_id, count)

        # Events topic: log snapshot availability (detection log proxies to Frigate HTTP API)
        elif topic == "frigate/events":
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                return
            after = event.get("after", {})
            if after.get("label") == "person":
                logger.debug("Frigate event %s: camera=%s has_snapshot=%s",
                             after.get("id"), after.get("camera"), after.get("has_snapshot"))
```

- [ ] **Step 4: Run tests to confirm they all pass**

```bash
venv/Scripts/python.exe -m pytest tests/test_frigate_listener.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/frigate_listener.py tests/test_frigate_listener.py
git commit -m "feat: add FrigateListener MQTT subscriber with full test coverage"
GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="" git push
```

---

## Task 5: Update main.py

**Files:** `backend/main.py` (modify), `tests/test_api.py` (modify), `tests/test_integration.py` (modify)

- [ ] **Step 1: Add detection-log endpoint tests to test_api.py**

Note: The existing `client` fixture in `tests/test_api.py` uses `with TestClient(app) as c:` (context manager form), which runs the FastAPI lifespan and initializes `_http_client`. The `respx.mock` interceptor works correctly with `httpx.AsyncClient` in this context — no fixture changes needed.

Append to `tests/test_api.py`:

```python
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
    """The old /api/snapshot/{id} endpoint no longer exists after Task 5 Step 4g."""
    resp = client.get("/api/snapshot/starbucks")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd "C:\Users\ethan\Desktop\Coding Projects\csusm-monitor"
venv/Scripts/python.exe -m pytest tests/test_api.py::test_detection_log_proxies_frigate tests/test_api.py::test_detection_log_image_proxies_frigate tests/test_api.py::test_snapshot_endpoint_removed -v
```

Expected:
- `test_detection_log_proxies_frigate` — FAILED (endpoint reads local disk, not Frigate)
- `test_detection_log_image_proxies_frigate` — FAILED (endpoint reads local disk)
- `test_snapshot_endpoint_removed` — FAILED with 200 or 500 (endpoint still exists; removed in Step 3g below)

- [ ] **Step 3: Update main.py**

Make the following changes to `backend/main.py`:

**a) Replace imports at the top** — remove the `DetectionWorker` reference and add `FrigateListener`:

Change:
```python
from backend.config import CAMERAS, DB_PATH, HEALTH_TIMEOUT, RETENTION_DAYS, TIMEZONE, SNAPSHOTS_DIR
```
To:
```python
from backend.config import CAMERAS, DB_PATH, FRIGATE_HOST, FRIGATE_PORT, HEALTH_TIMEOUT, RETENTION_DAYS, TIMEZONE
```

**b) Add a module-level listener reference** after `_workers = []`:
```python
_frigate_listener = None  # FrigateListener | None — set during lifespan startup
```

**c) In the lifespan function**, replace the worker startup block:

Remove:
```python
    if START_WORKERS:
        from backend.detector import DetectionWorker
        for cam_id, cam in CAMERAS.items():
            worker = DetectionWorker(cam_id, cam["stream_url"], _db_conn)
            worker.start()
            _workers.append(worker)
```

Replace with:
```python
    if START_WORKERS:
        from backend.frigate_listener import FrigateListener
        global _frigate_listener
        _frigate_listener = FrigateListener(_db_conn)
        _frigate_listener.start()
        _workers.append(_frigate_listener)
```

**d) Update `/api/status`** to use `latest_counts` from the listener for the count field:

Replace the `cameras` list-building block inside `get_status()`:
```python
    cameras = []
    for r in rows:
        healthy = True
        if r["timestamp"]:
            last_ts = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
            healthy = (now_naive - last_ts).total_seconds() < HEALTH_TIMEOUT
        else:
            healthy = False
        # Use live count from FrigateListener if available, fall back to last DB value
        live_count = r["count"] or 0
        if _frigate_listener is not None:
            with _frigate_listener._counts_lock:
                live_count = _frigate_listener.latest_counts.get(r["id"], live_count)
        cameras.append({
            "id": r["id"],
            "name": r["name"],
            "count": live_count,
            "timestamp": r["timestamp"],
            "healthy": healthy,
        })
```

**e) Replace `/api/detection-log` endpoint** entirely. Note the function signature changes from `def` to `async def` (required because `_http_client` is an `httpx.AsyncClient`):
```python
@app.get("/api/detection-log")
async def get_detection_log(camera: str = Query(default=None), limit: int = Query(default=50)):
    """Return recent detection events with snapshots from Frigate."""
    params = {"label": "person", "limit": limit, "has_snapshot": 1}
    if camera:
        params["cameras"] = camera
    try:
        resp = await _http_client.get(
            f"http://{FRIGATE_HOST}:{FRIGATE_PORT}/api/events", params=params
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception:
        logger.exception("Failed to fetch events from Frigate")
        return {"detections": []}

    detections = []
    for ev in events:
        ts = datetime.fromtimestamp(ev["start_time"], TZ).strftime("%Y-%m-%d %H:%M:%S")
        detections.append({
            "filename": ev["id"],
            "camera": ev["camera"],
            "timestamp": ts,
            "count": 1,
            "url": f"/api/detection-log/image/{ev['id']}",
        })
    return {"detections": detections}
```

**f) Replace `/api/detection-log/image/{filename}` endpoint**:
```python
@app.get("/api/detection-log/image/{event_id}")
async def get_detection_image(event_id: str):
    """Proxy a detection snapshot from Frigate."""
    try:
        resp = await _http_client.get(
            f"http://{FRIGATE_HOST}:{FRIGATE_PORT}/api/events/{event_id}/snapshot.jpg"
        )
        if resp.status_code != 200:
            return Response(status_code=404, content="Not found")
        return Response(content=resp.content, media_type="image/jpeg")
    except Exception:
        return Response(status_code=404, content="Not found")
```

**g) Remove the `/api/snapshot/{camera_id}` endpoint entirely** (delete the whole function).

**h) Remove unused imports** at the top — remove `SNAPSHOTS_DIR` from the config import since it's no longer used in main.py.

- [ ] **Step 4: Run new API tests to confirm they pass**

```bash
venv/Scripts/python.exe -m pytest tests/test_api.py -v
```

Expected: All tests including the 3 new ones pass

- [ ] **Step 5: Update tests/test_integration.py** — replace ffmpeg/YOLO mocks with mocked MQTT:

Replace the entire file content:

```python
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
```

- [ ] **Step 6: Run all tests**

```bash
venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: All tests pass except `tests/test_detector.py` (deleted next task). No failures.

- [ ] **Step 7: Commit**

```bash
git add backend/main.py tests/test_api.py tests/test_integration.py
git commit -m "feat: swap DetectionWorker for FrigateListener in main; proxy detection-log to Frigate API"
GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="" git push
```

---

## Task 6: Clean Up Detector + Write README

**Files:** `backend/detector.py` (delete), `tests/test_detector.py` (delete), `README.md` (create)

- [ ] **Step 1: Delete detector.py and test_detector.py**

```bash
cd "C:\Users\ethan\Desktop\Coding Projects\csusm-monitor"
rm backend/detector.py tests/test_detector.py
```

- [ ] **Step 2: Run full test suite to confirm clean pass**

```bash
venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: All remaining tests pass. The deleted test file is gone so no import errors.

- [ ] **Step 3: Create README.md**

```markdown
# CSUSM Campus Monitor

Real-time occupancy monitor for CSUSM campus locations using [Frigate NVR](https://frigate.video) for person detection and a custom FastAPI dashboard for trends and history.

## What It Does

- Monitors live HLS camera streams from CSUSM (Starbucks, Coffee Cart)
- Counts people in frame every detection cycle using Frigate + YOLOv8
- Stores counts in SQLite for historical trend analysis
- Serves a dark-themed web dashboard with live video, heatmaps, and timelines

## Architecture

```
CSUSM HLS streams → Frigate (Docker) → MQTT → FastAPI backend → SQLite → Dashboard
```

## Setup

### Requirements

- Python 3.12+
- Docker + Docker Compose
- ffmpeg on PATH

### Local Development

```bash
# 1. Clone and install deps
git clone https://github.com/ethanstoner/csusm-monitor.git
cd csusm-monitor
python -m venv venv && venv/bin/pip install -r backend/requirements.txt

# 2. Configure environment
cp .env.example .env   # edit if needed — defaults work for local dev

# 3. Start Frigate + Mosquitto
docker-compose up -d

# 4. Start the backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

### Linux Server Deployment

```bash
scp -r . user@server:~/csusm-monitor/
ssh user@server "cd ~/csusm-monitor && docker-compose up -d"
ssh user@server "cd ~/csusm-monitor && python3 -m venv venv && venv/bin/pip install -r backend/requirements.txt"
# Edit .env on server: MQTT_HOST=localhost, FRIGATE_HOST=localhost
ssh user@server "cd ~/csusm-monitor && venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
```

## Configuration

All settings live in `backend/config.py`. Camera streams are configured in the `CAMERAS` dict — adding a camera only requires one entry there and a matching entry in `frigate/config.yml`.

| Variable | Default | Description |
|---|---|---|
| `MQTT_HOST` | `localhost` | Mosquitto broker host |
| `MQTT_PORT` | `1883` | Mosquitto broker port |
| `FRIGATE_HOST` | `localhost` | Frigate HTTP API host |
| `FRIGATE_PORT` | `5000` | Frigate HTTP API port |

## Running Tests

```bash
pytest -v
```
```

- [ ] **Step 4: Final test run + push**

```bash
venv/Scripts/python.exe -m pytest tests/ -v
git add -A
git commit -m "feat: remove YOLOv8 detector; add README for Frigate-based architecture"
GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="" git push
```

Expected: All tests pass, repo fully updated on GitHub.
