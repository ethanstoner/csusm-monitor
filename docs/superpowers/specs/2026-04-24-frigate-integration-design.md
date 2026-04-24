# Frigate Integration Design

**Date:** 2026-04-24
**Project:** csusm-monitor
**Status:** Approved

## Overview

Replace the current in-process YOLOv8/ffmpeg detection pipeline with Frigate — a dedicated open-source NVR — as the detection backend. Frigate runs in Docker (target: linux-box at 192.168.1.199), publishes person counts and event metadata via MQTT, and the FastAPI backend consumes those messages to populate the existing SQLite store. The dashboard, REST API, and occupancy trend charts require no changes. The detection-log image URL format changes internally but the frontend consumes it transparently via the dynamic `url` field returned by the API.

## Goals

- Remove heavy ML dependencies (ultralytics, opencv) from the FastAPI process
- Delegate detection, snapshot saving, and stream management to Frigate
- Keep the existing SQLite schema, all REST API endpoints, and frontend intact
- Make linux-box deployment a config-only change (`.env` file)
- Push the project to GitHub for version control and portfolio visibility

## Non-Goals

- Using Frigate's built-in web UI (keep the custom dashboard)
- Adding new detection labels beyond `person`
- Integrating Frigate's recording/clips features
- Frigate GenAI descriptions (out of scope for now)

## Architecture

```
CSUSM HLS streams ──► Frigate (Docker, linux-box)
                           │  detects persons
                           │  saves snapshots
                           ▼
                     Mosquitto MQTT (Docker, linux-box)
                           │
                  frigate/<cam>/person  ← live counts (integer string)
                  frigate/events        ← event metadata + snapshot IDs
                           │
                           ▼
                     FastAPI backend
                      FrigateListener thread ──► SQLite (unchanged)
                      REST API (unchanged)  ──────► Dashboard
                           │
                           ▼  (HLS proxy, unchanged)
                     CSUSM streams ──► Dashboard live video
```

## Components

### Deleted

- `backend/detector.py` — YOLOv8 model, ffmpeg capture, StaticObjectFilter, DetectionWorker

### New Files

**`backend/frigate_listener.py`**

A single background thread running a `paho-mqtt` client. Responsibilities:

- Use `paho-mqtt` v2 API: `mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)`
- Subscribe to `frigate/+/person` — Frigate publishes the integer count of detected persons per camera on this topic (e.g., `frigate/starbucks/person` = `"2"`). On each message, parse the payload as int and call `insert_detection(db, cam_id, count, datetime.now(TZ))`. Note: pass `datetime.now(TZ)` (timezone-aware Pacific time) — `insert_detection` accepts both naive and aware datetimes and converts to Pacific time internally.
- Subscribe to `frigate/events` — parse JSON payload. On `type=new` or `type=update` where `after.label == "person"` and `after.has_snapshot == true`, no local caching is needed — the detection log proxies directly to Frigate's HTTP API. The `frigate/events` subscription is kept only to update `latest_counts` if a count topic message was missed; it is not used for snapshot storage.
- Expose `latest_counts: dict[str, int]` protected by a `threading.Lock` (`_counts_lock`). All reads and writes to `latest_counts` must hold this lock.
- Expose `.start()` method: starts the MQTT loop thread (mirrors `DetectionWorker.start()`)
- Expose `.stop()` method: calls `client.disconnect()` and joins the thread with a 10s timeout (mirrors `DetectionWorker.stop()`)
- Auto-reconnect on disconnect: implement `on_disconnect` callback with exponential backoff (1s, 2s, 4s … max 30s) before calling `client.reconnect()`

**`frigate/config.yml`**

Frigate configuration template checked into the repo. The HLS stream URLs match the values already in `backend/config.py`'s `CAMERAS` dict — no changes needed there:
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

**`mosquitto/mosquitto.conf`**

Required config file for the Mosquitto 2.x Docker image (anonymous connections are disabled by default since Mosquitto 2.0 and must be explicitly enabled):
```
allow_anonymous true
listener 1883
persistence false
```

**`docker-compose.yml`**

Runs Frigate + Mosquitto. FastAPI runs separately outside Docker.
```yaml
services:
  mosquitto:
    image: eclipse-mosquitto:2
    ports: ["1883:1883"]
    volumes:
      - ./mosquitto/mosquitto.conf:/mosquitto/config/mosquitto.conf
  frigate:
    image: ghcr.io/blakeblackshear/frigate:stable
    ports: ["5000:5000"]
    volumes:
      - ./frigate/config.yml:/config/config.yml
      - frigate-media:/media/frigate
    depends_on: [mosquitto]
volumes:
  frigate-media:
```

**`.env.example`**

```bash
MQTT_HOST=localhost
MQTT_PORT=1883
FRIGATE_HOST=localhost
FRIGATE_PORT=5000
```

### Modified Files

**`backend/config.py`**

Add four new env-driven settings read via `os.getenv` (add `import os` at top):
```python
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "localhost")
FRIGATE_PORT = int(os.getenv("FRIGATE_PORT", "5000"))
```

**`backend/main.py`**

- Remove import of `DetectionWorker`; import `FrigateListener` instead
- In lifespan startup: instantiate one `FrigateListener(db_conn)` instead of one `DetectionWorker` per camera. Call `.start()`. Append to `_workers` list.
- Lifespan shutdown loop `worker.stop()` works unchanged — `FrigateListener` exposes the same interface.
- Update `/api/detection-log`: instead of globbing local `SNAPSHOTS_DIR`, call `GET http://{FRIGATE_HOST}:{FRIGATE_PORT}/api/events?label=person&limit={limit}&has_snapshot=1` (and optionally `&cameras={camera}` when filter is set). Map Frigate event fields to the same response shape: `{filename: event.id, camera: event.camera, timestamp: event.start_time, count: 1, url: f"/api/detection-log/image/{event.id}"}`. The frontend uses `d.url` dynamically so no frontend changes are needed.
- Update `/api/detection-log/image/{event_id}`: proxy `GET http://{FRIGATE_HOST}:{FRIGATE_PORT}/api/events/{event_id}/snapshot.jpg`. Return 404 if Frigate returns non-200.
- Remove `/api/snapshot/{camera_id}` endpoint (Frigate handles snapshots natively; the endpoint is no longer backed by anything).
- Keep `/api/stream/` HLS proxy unchanged — live video still proxies CSUSM streams directly.
- Update `/api/status`: read `latest_counts` from the `FrigateListener` instance (with lock) rather than only from the DB for the count field. Keep existing `healthy` logic (timestamp staleness check) unchanged.

**`backend/requirements.txt`**

Remove: `ultralytics`, `opencv-python-headless`
Add: `paho-mqtt>=2.0`

**`.gitignore`**

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

## Data Flow

1. Frigate polls CSUSM HLS streams every detection cycle
2. Person detected → Frigate publishes `frigate/starbucks/person` = `"2"` (string integer)
3. `FrigateListener.on_message` receives count → acquires `_counts_lock` → updates `latest_counts["starbucks"] = 2` → releases lock → calls `insert_detection(db, "starbucks", 2, datetime.now(TZ))`
4. Frigate publishes `frigate/events` with `type=update`, `after.label=person`, `after.has_snapshot=true`, `after.id=<event_id>`
5. `FrigateListener` caches `{id, camera, start_time, top_score}` in `_event_cache`
6. Dashboard polls `/api/status` → acquires `_counts_lock`, reads `latest_counts`, releases lock; applies existing timestamp/health logic
7. User opens Activity Log → `GET /api/detection-log` → FastAPI proxies to `http://frigate:5000/api/events?label=person&...`
8. User views snapshot → `GET /api/detection-log/image/{event_id}` → FastAPI proxies to `http://frigate:5000/api/events/{event_id}/snapshot.jpg`

## Error Handling

- **Frigate/MQTT offline:** `FrigateListener` reconnects automatically with exponential backoff; `/api/status` returns last DB timestamp with `healthy: false` if stale (existing `HEALTH_TIMEOUT` logic unchanged)
- **Frigate snapshot unavailable:** `/api/detection-log/image/` returns 404 if Frigate returns non-200
- **Unknown camera in MQTT message:** log a warning if `cam_id` not in `CAMERAS` config; skip the insert

## Deployment

### Local Development (now)

```bash
# 1. Copy env config
cp .env.example .env          # defaults: localhost for all hosts

# 2. Start Frigate + Mosquitto
docker-compose up -d

# 3. Install updated dependencies
pip install -r backend/requirements.txt

# 4. Start FastAPI
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Linux-box Migration (later)

```bash
# 1. Copy project to linux-box
scp -r . linux-box:~/csusm-monitor/

# 2. On linux-box: start Docker stack
ssh linux-box "cd ~/csusm-monitor && docker-compose up -d"

# 3. On linux-box: set up Python env and install deps
ssh linux-box "cd ~/csusm-monitor && python3 -m venv venv && venv/bin/pip install -r backend/requirements.txt"

# 4. Copy .env (or create on remote)
# Edit .env: MQTT_HOST=localhost, FRIGATE_HOST=localhost (co-located)
scp .env linux-box:~/csusm-monitor/.env

# 5. Start FastAPI on linux-box
ssh linux-box "cd ~/csusm-monitor && venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
```

Only `.env` changes between environments — no code modifications required.

## GitHub Setup

- Repo: `github.com/ethanstoner/csusm-monitor` (public)
- Branch: `main`
- `.gitignore` excludes: `data/`, `venv/`, `.env`, `__pycache__/`, `*.db`
- Initial commit: all current source files
- README.md: project description, setup instructions, architecture diagram

## Testing

**Tests that are deleted:**
- `tests/test_detector.py` — deleted along with `detector.py`

**Tests that need updating:**
- `tests/test_integration.py` — currently mocks ffmpeg/YOLO for the `live_client` fixture; replace mocks with a mock MQTT broker or patched `FrigateListener`. The fixture must not start a real MQTT connection.
- `tests/test_api.py` — any test that exercises `/api/detection-log/image/` or `/api/snapshot/` needs updating to mock the Frigate HTTP proxy. The `live_client` fixture may import from `backend.detector` (now deleted) — update to import from `backend.frigate_listener`.

**Tests that remain valid without changes:**
- `tests/test_config.py` — pure config value assertions, unaffected
- `tests/test_database.py` — all DB functions are unchanged

**New:**
- `tests/test_frigate_listener.py`: unit tests using `unittest.mock.patch` on the paho client to verify: count topic parsing → correct `insert_detection` call, event topic parsing → correct cache update, unknown camera warning, thread start/stop lifecycle

## File Change Summary

| File | Action |
|------|--------|
| `backend/detector.py` | Delete |
| `backend/frigate_listener.py` | Create |
| `backend/config.py` | Modify — add MQTT/Frigate env vars |
| `backend/main.py` | Modify — swap worker, update detection-log endpoints, remove snapshot endpoint |
| `backend/requirements.txt` | Modify — swap ML deps for paho-mqtt>=2.0 |
| `frontend/index.html` | No change |
| `backend/database.py` | No change |
| `frigate/config.yml` | Create |
| `mosquitto/mosquitto.conf` | Create |
| `docker-compose.yml` | Create |
| `.env.example` | Create |
| `.gitignore` | Create |
| `tests/test_detector.py` | Delete |
| `tests/test_frigate_listener.py` | Create |
| `tests/test_integration.py` | Modify — replace ffmpeg/YOLO mocks with MQTT mocks |
| `tests/test_api.py` | Modify — update detection-log image and snapshot test cases |
