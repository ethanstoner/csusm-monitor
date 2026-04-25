# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Or one-click: start.bat (checks ffmpeg, creates venv, installs deps, opens browser)

# Run all tests
pytest -v

# Run a single test file
pytest tests/test_detector.py -v

# Run a single test
pytest tests/test_api.py::test_get_status -v
```

**Requirements:** ffmpeg must be on PATH (used for HLS frame capture via subprocess).

## Architecture

Real-time campus occupancy monitor: captures HLS video frames → YOLOv8 person detection → SQLite storage → web dashboard.

**Backend (FastAPI):**
- `backend/main.py` — App with lifespan management. Starts one `DetectionWorker` thread per camera on startup, shuts them down on exit. Serves frontend, API endpoints, and HLS proxy.
- `backend/detector.py` — Detection pipeline. `DetectionWorker` runs a capture→detect→store loop every 5s per camera. `StaticObjectFilter` tracks box center positions over a 20-frame rolling window to suppress stationary false positives (signs, poles) that YOLO misidentifies as people.
- `backend/database.py` — SQLite with WAL mode. Schema: `detections` (camera, count, timestamp, day_of_week, hour) and `cameras` tables. Indexed for fast status lookups and trend aggregation.
- `backend/config.py` — All tunable parameters: detection thresholds, camera definitions, paths, timers.

**Frontend:** Single `frontend/index.html` — dark-themed dashboard with hls.js for live video, Chart.js for heatmap/timeline, detection log grid with camera filter. Polls `/api/status` every 3s.

**Data flow:** `DetectionWorker._loop()` → `capture_frame()` (ffmpeg subprocess) → `detect_people()` (YOLOv8) → `StaticObjectFilter.filter_boxes()` (suppress stationary FPs) → `insert_detection()` (SQLite) + `save_detection_snapshot()` (annotated JPEG).

## Key Design Decisions

- **Timestamps are naive Pacific local time** (not UTC). Stored as strings so SQLite date functions work directly. Timezone configured via `TIMEZONE` in config.
- **HLS proxy** (`/api/stream/`) rewrites manifests to serve only last 6 segments, preventing clients from buffering from playlist start.
- **Static object filter** uses spatial proximity (40px center radius) over temporal history (12/20 frame hits) — no hardcoded exclusion zones needed. ~60s learning period after restart.
- **Confidence threshold is 0.45** — low enough to catch partially-occluded people while the `StaticObjectFilter` handles the false positives this creates from stationary objects.
- **Thread safety:** `_detections_lock` guards `latest_detections` dict; SQLite connection uses `check_same_thread=False`.

## Test Fixtures

- `client` — Fresh app with temp DB, no detection workers running
- `seeded_client` — Pre-populated with detection history data
- `live_client` — Full pipeline with mocked ffmpeg/YOLO (integration smoke test)

## Camera Configuration

Cameras are defined in `backend/config.py` `CAMERAS` dict. Each has an id, display name, and HLS stream URL. Adding a camera only requires adding an entry there — workers are auto-spawned from the dict at startup.
