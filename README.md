# CSUSM Campus Monitor

**Real-time occupancy tracking for campus locations using computer vision and live video streams.**

Built to solve a real problem: CSUSM's Starbucks and Coffee Cart have no way to tell how busy they are before you walk across campus. This project taps into the university's public HLS camera streams, runs YOLOv8 person detection on each frame, and serves a live dashboard showing current crowd counts, historical trends, and the best times to visit.

---

## The Problem

CSUSM students waste time walking to campus food spots only to find long lines. The university streams live camera feeds from these locations, but there's no occupancy data — just raw video. Students have no way to check crowd levels before committing to the walk.

## The Solution

I built a full-stack monitoring system that:

1. **Captures frames** from CSUSM's live HLS camera streams using ffmpeg
2. **Detects people** in each frame using YOLOv8 (with custom filtering to suppress false positives from stationary objects like signs and poles)
3. **Stores counts** in SQLite with time-series indexing for fast trend queries
4. **Serves a dashboard** with live video, real-time counts, heatmaps, and "best times to visit" recommendations

The system runs a detection cycle every 5 seconds per camera and can optionally offload detection to [Frigate NVR](https://frigate.video) via MQTT for GPU-accelerated inference on a separate server.

---

## Architecture

```
CSUSM HLS Streams
       |
       v
  +-----------+       +----------------+       +---------+
  |  ffmpeg   | ----> | YOLOv8 (local) | ----> | SQLite  |
  | (capture) |       | person detect  |       |  (WAL)  |
  +-----------+       +----------------+       +---------+
       |                                            |
       v                                            v
  +-----------+                              +------------+
  | HLS Proxy | <-------------------------> |  FastAPI    |
  | (CORS fix)|                              | REST API   |
  +-----------+                              +------------+
                                                    |
                                                    v
                                             +------------+
                                             |  Dashboard  |
                                             | (hls.js +   |
                                             |  Chart.js)  |
                                             +------------+
```

**Optional Frigate path** (for GPU-accelerated detection):
```
CSUSM HLS Streams --> Frigate (Docker) --> MQTT --> FastAPI --> SQLite
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| **Detection Worker** | `backend/detector.py` | Captures HLS frames via ffmpeg, runs YOLOv8 inference, filters static objects, saves annotated snapshots |
| **Frigate Listener** | `backend/frigate_listener.py` | MQTT subscriber that receives person counts from Frigate NVR as an alternative detection backend |
| **Database** | `backend/database.py` | SQLite with WAL mode, time-series schema with composite indexes, retention cleanup |
| **API Server** | `backend/main.py` | FastAPI app with lifespan management, HLS proxy, REST endpoints for status/history/heatmap |
| **Dashboard** | `frontend/index.html` | Dark-themed SPA with live video (hls.js), interactive charts (Chart.js), detection log with lightbox |
| **Config** | `backend/config.py` | All tunable parameters — detection thresholds, camera definitions, paths, timers |

---

## Technical Challenges & Solutions

### 1. False Positives from Stationary Objects
**Problem:** YOLOv8 at low confidence thresholds (needed to catch partially-occluded people) frequently misidentifies signs, poles, and furniture as people.

**Solution:** Built a `StaticObjectFilter` that tracks bounding box center positions over a 20-frame rolling window. Objects that remain within a 40px radius for 12+ consecutive frames are classified as stationary and suppressed. This eliminates false positives without hardcoding exclusion zones, and self-calibrates ~60 seconds after startup.

### 2. HLS Manifest Bloat
**Problem:** CSUSM's HLS streams never reset `EXT-X-MEDIA-SEQUENCE`, causing manifests to grow past 360KB by midday. hls.js would try to buffer from the playlist start, causing massive latency.

**Solution:** Built an HLS proxy that intercepts manifest requests, trims to only the last 6 segments, updates the media sequence counter, and caches the result for 3 seconds (just under one segment duration). Clients stay within ~24 seconds of live.

### 3. Corrupted / Dark Frames
**Problem:** HLS segments occasionally deliver black frames or static title cards (especially during stream restarts), causing false zero-counts.

**Solution:** Added pre-detection frame validation: brightness check (mean pixel value > 15) and edge density analysis (Canny edge detection, mean > 1.0) to skip corrupted and placeholder frames.

### 4. CORS Restrictions on University Streams
**Problem:** Browser-based HLS playback fails because CSUSM's stream server doesn't set CORS headers.

**Solution:** FastAPI proxies all `.m3u8` and `.ts` requests through `/api/stream/{camera_id}/`, transparently handling content types and caching.

### 5. Dual Detection Backend
**Problem:** Local YOLO detection works but is CPU-intensive. Wanted the option to offload to a GPU server without rewriting the app.

**Solution:** Designed a pluggable architecture — `DetectionWorker` (local YOLO) runs by default, and `FrigateListener` (MQTT subscriber for Frigate NVR) starts optionally alongside it. Both write to the same SQLite schema. The API layer merges live counts from whichever backend is active.

---

## Features

- **Live Video Feeds** — HLS streams proxied through FastAPI with hls.js playback
- **Real-time People Counts** — Updated every 5 seconds with health status indicators
- **Detection Snapshots** — Annotated JPEG captures with bounding boxes when people are detected
- **Weekly Heatmap** — Day-of-week × hour-of-day grid showing average crowd density
- **Hourly Averages** — 30-day rolling average by hour, filterable by weekday/weekend
- **Daily Trends** — Total and average counts per day with interactive zoom/pan
- **Best Times to Visit** — Ranked hours by lowest average crowd count per camera
- **Automatic Data Cleanup** — Retention policy deletes rows older than 30 days

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Detection** | YOLOv8 (ultralytics), OpenCV, ffmpeg |
| **Backend** | Python, FastAPI, SQLite (WAL mode), paho-mqtt |
| **Frontend** | Vanilla JS, hls.js, Chart.js (with zoom plugin) |
| **Infrastructure** | Docker Compose (Frigate + Mosquitto), uvicorn |

---

## Quick Start

**Requirements:** Python 3.12+, ffmpeg on PATH

```bash
git clone https://github.com/ethanstoner/csusm-monitor.git
cd csusm-monitor

# Option A: One-click (Windows)
start.bat

# Option B: Manual
python -m venv venv
venv/bin/pip install -r backend/requirements.txt   # or venv\Scripts\pip on Windows
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

### With Frigate (GPU-accelerated detection)

```bash
cp .env.example .env        # defaults work for local dev
docker-compose up -d         # starts Frigate + Mosquitto
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Current count + health per camera |
| `GET` | `/api/cameras` | Camera list with proxied stream URLs |
| `GET` | `/api/detection-log` | Recent detection snapshots (Frigate or local) |
| `GET` | `/api/history/heatmap?camera=X` | Day × hour average counts |
| `GET` | `/api/history/hourly?camera=X` | Hourly averages (weekday/weekend filter) |
| `GET` | `/api/history/timeline?camera=X` | Minute-by-minute time series |
| `GET` | `/api/history/best-times?camera=X` | Hours ranked by lowest crowd |
| `GET` | `/api/history/daily?camera=X` | Daily totals and averages |
| `GET` | `/api/stream/{id}/{path}` | HLS proxy (manifest trimming + CORS) |

---

## Running Tests

```bash
pytest -v
```

Tests cover the API layer, database operations, configuration validation, Frigate MQTT listener, and a full integration smoke test with mocked detection pipeline.

---

## Project Structure

```
csusm-monitor/
├── backend/
│   ├── main.py              # FastAPI app, lifespan, API routes
│   ├── detector.py           # YOLOv8 detection worker + frame filters
│   ├── frigate_listener.py   # MQTT subscriber for Frigate NVR
│   ├── database.py           # SQLite schema, queries, cleanup
│   ├── config.py             # All tunable parameters
│   └── requirements.txt
├── frontend/
│   └── index.html            # Dashboard SPA
├── frigate/
│   └── config.yml            # Frigate camera config
├── mosquitto/
│   └── mosquitto.conf        # MQTT broker config
├── tests/
│   ├── test_api.py           # API endpoint tests
│   ├── test_config.py        # Configuration validation
│   ├── test_database.py      # SQLite schema & query tests
│   ├── test_detector.py      # StaticObjectFilter unit tests
│   ├── test_frigate_listener.py  # MQTT listener unit tests
│   └── test_integration.py   # End-to-end smoke test
├── docker-compose.yml        # Frigate + Mosquitto stack
├── start.bat                 # One-click Windows launcher
├── .env.example              # Environment variable template
└── CLAUDE.md                 # AI assistant context
```

---

## Skills Demonstrated

- **Computer Vision** — YOLOv8 object detection with custom post-processing (static object filtering, frame validation)
- **Full-Stack Development** — Python backend + JavaScript frontend, REST API design, real-time data visualization
- **Systems Integration** — HLS video proxying, MQTT pub/sub, Docker containerization, multi-backend architecture
- **Data Engineering** — Time-series SQLite schema design with composite indexes, WAL mode for concurrent reads, automated retention
- **Problem Solving** — Each technical challenge (manifest bloat, false positives, CORS, dark frames) required a distinct engineering approach rather than an off-the-shelf solution
