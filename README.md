# CSUSM Campus Monitor

[![Tests](https://github.com/ethanstoner/csusm-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/ethanstoner/csusm-monitor/actions/workflows/ci.yml)

**Real-time occupancy tracking for campus locations using computer vision and live video streams.**

Built to solve a real problem: CSUSM's Starbucks and Coffee Cart have no way to tell how busy they are before you walk across campus. This project taps into the university's public HLS camera streams, runs YOLOv8 person detection on each frame, and serves a live dashboard showing current crowd counts, historical trends, and the best times to visit.

### Live Feeds & Detection Log
![Dashboard showing live camera feeds with people count badges and detection log with YOLOv8 bounding boxes](assets/dashboard-live.png)

### Analytics — Best Times & Weekly Heatmap
![Analytics section showing best times to visit per camera and weekly activity heatmap](assets/dashboard-analytics.png)

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

**Optional open-vocabulary path** (natural-language queries, off by default). Reuses the
frame the YOLO worker already captured, so there is no second ffmpeg and both models
score identical pixels:
```
captured frame --> HTTP --> LocateAnything-3B service --> boxes --> observations table
   (every 300s)              (separate process, GPU)
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| **Detection Worker** | `backend/detector.py` | Captures HLS frames via ffmpeg, runs YOLOv8 inference, filters static objects, saves annotated snapshots |
| **Frigate Listener** | `backend/frigate_listener.py` | MQTT subscriber that receives person counts from Frigate NVR as an alternative detection backend |
| **Open-Vocabulary Backend** | `backend/vlm.py` | Decodes LocateAnything-3B coordinate tokens, talks to the grounding service, polls natural-language queries on a slow interval |
| **Grounding Service** | `services/locate_anything/server.py` | Out-of-process LocateAnything-3B host (Linux + NVIDIA GPU, optional) |
| **Database** | `backend/database.py` | SQLite with WAL mode, time-series schema with composite indexes, retention cleanup |
| **API Server** | `backend/main.py` | FastAPI app with lifespan management, HLS proxy, REST endpoints for status/history/heatmap |
| **Dashboard** | `frontend/index.html` | Dark-themed SPA with live video (hls.js), interactive charts (Chart.js), detection log with lightbox |
| **Config** | `backend/config.py` | All tunable parameters — detection thresholds, camera definitions, paths, timers |

---

## Technical Challenges & Solutions

### 1. False Positives from Stationary Objects
**Problem:** YOLOv8 at low confidence thresholds (needed to catch partially-occluded people) frequently misidentifies signs, poles, and furniture as people.

**Solution:** Built a `StaticObjectFilter` that tracks bounding box center positions over a 20-frame rolling window. Objects that remain within a 40px radius for 12+ consecutive frames are classified as stationary and suppressed. This eliminates false positives without hardcoding exclusion zones, and self-calibrates ~60 seconds after startup.

The self-calibration has a cost worth naming: for those first 12 frames the filter has no history, so it passes everything through and the counts are inflated by exactly the objects it exists to remove. Those counts are shown live but deliberately **not** written to the database, because every trend in the dashboard is an average and a restart would otherwise bake a minute of false positives into the seven-day view.

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

### 6. A Missing Camera Is Not the Same as an Empty One
**Problem:** These are public university streams and they go down. When one 404s, ffmpeg fails against it in about 0.2 seconds, so a fixed 5-second retry meant spawning a process and logging a warning twelve times a minute, indefinitely, for a camera that was never coming back that day. Worse, the resulting hole in the data was invisible downstream — an hour with four readings and an hour with seven hundred rendered identically in the heatmap.

**Solution:** Consecutive capture failures back off geometrically from 5s to a 60s ceiling and reset on the first good frame; a dark or static frame means the stream is alive, so those don't count as failures. The inter-cycle wait is an `Event`, so shutdown interrupts a backoff instead of waiting it out. Trend queries return a sample count per bucket, and the heatmap hatches any hour with sparse coverage so a gap reads as a gap rather than as a quiet hour.

### 7. A 200 OK That Means the Camera Is Gone
**Problem:** CSUSM took the Starbucks camera down, and the URL kept answering `200 OK` — with a zero-byte manifest. Everything downstream believed it. The HLS proxy forwarded a body of `b'\n'` as a valid playlist, so hls.js sat on a manifest it could never parse and the card spun indefinitely with no way to distinguish "loading" from "decommissioned". Worse, `/api/status` fell back to the newest row in the table for a camera that had not produced a frame since May, so a four-month-old crowd count was served as current occupancy and summed into the campus total.

**Solution:** The proxy now rejects an upstream response that carries no `#EXTINF` segment, returning `502` with an `X-Stream-Status: offline` header instead of a playlist that cannot work. Detection workers record capture outcomes in a shared health map; three consecutive failures mark a camera offline, and one good frame clears it, so a single dropped segment does not flap the dashboard. An offline camera reports `count: null` rather than a stale number, the dashboard renders an explicit offline card with the reason instead of mounting a doomed video element, and the campus total excludes cameras that are not reporting rather than counting them as zero.

### 8. Counting Things Nobody Trained a Class For
**Problem:** The monitor counts exactly one thing, because YOLOv8n was trained to. Every question beyond "how many people" — how many bicycles are at the rack, how long the queue is as distinct from people merely standing nearby — needs a different model or a fine-tune, which is a semester of labelling rather than a weekend.

**Solution:** Added [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B), an open-vocabulary grounding model, as a **third** detection backend — prompted in English, no retraining. It does not replace YOLO. It runs out-of-process behind HTTP on a long interval, writes to its own `observations` table, and reuses the frame the YOLO worker already captured so both models score identical pixels.

Measuring it against YOLOv8n on 240 real frames — including 180 with hand-labelled ground truth, every frame reviewed by eye — produced the argument for that split. On person counting the two tie exactly (180/180 frames correct, 7/7 people, no false positives either way) and LocateAnything is **69× slower** for the identical answer. What it buys is the question YOLO cannot answer at all: prompt it `trash can` and it finds all six, having never been trained on the class. Four things had to be fixed before any of it was trustworthy, including a sampling default that made the same frame return different counts and a generation runaway that returned 340 boxes for a 6-object query. All of it, with the numbers, is in **[bench/README.md](bench/README.md)**.

> **Licence note.** The MIT `LICENSE` at the root covers *this code*. It does not cover the LocateAnything-3B weights, which are released under the [NVIDIA License](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE) for **academic and non-profit research purposes only**. That is why this backend is off unless `VLM_ENABLED=1`, why the model runs behind a process boundary rather than being imported, and why nothing in this repo ships the weights. If you clone this and intend to deploy it commercially, that backend has to be swapped for something you are licensed to use.

---

## Features

- **Live Video Feeds** — HLS streams proxied through FastAPI with hls.js playback
- **Real-time People Counts** — Updated every 5 seconds with health status indicators
- **Detection Snapshots** — Annotated JPEG captures with bounding boxes when people are detected
- **Weekly Heatmap** — Day-of-week × hour-of-day grid showing average crowd density
- **Hourly Averages** — 30-day rolling average by hour, filterable by weekday/weekend
- **Daily Trends** — Total and average counts per day with interactive zoom/pan
- **Best Times to Visit** — Ranked hours by lowest average crowd, scoped to each location's opening hours
- **Campus Context** — Weather, air quality, parking availability, Sprinter departures and upcoming events alongside the feeds
- **Ask the Campus** — Optional open-vocabulary counts driven by a natural-language prompt, on their own schedule and their own table
- **Coverage Honesty** — Hours backed by too few readings are marked, so outages don't masquerade as quiet periods; a camera that is down reports no count rather than its last one
- **Automatic Data Cleanup** — Retention policy deletes rows older than 30 days

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Detection** | YOLOv8 (ultralytics), OpenCV, ffmpeg |
| **Open-vocabulary** | NVIDIA LocateAnything-3B (Qwen2.5-3B + MoonViT), PyTorch, CUDA |
| **Backend** | Python, FastAPI, SQLite (WAL mode), paho-mqtt |
| **Frontend** | Vanilla JS, hls.js, Chart.js (with zoom plugin) |
| **Infrastructure** | Docker Compose (full stack), Dockerfile, uvicorn |

---

## Quick Start

### Docker (recommended)

**Requirements:** Docker + Docker Compose

```bash
git clone https://github.com/ethanstoner/csusm-monitor.git
cd csusm-monitor
docker compose up -d
```

That's it — opens the dashboard at [http://localhost:8000](http://localhost:8000). Brings up the FastAPI backend, Frigate NVR, and Mosquitto MQTT broker as a single stack. The YOLO model weights are baked into the image at build time.

### Local development (no Docker)

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

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Current count + health per camera |
| `GET` | `/api/cameras` | Camera list with proxied stream URLs |
| `GET` | `/api/detection-log` | Recent detection snapshots (Frigate or local) |
| `GET` | `/api/cameras/{id}/hours` | Operating-hours window analytics are scoped to |
| `GET` | `/api/history/heatmap?camera=X` | Day × hour average counts, with sample counts |
| `GET` | `/api/history/hourly?camera=X` | Hourly averages (weekday/weekend filter) |
| `GET` | `/api/history/timeline?camera=X` | Minute-by-minute time series |
| `GET` | `/api/history/best-times?camera=X` | Hours ranked by lowest crowd, within opening hours |
| `GET` | `/api/history/daily?camera=X` | Daily totals and averages |
| `GET` | `/api/observations` | Latest open-vocabulary count per (camera, label) |
| `GET` | `/api/observations/history?camera=X&label=Y` | Hourly averages for one label |
| `GET` | `/api/conditions` | Latest weather and air quality |
| `GET` | `/api/parking` | Latest campus parking availability |
| `GET` | `/api/parking/trends?lot=X` | Parking availability by day × hour |
| `GET` | `/api/transit` | Next NCTD Sprinter departures from CSUSM |
| `GET` | `/api/events` | Upcoming campus events |
| `GET` | `/api/stream/{id}/{path}` | HLS proxy (manifest trimming + CORS) |

---

## Running Tests

```bash
pytest -v
```

114 tests covering the API layer, database and trend queries, the data collectors' HTML/GTFS parsing, configuration validation, the `StaticObjectFilter` and detection-worker lifecycle (backoff, prompt shutdown, warm-up suppression), camera-health and offline-stream handling, the open-vocabulary box decoder and every grounding-service failure mode (unreachable, timeout, malformed payload, runaway generation), the Frigate MQTT listener, and a full integration smoke test with a mocked detection pipeline.

---

## Project Structure

```
csusm-monitor/
├── backend/
│   ├── main.py              # FastAPI app, lifespan, API routes
│   ├── detector.py           # YOLOv8 detection worker + frame filters
│   ├── collectors.py         # Weather, parking, AQI, transit, events collectors
│   ├── frigate_listener.py   # MQTT subscriber for Frigate NVR
│   ├── vlm.py                # Open-vocabulary backend: box decoding + worker
│   ├── database.py           # SQLite schema, queries, cleanup
│   ├── config.py             # All tunable parameters
│   └── requirements.txt
├── services/
│   └── locate_anything/
│       └── server.py         # Out-of-process LocateAnything-3B HTTP service
├── bench/
│   ├── README.md             # Measured latency, VRAM, accuracy; how to reproduce
│   ├── la3b_bench.py         # LocateAnything-3B benchmark harness
│   ├── yolo_bench.py         # YOLOv8n baseline over the same frames
│   ├── capture_frames.sh     # Freeze a benchmark frame set from a live stream
│   └── sweep_resolution.sh   # Latency/VRAM/recall vs input resolution
├── frontend/
│   └── index.html            # Dashboard SPA
├── frigate/
│   └── config.yml            # Frigate camera config
├── mosquitto/
│   └── mosquitto.conf        # MQTT broker config
├── tests/
│   ├── test_api.py           # API endpoint tests
│   ├── test_collectors.py    # Collector parsing + collector API tests
│   ├── test_config.py        # Configuration validation
│   ├── test_database.py      # SQLite schema & query tests
│   ├── test_detector.py      # StaticObjectFilter + worker lifecycle tests
│   ├── test_frigate_listener.py  # MQTT listener unit tests
│   ├── test_vlm.py           # Box decoding, service failure modes, worker
│   └── test_integration.py   # End-to-end smoke test
├── Dockerfile                # Backend container (Python + ffmpeg + YOLO)
├── .dockerignore             # Docker build exclusions
├── docker-compose.yml        # Full stack: monitor + Frigate + Mosquitto
├── start.bat                 # One-click Windows launcher
└── .env.example              # Environment variable template
```

---

## Skills Demonstrated

- **Computer Vision** — YOLOv8 object detection with custom post-processing (static object filtering, frame validation)
- **Full-Stack Development** — Python backend + JavaScript frontend, REST API design, real-time data visualization
- **Systems Integration** — HLS video proxying, MQTT pub/sub, Docker containerization, multi-backend architecture
- **Data Engineering** — Time-series SQLite schema design with composite indexes, WAL mode for concurrent reads, automated retention
- **Problem Solving** — Each technical challenge (manifest bloat, false positives, CORS, dark frames) required a distinct engineering approach rather than an off-the-shelf solution
