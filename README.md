# CSUSM Campus Monitor

Real-time occupancy monitor for CSUSM campus locations. Uses [Frigate NVR](https://frigate.video) for person detection and a custom FastAPI dashboard for live counts, heatmaps, and history.

## What It Does

- Monitors live HLS camera streams (Starbucks, Coffee Cart)
- Counts people in frame every detection cycle via Frigate + YOLOv8
- Stores counts in SQLite for trend analysis
- Serves a web dashboard with live video, heatmaps, and timelines

## Architecture

```
CSUSM HLS streams → Frigate (Docker) → MQTT → FastAPI → SQLite → Dashboard
```

Frigate handles detection and publishes person counts over MQTT. The FastAPI backend subscribes via `FrigateListener`, writes counts to SQLite, and serves the dashboard + REST API.

## Setup

**Requirements:** Python 3.12+, Docker + Docker Compose

```bash
# 1. Clone and install deps
git clone https://github.com/ethanstoner/csusm-monitor.git
cd csusm-monitor
python -m venv venv && venv/bin/pip install -r backend/requirements.txt

# 2. Configure environment
cp .env.example .env   # defaults work for local dev

# 3. Start Frigate + Mosquitto
docker-compose up -d

# 4. Start the backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

## Linux Server Deployment

```bash
scp -r . user@server:~/csusm-monitor/
ssh user@server "cd ~/csusm-monitor && docker-compose up -d"
ssh user@server "cd ~/csusm-monitor && python3 -m venv venv && venv/bin/pip install -r backend/requirements.txt"
ssh user@server "cd ~/csusm-monitor && venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
```

Only `.env` changes between environments — no code modifications required.

## Configuration

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
