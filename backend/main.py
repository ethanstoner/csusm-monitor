import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from backend.config import CAMERAS, DB_PATH, FRIGATE_HOST, FRIGATE_PORT, HEALTH_TIMEOUT, RETENTION_DAYS, SNAPSHOTS_DIR, TIMEZONE, VLM_ENABLED  # noqa: F401 — FRIGATE_* used by detection-log fallback
from backend.database import (
    cleanup_old_data,
    get_best_times,
    get_daily_totals,
    get_detection_sources,
    get_heatmap_data,
    get_hourly_averages,
    get_latest_counts,
    get_latest_observations,
    get_observation_history,
    get_timeline_data,
    get_latest_weather,
    get_latest_air_quality,
    get_latest_parking,
    get_parking_trends,
    get_upcoming_events,
    init_db,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

START_WORKERS = True  # Set to False in tests
CLEANUP_INTERVAL = 86400  # seconds between retention sweeps (24 hours)
TZ = ZoneInfo(TIMEZONE)

_db_conn = None
_workers = []
_http_client: httpx.AsyncClient | None = None
_frigate_listener = None  # FrigateListener | None — set during lifespan startup
_manifest_cache: dict[str, tuple[float, bytes]] = {}  # camera_id → (timestamp, trimmed_bytes)
_MANIFEST_TTL = 3.0  # seconds — just under one segment duration (~4s)


def get_db():
    return _db_conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_conn, _http_client, _frigate_listener
    _http_client = httpx.AsyncClient(timeout=10)
    _db_conn = init_db(DB_PATH)
    logger.info("Database initialized at %s", DB_PATH)

    # Cleanup old data on startup
    deleted = cleanup_old_data(_db_conn, RETENTION_DAYS)
    if deleted:
        logger.info("Cleaned up %d old detection rows", deleted)

    if START_WORKERS:
        # One detection worker per camera. Which model each one actually uses is
        # decided per cycle by PersonCounters (DETECTION_BACKEND), not here.
        from backend.detector import DetectionWorker
        for cam_id, cam in CAMERAS.items():
            worker = DetectionWorker(cam_id, cam["stream_url"], _db_conn)
            worker.start()
            _workers.append(worker)

        # Optionally start the open-vocabulary workers. Off unless a grounding
        # service is configured: the model is Linux-only, NVIDIA-licensed for
        # research use, and nothing on the person-count path depends on it.
        if VLM_ENABLED:
            from backend.vlm import OpenVocabWorker
            for cam_id in CAMERAS:
                worker = OpenVocabWorker(cam_id, _db_conn)
                worker.start()
                if worker.running:
                    _workers.append(worker)

        # Optionally start Frigate MQTT listener if MQTT_HOST is configured
        if os.getenv("MQTT_HOST"):
            try:
                from backend.frigate_listener import FrigateListener
                _frigate_listener = FrigateListener(_db_conn)
                _frigate_listener.start()
                _workers.append(_frigate_listener)
            except Exception:
                logger.info("Frigate listener not available, using local YOLO detection only")

        # Data collectors
        from backend.collectors import (
            WeatherCollector, ParkingCollector, AirQualityCollector,
            TransitCollector, EventsCollector,
        )
        for CollectorClass in [WeatherCollector, ParkingCollector,
                               AirQualityCollector, TransitCollector,
                               EventsCollector]:
            collector = CollectorClass(_db_conn)
            collector.start()
            _workers.append(collector)

    # Start daily cleanup thread. The wait is an Event, not time.sleep: a
    # 24-hour sleep meant the flag was only ever read once a day, so the thread
    # outlived shutdown holding _db_conn and would eventually run a query
    # against a connection the app had already closed.
    cleanup_stop = threading.Event()

    def _cleanup_loop():
        while not cleanup_stop.wait(CLEANUP_INTERVAL):
            try:
                deleted = cleanup_old_data(_db_conn, RETENTION_DAYS)
                if deleted:
                    logger.info("Daily cleanup: removed %d old rows", deleted)
            except Exception:
                logger.exception("Daily cleanup failed")

    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True, name="daily-cleanup")
    cleanup_thread.start()

    yield

    # Shutdown — stop every thread that touches the DB before closing it
    cleanup_stop.set()
    for worker in _workers:
        worker.stop()
    _workers.clear()
    cleanup_thread.join(timeout=5)
    if cleanup_thread.is_alive():
        logger.warning("Daily cleanup thread did not exit; leaving DB connection open")
        return
    if _http_client:
        await _http_client.aclose()
    if _db_conn:
        _db_conn.close()


app = FastAPI(title="CSUSM Campus Monitor", lifespan=lifespan)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>CSUSM Campus Monitor</h1><p>Frontend not built yet.</p>")


@app.get("/api/status")
async def get_status():
    # Imported here, not at module scope: backend.detector loads the YOLO
    # weights on import, which the API has no reason to pay for at startup.
    from backend.detector import get_camera_health
    rows = get_latest_counts(_db_conn)
    # Timestamps stored as naive Pacific local time strings; compare against naive local now
    now_naive = datetime.now(TZ).replace(tzinfo=None)
    cameras = []
    for r in rows:
        healthy = True
        if r["timestamp"]:
            last_ts = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
            healthy = (now_naive - last_ts).total_seconds() < HEALTH_TIMEOUT
        else:
            healthy = False
        # Use live count from local detector or FrigateListener, fall back to last DB value
        live_count = r["count"] or 0
        detector = None
        from backend.detector import latest_detections, _detections_lock
        with _detections_lock:
            if r["id"] in latest_detections:
                live_count = latest_detections[r["id"]]["count"]
                detector = latest_detections[r["id"]].get("source")
        if _frigate_listener is not None:
            with _frigate_listener._counts_lock:
                if r["id"] in _frigate_listener.latest_counts:
                    live_count = _frigate_listener.latest_counts[r["id"]]
        health = get_camera_health(r["id"])
        # A camera that cannot produce a frame has no current occupancy. Serving
        # the last row from the table instead means a stream that died in May
        # keeps reporting May's crowd as if it were now.
        if health["stream_status"] == "offline":
            live_count = None
        cameras.append({
            "id": r["id"],
            "name": r["name"],
            "count": live_count,
            "timestamp": r["timestamp"],
            "healthy": healthy,
            "stream_status": health["stream_status"],
            "last_error": health["last_error"],
            # Which model produced this count. Two can, and they trade off
            # automatically, so the dashboard should never have to guess.
            "detector": detector,
        })
    return {"cameras": cameras}


@app.get("/api/cameras")
async def get_cameras():
    from backend.detector import get_camera_health
    rows = get_latest_counts(_db_conn)
    now_naive = datetime.now(TZ).replace(tzinfo=None)
    cameras = []
    for r in rows:
        healthy = True
        if r["timestamp"]:
            last_ts = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
            healthy = (now_naive - last_ts).total_seconds() < HEALTH_TIMEOUT
        else:
            healthy = False
        # Use proxy URL so browser can load HLS without CORS issues
        stream_filename = r["stream_url"].rsplit("/", 1)[-1]
        proxy_url = f"/api/stream/{r['id']}/{stream_filename}"
        health = get_camera_health(r["id"])
        cameras.append({
            "id": r["id"],
            "name": r["name"],
            "stream_url": proxy_url,
            "active": r["active"],
            "healthy": healthy,
            "stream_status": health["stream_status"],
            "last_error": health["last_error"],
        })
    return {"cameras": cameras}


@app.get("/api/detection-log")
async def get_detection_log(camera: str = Query(default=None), limit: int = Query(default=20)):
    """Return recent detection snapshots. Tries Frigate first, falls back to local snapshots."""
    # Try Frigate API first
    try:
        params = {"label": "person", "limit": limit, "has_snapshot": 1}
        if camera:
            params["cameras"] = camera
        resp = await _http_client.get(
            f"http://{FRIGATE_HOST}:{FRIGATE_PORT}/api/events", params=params
        )
        resp.raise_for_status()
        events = resp.json()
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
    except Exception:
        pass

    # Fall back to local snapshots
    if not SNAPSHOTS_DIR.exists():
        return {"detections": []}
    snaps = sorted(SNAPSHOTS_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    if camera:
        snaps = [s for s in snaps if s.name.startswith(f"{camera}_")]
    snaps = snaps[:limit]
    detections = []
    for s in snaps:
        parts = s.stem.split("_")
        cam_id = parts[0]
        count = int(parts[-1].replace("p", "")) if parts[-1].endswith("p") else 1
        ts_str = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]} {parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:6]}"
        detections.append({
            "filename": s.name,
            "camera": cam_id,
            "timestamp": ts_str,
            "count": count,
            "url": f"/api/detection-log/image/{s.name}",
        })
    return {"detections": detections}


@app.get("/api/detection-log/image/{image_name:path}")
async def get_detection_image(image_name: str):
    """Serve a detection snapshot — tries Frigate first, falls back to local file."""
    # Try Frigate proxy
    try:
        resp = await _http_client.get(
            f"http://{FRIGATE_HOST}:{FRIGATE_PORT}/api/events/{image_name}/snapshot.jpg"
        )
        if resp.status_code == 200:
            return Response(
                content=resp.content,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400, immutable"},
            )
    except Exception:
        pass

    # Fall back to local snapshot file
    local_path = SNAPSHOTS_DIR / image_name
    if local_path.exists() and local_path.is_file() and SNAPSHOTS_DIR in local_path.resolve().parents:
        return FileResponse(
            local_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )
    return Response(status_code=404, content="Not found")


@app.get("/api/stream/{camera_id}/{path:path}")
async def proxy_stream(camera_id: str, path: str):
    """Proxy HLS stream to bypass CORS restrictions."""
    if camera_id not in CAMERAS:
        return Response(status_code=404, content="Camera not found")
    base = CAMERAS[camera_id]["stream_url"].rsplit("/", 1)[0]
    url = f"{base}/{path}"

    if path.endswith(".m3u8"):
        # Cache the trimmed manifest to avoid re-downloading the full (and ever-growing)
        # raw manifest on every hls.js poll (~every 4s). The raw manifest can exceed
        # 360KB by midday as CSUSM never resets EXT-X-MEDIA-SEQUENCE.
        cached = _manifest_cache.get(camera_id)
        if cached and (time.time() - cached[0]) < _MANIFEST_TTL:
            return Response(content=cached[1], media_type="application/vnd.apple.mpegurl")
        try:
            resp = await _http_client.get(url)
        except httpx.HTTPError as e:
            return _offline_manifest_response(camera_id, f"{type(e).__name__}: {e}")
        if resp.status_code != 200:
            return _offline_manifest_response(camera_id, f"upstream HTTP {resp.status_code}")
        content = _trim_manifest(resp.content.decode())
        # A decommissioned CSUSM camera keeps answering 200 with a zero-byte
        # manifest. Forwarding that as 200 gave hls.js something it could never
        # parse and no way for the dashboard to tell an outage from a slow load.
        if not _manifest_has_segments(content):
            return _offline_manifest_response(camera_id, "upstream manifest has no segments")
        _manifest_cache[camera_id] = (time.time(), content)
        return Response(content=content, media_type="application/vnd.apple.mpegurl")

    # Segments get the same treatment as manifests: a stream that has gone away
    # mid-playback should read as an outage, not as a fault in this API.
    try:
        resp = await _http_client.get(url)
    except httpx.HTTPError as e:
        return _offline_manifest_response(camera_id, f"{type(e).__name__}: {e}")
    content_type = resp.headers.get("content-type", "application/octet-stream")
    if path.endswith(".ts"):
        content_type = "video/mp2t"
    return Response(content=resp.content, status_code=resp.status_code, media_type=content_type)


def _manifest_has_segments(manifest: bytes) -> bool:
    """True if a trimmed manifest actually references at least one segment."""
    return b"#EXTINF:" in manifest


def _offline_manifest_response(camera_id: str, reason: str) -> Response:
    """Tell the client the stream is down, in a way the dashboard can act on."""
    logger.warning("[%s] Stream unavailable: %s", camera_id, reason)
    return Response(
        status_code=502,
        content=f"camera stream unavailable: {reason}",
        media_type="text/plain",
        headers={"X-Stream-Status": "offline", "Cache-Control": "no-store"},
    )


def _trim_manifest(manifest: str) -> bytes:
    """Trim an HLS manifest to only the last ~6 segments for live playback."""
    lines = manifest.strip().split("\n")
    # Separate header lines from segment entries
    header = []
    segments = []  # pairs of (EXTINF line, URI line)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#EXTINF:"):
            if i + 1 < len(lines):
                segments.append((line, lines[i + 1]))
                i += 2
                continue
        elif not segments:
            # Still in the header
            header.append(line)
        i += 1

    # Keep only the last 6 segments
    keep = segments[-6:] if len(segments) > 6 else segments

    # Update media sequence to reflect the trimmed position
    seq_offset = len(segments) - len(keep)
    new_header = []
    for h in header:
        if h.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            old_seq = int(h.split(":")[1])
            new_header.append(f"#EXT-X-MEDIA-SEQUENCE:{old_seq + seq_offset}")
        else:
            new_header.append(h)

    result = "\n".join(new_header)
    for extinf, uri in keep:
        result += f"\n{extinf}\n{uri}"
    result += "\n"
    return result.encode()


@app.get("/api/history/heatmap")
async def get_heatmap(camera: str = Query(...), days: int = Query(default=7)):
    data = get_heatmap_data(_db_conn, camera, days)
    return {"camera": camera, "days": days, "data": data}


@app.get("/api/history/hourly")
async def get_hourly(camera: str = Query(...), day_type: str = Query(default="all"), days: int = Query(default=30)):
    """Average people count by hour, filtered by weekday/weekend/all."""
    data = get_hourly_averages(_db_conn, camera, day_type, days)
    return {"camera": camera, "day_type": day_type, "data": data}


@app.get("/api/history/timeline")
async def get_timeline(camera: str = Query(...), date: str = Query(default=None)):
    if date is None:
        date = datetime.now(TZ).strftime("%Y-%m-%d")
    data = get_timeline_data(_db_conn, camera, date)
    return {"camera": camera, "date": date, "data": data}


@app.get("/api/history/best-times")
async def best_times(camera: str = Query(...), days: int = Query(default=7)):
    data = get_best_times(_db_conn, camera, days)
    return {"camera": camera, "days": days, "data": data}


@app.get("/api/cameras/{camera_id}/hours")
async def get_camera_hours(camera_id: str):
    """Return the (start_hour, end_hour) window analytics are scoped to."""
    cam = CAMERAS.get(camera_id)
    if not cam:
        return JSONResponse(status_code=404, content={"detail": "Camera not found"})
    return {"camera": camera_id, "open_hours": cam.get("open_hours")}


@app.get("/api/history/daily")
async def get_daily(camera: str = Query(...), days: int = Query(default=30)):
    data = get_daily_totals(_db_conn, camera, days)
    return {"camera": camera, "days": days, "data": data}


@app.get("/api/history/sources")
async def detector_sources(camera: str = Query(...), days: int = Query(default=30)):
    """Which detectors produced a camera's stored history, and how much of it.

    A trend line averaged across two models is only honest if you can see the
    mix behind it.
    """
    return {"camera": camera, "days": days,
            "data": get_detection_sources(_db_conn, camera, days)}


@app.get("/api/observations")
async def get_observations(camera: str = Query(default=None)):
    """Latest open-vocabulary count per (camera, label).

    `enabled` reports whether a grounding service is configured at all, so the
    dashboard can tell "the model is off" from "the model found nothing".
    """
    return {
        "enabled": VLM_ENABLED,
        "observations": get_latest_observations(_db_conn, camera),
    }


@app.get("/api/observations/history")
async def observation_history(camera: str = Query(...), label: str = Query(...),
                              days: int = Query(default=7)):
    data = get_observation_history(_db_conn, camera, label, days)
    return {"camera": camera, "label": label, "days": days, "data": data}


@app.get("/api/conditions")
async def get_conditions():
    weather = get_latest_weather(_db_conn)
    aqi = get_latest_air_quality(_db_conn)
    return {"weather": weather, "aqi": aqi}


@app.get("/api/parking")
async def get_parking():
    latest = get_latest_parking(_db_conn)
    lots = [latest] if latest else []
    return {"lots": lots}


@app.get("/api/parking/trends")
async def parking_trends(lot: str = Query(...), days: int = Query(default=7)):
    data = get_parking_trends(_db_conn, lot, days)
    return {"lot": lot, "days": days, "data": data}


@app.get("/api/transit")
async def get_transit():
    for w in _workers:
        if hasattr(w, "get_next_departures"):
            deps = w.get_next_departures()
            return {"station": "Cal State San Marcos", "departures": deps}
    return {"station": "Cal State San Marcos", "departures": []}


@app.get("/api/events")
async def get_events():
    events = get_upcoming_events(_db_conn)
    return {"events": events}
