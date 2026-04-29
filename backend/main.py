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
from fastapi.responses import FileResponse, HTMLResponse, Response

from backend.config import CAMERAS, DB_PATH, FRIGATE_HOST, FRIGATE_PORT, HEALTH_TIMEOUT, RETENTION_DAYS, SNAPSHOTS_DIR, TIMEZONE  # noqa: F401 — FRIGATE_* used by detection-log fallback
from backend.database import (
    cleanup_old_data,
    get_best_times,
    get_daily_totals,
    get_heatmap_data,
    get_hourly_averages,
    get_latest_counts,
    get_timeline_data,
    init_db,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

START_WORKERS = True  # Set to False in tests
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
        # Start local YOLO detection workers for each camera
        from backend.detector import DetectionWorker
        for cam_id, cam in CAMERAS.items():
            worker = DetectionWorker(cam_id, cam["stream_url"], _db_conn)
            worker.start()
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

    # Start daily cleanup thread
    _cleanup_running = True

    def _cleanup_loop():
        while _cleanup_running:
            time.sleep(86400)  # 24 hours
            try:
                deleted = cleanup_old_data(_db_conn, RETENTION_DAYS)
                if deleted:
                    logger.info("Daily cleanup: removed %d old rows", deleted)
            except Exception:
                logger.exception("Daily cleanup failed")

    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    cleanup_thread.start()

    yield
    _cleanup_running = False

    # Shutdown
    for worker in _workers:
        worker.stop()
    _workers.clear()
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
        from backend.detector import latest_detections, _detections_lock
        with _detections_lock:
            if r["id"] in latest_detections:
                live_count = latest_detections[r["id"]]["count"]
        if _frigate_listener is not None:
            with _frigate_listener._counts_lock:
                if r["id"] in _frigate_listener.latest_counts:
                    live_count = _frigate_listener.latest_counts[r["id"]]
        cameras.append({
            "id": r["id"],
            "name": r["name"],
            "count": live_count,
            "timestamp": r["timestamp"],
            "healthy": healthy,
        })
    return {"cameras": cameras}


@app.get("/api/cameras")
async def get_cameras():
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
        cameras.append({
            "id": r["id"],
            "name": r["name"],
            "stream_url": proxy_url,
            "active": r["active"],
            "healthy": healthy,
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
        resp = await _http_client.get(url)
        content = _trim_manifest(resp.content.decode())
        _manifest_cache[camera_id] = (time.time(), content)
        return Response(content=content, media_type="application/vnd.apple.mpegurl")

    resp = await _http_client.get(url)
    content_type = resp.headers.get("content-type", "application/octet-stream")
    if path.endswith(".ts"):
        content_type = "video/mp2t"
    return Response(content=resp.content, status_code=resp.status_code, media_type=content_type)


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


@app.get("/api/history/daily")
async def get_daily(camera: str = Query(...), days: int = Query(default=30)):
    data = get_daily_totals(_db_conn, camera, days)
    return {"camera": camera, "days": days, "data": data}
