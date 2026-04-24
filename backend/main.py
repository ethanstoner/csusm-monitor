import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, Response

from backend.config import CAMERAS, DB_PATH, HEALTH_TIMEOUT, RETENTION_DAYS, TIMEZONE, SNAPSHOTS_DIR
from backend.database import (
    cleanup_old_data,
    get_heatmap_data,
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


def get_db():
    return _db_conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_conn, _http_client
    _http_client = httpx.AsyncClient(timeout=10)
    _db_conn = init_db(DB_PATH)
    logger.info("Database initialized at %s", DB_PATH)

    # Cleanup old data on startup
    deleted = cleanup_old_data(_db_conn, RETENTION_DAYS)
    if deleted:
        logger.info("Cleaned up %d old detection rows", deleted)

    # Start detection workers
    if START_WORKERS:
        from backend.detector import DetectionWorker
        for cam_id, cam in CAMERAS.items():
            worker = DetectionWorker(cam_id, cam["stream_url"], _db_conn)
            worker.start()
            _workers.append(worker)

    # Start daily cleanup thread
    import time as _time
    _cleanup_running = True

    def _cleanup_loop():
        while _cleanup_running:
            _time.sleep(86400)  # 24 hours
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
        cameras.append({
            "id": r["id"],
            "name": r["name"],
            "count": r["count"] or 0,
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
async def get_detection_log(camera: str = Query(default=None), limit: int = Query(default=50)):
    """Return list of saved detection snapshots (only when people were detected)."""
    if not SNAPSHOTS_DIR.exists():
        return {"detections": []}
    snaps = sorted(SNAPSHOTS_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for snap in snaps:
        # Filename format: camera_YYYYMMDD_HHMMSS_Np.jpg
        parts = snap.stem.split("_")
        cam_id = parts[0]
        if camera and cam_id != camera:
            continue
        date_str = parts[1]  # YYYYMMDD
        time_str = parts[2]  # HHMMSS
        count_str = parts[3] if len(parts) > 3 else "0p"
        ts = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
        results.append({
            "filename": snap.name,
            "camera": cam_id,
            "timestamp": ts,
            "count": int(count_str.replace("p", "")),
            "url": f"/api/detection-log/image/{snap.name}",
        })
        if len(results) >= limit:
            break
    return {"detections": results}


@app.get("/api/detection-log/image/{filename}")
async def get_detection_image(filename: str):
    """Serve a saved detection snapshot image."""
    filepath = SNAPSHOTS_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".jpg":
        return Response(status_code=404, content="Not found")
    return FileResponse(filepath, media_type="image/jpeg")


@app.get("/api/snapshot/{camera_id}")
async def get_snapshot(camera_id: str):
    """Return the latest analyzed frame with detection boxes drawn on it as JPEG."""
    import cv2
    from backend.detector import latest_detections, _detections_lock
    with _detections_lock:
        data = latest_detections.get(camera_id)
    if not data or data["frame"] is None:
        return Response(status_code=404, content="No snapshot available yet")
    frame = data["frame"].copy()
    for box in data["boxes"]:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        conf = box["confidence"]
        # Green box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Label with confidence
        label = f"Person {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@app.get("/api/stream/{camera_id}/{path:path}")
async def proxy_stream(camera_id: str, path: str):
    """Proxy HLS stream to bypass CORS restrictions."""
    if camera_id not in CAMERAS:
        return Response(status_code=404, content="Camera not found")
    base = CAMERAS[camera_id]["stream_url"].rsplit("/", 1)[0]
    url = f"{base}/{path}"
    resp = await _http_client.get(url)
    content = resp.content
    content_type = resp.headers.get("content-type", "application/octet-stream")
    if path.endswith(".m3u8"):
        content_type = "application/vnd.apple.mpegurl"
        # Trim manifest to last ~6 segments so hls.js starts at the live edge
        # instead of buffering from the beginning of a 200KB+ manifest
        content = _trim_manifest(content.decode())
    elif path.endswith(".ts"):
        content_type = "video/mp2t"
    return Response(content=content, status_code=resp.status_code, media_type=content_type)


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


@app.get("/api/history/timeline")
async def get_timeline(camera: str = Query(...), date: str = Query(default=None)):
    if date is None:
        date = datetime.now(TZ).strftime("%Y-%m-%d")
    data = get_timeline_data(_db_conn, camera, date)
    return {"camera": camera, "date": date, "data": data}
