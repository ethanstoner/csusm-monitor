import logging
import subprocess
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from ultralytics import YOLO

from backend.config import DETECTION_INTERVAL, FFMPEG_TIMEOUT, TIMEZONE, SNAPSHOTS_DIR, MAX_SNAPSHOTS, CONFIDENCE_THRESHOLD, MIN_FRAME_BRIGHTNESS, MIN_BOX_AREA

logger = logging.getLogger(__name__)

# Load YOLO model once at module level (downloads yolov8n.pt on first run)
model = YOLO("yolov8n.pt")

TZ = ZoneInfo(TIMEZONE)

# Store latest detection results per camera (frame + boxes)
# Accessed by the API to serve annotated snapshots
latest_detections: dict[str, dict] = {}
_detections_lock = threading.Lock()

# --- Static object filter ---
# Tracks bounding box centers over a rolling window to suppress stationary
# false positives (signs, poles, furniture) that YOLO misidentifies as people.
_STATIC_WINDOW = 20      # frames of history to keep
_STATIC_HIT_THRESHOLD = 12  # hits in window to consider "static"
_STATIC_RADIUS = 40       # px — max center drift to count as same object


class StaticObjectFilter:
    """Suppress detections that remain in the same spot across many frames."""

    def __init__(self):
        self._history: list[list[tuple[float, float]]] = []  # ring of center-point lists

    def filter_boxes(self, boxes: list[dict]) -> list[dict]:
        """Return only boxes whose centers have NOT been static over the window."""
        centers = [((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2) for b in boxes]
        self._history.append(centers)
        if len(self._history) > _STATIC_WINDOW:
            self._history.pop(0)

        if len(self._history) < _STATIC_HIT_THRESHOLD:
            return boxes  # not enough history yet — pass everything through

        kept = []
        for box, center in zip(boxes, centers):
            hits = 0
            for past_centers in self._history[:-1]:
                if any(
                    abs(pc[0] - center[0]) < _STATIC_RADIUS and abs(pc[1] - center[1]) < _STATIC_RADIUS
                    for pc in past_centers
                ):
                    hits += 1
            if hits < _STATIC_HIT_THRESHOLD:
                kept.append(box)
        return kept


# One filter per camera (populated lazily in DetectionWorker._loop)
_static_filters: dict[str, StaticObjectFilter] = {}


def capture_frame(stream_url: str, timeout: int = FFMPEG_TIMEOUT) -> np.ndarray | None:
    """Pull a single frame from an HLS stream using ffmpeg. Returns BGR numpy array or None."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", stream_url,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-loglevel", "error",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if proc.returncode != 0 or len(proc.stdout) == 0:
            logger.warning("ffmpeg returned no data for %s", stream_url)
            return None
        frame = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
        return frame
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("ffmpeg capture failed for %s: %s", stream_url, e)
        return None


def is_frame_too_dark(frame: np.ndarray) -> bool:
    """Check if a frame is too dark (black/corrupted HLS segment)."""
    return float(np.mean(frame)) < MIN_FRAME_BRIGHTNESS


def is_static_frame(frame: np.ndarray) -> bool:
    """Detect static/placeholder frames (title cards, solid colors) via edge density."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges)) < 1.0


def detect_people(frame: np.ndarray) -> tuple[int, list[dict]]:
    """Run YOLOv8 person detection on a frame. Returns (count, list of box dicts).
    Filters out detections below CONFIDENCE_THRESHOLD."""
    results = model(frame, verbose=False)
    count = 0
    boxes = []
    for result in results:
        if result.boxes is not None and len(result.boxes.cls) > 0:
            cls = result.boxes.cls
            cls_np = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls)
            xyxy = result.boxes.xyxy
            xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
            conf = result.boxes.conf
            conf_np = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
            for i in range(len(cls_np)):
                box_area = (xyxy_np[i][2] - xyxy_np[i][0]) * (xyxy_np[i][3] - xyxy_np[i][1])
                if cls_np[i] == 0 and conf_np[i] >= CONFIDENCE_THRESHOLD and box_area >= MIN_BOX_AREA:
                    count += 1
                    boxes.append({
                        "x1": int(xyxy_np[i][0]),
                        "y1": int(xyxy_np[i][1]),
                        "x2": int(xyxy_np[i][2]),
                        "y2": int(xyxy_np[i][3]),
                        "confidence": round(float(conf_np[i]), 2),
                    })
    return count, boxes


def save_detection_snapshot(camera_id: str, frame: np.ndarray, boxes: list[dict], count: int, ts: datetime) -> str | None:
    """Save an annotated frame to disk when people are detected. Returns filename or None."""
    if count == 0:
        return None
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    # Draw boxes on a copy
    annotated = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        conf = box["confidence"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # Filename: camera_timestamp_count.jpg
    filename = f"{camera_id}_{ts.strftime('%Y%m%d_%H%M%S')}_{count}p.jpg"
    filepath = SNAPSHOTS_DIR / filename
    cv2.imwrite(str(filepath), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # Cleanup old snapshots if over limit
    all_snaps = sorted(SNAPSHOTS_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    while len(all_snaps) > MAX_SNAPSHOTS:
        all_snaps.pop(0).unlink()
    return filename


class DetectionWorker:
    """Background worker that captures frames and detects people for one camera."""

    def __init__(self, camera_id: str, stream_url: str, db_conn):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.db_conn = db_conn
        self.running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Start the detection loop in a background thread."""
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Detection worker started for %s", self.camera_id)

    def stop(self):
        """Signal the worker to stop."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=15)
        logger.info("Detection worker stopped for %s", self.camera_id)

    def _loop(self):
        """Main detection loop: capture -> detect -> store -> sleep remainder."""
        from backend.database import insert_detection

        while self.running:
            start_time = time.time()
            try:
                frame = capture_frame(self.stream_url)
                if frame is None:
                    logger.warning("[%s] Frame capture failed", self.camera_id)
                elif is_frame_too_dark(frame):
                    logger.warning("[%s] Frame too dark — skipping", self.camera_id)
                elif is_static_frame(frame):
                    logger.info("[%s] Static/placeholder frame — skipping detection", self.camera_id)
                else:
                    count, boxes = detect_people(frame)
                    # Suppress stationary false positives (signs, poles)
                    if self.camera_id not in _static_filters:
                        _static_filters[self.camera_id] = StaticObjectFilter()
                    boxes = _static_filters[self.camera_id].filter_boxes(boxes)
                    count = len(boxes)
                    now = datetime.now(TZ)
                    insert_detection(self.db_conn, self.camera_id, count, now)
                    with _detections_lock:
                        latest_detections[self.camera_id] = {
                            "frame": frame,
                            "boxes": boxes,
                            "count": count,
                            "timestamp": now.isoformat(),
                        }
                    if count > 0:
                        save_detection_snapshot(self.camera_id, frame, boxes, count, now)
                    logger.info("[%s] Detected %d people", self.camera_id, count)
            except Exception:
                logger.exception("[%s] Detection cycle error", self.camera_id)

            # Sleep for remainder of interval
            elapsed = time.time() - start_time
            sleep_time = max(0, DETECTION_INTERVAL - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
