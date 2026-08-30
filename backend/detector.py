import logging
import subprocess
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from ultralytics import YOLO

from backend.config import DETECTION_BACKEND, DETECTION_INTERVAL, PERSON_QUERY, FFMPEG_TIMEOUT, TIMEZONE, SNAPSHOTS_DIR, MAX_SNAPSHOTS, CONFIDENCE_THRESHOLD, MIN_FRAME_BRIGHTNESS, MIN_BOX_AREA, VLM_PROBE_INTERVAL

logger = logging.getLogger(__name__)

# Load YOLO model once at module level (downloads yolov8n.pt on first run).
# One model instance is shared by every camera thread, so inference is
# serialised — ultralytics makes no thread-safety guarantee.
model = YOLO("yolov8n.pt")
_model_lock = threading.Lock()

TZ = ZoneInfo(TIMEZONE)

# A camera whose stream is gone fails in a fraction of a second, so retrying
# every DETECTION_INTERVAL spawns ffmpeg 12x a minute forever. Back off
# geometrically instead, up to this ceiling, and reset on the first good frame.
MAX_BACKOFF = 60  # seconds

# Store latest detection results per camera (frame + boxes)
# Accessed by the API to serve annotated snapshots
latest_detections: dict[str, dict] = {}
_detections_lock = threading.Lock()

# --- Camera health ---
# A camera that has been taken down still answers HTTP, so the only thing that
# actually knows a stream is dead is the worker that keeps failing to pull a
# frame from it. Record that here so the API can say "offline" instead of
# handing the dashboard a video element that will never load.
OFFLINE_AFTER_FAILURES = 3  # consecutive failed captures before we call it dead
camera_health: dict[str, dict] = {}
_health_lock = threading.Lock()


def record_camera_health(camera_id: str, ok: bool, error: str | None = None) -> None:
    """Record the outcome of one capture attempt for a camera.

    A single success clears the failure streak: a stream that drops one segment
    is not an outage, and treating it as one would flap the dashboard.
    """
    now = datetime.now(TZ).isoformat()
    with _health_lock:
        health = camera_health.setdefault(
            camera_id, {"consecutive_failures": 0, "last_success": None, "last_error": None}
        )
        if ok:
            health["consecutive_failures"] = 0
            health["last_success"] = now
            health["last_error"] = None
        else:
            health["consecutive_failures"] += 1
            health["last_error"] = error


def get_camera_health(camera_id: str) -> dict:
    """Return a snapshot of one camera's stream health.

    `stream_status` is "offline" once a camera has failed OFFLINE_AFTER_FAILURES
    captures in a row, and "live" otherwise — including for a camera no worker
    has reported on yet, since "we have not looked" is not evidence of an outage.
    """
    with _health_lock:
        health = camera_health.get(camera_id)
        if health is None:
            return {"stream_status": "live", "last_error": None, "consecutive_failures": 0}
        offline = health["consecutive_failures"] >= OFFLINE_AFTER_FAILURES
        return {
            "stream_status": "offline" if offline else "live",
            "last_error": health["last_error"] if offline else None,
            "consecutive_failures": health["consecutive_failures"],
        }

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

    @property
    def is_warm(self) -> bool:
        """True once enough history exists for filter_boxes to actually filter.

        Until then every box is passed through, so counts include the very
        stationary false positives this class exists to remove.
        """
        return len(self._history) >= _STATIC_HIT_THRESHOLD

    @property
    def warmup_progress(self) -> tuple[int, int]:
        """(frames seen, frames needed) — for logging startup state."""
        return min(len(self._history), _STATIC_HIT_THRESHOLD), _STATIC_HIT_THRESHOLD

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
    with _model_lock:
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


YOLO_SOURCE = "yolov8n"
VLM_SOURCE = "LocateAnything-3B"
_VALID_BACKENDS = ("auto", "vlm", "yolo")


class PersonCounters:
    """Counts people with the grounding model, falling back to YOLO.

    LocateAnything-3B is the primary detector. It is not better at counting
    people — measured against hand-labelled ground truth the two are
    indistinguishable, and it costs ~69x more per frame (see bench/README.md).
    It is primary because it is the same weights that answer everything else,
    so one model serves the whole product instead of one model per question.

    YOLOv8n stays as the fallback rather than being deleted, because the
    grounding model needs a Linux host with an NVIDIA GPU and a dashboard that
    shows nothing without one is worse than a dashboard running YOLO.
    """

    def __init__(self, backend: str = DETECTION_BACKEND, client=None,
                 probe_interval: float = VLM_PROBE_INTERVAL,
                 query: str = PERSON_QUERY):
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"DETECTION_BACKEND must be one of {_VALID_BACKENDS}, got {backend!r}"
            )
        self.backend = backend
        self.probe_interval = probe_interval
        self.query = query
        self.active_source: str | None = None
        self._client = client
        self._available = True      # optimistic: try once before writing it off
        self._probed_at = 0.0
        self._logged_state: bool | None = None

    @property
    def client(self):
        """Built lazily so importing this module never requires the VLM config."""
        if self._client is None:
            from backend.vlm import LocateAnythingClient
            self._client = LocateAnythingClient()
        return self._client

    def _should_try_vlm(self) -> bool:
        """True unless a recent probe failed and the backoff has not expired."""
        if self.backend == "yolo":
            return False
        if self._available:
            return True
        return (time.time() - self._probed_at) >= self.probe_interval

    def _note_vlm_state(self, available: bool, error: str | None = None) -> None:
        self._available = available
        self._probed_at = time.time()
        if self._logged_state == available:
            return  # only log transitions, not every cycle
        self._logged_state = available
        if available:
            logger.info("Grounding service reachable — counting people with %s", VLM_SOURCE)
        elif self.backend == "auto":
            logger.warning("Grounding service unavailable (%s) — falling back to %s",
                           error, YOLO_SOURCE)
        else:
            logger.warning("Grounding service unavailable (%s) and backend is 'vlm' "
                           "— this cycle produces no reading", error)

    def count_people(self, frame) -> tuple[int, list[dict], str] | None:
        """Return (count, boxes, source), or None if no backend could answer."""
        if self._should_try_vlm():
            result = self.client.locate(frame, self.query)
            if result is not None:
                self._note_vlm_state(True)
                self.active_source = VLM_SOURCE
                return result["count"], result["boxes"], VLM_SOURCE
            self._note_vlm_state(False, getattr(self.client, "last_error", None))

        if self.backend == "vlm":
            # Asked for the grounding model specifically. A reading silently
            # produced by a different detector is worse than no reading, because
            # the stored row looks identical either way.
            self.active_source = None
            return None

        count, boxes = detect_people(frame)
        self.active_source = YOLO_SOURCE
        return count, boxes, YOLO_SOURCE


class DetectionWorker:
    """Background worker that captures frames and detects people for one camera."""

    def __init__(self, camera_id: str, stream_url: str, db_conn, counters: PersonCounters | None = None):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.db_conn = db_conn
        self.running = False
        self.consecutive_failures = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._filter = StaticObjectFilter()
        self.counters = counters if counters is not None else PersonCounters()

    def apply_filters(self, boxes: list[dict], source: str) -> list[dict]:
        """Post-process boxes for the detector that produced them.

        StaticObjectFilter was built and validated against YOLO's specific
        failure mode — calling signs and poles people at a 0.45 threshold. It
        suppresses anything that holds still, so pointing it at a grounding
        model would equally suppress a person sitting at a table: a real
        detection removed to fix a problem that model may not have. It stays on
        the path it was measured on.
        """
        if source != YOLO_SOURCE:
            return boxes
        return self._filter.filter_boxes(boxes)

    def should_record(self, source: str) -> bool:
        """Whether a reading from `source` belongs in the trend history.

        The warm-up discard exists because StaticObjectFilter cannot tell a
        person from a sign until it has history. There is no filter on the
        grounding path, so there is nothing to warm up and no reason to throw
        away its first minute of real readings.
        """
        return True if source != YOLO_SOURCE else self._filter.is_warm

    def start(self):
        """Start the detection loop in a background thread."""
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Detection worker started for %s", self.camera_id)

    def stop(self):
        """Signal the worker to stop and wait for the current cycle to finish."""
        self.running = False
        self._stop_event.set()  # interrupts the inter-cycle wait immediately
        if self._thread:
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                logger.warning(
                    "[%s] Detection worker did not exit within 15s; it may still "
                    "hold the database connection", self.camera_id,
                )
                return
        logger.info("Detection worker stopped for %s", self.camera_id)

    def next_interval(self) -> float:
        """Seconds to wait before the next cycle, backing off while failing."""
        if not self.consecutive_failures:
            return DETECTION_INTERVAL
        # Exponent is capped so a camera down for a week does not compute 2**120k
        return min(DETECTION_INTERVAL * 2 ** min(self.consecutive_failures, 16), MAX_BACKOFF)

    def _loop(self):
        """Main detection loop: capture -> detect -> store -> wait remainder."""
        from backend.database import insert_detection

        while self.running:
            start_time = time.time()
            try:
                frame = capture_frame(self.stream_url)
                if frame is None:
                    self.consecutive_failures += 1
                    record_camera_health(self.camera_id, ok=False, error="no frame from stream")
                    logger.warning(
                        "[%s] Frame capture failed (%d in a row) — retrying in %.0fs",
                        self.camera_id, self.consecutive_failures, self.next_interval(),
                    )
                elif is_frame_too_dark(frame):
                    # The stream is alive, so this is not a failure — an unlit
                    # scene at 3 AM should not push the camera into backoff.
                    self.consecutive_failures = 0
                    record_camera_health(self.camera_id, ok=True)
                    logger.warning("[%s] Frame too dark — skipping", self.camera_id)
                elif is_static_frame(frame):
                    self.consecutive_failures = 0
                    record_camera_health(self.camera_id, ok=True)
                    logger.info("[%s] Static/placeholder frame — skipping detection", self.camera_id)
                else:
                    self.consecutive_failures = 0
                    record_camera_health(self.camera_id, ok=True)
                    result = self.counters.count_people(frame)
                    if result is None:
                        # backend="vlm" with no service. Not a stream failure, so
                        # it must not push the camera into capture backoff.
                        logger.warning("[%s] No detection backend available — skipping",
                                       self.camera_id)
                        self._stop_event.wait(max(0, self.next_interval() - (time.time() - start_time)))
                        continue
                    count, boxes, source = result
                    boxes = self.apply_filters(boxes, source)
                    count = len(boxes)
                    now = datetime.now(TZ)
                    recorded = self.should_record(source)
                    if recorded:
                        insert_detection(self.db_conn, self.camera_id, count, now, source=source)
                        if count > 0:
                            save_detection_snapshot(self.camera_id, frame, boxes, count, now)
                        logger.info("[%s] Detected %d people (%s)", self.camera_id, count, source)
                    else:
                        seen, needed = self._filter.warmup_progress
                        logger.info(
                            "[%s] Static filter warming up (%d/%d frames) — %d people "
                            "shown live but not recorded",
                            self.camera_id, seen, needed, count,
                        )
                    with _detections_lock:
                        latest_detections[self.camera_id] = {
                            "frame": frame,
                            "boxes": boxes,
                            "count": count,
                            "timestamp": now.isoformat(),
                            "recorded": recorded,
                            "source": source,
                        }
            except Exception as e:
                self.consecutive_failures += 1
                record_camera_health(self.camera_id, ok=False, error=f"{type(e).__name__}: {e}")
                logger.exception("[%s] Detection cycle error", self.camera_id)

            # Wait out the remainder of the interval, but wake instantly on stop()
            elapsed = time.time() - start_time
            self._stop_event.wait(max(0, self.next_interval() - elapsed))
