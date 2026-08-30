"""Open-vocabulary detection backend, driven by NVIDIA LocateAnything-3B.

This is a third detection backend alongside `DetectionWorker` (local YOLO) and
`FrigateListener` (MQTT), and it deliberately does not replace either. YOLOv8n
keeps the 5-second person-count hot path: it is fast, it is good enough at the
one thing it does, and months of trend history were computed with it — swapping
the model underneath the heatmap would silently change what the historical
average means.

What this adds is the thing YOLO cannot do at all: counting something nobody
trained a class for, asked in English. "How many bicycles are at the rack", "how
long is the queue", "is the shuttle at the stop" are all the same weights and
the same code path, with a different string.

The model itself runs out-of-process (it is Linux-only and does not fit the
5-second budget), reached over HTTP. Everything here is written on the
assumption that the service is absent, slow, or wrong, because for most of this
module's life it will be at least one of those.
"""
import base64
import logging
import re
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.config import (
    OPEN_VOCAB_INTERVAL,
    OPEN_VOCAB_QUERIES,
    TIMEZONE,
    VLM_BASE_URL,
    VLM_MAX_IMAGE_SIDE,
    VLM_TIMEOUT,
)

logger = logging.getLogger(__name__)
TZ = ZoneInfo(TIMEZONE)

# LocateAnything emits coordinates as dedicated <0>..<1000> vocabulary tokens
# wrapped in <box>...</box>, normalised to a 0-1000 grid regardless of the
# input resolution.
BOX_RE = re.compile(r"<box>((?:<\d+>)+)</box>")
COORD_RE = re.compile(r"<(\d+)>")
COORD_SCALE = 1000.0

# The model card documents box order as x1 y1 x2 y2; generate_utils.py in the
# same release says x1 x2 y1 y2. They disagree, so the order is verified against
# annotated frames (see bench/la3b_bench.py --annotate) rather than trusted.
COORD_ORDER = "xyxy"

# A grounding model can return a truncated box, a coordinate outside the grid,
# or an inverted rectangle. Anything malformed is dropped rather than repaired:
# a detector that invents a plausible box out of a broken one is worse than one
# that reports fewer, because nothing downstream can tell the two apart.

# It can also fail in a way no per-box check catches: run away and emit boxes
# until it hits the token budget. Observed in practice — 340 boxes returned for
# a query whose correct answer was 6, every one of them individually well
# formed. A count that large is not an occupancy reading, it is a broken
# generation, and the honest thing to record for it is nothing at all.
MAX_PLAUSIBLE_BOXES = 100


def parse_boxes(answer: str, width: int, height: int, order: str = COORD_ORDER) -> list[dict]:
    """Decode <box> tokens into pixel boxes, dropping anything malformed."""
    boxes: list[dict] = []
    if not answer:
        return boxes
    for match in BOX_RE.finditer(answer):
        coords = [int(c) for c in COORD_RE.findall(match.group(1))]
        if len(coords) != 4:
            continue
        if any(c < 0 or c > 1000 for c in coords):
            continue
        if order == "xyxy":
            x1, y1, x2, y2 = coords
        else:  # "xxyy"
            x1, x2, y1, y2 = coords
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({
            "x1": int(x1 / COORD_SCALE * width),
            "y1": int(y1 / COORD_SCALE * height),
            "x2": int(x2 / COORD_SCALE * width),
            "y2": int(y2 / COORD_SCALE * height),
        })
    return boxes


class LocateAnythingClient:
    """HTTP client for the out-of-process grounding service.

    Every failure mode collapses to None. Callers get "no answer this cycle",
    never an exception, because the alternative is one unreachable GPU box
    taking down a dashboard whose main job — counting people — does not need it.
    """

    def __init__(self, base_url: str = VLM_BASE_URL, timeout: float = VLM_TIMEOUT,
                 max_image_side: int = VLM_MAX_IMAGE_SIDE):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_image_side = max_image_side
        self.last_error: str | None = None

    def locate(self, frame, query: str) -> dict | None:
        """Ask the service to locate `query` in `frame` (a BGR numpy array).

        Returns {"boxes": [...], "count": int, "latency_ms": float} or None.
        """
        import cv2
        import httpx

        height, width = frame.shape[:2]
        # Downscale before sending. The model returns coordinates on a
        # resolution-independent 0-1000 grid, so boxes are still decoded against
        # the *original* frame size — the dashboard gets full-resolution boxes
        # from a fraction of the inference cost.
        sent = frame
        longest = max(width, height)
        if self.max_image_side and longest > self.max_image_side:
            scale = self.max_image_side / longest
            sent = cv2.resize(frame, (round(width * scale), round(height * scale)),
                              interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", sent, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            self.last_error = "frame could not be JPEG-encoded"
            return None

        started = time.perf_counter()
        try:
            resp = httpx.post(
                f"{self.base_url}/locate",
                json={"image": base64.b64encode(buf.tobytes()).decode(), "query": query},
                timeout=self.timeout,
            )
        except Exception as e:  # httpx raises a family of unrelated errors
            self.last_error = f"{type(e).__name__}: {e}"
            return None
        if resp.status_code != 200:
            self.last_error = f"grounding service returned HTTP {resp.status_code}"
            return None
        try:
            payload = resp.json()
        except ValueError:
            self.last_error = "grounding service returned non-JSON"
            return None

        raw = payload.get("raw")
        if not isinstance(raw, str):
            self.last_error = "grounding service response had no raw text"
            return None

        boxes = parse_boxes(raw, width, height)
        if len(boxes) > MAX_PLAUSIBLE_BOXES:
            self.last_error = (
                f"discarded a runaway generation: {len(boxes)} boxes for {query!r}"
            )
            return None
        self.last_error = None
        return {
            "boxes": boxes,
            "count": len(boxes),
            # Prefer the service's own timing when it reports one; it excludes
            # request overhead and is the number the roadmap argues about.
            "latency_ms": payload.get("latency_ms", (time.perf_counter() - started) * 1000),
            "raw": raw,
        }


# Latest open-vocabulary results per camera, for the API to serve without a
# database round trip. Shape: {camera_id: {label: {...}}}
latest_observations: dict[str, dict] = {}
_observations_lock = threading.Lock()


class OpenVocabWorker:
    """Answer a fixed set of natural-language queries about one camera.

    Runs on OPEN_VOCAB_INTERVAL, far slower than the 5-second person cycle, and
    reuses the frame the YOLO worker already captured instead of starting a
    second ffmpeg. That is not only cheaper: it means both models score the same
    pixels, so their disagreement is about the models and not about the two
    frames happening to be a few seconds apart.
    """

    def __init__(self, camera_id: str, db_conn, client: LocateAnythingClient | None = None,
                 queries: list[str] | None = None, interval: float = OPEN_VOCAB_INTERVAL):
        self.camera_id = camera_id
        self.db_conn = db_conn
        self.client = client or LocateAnythingClient()
        self.queries = queries if queries is not None else OPEN_VOCAB_QUERIES.get(camera_id, [])
        self.interval = interval
        self.running = False
        self.last_error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self):
        if not self.queries:
            logger.info("[%s] No open-vocabulary queries configured — worker not started",
                        self.camera_id)
            return
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"openvocab-{self.camera_id}")
        self._thread.start()
        logger.info("[%s] Open-vocabulary worker started (%d queries, every %.0fs)",
                    self.camera_id, len(self.queries), self.interval)

    def stop(self):
        self.running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(15, self.client.timeout + 5))
            if self._thread.is_alive():
                logger.warning("[%s] Open-vocabulary worker did not exit in time",
                               self.camera_id)
                return
        logger.info("[%s] Open-vocabulary worker stopped", self.camera_id)

    def _current_frame(self):
        """Borrow the newest frame the YOLO worker captured for this camera."""
        from backend.detector import _detections_lock, latest_detections
        with _detections_lock:
            entry = latest_detections.get(self.camera_id)
            return None if entry is None else entry.get("frame")

    def run_once(self) -> int:
        """Score every configured query against the current frame.

        Returns the number of queries that produced an answer.
        """
        from backend.database import insert_observation

        frame = self._current_frame()
        if frame is None:
            self.last_error = "no frame available from the camera"
            return 0

        answered = 0
        for query in self.queries:
            if self._stop_event.is_set():
                break
            result = self.client.locate(frame, query)
            if result is None:
                self.last_error = self.client.last_error
                logger.warning("[%s] '%s' unanswered: %s",
                               self.camera_id, query, self.last_error)
                continue
            now = datetime.now(TZ)
            insert_observation(self.db_conn, self.camera_id, query, result["count"], now,
                               latency_ms=int(result["latency_ms"]))
            with _observations_lock:
                latest_observations.setdefault(self.camera_id, {})[query] = {
                    "count": result["count"],
                    "boxes": result["boxes"],
                    "timestamp": now.isoformat(),
                    "latency_ms": int(result["latency_ms"]),
                }
            answered += 1
            self.last_error = None
            logger.info("[%s] '%s' -> %d (%.0fms)",
                        self.camera_id, query, result["count"], result["latency_ms"])
        return answered

    def _loop(self):
        while self.running:
            started = time.time()
            try:
                self.run_once()
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                logger.exception("[%s] Open-vocabulary cycle error", self.camera_id)
            self._stop_event.wait(max(0, self.interval - (time.time() - started)))
