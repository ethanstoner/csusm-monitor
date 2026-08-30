"""Tests for the detection pipeline utilities."""
import time
from unittest.mock import patch

import pytest

import numpy as np

from backend.config import DETECTION_INTERVAL
from backend.detector import MAX_BACKOFF, DetectionWorker, PersonCounters, StaticObjectFilter


def _yolo_worker(camera_id, stream_url, db_conn=None):
    """A worker pinned to YOLO.

    These tests are about the capture loop — backoff, shutdown, warm-up — not
    about backend selection, which has its own file. Pinning the backend keeps
    them off the network: on "auto" the worker would try to reach a grounding
    service that is not running and pay its timeout every cycle.
    """
    return DetectionWorker(camera_id, stream_url, db_conn,
                           counters=PersonCounters(backend="yolo"))


def _box(x1, y1, x2, y2, conf=0.9):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf}


@pytest.fixture
def snapshots_dir(tmp_path):
    """Redirect snapshot writes away from the real data/snapshots directory.

    Without this a worker test writes JPEGs into live data and, once the
    directory is at MAX_SNAPSHOTS, deletes real captures to make room.
    """
    d = tmp_path / "snapshots"
    d.mkdir()
    with patch("backend.detector.SNAPSHOTS_DIR", d):
        yield d


def test_static_filter_passes_during_warmup():
    """All boxes should pass through until the filter has enough history."""
    sf = StaticObjectFilter()
    boxes = [_box(100, 100, 200, 200)]
    for _ in range(11):
        result = sf.filter_boxes(boxes)
    assert len(result) == 1  # still warming up, should pass


def test_static_filter_suppresses_stationary_objects():
    """A box that stays in the same spot for 20+ frames gets filtered out."""
    sf = StaticObjectFilter()
    stationary = [_box(100, 100, 200, 200)]
    for _ in range(25):
        result = sf.filter_boxes(stationary)
    assert len(result) == 0  # should be suppressed now


def test_static_filter_keeps_moving_objects():
    """A box that moves significantly each frame should not be filtered."""
    sf = StaticObjectFilter()
    for i in range(25):
        moving = [_box(i * 50, 100, i * 50 + 100, 200)]
        result = sf.filter_boxes(moving)
    assert len(result) == 1  # always moving, never suppressed


def test_static_filter_mixed():
    """Static objects are removed while moving ones are kept."""
    sf = StaticObjectFilter()
    for i in range(25):
        boxes = [
            _box(100, 100, 200, 200),          # stationary
            _box(i * 50, 300, i * 50 + 100, 400),  # moving
        ]
        result = sf.filter_boxes(boxes)
    assert len(result) == 1
    assert result[0]["y1"] == 300  # the moving box survives


def test_static_filter_reports_warmup_state():
    """is_warm flips exactly when filter_boxes starts actually filtering."""
    sf = StaticObjectFilter()
    boxes = [_box(100, 100, 200, 200)]
    for _ in range(11):
        assert sf.filter_boxes(boxes) == boxes  # passes everything through
        assert not sf.is_warm
    sf.filter_boxes(boxes)
    assert sf.is_warm
    assert sf.warmup_progress == (12, 12)


def test_worker_backs_off_on_repeated_capture_failure():
    """A dead stream must not be retried every DETECTION_INTERVAL forever."""
    w = _yolo_worker("starbucks", "http://dead/x.m3u8")
    assert w.next_interval() == DETECTION_INTERVAL
    intervals = []
    for _ in range(8):
        w.consecutive_failures += 1
        intervals.append(w.next_interval())
    assert intervals[0] > DETECTION_INTERVAL
    assert intervals == sorted(intervals)          # monotonically increasing
    assert max(intervals) == MAX_BACKOFF           # and bounded
    w.consecutive_failures = 0
    assert w.next_interval() == DETECTION_INTERVAL  # recovers on a good frame


def test_worker_recovers_after_transient_failure(snapshots_dir):
    """One bad capture followed by good ones leaves no lingering backoff."""
    frame = np.full((80, 80, 3), 200, dtype=np.uint8)
    frame[::4, :, :] = 0  # enough edges to clear the static-frame check
    captures = [None]  # first cycle fails, every later cycle succeeds
    inserted = []

    def fake_capture(*_a, **_kw):
        return captures.pop(0) if captures else frame

    with patch("backend.detector.capture_frame", side_effect=fake_capture), \
         patch("backend.detector.detect_people", return_value=(2, [_box(10, 10, 60, 60)])), \
         patch("backend.detector.DETECTION_INTERVAL", 0.2), \
         patch("backend.database.insert_detection", side_effect=lambda *a, **kw: inserted.append((a, kw))):
        w = _yolo_worker("coffeecart", "http://x/y.m3u8")
        w._filter._history = [[] for _ in range(20)]  # pretend the filter is warm
        w.start()
        deadline = time.time() + 30
        while not inserted and time.time() < deadline:
            time.sleep(0.05)
        w.stop()

    assert inserted, "worker never recorded a detection after recovering"
    assert w.consecutive_failures == 0, "backoff should reset after a good frame"


def test_worker_stop_is_prompt_during_backoff():
    """stop() must not block for the full backoff window."""
    with patch("backend.detector.capture_frame", return_value=None):
        w = _yolo_worker("starbucks", "http://dead/x.m3u8")
        w.consecutive_failures = 10  # would otherwise wait MAX_BACKOFF seconds
        w.start()
        time.sleep(0.5)
        t0 = time.time()
        w.stop()
        elapsed = time.time() - t0
    assert elapsed < 5, f"stop() took {elapsed:.1f}s, backoff was {MAX_BACKOFF}s"


def test_warmup_frames_are_not_recorded(snapshots_dir):
    """Counts taken before the static filter is warm stay out of the database."""
    frame = np.full((80, 80, 3), 200, dtype=np.uint8)
    frame[::4, :, :] = 0
    inserted = []

    with patch("backend.detector.capture_frame", return_value=frame), \
         patch("backend.detector.detect_people", return_value=(1, [_box(10, 10, 60, 60)])), \
         patch("backend.database.insert_detection", side_effect=lambda *a, **kw: inserted.append((a, kw))):
        w = _yolo_worker("starbucks", "http://x/y.m3u8")
        w.start()
        time.sleep(1.5)   # several cycles, all inside the warmup window
        w.stop()

    assert inserted == [], "inflated warmup counts must not reach the detections table"
    from backend.detector import latest_detections
    assert latest_detections["starbucks"]["recorded"] is False
    assert latest_detections["starbucks"]["count"] == 1  # still shown live
