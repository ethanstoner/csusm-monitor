"""Tests for choosing between the grounding model and YOLO at detection time.

LocateAnything-3B is the primary person detector; YOLOv8n is the fallback. The
interesting behaviour is not either model — it is what happens at the seam,
because the seam is crossed by a GPU service on another operating system that
will not always be there.
"""
import time
from unittest.mock import patch

import numpy as np
import pytest

from backend.database import init_db
from backend.detector import PersonCounters


def _box(x1, y1, x2, y2, conf=0.9):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf}


@pytest.fixture
def frame():
    f = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    f[::4, :, :] = 0  # edges, so is_static_frame does not skip it
    return f


@pytest.fixture
def conn(tmp_path):
    c = init_db(tmp_path / "test.db")
    yield c
    c.close()


class _Service:
    """Grounding service stand-in whose availability can be flipped mid-test."""

    def __init__(self, up=True, count=3):
        self.up = up
        self.count = count
        self.calls = 0
        self.last_error = "grounding service unreachable"
        self.timeout = 5
        self.max_image_side = 1440

    def locate(self, frame, query):
        self.calls += 1
        if not self.up:
            return None
        boxes = [_box(10 * i, 10, 10 * i + 50, 200) for i in range(self.count)]
        return {"boxes": boxes, "count": self.count, "latency_ms": 1200.0, "raw": ""}


# --- selection ------------------------------------------------------------

def test_auto_prefers_the_grounding_model(frame):
    counters = PersonCounters(backend="auto", client=_Service(up=True))
    count, boxes, source = counters.count_people(frame)
    assert source == "LocateAnything-3B"
    assert count == 3


def test_auto_falls_back_to_yolo_when_the_service_is_down(frame):
    service = _Service(up=False)
    counters = PersonCounters(backend="auto", client=service)
    with patch("backend.detector.detect_people", return_value=(2, [_box(0, 0, 10, 10)])) as yolo:
        count, boxes, source = counters.count_people(frame)
    assert service.calls == 1, "it should try the grounding model first"
    assert yolo.called
    assert source == "yolov8n"
    assert count == 2


def test_vlm_only_mode_does_not_fall_back(frame):
    """Explicitly asking for the grounding model means a gap, not a substitute.

    A reading silently produced by a different detector than the one requested
    is worse than no reading, because the row looks identical either way.
    """
    counters = PersonCounters(backend="vlm", client=_Service(up=False))
    with patch("backend.detector.detect_people") as yolo:
        result = counters.count_people(frame)
    assert result is None
    assert not yolo.called


def test_yolo_only_mode_never_calls_the_service(frame):
    service = _Service(up=True)
    counters = PersonCounters(backend="yolo", client=service)
    with patch("backend.detector.detect_people", return_value=(5, [])):
        count, _boxes, source = counters.count_people(frame)
    assert service.calls == 0
    assert source == "yolov8n"
    assert count == 5


def test_unknown_backend_name_is_rejected_at_construction():
    """A typo in DETECTION_BACKEND must fail loudly, not silently pick one."""
    with pytest.raises(ValueError, match="DETECTION_BACKEND"):
        PersonCounters(backend="locateanything", client=_Service())


# --- probe caching --------------------------------------------------------

def test_a_down_service_is_not_retried_every_cycle(frame):
    """Each retry costs a connection attempt and its timeout before falling back."""
    service = _Service(up=False)
    counters = PersonCounters(backend="auto", client=service, probe_interval=60)
    with patch("backend.detector.detect_people", return_value=(0, [])):
        for _ in range(5):
            counters.count_people(frame)
    assert service.calls == 1, f"probed {service.calls} times inside the backoff window"


def test_the_service_is_retried_once_the_probe_window_expires(frame):
    service = _Service(up=False)
    counters = PersonCounters(backend="auto", client=service, probe_interval=0.05)
    with patch("backend.detector.detect_people", return_value=(0, [])):
        counters.count_people(frame)
        time.sleep(0.08)
        counters.count_people(frame)
    assert service.calls == 2


def test_recovery_switches_back_to_the_grounding_model(frame):
    service = _Service(up=False)
    counters = PersonCounters(backend="auto", client=service, probe_interval=0)
    with patch("backend.detector.detect_people", return_value=(1, [])):
        _, _, first = counters.count_people(frame)
    service.up = True
    _, _, second = counters.count_people(frame)

    assert first == "yolov8n"
    assert second == "LocateAnything-3B", "worker never came back after the service returned"


def test_active_source_is_reported_for_the_api(frame):
    counters = PersonCounters(backend="auto", client=_Service(up=True))
    assert counters.active_source is None, "nothing is known before the first cycle"
    counters.count_people(frame)
    assert counters.active_source == "LocateAnything-3B"


# --- the static filter belongs to YOLO ------------------------------------

def test_static_filter_is_applied_only_to_yolo_output(frame, conn):
    """StaticObjectFilter was built and validated against YOLO's failure mode.

    It suppresses anything that holds still, which for YOLO means signs and
    poles. Applied to a grounding model it would equally suppress a person
    sitting at a table — a real detection removed to fix a problem that model
    may not have. It stays on the path it was measured on.
    """
    from backend.detector import DetectionWorker

    stationary = [_box(100, 100, 200, 300)]
    service = _Service(up=True, count=1)
    worker = DetectionWorker("coffeecart", "http://x/y.m3u8", conn,
                             counters=PersonCounters(backend="auto", client=service))

    # Many cycles of an unmoving box: YOLO's would be filtered away by now.
    for _ in range(30):
        count, boxes, source = worker.counters.count_people(frame)
        boxes = worker.apply_filters(boxes, source)
    assert source == "LocateAnything-3B"
    assert len(boxes) == 1, "grounding-model boxes must not go through the YOLO filter"

    # The same worker, on the YOLO path, still filters.
    with patch("backend.detector.detect_people", return_value=(1, stationary)):
        service.up = False
        worker.counters._available = False
        worker.counters._probed_at = time.time()
        for _ in range(30):
            count, boxes, source = worker.counters.count_people(frame)
            boxes = worker.apply_filters(boxes, source)
    assert source == "yolov8n"
    assert boxes == [], "stationary YOLO box should have been suppressed"


def test_grounding_model_readings_are_recorded_immediately(frame, conn):
    """No warm-up discard on the grounding path.

    The 60-second warm-up exists because StaticObjectFilter cannot tell a person
    from a sign until it has history. With no filter on this path there is
    nothing to warm up, and throwing away the first minute of real readings
    would be superstition.
    """
    from backend.detector import DetectionWorker

    worker = DetectionWorker("coffeecart", "http://x/y.m3u8", conn,
                             counters=PersonCounters(backend="auto", client=_Service(up=True)))
    assert worker.should_record("LocateAnything-3B") is True
    assert worker.should_record("yolov8n") is False  # filter still cold
