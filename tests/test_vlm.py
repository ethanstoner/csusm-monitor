"""Tests for the open-vocabulary detection backend.

None of these need a GPU or the model. That is the point: the parts most likely
to break in production are the ones that handle the grounding service being
absent, slow, or returning something the parser has never seen, and those must
be exercised on every CI run — not only on the one machine with a 4090 in it.
"""
import base64
import threading
import time
from datetime import datetime

import numpy as np
import pytest

from backend.database import init_db
from backend.vlm import (
    LocateAnythingClient,
    OpenVocabWorker,
    latest_observations,
    parse_boxes,
)


def _tokens(*coords):
    return "<box>" + "".join(f"<{c}>" for c in coords) + "</box>"


@pytest.fixture(autouse=True)
def _clean_module_state():
    from backend.detector import _detections_lock, latest_detections
    from backend.vlm import _observations_lock
    with _detections_lock:
        latest_detections.clear()
    with _observations_lock:
        latest_observations.clear()
    yield
    with _detections_lock:
        latest_detections.clear()
    with _observations_lock:
        latest_observations.clear()


@pytest.fixture
def frame():
    return np.full((1080, 1920, 3), 128, dtype=np.uint8)


@pytest.fixture
def conn(tmp_path):
    c = init_db(tmp_path / "test.db")
    yield c
    c.close()


# --- coordinate decoding -------------------------------------------------

def test_parse_boxes_scales_the_0_1000_grid_to_pixels():
    """Coordinates are normalised to 0-1000 regardless of input resolution."""
    boxes = parse_boxes(_tokens(100, 200, 300, 400), width=1920, height=1080)
    assert boxes == [{"x1": 192, "y1": 216, "x2": 576, "y2": 432}]


def test_parse_boxes_reads_several_boxes():
    answer = _tokens(0, 0, 100, 100) + " " + _tokens(500, 500, 600, 600)
    assert len(parse_boxes(answer, 1000, 1000)) == 2


def test_parse_boxes_honours_the_alternate_coordinate_order():
    """The release ships two contradictory box orders; both must be decodable."""
    boxes = parse_boxes(_tokens(100, 300, 200, 400), 1000, 1000, order="xxyy")
    assert boxes == [{"x1": 100, "y1": 200, "x2": 300, "y2": 400}]


def test_parse_boxes_drops_truncated_boxes():
    """A box cut short by the token budget is not a box."""
    assert parse_boxes(_tokens(100, 200, 300), 1000, 1000) == []


def test_parse_boxes_drops_out_of_grid_coordinates():
    assert parse_boxes(_tokens(100, 200, 1200, 400), 1000, 1000) == []


def test_parse_boxes_drops_inverted_rectangles():
    """x2 <= x1 is not a rectangle; repairing it would invent a detection."""
    assert parse_boxes(_tokens(500, 200, 100, 400), 1000, 1000) == []
    assert parse_boxes(_tokens(100, 400, 300, 200), 1000, 1000) == []


def test_parse_boxes_handles_a_refusal_or_empty_answer():
    """'<box>none</box>' and prose both mean zero, not a crash."""
    assert parse_boxes("<box>none</box>", 1000, 1000) == []
    assert parse_boxes("I cannot find any people in this image.", 1000, 1000) == []
    assert parse_boxes("", 1000, 1000) == []


def test_parse_boxes_keeps_good_boxes_from_a_partly_broken_answer():
    """One malformed box must not discard the valid ones beside it."""
    answer = _tokens(100, 100, 200, 200) + _tokens(9999, 0, 1, 2) + _tokens(300, 300, 400, 400)
    assert len(parse_boxes(answer, 1000, 1000)) == 2


# --- client failure modes ------------------------------------------------

def test_client_returns_boxes_on_a_good_response(frame):
    import httpx
    import respx
    with respx.mock:
        respx.post("http://localhost:8100/locate").mock(
            return_value=httpx.Response(200, json={"raw": _tokens(0, 0, 500, 500), "latency_ms": 812.0})
        )
        result = LocateAnythingClient().locate(frame, "person")
    assert result["count"] == 1
    assert result["latency_ms"] == 812.0
    assert result["boxes"][0]["x2"] == 960


def test_client_downscales_but_still_returns_full_resolution_boxes(frame):
    """Downscaling is a cost optimisation, not a change to the output.

    Coordinates come back on a resolution-independent 0-1000 grid, so a frame
    sent at 960px wide still decodes to boxes in the original 1920x1080 space.
    """
    import cv2
    import httpx
    import respx

    sent_sizes = []

    def _capture(request):
        payload = __import__("json").loads(request.content)
        buf = np.frombuffer(base64.b64decode(payload["image"]), np.uint8)
        sent_sizes.append(cv2.imdecode(buf, cv2.IMREAD_COLOR).shape[:2])
        return httpx.Response(200, json={"raw": _tokens(0, 0, 1000, 1000), "latency_ms": 1.0})

    with respx.mock:
        respx.post("http://localhost:8100/locate").mock(side_effect=_capture)
        result = LocateAnythingClient(max_image_side=960).locate(frame, "person")

    assert sent_sizes == [(540, 960)], "frame was not downscaled before transmission"
    assert result["boxes"][0] == {"x1": 0, "y1": 0, "x2": 1920, "y2": 1080}


def test_client_does_not_upscale_a_small_frame(frame):
    """A frame already under the limit is sent untouched."""
    import cv2
    import httpx
    import respx

    small = np.full((360, 640, 3), 128, dtype=np.uint8)
    sent_sizes = []

    def _capture(request):
        payload = __import__("json").loads(request.content)
        buf = np.frombuffer(base64.b64decode(payload["image"]), np.uint8)
        sent_sizes.append(cv2.imdecode(buf, cv2.IMREAD_COLOR).shape[:2])
        return httpx.Response(200, json={"raw": "", "latency_ms": 1.0})

    with respx.mock:
        respx.post("http://localhost:8100/locate").mock(side_effect=_capture)
        LocateAnythingClient(max_image_side=960).locate(small, "person")

    assert sent_sizes == [(360, 640)]


def test_client_returns_none_when_the_service_is_down(frame):
    """An unreachable GPU box must not raise into the worker loop."""
    import httpx
    import respx
    with respx.mock:
        respx.post("http://localhost:8100/locate").mock(side_effect=httpx.ConnectError("refused"))
        client = LocateAnythingClient()
        assert client.locate(frame, "person") is None
    assert "ConnectError" in client.last_error


def test_client_discards_a_runaway_generation(frame):
    """340 well-formed boxes for a 6-object query is a broken generation.

    No per-box check catches this: every box is individually valid. Recording
    the count would put a number two orders of magnitude too large into a time
    series that gets averaged.
    """
    import httpx
    import respx
    from backend.vlm import MAX_PLAUSIBLE_BOXES

    runaway = "".join(_tokens(i, i, i + 1, i + 1) for i in range(MAX_PLAUSIBLE_BOXES + 5))
    with respx.mock:
        respx.post("http://localhost:8100/locate").mock(
            return_value=httpx.Response(200, json={"raw": runaway, "latency_ms": 30000.0})
        )
        client = LocateAnythingClient()
        assert client.locate(frame, "trash can") is None
    assert "runaway" in client.last_error


def test_client_accepts_a_large_but_plausible_count(frame):
    """The guard must not clip a genuinely busy scene."""
    import httpx
    import respx
    from backend.vlm import MAX_PLAUSIBLE_BOXES

    busy = "".join(_tokens(i, i, i + 1, i + 1) for i in range(MAX_PLAUSIBLE_BOXES))
    with respx.mock:
        respx.post("http://localhost:8100/locate").mock(
            return_value=httpx.Response(200, json={"raw": busy, "latency_ms": 1200.0})
        )
        result = LocateAnythingClient().locate(frame, "person")
    assert result["count"] == MAX_PLAUSIBLE_BOXES


def test_client_returns_none_on_a_server_error(frame):
    import httpx
    import respx
    with respx.mock:
        respx.post("http://localhost:8100/locate").mock(return_value=httpx.Response(500))
        client = LocateAnythingClient()
        assert client.locate(frame, "person") is None
    assert "500" in client.last_error


def test_client_returns_none_on_a_timeout(frame):
    import httpx
    import respx
    with respx.mock:
        respx.post("http://localhost:8100/locate").mock(
            side_effect=httpx.ReadTimeout("too slow")
        )
        client = LocateAnythingClient()
        assert client.locate(frame, "person") is None
    assert "ReadTimeout" in client.last_error


def test_client_returns_none_on_a_malformed_payload(frame):
    """A 200 carrying the wrong shape is still a failed cycle."""
    import httpx
    import respx
    with respx.mock:
        respx.post("http://localhost:8100/locate").mock(
            return_value=httpx.Response(200, json={"unexpected": True})
        )
        client = LocateAnythingClient()
        assert client.locate(frame, "person") is None
    assert "raw text" in client.last_error


# --- worker --------------------------------------------------------------

class _FakeClient:
    """Stand-in for the grounding service with scriptable answers."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []
        self.last_error = "service unavailable"
        self.timeout = 5

    def locate(self, frame, query):
        self.calls.append(query)
        return self.answers.get(query)


def _publish_frame(camera_id, frame):
    from backend.detector import _detections_lock, latest_detections
    with _detections_lock:
        latest_detections[camera_id] = {"frame": frame, "boxes": [], "count": 0,
                                        "timestamp": datetime.now().isoformat(),
                                        "recorded": True}


def test_worker_records_one_observation_per_query(conn, frame):
    _publish_frame("coffeecart", frame)
    client = _FakeClient({
        "person": {"boxes": [], "count": 4, "latency_ms": 900.0, "raw": ""},
        "bicycle": {"boxes": [], "count": 2, "latency_ms": 880.0, "raw": ""},
    })
    worker = OpenVocabWorker("coffeecart", conn, client=client, queries=["person", "bicycle"])

    assert worker.run_once() == 2
    rows = conn.execute("SELECT label, count, latency_ms FROM observations ORDER BY label").fetchall()
    assert rows == [("bicycle", 2, 880), ("person", 4, 900)]


def test_worker_reuses_the_frame_yolo_already_captured(conn, frame):
    """No second ffmpeg, and both models score identical pixels."""
    _publish_frame("coffeecart", frame)
    seen = []

    class _Recording(_FakeClient):
        def locate(self, f, query):
            seen.append(f)
            return super().locate(f, query)

    client = _Recording({"person": {"boxes": [], "count": 1, "latency_ms": 5.0, "raw": ""}})
    OpenVocabWorker("coffeecart", conn, client=client, queries=["person"]).run_once()

    assert len(seen) == 1
    assert seen[0] is frame


def test_worker_records_nothing_when_the_camera_has_no_frame(conn):
    """An offline camera produces no observations rather than a zero count."""
    client = _FakeClient({"person": {"boxes": [], "count": 3, "latency_ms": 5.0, "raw": ""}})
    worker = OpenVocabWorker("coffeecart", conn, client=client, queries=["person"])

    assert worker.run_once() == 0
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 0
    assert client.calls == []
    assert "no frame" in worker.last_error


def test_worker_survives_the_service_being_down(conn, frame):
    """A dead grounding service costs the cycle, not the worker."""
    _publish_frame("coffeecart", frame)
    client = _FakeClient({})  # every query returns None
    worker = OpenVocabWorker("coffeecart", conn, client=client, queries=["person", "bicycle"])

    assert worker.run_once() == 0
    assert client.calls == ["person", "bicycle"], "one failure must not skip the rest"
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 0
    assert worker.last_error == "service unavailable"


def test_worker_records_the_queries_that_did_answer(conn, frame):
    """A partial outage stores what it got instead of discarding the cycle."""
    _publish_frame("coffeecart", frame)
    client = _FakeClient({"person": {"boxes": [], "count": 6, "latency_ms": 700.0, "raw": ""}})
    worker = OpenVocabWorker("coffeecart", conn, client=client, queries=["person", "bicycle"])

    assert worker.run_once() == 1
    assert conn.execute("SELECT label FROM observations").fetchall() == [("person",)]


def test_worker_publishes_latest_observations_for_the_api(conn, frame):
    _publish_frame("coffeecart", frame)
    client = _FakeClient({"person": {"boxes": [{"x1": 1, "y1": 2, "x2": 3, "y2": 4}],
                                     "count": 1, "latency_ms": 700.0, "raw": ""}})
    OpenVocabWorker("coffeecart", conn, client=client, queries=["person"]).run_once()

    assert latest_observations["coffeecart"]["person"]["count"] == 1
    assert latest_observations["coffeecart"]["person"]["latency_ms"] == 700


def test_worker_without_queries_never_starts(conn):
    """A camera with nothing to ask must not spawn an idle thread."""
    before = threading.active_count()
    worker = OpenVocabWorker("coffeecart", conn, client=_FakeClient({}), queries=[])
    worker.start()
    try:
        assert threading.active_count() == before
        assert not worker.running
    finally:
        worker.stop()


def test_worker_stop_is_prompt(conn, frame):
    """stop() must not wait out a 5-minute interval."""
    _publish_frame("coffeecart", frame)
    client = _FakeClient({"person": {"boxes": [], "count": 0, "latency_ms": 1.0, "raw": ""}})
    worker = OpenVocabWorker("coffeecart", conn, client=client, queries=["person"], interval=300)
    worker.start()
    deadline = time.time() + 5
    while not client.calls and time.time() < deadline:
        time.sleep(0.02)
    t0 = time.time()
    worker.stop()
    elapsed = time.time() - t0

    assert client.calls, "worker never ran a cycle"
    assert elapsed < 5, f"stop() took {elapsed:.1f}s against a 300s interval"


def test_worker_keeps_looping_after_an_unexpected_error(conn, frame):
    """An exception mid-cycle is logged and the next cycle still runs."""
    _publish_frame("coffeecart", frame)
    calls = []

    class _Exploding(_FakeClient):
        def locate(self, f, query):
            calls.append(query)
            raise RuntimeError("boom")

    worker = OpenVocabWorker("coffeecart", conn, client=_Exploding({}),
                             queries=["person"], interval=0.05)
    worker.start()
    deadline = time.time() + 5
    while len(calls) < 2 and time.time() < deadline:
        time.sleep(0.02)
    worker.stop()

    assert len(calls) >= 2, "worker died on the first exception"
    assert "RuntimeError" in worker.last_error
