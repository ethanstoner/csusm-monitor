"""Tests for the detection pipeline utilities."""
from backend.detector import StaticObjectFilter


def _box(x1, y1, x2, y2, conf=0.9):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf}


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
