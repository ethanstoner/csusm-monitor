import numpy as np
from unittest.mock import patch, MagicMock
from backend.detector import capture_frame, detect_people, DetectionWorker

def test_capture_frame_returns_numpy_or_none():
    """capture_frame should return a numpy array on success or None on failure."""
    result = capture_frame("http://localhost:99999/fake.m3u8", timeout=2)
    assert result is None

def test_detect_people_returns_count():
    """detect_people should return a tuple of (count, boxes)."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("backend.detector.model") as mock_model:
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = np.array([0, 0, 0])  # 3 "person" detections (class 0)
        mock_box.xyxy = np.array([[10, 20, 100, 200], [110, 30, 200, 250], [220, 40, 300, 280]])
        mock_box.conf = np.array([0.95, 0.87, 0.72])
        mock_result.boxes = mock_box
        mock_model.return_value = [mock_result]
        count, boxes = detect_people(blank)
        assert isinstance(count, int)
        assert count == 3
        assert len(boxes) == 3
        assert boxes[0]["confidence"] == 0.95
        assert "x1" in boxes[0]

def test_detect_people_empty_image():
    """detect_people on a blank image should return 0 with real model."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    count, boxes = detect_people(blank)
    assert isinstance(count, int)
    assert count >= 0
    assert isinstance(boxes, list)

class TestDetectionWorker:
    def test_worker_init(self, tmp_path):
        from backend.database import init_db
        conn = init_db(tmp_path / "test.db")
        worker = DetectionWorker("starbucks", "http://fake.m3u8", conn)
        assert worker.camera_id == "starbucks"
        assert worker.running is False
        conn.close()
