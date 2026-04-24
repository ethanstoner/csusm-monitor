import json
import threading
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest
from backend.database import init_db, get_latest_counts


def make_msg(topic, payload):
    """Build a mock paho MQTT message."""
    msg = MagicMock()
    msg.topic = topic
    msg.payload = payload.encode() if isinstance(payload, str) else payload
    return msg


@pytest.fixture
def db(tmp_path):
    return init_db(tmp_path / "test.db")


def make_listener(db):
    """Create a FrigateListener with a mocked MQTT client (no real connection)."""
    with patch("backend.frigate_listener.mqtt.Client") as MockClient:
        MockClient.return_value = MagicMock()
        from backend.frigate_listener import FrigateListener
        return FrigateListener(db)


def test_count_topic_updates_latest_counts(db):
    """Person count MQTT message updates latest_counts dict."""
    listener = make_listener(db)
    msg = make_msg("frigate/starbucks/person", "3")
    listener._on_message(None, None, msg)
    with listener._counts_lock:
        assert listener.latest_counts["starbucks"] == 3


def test_count_topic_inserts_detection(db):
    """Person count MQTT message writes a row to SQLite."""
    listener = make_listener(db)
    msg = make_msg("frigate/starbucks/person", "2")
    listener._on_message(None, None, msg)
    rows = get_latest_counts(db)
    starbucks = next(r for r in rows if r["id"] == "starbucks")
    assert starbucks["count"] == 2


def test_zero_count_still_inserts(db):
    """Zero count is valid (empty camera) and must be written to DB."""
    listener = make_listener(db)
    msg = make_msg("frigate/coffeecart/person", "0")
    listener._on_message(None, None, msg)
    rows = get_latest_counts(db)
    cart = next(r for r in rows if r["id"] == "coffeecart")
    assert cart["count"] == 0


def test_unknown_camera_logs_warning_and_skips(db, caplog):
    """MQTT message for an unknown camera logs a warning and does not insert."""
    import logging
    listener = make_listener(db)
    msg = make_msg("frigate/phantom_cam/person", "1")
    with caplog.at_level(logging.WARNING, logger="backend.frigate_listener"):
        listener._on_message(None, None, msg)
    assert "phantom_cam" in caplog.text
    rows = get_latest_counts(db)
    assert all(r["count"] is None for r in rows)


def test_non_integer_payload_logs_warning(db, caplog):
    """Malformed payload does not crash the listener."""
    import logging
    listener = make_listener(db)
    msg = make_msg("frigate/starbucks/person", "not_a_number")
    with caplog.at_level(logging.WARNING, logger="backend.frigate_listener"):
        listener._on_message(None, None, msg)
    rows = get_latest_counts(db)
    assert all(r["count"] is None for r in rows)


def test_start_stop_lifecycle(db):
    """FrigateListener.start() launches a thread; stop() terminates it cleanly."""
    mock_mqtt = MagicMock()
    # loop_forever blocks forever — make it a no-op so the thread exits cleanly
    mock_mqtt.loop_forever.side_effect = lambda: None
    with patch("backend.frigate_listener.mqtt.Client", return_value=mock_mqtt):
        from backend.frigate_listener import FrigateListener
        listener = FrigateListener(db)
        assert listener.running is False

        listener.start()
        assert listener.running is True
        assert listener._thread is not None

        listener.stop()
        assert listener.running is False
        mock_mqtt.disconnect.assert_called_once()
