from backend.config import CAMERAS, DETECTION_INTERVAL, DB_PATH, TIMEZONE, HEALTH_TIMEOUT, MQTT_HOST, MQTT_PORT, FRIGATE_HOST, FRIGATE_PORT

def test_cameras_defined():
    assert len(CAMERAS) == 2
    assert "starbucks" in CAMERAS
    assert "coffeecart" in CAMERAS
    for cam_id, cam in CAMERAS.items():
        assert "name" in cam
        assert "stream_url" in cam
        assert cam["stream_url"].endswith(".m3u8")

def test_detection_interval():
    assert DETECTION_INTERVAL == 5

def test_timezone():
    assert TIMEZONE == "America/Los_Angeles"

def test_health_timeout():
    assert HEALTH_TIMEOUT == 30

def test_db_path():
    assert "data" in str(DB_PATH)
    assert str(DB_PATH).endswith("history.db")

def test_mqtt_config_defaults():
    assert isinstance(MQTT_HOST, str)
    assert isinstance(MQTT_PORT, int)
    assert isinstance(FRIGATE_HOST, str)
    assert isinstance(FRIGATE_PORT, int)
