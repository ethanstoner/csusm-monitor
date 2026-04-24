import os
from pathlib import Path

DETECTION_INTERVAL = 5  # seconds between detection cycles
HEALTH_TIMEOUT = 30     # seconds before camera is marked unhealthy
TIMEZONE = "America/Los_Angeles"
FFMPEG_TIMEOUT = 10     # seconds before ffmpeg capture times out
CONFIDENCE_THRESHOLD = 0.45  # minimum YOLO confidence to count as a person
MIN_FRAME_BRIGHTNESS = 15  # skip frames darker than this (0-255 avg)
MIN_BOX_AREA = 1500  # minimum bounding box area in pixels to count (filters tiny false positives)
RETENTION_DAYS = 30     # days to keep raw detection data

DB_PATH = Path(__file__).parent.parent / "data" / "history.db"
SNAPSHOTS_DIR = Path(__file__).parent.parent / "data" / "snapshots"
MAX_SNAPSHOTS = 200  # keep last 200 detection images

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "localhost")
FRIGATE_PORT = int(os.getenv("FRIGATE_PORT", "5000"))

CAMERAS = {
    "starbucks": {
        "name": "Starbucks (USU)",
        "stream_url": "https://stream.csusm.edu/starbucks.m3u8",
    },
    "coffeecart": {
        "name": "Campus Coffee Cart",
        "stream_url": "https://stream.csusm.edu/coffeecart.m3u8",
    },
}
