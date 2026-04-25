import os
from pathlib import Path

# --- Detection pipeline ---
DETECTION_INTERVAL = 5      # seconds between detection cycles per camera
CONFIDENCE_THRESHOLD = 0.45 # minimum YOLOv8 confidence to count as person
MIN_FRAME_BRIGHTNESS = 15   # skip frames darker than this (0-255 mean)
MIN_BOX_AREA = 1500         # minimum bounding box area (px²) — filters tiny FPs
FFMPEG_TIMEOUT = 10         # seconds before ffmpeg capture is killed

# --- Health & retention ---
HEALTH_TIMEOUT = 30         # seconds of silence before camera is "unhealthy"
RETENTION_DAYS = 30         # days of raw detection rows to keep

# --- Paths ---
TIMEZONE = "America/Los_Angeles"
DB_PATH = Path(__file__).parent.parent / "data" / "history.db"
SNAPSHOTS_DIR = Path(__file__).parent.parent / "data" / "snapshots"
MAX_SNAPSHOTS = 200

# --- Frigate / MQTT (override via .env) ---
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "localhost")
FRIGATE_PORT = int(os.getenv("FRIGATE_PORT", "5000"))

# --- Camera definitions ---
# Adding a camera only requires a new entry here; a DetectionWorker
# is auto-spawned for each key at startup.
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
