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

# --- Data collectors ---
WEATHER_INTERVAL = 900       # 15 min
PARKING_INTERVAL = 300       # 5 min
AQI_INTERVAL = 1800          # 30 min
TRANSIT_REFRESH_INTERVAL = 86400  # daily GTFS refresh
EVENTS_INTERVAL = 21600      # 6 hours

# --- CSUSM location ---
CAMPUS_LAT = 33.1284
CAMPUS_LON = -117.1597

# --- Data paths ---
GTFS_DIR = Path(__file__).parent.parent / "data" / "gtfs"

# --- Open-vocabulary detection (LocateAnything-3B, out of process) ---
# Off unless a grounding service is configured. The weights are NVIDIA-licensed
# for academic and non-profit research only, so this cannot be the default
# backend of a repo other people clone (see the licence note in the README).
VLM_ENABLED = os.getenv("VLM_ENABLED", "0") == "1"
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8100")
VLM_TIMEOUT = float(os.getenv("VLM_TIMEOUT", "120"))  # a 3B VLM is not a 10s request
# MoonViT tokenises at the image's native resolution, so a 1920x1080 frame is
# the single largest cost in the pipeline. See bench/README.md for the measured
# latency and VRAM at each setting — this default is chosen from that sweep, not
# from taste.
VLM_MAX_IMAGE_SIDE = int(os.getenv("VLM_MAX_IMAGE_SIDE", "1440"))
OPEN_VOCAB_INTERVAL = float(os.getenv("OPEN_VOCAB_INTERVAL", "300"))  # seconds per camera

# Natural-language queries asked of each camera. Adding a question here is the
# entire cost of adding one — no retraining, no labelled data, no new model.
OPEN_VOCAB_QUERIES = {
    "coffeecart": ["person", "bicycle", "backpack"],
    "starbucks": ["person", "bicycle", "backpack"],
}

# --- Frigate / MQTT (override via .env) ---
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "localhost")
FRIGATE_PORT = int(os.getenv("FRIGATE_PORT", "5000"))

# --- Camera definitions ---
# Adding a camera only requires a new entry here; a DetectionWorker
# is auto-spawned for each key at startup.
# open_hours: (start_hour, end_hour) in 24h format — used to filter analytics
CAMERAS = {
    "starbucks": {
        "name": "Starbucks (USU)",
        "stream_url": "https://stream.csusm.edu/starbucks.m3u8",
        "open_hours": (6, 21),  # 6 AM - 9 PM
    },
    "coffeecart": {
        "name": "Campus Coffee Cart",
        "stream_url": "https://stream.csusm.edu/coffeecart.m3u8",
        "open_hours": (7, 17),  # 7 AM - 5 PM
    },
}
