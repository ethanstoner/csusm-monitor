import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from backend.config import CAMERAS, TIMEZONE, RETENTION_DAYS

TZ = ZoneInfo(TIMEZONE)


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize database, create tables, seed cameras, enable WAL mode."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            camera      TEXT NOT NULL,
            count       INTEGER NOT NULL,
            timestamp   DATETIME NOT NULL,
            day_of_week INTEGER NOT NULL,
            hour        INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_camera_time ON detections(camera, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_trends ON detections(camera, day_of_week, hour)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id         TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            stream_url TEXT NOT NULL,
            active     BOOLEAN NOT NULL DEFAULT 1
        )
    """)
    for cam_id, cam in CAMERAS.items():
        conn.execute(
            "INSERT OR IGNORE INTO cameras (id, name, stream_url) VALUES (?, ?, ?)",
            (cam_id, cam["name"], cam["stream_url"]),
        )
    conn.commit()
    return conn


def insert_detection(conn: sqlite3.Connection, camera: str, count: int, ts: datetime) -> None:
    """Insert a detection result. Stores timestamp as naive Pacific local time string."""
    pacific_ts = ts.replace(tzinfo=TZ) if ts.tzinfo is None else ts.astimezone(TZ)
    # Store as naive local time so SQLite date/time functions work correctly
    naive_str = pacific_ts.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO detections (camera, count, timestamp, day_of_week, hour) VALUES (?, ?, ?, ?, ?)",
        (camera, count, naive_str, pacific_ts.weekday(), pacific_ts.hour),
    )
    conn.commit()


def get_latest_counts(conn: sqlite3.Connection) -> list[dict]:
    """Get most recent detection per camera with name and stream_url."""
    rows = conn.execute("""
        SELECT c.id, c.name, c.stream_url, c.active,
               d.count, d.timestamp
        FROM cameras c
        LEFT JOIN detections d ON d.camera = c.id
            AND d.id = (SELECT MAX(id) FROM detections WHERE camera = c.id)
        WHERE c.active = 1
    """).fetchall()
    return [
        {
            "id": r[0], "name": r[1], "stream_url": r[2], "active": bool(r[3]),
            "count": r[4], "timestamp": r[5],
        }
        for r in rows
    ]


def get_heatmap_data(conn: sqlite3.Connection, camera: str, days: int = 7) -> list[dict]:
    """Get average count by day_of_week and hour for the given lookback period."""
    cutoff = (datetime.now(TZ) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    rows = conn.execute("""
        SELECT day_of_week, hour, AVG(count) as avg_count
        FROM detections
        WHERE camera = ? AND timestamp >= ?
        GROUP BY day_of_week, hour
        ORDER BY day_of_week, hour
    """, (camera, cutoff)).fetchall()
    return [{"day_of_week": r[0], "hour": r[1], "avg_count": round(r[2], 1)} for r in rows]


def get_timeline_data(conn: sqlite3.Connection, camera: str, date: str) -> list[dict]:
    """Get 1-minute averaged time series for a specific date."""
    rows = conn.execute("""
        SELECT strftime('%H:%M', timestamp) as time_min, AVG(count) as avg_count
        FROM detections
        WHERE camera = ? AND date(timestamp) = ?
        GROUP BY time_min
        ORDER BY time_min
    """, (camera, date)).fetchall()
    return [{"time": r[0], "avg_count": round(r[1], 1)} for r in rows]


def get_hourly_averages(conn: sqlite3.Connection, camera: str, day_type: str = "all", days: int = 30) -> list[dict]:
    """Average count by hour of day, filtered by weekday/weekend/all."""
    cutoff = (datetime.now(TZ) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    if day_type == "weekday":
        dow_filter = "AND day_of_week < 5"
    elif day_type == "weekend":
        dow_filter = "AND day_of_week >= 5"
    else:
        dow_filter = ""
    rows = conn.execute(f"""
        SELECT hour, AVG(count) as avg_count
        FROM detections
        WHERE camera = ? AND timestamp >= ? {dow_filter}
        GROUP BY hour
        ORDER BY hour
    """, (camera, cutoff)).fetchall()
    return [{"hour": r[0], "avg_count": round(r[1], 1)} for r in rows]


def cleanup_old_data(conn: sqlite3.Connection, retention_days: int = RETENTION_DAYS) -> int:
    """Delete detections older than retention_days. Returns number of rows deleted."""
    cutoff = (datetime.now(TZ) - timedelta(days=retention_days)).strftime("%Y-%m-%d %H:%M:%S")
    cursor = conn.execute("DELETE FROM detections WHERE timestamp < ?", (cutoff,))
    conn.commit()
    return cursor.rowcount
