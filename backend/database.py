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

    # --- Collector tables ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL, apparent_temperature REAL,
            humidity REAL, wind_speed REAL, wind_direction REAL,
            weather_code INTEGER, uv_index REAL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lot_id TEXT NOT NULL, available INTEGER NOT NULL,
            total INTEGER NOT NULL, timestamp TEXT NOT NULL,
            day_of_week INTEGER, hour INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_parking_lot_time ON parking(lot_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_parking_trends ON parking(lot_id, day_of_week, hour)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS air_quality (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            aqi INTEGER, category TEXT, pollutant TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL, event_date TEXT,
            location TEXT, description TEXT,
            fetched_at TEXT NOT NULL,
            UNIQUE(title, event_date)
        )
    """)
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
    if camera == "_all":
        rows = conn.execute("""
            SELECT day_of_week, hour, AVG(count) as avg_count
            FROM detections WHERE timestamp >= ?
            GROUP BY day_of_week, hour
            ORDER BY day_of_week, hour
        """, (cutoff,)).fetchall()
    else:
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
    if camera == "_all":
        rows = conn.execute("""
            SELECT strftime('%H:%M', timestamp) as time_min, MAX(count) as peak_count
            FROM detections WHERE date(timestamp) = ?
            GROUP BY time_min
            ORDER BY time_min
        """, (date,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT strftime('%H:%M', timestamp) as time_min, MAX(count) as peak_count
            FROM detections
            WHERE camera = ? AND date(timestamp) = ?
            GROUP BY time_min
            ORDER BY time_min
        """, (camera, date)).fetchall()
    return [{"time": r[0], "avg_count": r[1]} for r in rows]


def get_hourly_averages(conn: sqlite3.Connection, camera: str, day_type: str = "all", days: int = 30) -> list[dict]:
    """Average count by hour of day, filtered by weekday/weekend/all."""
    cutoff = (datetime.now(TZ) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    if day_type == "weekday":
        dow_filter = "AND day_of_week < 5"
    elif day_type == "weekend":
        dow_filter = "AND day_of_week >= 5"
    else:
        dow_filter = ""
    if camera == "_all":
        rows = conn.execute(f"""
            SELECT hour, AVG(count) as avg_count
            FROM detections WHERE timestamp >= ? {dow_filter}
            GROUP BY hour
            ORDER BY hour
        """, (cutoff,)).fetchall()
    else:
        rows = conn.execute(f"""
            SELECT hour, AVG(count) as avg_count
            FROM detections
            WHERE camera = ? AND timestamp >= ? {dow_filter}
            GROUP BY hour
            ORDER BY hour
        """, (camera, cutoff)).fetchall()
    return [{"hour": r[0], "avg_count": round(r[1], 1)} for r in rows]


def get_best_times(conn: sqlite3.Connection, camera: str, days: int = 7) -> list[dict]:
    """Return hours ranked by lowest average people count (best times to visit)."""
    cutoff = (datetime.now(TZ) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    rows = conn.execute("""
        SELECT hour, AVG(count) as avg_count, COUNT(*) as samples
        FROM detections
        WHERE camera = ? AND timestamp >= ?
        GROUP BY hour
        HAVING samples >= 3
        ORDER BY avg_count ASC
    """, (camera, cutoff)).fetchall()
    return [{"hour": r[0], "avg_count": round(r[1], 1), "samples": r[2]} for r in rows]


def get_daily_totals(conn: sqlite3.Connection, camera: str, days: int = 30) -> list[dict]:
    """Return daily total/average people counts for the given lookback."""
    cutoff = (datetime.now(TZ) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    if camera == "_all":
        rows = conn.execute("""
            SELECT date(timestamp) as day, SUM(count) as total, AVG(count) as avg_count, COUNT(*) as samples
            FROM detections WHERE timestamp >= ?
            GROUP BY day ORDER BY day
        """, (cutoff,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT date(timestamp) as day, SUM(count) as total, AVG(count) as avg_count, COUNT(*) as samples
            FROM detections WHERE camera = ? AND timestamp >= ?
            GROUP BY day ORDER BY day
        """, (camera, cutoff)).fetchall()
    return [{"date": r[0], "total": r[1], "avg_count": round(r[2], 1), "samples": r[3]} for r in rows]


def cleanup_old_data(conn: sqlite3.Connection, retention_days: int = RETENTION_DAYS) -> int:
    """Delete detections older than retention_days. Returns number of rows deleted."""
    cutoff = (datetime.now(TZ) - timedelta(days=retention_days)).strftime("%Y-%m-%d %H:%M:%S")
    cursor = conn.execute("DELETE FROM detections WHERE timestamp < ?", (cutoff,))
    conn.commit()
    return cursor.rowcount


def insert_weather(conn, *, temperature, apparent_temperature, humidity,
                   wind_speed, wind_direction, weather_code, uv_index):
    ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO weather (temperature, apparent_temperature, humidity, wind_speed, wind_direction, weather_code, uv_index, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (temperature, apparent_temperature, humidity, wind_speed, wind_direction, weather_code, uv_index, ts),
    )
    conn.commit()


def insert_parking(conn, *, lot_id, available, total):
    now = datetime.now(TZ)
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO parking (lot_id, available, total, timestamp, day_of_week, hour) VALUES (?, ?, ?, ?, ?, ?)",
        (lot_id, available, total, ts, now.weekday(), now.hour),
    )
    conn.commit()


def insert_air_quality(conn, *, aqi, category, pollutant):
    ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO air_quality (aqi, category, pollutant, timestamp) VALUES (?, ?, ?, ?)",
        (aqi, category, pollutant, ts),
    )
    conn.commit()


def insert_event(conn, *, title, event_date, location, description):
    ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT OR IGNORE INTO events (title, event_date, location, description, fetched_at) VALUES (?, ?, ?, ?, ?)",
        (title, event_date, location, description, ts),
    )
    conn.commit()


def get_latest_weather(conn):
    row = conn.execute("SELECT temperature, apparent_temperature, humidity, wind_speed, wind_direction, weather_code, uv_index, timestamp FROM weather ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    return {"temperature": row[0], "apparent_temperature": row[1], "humidity": row[2],
            "wind_speed": row[3], "wind_direction": row[4], "weather_code": row[5],
            "uv_index": row[6], "timestamp": row[7]}


def get_latest_air_quality(conn):
    row = conn.execute("SELECT aqi, category, pollutant, timestamp FROM air_quality ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    return {"aqi": row[0], "category": row[1], "pollutant": row[2], "timestamp": row[3]}


def get_latest_parking(conn):
    row = conn.execute("SELECT lot_id, available, total, timestamp FROM parking ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    return {"lot_id": row[0], "available": row[1], "total": row[2],
            "percent_full": round((1 - row[1] / row[2]) * 100, 1) if row[2] > 0 else 0,
            "timestamp": row[3]}


def get_parking_trends(conn, lot_id, days=7):
    cutoff = (datetime.now(TZ) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    rows = conn.execute("""
        SELECT day_of_week, hour, AVG(available) as avg_available, AVG(total) as avg_total
        FROM parking WHERE lot_id = ? AND timestamp >= ?
        GROUP BY day_of_week, hour ORDER BY day_of_week, hour
    """, (lot_id, cutoff)).fetchall()
    return [{"day_of_week": r[0], "hour": r[1], "avg_available": round(r[2], 1),
             "avg_percent_full": round((1 - r[2] / r[3]) * 100, 1) if r[3] > 0 else 0}
            for r in rows]


def get_upcoming_events(conn, limit=5):
    rows = conn.execute("""
        SELECT title, event_date, location, description FROM events
        WHERE event_date >= date('now') ORDER BY event_date LIMIT ?
    """, (limit,)).fetchall()
    return [{"title": r[0], "date": r[1], "location": r[2], "description": r[3]} for r in rows]
