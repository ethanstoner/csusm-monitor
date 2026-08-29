import sqlite3
from datetime import datetime, timedelta
from backend.database import init_db, insert_detection, get_latest_counts, get_heatmap_data, get_timeline_data, get_best_times, cleanup_old_data

def test_init_db_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "detections" in tables
    assert "cameras" in tables
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    conn.close()

def test_init_db_seeds_cameras(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    rows = conn.execute("SELECT id, name, stream_url FROM cameras").fetchall()
    assert len(rows) == 2
    ids = {r[0] for r in rows}
    assert ids == {"starbucks", "coffeecart"}
    conn.close()

def test_insert_and_get_latest(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_detection(conn, "starbucks", 5, datetime(2026, 4, 12, 10, 30, 0))
    insert_detection(conn, "starbucks", 3, datetime(2026, 4, 12, 10, 30, 5))
    latest = get_latest_counts(conn)
    starbucks = [c for c in latest if c["id"] == "starbucks"][0]
    assert starbucks["count"] == 3
    conn.close()

def test_get_heatmap_data(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    today_10am = now.replace(hour=10, minute=0, second=0, microsecond=0)
    for i in range(5):
        insert_detection(conn, "starbucks", 4, today_10am.replace(second=i))
    data = get_heatmap_data(conn, "starbucks", days=7)
    assert len(data) > 0
    assert data[0]["day_of_week"] == today_10am.weekday()
    assert data[0]["hour"] == 10
    assert data[0]["avg_count"] == 4.0
    conn.close()

def test_get_timeline_data(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_detection(conn, "starbucks", 2, datetime(2026, 4, 12, 10, 0, 0))
    insert_detection(conn, "starbucks", 4, datetime(2026, 4, 12, 10, 0, 30))
    data = get_timeline_data(conn, "starbucks", "2026-04-12")
    assert len(data) > 0
    assert data[0]["time"] == "10:00"
    assert data[0]["avg_count"] == 4  # MAX of (2, 4) in the same minute
    conn.close()

def test_get_best_times_respects_open_hours(tmp_path):
    """Hours outside the camera's open window are never suggested."""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    from zoneinfo import ZoneInfo
    from backend.config import CAMERAS
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    base = now.replace(minute=0, second=0, microsecond=0) - timedelta(days=1)
    # 3 AM is closed, 10 AM is open — both get enough samples to clear HAVING
    for hour in (3, 10):
        for i in range(5):
            insert_detection(conn, "coffeecart", 1, base.replace(hour=hour, second=i))

    hours = {r["hour"] for r in get_best_times(conn, "coffeecart", days=7)}
    open_start, open_end = CAMERAS["coffeecart"]["open_hours"]
    assert 10 in hours
    assert 3 not in hours
    assert all(open_start <= h < open_end for h in hours)
    conn.close()

def test_get_best_times_unknown_camera_uses_full_day(tmp_path):
    """A camera with no open_hours entry falls back to all 24 hours."""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    base = (now - timedelta(days=1)).replace(hour=3, minute=0, second=0, microsecond=0)
    for i in range(5):
        insert_detection(conn, "mystery", 1, base.replace(second=i))
    assert [r["hour"] for r in get_best_times(conn, "mystery", days=7)] == [3]
    conn.close()

def test_cleanup_old_data(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    old = datetime.now() - timedelta(days=60)
    recent = datetime.now() - timedelta(days=5)
    insert_detection(conn, "starbucks", 5, old)
    insert_detection(conn, "starbucks", 3, recent)
    cleanup_old_data(conn, retention_days=30)
    rows = conn.execute("SELECT count(*) FROM detections").fetchone()[0]
    assert rows == 1
    conn.close()
