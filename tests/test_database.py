import sqlite3
from datetime import datetime
from backend.database import init_db, insert_detection, get_latest_counts, get_heatmap_data, get_timeline_data, cleanup_old_data

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

def test_cleanup_old_data(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_detection(conn, "starbucks", 5, datetime(2026, 2, 10, 10, 0, 0))
    insert_detection(conn, "starbucks", 3, datetime(2026, 4, 12, 10, 0, 0))
    cleanup_old_data(conn, retention_days=30)
    rows = conn.execute("SELECT count(*) FROM detections").fetchone()[0]
    assert rows == 1
    conn.close()
