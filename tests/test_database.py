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
    assert data[0]["samples"] == 5
    conn.close()

def test_trend_queries_report_sample_counts(tmp_path):
    """An hour averaged from 1 reading must be distinguishable from 200."""
    from zoneinfo import ZoneInfo
    from backend.database import get_hourly_averages
    conn = init_db(tmp_path / "test.db")
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    base = (now - timedelta(days=1)).replace(minute=0, second=0, microsecond=0)
    for i in range(40):                                   # well-covered hour
        insert_detection(conn, "starbucks", 6, base.replace(hour=11, second=i % 60, minute=i // 60))
    insert_detection(conn, "starbucks", 6, base.replace(hour=12))  # one lonely reading

    by_hour = {r["hour"]: r for r in get_hourly_averages(conn, "starbucks", days=7)}
    assert by_hour[11]["samples"] == 40
    assert by_hour[12]["samples"] == 1
    # Identical averages, wildly different confidence — the count is the signal
    assert by_hour[11]["avg_count"] == by_hour[12]["avg_count"] == 6.0

    hm = {(r["day_of_week"], r["hour"]): r["samples"] for r in get_heatmap_data(conn, "starbucks", days=7)}
    assert hm[(base.weekday(), 11)] == 40
    assert hm[(base.weekday(), 12)] == 1
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

def test_detections_record_which_model_produced_them(tmp_path):
    """A count is only interpretable if you know what counted.

    Two detectors now write to this table. Averaging a bucket produced by
    YOLOv8n together with one produced by a grounding model, with no way to
    tell them apart afterwards, is how a heatmap quietly starts meaning
    something different from what it meant last week.
    """
    conn = init_db(tmp_path / "test.db")
    insert_detection(conn, "coffeecart", 3, datetime(2026, 4, 12, 10, 0, 0), source="yolov8n")
    insert_detection(conn, "coffeecart", 4, datetime(2026, 4, 12, 10, 0, 5),
                     source="LocateAnything-3B")

    rows = conn.execute("SELECT count, source FROM detections ORDER BY id").fetchall()
    assert rows == [(3, "yolov8n"), (4, "LocateAnything-3B")]
    conn.close()


def test_source_defaults_when_the_caller_does_not_say(tmp_path):
    """An unlabelled write is recorded as unknown, never silently as YOLO."""
    conn = init_db(tmp_path / "test.db")
    insert_detection(conn, "coffeecart", 1, datetime(2026, 4, 12, 10, 0, 0))
    assert conn.execute("SELECT source FROM detections").fetchone()[0] == "unknown"
    conn.close()


def test_existing_database_gains_the_source_column(tmp_path):
    """init_db migrates a table created before the column existed."""
    import sqlite3 as sq
    db = tmp_path / "legacy.db"
    legacy = sq.connect(str(db))
    legacy.execute("""
        CREATE TABLE detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera TEXT NOT NULL, count INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            day_of_week INTEGER NOT NULL, hour INTEGER NOT NULL
        )
    """)
    legacy.execute(
        "INSERT INTO detections (camera, count, timestamp, day_of_week, hour)"
        " VALUES ('coffeecart', 9, '2026-04-12 10:00:00', 6, 10)")
    legacy.commit()
    legacy.close()

    conn = init_db(db)
    row = conn.execute("SELECT count, source FROM detections").fetchone()
    # The pre-existing row predates the split, so it is labelled with the only
    # detector that existed then rather than left null.
    assert row == (9, "yolov8n")
    conn.close()


def test_source_breakdown_reports_what_produced_a_camera_history(tmp_path):
    from zoneinfo import ZoneInfo
    from backend.database import get_detection_sources
    conn = init_db(tmp_path / "test.db")
    # Inside the default 30-day lookback, or the query correctly excludes it.
    base = (datetime.now(ZoneInfo("America/Los_Angeles")) - timedelta(days=1)).replace(
        minute=0, second=0, microsecond=0)
    for i in range(3):
        insert_detection(conn, "coffeecart", 1, base.replace(second=i), source="yolov8n")
    insert_detection(conn, "coffeecart", 1, base.replace(second=30),
                     source="LocateAnything-3B")

    assert get_detection_sources(conn, "coffeecart") == [
        {"source": "yolov8n", "samples": 3},
        {"source": "LocateAnything-3B", "samples": 1},
    ]
    conn.close()


def test_observations_are_stored_separately_from_detections(tmp_path):
    """Open-vocabulary counts must not land in the person-count history.

    `detections.count` carries no label — every trend in the dashboard reads it
    as 'people'. Writing a bicycle count into the same column would silently
    change what months of heatmap history mean.
    """
    from backend.database import insert_observation
    conn = init_db(tmp_path / "test.db")
    insert_detection(conn, "coffeecart", 4, datetime(2026, 4, 12, 10, 0, 0))
    insert_observation(conn, "coffeecart", "bicycle", 9,
                       datetime(2026, 4, 12, 10, 0, 0), latency_ms=1234)

    assert conn.execute("SELECT SUM(count) FROM detections").fetchone()[0] == 4
    assert conn.execute("SELECT SUM(count) FROM observations").fetchone()[0] == 9
    conn.close()


def test_get_latest_observations_returns_one_row_per_label(tmp_path):
    """The newest reading for each (camera, label) pair, not the newest overall."""
    from backend.database import insert_observation, get_latest_observations
    conn = init_db(tmp_path / "test.db")
    base = datetime(2026, 4, 12, 10, 0, 0)
    insert_observation(conn, "coffeecart", "person in line", 2, base)
    insert_observation(conn, "coffeecart", "person in line", 7, base.replace(second=30))
    insert_observation(conn, "coffeecart", "bicycle", 3, base.replace(second=10))

    latest = {(r["camera"], r["label"]): r for r in get_latest_observations(conn)}
    assert latest[("coffeecart", "person in line")]["count"] == 7
    assert latest[("coffeecart", "bicycle")]["count"] == 3
    conn.close()


def test_get_latest_observations_filters_by_camera(tmp_path):
    from backend.database import insert_observation, get_latest_observations
    conn = init_db(tmp_path / "test.db")
    ts = datetime(2026, 4, 12, 10, 0, 0)
    insert_observation(conn, "coffeecart", "bicycle", 3, ts)
    insert_observation(conn, "starbucks", "bicycle", 8, ts)

    rows = get_latest_observations(conn, camera="starbucks")
    assert [r["count"] for r in rows] == [8]
    conn.close()


def test_observation_history_is_scoped_to_one_label(tmp_path):
    """Two labels on the same camera must not average into each other."""
    from zoneinfo import ZoneInfo
    from backend.database import insert_observation, get_observation_history
    conn = init_db(tmp_path / "test.db")
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    base = (now - timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0)
    for i in range(4):
        insert_observation(conn, "coffeecart", "person in line", 10, base.replace(second=i))
        insert_observation(conn, "coffeecart", "bicycle", 0, base.replace(second=i))

    queue = get_observation_history(conn, "coffeecart", "person in line", days=7)
    assert [r["avg_count"] for r in queue] == [10.0]
    assert queue[0]["samples"] == 4
    conn.close()


def test_cleanup_removes_old_observations_too(tmp_path):
    """Retention has to cover every time-series table, not just detections."""
    from backend.database import insert_observation
    conn = init_db(tmp_path / "test.db")
    insert_observation(conn, "coffeecart", "bicycle", 1, datetime.now() - timedelta(days=60))
    insert_observation(conn, "coffeecart", "bicycle", 2, datetime.now() - timedelta(days=5))
    cleanup_old_data(conn, retention_days=30)
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 1
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
