import sqlite3
import io
import zipfile
import pytest
from datetime import datetime
from pathlib import Path
from fastapi.testclient import TestClient


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Create a fresh DB with all tables."""
    import backend.config as cfg
    monkeypatch.setattr(cfg, "DB_PATH", tmp_path / "test.db")
    from backend.database import init_db
    conn = init_db(tmp_path / "test.db")
    yield conn
    conn.close()


def test_weather_table_exists(db):
    rows = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='weather'").fetchall()
    assert len(rows) == 1


def test_parking_table_exists(db):
    rows = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parking'").fetchall()
    assert len(rows) == 1


def test_air_quality_table_exists(db):
    rows = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='air_quality'").fetchall()
    assert len(rows) == 1


def test_events_table_exists(db):
    rows = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'").fetchall()
    assert len(rows) == 1


def test_insert_weather(db):
    from backend.database import insert_weather
    insert_weather(db, temperature=72.5, apparent_temperature=70.0, humidity=45.0,
                   wind_speed=8.5, wind_direction=180.0, weather_code=1, uv_index=5.0)
    rows = db.execute("SELECT temperature, humidity FROM weather").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 72.5


def test_insert_parking(db):
    from backend.database import insert_parking
    insert_parking(db, lot_id="F", available=766, total=1240)
    rows = db.execute("SELECT lot_id, available, total, day_of_week, hour FROM parking").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "F"
    assert rows[0][1] == 766
    assert rows[0][3] is not None  # day_of_week computed


def test_insert_air_quality(db):
    from backend.database import insert_air_quality
    insert_air_quality(db, aqi=42, category="Good", pollutant="PM2.5")
    rows = db.execute("SELECT aqi, category FROM air_quality").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 42


def test_insert_event_dedup(db):
    from backend.database import insert_event
    insert_event(db, title="ASI Market", event_date="2026-05-04", location="USU", description="Free food")
    insert_event(db, title="ASI Market", event_date="2026-05-04", location="USU", description="Free food")
    rows = db.execute("SELECT COUNT(*) FROM events").fetchone()
    assert rows[0] == 1  # deduped


def test_get_parking_trends(db):
    from backend.database import get_parking_trends
    for hour in range(8, 12):
        db.execute(
            "INSERT INTO parking (lot_id, available, total, timestamp, day_of_week, hour) VALUES (?, ?, ?, ?, ?, ?)",
            ("F", 1000 - hour * 50, 1240, f"2026-05-01 {hour:02d}:00:00", 3, hour),
        )
    db.commit()
    data = get_parking_trends(db, "F", days=7)
    assert len(data) > 0
    assert "avg_available" in data[0]
    assert "avg_percent_full" in data[0]


def test_weather_collector_parse(db, monkeypatch):
    """Test WeatherCollector parses Open-Meteo JSON and stores to DB."""
    import httpx
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "current": {
            "temperature_2m": 72.5,
            "apparent_temperature": 70.0,
            "relative_humidity_2m": 45,
            "wind_speed_10m": 8.5,
            "wind_direction_10m": 180,
            "weather_code": 1,
            "uv_index": 5.0,
        }
    }
    mock_response.raise_for_status = MagicMock()

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_response)

    from backend.collectors import WeatherCollector
    collector = WeatherCollector(db)
    collector.collect()

    rows = db.execute("SELECT temperature, humidity FROM weather").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 72.5
    assert rows[0][1] == 45

    with collector._lock:
        assert collector.latest["temperature"] == 72.5


def test_parking_collector_parse(db, monkeypatch):
    import httpx
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = """
    <html><body>
    <h1>Parking List</h1>
    <p>5/1/2026 10:09 AM</p>
    <div>Lot F</div>
    <div>766/1240 Spaces available</div>
    </body></html>
    """
    mock_response.raise_for_status = MagicMock()
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_response)

    from backend.collectors import ParkingCollector
    collector = ParkingCollector(db)
    collector.collect()

    rows = db.execute("SELECT lot_id, available, total FROM parking").fetchall()
    assert len(rows) == 1
    assert rows[0] == ("F", 766, 1240)


def test_aqi_collector_no_key(db, monkeypatch):
    import backend.config as cfg
    monkeypatch.setattr(cfg, "AIRNOW_API_KEY", "")

    from backend.collectors import AirQualityCollector
    collector = AirQualityCollector(db)
    collector.collect()

    rows = db.execute("SELECT COUNT(*) FROM air_quality").fetchone()
    assert rows[0] == 0


def test_aqi_collector_with_key(db, monkeypatch):
    import httpx
    import backend.config as cfg
    from unittest.mock import MagicMock

    monkeypatch.setattr(cfg, "AIRNOW_API_KEY", "test-key-123")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"AQI": 42, "Category": {"Name": "Good"}, "ParameterName": "PM2.5"},
        {"AQI": 35, "Category": {"Name": "Good"}, "ParameterName": "O3"},
    ]
    mock_response.raise_for_status = MagicMock()
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_response)

    from backend.collectors import AirQualityCollector
    collector = AirQualityCollector(db)
    collector.collect()

    rows = db.execute("SELECT aqi, pollutant FROM air_quality").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 42

    with collector._lock:
        assert collector.latest["aqi"] == 42


def _make_gtfs_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("stops.txt", "stop_id,stop_name,stop_lat,stop_lon\n"
                     "100,Cal State San Marcos,33.1284,-117.1597\n"
                     "200,Other Stop,33.0,-117.0\n")
        zf.writestr("routes.txt", "route_id,route_short_name,route_long_name,route_type\n"
                     "SPRINTER,SPRINTER,Sprinter,0\n")
        zf.writestr("trips.txt", "route_id,service_id,trip_id,direction_id,trip_headsign\n"
                     "SPRINTER,weekday,T1,0,Escondido\n"
                     "SPRINTER,weekday,T2,1,Oceanside\n")
        zf.writestr("stop_times.txt", "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                     "T1,08:30:00,08:30:00,100,5\n"
                     "T1,09:00:00,09:00:00,200,6\n"
                     "T2,09:15:00,09:15:00,100,3\n"
                     "T2,09:45:00,09:45:00,200,4\n")
        zf.writestr("calendar.txt", "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
                     "weekday,1,1,1,1,1,0,0,20260101,20261231\n")
    buf.seek(0)
    return buf.read()


def test_transit_collector_parse(db, tmp_path, monkeypatch):
    import httpx
    import backend.config as cfg
    from unittest.mock import MagicMock

    monkeypatch.setattr(cfg, "GTFS_DIR", tmp_path / "gtfs")

    gtfs_bytes = _make_gtfs_zip()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = gtfs_bytes
    mock_response.raise_for_status = MagicMock()
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_response)

    from backend.collectors import TransitCollector
    collector = TransitCollector(db)
    collector.collect()

    departures = collector.get_next_departures(current_time="08:00:00", current_weekday=0)
    assert len(departures) == 2
    assert departures[0]["time"] == "08:30:00"
    assert departures[0]["direction"] == "Escondido"
    assert departures[0]["minutes_away"] == 30


def test_events_collector_parse(db, monkeypatch):
    import httpx
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = """
    <html><body>
    <div class="event-item">
        <span class="event-date">May 4, 11:00 AM</span>
        <span class="event-title">ASI Fresh Market Mondays</span>
        <span class="event-location">USU Quad</span>
        <span class="event-description">Free fruits and veggies</span>
    </div>
    <div class="event-item">
        <span class="event-date">May 5, 12:00 PM</span>
        <span class="event-title">Chow on Deck</span>
        <span class="event-location">Forum Plaza</span>
        <span class="event-description">End of semester celebration</span>
    </div>
    </body></html>
    """
    mock_response.raise_for_status = MagicMock()
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_response)

    from backend.collectors import EventsCollector
    collector = EventsCollector(db)
    collector.collect()

    rows = db.execute("SELECT title, event_date, location FROM events").fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "ASI Fresh Market Mondays"


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """Test client with collector tables initialized."""
    import backend.config as cfg
    monkeypatch.setattr(cfg, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("backend.main.START_WORKERS", False)
    from backend.main import app, get_db
    with TestClient(app) as c:
        conn = get_db()
        from backend.database import insert_weather, insert_parking, insert_air_quality, insert_event
        insert_weather(conn, temperature=72.5, apparent_temperature=70.0, humidity=45.0,
                       wind_speed=8.5, wind_direction=180.0, weather_code=1, uv_index=5.0)
        insert_parking(conn, lot_id="F", available=766, total=1240)
        insert_air_quality(conn, aqi=42, category="Good", pollutant="PM2.5")
        insert_event(conn, title="Test Event", event_date="2026-12-01",
                     location="USU", description="A test event")
        yield c


def test_get_conditions(api_client):
    resp = api_client.get("/api/conditions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["weather"]["temperature"] == 72.5
    assert data["aqi"]["aqi"] == 42


def test_get_parking_api(api_client):
    resp = api_client.get("/api/parking")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["lots"]) == 1
    assert data["lots"][0]["available"] == 766


def test_get_parking_trends_api(api_client):
    resp = api_client.get("/api/parking/trends?lot=F&days=7")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lot"] == "F"


def test_get_transit_api(api_client):
    resp = api_client.get("/api/transit")
    assert resp.status_code == 200
    data = resp.json()
    assert "station" in data
    assert "departures" in data


def test_get_events_api(api_client):
    resp = api_client.get("/api/events")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["events"]) == 1
    assert data["events"][0]["title"] == "Test Event"
