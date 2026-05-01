import csv
import io
import logging
import re
import sqlite3
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

import backend.config as cfg
from backend.database import (
    insert_air_quality, insert_event, insert_parking, insert_weather,
)

logger = logging.getLogger(__name__)
TZ = ZoneInfo(cfg.TIMEZONE)


class BaseCollector(threading.Thread):
    """Base class for data collection workers.
    Subclasses set NAME and INTERVAL as class attributes, accept only db_conn."""
    NAME = "collector"
    INTERVAL = 300

    def __init__(self, db_conn):
        super().__init__(daemon=True, name=self.NAME)
        self.interval = self.INTERVAL
        self._main_db = db_conn
        self._running = True
        self.latest = {}
        self._lock = threading.Lock()

    def _open_conn(self):
        """Open a thread-local SQLite connection."""
        conn = sqlite3.connect(str(cfg.DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def run(self):
        self._conn = self._open_conn()
        while self._running:
            try:
                self.collect()
            except Exception:
                logger.exception(f"{self.name} collection failed")
            time.sleep(self.interval)
        self._conn.close()

    def collect(self):
        raise NotImplementedError

    def stop(self):
        self._running = False
        self.join(timeout=5)


class WeatherCollector(BaseCollector):
    NAME = "weather-collector"
    INTERVAL = cfg.WEATHER_INTERVAL

    def collect(self):
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={cfg.CAMPUS_LAT}&longitude={cfg.CAMPUS_LON}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
            f"wind_direction_10m,weather_code,apparent_temperature,uv_index"
            f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
            f"&timezone=America/Los_Angeles"
        )
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()["current"]
        weather = {
            "temperature": data["temperature_2m"],
            "apparent_temperature": data["apparent_temperature"],
            "humidity": data["relative_humidity_2m"],
            "wind_speed": data["wind_speed_10m"],
            "wind_direction": data["wind_direction_10m"],
            "weather_code": data["weather_code"],
            "uv_index": data["uv_index"],
        }
        conn = getattr(self, "_conn", self._main_db)
        insert_weather(conn, **weather)
        with self._lock:
            self.latest = weather
        logger.info("Weather: %.0f°F, humidity %d%%", weather["temperature"], weather["humidity"])


class ParkingCollector(BaseCollector):
    NAME = "parking-collector"
    INTERVAL = cfg.PARKING_INTERVAL

    def collect(self):
        resp = httpx.get("https://parkingstatus.csusm.edu", timeout=10)
        resp.raise_for_status()
        match = re.search(r"(\d+)\s*/\s*(\d+)\s*Spaces available", resp.text)
        if not match:
            logger.warning("Parking: could not parse HTML")
            return
        available = int(match.group(1))
        total = int(match.group(2))
        lot_match = re.search(r"Lot\s+(\w+)", resp.text)
        lot_id = lot_match.group(1) if lot_match else "unknown"

        conn = getattr(self, "_conn", self._main_db)
        insert_parking(conn, lot_id=lot_id, available=available, total=total)
        with self._lock:
            self.latest = {"lot_id": lot_id, "available": available, "total": total}
        logger.info("Parking Lot %s: %d/%d available", lot_id, available, total)


class AirQualityCollector(BaseCollector):
    NAME = "aqi-collector"
    INTERVAL = cfg.AQI_INTERVAL

    def collect(self):
        if not cfg.AIRNOW_API_KEY:
            return
        url = "https://www.airnowapi.org/aq/observation/zipCode/current/"
        params = {
            "zipCode": cfg.CAMPUS_ZIP,
            "format": "application/json",
            "API_KEY": cfg.AIRNOW_API_KEY,
        }
        resp = httpx.get(url, params=params, timeout=10)
        if resp.status_code == 401:
            logger.warning("AirNow API key is invalid or expired (HTTP 401)")
            return
        resp.raise_for_status()
        readings = resp.json()
        if not readings:
            return
        dominant = max(readings, key=lambda r: r.get("AQI", 0))
        aqi = dominant["AQI"]
        category = dominant.get("Category", {}).get("Name", "Unknown")
        pollutant = dominant.get("ParameterName", "Unknown")

        conn = getattr(self, "_conn", self._main_db)
        insert_air_quality(conn, aqi=aqi, category=category, pollutant=pollutant)
        with self._lock:
            self.latest = {"aqi": aqi, "category": category, "pollutant": pollutant}
        logger.info("AQI: %d (%s) — %s", aqi, category, pollutant)


class TransitCollector(BaseCollector):
    NAME = "transit-collector"
    INTERVAL = cfg.TRANSIT_REFRESH_INTERVAL

    def __init__(self, db_conn):
        super().__init__(db_conn)
        self._schedule = []
        self._services = {}
        self._last_download = 0

    def collect(self):
        cfg.GTFS_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = cfg.GTFS_DIR / "google_transit.zip"

        resp = httpx.get("https://lfportal.nctd.org/staticGTFS/google_transit.zip", timeout=30)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

        self._parse_gtfs(zip_path)
        self._last_download = time.time()
        logger.info("Transit: loaded %d departures from CSUSM station", len(self._schedule))

    def _parse_gtfs(self, zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            stops = list(csv.DictReader(io.TextIOWrapper(zf.open("stops.txt"))))
            csusm_ids = {s["stop_id"] for s in stops if "cal state" in s["stop_name"].lower() or "csusm" in s["stop_name"].lower()}
            if not csusm_ids:
                logger.warning("Transit: CSUSM station not found in GTFS stops")
                return

            self._services = {}
            cal = list(csv.DictReader(io.TextIOWrapper(zf.open("calendar.txt"))))
            for row in cal:
                self._services[row["service_id"]] = {
                    "days": [int(row[d]) for d in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]],
                    "start": row["start_date"],
                    "end": row["end_date"],
                }

            trips = list(csv.DictReader(io.TextIOWrapper(zf.open("trips.txt"))))
            trip_info = {t["trip_id"]: t for t in trips}

            stop_times = list(csv.DictReader(io.TextIOWrapper(zf.open("stop_times.txt"))))
            self._schedule = []
            for st in stop_times:
                if st["stop_id"] in csusm_ids and st["trip_id"] in trip_info:
                    trip = trip_info[st["trip_id"]]
                    self._schedule.append({
                        "trip_id": st["trip_id"],
                        "route": trip.get("route_id", ""),
                        "direction": trip.get("trip_headsign", ""),
                        "stop_time": st["departure_time"],
                        "service_id": trip.get("service_id", ""),
                    })
            self._schedule.sort(key=lambda x: x["stop_time"])

    def get_next_departures(self, n=6, current_time=None, current_weekday=None):
        now = datetime.now(TZ)
        if current_time is None:
            current_time = now.strftime("%H:%M:%S")
        if current_weekday is None:
            current_weekday = now.weekday()

        results = []
        for dep in self._schedule:
            svc = self._services.get(dep["service_id"])
            if svc and not svc["days"][current_weekday]:
                continue
            if dep["stop_time"] >= current_time:
                dep_parts = dep["stop_time"].split(":")
                cur_parts = current_time.split(":")
                dep_mins = int(dep_parts[0]) * 60 + int(dep_parts[1])
                cur_mins = int(cur_parts[0]) * 60 + int(cur_parts[1])
                results.append({
                    "route": dep["route"],
                    "direction": dep["direction"],
                    "time": dep["stop_time"],
                    "minutes_away": dep_mins - cur_mins,
                })
            if len(results) >= n:
                break
        return results


class EventsCollector(BaseCollector):
    NAME = "events-collector"
    INTERVAL = cfg.EVENTS_INTERVAL

    def collect(self):
        resp = httpx.get("https://m.csusm.edu/default/events/index", timeout=15)
        resp.raise_for_status()
        events = self._parse_events(resp.text)
        if not events:
            logger.info("Events: no events found on Kurogo page")
            return

        conn = getattr(self, "_conn", self._main_db)
        for ev in events:
            insert_event(conn, **ev)
        with self._lock:
            self.latest = {"events": events, "count": len(events)}
        logger.info("Events: stored %d events", len(events))

    def _parse_events(self, html):
        events = []
        title_matches = re.findall(r'class="event-title"[^>]*>([^<]+)', html)
        date_matches = re.findall(r'class="event-date"[^>]*>([^<]+)', html)
        loc_matches = re.findall(r'class="event-location"[^>]*>([^<]+)', html)
        desc_matches = re.findall(r'class="event-description"[^>]*>([^<]+)', html)

        for i, title in enumerate(title_matches):
            events.append({
                "title": title.strip(),
                "event_date": date_matches[i].strip() if i < len(date_matches) else None,
                "location": loc_matches[i].strip() if i < len(loc_matches) else None,
                "description": desc_matches[i].strip() if i < len(desc_matches) else None,
            })

        if not events:
            for match in re.finditer(r'(\w+ \d+,?\s*\d{1,2}:\d{2}\s*[AP]M)\s*[-–]?\s*(.+?)(?:<|$)', html):
                events.append({
                    "title": match.group(2).strip(),
                    "event_date": match.group(1).strip(),
                    "location": None,
                    "description": None,
                })

        return events[:20]
