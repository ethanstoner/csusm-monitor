import csv
import io
import logging
import re
import sqlite3
import subprocess
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
        """Open a thread-local SQLite connection with retry for locked DB."""
        conn = None
        for attempt in range(3):
            try:
                conn = sqlite3.connect(str(cfg.DB_PATH), timeout=5, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                return conn
            except sqlite3.OperationalError:
                # The PRAGMA can fail after connect() succeeded — don't leak the handle.
                if conn is not None:
                    conn.close()
                    conn = None
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))

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

    # Regex patterns tried in order — first match wins
    _PATTERNS = [
        re.compile(r"<b>\s*(\d+)\s*/\s*(\d+)"),                    # <b>733/1240
        re.compile(r"(\d+)\s*/\s*(\d+)\s*<small[^>]*>\s*Spaces"),  # N/M <small>Spaces
        re.compile(r"(\d+)\s*/\s*(\d+)\s*Spaces\s+available"),     # N/M Spaces available
        re.compile(r"aria-valuenow=[\"']?(\d+).*?aria-valuemax=[\"']?(\d+)"),  # progress bar
    ]

    def collect(self):
        resp = httpx.get("https://parkingstatus.csusm.edu", timeout=10)
        resp.raise_for_status()
        html = resp.text

        available = total = None
        for pat in self._PATTERNS:
            m = pat.search(html)
            if m:
                available, total = int(m.group(1)), int(m.group(2))
                break
        if available is None:
            logger.warning("Parking: could not parse HTML — no pattern matched")
            return

        lot_match = re.search(r"Lot\s+(\w+)", html)
        lot_id = lot_match.group(1) if lot_match else "unknown"

        conn = getattr(self, "_conn", self._main_db)
        insert_parking(conn, lot_id=lot_id, available=available, total=total)
        with self._lock:
            self.latest = {"lot_id": lot_id, "available": available, "total": total}
        logger.info("Parking Lot %s: %d/%d available", lot_id, available, total)


class AirQualityCollector(BaseCollector):
    NAME = "aqi-collector"
    INTERVAL = cfg.AQI_INTERVAL

    # US AQI breakpoints for PM2.5 (µg/m³) — EPA standard
    _PM25_BP = [
        (0.0, 12.0, 0, 50, "Good"),
        (12.1, 35.4, 51, 100, "Moderate"),
        (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups"),
        (55.5, 150.4, 151, 200, "Unhealthy"),
        (150.5, 250.4, 201, 300, "Very Unhealthy"),
        (250.5, 500.4, 301, 500, "Hazardous"),
    ]

    @classmethod
    def _pm25_to_aqi(cls, pm25: float) -> tuple[int, str]:
        """Convert PM2.5 concentration to US AQI and category."""
        for c_lo, c_hi, i_lo, i_hi, cat in cls._PM25_BP:
            if pm25 <= c_hi:
                aqi = round((i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo)
                return aqi, cat
        return 500, "Hazardous"

    def collect(self):
        url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={cfg.CAMPUS_LAT}&longitude={cfg.CAMPUS_LON}"
            f"&current=pm2_5,pm10"
        )
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        current = resp.json().get("current", {})
        pm25 = current.get("pm2_5")
        if pm25 is None:
            return
        aqi, category = self._pm25_to_aqi(pm25)
        pollutant = "PM2.5"

        conn = getattr(self, "_conn", self._main_db)
        insert_air_quality(conn, aqi=aqi, category=category, pollutant=pollutant)
        with self._lock:
            self.latest = {"aqi": aqi, "category": category, "pollutant": pollutant}
        logger.info("AQI: %d (%s) — %s (PM2.5=%.1f)", aqi, category, pollutant, pm25)


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

        # Use curl for download — NCTD server rejects httpx/python TLS connections
        result = subprocess.run(
            ["curl", "-sL", "-A", "Mozilla/5.0", "-o", str(zip_path),
             "https://lfportal.nctd.org/staticGTFS/google_transit.zip"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0 or not zip_path.exists() or zip_path.stat().st_size < 1000:
            logger.warning("Transit: GTFS download failed (curl rc=%d)", result.returncode)
            return

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

    # CSUSM academic calendar dates (static, updated per semester)
    # Source: https://www.csusm.edu/academic_programs/calendars.html
    ACADEMIC_EVENTS = [
        {"title": "Last Day of Classes", "event_date": "2026-05-15", "location": "Campus", "description": "Spring 2026"},
        {"title": "Finals Week", "event_date": "2026-05-18", "location": "Campus", "description": "Spring 2026 final exams May 18-22"},
        {"title": "Commencement", "event_date": "2026-05-22", "location": "Campus", "description": "Spring 2026 commencement ceremony"},
        {"title": "Summer Session Begins", "event_date": "2026-06-01", "location": "Campus", "description": "Summer 2026"},
        {"title": "Independence Day (No Classes)", "event_date": "2026-07-03", "location": "Campus", "description": "Campus closed"},
        {"title": "Fall Semester Begins", "event_date": "2026-08-24", "location": "Campus", "description": "Fall 2026"},
        {"title": "Labor Day (No Classes)", "event_date": "2026-09-07", "location": "Campus", "description": "Campus closed"},
        {"title": "Veterans Day (No Classes)", "event_date": "2026-11-11", "location": "Campus", "description": "Campus closed"},
        {"title": "Thanksgiving Break", "event_date": "2026-11-23", "location": "Campus", "description": "Nov 23-27, no classes"},
        {"title": "Last Day of Classes (Fall)", "event_date": "2026-12-11", "location": "Campus", "description": "Fall 2026"},
    ]

    def collect(self):
        # Try scraping dynamic events first
        events = []
        try:
            resp = httpx.get("https://m.csusm.edu/default/events/index", timeout=15)
            resp.raise_for_status()
            events = self._parse_events(resp.text)
        except Exception:
            logger.info("Events: could not fetch Kurogo page, using academic calendar")

        # Fall back to academic calendar if no dynamic events found
        if not events:
            events = self.ACADEMIC_EVENTS

        conn = getattr(self, "_conn", self._main_db)
        for ev in events:
            insert_event(conn, **ev)
        with self._lock:
            self.latest = {"events": events, "count": len(events)}
        logger.info("Events: stored %d events", len(events))

    @staticmethod
    def _normalize_event_date(date_str, time_str=""):
        """Convert a Kurogo 'May 7' / '5:30 PM' pair into 'YYYY-MM-DD HH:MM'.

        Kurogo omits the year. Listings are forward-looking, so a month/day that
        already passed is assumed to belong to next year. Returns None if the
        date cannot be parsed — ordering in get_upcoming_events() is a string
        comparison, so an unparseable date must never reach the database.
        """
        now = datetime.now(TZ)
        date_str = date_str.strip()

        # Already ISO (the academic-calendar fallback) — leave it alone.
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass

        day = None
        for fmt in ("%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y"):
            try:
                day = datetime.strptime(date_str, fmt).date()
                break
            except ValueError:
                continue

        if day is None:
            # "May 4, 11:00 AM" — the trailing field is a time, not a year.
            head, sep, tail = date_str.partition(",")
            if sep and not time_str:
                time_str = tail
            for fmt in ("%b %d", "%B %d"):
                try:
                    parsed = datetime.strptime(head.strip(), fmt)
                except ValueError:
                    continue
                day = parsed.replace(year=now.year).date()
                # Listings only run forward; a date well in the past is next year's.
                if (now.date() - day).days > 60:
                    day = day.replace(year=now.year + 1)
                break

        if day is None:
            return None

        hhmm = "00:00"
        for fmt in ("%I:%M %p", "%I %p", "%H:%M"):
            try:
                hhmm = datetime.strptime(time_str.strip(), fmt).strftime("%H:%M")
                break
            except ValueError:
                continue
        return f"{day.isoformat()} {hhmm}"

    def _parse_events(self, html):
        events = []

        # Strategy 1: Kurogo kgo-list-item — sequential child <div>s:
        #   div[0]=date, div[1]=time, div[2]=title, div[3]=org/desc
        items = re.findall(
            r'class="kgo-list-item"[^>]*>(.*?)</div>\s*</a>',
            html, re.DOTALL,
        )
        for item_html in items:
            divs = re.findall(r'<div[^>]*>([^<]+)</div>', item_html)
            if len(divs) >= 3:
                date_str = divs[0].strip()
                time_str = divs[1].strip() if len(divs) > 1 else ""
                title = divs[2].strip()
                desc = divs[3].strip() if len(divs) > 3 else None
                event_date = self._normalize_event_date(date_str, time_str)
                if event_date is None:
                    logger.debug("Events: unparseable date %r for %r", date_str, title)
                    continue
                events.append({
                    "title": title,
                    "event_date": event_date,
                    "location": "CSUSM",
                    "description": desc,
                })

        # Strategy 2: class="event-title" / event-date pattern (generic)
        if not events:
            title_matches = re.findall(r'class="event-title"[^>]*>([^<]+)', html)
            date_matches = re.findall(r'class="event-date"[^>]*>([^<]+)', html)
            loc_matches = re.findall(r'class="event-location"[^>]*>([^<]+)', html)
            desc_matches = re.findall(r'class="event-description"[^>]*>([^<]+)', html)
            for i, title in enumerate(title_matches):
                raw_date = date_matches[i].strip() if i < len(date_matches) else ""
                event_date = self._normalize_event_date(raw_date) if raw_date else None
                if event_date is None:
                    logger.debug("Events: unparseable date %r for %r", raw_date, title)
                    continue
                events.append({
                    "title": title.strip(),
                    "event_date": event_date,
                    "location": loc_matches[i].strip() if i < len(loc_matches) else None,
                    "description": desc_matches[i].strip() if i < len(desc_matches) else None,
                })

        return events[:20]
