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
