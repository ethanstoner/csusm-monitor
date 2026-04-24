import json
import logging
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import paho.mqtt.client as mqtt

from backend.config import CAMERAS, MQTT_HOST, MQTT_PORT, TIMEZONE
from backend.database import insert_detection

logger = logging.getLogger(__name__)
TZ = ZoneInfo(TIMEZONE)


class FrigateListener:
    """Subscribes to Frigate MQTT topics and writes person counts to SQLite."""

    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.running = False
        self._thread: threading.Thread | None = None
        self.latest_counts: dict[str, int] = {}
        self._counts_lock = threading.Lock()
        self._reconnect_delay = 1

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("FrigateListener started — connecting to %s:%d", MQTT_HOST, MQTT_PORT)

    def stop(self):
        self.running = False
        self._client.disconnect()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("FrigateListener stopped")

    def _run(self):
        try:
            self._client.connect(MQTT_HOST, MQTT_PORT)
            self._client.loop_forever()
        except Exception:
            logger.exception("FrigateListener connection error")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self._reconnect_delay = 1
            client.subscribe("frigate/+/person")
            client.subscribe("frigate/events")
            logger.info("Connected to MQTT — subscribed to Frigate topics")
        else:
            logger.warning("MQTT connect failed: reason_code=%s", reason_code)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        if not self.running:
            return
        logger.warning("MQTT disconnected (reason=%s), retrying in %ds", reason_code, self._reconnect_delay)
        time.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, 30)
        try:
            client.reconnect()
        except Exception:
            logger.exception("MQTT reconnect failed")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode("utf-8", errors="replace")
        parts = topic.split("/")

        # Count topic: frigate/<camera>/person
        if len(parts) == 3 and parts[0] == "frigate" and parts[2] == "person":
            cam_id = parts[1]
            if cam_id not in CAMERAS:
                logger.warning("Count received for unknown camera '%s' — skipping", cam_id)
                return
            try:
                count = int(payload)
            except ValueError:
                logger.warning("Non-integer count payload on %s: %r", topic, payload)
                return
            with self._counts_lock:
                self.latest_counts[cam_id] = count
            insert_detection(self.db_conn, cam_id, count, datetime.now(TZ))
            logger.info("[%s] person count: %d", cam_id, count)

        # Events topic: log snapshot availability (detection log proxies to Frigate HTTP API)
        elif topic == "frigate/events":
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                return
            after = event.get("after", {})
            if after.get("label") == "person":
                logger.debug("Frigate event %s: camera=%s has_snapshot=%s",
                             after.get("id"), after.get("camera"), after.get("has_snapshot"))
