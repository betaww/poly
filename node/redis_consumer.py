"""
VPS Redis Consumer — subscribes to Alienware predictions and feeds them to strategies.

Bridges the brain node's Redis predictions into the local strategy engine.
Falls back to using market prices if Redis/brain is unavailable.
"""
import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass

import redis

from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A price prediction from the brain node."""
    asset: str
    cex_price: float
    confidence: float
    volatility: float
    sources: int
    timestamp: float
    age_ms: float = 0.0

    @property
    def is_stale(self) -> bool:
        return self.age_ms > 5000  # stale after 5 seconds


class RedisConsumer:
    """
    Subscribes to brain predictions via Redis pub/sub.
    Runs in a background thread to not block the asyncio event loop.
    Also listens for control commands from the Mac dashboard.
    """

    def __init__(self, config: Config):
        self.config = config
        self._redis: redis.Redis | None = None
        self._pubsub: redis.client.PubSub | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        # Latest predictions per asset
        self.predictions: dict[str, Prediction] = {}
        self._lock = threading.Lock()

        # Control command callback
        self._control_callback = None

    def start(self, control_callback=None):
        """Start background Redis subscriber thread."""
        self._control_callback = control_callback
        rc = self.config.redis

        try:
            kwargs = {
                "host": rc.host,
                "port": rc.port,
                "db": rc.db,
                "decode_responses": True,
            }
            if rc.password:
                kwargs["password"] = rc.password

            self._redis = redis.Redis(**kwargs)
            self._redis.ping()
            logger.info(f"Redis consumer connected: {rc.host}:{rc.port}")

            self._pubsub = self._redis.pubsub()
            self._pubsub.subscribe(
                rc.ch_prediction,
                rc.ch_control,
            )

            self._running = True
            self._thread = threading.Thread(target=self._listen, daemon=True)
            self._thread.start()
            logger.info("Redis consumer thread started")

        except redis.ConnectionError as e:
            logger.warning(f"Redis not available ({e}). Running in standalone mode.")
            self._redis = None

    def _listen(self):
        """Background thread: listen for Redis messages with auto-reconnect."""
        while self._running:
            try:
                if self._pubsub is None:
                    self._reconnect()
                msg = self._pubsub.get_message(timeout=0.1)
                if msg and msg["type"] == "message":
                    channel = msg["channel"]
                    data = json.loads(msg["data"])

                    if channel == self.config.redis.ch_prediction:
                        self._handle_prediction(data)
                    elif channel == self.config.redis.ch_control:
                        self._handle_control(data)

            except json.JSONDecodeError:
                pass
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(f"Redis connection lost: {e}. Reconnecting in 3s...")
                self._pubsub = None
                time.sleep(3)
            except Exception as e:
                logger.error(f"Redis consumer error: {e}")
                time.sleep(1)

    def _reconnect(self):
        """Rebuild Redis pubsub connection."""
        try:
            # Clean up old connection to prevent resource leak
            if self._redis:
                try:
                    self._redis.close()
                except Exception:
                    pass
            rc = self.config.redis
            kwargs = {
                "host": rc.host, "port": rc.port, "db": rc.db,
                "decode_responses": True,
            }
            if rc.password:
                kwargs["password"] = rc.password
            self._redis = redis.Redis(**kwargs)
            self._redis.ping()
            self._pubsub = self._redis.pubsub()
            self._pubsub.subscribe(rc.ch_prediction, rc.ch_control)
            logger.info("Redis consumer reconnected")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis reconnect failed: {e}. Retrying in 5s...")
            self._pubsub = None
            time.sleep(5)

    def _handle_prediction(self, data: dict):
        """Process a prediction message from the brain."""
        pred = Prediction(
            asset=data.get("asset", ""),
            cex_price=data.get("cex_price", 0),
            confidence=data.get("confidence", 0),
            volatility=data.get("volatility", 0.01),
            sources=data.get("sources", 0),
            timestamp=data.get("timestamp", 0),
            age_ms=(time.time() - data.get("timestamp", time.time())) * 1000,
        )

        with self._lock:
            self.predictions[pred.asset] = pred

    def _handle_control(self, data: dict):
        """Process a control command from the dashboard."""
        command = data.get("command", "")
        logger.info(f"Control command received: {command}")
        if self._control_callback:
            self._control_callback(command, data)

    def get_prediction(self, asset: str) -> Prediction | None:
        """Get latest prediction for an asset (thread-safe)."""
        with self._lock:
            pred = self.predictions.get(asset)
            if pred:
                pred.age_ms = (time.time() - pred.timestamp) * 1000
                if pred.is_stale:
                    return None
            return pred

    def publish_telemetry(self, data: dict):
        """Publish telemetry to Mac dashboard."""
        if self._redis:
            try:
                self._redis.publish(
                    self.config.redis.ch_telemetry,
                    json.dumps(data),
                )
            except Exception as e:
                logger.debug(f"Telemetry publish failed: {e}")

    def stop(self):
        """Stop the consumer thread."""
        self._running = False
        if self._pubsub:
            self._pubsub.unsubscribe()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Redis consumer stopped")

    @property
    def is_connected(self) -> bool:
        return self._redis is not None and self._running
