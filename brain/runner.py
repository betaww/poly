"""
Alienware Brain Runner — receives price feeds from VPS via Redis.

Architecture:
  VPS: feed_forwarder.py → Redis pub/sub (polymarket:feeds)
  Brain: Redis sub → SyntheticOracle → Redis pub (predictions)

The Brain no longer connects to exchanges directly — the VPS handles all
exchange WebSocket connections and forwards price ticks via Redis.
This avoids network restriction issues on the Alienware side.

Usage:
  python -m user_data.polymarket.brain.runner

Env vars:
  POLY_REDIS_HOST — Redis host (default: localhost)
  POLY_REDIS_PORT — Redis port (default: 6379)
  POLY_REDIS_PASSWORD — Redis password
"""
import asyncio
import json
import logging
import time
import threading

import redis

from ..config import Config
from .price_feeds import SyntheticOracle, PriceTick

logger = logging.getLogger(__name__)


class BrainRunner:
    """
    The Brain: receives CEX prices from VPS via Redis,
    predicts Chainlink settlement, publishes predictions back to Redis.
    """

    def __init__(self, config: Config):
        self.config = config
        self._redis: redis.Redis | None = None
        self._publish_interval = 0.1  # 100ms — publish predictions at 10 Hz
        self._last_publish = 0.0
        self._running = False
        self._tick_count = 0

        # Separate oracle per asset
        self._oracles: dict[str, SyntheticOracle] = {
            "btc": SyntheticOracle(config),
        }
        if "eth" in config.market.assets:
            self._oracles["eth"] = SyntheticOracle(config)

    def _connect_redis(self):
        """Connect to Redis."""
        rc = self.config.redis
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
        logger.info(f"Redis connected: {rc.host}:{rc.port}")

    def _on_tick(self, tick: PriceTick):
        """Handle price tick from Redis subscription."""
        asset = tick.asset
        if asset in self._oracles:
            self._oracles[asset].update(tick)

        self._tick_count += 1
        # Log every 100 ticks
        if self._tick_count % 100 == 0:
            sources = {a: len(o.prices) for a, o in self._oracles.items()}
            logger.info(f"Ticks received: {self._tick_count} | Sources: {sources}")

    def _publish_predictions(self):
        """Publish current predictions to Redis."""
        if not self._redis:
            return

        for asset, oracle in self._oracles.items():
            predicted_price, confidence = oracle.predict()
            if predicted_price == 0:
                continue

            volatility = oracle.get_volatility()

            msg = json.dumps({
                "asset": asset,
                "cex_price": predicted_price,
                "confidence": round(confidence, 4),
                "volatility": round(volatility, 6),
                "sources": len(oracle.prices),
                "timestamp": time.time(),
            })

            try:
                self._redis.publish(self.config.redis.ch_prediction, msg)
            except Exception as e:
                logger.error(f"Redis publish failed: {e}")

    def _subscribe_feeds(self):
        """Subscribe to VPS price feeds via Redis pub/sub (runs in a thread)."""
        rc = self.config.redis
        kwargs = {
            "host": rc.host, "port": rc.port, "db": rc.db,
            "decode_responses": True,
        }
        if rc.password:
            kwargs["password"] = rc.password

        sub_redis = redis.Redis(**kwargs)
        pubsub = sub_redis.pubsub()
        pubsub.subscribe("polymarket:feeds")
        logger.info("Subscribed to polymarket:feeds")

        for message in pubsub.listen():
            if not self._running:
                break
            if message["type"] != "message":
                continue
            try:
                d = json.loads(message["data"])
                tick = PriceTick(
                    exchange=d["exchange"],
                    asset=d["asset"],
                    price=d["price"],
                    bid=d.get("bid", 0),
                    ask=d.get("ask", 0),
                    timestamp=d.get("ts", time.time()),
                )
                self._on_tick(tick)
            except Exception as e:
                logger.error(f"Feed parse error: {e}")

        pubsub.close()
        sub_redis.close()

    async def start(self):
        """Start the brain node."""
        self._running = True
        self._connect_redis()

        logger.info(f"""
╔═══════════════════════════════════════════════════╗
║  Polymarket Brain Node (Alienware RTX 5090)       ║
║  Mode: Redis subscriber (VPS feeds all exchanges) ║
║  Assets: {', '.join(a.upper() for a in self._oracles):<40s}║
║  Publish rate: {1/self._publish_interval:.0f} Hz{' '*31}║
║  Redis: {f"{self.config.redis.host}:{self.config.redis.port}":<41s}║
╚═══════════════════════════════════════════════════╝
        """)

        # Start Redis subscriber in a background thread
        feed_thread = threading.Thread(target=self._subscribe_feeds, daemon=True)
        feed_thread.start()
        logger.info("Feed subscriber thread started")

        # Main loop: publish predictions at configured rate
        try:
            while self._running:
                self._publish_predictions()
                await asyncio.sleep(self._publish_interval)
        except asyncio.CancelledError:
            logger.info("Brain shutting down...")
        finally:
            self._running = False

    def stop(self):
        self._running = False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Brain Node (Alienware)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)-25s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config.from_env()
    config.node = "brain"
    runner = BrainRunner(config)
    asyncio.run(runner.start())


if __name__ == "__main__":
    main()
