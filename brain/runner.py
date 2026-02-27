"""
Alienware Brain Runner — entry point for the GPU prediction node.

Connects to exchange WebSockets, runs the Synthetic Oracle,
and publishes predictions to Redis for VPS consumption.

Usage:
  python -m user_data.polymarket.brain.runner

Env vars:
  POLY_REDIS_HOST — Redis host (default: localhost / Tailscale IP)
  POLY_REDIS_PORT — Redis port (default: 6379)
"""
import asyncio
import json
import logging
import time

import redis

from ..config import Config
from .price_feeds import BinanceFeed, CoinbaseFeed, SyntheticOracle, PriceTick

logger = logging.getLogger(__name__)


class BrainRunner:
    """
    The Brain: receives CEX prices, predicts Chainlink settlement, publishes to Redis.
    """

    def __init__(self, config: Config):
        self.config = config
        self.oracle = SyntheticOracle(config)

        # Price feeds
        self._feeds = {
            "binance_btc": BinanceFeed("btcusdt"),
            "coinbase_btc": CoinbaseFeed("BTC-USD"),
        }
        # Add ETH feeds if configured
        if "eth" in config.market.assets:
            self._feeds["binance_eth"] = BinanceFeed("ethusdt")
            self._feeds["coinbase_eth"] = CoinbaseFeed("ETH-USD")

        # Redis publisher
        self._redis: redis.Redis | None = None
        self._publish_interval = 0.1  # 100ms — publish predictions at 10 Hz
        self._last_publish = 0.0
        self._running = False

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

    async def _on_tick(self, tick: PriceTick):
        """Handle price tick from any exchange."""
        # Route to the correct per-asset oracle using tick.asset
        asset = tick.asset

        # Update the appropriate oracle
        if asset in self._oracles:
            self._oracles[asset].update(tick)

        # Publish at configured interval
        now = time.time()
        if now - self._last_publish >= self._publish_interval:
            self._last_publish = now
            await self._publish_predictions()

    async def _publish_predictions(self):
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

    async def start(self):
        """Start the brain node."""
        self._running = True
        self._connect_redis()

        logger.info(f"""
╔═══════════════════════════════════════════════════╗
║  Polymarket Brain Node (Alienware RTX 5090)       ║
║  Assets: {', '.join(a.upper() for a in self._oracles):<40s}║
║  Feeds: {len(self._feeds):<41d}║
║  Publish rate: {1/self._publish_interval:.0f} Hz{' '*31}║
║  Redis: {f"{self.config.redis.host}:{self.config.redis.port}":<41s}║
╚═══════════════════════════════════════════════════╝
        """)

        # Start all feeds concurrently, passing asset tag
        tasks = []
        asset_map = {
            "binance_btc": "btc", "coinbase_btc": "btc",
            "binance_eth": "eth", "coinbase_eth": "eth",
        }
        for name, feed in self._feeds.items():
            asset = asset_map.get(name, "btc")
            task = asyncio.create_task(feed.connect(self._on_tick, asset=asset))
            tasks.append(task)
            logger.info(f"Feed started: {name} → {asset}")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Brain shutting down...")
        finally:
            for feed in self._feeds.values():
                feed.stop()

    def stop(self):
        self._running = False
        for feed in self._feeds.values():
            feed.stop()


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
