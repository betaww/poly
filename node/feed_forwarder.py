"""
VPS Feed Forwarder — runs exchange WebSocket feeds and publishes to Redis.

Runs on VPS (London) where all exchanges are accessible.
Alienware Brain reads these prices from Redis instead of connecting directly.

Architecture:
  VPS: Binance WS + OKX WS + Coinbase WS → Redis pub/sub
  Alienware: Redis sub → SyntheticOracle → Redis pub (predictions)
"""
import asyncio
import json
import logging
import os
import time

import redis
import websockets

logger = logging.getLogger(__name__)


class FeedForwarder:
    """Connects to exchange WebSockets and forwards prices to Redis."""

    def __init__(self, redis_host: str, redis_port: int, redis_password: str):
        self._redis = redis.Redis(
            host=redis_host, port=redis_port, password=redis_password,
            decode_responses=True,
        )
        self._running = False
        logger.info(f"FeedForwarder Redis: {redis_host}:{redis_port}")

    async def start(self):
        self._running = True
        tasks = [
            asyncio.create_task(self._binance_feed("btcusdt", "btc")),
            asyncio.create_task(self._binance_feed("ethusdt", "eth")),
            asyncio.create_task(self._okx_feed("BTC-USDT", "btc")),
            asyncio.create_task(self._okx_feed("ETH-USDT", "eth")),
            asyncio.create_task(self._coinbase_feed("BTC-USD", "btc")),
            asyncio.create_task(self._coinbase_feed("ETH-USD", "eth")),
        ]
        logger.info("FeedForwarder started: 6 feeds (Binance+OKX+Coinbase × BTC+ETH)")
        await asyncio.gather(*tasks, return_exceptions=True)

    def _publish_tick(self, exchange: str, asset: str, price: float, bid: float, ask: float):
        """Publish a price tick to Redis."""
        tick = {
            "exchange": exchange,
            "asset": asset,
            "price": price,
            "bid": bid,
            "ask": ask,
            "ts": time.time(),
        }
        self._redis.publish("polymarket:feeds", json.dumps(tick))

    # ─── Binance ──────────────────────────────────────────────

    async def _binance_feed(self, symbol: str, asset: str):
        url = f"wss://stream.binance.com:9443/ws/{symbol}@ticker"
        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info(f"[VPS] Binance connected: {symbol}")
                    async for msg in ws:
                        if not self._running:
                            break
                        d = json.loads(msg)
                        self._publish_tick(
                            "binance", asset,
                            float(d.get("c", 0)),
                            float(d.get("b", 0)),
                            float(d.get("a", 0)),
                        )
            except Exception as e:
                logger.error(f"[VPS] Binance {symbol}: {e}")
            if self._running:
                await asyncio.sleep(2)

    # ─── OKX ──────────────────────────────────────────────────

    async def _okx_feed(self, inst_id: str, asset: str):
        url = "wss://ws.okx.com:8443/ws/v5/public"
        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    sub = {"op": "subscribe", "args": [{"channel": "tickers", "instId": inst_id}]}
                    await ws.send(json.dumps(sub))
                    logger.info(f"[VPS] OKX connected: {inst_id}")
                    async for msg in ws:
                        if not self._running:
                            break
                        d = json.loads(msg)
                        if "data" not in d:
                            continue
                        for item in d["data"]:
                            self._publish_tick(
                                "okx", asset,
                                float(item.get("last", 0)),
                                float(item.get("bidPx", 0)),
                                float(item.get("askPx", 0)),
                            )
            except Exception as e:
                logger.error(f"[VPS] OKX {inst_id}: {e}")
            if self._running:
                await asyncio.sleep(2)

    # ─── Coinbase ─────────────────────────────────────────────

    async def _coinbase_feed(self, product_id: str, asset: str):
        url = "wss://ws-feed.exchange.coinbase.com"
        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    sub = {"type": "subscribe", "product_ids": [product_id], "channels": ["ticker"]}
                    await ws.send(json.dumps(sub))
                    logger.info(f"[VPS] Coinbase connected: {product_id}")
                    async for msg in ws:
                        if not self._running:
                            break
                        d = json.loads(msg)
                        if d.get("type") != "ticker":
                            continue
                        self._publish_tick(
                            "coinbase", asset,
                            float(d.get("price", 0)),
                            float(d.get("best_bid", 0)),
                            float(d.get("best_ask", 0)),
                        )
            except Exception as e:
                logger.error(f"[VPS] Coinbase {product_id}: {e}")
            if self._running:
                await asyncio.sleep(2)

    def stop(self):
        self._running = False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)-30s | %(message)s",
        datefmt="%H:%M:%S",
    )
    host = os.environ.get("POLY_REDIS_HOST", "localhost")
    port = int(os.environ.get("POLY_REDIS_PORT", "6379"))
    pw = os.environ.get("POLY_REDIS_PASSWORD", "")

    forwarder = FeedForwarder(host, port, pw)
    try:
        asyncio.run(forwarder.start())
    except KeyboardInterrupt:
        forwarder.stop()
        print("\nFeedForwarder stopped.")
