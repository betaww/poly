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
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_password = redis_password
        self._redis = self._connect_redis()
        self._running = False
        # Heartbeat tracking
        self._tick_counts: dict[str, int] = {}   # {"binance:btc": count}
        self._last_tick_ts: dict[str, float] = {}  # {"binance:btc": timestamp}
        self._last_heartbeat = time.time()
        logger.info(f"FeedForwarder Redis: {redis_host}:{redis_port}")

    def _connect_redis(self) -> redis.Redis:
        """Create Redis connection."""
        return redis.Redis(
            host=self._redis_host, port=self._redis_port,
            password=self._redis_password, decode_responses=True,
        )

    async def start(self):
        self._running = True
        tasks = [
            asyncio.create_task(self._binance_feed("btcusdt", "btc")),
            asyncio.create_task(self._binance_feed("ethusdt", "eth")),
            asyncio.create_task(self._okx_feed("BTC-USDT", "btc")),
            asyncio.create_task(self._okx_feed("ETH-USDT", "eth")),
            asyncio.create_task(self._coinbase_feed("BTC-USD", "btc")),
            asyncio.create_task(self._coinbase_feed("ETH-USD", "eth")),
            asyncio.create_task(self._heartbeat_loop()),
        ]
        logger.info("FeedForwarder started: 6 feeds (Binance+OKX+Coinbase × BTC+ETH)")
        await asyncio.gather(*tasks, return_exceptions=True)

    def _publish_tick(self, exchange: str, asset: str, price: float, bid: float, ask: float,
                       exchange_ts: float = 0.0):
        """Publish a price tick to Redis with auto-reconnect.

        P2 FIX: Use bid-ask midpoint as primary price when both are available.
        P1 FIX: Preserve exchange event timestamp for accurate staleness.
        """
        # P2: Prefer bid-ask midpoint over last trade
        if bid > 0 and ask > 0:
            midpoint = (bid + ask) / 2
        else:
            midpoint = price  # fallback to last trade

        tick = {
            "exchange": exchange,
            "asset": asset,
            "price": midpoint,  # P2: midpoint instead of last trade
            "bid": bid,
            "ask": ask,
            "ts": exchange_ts if exchange_ts > 0 else time.time(),  # P1: exchange timestamp
            "local_ts": time.time(),  # local receipt time for latency measurement
        }
        # Track for heartbeat
        key = f"{exchange}:{asset}"
        self._tick_counts[key] = self._tick_counts.get(key, 0) + 1
        self._last_tick_ts[key] = time.time()

        try:
            self._redis.publish("polymarket:feeds", json.dumps(tick))
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis publish failed: {e}. Reconnecting...")
            try:
                self._redis = self._connect_redis()
                self._redis.publish("polymarket:feeds", json.dumps(tick))
            except Exception:
                pass  # will retry on next tick

    async def _heartbeat_loop(self):
        """Log tick counts per exchange every 60s and detect stale feeds."""
        while self._running:
            await asyncio.sleep(60)
            now = time.time()
            parts = []
            stale = []
            for key, count in sorted(self._tick_counts.items()):
                parts.append(f"{key}={count}")
                last = self._last_tick_ts.get(key, 0)
                if now - last > 30:
                    stale.append(key)
            logger.info(f"Heartbeat [60s]: {' | '.join(parts)}")
            if stale:
                logger.warning(f"⚠️ STALE FEEDS (>30s no data): {', '.join(stale)}")
            self._tick_counts = {}  # reset for next interval

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
                        # P1: Use Binance event time (ms epoch)
                        exchange_ts = d.get("E", 0) / 1000.0 if d.get("E") else 0.0
                        self._publish_tick(
                            "binance", asset,
                            float(d.get("c", 0)),
                            float(d.get("b", 0)),
                            float(d.get("a", 0)),
                            exchange_ts=exchange_ts,
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
                            # P1: OKX ts is ms epoch
                            exchange_ts = int(item.get("ts", 0)) / 1000.0 if item.get("ts") else 0.0
                            self._publish_tick(
                                "okx", asset,
                                float(item.get("last", 0)),
                                float(item.get("bidPx", 0)),
                                float(item.get("askPx", 0)),
                                exchange_ts=exchange_ts,
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
                        # P1: Coinbase provides ISO time, convert to epoch
                        cb_time = d.get("time", "")
                        exchange_ts = 0.0
                        if cb_time:
                            try:
                                from datetime import datetime, timezone
                                dt = datetime.fromisoformat(cb_time.replace('Z', '+00:00'))
                                exchange_ts = dt.timestamp()
                            except (ValueError, TypeError):
                                pass
                        self._publish_tick(
                            "coinbase", asset,
                            float(d.get("price", 0)),
                            float(d.get("best_bid", 0)),
                            float(d.get("best_ask", 0)),
                            exchange_ts=exchange_ts,
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
