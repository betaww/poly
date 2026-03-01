"""
VPS Feed Forwarder — runs exchange + oracle WebSocket feeds and publishes to Redis.

Runs on VPS (London) where all exchanges are accessible.
Alienware Brain reads these prices from Redis instead of connecting directly.

Architecture:
  VPS: Binance WS + OKX WS + Coinbase WS + Pyth WS + Chainlink Poll → Redis pub/sub
  Alienware: Redis sub → SyntheticOracle → Redis pub (predictions)

Data sources:
  - Binance, OKX, Coinbase: Direct CEX WebSocket (P1/P2 enriched)
  - Pyth Network (Hermes): Pull-based oracle, 400ms refresh, >0.999 correlation with Chainlink
  - Chainlink (Polygon): On-chain Price Feed poll, 0.1% deviation / 25s heartbeat (calibration only)
"""
import asyncio
import json
import logging
import os
import time

import redis
import websockets
from datetime import datetime, timezone  # M8 FIX: moved from inner loop

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

    # Pyth Hermes feed IDs (stable, hex-encoded)
    PYTH_FEEDS = {
        "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43": "btc",
        "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace": "eth",
    }
    # Chainlink Polygon mainnet Price Feed proxy contracts
    # Verified from https://data.chain.link/feeds/polygon/mainnet/
    CHAINLINK_FEEDS = {
        "btc": "0xc907E116054Ad103354f2D350FD2514433D57F6f",
        "eth": "0xF9680D99D6C9589e2a93a78A04A279e509205945",
    }
    # AggregatorV3Interface.latestRoundData() ABI (only what we need)
    AGGREGATOR_ABI = [{"inputs":[], "name":"latestRoundData", "outputs":[{"name":"roundId","type":"uint80"},{"name":"answer","type":"int256"},{"name":"startedAt","type":"uint256"},{"name":"updatedAt","type":"uint256"},{"name":"answeredInRound","type":"uint80"}], "stateMutability":"view", "type":"function"}]

    async def start(self):
        self._running = True
        tasks = [
            asyncio.create_task(self._binance_feed("btcusdt", "btc")),
            asyncio.create_task(self._binance_feed("ethusdt", "eth")),
            asyncio.create_task(self._okx_feed("BTC-USDT", "btc")),
            asyncio.create_task(self._okx_feed("ETH-USDT", "eth")),
            asyncio.create_task(self._coinbase_feed("BTC-USD", "btc")),
            asyncio.create_task(self._coinbase_feed("ETH-USD", "eth")),
            asyncio.create_task(self._pyth_feed()),
            asyncio.create_task(self._chainlink_poller()),
            asyncio.create_task(self._heartbeat_loop()),
        ]
        logger.info(
            "FeedForwarder started: 8 feeds "
            "(Binance+OKX+Coinbase+Pyth+Chainlink × BTC+ETH)"
        )
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

    # ─── Pyth Network (Hermes SSE) ────────────────────────────

    async def _pyth_feed(self):
        """Connect to Pyth Hermes SSE for BTC + ETH prices.

        Pyth uses pull-based architecture matching Chainlink Data Streams.
        400ms refresh, free public API, >0.999 correlation with Chainlink.
        Uses Server-Sent Events (SSE) via HTTP for maximum compatibility.
        """
        feed_ids = list(self.PYTH_FEEDS.keys())
        ids_param = "&ids[]=".join(feed_ids)
        url = f"https://hermes.pyth.network/v2/updates/price/stream?ids[]={ids_param}&parsed=true"

        while self._running:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        logger.info(f"[VPS] Pyth Hermes SSE connected ({len(feed_ids)} feeds)")
                        buffer = ""
                        async for chunk in resp.content:
                            if not self._running:
                                break
                            buffer += chunk.decode("utf-8", errors="ignore")
                            # L5 FIX: Smart buffer truncation — cut at event boundary
                            if len(buffer) > 65536:
                                last_boundary = buffer.rfind("\n\n", 0, -1)
                                if last_boundary > 0:
                                    buffer = buffer[last_boundary + 2:]
                                else:
                                    buffer = buffer[-32768:]  # fallback: hard cut
                            # SSE messages are separated by double newlines
                            while "\n\n" in buffer:
                                event_str, buffer = buffer.split("\n\n", 1)
                                self._process_pyth_event(event_str)
            except ImportError:
                logger.warning("[VPS] Pyth feed: aiohttp not installed. pip install aiohttp")
                await asyncio.sleep(300)  # don't spam
            except Exception as e:
                logger.error(f"[VPS] Pyth Hermes: {e}")
            if self._running:
                await asyncio.sleep(3)

    def _process_pyth_event(self, event_str: str):
        """Parse Pyth SSE event and publish ticks."""
        # SSE format: "data: {json}\n"
        for line in event_str.strip().split("\n"):
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                    parsed = data.get("parsed", [])
                    for feed in parsed:
                        feed_id = feed.get("id", "")
                        asset = self.PYTH_FEEDS.get(feed_id)
                        if not asset:
                            continue
                        price_data = feed.get("price", {})
                        if not price_data:
                            continue
                        # Pyth price is integer with exponent (e.g. price=8500000, expo=-2 → $85000.00)
                        raw_price = int(price_data.get("price", 0))
                        expo = int(price_data.get("expo", 0))
                        price = raw_price * (10 ** expo)
                        publish_time = int(price_data.get("publish_time", 0))
                        # Pyth confidence interval (same scale)
                        raw_conf = int(price_data.get("conf", 0))
                        conf = raw_conf * (10 ** expo)
                        # Use conf as synthetic bid/ask spread
                        bid = price - conf
                        ask = price + conf
                        if price > 0:
                            self._publish_tick(
                                "pyth", asset, price, bid, ask,
                                exchange_ts=float(publish_time),
                            )
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

    # ─── Chainlink Polygon On-Chain ──────────────────────────

    async def _chainlink_poller(self):
        """Poll Chainlink Price Feeds on Polygon mainnet every 10s.

        Used as calibration reference for drift detection.
        BTC/USD: 0.1% deviation threshold, ~25s heartbeat.
        NOT used as a trading signal — only for Oracle drift monitoring.
        """
        rpc_url = os.environ.get(
            "POLYGON_RPC_URL",
            "https://polygon.drpc.org",  # free public RPC (tested from VPS)
        )
        while self._running:
            try:
                from web3 import Web3
                w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
                if not w3.is_connected():
                    logger.warning("[VPS] Chainlink: Polygon RPC not connected")
                    await asyncio.sleep(30)
                    continue
                logger.info(f"[VPS] Chainlink Polygon poller started (10s interval)")
                # Pre-build contract objects (reuse across iterations)
                contracts = {}
                for asset, addr in self.CHAINLINK_FEEDS.items():
                    contracts[asset] = w3.eth.contract(
                        address=Web3.to_checksum_address(addr),
                        abi=self.AGGREGATOR_ABI,
                    )
                while self._running:
                    for asset, contract in contracts.items():
                        try:
                            result = contract.functions.latestRoundData().call()
                            # result = (roundId, answer, startedAt, updatedAt, answeredInRound)
                            answer = result[1]
                            updated_at = result[3]
                            # Chainlink USD feeds use 8 decimals
                            price = answer / 1e8
                            if price > 0:
                                self._publish_tick(
                                    "chainlink", asset, price, 0.0, 0.0,
                                    exchange_ts=float(updated_at),
                                )
                        except Exception as e:
                            logger.debug(f"[VPS] Chainlink {asset}: {e}")
                    await asyncio.sleep(10)
            except ImportError:
                logger.warning("[VPS] Chainlink poller: web3 not installed. pip install web3")
                await asyncio.sleep(300)  # don't spam
            except Exception as e:
                logger.error(f"[VPS] Chainlink poller: {e}")
            if self._running:
                await asyncio.sleep(10)

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
