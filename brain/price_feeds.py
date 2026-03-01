"""
Alienware Brain Node — Multi-Exchange Price Feeds.

Connects to Binance, Coinbase, Kraken, OKX via WebSocket.
Aggregates into a synthetic oracle prediction.
Publishes predictions to Redis for VPS consumption.

This node does ONE thing well: predict the Chainlink settlement price
faster than anyone else, using the RTX 5090.
"""
import asyncio
import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field

import websockets

from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class PriceTick:
    """A single price update from an exchange."""
    exchange: str
    asset: str  # "btc" or "eth"
    price: float
    timestamp: float
    bid: float = 0.0
    ask: float = 0.0


class BinanceFeed:
    """Binance WebSocket — primary price source (45% weight)."""

    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol
        self.url = f"wss://stream.binance.com:9443/ws/{symbol}@ticker"
        self.last_price: float = 0.0
        self._running = False

    async def connect(self, callback, asset: str = "btc"):
        """Connect and stream price updates."""
        self._running = True
        proxy_url = os.environ.get("POLY_SOCKS_PROXY", "")

        while self._running:
            try:
                # Create proxy socket fresh each attempt
                ws_kwargs = {}
                if proxy_url:
                    try:
                        from python_socks.async_.asyncio import Proxy
                        p = Proxy.from_url(proxy_url)
                        sock = await p.connect(
                            dest_host="stream.binance.com",
                            dest_port=9443,
                        )
                        # Pass raw socket; websockets handles TLS via wss:// URI
                        ws_kwargs["sock"] = sock
                        ws_kwargs["server_hostname"] = "stream.binance.com"
                        logger.info("Binance proxy socket ready")
                    except Exception as e:
                        logger.warning(f"Proxy failed: {e}, trying direct")

                async with websockets.connect(self.url, **ws_kwargs) as ws:
                    logger.info(f"Binance WS connected: {self.symbol}")
                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        tick = PriceTick(
                            exchange="binance",
                            asset=asset,
                            # P2 FIX: Use bid-ask midpoint instead of last trade
                            price=(
                                (float(data.get("b", 0)) + float(data.get("a", 0))) / 2
                                if float(data.get("b", 0)) > 0 and float(data.get("a", 0)) > 0
                                else float(data.get("c", 0))
                            ),
                            bid=float(data.get("b", 0)),
                            ask=float(data.get("a", 0)),
                            # P1 FIX: Use exchange event timestamp
                            timestamp=data.get("E", int(time.time() * 1000)) / 1000.0,
                        )
                        self.last_price = tick.price
                        await callback(tick)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Binance WS disconnected, reconnecting...")
            except Exception as e:
                logger.error(f"Binance WS error: {e}")
            if self._running:
                await asyncio.sleep(2)

    def stop(self):
        self._running = False


class CoinbaseFeed:
    """Coinbase WebSocket — secondary price source (30% weight)."""

    def __init__(self, symbol: str = "BTC-USD"):
        self.symbol = symbol
        self.url = "wss://ws-feed.exchange.coinbase.com"
        self.last_price: float = 0.0
        self._running = False

    async def connect(self, callback, asset: str = "btc"):
        self._running = True
        while self._running:
            try:
                async with websockets.connect(self.url) as ws:
                    sub = {
                        "type": "subscribe",
                        "product_ids": [self.symbol],
                        "channels": ["ticker"],
                    }
                    await ws.send(json.dumps(sub))
                    logger.info(f"Coinbase WS connected: {self.symbol}")

                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        if data.get("type") != "ticker":
                            continue
                        tick = PriceTick(
                            exchange="coinbase",
                            asset=asset,
                            # P2 FIX: Use bid-ask midpoint
                            price=(
                                (float(data.get("best_bid", 0)) + float(data.get("best_ask", 0))) / 2
                                if float(data.get("best_bid", 0)) > 0 and float(data.get("best_ask", 0)) > 0
                                else float(data.get("price", 0))
                            ),
                            bid=float(data.get("best_bid", 0)),
                            ask=float(data.get("best_ask", 0)),
                            timestamp=time.time(),
                        )
                        self.last_price = tick.price
                        await callback(tick)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Coinbase WS disconnected, reconnecting...")
            except Exception as e:
                logger.error(f"Coinbase WS error: {e}")
            if self._running:
                await asyncio.sleep(1)

    def stop(self):
        self._running = False


class OKXFeed:
    """OKX WebSocket — replaces Binance for restricted regions (no HTTP 451)."""

    # symbol map: btcusdt → BTC-USDT
    SYMBOL_MAP = {
        "btcusdt": "BTC-USDT",
        "ethusdt": "ETH-USDT",
    }

    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol
        self.okx_symbol = self.SYMBOL_MAP.get(symbol, symbol.upper().replace("usdt", "-USDT"))
        self.url = "wss://ws.okx.com:8443/ws/v5/public"
        self.last_price: float = 0.0
        self._running = False

    async def connect(self, callback, asset: str = "btc"):
        self._running = True
        while self._running:
            try:
                async with websockets.connect(self.url) as ws:
                    # Subscribe to ticker channel
                    sub = {
                        "op": "subscribe",
                        "args": [{"channel": "tickers", "instId": self.okx_symbol}],
                    }
                    await ws.send(json.dumps(sub))
                    logger.info(f"OKX WS connected: {self.okx_symbol}")

                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        # OKX sends: {"arg":..., "data":[{"last":"84500",...}]}
                        if "data" not in data:
                            continue
                        for item in data["data"]:
                            tick = PriceTick(
                                exchange="okx",
                                asset=asset,
                                # P2 FIX: Use bid-ask midpoint
                                price=(
                                    (float(item.get("bidPx", 0)) + float(item.get("askPx", 0))) / 2
                                    if float(item.get("bidPx", 0)) > 0 and float(item.get("askPx", 0)) > 0
                                    else float(item.get("last", 0))
                                ),
                                bid=float(item.get("bidPx", 0)),
                                ask=float(item.get("askPx", 0)),
                                # P1: OKX provides ms timestamp
                                timestamp=int(item.get("ts", int(time.time() * 1000))) / 1000.0,
                            )
                            self.last_price = tick.price
                            await callback(tick)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("OKX WS disconnected, reconnecting...")
            except Exception as e:
                logger.error(f"OKX WS error: {e}")
            if self._running:
                await asyncio.sleep(1)

    def stop(self):
        self._running = False


class SyntheticOracle:
    """
    Weighted price aggregator that mimics Chainlink Data Streams.

    Produces a synthetic settlement price prediction by combining
    multiple exchange feeds with adaptive weights.
    """

    def __init__(self, config: Config):
        self.config = config
        self.weights = dict(config.oracle.weights)
        self.prices: dict[str, float] = {}
        self.update_times: dict[str, float] = {}
        self._prediction_history: deque = deque(maxlen=300)  # M8 FIX: bounded deque
        # P5: Raw tick deque for accurate volatility calculation
        self._raw_ticks: deque = deque(maxlen=500)
        # P6: Per-exchange latency tracking
        self._exchange_latencies: dict[str, deque] = {}

    def update(self, tick: PriceTick):
        """Process a new price tick."""
        self.prices[tick.exchange] = tick.price
        self.update_times[tick.exchange] = tick.timestamp
        # P5: Store raw ticks for volatility
        self._raw_ticks.append((tick.timestamp, tick.price))
        # P6: Track observed latency (exchange timestamp vs local receipt)
        local_now = time.time()
        if tick.timestamp > 0 and tick.timestamp < local_now + 1:  # sanity check
            latency_ms = (local_now - tick.timestamp) * 1000
            if 0 < latency_ms < 5000:  # filter unreasonable values
                if tick.exchange not in self._exchange_latencies:
                    self._exchange_latencies[tick.exchange] = deque(maxlen=100)
                self._exchange_latencies[tick.exchange].append(latency_ms)

    def predict(self) -> tuple[float, float]:
        """
        Generate synthetic Chainlink settlement price prediction.
        Returns (predicted_price, confidence).
        """
        if not self.prices:
            return 0.0, 0.0

        # Weighted average of available prices
        total_weight = 0.0
        weighted_sum = 0.0
        available_count = 0

        for exchange, weight in self.weights.items():
            if exchange in self.prices:
                # P3 FIX: Exponential staleness decay (8s half-life)
                # Linear was too aggressive: old code killed weight at 5s
                age = time.time() - self.update_times.get(exchange, 0)
                staleness_factor = max(0.1, math.exp(-age * 0.693 / 8.0))  # ln(2)/8 ≈ 0.0866

                # P6: Latency penalty — adjust weight for systematically slower exchanges
                lat_dq = self._exchange_latencies.get(exchange)
                if lat_dq and len(lat_dq) > 10:
                    avg_latency = sum(lat_dq) / len(lat_dq)
                    # Exchanges with >200ms extra latency get penalized
                    latency_penalty = max(0.7, 1.0 - max(0, avg_latency - 100) / 500)
                else:
                    latency_penalty = 1.0

                effective_weight = weight * staleness_factor * latency_penalty
                weighted_sum += self.prices[exchange] * effective_weight
                total_weight += effective_weight
                available_count += 1

        if total_weight == 0:
            return 0.0, 0.0

        predicted = weighted_sum / total_weight

        # P4 FIX: Use MAD (Median Absolute Deviation) instead of max_deviation
        # MAD is robust to single outlier exchanges (e.g., OKX flash crash)
        if len(self.prices) >= 2:
            prices = list(self.prices.values())
            median = sorted(prices)[len(prices) // 2]
            if median > 0:
                deviations = sorted(abs(p - median) / median for p in prices)
                mad = deviations[len(deviations) // 2]  # median of absolute deviations
                # 0.1% MAD → conf ~0.98, 0.5% MAD → conf ~0.90
                confidence = max(0.5, 1.0 - mad * 20)
            else:
                confidence = 0.5
            # Source penalty: fewer sources = lower confidence
            expected_sources = len(self.weights)
            source_penalty = available_count / max(expected_sources, 1)
            confidence *= source_penalty
            confidence = max(0.5, confidence)
        else:
            confidence = 0.6  # single source = lower confidence

        self._prediction_history.append(predicted)

        return predicted, min(confidence, 0.99)

    def get_volatility(self) -> float:
        """P5 FIX: Calculate volatility from RAW ticks, not smoothed predictions.

        Using raw ticks gives a more accurate picture of actual market volatility,
        since the weighted average prediction smooths out exchange-to-exchange noise.
        """
        # P5: Use raw ticks if available
        if len(self._raw_ticks) >= 10:
            recent = list(self._raw_ticks)[-60:]
            if len(recent) >= 2:
                prices_only = [p for _, p in recent]
                mean = sum(prices_only) / len(prices_only)
                if mean > 0:
                    returns = [
                        (prices_only[i] - prices_only[i-1]) / prices_only[i-1]
                        for i in range(1, len(prices_only))
                        if prices_only[i-1] != 0
                    ]
                    if returns:
                        variance = sum(r**2 for r in returns) / len(returns)
                        return max(0.001, variance ** 0.5)

        # Fallback: use prediction history
        if len(self._prediction_history) < 10:
            return 0.01
        recent = list(self._prediction_history)[-60:]
        if len(recent) < 2:
            return 0.01
        mean = sum(recent) / len(recent)
        if mean == 0:
            return 0.01
        returns = [(recent[i] - recent[i-1]) / recent[i-1]
                    for i in range(1, len(recent)) if recent[i-1] != 0]
        if not returns:
            return 0.01
        variance = sum(r**2 for r in returns) / len(returns)
        return max(0.001, variance ** 0.5)

    def get_direction(self, strike_price: float) -> tuple[str, float]:
        """
        Predict settlement direction relative to strike.
        Returns (direction, confidence).
        """
        predicted, base_conf = self.predict()
        if predicted == 0 or strike_price == 0:
            return "Unknown", 0.0

        distance = (predicted - strike_price) / strike_price
        direction = "Up" if distance > 0 else "Down"

        # Amplify confidence based on distance from strike
        abs_dist = abs(distance)
        if abs_dist > 0.001:
            conf = min(0.99, base_conf + abs_dist * 10)
        elif abs_dist > 0.0003:
            conf = base_conf * 0.95
        else:
            conf = 0.50  # too close to call

        return direction, conf
