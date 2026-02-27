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
import os
import time
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
                            price=float(data.get("c", 0)),  # last price
                            bid=float(data.get("b", 0)),
                            ask=float(data.get("a", 0)),
                            timestamp=time.time(),
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
                            price=float(data.get("price", 0)),
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
                                price=float(item.get("last", 0)),
                                bid=float(item.get("bidPx", 0)),
                                ask=float(item.get("askPx", 0)),
                                timestamp=time.time(),
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
        self._prediction_history: list[float] = []

    def update(self, tick: PriceTick):
        """Process a new price tick."""
        self.prices[tick.exchange] = tick.price
        self.update_times[tick.exchange] = tick.timestamp

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

        for exchange, weight in self.weights.items():
            if exchange in self.prices:
                # Staleness penalty: reduce weight if data is old
                age = time.time() - self.update_times.get(exchange, 0)
                staleness_factor = max(0.1, 1.0 - age / 10.0)  # decay over 10s

                effective_weight = weight * staleness_factor
                weighted_sum += self.prices[exchange] * effective_weight
                total_weight += effective_weight

        if total_weight == 0:
            return 0.0, 0.0

        predicted = weighted_sum / total_weight

        # Confidence based on source agreement
        if len(self.prices) >= 2:
            prices = list(self.prices.values())
            median = sorted(prices)[len(prices) // 2]
            max_deviation = max(abs(p - median) / median for p in prices) if median > 0 else 0
            # Low deviation = high confidence
            confidence = max(0.5, 1.0 - max_deviation * 100)
        else:
            confidence = 0.7  # single source = lower confidence

        self._prediction_history.append(predicted)
        if len(self._prediction_history) > 300:
            self._prediction_history = self._prediction_history[-300:]

        return predicted, min(confidence, 0.99)

    def get_volatility(self) -> float:
        """Calculate rolling volatility from prediction history."""
        if len(self._prediction_history) < 10:
            return 0.01
        recent = self._prediction_history[-60:]
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
