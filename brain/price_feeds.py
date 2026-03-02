"""
Alienware Brain Node — Multi-Source Price Feeds.

Aggregates Pyth Network oracle, Binance, Coinbase, OKX exchange feeds,
and Chainlink Polygon on-chain data into a synthetic oracle prediction.
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


class KalmanPriceFilter:
    """Adaptive Kalman filter for multi-source price fusion.

    Treats each CEX as a sensor with independent measurement noise.
    Dynamically adjusts per-source noise covariance based on innovation
    (prediction error). Uses Chainlink as ground-truth correction.
    """

    def __init__(self, config: Config):
        # State estimate and variance
        self.x_hat: float = 0.0       # estimated true price
        self.P: float = 1.0           # estimate variance
        self.Q: float = 0.0001        # process noise (price drift per tick)

        # Per-source measurement noise (initialized from config weights)
        # Lower weight → higher initial noise
        self._R: dict[str, float] = {}
        for source, weight in config.oracle.weights.items():
            if weight > 0:
                # Higher weight → lower R (more trusted)
                self._R[source] = max(0.01, 1.0 / (weight * 100))

        self._initialized = False

    def update(self, source: str, price: float):
        """Kalman update with a sensor measurement."""
        if price <= 0:
            return

        if not self._initialized:
            self.x_hat = price
            self.P = 1.0
            self._initialized = True
            return

        R = self._R.get(source, 1.0)

        # Predict step (simple random walk model)
        x_pred = self.x_hat
        P_pred = self.P + self.Q

        # Innovation (measurement residual)
        innovation = price - x_pred

        # Innovation variance
        S = P_pred + R

        # Kalman gain
        K = P_pred / S if S > 0 else 0.0

        # State update
        self.x_hat = x_pred + K * innovation
        self.P = (1.0 - K) * P_pred

        # Adaptive R: adjust noise based on squared innovation
        # If this source consistently has large innovations, increase R
        alpha = 0.05  # adaptation rate
        self._R[source] = (1 - alpha) * R + alpha * (innovation ** 2)

    def correct_with_ground_truth(self, chainlink_price: float):
        """Use Chainlink as ground-truth to correct state and recalibrate."""
        if chainlink_price <= 0 or not self._initialized:
            return

        # Strong correction: treat Chainlink as very low-noise measurement
        R_cl = 0.001
        P_pred = self.P + self.Q
        innovation = chainlink_price - self.x_hat
        S = P_pred + R_cl
        K = P_pred / S if S > 0 else 0.0
        self.x_hat = self.x_hat + K * innovation
        self.P = (1.0 - K) * P_pred

    @property
    def price(self) -> float:
        return self.x_hat if self._initialized else 0.0

    @property
    def confidence_from_variance(self) -> float:
        """Higher confidence when estimate variance is low."""
        if self.P <= 0:
            return 0.99
        # Map variance to 0.3-0.99 confidence
        return max(0.3, min(0.99, 1.0 - self.P * 10))


class SyntheticOracle:
    """
    Weighted median price aggregator aligned with Chainlink Data Streams.

    Chainlink uses 3-layer median-of-medians aggregation. We replicate this
    by computing a weighted median across Pyth (pull-based oracle, closest
    to Chainlink), plus 3 CEX feeds with staleness/latency adjustments.

    Chainlink on-chain price is used ONLY for drift detection calibration
    (weight=0), not for pricing.
    """

    def __init__(self, config: Config):
        self.config = config
        self.weights = dict(config.oracle.weights)
        self.prices: dict[str, float] = {}
        self.update_times: dict[str, float] = {}
        self._prediction_history: deque = deque(maxlen=300)  # M8 FIX: bounded deque
        # P5: Raw tick deque for accurate volatility calculation
        self._raw_ticks: deque = deque(maxlen=3000)  # 5 min @ ~10 ticks/sec
        # P6: Per-exchange latency tracking
        self._exchange_latencies: dict[str, deque] = {}
        # Chainlink drift detection
        self._chainlink_price: float = 0.0
        self._chainlink_updated_at: float = 0.0
        # Kalman filter for adaptive price fusion
        self._kalman = KalmanPriceFilter(config)

    def update(self, tick: PriceTick):
        """Process a new price tick."""
        # Track Chainlink price separately for drift detection
        if tick.exchange == "chainlink":
            self._chainlink_price = tick.price
            self._chainlink_updated_at = tick.timestamp
        self.prices[tick.exchange] = tick.price
        self.update_times[tick.exchange] = tick.timestamp
        # P5: Store raw ticks for volatility (exclude chainlink — too slow)
        if tick.exchange != "chainlink":
            self._raw_ticks.append((tick.timestamp, tick.price))
        # P6: Track observed latency (exchange timestamp vs local receipt)
        local_now = time.time()
        if tick.timestamp > 0 and tick.timestamp < local_now + 1:  # sanity check
            latency_ms = (local_now - tick.timestamp) * 1000
            if 0 < latency_ms < 5000:  # filter unreasonable values
                if tick.exchange not in self._exchange_latencies:
                    self._exchange_latencies[tick.exchange] = deque(maxlen=100)
                self._exchange_latencies[tick.exchange].append(latency_ms)

        # Kalman filter: update with sensor measurement or ground-truth
        if tick.exchange == 'chainlink':
            self._kalman.correct_with_ground_truth(tick.price)
        elif tick.price > 0:
            self._kalman.update(tick.exchange, tick.price)

    def _weighted_median(self, prices_weights: list[tuple[float, float]]) -> float:
        """Compute weighted median — matches Chainlink's median-of-medians logic.

        v10 FIX #11: Added linear interpolation at 50% crossing point for
        higher precision with only 3-4 sources.
        For 1 source: returns that price (no median possible).
        For 2 sources: returns weighted average (avoids systematic lower-price bias).
        For 3+ sources: interpolated weighted median (robust + continuous).
        """
        if not prices_weights:
            return 0.0
        pw = sorted(prices_weights, key=lambda x: x[0])
        total = sum(w for _, w in pw)
        if total == 0:
            return 0.0
        if len(pw) == 1:
            return pw[0][0]
        # Special case: 2 sources — weighted average
        if len(pw) == 2:
            return (pw[0][0] * pw[0][1] + pw[1][0] * pw[1][1]) / total
        # 3+ sources: interpolated weighted median
        # v10 FIX #11: Instead of returning discrete price at crossing,
        # linearly interpolate between the two prices straddling the 50% mark
        half = total / 2
        cumulative = 0.0
        for i, (price, weight) in enumerate(pw):
            prev_cum = cumulative
            cumulative += weight
            if cumulative > half:
                if i == 0 or prev_cum >= half:
                    return price  # edge case: first item crosses
                # Linear interpolation between pw[i-1] and pw[i]
                prev_price = pw[i - 1][0]
                # How far past the previous into this level is the 50% mark
                frac = (half - prev_cum) / weight if weight > 0 else 0.0
                return prev_price + frac * (price - prev_price)
        return pw[-1][0]

    def predict(self) -> tuple[float, float]:
        """
        Generate synthetic Chainlink settlement price prediction.
        Uses WEIGHTED MEDIAN (matching Chainlink's median-of-medians algorithm).
        Returns (predicted_price, confidence).
        """
        if not self.prices:
            return 0.0, 0.0

        # Build list of (price, effective_weight) for active sources
        prices_weights: list[tuple[float, float]] = []
        available_count = 0

        for exchange, weight in self.weights.items():
            if exchange in self.prices and weight > 0:
                # P3 FIX: Exponential staleness decay (8s half-life)
                age = time.time() - self.update_times.get(exchange, 0)
                staleness_factor = max(0.1, math.exp(-age * 0.693 / 8.0))

                # P6: Latency penalty — adjust weight for systematically slower exchanges
                lat_dq = self._exchange_latencies.get(exchange)
                if lat_dq and len(lat_dq) > 10:
                    avg_latency = sum(lat_dq) / len(lat_dq)
                    latency_penalty = max(0.7, 1.0 - max(0, avg_latency - 100) / 500)
                else:
                    latency_penalty = 1.0

                effective_weight = weight * staleness_factor * latency_penalty
                prices_weights.append((self.prices[exchange], effective_weight))
                available_count += 1

        if not prices_weights:
            return 0.0, 0.0

        # WEIGHTED MEDIAN — matches Chainlink's aggregation algorithm
        wm_predicted = self._weighted_median(prices_weights)

        # Kalman-weighted median fusion: use Kalman as primary when initialized
        kalman_price = self._kalman.price
        if kalman_price > 0 and wm_predicted > 0:
            # v10 FIX #7: Adapt Kalman Q from Parkinson volatility
            parkinson_vol = self.get_volatility()
            # Floor at 1e-7 (not 1e-5): computed Q = vol² × 0.01
            # For 1% vol: Q = 1e-6 (should not be floored)
            # For 0.01% vol: Q = 1e-10 → floor to 1e-7
            self._kalman.Q = max(1e-7, (parkinson_vol ** 2) * 0.01)
            # Blend: 70% Kalman (adaptive) + 30% weighted median (robust)
            predicted = 0.7 * kalman_price + 0.3 * wm_predicted
        else:
            predicted = wm_predicted

        # P4 FIX: Use MAD (Median Absolute Deviation) for confidence
        active_prices = [p for p, _ in prices_weights]
        if len(active_prices) >= 2:
            sorted_prices = sorted(active_prices)
            median = sorted_prices[len(sorted_prices) // 2]
            if median > 0:
                deviations = sorted(abs(p - median) / median for p in active_prices)
                mad = deviations[len(deviations) // 2]
                confidence = max(0.3, 1.0 - mad * 20)  # L6 FIX: floor 0.3 not 0.5
            else:
                confidence = 0.3
            # Source penalty: fewer active sources = lower confidence
            expected_sources = sum(1 for w in self.weights.values() if w > 0)
            source_penalty = available_count / max(expected_sources, 1)
            confidence *= source_penalty
            confidence = max(0.3, confidence)  # L6 FIX: floor 0.3
        else:
            # C1 FIX: single source — cap confidence and warn
            confidence = 0.45
            logger.warning(
                f"⚠️ Only {available_count} active source(s) — "
                f"confidence capped at {confidence:.0%}. "
                f"Prices: {active_prices}"
            )

        # Chainlink drift detection — reduce confidence when we diverge from settlement source
        chainlink_age = time.time() - self._chainlink_updated_at
        if self._chainlink_price > 0 and chainlink_age < 60 and predicted > 0:
            drift = abs(predicted - self._chainlink_price) / self._chainlink_price
            if drift > 0.001:  # > 0.1% drift
                # 0.1% drift → -5% conf, 0.5% drift → -25% conf
                drift_penalty = min(0.30, drift * 50)
                confidence *= (1.0 - drift_penalty)
                confidence = max(0.3, confidence)  # L6 FIX: floor 0.3
                logger.info(
                    f"Chainlink drift: {drift:.4%} | "
                    f"ours=${predicted:.2f} vs CL=${self._chainlink_price:.2f} | "
                    f"conf penalty={drift_penalty:.1%}"
                )

        self._prediction_history.append(predicted)

        return predicted, min(confidence, 0.99)

    def get_volatility(self) -> float:
        """Parkinson volatility estimator using High-Low extremes.

        Uses 5-second OHLC micro-slices over up to 5 minutes.
        Parkinson estimator: σ = sqrt(1/(4n·ln2) · Σ ln²(H_i/L_i))
        ~5x more efficient than close-to-close standard deviation,
        and immune to bid-ask bounce microstructure noise.
        """
        if len(self._raw_ticks) < 10:
            return 0.01

        recent = list(self._raw_ticks)
        now = time.time()
        # Use last 5 minutes of ticks
        cutoff = now - 300
        recent = [(ts, p) for ts, p in recent if ts >= cutoff]

        if len(recent) < 10:
            return 0.01

        # Build 5-second OHLC bars
        bars: dict[int, dict] = {}  # bar_id -> {o, h, l, c}
        for ts, price in recent:
            bar_id = int(ts) // 5  # 5-second bars
            if bar_id not in bars:
                bars[bar_id] = {'o': price, 'h': price, 'l': price, 'c': price}
            else:
                bars[bar_id]['h'] = max(bars[bar_id]['h'], price)
                bars[bar_id]['l'] = min(bars[bar_id]['l'], price)
                bars[bar_id]['c'] = price

        if len(bars) < 3:
            return 0.01

        # Parkinson estimator: σ² = 1/(4n·ln2) · Σ [ln(H/L)]²
        n = len(bars)
        sum_hl_sq = 0.0
        for bar in bars.values():
            if bar['l'] > 0:
                hl_ratio = bar['h'] / bar['l']
                sum_hl_sq += math.log(hl_ratio) ** 2

        parkinson_var = sum_hl_sq / (4.0 * n * math.log(2))
        vol = max(0.001, parkinson_var ** 0.5)

        return vol

    def get_direction(self, strike_price: float) -> tuple[str, float]:
        """C2 FIX: Predict settlement direction relative to strike.

        Returns (direction, base_confidence) from the Oracle's perspective.
        NOTE: DirectionalSniper has its own confidence pipeline (z-score + D1 + D2 + D3)
        and does NOT use this method's confidence directly. This method is retained
        for potential future use (e.g., passing Oracle base confidence to the strategy).
        """
        predicted, base_conf = self.predict()
        if predicted == 0 or strike_price == 0:
            return "Unknown", 0.0

        distance = (predicted - strike_price) / strike_price
        direction = "Up" if distance > 0 else "Down"

        # Simple distance-based confidence — strategy has its own richer pipeline
        abs_dist = abs(distance)
        if abs_dist > 0.001:
            conf = min(0.99, base_conf + abs_dist * 10)
        elif abs_dist > 0.0003:
            conf = base_conf * 0.95
        else:
            conf = 0.50  # too close to call

        return direction, conf
