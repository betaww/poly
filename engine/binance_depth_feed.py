"""
Binance L2 Depth Feed — Order Flow Imbalance (OFI) signal.

Connects to Binance WebSocket depth stream and computes real-time OFI
from top-of-book order flow changes. OFI captures the net buying/selling
pressure that drives short-term price movements.

OFI = Σ (Δbid_volume - Δask_volume) at best N levels

A positive OFI indicates net buying pressure (more bid additions or ask
removals), a negative OFI indicates net selling pressure.
"""
import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import websockets

logger = logging.getLogger(__name__)

# Top-20 levels @ 100ms updates
WS_URL_TEMPLATE = "wss://stream.binance.com:9443/ws/{symbol}@depth20@100ms"

SYMBOL_MAP = {
    "btc": "btcusdt",
    "eth": "ethusdt",
}


@dataclass
class DepthLevel:
    """A single price level in the orderbook."""
    price: float
    qty: float


class BinanceDepthFeed:
    """Binance L2 depth WebSocket → Order Flow Imbalance signal.

    Usage:
        feed = BinanceDepthFeed()
        feed.start(["btc", "eth"])
        ofi = feed.get_ofi("btc")  # -1.0 to +1.0
    """

    def __init__(self, levels: int = 5, window_sec: float = 10.0):
        """
        Args:
            levels: number of top orderbook levels to track for OFI
            window_sec: rolling OFI accumulation window
        """
        self._levels = levels
        self._window_sec = window_sec
        self._running = False
        self._thread: threading.Thread | None = None

        # Per-asset state
        self._prev_bids: dict[str, list[DepthLevel]] = {}
        self._prev_asks: dict[str, list[DepthLevel]] = {}
        self._ofi_history: dict[str, deque] = {}  # (timestamp, ofi_tick)
        self._assets: list[str] = []
        self._lock = threading.Lock()

    def start(self, assets: list[str]):
        """Start background thread for depth streams."""
        if self._running:
            return
        self._assets = assets
        for asset in assets:
            self._ofi_history[asset] = deque(maxlen=1000)
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"BinanceDepthFeed started: {assets}")

    def _run_loop(self):
        """Background thread: run asyncio loop for WS."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect_and_listen())

    async def _connect_and_listen(self):
        """Connect to Binance WS and listen for depth updates."""
        # Multi-stream URL
        streams = [f"{SYMBOL_MAP.get(a, a)}@depth20@100ms" for a in self._assets]
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info(f"BinanceDepthFeed connected: {url}")
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            self._process_message(msg)
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            logger.debug(f"Depth parse error: {e}")
            except Exception as e:
                logger.warning(f"BinanceDepthFeed disconnected: {e}")
                if self._running:
                    await asyncio.sleep(2)

    def _process_message(self, msg: dict):
        """Process a combined stream depth message."""
        stream = msg.get("stream", "")
        data = msg.get("data", {})

        # Determine asset from stream name
        asset = None
        for a, symbol in SYMBOL_MAP.items():
            if stream.startswith(symbol):
                asset = a
                break
        if not asset:
            return

        # Parse bids and asks (top N levels)
        raw_bids = data.get("bids", [])[:self._levels]
        raw_asks = data.get("asks", [])[:self._levels]

        bids = [DepthLevel(float(p), float(q)) for p, q in raw_bids]
        asks = [DepthLevel(float(p), float(q)) for p, q in raw_asks]

        # Compute OFI tick
        with self._lock:
            ofi_tick = self._compute_ofi_tick(asset, bids, asks)
            self._prev_bids[asset] = bids
            self._prev_asks[asset] = asks

            if ofi_tick is not None:
                self._ofi_history[asset].append((time.time(), ofi_tick))

    def _compute_ofi_tick(
        self, asset: str, bids: list[DepthLevel], asks: list[DepthLevel]
    ) -> float | None:
        """Compute single OFI tick from depth delta.

        OFI = Δ(total bid qty) - Δ(total ask qty) at top N levels.
        This captures the net arrival of buying vs selling interest.
        """
        if asset not in self._prev_bids:
            return None  # first snapshot, no delta

        prev_bid_qty = sum(l.qty for l in self._prev_bids[asset])
        prev_ask_qty = sum(l.qty for l in self._prev_asks[asset])
        curr_bid_qty = sum(l.qty for l in bids)
        curr_ask_qty = sum(l.qty for l in asks)

        delta_bid = curr_bid_qty - prev_bid_qty
        delta_ask = curr_ask_qty - prev_ask_qty

        # OFI = bid growth - ask growth
        # Positive: buying pressure (bids increasing or asks decreasing)
        # Negative: selling pressure
        return delta_bid - delta_ask

    def get_ofi(self, asset: str) -> float:
        """Get normalized OFI for an asset over the rolling window.

        Returns:
            float in [-1.0, +1.0], where:
            +1.0 = extreme buying pressure
            -1.0 = extreme selling pressure
             0.0 = balanced or no data
        """
        with self._lock:
            history = self._ofi_history.get(asset)
            if not history:
                return 0.0

            now = time.time()
            cutoff = now - self._window_sec

            # Sum OFI ticks within window
            raw_ofi = sum(ofi for ts, ofi in history if ts >= cutoff)

            # Normalize: use a reference scale based on typical BTC/ETH L2 depth
            # BTC top-5 depth is typically 5-50 BTC, so a ±10 OFI is significant
            if asset == "btc":
                scale = 10.0  # ~10 BTC net flow = strong signal
            elif asset == "eth":
                scale = 100.0  # ~100 ETH net flow = strong signal
            else:
                scale = 10.0

            normalized = max(-1.0, min(1.0, raw_ofi / scale))
            return normalized

    def stop(self):
        """Stop the depth feed."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("BinanceDepthFeed stopped")
