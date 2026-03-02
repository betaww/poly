"""
Multi-Exchange L2 Depth Feed — feeds LWBA engine with orderbook snapshots.

Collects L2 orderbook depth from Binance, OKX, and Coinbase via WebSocket.
Each exchange runs in a background thread with auto-reconnect.

Reuses the existing BinanceDepthFeed's raw depth data to avoid duplicate
connections — extends it by also feeding book snapshots to the LWBA engine.
"""
import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field

import websockets

from .lwba_engine import OrderBookSnapshot

logger = logging.getLogger(__name__)


# ─── Exchange-specific stream configs ────────────────────────────────────────

BINANCE_SYMBOLS = {"btc": "btcusdt", "eth": "ethusdt"}
OKX_INST_IDS = {"btc": "BTC-USDT", "eth": "ETH-USDT"}
COINBASE_PRODUCTS = {"btc": "BTC-USD", "eth": "ETH-USD"}


class _ExchangeDepthStream:
    """Base class for exchange depth WebSocket streams."""

    exchange: str = ""

    def __init__(self, assets: list[str]):
        self._assets = assets
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._books: dict[str, OrderBookSnapshot] = {}

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name=f"{self.exchange}-depth"
        )
        self._thread.start()
        logger.info(f"{self.exchange} depth feed started: {self._assets}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def get_book(self, asset: str) -> OrderBookSnapshot | None:
        with self._lock:
            return self._books.get(asset)

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect())

    async def _connect(self):
        raise NotImplementedError

    def _update_book(self, asset: str,
                     bids: list[tuple[float, float]],
                     asks: list[tuple[float, float]]):
        """Thread-safe update of a book snapshot."""
        with self._lock:
            self._books[asset] = OrderBookSnapshot(
                exchange=self.exchange,
                asset=asset,
                bids=bids,
                asks=asks,
                timestamp=time.time(),
            )


class BinanceDepthStream(_ExchangeDepthStream):
    """Binance depth20@100ms — top 20 levels."""

    exchange = "binance"

    async def _connect(self):
        streams = [
            f"{BINANCE_SYMBOLS.get(a, a)}@depth20@100ms"
            for a in self._assets
        ]
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info(f"Binance depth connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            self._handle(msg)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Binance depth disconnected: {e}")
                if self._running:
                    await asyncio.sleep(2)

    def _handle(self, msg: dict):
        stream = msg.get("stream", "")
        data = msg.get("data", {})
        asset = None
        for a, sym in BINANCE_SYMBOLS.items():
            if stream.startswith(sym):
                asset = a
                break
        if not asset:
            return

        bids = [(float(p), float(q)) for p, q in data.get("bids", [])[:20]]
        asks = [(float(p), float(q)) for p, q in data.get("asks", [])[:20]]
        self._update_book(asset, bids, asks)


class OKXDepthStream(_ExchangeDepthStream):
    """OKX books5 (top 5 levels, ~400ms updates)."""

    exchange = "okx"

    async def _connect(self):
        url = "wss://ws.okx.com:8443/ws/v5/public"

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    # Subscribe to depth channels
                    sub_args = [
                        {"channel": "books5", "instId": OKX_INST_IDS[a]}
                        for a in self._assets if a in OKX_INST_IDS
                    ]
                    await ws.send(json.dumps({
                        "op": "subscribe", "args": sub_args
                    }))
                    logger.info(f"OKX depth connected")

                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            self._handle(msg)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"OKX depth disconnected: {e}")
                if self._running:
                    await asyncio.sleep(2)

    def _handle(self, msg: dict):
        # OKX format: {"arg": {"channel": "books5", "instId": "BTC-USDT"},
        #              "data": [{"bids": [["price","qty",...]], "asks": [...]}]}
        arg = msg.get("arg", {})
        data_list = msg.get("data", [])
        if not data_list or "channel" not in arg:
            return

        inst_id = arg.get("instId", "")
        asset = None
        for a, iid in OKX_INST_IDS.items():
            if inst_id == iid:
                asset = a
                break
        if not asset:
            return

        data = data_list[0]
        # OKX bids/asks format: [["price","qty","_","_"], ...]
        bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])[:20]]
        asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])[:20]]
        self._update_book(asset, bids, asks)


class CoinbaseDepthStream(_ExchangeDepthStream):
    """Coinbase level2_batch — L2 orderbook snapshots + updates."""

    exchange = "coinbase"

    def __init__(self, assets: list[str]):
        super().__init__(assets)
        # Full book state (maintained via diffs)
        self._full_bids: dict[str, dict[float, float]] = {}  # {asset: {price: qty}}
        self._full_asks: dict[str, dict[float, float]] = {}
        for a in assets:
            self._full_bids[a] = {}
            self._full_asks[a] = {}

    async def _connect(self):
        url = "wss://ws-feed.exchange.coinbase.com"
        product_ids = [
            COINBASE_PRODUCTS[a] for a in self._assets if a in COINBASE_PRODUCTS
        ]

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "product_ids": product_ids,
                        "channels": ["level2_batch"],
                    }))
                    logger.info(f"Coinbase depth connected")

                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            self._handle(msg)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Coinbase depth disconnected: {e}")
                if self._running:
                    await asyncio.sleep(2)

    def _handle(self, msg: dict):
        msg_type = msg.get("type", "")
        product_id = msg.get("product_id", "")

        asset = None
        for a, pid in COINBASE_PRODUCTS.items():
            if product_id == pid:
                asset = a
                break
        if not asset:
            return

        if msg_type == "snapshot":
            # Full orderbook snapshot
            self._full_bids[asset] = {
                float(p): float(q) for p, q in msg.get("bids", [])
            }
            self._full_asks[asset] = {
                float(p): float(q) for p, q in msg.get("asks", [])
            }
        elif msg_type == "l2update":
            # Incremental updates
            for change in msg.get("changes", []):
                side, price_str, qty_str = change[0], change[1], change[2]
                price = float(price_str)
                qty = float(qty_str)
                book = self._full_bids[asset] if side == "buy" else self._full_asks[asset]
                if qty == 0:
                    book.pop(price, None)
                else:
                    book[price] = qty
        else:
            return

        # Build sorted snapshot (top 20 levels)
        sorted_bids = sorted(
            self._full_bids[asset].items(), key=lambda x: x[0], reverse=True
        )[:20]
        sorted_asks = sorted(
            self._full_asks[asset].items(), key=lambda x: x[0]
        )[:20]
        self._update_book(asset, sorted_bids, sorted_asks)


# ─── Aggregator ──────────────────────────────────────────────────────────────

class MultiExchangeDepthFeeds:
    """Aggregates L2 orderbook depth from multiple exchanges.

    Usage:
        feeds = MultiExchangeDepthFeeds(["btc", "eth"])
        feeds.start()
        books = feeds.get_all_books("btc")
        # → {"binance": OrderBookSnapshot, "okx": ..., "coinbase": ...}
    """

    def __init__(self, assets: list[str]):
        self._assets = assets
        self._streams: dict[str, _ExchangeDepthStream] = {
            "binance": BinanceDepthStream(assets),
            "okx": OKXDepthStream(assets),
            "coinbase": CoinbaseDepthStream(assets),
        }

    def start(self):
        """Start all exchange depth feeds."""
        for stream in self._streams.values():
            stream.start()
        logger.info(
            f"MultiExchangeDepthFeeds started: "
            f"{list(self._streams.keys())} × {self._assets}"
        )

    def stop(self):
        """Stop all feeds."""
        for stream in self._streams.values():
            stream.stop()

    def get_all_books(self, asset: str) -> dict[str, OrderBookSnapshot]:
        """Get current L2 snapshots from all exchanges for an asset.

        Returns: {exchange_name: OrderBookSnapshot}
        Only includes exchanges with valid (non-None) books.
        """
        result = {}
        for name, stream in self._streams.items():
            book = stream.get_book(asset)
            if book is not None and book.is_valid:
                result[name] = book
        return result

    def get_book(self, exchange: str, asset: str) -> OrderBookSnapshot | None:
        """Get book from a specific exchange."""
        stream = self._streams.get(exchange)
        return stream.get_book(asset) if stream else None
