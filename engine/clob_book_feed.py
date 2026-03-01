"""
Polymarket CLOB Orderbook Feed — real-time L2 depth data.

Connects to wss://ws-subscriptions-clob.polymarket.com/ws/market
and maintains a local orderbook snapshot per token.

Provides:
  - Real best bid/ask prices
  - Total depth per side (USD)
  - Spread calculation
  - Top-N levels for paper simulator

Used by:
  - CryptoMM: accurate spread pricing around real best bid/ask
  - PaperSimulator: realistic fill probability using actual depth
  - RiskManager: exposure validation against real liquidity
"""
import asyncio
import json
import logging
import queue
import time
import threading
from dataclasses import dataclass, field

import websockets

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class OrderBookLevel:
    """A single price level in the orderbook."""
    price: float
    size: float  # in shares (not USD)

    @property
    def size_usd(self) -> float:
        return self.price * self.size


@dataclass
class OrderBookSnapshot:
    """Local orderbook state for one token."""
    token_id: str
    asset: str      # "btc", "eth"
    outcome: str    # "Up", "Down"
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    last_update: float = 0.0

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 1.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def midpoint(self) -> float:
        if self.best_bid > 0 and self.best_ask < 1:
            return (self.best_bid + self.best_ask) / 2
        return 0.5

    @property
    def total_bid_depth_usd(self) -> float:
        return sum(lvl.size_usd for lvl in self.bids)

    @property
    def total_ask_depth_usd(self) -> float:
        return sum(lvl.size_usd for lvl in self.asks)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.last_update > 10.0

    def depth_at_price(self, price: float, is_buy: bool) -> float:
        """Get total depth available at or better than a price level."""
        total = 0.0
        if is_buy:
            # For a buy order, we consume asks at or below our price
            for lvl in self.asks:
                if lvl.price <= price:
                    total += lvl.size_usd
        else:
            # For a sell order, we consume bids at or above our price
            for lvl in self.bids:
                if lvl.price >= price:
                    total += lvl.size_usd
        return total


class CLOBBookFeed:
    """
    Connects to Polymarket CLOB WebSocket and maintains local orderbook snapshots.

    Usage:
        feed = CLOBBookFeed()
        feed.start(token_ids={"token_up": ("btc", "Up"), "token_down": ("btc", "Down")})

        book = feed.get_book("token_up")
        if book:
            print(f"Best bid: {book.best_bid}, Ask: {book.best_ask}")
    """

    def __init__(self):
        self._books: dict[str, OrderBookSnapshot] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._token_info: dict[str, tuple[str, str]] = {}  # token_id → (asset, outcome)
        self._ws_connected = False
        self._pending_subs: queue.Queue = queue.Queue()  # M5 FIX: thread-safe queue

    def start(self, token_ids: dict[str, tuple[str, str]]):
        """
        Start background thread.

        Args:
            token_ids: dict mapping token_id to (asset, outcome) tuple
                       e.g. {"tok_abc": ("btc", "Up"), "tok_def": ("btc", "Down")}
        """
        self._token_info = token_ids
        for token_id, (asset, outcome) in token_ids.items():
            self._books[token_id] = OrderBookSnapshot(
                token_id=token_id, asset=asset, outcome=outcome
            )

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"CLOB BookFeed started: {len(token_ids)} tokens")

    def subscribe_token(self, token_id: str, asset: str, outcome: str):
        """Subscribe to an additional token after start().

        Thread-safe: new tokens are added to internal state and will be
        subscribed on the next WS reconnect cycle. For immediate subscription,
        we also store a pending queue.
        """
        if token_id in self._token_info:
            return  # already subscribed
        with self._lock:
            self._token_info[token_id] = (asset, outcome)
            self._books[token_id] = OrderBookSnapshot(
                token_id=token_id, asset=asset, outcome=outcome
            )
        self._pending_subs.put(token_id)  # M5 FIX: thread-safe
        logger.info(f"CLOB BookFeed: queued subscription for {asset}_{outcome} {token_id[:16]}...")

    def _run_loop(self):
        """Background thread: run asyncio loop for WS."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect_and_listen())

    async def _connect_and_listen(self):
        """Connect to Polymarket WS and listen for book updates."""
        while self._running:
            try:
                async with websockets.connect(WS_URL, ping_interval=20) as ws:
                    self._ws_connected = True
                    logger.info(f"CLOB WS connected: {WS_URL}")

                    # Subscribe to book channel for each token
                    for token_id in self._token_info:
                        sub_msg = {
                            "type": "subscribe",
                            "channel": "book",
                            "assets_id": token_id,  # Polymarket uses assets_id
                        }
                        await ws.send(json.dumps(sub_msg))
                        logger.info(f"Subscribed to book: {token_id[:16]}...")

                    # Listen for messages
                    async for msg in ws:
                        if not self._running:
                            break
                        # M5 FIX: Process pending late subscriptions (thread-safe)
                        while not self._pending_subs.empty():
                            try:
                                tid = self._pending_subs.get_nowait()
                                sub_msg = {
                                    "type": "subscribe",
                                    "channel": "book",
                                    "assets_id": tid,
                                }
                                await ws.send(json.dumps(sub_msg))
                                logger.info(f"Late-subscribed to book: {tid[:16]}...")
                            except queue.Empty:
                                break
                        try:
                            self._process_message(json.loads(msg))
                        except json.JSONDecodeError:
                            pass  # ACK/ping frames are not JSON

            except websockets.exceptions.ConnectionClosed:
                logger.warning("CLOB WS disconnected, reconnecting in 2s...")
                self._ws_connected = False
            except Exception as e:
                logger.error(f"CLOB WS error: {e}")
                self._ws_connected = False

            if self._running:
                await asyncio.sleep(2)

    def _process_message(self, msg: dict):
        """Process a WebSocket message and update local orderbook."""
        msg_type = msg.get("type", "")

        if msg_type == "book":
            # Full snapshot
            self._handle_book_snapshot(msg)
        elif msg_type == "book_delta":
            # Incremental update
            self._handle_book_delta(msg)
        elif msg_type == "price_change":
            # Price change event — also useful
            self._handle_price_change(msg)

    def _handle_book_snapshot(self, msg: dict):
        """Process a full book snapshot."""
        asset_id = msg.get("asset_id", "")
        if asset_id not in self._books:
            return

        bids = [
            OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in msg.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in msg.get("asks", [])
        ]

        # Sort: bids descending, asks ascending
        bids.sort(key=lambda l: l.price, reverse=True)
        asks.sort(key=lambda l: l.price)

        with self._lock:
            self._books[asset_id].bids = bids
            self._books[asset_id].asks = asks
            self._books[asset_id].last_update = time.time()

    def _handle_book_delta(self, msg: dict):
        """Process incremental book update."""
        asset_id = msg.get("asset_id", "")
        if asset_id not in self._books:
            return

        with self._lock:
            book = self._books[asset_id]

            # Apply bid changes
            for change in msg.get("bids", []):
                price = float(change["price"])
                size = float(change["size"])
                if size == 0:
                    # Remove level
                    book.bids = [b for b in book.bids if abs(b.price - price) > 0.001]
                else:
                    # Update or insert
                    found = False
                    for b in book.bids:
                        if abs(b.price - price) < 0.001:
                            b.size = size
                            found = True
                            break
                    if not found:
                        book.bids.append(OrderBookLevel(price=price, size=size))
                    book.bids.sort(key=lambda l: l.price, reverse=True)

            # Apply ask changes
            for change in msg.get("asks", []):
                price = float(change["price"])
                size = float(change["size"])
                if size == 0:
                    book.asks = [a for a in book.asks if abs(a.price - price) > 0.001]
                else:
                    found = False
                    for a in book.asks:
                        if abs(a.price - price) < 0.001:
                            a.size = size
                            found = True
                            break
                    if not found:
                        book.asks.append(OrderBookLevel(price=price, size=size))
                    book.asks.sort(key=lambda l: l.price)

            book.last_update = time.time()

    def _handle_price_change(self, msg: dict):
        """Process a price change event (lighter than full book)."""
        # These are useful for quick midpoint updates
        pass  # We primarily use book/book_delta

    def get_book(self, token_id: str) -> OrderBookSnapshot | None:
        """Get current orderbook snapshot for a token (thread-safe)."""
        with self._lock:
            book = self._books.get(token_id)
            if book and not book.is_stale:
                return book
        return None

    def get_best_prices(self, token_id: str) -> tuple[float, float]:
        """Get (best_bid, best_ask) for a token. Returns (0, 1) if unavailable."""
        book = self.get_book(token_id)
        if book:
            return book.best_bid, book.best_ask
        return 0.0, 1.0

    def get_depth(self, token_id: str) -> tuple[float, float]:
        """Get (bid_depth_usd, ask_depth_usd) for a token."""
        book = self.get_book(token_id)
        if book:
            return book.total_bid_depth_usd, book.total_ask_depth_usd
        return 0.0, 0.0

    @property
    def is_connected(self) -> bool:
        return self._ws_connected

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("CLOB BookFeed stopped")

    def get_status(self) -> dict:
        """Get feed status for telemetry."""
        with self._lock:
            books = {}
            for tid, book in self._books.items():
                info = self._token_info.get(tid, ("?", "?"))
                books[f"{info[0]}_{info[1]}"] = {
                    "bid": book.best_bid,
                    "ask": book.best_ask,
                    "spread": round(book.spread, 4),
                    "bid_depth": round(book.total_bid_depth_usd, 0),
                    "ask_depth": round(book.total_ask_depth_usd, 0),
                    "age_ms": round((time.time() - book.last_update) * 1000),
                }
            return {
                "connected": self._ws_connected,
                "tokens": len(self._books),
                "books": books,
            }
