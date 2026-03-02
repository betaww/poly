"""
LWBA Engine — Chainlink shadow price replication.

Implements the Liquidity-Weighted Bid-Ask (LWBA) algorithm used by
Chainlink Data Streams for Polymarket 5-minute crypto settlement.

Algorithm (from Chainlink docs):
  LWBA_Bid = Σ(qty_i × price_i) / Σ(qty_i)   for all bid levels
  LWBA_Ask = Σ(qty_j × price_j) / Σ(qty_j)   for all ask levels
  LWBA_Mid = (Bid + Ask) / 2

Multi-exchange aggregation (replicating DON consensus):
  Final_Bid = Median(LWBA_Bid from each exchange)
  Final_Ask = Median(LWBA_Ask from each exchange)
  Final_Mid = (Final_Bid + Final_Ask) / 2
"""
import logging
import statistics
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """L2 orderbook snapshot from a single exchange."""
    exchange: str
    asset: str
    bids: list[tuple[float, float]]  # [(price, qty), ...] descending by price
    asks: list[tuple[float, float]]  # [(price, qty), ...] ascending by price
    timestamp: float = 0.0

    @property
    def is_valid(self) -> bool:
        return len(self.bids) >= 2 and len(self.asks) >= 2

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return 0.0


@dataclass
class LWBAResult:
    """Result of a multi-exchange LWBA computation."""
    bid: float = 0.0          # LWBA weighted bid (median across exchanges)
    ask: float = 0.0          # LWBA weighted ask (median across exchanges)
    mid: float = 0.0          # Shadow price = (bid + ask) / 2
    spread: float = 0.0       # ask - bid
    spread_bps: float = 0.0   # spread in basis points
    n_sources: int = 0        # number of contributing exchanges
    timestamp: float = 0.0
    # Per-exchange breakdown for diagnostics
    exchange_mids: dict[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.n_sources >= 1 and self.mid > 0

    @property
    def is_high_quality(self) -> bool:
        """True if 3+ sources with tight spread."""
        return self.n_sources >= 3 and self.spread_bps < 10.0  # < 10 bps


class LWBAEngine:
    """Replicate Chainlink's LWBA shadow price algorithm.

    Usage:
        engine = LWBAEngine()
        books = {
            "binance": OrderBookSnapshot(...),
            "okx":     OrderBookSnapshot(...),
            "coinbase": OrderBookSnapshot(...),
        }
        result = engine.aggregate(books)
        shadow_price = result.mid  # Use this instead of spot price
    """

    def __init__(self, max_levels: int = 20, stale_threshold_s: float = 10.0):
        """
        Args:
            max_levels: max orderbook depth levels to use per side
            stale_threshold_s: discard books older than this
        """
        self._max_levels = max_levels
        self._stale_threshold = stale_threshold_s
        # History for diagnostics
        self._last_result: dict[str, LWBAResult] = {}  # per asset

    def compute_lwba(self, book: OrderBookSnapshot) -> tuple[float, float]:
        """Compute LWBA Bid and Ask for a single exchange.

        LWBA_Bid = Σ(qty_i × price_i) / Σ(qty_i)
        LWBA_Ask = Σ(qty_j × price_j) / Σ(qty_j)

        Returns: (lwba_bid, lwba_ask)
        """
        bids = book.bids[:self._max_levels]
        asks = book.asks[:self._max_levels]

        if not bids or not asks:
            return 0.0, 0.0

        # Bid side: volume-weighted average price
        bid_notional = sum(qty * price for price, qty in bids)
        bid_volume = sum(qty for _, qty in bids)
        lwba_bid = bid_notional / bid_volume if bid_volume > 0 else 0.0

        # Ask side: volume-weighted average price
        ask_notional = sum(qty * price for price, qty in asks)
        ask_volume = sum(qty for _, qty in asks)
        lwba_ask = ask_notional / ask_volume if ask_volume > 0 else 0.0

        return lwba_bid, lwba_ask

    def aggregate(self, books: dict[str, OrderBookSnapshot],
                  asset: str = "") -> LWBAResult:
        """Multi-exchange LWBA aggregation — median of independent LWBA.

        Replicates Chainlink DON consensus:
        1. Each exchange independently computes LWBA Bid/Ask
        2. Final Bid = Median(all LWBA Bids)
        3. Final Ask = Median(all LWBA Asks)
        4. Mid = (Bid + Ask) / 2  ← settlement shadow price

        Args:
            books: {exchange_name: OrderBookSnapshot}
            asset: asset name for logging

        Returns: LWBAResult with aggregated shadow price
        """
        now = time.time()
        bids = []
        asks = []
        exchange_mids = {}

        for exchange, book in books.items():
            if not book.is_valid:
                continue
            # Skip stale books
            if book.timestamp > 0 and (now - book.timestamp) > self._stale_threshold:
                logger.debug(
                    f"LWBA: skipping stale {exchange} book "
                    f"(age={now - book.timestamp:.1f}s)"
                )
                continue

            lwba_bid, lwba_ask = self.compute_lwba(book)
            if lwba_bid > 0 and lwba_ask > 0 and lwba_ask >= lwba_bid:
                bids.append(lwba_bid)
                asks.append(lwba_ask)
                exchange_mids[exchange] = (lwba_bid + lwba_ask) / 2

        if not bids:
            return LWBAResult(timestamp=now)

        # Median aggregation (matches Chainlink DON consensus)
        final_bid = statistics.median(bids)
        final_ask = statistics.median(asks)
        final_mid = (final_bid + final_ask) / 2
        spread = final_ask - final_bid
        spread_bps = (spread / final_mid * 10000) if final_mid > 0 else 0.0

        result = LWBAResult(
            bid=round(final_bid, 8),
            ask=round(final_ask, 8),
            mid=round(final_mid, 8),
            spread=round(spread, 8),
            spread_bps=round(spread_bps, 2),
            n_sources=len(bids),
            timestamp=now,
            exchange_mids=exchange_mids,
        )

        self._last_result[asset] = result
        return result

    def get_last_result(self, asset: str) -> LWBAResult | None:
        """Get the most recent LWBA result for an asset."""
        return self._last_result.get(asset)
