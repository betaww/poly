"""
CLOB Order Executor — handles order lifecycle using py-clob-client.

Wraps the official Polymarket Python SDK for:
  - Order creation (limit, market, FOK)
  - Cancel / cancel-all
  - Order status polling
  - Balance queries

Paper mode: logs orders without submitting to CLOB.
E3: Paper sim uses real CLOB depth when available.
E5: Live mode uses WebSocket fill tracker when available.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import Config
from ..strategies.base import Signal, Side

logger = logging.getLogger(__name__)


@dataclass
class OrderRecord:
    """Local order tracking."""
    order_id: str
    signal: Signal
    status: str = "pending"    # pending, live, matched, canceled, failed
    fill_price: float = 0.0
    fill_size: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.status in ("matched", "canceled", "failed")


class OrderExecutor:
    """
    Executes orders on Polymarket CLOB.

    In paper mode, uses PaperSimulator for realistic fills.
    In live mode, uses py-clob-client for real order submission.
    """

    def __init__(self, config: Config):
        self.config = config
        self.is_paper = config.mode == "paper"
        self._orders: dict[str, OrderRecord] = {}
        self._order_counter = 0
        self._clob_client = None

        # Realistic paper trading simulator
        self._simulator = None
        if self.is_paper:
            from .paper_simulator import PaperSimulator, LatencyConfig, OrderBookConfig
            psc = config.paper_sim
            latency = LatencyConfig(
                mean_ms=psc.latency_mean_ms if config.node == "vps" else psc.latency_mean_ms * 1.8,
                stddev_ms=psc.latency_stddev_ms if config.node == "vps" else psc.latency_stddev_ms * 1.5,
                min_ms=psc.latency_min_ms,
                max_ms=psc.latency_max_ms,
            )
            book = OrderBookConfig(
                total_depth_usd=psc.book_depth_usd,
                near_concentration=psc.book_near_concentration,
            )
            self._simulator = PaperSimulator(
                latency_config=latency,
                book_config=book,
                rate_limit=psc.rate_limit_per_min,
            )
        else:
            self._init_clob_client()

        # E3: Optional CLOB book feed reference for real depth data
        self._clob_feed = None
        # E5: Optional live fill tracker
        self._live_fill_tracker = None

    def _init_clob_client(self):
        """Initialize the official py-clob-client."""
        try:
            from py_clob_client.client import ClobClient

            self._clob_client = ClobClient(
                self.config.api.clob_host,
                key=self.config.wallet.private_key,
                chain_id=self.config.wallet.chain_id,
                signature_type=self.config.wallet.signature_type,
                funder=self.config.wallet.funder_address or None,
            )
            # Derive or set API credentials
            if self.config.api.api_key:
                self._clob_client.set_api_creds({
                    "apiKey": self.config.api.api_key,
                    "secret": self.config.api.api_secret,
                    "passphrase": self.config.api.api_passphrase,
                })
            else:
                creds = self._clob_client.create_or_derive_api_creds()
                self._clob_client.set_api_creds(creds)

            logger.info("CLOB client initialized (LIVE mode)")

        except ImportError:
            logger.error("py-clob-client not installed. Run: pip install py-clob-client")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize CLOB client: {e}")
            raise

    async def place_order(self, signal: Signal) -> OrderRecord:
        """Place an order based on a strategy signal."""
        self._order_counter += 1
        order_id = f"{'paper' if self.is_paper else 'live'}_{self._order_counter}_{int(time.time())}"

        record = OrderRecord(
            order_id=order_id,
            signal=signal,
            created_at=time.time(),
        )

        if self.is_paper:
            return self._paper_fill(record)
        else:
            return await self._live_place(record)

    def _paper_fill(self, record: OrderRecord) -> OrderRecord:
        """Simulate fill using realistic PaperSimulator."""
        signal = record.signal
        # C4 FIX: Use the market's current midpoint which is now updated
        # in real-time from CLOB book feed (see vps_runner C3 fix).
        # Falls back to price_up if book feed is unavailable.
        midpoint = signal.market.midpoint
        is_buy = signal.side == Side.BUY

        # E3: Use real CLOB depth to dynamically adjust book model
        if self._clob_feed:
            token_id = signal.token_id
            book = self._clob_feed.get_book(token_id)
            if book and book.total_bid_depth_usd > 0:
                # Override parameterized depth with real CLOB depth
                real_depth = book.total_bid_depth_usd if is_buy else book.total_ask_depth_usd
                if real_depth > 0:
                    self._simulator.book.total_depth_usd = real_depth
                    # Also use real midpoint
                    if book.midpoint > 0:
                        midpoint = book.midpoint

        if signal.order_type == "FOK":
            # Taker order: slippage + fees + latency
            result = self._simulator.simulate_fok_order(
                price=signal.price,
                size_usd=signal.size_usd,
                midpoint=midpoint,
                is_buy=is_buy,
            )
        else:
            # Maker order: probabilistic fill, 0% fee
            # M1 FIX: For DirectionalSniper, the GTC order rests from
            # signal time until blackout window (T-5s) or round end.
            # Use actual seconds_remaining as upper bound, minus 5s blackout.
            remaining = signal.market.seconds_remaining
            resting_time = max(1.0, min(remaining - 5.0, 10.0))  # 1-10s realistic range
            # Use config volatility if available, fallback to 5%
            vol = getattr(self.config, 'paper_sim', None)
            default_vol = vol.default_volatility if vol else 0.05
            result = self._simulator.simulate_gtc_order(
                price=signal.price,
                size_usd=signal.size_usd,
                midpoint=midpoint,
                is_buy=is_buy,
                seconds_resting=resting_time,
                volatility=default_vol,
            )

        if result.filled:
            record.status = "matched"
            record.fill_price = result.fill_price
            record.fill_size = result.fill_size
            fee_str = f" fee=${result.fee_usd:.3f}" if result.fee_usd > 0 else ""
            slip_str = f" slip={result.slippage_ticks:.1f}t" if result.slippage_ticks > 0.1 else ""
            logger.info(
                f"[PAPER] {signal.side.value} {signal.outcome.value} @ {result.fill_price:.3f} "
                f"x {result.fill_size:.1f} shares (${signal.size_usd:.2f})"
                f"{fee_str}{slip_str} | "
                f"market={signal.market.asset.upper()} {signal.market.slug[-10:]}"
            )
        else:
            record.status = "failed"
            logger.debug(
                f"[PAPER] {signal.side.value} {signal.outcome.value} REJECTED: {result.reject_reason}"
            )

        record.updated_at = time.time()
        self._orders[record.order_id] = record

        # Prune old entries to prevent memory growth
        self._prune_orders()

        return record

    def _prune_orders(self, max_size: int = 500):
        """Remove oldest terminal orders to prevent unbounded memory growth."""
        if len(self._orders) > max_size:
            # Keep only the most recent orders
            sorted_orders = sorted(
                self._orders.items(),
                key=lambda x: x[1].updated_at,
                reverse=True,
            )
            self._orders = dict(sorted_orders[:max_size])

    async def _live_place(self, record: OrderRecord) -> OrderRecord:
        """Place order via py-clob-client."""
        signal = record.signal
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            side = BUY if signal.side == Side.BUY else SELL

            # Determine order type
            if signal.order_type == "FOK":
                # Market order (fill-or-kill)
                from py_clob_client.clob_types import MarketOrderArgs
                order_args = MarketOrderArgs(
                    token_id=signal.token_id,
                    amount=signal.size_usd,
                    side=side,
                    order_type=OrderType.FOK,
                )
                signed_order = self._clob_client.create_market_order(order_args)
                resp = self._clob_client.post_order(signed_order, OrderType.FOK)
            else:
                # Limit order (GTC)
                order_args = OrderArgs(
                    token_id=signal.token_id,
                    price=signal.price,
                    size=signal.size_usd / signal.price,  # shares
                    side=side,
                )
                tick_size = signal.market.tick_size
                neg_risk = signal.market.neg_risk
                # C7 FIX: pass tick_size and neg_risk to create_order
                # for correct EIP-712 signature. Without these, live
                # orders may be rejected with "invalid signature".
                signed_order = self._clob_client.create_order(
                    order_args,
                    options={"tick_size": tick_size, "neg_risk": neg_risk},
                )
                resp = self._clob_client.post_order(
                    signed_order,
                    OrderType.GTC,
                    tick_size=tick_size,
                    neg_risk=neg_risk,
                )

            # Parse response
            if resp and isinstance(resp, dict):
                record.order_id = resp.get("orderID", record.order_id)
                record.status = "live"
            else:
                record.status = "failed"

            record.updated_at = time.time()
            self._orders[record.order_id] = record

            logger.info(
                f"[LIVE] {signal.side.value} {signal.outcome.value} @ {signal.price:.3f} "
                f"| order_id={record.order_id} | status={record.status}"
            )
            return record

        except Exception as e:
            record.status = "failed"
            record.updated_at = time.time()
            self._orders[record.order_id] = record
            logger.error(f"Order placement failed: {e}", exc_info=True)
            return record

    async def poll_live_fills(self) -> list[OrderRecord]:
        """CRIT2 FIX: Poll live orders for fills.

        In production, this should be replaced by User channel WebSocket
        (wss://ws-subscriptions-clob.polymarket.com/ws/user) for instant
        fill notifications. Polling is a reliable fallback.
        """
        filled = []
        if self.is_paper or not self._clob_client:
            return filled

        for oid, record in list(self._orders.items()):
            if record.status != "live":
                continue
            try:
                order = self._clob_client.get_order(oid)
                if order and isinstance(order, dict):
                    status = order.get("status", "")
                    if status == "matched":
                        record.status = "matched"
                        record.fill_price = float(order.get("price", record.signal.price))
                        record.fill_size = float(order.get("size_matched", 0))
                        record.updated_at = time.time()
                        filled.append(record)
                        logger.info(
                            f"[LIVE] Fill detected: {oid} @ {record.fill_price:.3f} "
                            f"x {record.fill_size:.1f}"
                        )
                    elif status in ("canceled", "expired"):
                        record.status = "canceled"
                        record.updated_at = time.time()
            except Exception as e:
                logger.debug(f"Poll failed for {oid}: {e}")

        return filled

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        if self.is_paper:
            if order_id in self._orders:
                self._orders[order_id].status = "canceled"
                logger.info(f"[PAPER] Canceled order {order_id}")
                return True
            return False

        try:
            self._clob_client.cancel(order_id)
            if order_id in self._orders:
                self._orders[order_id].status = "canceled"
            logger.info(f"[LIVE] Canceled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed for {order_id}: {e}")
            return False

    async def cancel_all(self) -> int:
        """Cancel all active orders."""
        if self.is_paper:
            count = 0
            for oid, record in self._orders.items():
                if record.status == "live":
                    record.status = "canceled"
                    count += 1
            logger.info(f"[PAPER] Canceled {count} orders")
            return count

        try:
            if self._clob_client:
                self._clob_client.cancel_all()
            count = 0
            for record in self._orders.values():
                if record.status == "live":
                    record.status = "canceled"
                    count += 1
            logger.info(f"[LIVE] Canceled all orders ({count})")
            return count
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
            return 0

    async def get_balance(self) -> float:
        """Get USDC balance."""
        if self.is_paper:
            return 1000.0  # simulated balance

        try:
            if self._clob_client:
                balance = self._clob_client.get_balance()
                return float(balance) if balance else 0.0
            return 0.0
        except Exception as e:
            logger.error(f"Balance check failed: {e}")
            return 0.0

    def get_active_orders(self) -> list[OrderRecord]:
        """Return non-terminal orders."""
        return [r for r in self._orders.values() if not r.is_terminal]

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        return self._orders.get(order_id)
