"""
Realistic Paper Trading Simulator for Polymarket 5-minute crypto markets.

Models 5 dimensions of real trading:
  1. Dynamic taker fees  — exact Polymarket formula: C × 0.25 × (p(1-p))²
  2. Latency             — API round-trip + price drift during delay
  3. Order book depth    — $5K-$15K per side, slippage on large orders
  4. Fill probability    — GTC limits fill probabilistically, not instantly
  5. Rate limiting       — 60 orders/min cap

Research basis (Feb 2026):
  - Taker fee: feeRate=0.25, exponent=2, max effective ~1.56% at p=0.50
  - Maker fee: 0% + rebate
  - 500ms taker delay removed Feb 18 2026
  - Order book depth: $5K-$15K per side during active sessions
  - Rate limit: 60 orders/min/API key
"""
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ─── Fee Model ────────────────────────────────────────────────────────────────

class FeeType(str, Enum):
    MAKER = "maker"
    TAKER = "taker"


@dataclass
class FeeResult:
    """Result of fee calculation."""
    fee_usd: float
    fee_rate_effective: float  # as a fraction (0.0156 = 1.56%)
    fee_type: FeeType


def calculate_taker_fee(shares: float, price: float,
                        fee_rate: float = 0.25,
                        exponent: int = 2) -> FeeResult:
    """
    Polymarket dynamic taker fee for 5m/15m crypto markets.

    Formula: fee = C × feeRate × (p × (1 - p))^exponent

    Where:
      C = number of shares
      feeRate = 0.25 (for crypto markets)
      p = price per share (0.00 to 1.00)
      exponent = 2 (for crypto markets)

    Max effective rate: at p=0.50 → 0.25 × 0.0625 = 0.015625 (1.5625%)

    Examples:
      100 shares @ $0.50 → fee = 100 × 0.25 × (0.25)^2 = $0.39
      100 shares @ $0.90 → fee = 100 × 0.25 × (0.09)^2 = $0.020
      100 shares @ $0.10 → fee = 100 × 0.25 × (0.09)^2 = $0.020
    """
    if price <= 0 or price >= 1:
        return FeeResult(fee_usd=0.0, fee_rate_effective=0.0, fee_type=FeeType.TAKER)

    p_factor = (price * (1.0 - price)) ** exponent
    fee_usd = shares * fee_rate * p_factor
    # Effective rate relative to notional value (shares × price)
    notional = shares * price
    effective_rate = fee_usd / notional if notional > 0 else 0.0

    return FeeResult(
        fee_usd=round(fee_usd, 6),
        fee_rate_effective=round(effective_rate, 6),
        fee_type=FeeType.TAKER,
    )


def calculate_maker_fee(shares: float, price: float) -> FeeResult:
    """Maker orders are free on Polymarket (0% fee + potential rebate)."""
    return FeeResult(fee_usd=0.0, fee_rate_effective=0.0, fee_type=FeeType.MAKER)


# ─── Latency Model ───────────────────────────────────────────────────────────

@dataclass
class LatencyConfig:
    """Latency model parameters."""
    # API round-trip time (milliseconds)
    mean_ms: float = 80.0      # VPS in same region as CLOB
    stddev_ms: float = 30.0    # jitter
    # Price drift volatility per second (as fraction of price)
    tick_volatility: float = 0.0005  # 0.05% per second typical for BTC
    # Minimum latency floor (ms)
    min_ms: float = 20.0
    # Maximum latency cap (ms) — network timeout
    max_ms: float = 500.0


def simulate_latency(config: LatencyConfig) -> float:
    """
    Simulate API round-trip latency.
    Returns latency in seconds.
    """
    latency_ms = random.gauss(config.mean_ms, config.stddev_ms)
    latency_ms = max(config.min_ms, min(config.max_ms, latency_ms))
    return latency_ms / 1000.0


def simulate_price_drift(price: float, latency_s: float,
                         tick_volatility: float = 0.0005) -> float:
    """
    Simulate price movement during order transit latency.

    During the time our order is in transit, the true price can move.
    We model this as a random walk: ΔP ~ Normal(0, σ × √t)

    Returns the execution price (may be worse than submitted price).
    """
    if latency_s <= 0:
        return price
    sigma = price * tick_volatility * math.sqrt(latency_s)
    drift = random.gauss(0, sigma)
    return round(price + drift, 4)


# ─── Order Book Model ─────────────────────────────────────────────────────────

@dataclass
class OrderBookConfig:
    """Simulated order book parameters."""
    # Total depth per side (USD)
    total_depth_usd: float = 10000.0   # $10K per side (mid-range estimate)
    # Depth concentration: fraction of depth within 1 tick of midpoint
    near_concentration: float = 0.3     # 30% depth within 1 tick
    # Number of price levels modeled
    num_levels: int = 10
    # Tick size
    tick_size: float = 0.01


def simulate_slippage(order_size_usd: float, price: float,
                      config: OrderBookConfig) -> tuple[float, float]:
    """
    Simulate slippage from eating through the order book.

    Returns (execution_price, fill_ratio).

    For small orders ($10-$50) in $10K depth → negligible slippage.
    For large orders ($1000+) → significant slippage.

    Model: exponential depth decay from midpoint.
    Level i has depth = total × concentration × exp(-i × decay)
    """
    if order_size_usd <= 0 or price <= 0:
        return price, 1.0

    tick = config.tick_size
    remaining = order_size_usd
    weighted_price_sum = 0.0
    total_filled = 0.0

    # Decay factor: depth decreases exponentially away from midpoint
    decay = 0.5  # each level has ~60% of previous level's depth

    for i in range(config.num_levels):
        # Depth at this level (USD)
        if i == 0:
            level_depth = config.total_depth_usd * config.near_concentration
        else:
            level_depth = (config.total_depth_usd * (1 - config.near_concentration)
                           * (1 - decay) * decay ** (i - 1))

        # Price at this level (worse by i ticks)
        level_price = price + (i * tick)  # buying: higher price = worse
        level_price = min(0.99, level_price)  # cap at 99¢

        # Fill from this level
        fill_from_level = min(remaining, level_depth)
        weighted_price_sum += fill_from_level * level_price
        total_filled += fill_from_level
        remaining -= fill_from_level

        if remaining <= 0:
            break

    if total_filled == 0:
        return price, 0.0

    avg_execution_price = weighted_price_sum / total_filled
    fill_ratio = total_filled / order_size_usd

    return round(avg_execution_price, 4), min(1.0, fill_ratio)


# ─── Fill Probability Model ──────────────────────────────────────────────────

def estimate_gtc_fill_probability(
    limit_price: float,
    midpoint: float,
    seconds_resting: float,
    is_buy: bool,
    volatility: float = 0.01,
) -> float:
    """
    Estimate probability that a GTC limit order fills.

    Factors:
      - Distance from midpoint (closer = more likely)
      - Time resting (longer = more likely)
      - Market volatility (higher vol = more likely to fill)

    For buy orders: fills if market drops to our bid
    For sell orders: fills if market rises to our ask

    Returns probability [0, 1].
    """
    if midpoint <= 0:
        return 0.0

    # Distance from midpoint (in ticks, positive = favorable)
    if is_buy:
        # Buy: our bid is below midpoint, distance = how much below
        distance = midpoint - limit_price
    else:
        # Sell: our ask is above midpoint, distance = how much above
        distance = limit_price - midpoint

    if distance < 0:
        # Price is crossing the midpoint (aggressive) — very likely to fill
        return 0.95

    # Normalize distance by volatility
    distance_pct = distance / midpoint
    vol_adjusted = distance_pct / max(0.001, volatility)

    # Base probability: exponential decay with distance
    # At midpoint (distance=0): ~80% fill in 1 second
    # At 1 tick out: ~50%
    # At 2+ ticks out: drops sharply
    base_prob = math.exp(-vol_adjusted * 5.0)

    # Time factor: more resting time = higher probability
    # Saturates at ~10 seconds
    time_factor = 1.0 - math.exp(-seconds_resting / 5.0)

    probability = base_prob * time_factor * 0.85  # 85% cap (never 100% certain)

    return max(0.0, min(0.85, probability))


# ─── Rate Limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Enforces Polymarket's 60 orders/min rate limit.
    """

    def __init__(self, max_per_minute: int = 60):
        self.max_per_minute = max_per_minute
        self._timestamps: list[float] = []

    def check(self) -> bool:
        """Check if we can place an order. Returns True if allowed."""
        now = time.time()
        cutoff = now - 60.0

        # Prune old timestamps
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        if len(self._timestamps) >= self.max_per_minute:
            return False

        self._timestamps.append(now)
        return True

    def remaining(self) -> int:
        """How many orders can we still place this minute."""
        now = time.time()
        cutoff = now - 60.0
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        return max(0, self.max_per_minute - len(self._timestamps))


# ─── Paper Simulator (integrates all models) ──────────────────────────────────

@dataclass
class SimulationResult:
    """Result of a simulated order execution."""
    filled: bool
    fill_price: float = 0.0
    fill_size: float = 0.0       # in shares
    fee_usd: float = 0.0
    slippage_ticks: float = 0.0  # how many ticks worse than intended
    latency_ms: float = 0.0
    reject_reason: str = ""


class PaperSimulator:
    """
    Realistic paper trading simulator.

    Replaces the naive immediate-fill logic with a multi-factor model
    that accounts for real Polymarket trading conditions.
    """

    def __init__(
        self,
        latency_config: LatencyConfig | None = None,
        book_config: OrderBookConfig | None = None,
        fee_rate: float = 0.25,
        fee_exponent: int = 2,
        rate_limit: int = 60,
    ):
        self.latency = latency_config or LatencyConfig()
        self.book = book_config or OrderBookConfig()
        self.fee_rate = fee_rate
        self.fee_exponent = fee_exponent
        self.rate_limiter = RateLimiter(rate_limit)

        # Stats
        self.total_fees_paid: float = 0.0
        self.total_slippage_usd: float = 0.0
        self.orders_rejected: int = 0
        self.orders_rate_limited: int = 0

    def simulate_fok_order(
        self,
        price: float,
        size_usd: float,
        midpoint: float,
        is_buy: bool,
    ) -> SimulationResult:
        """
        Simulate a Fill-or-Kill (taker) order.

        FOK orders cross the spread and take liquidity.
        They incur taker fees and potential slippage.
        """
        # Rate limit check
        if not self.rate_limiter.check():
            self.orders_rate_limited += 1
            return SimulationResult(
                filled=False,
                reject_reason=f"Rate limited ({self.rate_limiter.max_per_minute}/min)",
            )

        # Simulate latency
        latency_s = simulate_latency(self.latency)
        latency_ms = latency_s * 1000

        # Price may have drifted during latency
        drifted_price = simulate_price_drift(
            price, latency_s, self.latency.tick_volatility
        )

        # Check if drifted price is still favorable
        if is_buy and drifted_price > price + 0.03:
            # Price moved too far against us
            self.orders_rejected += 1
            return SimulationResult(
                filled=False,
                latency_ms=latency_ms,
                reject_reason=f"Price moved against: {price:.3f} → {drifted_price:.3f}",
            )
        if not is_buy and drifted_price < price - 0.03:
            self.orders_rejected += 1
            return SimulationResult(
                filled=False,
                latency_ms=latency_ms,
                reject_reason=f"Price moved against: {price:.3f} → {drifted_price:.3f}",
            )

        # Simulate slippage from order book depth
        exec_price, fill_ratio = simulate_slippage(
            size_usd, drifted_price, self.book
        )

        if fill_ratio < 0.5:
            # FOK: reject if can't fill most of the order
            self.orders_rejected += 1
            return SimulationResult(
                filled=False,
                latency_ms=latency_ms,
                reject_reason=f"Insufficient depth: fill_ratio={fill_ratio:.1%}",
            )

        # Calculate shares and fee
        shares = (size_usd * fill_ratio) / exec_price if exec_price > 0 else 0
        fee_result = calculate_taker_fee(
            shares, exec_price, self.fee_rate, self.fee_exponent
        )

        slippage_ticks = abs(exec_price - price) / max(0.01, self.book.tick_size)

        # Track stats
        self.total_fees_paid += fee_result.fee_usd
        self.total_slippage_usd += abs(exec_price - price) * shares

        return SimulationResult(
            filled=True,
            fill_price=exec_price,
            fill_size=shares,
            fee_usd=fee_result.fee_usd,
            slippage_ticks=slippage_ticks,
            latency_ms=latency_ms,
        )

    def simulate_gtc_order(
        self,
        price: float,
        size_usd: float,
        midpoint: float,
        is_buy: bool,
        seconds_resting: float = 1.0,
        volatility: float = 0.01,
    ) -> SimulationResult:
        """
        Simulate a Good-Till-Canceled (maker) limit order.

        GTC orders rest on the book. They may or may not fill
        depending on market movement. Maker orders pay 0% fee.
        """
        # Rate limit check
        if not self.rate_limiter.check():
            self.orders_rate_limited += 1
            return SimulationResult(
                filled=False,
                reject_reason=f"Rate limited ({self.rate_limiter.max_per_minute}/min)",
            )

        # Estimate fill probability
        fill_prob = estimate_gtc_fill_probability(
            limit_price=price,
            midpoint=midpoint,
            seconds_resting=seconds_resting,
            is_buy=is_buy,
            volatility=volatility,
        )

        # Roll the dice
        if random.random() > fill_prob:
            return SimulationResult(
                filled=False,
                reject_reason=f"GTC not filled (prob={fill_prob:.1%})",
            )

        # Filled at our limit price (maker = no slippage beyond our price)
        shares = size_usd / price if price > 0 else 0

        # Maker fee = 0
        fee_result = calculate_maker_fee(shares, price)

        return SimulationResult(
            filled=True,
            fill_price=price,
            fill_size=shares,
            fee_usd=fee_result.fee_usd,
            slippage_ticks=0.0,
            latency_ms=simulate_latency(self.latency) * 1000,
        )

    def get_stats(self) -> dict:
        """Return simulation statistics."""
        return {
            "total_fees_paid": round(self.total_fees_paid, 4),
            "total_slippage_usd": round(self.total_slippage_usd, 4),
            "orders_rejected": self.orders_rejected,
            "orders_rate_limited": self.orders_rate_limited,
            "rate_limit_remaining": self.rate_limiter.remaining(),
        }
