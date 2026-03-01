"""
Strategy: Directional Sniper — T-10s late-entry maker for 5-min crypto markets.

Replaces CryptoMM (continuous quoting) and OracleArb (T-10s direction).
Merges their best features into a single strategy that avoids adverse selection.

How it works:
  1. T-300s to T-10s: OBSERVATION — watch CEX price vs strike, track stability
  2. T-10s to T-5s:  COMMITMENT — if CEX clearly above strike, BUY Up @ $0.60-0.65
  3. T-5s to T-0s:   BLACKOUT — too late for maker fills
  4. Settlement:     Up wins → $1.00 payout (+$0.35-0.40 profit); Down → -$0.60

Why this works:
  - Avoids adverse selection: no orders resting during price moves
  - Up-only: only buy when CEX > strike (directional conviction)
  - 0% maker fee + daily USDC rebate
  - maker_price $0.60 needs only 60% accuracy to break even
  - T-10s CEX signal is ~75% accurate → substantial positive EV

Key parameter: MAKER_PRICE
  - $0.90-0.95: needs 90%+ accuracy (too risky, old OracleArb approach)
  - $0.60-0.65: needs 60% accuracy (matches our 74.6% observed win rate)
  - Lower price = lower fill rate but higher edge per fill
"""
import logging
import math
import time
from typing import Optional

from .base import BaseStrategy, Signal, Side, Outcome
from ..config import Config
from ..market_scanner import MarketInfo

logger = logging.getLogger(__name__)


class DirectionalSniper(BaseStrategy):
    """
    T-10s Late-Entry Directional Maker Strategy.

    Only trades in the last 10 seconds of each 5-minute round.
    Only buys Up tokens when CEX price > strike (dynamic, set at round start).
    Uses maker orders at $0.60-0.65 for 0% fee + rebate.
    """

    def __init__(self, config: Config):
        super().__init__(config, name="Sniper")

        # CEX price and strike (from Brain predictions + dynamic strike)
        self._cex_price: float = 0.0
        self._strike_price: float = 0.0
        self._volatility: float = 0.01

        # Round tracking
        self._signal_sent: bool = False
        self._direction_samples: list[str] = []  # track direction stability
        self._last_sample_time: float = 0
        self._reversal_count: int = 0

        # PnL tracking (Up-only inventory)
        self._buy_cost: float = 0.0
        self._shares_bought: float = 0.0

    # --- External setters (called by vps_runner) ---

    def set_cex_price(self, price: float):
        self._cex_price = price

    def set_strike_price(self, strike: float):
        self._strike_price = strike

    def set_volatility(self, vol: float):
        self._volatility = max(0.001, vol)

    # --- Core logic ---

    def _get_direction_confidence(self) -> tuple[str, float]:
        """
        Determine direction and confidence from CEX vs strike.

        Returns ("Up"/"Down"/"Unknown", confidence 0.0-1.0).
        """
        if self._cex_price <= 0 or self._strike_price <= 0:
            return "Unknown", 0.0

        distance = (self._cex_price - self._strike_price) / self._strike_price
        abs_distance = abs(distance)

        if abs_distance < 0.00005:  # < 0.005% — effectively at strike
            return "Unknown", 0.50

        direction = "Up" if distance > 0 else "Down"

        # Confidence curve: distance → probability
        if abs_distance > 0.002:       # > 0.2%
            confidence = 0.85 + min(abs_distance * 20, 0.10)  # 85-95%
        elif abs_distance > 0.001:     # > 0.1%
            confidence = 0.75 + (abs_distance - 0.001) * 100  # 75-85%
        elif abs_distance > 0.0005:    # > 0.05%
            confidence = 0.65 + (abs_distance - 0.0005) * 200  # 65-75%
        else:
            confidence = 0.55 + abs_distance * 200             # 55-65%

        return direction, min(confidence, 0.95)

    def _calculate_maker_price(self, confidence: float) -> float:
        """
        Calculate optimal maker bid price.

        Lower price = safer (need lower win rate to break even)
        But also lower fill probability from paper sim.

        Break-even: maker_price = win_rate
        We target $0.55-0.65 depending on confidence.
        """
        sc = self.config.strategy

        # Base: use configured sniper_base_price (default $0.60)
        base = getattr(sc, 'sniper_base_price', 0.60)

        # Adjust by confidence: higher confidence → can bid higher
        if confidence >= 0.90:
            price = base + 0.05  # $0.65
        elif confidence >= 0.80:
            price = base + 0.03  # $0.63
        elif confidence >= 0.70:
            price = base         # $0.60
        else:
            price = base - 0.05  # $0.55

        return round(max(0.50, min(0.70, price)), 2)

    async def on_market_update(self, market: MarketInfo) -> list[Signal]:
        """
        Main strategy loop — only fires in T-10s to T-5s window.
        """
        if self.should_pause():
            return []

        t = market.seconds_remaining

        # --- BLACKOUT (T < 5s): too late for maker fills ---
        if t < 5:
            return []

        # --- OBSERVATION (T > 10s): just track direction stability ---
        if t > 10:
            # Sample direction every 2 seconds for stability tracking
            now = time.time()
            if now - self._last_sample_time > 2.0:
                self._last_sample_time = now
                direction, _ = self._get_direction_confidence()
                if direction in ("Up", "Down"):
                    if self._direction_samples and self._direction_samples[-1] != direction:
                        self._reversal_count += 1
                    self._direction_samples.append(direction)
            return []

        # --- COMMITMENT (T-10s to T-5s): fire if confident ---
        if self._signal_sent:
            return []

        direction, confidence = self._get_direction_confidence()

        # UP-ONLY: we only buy when direction is Up
        if direction != "Up":
            self._signal_sent = True  # don't retry this round
            self._logger.debug(
                f"Skip: direction={direction} conf={confidence:.1%} | "
                f"cex=${self._cex_price:.2f} strike=${self._strike_price:.2f}"
            )
            return []

        # Stability check: too many reversals = unstable market
        if self._reversal_count > 3:
            self._signal_sent = True
            self._logger.info(
                f"Skip: {self._reversal_count} reversals (unstable) | T-{t:.0f}s"
            )
            return []

        # Confidence threshold
        min_confidence = getattr(self.config.strategy, 'sniper_min_confidence', 0.65)
        if confidence < min_confidence:
            self._logger.debug(
                f"Skip: confidence {confidence:.1%} < {min_confidence:.1%}"
            )
            return []

        # Calculate maker price
        maker_price = self._calculate_maker_price(confidence)

        # Order size
        order_size = self.config.strategy.order_size_usd

        signal = Signal(
            market=market,
            outcome=Outcome.UP,
            side=Side.BUY,
            price=maker_price,
            size_usd=order_size,
            confidence=confidence,
            reason=(
                f"Sniper: Up GTC@${maker_price:.2f} | "
                f"conf={confidence:.1%} | "
                f"cex=${self._cex_price:.2f} vs strike=${self._strike_price:.2f} | "
                f"reversals={self._reversal_count} | T-{t:.0f}s"
            ),
            order_type="GTC",
        )

        self._signal_sent = True
        self._logger.info(
            f"🎯 SIGNAL: Up GTC@${maker_price:.2f} × ${order_size:.0f} | "
            f"conf={confidence:.1%} | T-{t:.0f}s | "
            f"cex=${self._cex_price:.2f} vs strike=${self._strike_price:.2f}"
        )

        return [signal]

    # --- Lifecycle ---

    async def on_round_start(self, market: MarketInfo):
        await super().on_round_start(market)
        self._signal_sent = False
        self._direction_samples = []
        self._reversal_count = 0
        self._last_sample_time = 0
        self._buy_cost = 0.0
        self._shares_bought = 0.0

        if market.strike_price > 0:
            self._strike_price = market.strike_price

    async def on_fill(self, signal: Signal, fill_price: float, fill_size: float):
        await super().on_fill(signal, fill_price, fill_size)
        self._buy_cost += fill_price * fill_size
        self._shares_bought += fill_size

    async def on_round_end(self, market: MarketInfo, settled_outcome: Optional[str] = None):
        """
        Calculate PnL. Simple: all shares are unsold at settlement.
        Up wins → $1.00/share; Down wins → $0.00/share.
        """
        if self._shares_bought > 0:
            avg_cost = self._buy_cost / self._shares_bought
            if settled_outcome == "Up":
                payout = self._shares_bought * 1.0
            else:
                payout = 0.0
            pnl = payout - self._buy_cost
            outcome_emoji = "✅" if pnl > 0 else "❌"
        else:
            pnl = 0.0
            avg_cost = 0.0
            outcome_emoji = "⏭️"

        self.state.total_pnl_usd += pnl
        self.state.daily_pnl_usd += pnl

        if pnl > 0:
            self.state.rounds_won += 1
            self.state.consecutive_losses = 0
        elif pnl < 0:
            self.state.consecutive_losses += 1

        self.state.rounds_traded += 1
        self.state.reset_round()

        if self._shares_bought > 0:
            self._logger.info(
                f"{outcome_emoji} Round PnL: ${pnl:+.2f} | "
                f"settled={settled_outcome} | "
                f"{self._shares_bought:.1f}sh @${avg_cost:.3f} | "
                f"Daily: ${self.state.daily_pnl_usd:+.2f} | "
                f"Win rate: {self.state.win_rate:.1%}"
            )
        else:
            self._logger.debug(f"Round skip (no fills) | settled={settled_outcome}")
