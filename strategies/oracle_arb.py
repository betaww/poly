"""
Strategy 2: Oracle-Informed Maker Strategy (5-minute version).

Post-Feb 2026 Polymarket rules: taker fees are 1.56% at p=0.50,
but GTC maker orders have 0% fee + 20% taker fee rebate.

How it works:
  1. OBSERVATION (T > 15s): Build price context, no trading
  2. PREPARATION (T <= 15s): Start tracking direction, watch for reversals
  3. COMMITMENT (T-10s to T-5s): Place GTC MAKER orders if confident
  4. BLACKOUT (T < 5s): Too late for maker fills

Key post-Feb-2026 changes:
  - GTC (maker) instead of FOK (taker) → 0% fee vs 1.56%
  - T-10s entry instead of T-7s → backed by backtest showing T-10s
    determines 15-20% of 5-min candles
  - Maker rebate income on top of settlement profit
"""
import logging
import time
import math
from enum import Enum
from typing import Optional

from .base import BaseStrategy, Signal, Side, Outcome
from ..config import Config
from ..market_scanner import MarketInfo

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    OBSERVATION = "OBSERVATION"
    PREPARATION = "PREPARATION"
    COMMITMENT = "COMMITMENT"
    BLACKOUT = "BLACKOUT"


class OracleArbStrategy(BaseStrategy):
    """
    Oracle-Informed Maker Strategy — places GTC maker orders based on
    CEX price direction in the final 10 seconds of each round.

    Key post-Feb-2026 design:
      - 0% maker fee + daily USDC rebates
      - GTC limit orders at 0.90-0.95 based on confidence
      - Entry at T-10s backed by backtest data
    """

    def __init__(self, config: Config):
        super().__init__(config, name="OracleArb")

        # Synthetic oracle inputs
        self._cex_price: float = 0.0
        self._strike_price: float = 0.0
        self._volatility: float = 0.01  # from brain predictions

        # Tracking
        self._direction_history: list[str] = []  # "Up" or "Down"
        self._reversal_count: int = 0
        self._last_direction: Optional[str] = None
        self._signal_sent_this_round: bool = False
        self._round_cost: float = 0.0  # cost basis for this round

    def _get_phase(self, market: MarketInfo) -> Phase:
        """Determine current trading phase based on time remaining."""
        t = market.seconds_remaining
        sc = self.config.strategy
        # POST-FEB-2026 TIMING: shifted earlier for GTC maker orders
        # commitment_window_start=10, commitment_window_end=5
        # OBSERVATION: t > 15s      (too early)
        # PREPARATION: 10s < t <= 15s (tracking, no trading)
        # COMMITMENT:  5s < t <= 10s  (place GTC makers if confident)
        # BLACKOUT:    t <= 5s        (too late for maker fills)
        if t > 15:
            return Phase.OBSERVATION
        elif t > sc.commitment_window_start:  # t > 10
            return Phase.PREPARATION
        elif t > sc.commitment_window_end:    # t > 5
            return Phase.COMMITMENT
        else:
            return Phase.BLACKOUT

    def set_cex_price(self, price: float):
        self._cex_price = price

    def set_strike_price(self, strike: float):
        self._strike_price = strike

    def set_volatility(self, vol: float):
        """Update volatility from brain predictions."""
        self._volatility = max(0.001, vol)

    def _predict_direction(self) -> tuple[str, float]:
        """
        Predict settlement direction using synthetic oracle.
        Returns (direction, confidence).
        """
        if not self._cex_price or not self._strike_price:
            return "Unknown", 0.0

        distance = (self._cex_price - self._strike_price) / self._strike_price
        abs_distance = abs(distance)

        # BUG FIX: exact match = too close to call, not "Down"
        if abs_distance < 0.00001:  # < 0.001% — effectively at strike
            return "Unknown", 0.50

        # Direction
        direction = "Up" if distance > 0 else "Down"

        # Confidence based on distance magnitude
        # BUG FIX: ensure continuous confidence curve (no overflow)
        # Ranges: [0.001, ∞) → 95-99%, [0.0005, 0.001) → 90-95%,
        #         [0.0001, 0.0005) → 70-90%, [0, 0.0001) → 50-70%
        if abs_distance > 0.001:        # > 0.1% from strike
            confidence = 0.95 + min(abs_distance * 10, 0.04)  # 95-99%
        elif abs_distance > 0.0005:     # > 0.05%
            # Linear: 0.0005 → 0.90, 0.001 → 0.95
            confidence = 0.90 + (abs_distance - 0.0005) * 100  # 90-95%
        elif abs_distance > 0.0001:     # > 0.01%
            # Linear: 0.0001 → 0.70, 0.0005 → 0.90
            confidence = 0.70 + (abs_distance - 0.0001) * 500  # 70-90%
        else:                           # < 0.01% — low confidence
            confidence = 0.50 + abs_distance * 2000            # 50-70%

        return direction, min(confidence, 0.99)

    def _calculate_edge(self, direction: str, confidence: float, market: MarketInfo) -> float:
        """
        Calculate trading edge for GTC maker orders.

        Post-Feb 2026: GTC maker orders have 0% fee + rebate.
        Edge = P(win) - entry_price  (no taker fee deduction)
        """
        # For GTC maker, we set the price ourselves (typically 0.90-0.95)
        # Edge = confidence - entry_price (no fee for maker)
        maker_price = self._get_maker_price(confidence)
        edge = confidence - maker_price

        return edge

    def _get_maker_price(self, confidence: float) -> float:
        """
        Calculate optimal GTC maker order price.

        Higher confidence → can afford higher price (still profitable).
        Target: 0.90-0.95 range for high-confidence directions.
        """
        if confidence >= 0.98:
            return 0.95  # very confident, place near top
        elif confidence >= 0.95:
            return 0.92  # standard high-confidence
        elif confidence >= 0.90:
            return 0.88  # moderate confidence, want more margin
        else:
            return 0.85  # lower confidence, need bigger edge

    def _calculate_size(self, edge: float, confidence: float) -> float:
        """Kelly criterion position sizing (1/4 Kelly for safety)."""
        sc = self.config.strategy

        if edge <= 0:
            return 0.0

        # Full Kelly = edge / (odds - 1), simplified for binary
        # 1/4 Kelly for conservative sizing
        kelly_size = sc.kelly_fraction * edge * sc.max_position_usd

        # I5 FIX: volatility-aware sizing — reduce when vol > 2x normal
        if self._volatility > 0.02:  # > 2% (2x the 1% normal)
            vol_factor = 0.5  # halve size in high vol
            kelly_size *= vol_factor

        return max(sc.order_size_usd, min(kelly_size, sc.max_position_usd))

    async def on_round_start(self, market: MarketInfo):
        """Reset tracking for new round."""
        await super().on_round_start(market)
        self._direction_history = []
        self._reversal_count = 0
        self._last_direction = None
        self._signal_sent_this_round = False
        self._round_cost = 0.0

        # Use the strike price parsed from market question (e.g. "BTC above $84,500")
        if market.strike_price > 0:
            self._strike_price = market.strike_price
            self._logger.info(
                f"Round strike: ${self._strike_price:,.2f} | CEX: ${self._cex_price:,.2f}"
            )
        elif self._cex_price > 0:
            # Fallback: use CEX price at round start (less accurate)
            self._strike_price = self._cex_price
            self._logger.warning("Strike not parsed, using CEX price as fallback")

    async def on_market_update(self, market: MarketInfo) -> list[Signal]:
        """
        Late Commitment protocol — only trade in the last few seconds.
        """
        if self.should_pause():
            return []

        phase = self._get_phase(market)

        # --- OBSERVATION: just build context, no tracking ---
        if phase == Phase.OBSERVATION:
            return []

        direction, confidence = self._predict_direction()

        # Track direction changes (only in PREP/COMMIT to avoid unbounded list)
        if direction in ("Up", "Down"):
            if self._last_direction and direction != self._last_direction:
                self._reversal_count += 1
            self._last_direction = direction
            self._direction_history.append(direction)

        # --- PREPARATION: track but don't trade ---
        if phase == Phase.PREPARATION:
            if len(self._direction_history) % 10 == 0:
                self._logger.debug(
                    f"PREP: direction={direction} conf={confidence:.1%} "
                    f"reversals={self._reversal_count} T-{market.seconds_remaining:.0f}s"
                )
            return []

        # --- BLACKOUT: too late ---
        if phase == Phase.BLACKOUT:
            return []

        # --- COMMITMENT: execute if conditions met ---
        if self._signal_sent_this_round:
            return []  # One signal per round

        sc = self.config.strategy

        # Check confidence threshold
        if confidence < sc.oracle_confidence_threshold:
            self._logger.debug(
                f"COMMIT: confidence {confidence:.1%} < threshold {sc.oracle_confidence_threshold:.1%}"
            )
            return []

        # Check for excessive reversals (unstable market)
        if self._reversal_count > 2:
            self._logger.info(
                f"COMMIT: skipping — {self._reversal_count} reversals (unstable)"
            )
            return []

        # Calculate edge
        edge = self._calculate_edge(direction, confidence, market)

        if edge < sc.oracle_min_edge:
            self._logger.debug(
                f"COMMIT: edge {edge:.1%} < min {sc.oracle_min_edge:.1%}"
            )
            return []

        # Calculate position size
        size = self._calculate_size(edge, confidence)

        # Build signal — GTC MAKER order (0% fee + rebate)
        outcome = Outcome.UP if direction == "Up" else Outcome.DOWN
        maker_price = self._get_maker_price(confidence)

        signal = Signal(
            market=market,
            outcome=outcome,
            side=Side.BUY,
            price=maker_price,  # GTC limit price (0.90-0.95)
            size_usd=size,
            confidence=confidence,
            reason=(
                f"OracleArb: {direction} | conf={confidence:.1%} | edge={edge:.1%} | "
                f"cex={self._cex_price:.2f} vs strike={self._strike_price:.2f} | "
                f"maker@${maker_price:.2f} | T-{market.seconds_remaining:.0f}s"
            ),
            order_type="GTC",  # POST-FEB-2026: GTC maker (0% fee + rebate)
        )

        self._signal_sent_this_round = True
        self._logger.info(
            f"🎯 SIGNAL: {direction} GTC@${maker_price:.2f} × ${size:.2f} | "
            f"conf={confidence:.1%} edge={edge:.1%} | "
            f"T-{market.seconds_remaining:.0f}s | reversals={self._reversal_count}"
        )

        return [signal]

    async def on_fill(self, signal: Signal, fill_price: float, fill_size: float):
        """Track fill cost for P&L."""
        await super().on_fill(signal, fill_price, fill_size)
        self._round_cost += fill_price * fill_size

    async def on_round_end(self, market: MarketInfo, settled_outcome: Optional[str] = None):
        """Track oracle prediction accuracy and calculate P&L."""
        pnl = 0.0
        if self._signal_sent_this_round and self._last_direction:
            correct = self._last_direction == settled_outcome

            # Calculate dollar P&L
            if correct:
                # We bought the winning outcome token → pays $1 per token
                if self._last_direction == "Up":
                    payout = max(0, self.state.current_position_up) * 1.0
                else:
                    payout = max(0, self.state.current_position_down) * 1.0
                pnl = payout - self._round_cost
                self.state.rounds_won += 1
                self.state.consecutive_losses = 0
                self._logger.info(f"✅ Correct: {settled_outcome} | P&L=${pnl:+.2f}")
            else:
                # Wrong direction → our tokens pay $0
                pnl = -self._round_cost
                self.state.consecutive_losses += 1
                self._logger.info(
                    f"❌ Wrong: predicted={self._last_direction} actual={settled_outcome} | "
                    f"Loss=${pnl:+.2f}"
                )

            self.state.daily_pnl_usd += pnl
            self.state.total_pnl_usd += pnl

            # BUG4 FIX: Don't call super().on_round_end() — it would double-count rounds_traded
            self.state.rounds_traded += 1
            self.state.reset_round()  # clear positions for next round
            self._logger.info(
                f"Round end: {market.asset.upper()} | Outcome={settled_outcome} | "
                f"PnL={self.state.daily_pnl_usd:+.2f}"
            )
        else:
            # No trade this round — skip rounds_traded increment
            self.state.rounds_skipped += 1
            self.state.reset_round()  # still need to clear positions

        # Reset strike for next round
        self._strike_price = 0.0
