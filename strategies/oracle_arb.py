"""
Strategy 2: Oracle Latency Arbitrage (5-minute version).

Ported from Alpha-Sigma V7's 15-minute strategy.
Exploits the 50-2000ms delay between CEX price discovery and Chainlink settlement.

How it works:
  1. OBSERVATION (T > 15s): Build price context, no trading
  2. PREPARATION (T <= 15s): Start tracking direction, watch for reversals
  3. COMMITMENT (T-7s to T-3s): Execute if confidence > 95%
  4. BLACKOUT (T < 3s): Too late, no trading

Uses synthetic oracle (weighted average of Binance/Coinbase/Kraken/OKX)
to predict Chainlink settlement price before it's finalized.
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
    Oracle Latency Arbitrage — predicts Chainlink settlement 50-200ms in advance.

    Key difference from Alpha-Sigma V7 (15m):
      - 5-minute windows = 3x more trading opportunities
      - Same synthetic oracle logic, adapted commitment window
    """

    def __init__(self, config: Config):
        super().__init__(config, name="OracleArb")

        # Synthetic oracle inputs
        self._cex_price: float = 0.0
        self._strike_price: float = 0.0

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
        # commitment_window_start=7, commitment_window_end=3
        # OBSERVATION: t > 15s      (too early)
        # PREPARATION: 7s < t <= 15s (tracking, no trading)
        # COMMITMENT:  3s < t <= 7s  (execute if confident)
        # BLACKOUT:    t <= 3s       (too late)
        if t > 15:
            return Phase.OBSERVATION
        elif t > sc.commitment_window_start:  # t > 7
            return Phase.PREPARATION
        elif t > sc.commitment_window_end:    # t > 3
            return Phase.COMMITMENT
        else:
            return Phase.BLACKOUT

    def set_cex_price(self, price: float):
        self._cex_price = price

    def set_strike_price(self, strike: float):
        self._strike_price = strike

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
        Calculate trading edge.
        Edge = P(win) - cost_of_entry

        Uses Polymarket dynamic taker fee:
          fee = C × feeRate × (p(1-p))^exponent
          effective_rate ≈ feeRate × (p(1-p))^exponent / p
        For FOK orders at p=0.50: effective rate ≈ 1.56%
        """
        if direction == "Up":
            entry_cost = market.price_up  # cost to buy "Up" token
        else:
            entry_cost = market.price_down  # cost to buy "Down" token

        # Dynamic fee: feeRate × (p(1-p))^exponent
        p = entry_cost
        fee_factor = market.fee_rate * (p * (1 - p)) ** market.fee_exponent
        # fee_factor is per-share, effective cost increase = fee_factor / p
        effective_cost = entry_cost + fee_factor

        # Edge = predicted probability - effective cost
        edge = confidence - effective_cost

        return edge

    def _calculate_size(self, edge: float, confidence: float) -> float:
        """Kelly criterion position sizing (1/4 Kelly for safety)."""
        sc = self.config.strategy

        if edge <= 0:
            return 0.0

        # Full Kelly = edge / (odds - 1), simplified for binary
        # 1/4 Kelly for conservative sizing
        kelly_size = sc.kelly_fraction * edge * sc.max_position_usd

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

        # Build signal
        outcome = Outcome.UP if direction == "Up" else Outcome.DOWN
        target_price = market.price_up if direction == "Up" else market.price_down

        signal = Signal(
            market=market,
            outcome=outcome,
            side=Side.BUY,
            price=min(target_price + 0.02, 0.99),  # aggressive price for fill
            size_usd=size,
            confidence=confidence,
            reason=(
                f"OracleArb: {direction} | conf={confidence:.1%} | edge={edge:.1%} | "
                f"cex={self._cex_price:.2f} vs strike={self._strike_price:.2f} | "
                f"T-{market.seconds_remaining:.0f}s"
            ),
            order_type="FOK",  # Fill-or-kill for speed
        )

        self._signal_sent_this_round = True
        self._logger.info(
            f"🎯 SIGNAL: {direction} @ ${size:.2f} | "
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

            # BUG FIX: Only increment rounds_traded for rounds we actually traded
            self.state.rounds_traded += 1
            self._logger.info(
                f"Round end: {market.asset.upper()} | Outcome={settled_outcome} | "
                f"PnL={self.state.daily_pnl_usd:+.2f}"
            )
        else:
            # No trade this round — skip rounds_traded increment
            # Don't call super() which would incorrectly inflate rounds_traded
            pass

        # Reset strike for next round
        self._strike_price = 0.0
