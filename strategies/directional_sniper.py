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
  - Bidirectional: buys Up when CEX > strike, buys Down when CEX < strike
  - 0% maker fee + daily USDC rebate
  - maker_price $0.60 needs only 60% accuracy to break even
  - T-10s CEX signal is ~75% accurate → substantial positive EV
  - Kelly sizing: larger bets when confidence is high, smaller when low

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
    E1: Buys Up when CEX > strike, buys Down when CEX < strike.
    E2: Uses half-Kelly criterion for position sizing.
    Uses maker orders at $0.55-0.65 for 0% fee + rebate.
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
        self._reversal_times: list[float] = []  # D4: timestamps of reversals

        # D1: EMA trend tracking
        self._ema_fast: float = 0.0
        self._ema_slow: float = 0.0
        self._ema_initialized: bool = False

        # D2: CLOB midpoint for Bayesian fusion
        self._clob_midpoint: float = 0.0  # Up token midpoint from CLOB

        # PnL tracking (directional inventory)
        self._buy_cost: float = 0.0
        self._shares_bought: float = 0.0
        self._trade_direction: str = ""  # E1: "Up" or "Down" — which token we bought

        # E2: bankroll for Kelly sizing (can be set from config or live balance)
        self._bankroll: float = getattr(config.strategy, 'sniper_bankroll', 100.0)

    # --- External setters (called by vps_runner) ---

    def set_cex_price(self, price: float):
        self._cex_price = price
        # D1: Update EMAs on each price update
        if not self._ema_initialized and price > 0:
            self._ema_fast = price
            self._ema_slow = price
            self._ema_initialized = True
        elif self._ema_initialized:
            self._ema_fast = 0.8 * self._ema_fast + 0.2 * price   # α=0.2 (fast)
            self._ema_slow = 0.95 * self._ema_slow + 0.05 * price  # α=0.05 (slow)

    def set_strike_price(self, strike: float):
        self._strike_price = strike

    def set_volatility(self, vol: float):
        self._volatility = max(0.001, vol)

    def set_clob_midpoint(self, midpoint: float):
        """D2: Set CLOB Up token midpoint for Bayesian fusion."""
        self._clob_midpoint = midpoint

    # --- Core logic ---

    def _get_direction_confidence(self) -> tuple[str, float]:
        """
        Determine direction and confidence from CEX vs strike.

        Returns ("Up"/"Down"/"Unknown", confidence 0.0-1.0).

        M2 FIX: Confidence now uses volatility-normalized distance
        instead of fixed percentage thresholds. This prevents
        triggering on noise-level moves in low-vol conditions.
        """
        if self._cex_price <= 0 or self._strike_price <= 0:
            return "Unknown", 0.0

        distance = (self._cex_price - self._strike_price) / self._strike_price
        abs_distance = abs(distance)

        if abs_distance < 0.00005:  # < 0.005% — effectively at strike
            return "Unknown", 0.50

        direction = "Up" if distance > 0 else "Down"

        # M2 FIX: Normalize by volatility — distance that is large relative
        # to recent vol deserves high confidence; same distance in high vol
        # is just noise. vol is typically 0.001-0.05 for 5-min BTC.
        vol = max(self._volatility, 0.001)
        z_score = abs_distance / vol  # "standard deviations" of move

        if z_score > 3.0:            # 3+ sigma — very strong signal
            confidence = 0.90 + min(z_score - 3.0, 0.05)  # 90-95%
        elif z_score > 2.0:          # 2-3 sigma — solid signal
            confidence = 0.80 + (z_score - 2.0) * 0.10     # 80-90%
        elif z_score > 1.0:          # 1-2 sigma — moderate signal
            confidence = 0.65 + (z_score - 1.0) * 0.15     # 65-80%
        elif z_score > 0.5:          # 0.5-1 sigma — weak signal
            confidence = 0.55 + (z_score - 0.5) * 0.20     # 55-65%
        else:
            confidence = 0.50 + z_score * 0.10              # 50-55%

        # D1: EMA trend adjustment — reward momentum alignment, punish divergence
        conf_after_zscore = confidence  # diagnostic
        if self._ema_initialized and self._ema_slow > 0:
            trend = (self._ema_fast - self._ema_slow) / self._ema_slow
            if (direction == "Up" and trend > 0) or (direction == "Down" and trend < 0):
                # Trend confirms direction → boost confidence (max +10%)
                confidence *= 1.0 + min(abs(trend) * 500, 0.10)
            else:
                # Trend opposes direction → reduce confidence (max -10%)
                # Reduced from -20% — 5-min markets have weak trend signals
                confidence *= max(0.90, 1.0 - abs(trend) * 300)

        conf_after_d1 = confidence  # diagnostic

        # D2: Bayesian fusion with CLOB implied probability
        # Skip when CLOB is in 0.45-0.55 "uninformed" zone (market is 50/50)
        if (0.05 < self._clob_midpoint < 0.45) or (0.55 < self._clob_midpoint < 0.95):
            p_clob = self._clob_midpoint  # CLOB's market-implied P(Up)
            p_cex = confidence if direction == "Up" else (1.0 - confidence)
            # Guard: avoid log(0) or division-by-zero with extreme values
            p_cex = max(0.02, min(0.98, p_cex))
            # Bayesian update: P(Up|both) ∝ P(Up|CEX) × P(Up|CLOB)
            odds_cex = p_cex / (1 - p_cex)
            odds_clob = p_clob / (1 - p_clob)
            combined_odds = odds_cex * odds_clob  # assume 0.5 prior → cancels
            fused_up = combined_odds / (1 + combined_odds)
            # Convert back to directional confidence, clamp to valid range
            confidence = fused_up if direction == "Up" else (1.0 - fused_up)
            confidence = max(0.50, min(0.95, confidence))

        # Diagnostic: log pipeline stages
        self._logger.debug(
            f"Conf pipeline: z={conf_after_zscore:.3f} → D1={conf_after_d1:.3f} → D2={confidence:.3f} | "
            f"dir={direction} clob={self._clob_midpoint:.3f} vol={self._volatility:.5f}"
        )

        return direction, min(confidence, 0.95)

    def _calculate_maker_price(self, confidence: float) -> float:
        """D5 FIX: Continuous maker price interpolation.

        Old code used step function with $0.02 jumps at confidence boundaries.
        New: linear interpolation from $0.50 (conf=0.50) to $0.70 (conf=0.95).
        """
        sc = self.config.strategy
        min_price = 0.50
        max_price = 0.70
        min_conf = 0.50
        max_conf = 0.95

        # D5: Linear interpolation
        t = (confidence - min_conf) / max(max_conf - min_conf, 0.01)
        t = max(0.0, min(1.0, t))  # clamp to [0, 1]
        price = min_price + t * (max_price - min_price)

        return round(max(min_price, min(max_price, price)), 2)

    def _kelly_size(self, confidence: float, maker_price: float) -> float:
        """
        E2: Calculate order size using half-Kelly criterion.

        Kelly fraction: f* = (b*p - q) / b
        where b = payout odds = (1/price - 1), p = win probability, q = 1-p

        We use HALF Kelly for safety (lower variance, captures ~75% of optimal growth).
        """
        sc = self.config.strategy
        min_size = getattr(sc, 'sniper_min_size', 5.0)
        max_fraction = getattr(sc, 'sniper_max_kelly_fraction', 0.15)

        if maker_price <= 0 or maker_price >= 1 or confidence <= 0:
            return min_size

        b = (1.0 / maker_price) - 1.0  # odds ratio (e.g., $0.60 → 0.667)

        # D6: Use empirical win-rate lower bound blended with model confidence
        # This prevents Kelly from over-betting based on untested model estimates
        if self.state.rounds_traded > 20:
            n = self.state.rounds_traded
            wins = self.state.rounds_won
            p_hat = wins / n
            z = 1.645  # 90% CI
            denom = 1 + z * z / n
            center = (p_hat + z * z / (2 * n)) / denom
            offset = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
            empirical_lower = max(0.01, center - offset)
            # Blend: 70% model confidence + 30% empirical lower bound
            p = 0.7 * confidence + 0.3 * empirical_lower
        else:
            p = confidence

        q = 1.0 - p

        kelly_f = (b * p - q) / b
        if kelly_f <= 0:
            return min_size  # negative Kelly = no edge

        half_kelly = kelly_f / 2.0
        clamped = min(half_kelly, max_fraction)
        size = self._bankroll * clamped

        return round(max(min_size, min(size, self._bankroll * max_fraction)), 2)

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
                        self._reversal_times.append(now)  # D4: track reversal time
                        # Bound list to prevent memory leak in long sessions
                        if len(self._reversal_times) > 50:
                            self._reversal_times = self._reversal_times[-50:]
                    self._direction_samples.append(direction)
            return []

        # --- COMMITMENT (T-10s to T-5s): fire if confident ---
        if self._signal_sent:
            return []

        direction, confidence = self._get_direction_confidence()

        # E1: Bidirectional — trade both Up and Down
        if direction not in ("Up", "Down"):
            if t <= 7:
                self._signal_sent = True
            self._logger.debug(
                f"Skip: direction={direction} conf={confidence:.1%} | "
                f"cex=${self._cex_price:.2f} strike=${self._strike_price:.2f}"
            )
            return []

        # D4: Time-weighted reversal scoring (recent reversals hurt more)
        now = time.time()
        weighted_reversals = sum(
            1.0 / (1.0 + (now - rt) / 10.0)
            for rt in self._reversal_times
        )
        if weighted_reversals > 2.5:
            if t <= 7:
                self._signal_sent = True
            self._logger.info(
                f"Skip: weighted reversals={weighted_reversals:.1f} (unstable) | T-{t:.0f}s"
            )
            return []

        # Confidence threshold
        min_confidence = getattr(self.config.strategy, 'sniper_min_confidence', 0.55)

        # D3: Time-decay urgency bonus — later signals are more informative
        # T-10s → +0.00, T-5s → +0.05
        time_bonus = 0.05 * (1.0 - (t - 5.0) / 5.0) if 5 <= t <= 10 else 0.0
        confidence += time_bonus

        if confidence < min_confidence:
            self._logger.info(
                f"Skip: confidence {confidence:.1%} < {min_confidence:.1%} | "
                f"cex=${self._cex_price:.2f} strike=${self._strike_price:.2f} T-{t:.0f}s"
            )
            return []

        # Calculate maker price
        maker_price = self._calculate_maker_price(confidence)

        # E2: Kelly-sized order
        order_size = self._kelly_size(confidence, maker_price)

        # E1: Select outcome based on direction
        outcome = Outcome.UP if direction == "Up" else Outcome.DOWN

        signal = Signal(
            market=market,
            outcome=outcome,
            side=Side.BUY,
            price=maker_price,
            size_usd=order_size,
            confidence=confidence,
            reason=(
                f"Sniper: {direction} GTC@${maker_price:.2f} × ${order_size:.0f} | "
                f"conf={confidence:.1%} | "
                f"cex=${self._cex_price:.2f} vs strike=${self._strike_price:.2f} | "
                f"reversals={self._reversal_count} | T-{t:.0f}s"
            ),
            order_type="GTC",
        )

        self._signal_sent = True
        self._trade_direction = direction  # E1: remember which direction we traded
        self._logger.info(
            f"🎯 SIGNAL: {direction} GTC@${maker_price:.2f} × ${order_size:.0f} | "
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
        self._trade_direction = ""
        self._reversal_times = []  # D4
        self._clob_midpoint = 0.0  # D2
        # D1: Do NOT reset EMAs — trend context is valuable across rounds

        if market.strike_price > 0:
            self._strike_price = market.strike_price

    async def on_fill(self, signal: Signal, fill_price: float, fill_size: float):
        await super().on_fill(signal, fill_price, fill_size)
        self._buy_cost += fill_price * fill_size
        self._shares_bought += fill_size

    async def on_round_end(self, market: MarketInfo, settled_outcome: Optional[str] = None):
        """
        Calculate PnL. E1: Supports both Up and Down positions.

        If we bought Up tokens:  Up settles → $1.00/share,   Down settles → $0.00
        If we bought Down tokens: Down settles → $1.00/share, Up settles → $0.00

        C2 NOTE: We intentionally DO NOT call super().on_round_end() because
        we fully override PnL tracking and rounds_traded counting here.
        BaseStrategy.on_round_end() would double-count rounds_traded.

        ⚠️ MAINTENANCE: If you add NEW logic to BaseStrategy.on_round_end(),
        you MUST mirror it here manually. This override is the reason.
        """
        if self._shares_bought > 0:
            avg_cost = self._buy_cost / self._shares_bought
            # E1: Payout depends on which direction we traded
            if self._trade_direction == "Up":
                payout = self._shares_bought * 1.0 if settled_outcome == "Up" else 0.0
            elif self._trade_direction == "Down":
                payout = self._shares_bought * 1.0 if settled_outcome == "Down" else 0.0
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

        # E2: Update bankroll for next round's Kelly calculation
        self._bankroll = max(10.0, self._bankroll + pnl)

        if self._shares_bought > 0:
            self._logger.info(
                f"{outcome_emoji} Round PnL: ${pnl:+.2f} | "
                f"dir={self._trade_direction} settled={settled_outcome} | "
                f"{self._shares_bought:.1f}sh @${avg_cost:.3f} | "
                f"Daily: ${self.state.daily_pnl_usd:+.2f} | "
                f"Win rate: {self.state.win_rate:.1%} | "
                f"Bankroll: ${self._bankroll:.0f}"
            )
        else:
            self._logger.debug(f"Round skip (no fills) | settled={settled_outcome}")
