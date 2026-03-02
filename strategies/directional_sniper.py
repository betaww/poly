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

from scipy.stats import t as t_dist

# Log-odds helpers for independent D1/D2/D3 fusion
def _to_log_odds(p: float) -> float:
    p = max(0.01, min(0.99, p))
    return math.log(p / (1.0 - p))

def _from_log_odds(lo: float) -> float:
    lo = max(-20.0, min(20.0, lo))  # guard against exp overflow
    return 1.0 / (1.0 + math.exp(-lo))

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

        # D3: Binance OFI (Order Flow Imbalance) signal
        self._ofi: float = 0.0  # -1.0 (strong sell) to +1.0 (strong buy)

        # v11: CLOB best bid for maker pricing guard
        self._clob_best_bid: float = 0.0

        # PnL tracking (directional inventory)
        self._buy_cost: float = 0.0
        self._shares_bought: float = 0.0
        self._trade_direction: str = ""  # E1: "Up" or "Down" — which token we bought

        # E2: bankroll for Kelly sizing (can be set from config or live balance)
        self._bankroll: float = getattr(config.strategy, 'sniper_bankroll', 100.0)

        # v11: Sharpe / drawdown / profit factor tracking
        self._pnl_history: list[float] = []
        self._peak_bankroll: float = self._bankroll
        self._max_drawdown: float = 0.0
        self._gross_wins: float = 0.0
        self._gross_losses: float = 0.0

        # v11: Last signal delta breakdown for analysis
        self._last_d1: float = 0.0
        self._last_d2: float = 0.0
        self._last_d3: float = 0.0
        self._last_d4: float = 0.0  # v12: LWBA spread delta

        # v11: Cross-asset correlation flag (set by vps_runner)
        self._correlation_penalty: float = 1.0  # 1.0 = no penalty, 0.5 = halved

        # v12: LWBA shadow price (Chainlink settlement replica)
        self._lwba_shadow_price: float = 0.0  # LWBA Mid from multi-exchange depth
        self._lwba_spread_bps: float = 0.0    # LWBA spread in basis points

    # --- External setters (called by vps_runner) ---

    def set_cex_price(self, price: float):
        self._cex_price = price
        # D1: Update EMAs on each price update
        # v10 FIX #6: Slowed EMAs to match 5-min window (was α=0.2/0.05, too noisy at 1Hz)
        if not self._ema_initialized and price > 0:
            self._ema_fast = price
            self._ema_slow = price
            self._ema_initialized = True
        elif self._ema_initialized:
            self._ema_fast = 0.95 * self._ema_fast + 0.05 * price   # α=0.05 (half-life ~14s)
            self._ema_slow = 0.99 * self._ema_slow + 0.01 * price   # α=0.01 (half-life ~70s)

    def set_strike_price(self, strike: float):
        self._strike_price = strike

    def set_volatility(self, vol: float):
        self._volatility = max(0.001, vol)

    def set_clob_midpoint(self, midpoint: float):
        """D2: Set CLOB Up token midpoint for Bayesian fusion."""
        self._clob_midpoint = midpoint

    def set_ofi(self, ofi: float):
        """D3: Set Binance Order Flow Imbalance signal."""
        self._ofi = max(-1.0, min(1.0, ofi))

    def set_clob_best_bid(self, best_bid: float):
        """v11: Set current CLOB best bid for maker pricing guard."""
        self._clob_best_bid = best_bid

    def set_correlation_penalty(self, penalty: float):
        """v11: Set cross-asset correlation penalty (0.5 = halved size)."""
        self._correlation_penalty = max(0.1, min(1.0, penalty))

    def set_lwba_shadow_price(self, price: float):
        """v12: Set LWBA shadow price (Chainlink settlement replica)."""
        if price > 0:
            self._lwba_shadow_price = price

    def set_lwba_spread(self, spread_bps: float):
        """v12: Set LWBA bid-ask spread in basis points."""
        self._lwba_spread_bps = spread_bps

    # --- Core logic ---

    def _get_direction_confidence(self) -> tuple[str, float]:
        """
        Determine direction and confidence from CEX vs strike.

        Returns ("Up"/"Down"/"Unknown", confidence 0.0-1.0).

        v10 FIX #3: D1/D2/D3 now fuse independently in log-odds space
        instead of sequential multiplication (avoids double-counting).
        v10 FIX #1: Deadzone widened to 0.05% (was 0.005%) to filter
        boundary rounds that cause MISMATCH with Chainlink.
        """
        if self._cex_price <= 0 or self._strike_price <= 0:
            return "Unknown", 0.0

        # v12: Use LWBA shadow price if available (matches Chainlink settlement)
        # Fallback to CEX spot if LWBA unavailable
        effective_price = self._lwba_shadow_price if self._lwba_shadow_price > 0 else self._cex_price

        distance = (effective_price - self._strike_price) / self._strike_price
        abs_distance = abs(distance)

        # v10 FIX #1: Widened deadzone (was 0.00005 = 0.005%)
        # Settlement Verifier showed ALL mismatches at CEX ≈ strike boundary
        if abs_distance < 0.0005:  # < 0.05% — boundary zone, too risky
            return "Unknown", 0.50

        direction = "Up" if distance > 0 else "Down"

        # M2 FIX: Normalize by volatility
        vol = max(self._volatility, 0.001)
        z_score = abs_distance / vol

        # Student-t CDF (df=4): fat-tail correction
        raw_conf = t_dist.cdf(z_score, df=4)
        base_conf = max(0.50, min(0.95, raw_conf))

        # --- v10 FIX #3: Independent fusion in log-odds space ---
        # Each stage computes an *additive delta* in log-odds, preventing
        # sequential dependence and double-counting.
        base_log_odds = _to_log_odds(base_conf)
        d1_delta = 0.0  # EMA trend
        d2_delta = 0.0  # CLOB Bayesian
        d3_delta = 0.0  # OFI

        # D1: EMA trend adjustment
        # v10 FIX #6: Added trend magnitude threshold (0.1%) to filter noise
        if self._ema_initialized and self._ema_slow > 0:
            trend = (self._ema_fast - self._ema_slow) / self._ema_slow
            if abs(trend) > 0.001:  # only act on > 0.1% trend
                if (direction == "Up" and trend > 0) or (direction == "Down" and trend < 0):
                    d1_delta = min(abs(trend) * 200, 0.40)   # max +10% conf equiv
                else:
                    d1_delta = -min(abs(trend) * 150, 0.40)  # max -10% conf equiv

        # D2: CLOB Bayesian fusion
        # clob_log_odds = ln(p_up/(1-p_up)): positive if CLOB favors Up
        # For Up direction: positive CLOB LO → confirms → add
        # For Down direction: positive CLOB LO → contradicts → subtract
        if (0.05 < self._clob_midpoint < 0.45) or (0.55 < self._clob_midpoint < 0.95):
            p_clob = self._clob_midpoint
            clob_log_odds = _to_log_odds(p_clob)
            if direction == "Up":
                d2_delta = clob_log_odds * 0.5
            else:
                d2_delta = -clob_log_odds * 0.5

        # D3: OFI confirmation
        if abs(self._ofi) > 0.2:
            if (direction == "Up" and self._ofi > 0) or (direction == "Down" and self._ofi < 0):
                d3_delta = min(abs(self._ofi) * 0.6, 0.40)   # confirms
            else:
                d3_delta = -min(abs(self._ofi) * 0.4, 0.40)  # opposes

        # Fuse in log-odds space and convert back
        fused_log_odds = base_log_odds + d1_delta + d2_delta + d3_delta

        # v12: LWBA spread confidence modifier
        # Narrow spread (< 3 bps) → +2% confidence (market is liquid/certain)
        # Wide spread (> 10 bps) → -5% confidence (market is uncertain)
        d4_delta = 0.0
        if self._lwba_spread_bps > 0:
            if self._lwba_spread_bps < 3.0:
                d4_delta = 0.10   # narrow → bonus
            elif self._lwba_spread_bps > 10.0:
                d4_delta = -0.20  # wide → penalty
            fused_log_odds += d4_delta

        confidence = _from_log_odds(fused_log_odds)
        confidence = max(0.50, min(0.95, confidence))

        # v11: Store delta decomposition for logging and analysis
        self._last_d1 = d1_delta
        self._last_d2 = d2_delta
        self._last_d3 = d3_delta
        self._last_d4 = d4_delta # v12: LWBA spread delta
        self._last_confidence = confidence
        self._last_z_score = base_conf  # store base for diagnostics

        return direction, confidence

    def _calculate_maker_price(self, confidence: float) -> float:
        """D5 FIX: Continuous maker price interpolation.

        v11: Raised cap from $0.70 to $0.80 — high confidence signals were
        being capped at $0.70 which systematically limited profits.
        v11: Guard against accidentally taker-matching by checking CLOB best bid.
        """
        sc = self.config.strategy
        min_price = 0.50
        max_price = 0.80  # v11: was 0.70 — too restrictive for high conf
        min_conf = 0.50
        max_conf = 0.95

        # D5: Linear interpolation
        t = (confidence - min_conf) / max(max_conf - min_conf, 0.01)
        t = max(0.0, min(1.0, t))  # clamp to [0, 1]
        price = min_price + t * (max_price - min_price)
        price = round(max(min_price, min(max_price, price)), 2)

        # v11 #17: Ensure we don't accidentally cross the spread and become taker
        # If our price >= CLOB best ask (approximated as 1 - best_bid for the
        # complement token), cap at best_bid + 1 tick to stay as maker
        if self._clob_best_bid > 0 and price > self._clob_best_bid:
            price = round(min(price, self._clob_best_bid), 2)

        return price

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
        # v10 FIX #4: Raised threshold from 20 to 50 rounds — Wilson CI is too
        # wide with <50 observations, causing systematic position under-sizing.
        if self.state.rounds_traded > 50:
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
        Main strategy loop — fires in dynamic commitment window.

        v11: Dynamic window based on volatility (high vol = shorter window).
        v11: T-5s to T-3s re-check: cancel order if direction reversed.
        """
        if self.should_pause():
            return []

        t = market.seconds_remaining
        _t0 = time.time()  # v11 #14: pipeline timing

        # --- BLACKOUT (T < 3s): too late even for cancel ---
        if t < 3:
            return []

        # v11 #2: T-5s to T-3s RE-CHECK — cancel if direction reversed
        if self._signal_sent and 3 <= t <= 5:
            direction, _ = self._get_direction_confidence()
            if direction != self._trade_direction and direction in ("Up", "Down"):
                self._logger.info(
                    f"⚠️ Direction reversed! {self._trade_direction}→{direction} at T-{t:.0f}s — cancel signal"
                )
                self._signal_sent = False
                self._trade_direction = ""
                # Return empty — vps_runner will call cancel_all() based on this
                return [Signal(
                    market=market,
                    outcome=Outcome.UP,  # dummy
                    side=Side.BUY,
                    price=0.0,
                    size_usd=0.0,
                    confidence=0.0,
                    reason="CANCEL:direction_reversed",
                    order_type="CANCEL",
                )]
            return []

        # v11 #4: Dynamic commitment window based on volatility
        # High vol (>2%) → window starts at T-7s (less time for reversal)
        # Low vol (<0.5%) → window starts at T-12s (can commit earlier)
        vol_pct = self._volatility * 100  # e.g., 0.01 → 1.0%
        window_start = max(7, min(12, int(10 / max(vol_pct, 0.3) * 1.0)))

        # --- OBSERVATION (T > window_start): just track direction stability ---
        if t > window_start:
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

        # --- COMMITMENT (T-window to T-5s): fire if confident ---
        if self._signal_sent:
            return []

        direction, confidence = self._get_direction_confidence()

        # v11 #14: Pipeline timing
        pipeline_ms = (time.time() - _t0) * 1000

        # v11 #12: Log with D1/D2/D3 delta decomposition
        self._logger.info(
            f"⏱️ T-{t:.0f}s | cex=${self._cex_price:.2f} strike=${self._strike_price:.2f} "
            f"vol={self._volatility:.5f} | dir={direction} conf={confidence:.1%} | "
            f"z={getattr(self, '_last_z_score', 0):.3f} clob={self._clob_midpoint:.3f} | "
            f"d1={self._last_d1:+.3f} d2={self._last_d2:+.3f} d3={self._last_d3:+.3f} | "
            f"{pipeline_ms:.1f}ms"
        )

        # E1: Bidirectional — trade both Up and Down
        if direction not in ("Up", "Down"):
            if t <= 7:
                self._signal_sent = True
            # A4 FIX: promote from DEBUG → INFO (was the silent killer)
            self._logger.info(
                f"Skip: direction={direction} conf={confidence:.1%} | "
                f"cex=${self._cex_price:.2f} strike=${self._strike_price:.2f}"
            )
            return []

        # D4: Time-weighted reversal scoring (recent reversals hurt more)
        # v10 FIX #15: Lowered threshold from 2.5 to 1.5 — old value almost
        # never triggered (needed 5+ rapid reversals). Now 3 reversals in
        # 10 seconds will trigger (0.5 × 3 ≈ 1.5).
        now = time.time()
        weighted_reversals = sum(
            1.0 / (1.0 + (now - rt) / 10.0)
            for rt in self._reversal_times
        )
        if weighted_reversals > 1.5:
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

        # v11 #15: Apply cross-asset correlation penalty
        order_size = round(order_size * self._correlation_penalty, 2)
        order_size = max(getattr(self.config.strategy, 'sniper_min_size', 5.0), order_size)

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
        corr_str = f" corr_pen={self._correlation_penalty:.1f}" if self._correlation_penalty < 1.0 else ""
        self._logger.info(
            f"🎯 SIGNAL: {direction} GTC@${maker_price:.2f} × ${order_size:.0f} | "
            f"conf={confidence:.1%} | T-{t:.0f}s | "
            f"cex=${self._cex_price:.2f} vs strike=${self._strike_price:.2f}{corr_str}"
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
        self._clob_best_bid = 0.0  # v11
        self._correlation_penalty = 1.0  # v11: reset per round
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
            self._gross_wins += pnl
        elif pnl < 0:
            self.state.consecutive_losses += 1
            self._gross_losses += abs(pnl)

        self.state.rounds_traded += 1
        self.state.reset_round()

        # E2: Update bankroll for next round's Kelly calculation
        self._bankroll = max(10.0, self._bankroll + pnl)

        # v11 #10: Track drawdown and PnL history
        if self._shares_bought > 0:
            self._pnl_history.append(pnl)
            if self._bankroll > self._peak_bankroll:
                self._peak_bankroll = self._bankroll
            dd = (self._peak_bankroll - self._bankroll) / self._peak_bankroll if self._peak_bankroll > 0 else 0
            self._max_drawdown = max(self._max_drawdown, dd)

        if self._shares_bought > 0:
            # v11 #10: Compute running Sharpe (annualized per-round)
            sharpe_str = ""
            if len(self._pnl_history) >= 10:
                import statistics
                mu = statistics.mean(self._pnl_history)
                sigma = statistics.stdev(self._pnl_history) if len(self._pnl_history) > 1 else 1.0
                sharpe = mu / sigma if sigma > 0 else 0.0
                pf = self._gross_wins / self._gross_losses if self._gross_losses > 0 else float('inf')
                sharpe_str = f" | SR={sharpe:.2f} PF={pf:.1f} MDD={self._max_drawdown:.1%}"

            self._logger.info(
                f"{outcome_emoji} Round PnL: ${pnl:+.2f} | "
                f"dir={self._trade_direction} settled={settled_outcome} | "
                f"{self._shares_bought:.1f}sh @${avg_cost:.3f} | "
                f"Daily: ${self.state.daily_pnl_usd:+.2f} | "
                f"Win rate: {self.state.win_rate:.1%} | "
                f"Bankroll: ${self._bankroll:.0f}{sharpe_str}"
            )
        else:
            self._logger.debug(f"Round skip (no fills) | settled={settled_outcome}")
