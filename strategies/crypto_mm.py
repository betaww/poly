"""
Strategy 1: 5-Minute Crypto Market Making.

The primary strategy. Places two-sided quotes on BTC/ETH 5-minute Up/Down markets.

How it works:
  1. Estimate settlement probability using mid-price from Binance/Coinbase vs Chainlink strike
  2. Place bid (buy) at probability - spread/2, ask (sell) at probability + spread/2
  3. Adjust spread dynamically based on volatility and time remaining
  4. Every 5 minutes the market settles automatically — no overnight risk

Key advantage: 0% maker fees (100% rebate) = lowest-cost market maker in the pool.
"""
import logging
import time
import math
from typing import Optional

from .base import BaseStrategy, Signal, Side, Outcome
from ..config import Config
from ..market_scanner import MarketInfo

logger = logging.getLogger(__name__)


class CryptoMarketMaker(BaseStrategy):
    """
    5-Minute Binary Options Market Maker.

    Earns spread by providing liquidity on both "Up" and "Down" outcomes.
    Uses CEX price data to estimate fair probability, then quotes around it.
    """

    def __init__(self, config: Config):
        super().__init__(config, name="CryptoMM")
        self._last_quote_time: float = 0
        self._cex_price: Optional[float] = None
        self._strike_price: Optional[float] = None
        self._volatility: float = 0.01  # initial estimate
        self._directional_sent: bool = False  # one directional signal per round

        # Track cost per side for accurate P&L
        self._cost_up: float = 0.0     # total USD spent buying Up tokens
        self._cost_down: float = 0.0   # total USD spent buying Down tokens

        # Real CLOB orderbook data (populated by vps_runner from clob_book_feed)
        self._book_bid_up: float = 0.0
        self._book_ask_up: float = 0.0
        self._book_bid_down: float = 0.0
        self._book_ask_down: float = 0.0
        self._book_depth_bid: float = 0.0   # total bid depth USD
        self._book_depth_ask: float = 0.0   # total ask depth USD
        self._has_book_data: bool = False

    def set_cex_price(self, price: float):
        """Update current CEX price (called by data feed)."""
        self._cex_price = price

    def set_strike_price(self, price: float):
        """Update the strike price for current round."""
        self._strike_price = price

    def set_volatility(self, vol: float):
        """Update volatility estimate (from rolling std of returns)."""
        self._volatility = max(0.001, vol)

    def set_book_data(self, best_bid_up: float, best_ask_up: float,
                      best_bid_down: float, best_ask_down: float,
                      depth_bid_usd: float, depth_ask_usd: float):
        """Update real CLOB orderbook data."""
        self._book_bid_up = best_bid_up
        self._book_ask_up = best_ask_up
        self._book_bid_down = best_bid_down
        self._book_ask_down = best_ask_down
        self._book_depth_bid = depth_bid_usd
        self._book_depth_ask = depth_ask_usd
        self._has_book_data = True

    def _estimate_fair_probability(self, market: MarketInfo) -> float:
        """
        Estimate probability of "Up" outcome.

        If we have CEX price and strike:
          - P(Up) = sigmoid of (cex_price - strike) / (strike * volatility * sqrt(T))
        
        If not, use current market midpoint as proxy.
        """
        if self._cex_price and self._strike_price and self._strike_price > 0:
            distance = (self._cex_price - self._strike_price) / self._strike_price

            # Time factor: more time = more uncertainty = closer to 50%
            t_remaining = max(1, market.seconds_remaining)
            t_total = market.duration_seconds
            time_factor = math.sqrt(t_remaining / t_total)

            # Normalized distance
            z = distance / (self._volatility * time_factor)

            # Sigmoid mapping to probability (clamped 5-95%)
            # Guard against overflow: exp(-x) overflows for |x| > ~700
            z_clamped = max(-50, min(50, z * 10))
            prob_up = 1.0 / (1.0 + math.exp(-z_clamped))
            prob_up = max(0.05, min(0.95, prob_up))

            return prob_up
        else:
            # Fallback: use market midpoint
            return market.price_up

    def _calculate_spread(self, market: MarketInfo) -> float:
        """
        Dynamic spread based on:
          - Real CLOB spread (if available) — BLIND SPOT FIX
          - Base spread from config (fallback)
          - Volatility multiplier
          - Time remaining (wider spread near settlement)
        """
        sc = self.config.strategy

        # BLIND SPOT FIX: Use real CLOB spread as baseline if available
        if self._has_book_data and self._book_bid_up > 0 and self._book_ask_up < 1:
            real_spread = self._book_ask_up - self._book_bid_up
            # Don't go tighter than real market spread (we'd get picked off)
            base = max(sc.base_spread, real_spread * 1.1)  # 10% wider than market
        else:
            base = sc.base_spread

        # Volatility adjustment: higher vol = wider spread
        vol_mult = 1.0 + (self._volatility / 0.01) * 0.5

        # Time decay: widen spread in last 30 seconds
        t_remaining = market.seconds_remaining
        if t_remaining < 30:
            time_mult = 1.5 + (30 - t_remaining) / 30 * 1.5  # 1.5x to 3x
        elif t_remaining < 60:
            time_mult = 1.2
        else:
            time_mult = 1.0

        # Depth adjustment: thin book = wider spread
        if self._has_book_data and self._book_depth_bid > 0:
            if self._book_depth_bid < 2000:  # less than $2K depth
                depth_mult = 1.5  # much wider
            elif self._book_depth_bid < 5000:
                depth_mult = 1.2  # slightly wider
            else:
                depth_mult = 1.0
        else:
            depth_mult = 1.0

        spread = base * vol_mult * time_mult * depth_mult
        return max(sc.min_spread, min(sc.max_spread, spread))

    def _round_to_tick(self, price: float, tick_size: str) -> float:
        """Round price to valid tick size."""
        tick = float(tick_size)
        return round(round(price / tick) * tick, 2)

    async def on_market_update(self, market: MarketInfo) -> list[Signal]:
        """
        Generate two-sided quotes on each market update.

        Post-Feb 2026 enhancement: in the last 10 seconds, if CEX price
        clearly favors one direction, switch to one-sided directional
        maker orders for settlement profit.
        """
        # Don't quote if paused
        if self.should_pause():
            return []

        # Don't quote too close to settlement
        if market.seconds_remaining < 3:
            return []

        # --- T-10s DIRECTIONAL MODE ---
        # In last 10 seconds, if direction is clear, go one-sided (once per round)
        if (market.seconds_remaining <= 10 and not self._directional_sent and
            self._cex_price and self._cex_price > 0 and
            self._strike_price and self._strike_price > 0):

            distance = (self._cex_price - self._strike_price) / self._strike_price
            abs_distance = abs(distance)

            # If CEX is >0.1% from strike, direction is highly likely
            if abs_distance > 0.001:
                direction = "Up" if distance > 0 else "Down"
                self._directional_sent = True
                return self._generate_directional_signal(market, direction)

        # Don't quote too frequently
        now = time.time()
        refresh_interval = self.config.strategy.refresh_interval_ms / 1000.0
        if now - self._last_quote_time < refresh_interval:
            return []
        self._last_quote_time = now

        # Estimate fair probability
        prob_up = self._estimate_fair_probability(market)
        prob_down = 1.0 - prob_up

        # Calculate spread
        spread = self._calculate_spread(market)
        half_spread = spread / 2.0

        # Generate quotes
        signals = []
        tick = market.tick_size
        order_size = self.config.strategy.order_size_usd

        # --- "Up" side ---
        bid_up_price = self._round_to_tick(prob_up - half_spread, tick)
        ask_up_price = self._round_to_tick(prob_up + half_spread, tick)

        if 0.01 <= bid_up_price <= 0.99:
            signals.append(Signal(
                market=market,
                outcome=Outcome.UP,
                side=Side.BUY,
                price=bid_up_price,
                size_usd=order_size,
                confidence=0.6,
                reason=f"MM bid Up @ {bid_up_price:.2f} (fair={prob_up:.3f}, spread={spread:.3f})",
                order_type="GTC",
            ))

        if 0.01 <= ask_up_price <= 0.99:
            signals.append(Signal(
                market=market,
                outcome=Outcome.UP,
                side=Side.SELL,
                price=ask_up_price,
                size_usd=order_size,
                confidence=0.6,
                reason=f"MM ask Up @ {ask_up_price:.2f} (fair={prob_up:.3f}, spread={spread:.3f})",
                order_type="GTC",
            ))

        # --- "Down" side ---
        bid_down_price = self._round_to_tick(prob_down - half_spread, tick)
        ask_down_price = self._round_to_tick(prob_down + half_spread, tick)

        if 0.01 <= bid_down_price <= 0.99:
            signals.append(Signal(
                market=market,
                outcome=Outcome.DOWN,
                side=Side.BUY,
                price=bid_down_price,
                size_usd=order_size,
                confidence=0.6,
                reason=f"MM bid Down @ {bid_down_price:.2f} (fair={prob_down:.3f})",
                order_type="GTC",
            ))

        if 0.01 <= ask_down_price <= 0.99:
            signals.append(Signal(
                market=market,
                outcome=Outcome.DOWN,
                side=Side.SELL,
                price=ask_down_price,
                size_usd=order_size,
                confidence=0.6,
                reason=f"MM ask Down @ {ask_down_price:.2f} (fair={prob_down:.3f})",
                order_type="GTC",
            ))

        if signals:
            self._logger.info(
                f"Quotes: Up bid={bid_up_price:.2f}/ask={ask_up_price:.2f} | "
                f"Down bid={bid_down_price:.2f}/ask={ask_down_price:.2f} | "
                f"spread={spread:.3f} | T-{market.seconds_remaining:.0f}s"
            )

        return signals

    def _generate_directional_signal(self, market: MarketInfo, direction: str) -> list[Signal]:
        """
        T-10s Directional Mode: place one-sided GTC maker order
        on the likely winning side for settlement profit.

        Called when CEX price is >0.1% from strike with <10s remaining.
        Dynamic pricing: more distance = higher price (more confident).
        """
        outcome = Outcome.UP if direction == "Up" else Outcome.DOWN

        # I1 FIX: Dynamic maker price based on distance from strike
        distance = abs(self._cex_price - self._strike_price) / self._strike_price
        if distance > 0.005:     # >0.5% from strike — very confident
            maker_price = 0.95
        elif distance > 0.002:   # >0.2% from strike
            maker_price = 0.92
        elif distance > 0.001:   # >0.1% from strike
            maker_price = 0.88
        else:
            maker_price = 0.85   # close to strike — need bigger edge

        order_size = self.config.strategy.order_size_usd

        signal = Signal(
            market=market,
            outcome=outcome,
            side=Side.BUY,
            price=maker_price,
            size_usd=order_size,
            confidence=0.95,
            reason=(
                f"MM T-10s directional: {direction} GTC@${maker_price:.2f} | "
                f"cex={self._cex_price:.2f} vs strike={self._strike_price:.2f} | "
                f"dist={distance:.4%} | T-{market.seconds_remaining:.0f}s"
            ),
            order_type="GTC",
        )

        self._logger.info(
            f"🎯 T-10s DIRECTIONAL: {direction} GTC@${maker_price:.2f} × ${order_size:.0f} | "
            f"T-{market.seconds_remaining:.0f}s"
        )

        return [signal]

    async def on_round_start(self, market: MarketInfo):
        """Reset round-specific tracking."""
        await super().on_round_start(market)
        self._cost_up = 0.0
        self._cost_down = 0.0
        self._last_quote_time = 0  # allow immediate first quote
        self._directional_sent = False  # reset for new round

    async def on_fill(self, signal: Signal, fill_price: float, fill_size: float):
        """Track cost per side for accurate P&L."""
        await super().on_fill(signal, fill_price, fill_size)
        cost = fill_price * fill_size
        if signal.side == Side.BUY:
            if signal.outcome == Outcome.UP:
                self._cost_up += cost   # spent money buying Up tokens
            else:
                self._cost_down += cost # spent money buying Down tokens
        else:
            # SELL: tokens sold = revenue = negative cost
            if signal.outcome == Outcome.UP:
                self._cost_up -= cost   # received money selling Up tokens
            else:
                self._cost_down -= cost # received money selling Down tokens

    async def on_round_end(self, market: MarketInfo, settled_outcome: Optional[str] = None):
        """Calculate round P&L and update state."""
        # Binary market settlement:
        #   "Up" wins  → Up tokens pay $1 each, Down tokens pay $0
        #   "Down" wins → Down tokens pay $1 each, Up tokens pay $0
        # Net cost per side = (bought - sold) in dollars
        # P&L = settlement_payout - net_cost_of_remaining_tokens
        payout = 0.0
        # net cost = positive means we're net long (spent > received)
        net_cost = self._cost_up + self._cost_down

        # Only tokens still HELD pay out; tokens already sold are captured in cost
        if settled_outcome == "Up":
            payout = max(0, self.state.current_position_up) * 1.0
        elif settled_outcome == "Down":
            payout = max(0, self.state.current_position_down) * 1.0

        pnl = payout - net_cost

        self.state.daily_pnl_usd += pnl
        self.state.total_pnl_usd += pnl

        if pnl > 0:
            self.state.rounds_won += 1
            self.state.consecutive_losses = 0
        elif pnl < 0:
            self.state.consecutive_losses += 1

        # CRIT4 FIX: Don't call super().on_round_end() — it would double-count rounds_traded
        self.state.rounds_traded += 1
        self.state.reset_round()  # clear position state for next round
        self._logger.info(
            f"Round P&L: ${pnl:+.2f} (payout=${payout:.2f} cost=${net_cost:.2f}) | "
            f"Daily: ${self.state.daily_pnl_usd:+.2f} | "
            f"Total: ${self.state.total_pnl_usd:+.2f} | "
            f"Win rate: {self.state.win_rate:.1%}"
        )
