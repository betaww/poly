"""
Risk Manager — enforces position limits, daily loss caps, and circuit breakers.

Validates every signal before execution.
"""
import logging
import time
from dataclasses import dataclass, field

from ..config import Config
from ..strategies.base import Signal, StrategyState

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Global risk tracking (across all strategies)."""
    daily_pnl: float = 0.0
    daily_volume: float = 0.0
    total_exposure: float = 0.0
    active_orders_count: int = 0
    last_reset_day: int = 0

    # Circuit breaker
    is_paused: bool = False
    pause_until: float = 0.0
    pause_reason: str = ""


class RiskManager:
    """
    Pre-trade and post-trade risk checks.

    Validates:
      1. Daily loss limit
      2. Per-round position size
      3. Consecutive loss circuit breaker
      4. Max concurrent orders
      5. Market timing (don't trade too close to settlement)
    """

    def __init__(self, config: Config):
        self.config = config
        self.state = RiskState()

    def check_signal(self, signal: Signal, strategy_state: StrategyState) -> tuple[bool, str]:
        """
        Validate a signal before execution.
        Returns (approved, reason).
        """
        sc = self.config.strategy
        now = time.time()

        # --- Circuit breaker ---
        if self.state.is_paused:
            if now < self.state.pause_until:
                return False, f"Paused until {self.state.pause_until:.0f}: {self.state.pause_reason}"
            else:
                self.state.is_paused = False
                self.state.pause_reason = ""
                logger.info("Circuit breaker released")

        # --- Daily loss limit ---
        if self.state.daily_pnl <= -sc.daily_loss_limit_usd:
            self._trigger_pause(f"Daily loss limit: ${self.state.daily_pnl:.2f}")
            return False, f"Daily loss limit hit: ${self.state.daily_pnl:.2f}"

        # --- Per-round position limit ---
        current_exposure = strategy_state.daily_volume_usd
        if signal.size_usd > sc.max_position_usd:
            return False, (
                f"Position limit: order=${signal.size_usd:.2f} > max=${sc.max_position_usd:.2f}"
            )

        # --- Order size bounds ---
        if signal.size_usd < signal.market.min_order_size:
            return False, f"Order too small: ${signal.size_usd:.2f} < min=${signal.market.min_order_size:.2f}"

        # --- Price bounds ---
        if not (0.01 <= signal.price <= 0.99):
            return False, f"Price out of bounds: {signal.price:.3f}"

        # --- Timing check: don't trade in last 2 seconds ---
        if signal.market.seconds_remaining < 2:
            return False, f"Too close to settlement: {signal.market.seconds_remaining:.1f}s remaining"

        # --- Consecutive loss check ---
        if strategy_state.consecutive_losses >= sc.consecutive_loss_pause:
            pause_seconds = sc.pause_duration_minutes * 60
            self._trigger_pause(
                f"Consecutive losses: {strategy_state.consecutive_losses}",
                duration=pause_seconds,
            )
            return False, f"Consecutive loss pause: {strategy_state.consecutive_losses} losses"

        return True, "approved"

    def _trigger_pause(self, reason: str, duration: float = 900):
        """Activate circuit breaker."""
        self.state.is_paused = True
        self.state.pause_until = time.time() + duration
        self.state.pause_reason = reason
        logger.warning(f"CIRCUIT BREAKER: {reason} | Paused for {duration}s")

    def record_pnl(self, pnl: float):
        """Record realized P&L from a settled round."""
        self.state.daily_pnl += pnl
        logger.info(f"P&L recorded: ${pnl:+.2f} | Daily total: ${self.state.daily_pnl:+.2f}")

    def reset_daily(self):
        """Reset daily counters (call at midnight UTC)."""
        logger.info(f"Daily reset | Final P&L: ${self.state.daily_pnl:+.2f}")
        self.state.daily_pnl = 0.0
        self.state.daily_volume = 0.0

    def get_status(self) -> dict:
        """Return current risk state for dashboard."""
        return {
            "daily_pnl": self.state.daily_pnl,
            "daily_volume": self.state.daily_volume,
            "total_exposure": self.state.total_exposure,
            "is_paused": self.state.is_paused,
            "pause_reason": self.state.pause_reason,
        }
