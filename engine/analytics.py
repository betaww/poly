"""
E4: Real-time Performance Analytics Engine.

Provides:
  - Wilson score confidence intervals for win rate
  - Rolling Sharpe ratio
  - Maximum drawdown tracking
  - Per-round EV estimation
  - Rolling fill rate monitoring

Usage:
    analytics = PerformanceAnalytics()
    analytics.record_round(pnl=0.35, was_fill=True)
    stats = analytics.get_summary()
"""
import math
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RoundRecord:
    """Single round for analytics."""
    timestamp: float
    pnl: float
    was_fill: bool
    direction: str  # "Up" or "Down"
    confidence: float
    maker_price: float
    order_size: float


class PerformanceAnalytics:
    """
    E4: Tracks and computes real-time trading performance metrics.

    All metrics are computed incrementally — no need to re-scan history.
    """

    def __init__(self, window_size: int = 100):
        self._rounds: deque[RoundRecord] = deque(maxlen=window_size)
        self._all_pnls: deque[float] = deque(maxlen=10000)  # bounded to prevent memory leak

        # Running counters
        self.total_rounds: int = 0
        self.total_fills: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.total_pnl: float = 0.0

        # Drawdown tracking
        self._peak_pnl: float = 0.0
        self._max_drawdown: float = 0.0
        self._current_drawdown: float = 0.0

        # Daily tracking
        self._daily_start_pnl: float = 0.0
        self._daily_start_time: float = time.time()

    def record_round(self, pnl: float, was_fill: bool,
                     direction: str = "", confidence: float = 0.0,
                     maker_price: float = 0.0, order_size: float = 0.0):
        """Record a completed round."""
        record = RoundRecord(
            timestamp=time.time(), pnl=pnl, was_fill=was_fill,
            direction=direction, confidence=confidence,
            maker_price=maker_price, order_size=order_size,
        )
        self._rounds.append(record)
        self._all_pnls.append(pnl)

        self.total_rounds += 1
        if was_fill:
            self.total_fills += 1
        if pnl > 0:
            self.total_wins += 1
        elif pnl < 0:
            self.total_losses += 1

        self.total_pnl += pnl

        # Drawdown tracking
        if self.total_pnl > self._peak_pnl:
            self._peak_pnl = self.total_pnl
        self._current_drawdown = self._peak_pnl - self.total_pnl
        self._max_drawdown = max(self._max_drawdown, self._current_drawdown)

    # ─── Statistical Methods ──────────────────────────────────────────────

    @staticmethod
    def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """
        Wilson score confidence interval for win rate.

        More accurate than naive wins/n, especially for small samples.
        z=1.96 → 95% CI, z=1.645 → 90% CI.
        """
        if n == 0:
            return 0.0, 1.0
        p = wins / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        offset = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
        return max(0.0, center - offset), min(1.0, center + offset)

    def get_win_rate_ci(self, z: float = 1.96) -> tuple[float, float, float]:
        """Returns (point_estimate, lower_95, upper_95)."""
        n = self.total_fills  # only count rounds with fills
        wins = self.total_wins
        point = wins / max(1, n)
        lower, upper = self.wilson_ci(wins, n, z)
        return point, lower, upper

    def get_sharpe_ratio(self, window: int = 50) -> float:
        """
        Rolling Sharpe ratio (no risk-free rate for 5-min rounds).

        Sharpe = mean(returns) / std(returns) × sqrt(rounds_per_day)
        ~288 rounds/day for 5-min markets.
        """
        recent = [r.pnl for r in self._rounds if r.was_fill][-window:]
        if len(recent) < 5:
            return 0.0

        mean_r = sum(recent) / len(recent)
        variance = sum((r - mean_r) ** 2 for r in recent) / len(recent)
        std_r = math.sqrt(variance) if variance > 0 else 1e-6

        annualization = math.sqrt(288)  # 288 five-min rounds per day
        return (mean_r / std_r) * annualization

    def get_fill_rate(self, window: int = 50) -> float:
        """Rolling fill rate over last N rounds."""
        recent = list(self._rounds)[-window:]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.was_fill) / len(recent)

    def get_ev_per_round(self) -> float:
        """Expected value per round (including skipped rounds)."""
        if self.total_rounds == 0:
            return 0.0
        return self.total_pnl / self.total_rounds

    def get_ev_per_fill(self) -> float:
        """Expected value per filled round."""
        if self.total_fills == 0:
            return 0.0
        return self.total_pnl / self.total_fills

    # ─── Summary ──────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Complete performance summary."""
        wr_point, wr_lower, wr_upper = self.get_win_rate_ci()
        return {
            "total_rounds": self.total_rounds,
            "total_fills": self.total_fills,
            "fill_rate": self.get_fill_rate(),
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(wr_point, 4),
            "win_rate_95ci": (round(wr_lower, 4), round(wr_upper, 4)),
            "sharpe_ratio": round(self.get_sharpe_ratio(), 2),
            "ev_per_round": round(self.get_ev_per_round(), 4),
            "ev_per_fill": round(self.get_ev_per_fill(), 4),
            "max_drawdown": round(self._max_drawdown, 2),
            "current_drawdown": round(self._current_drawdown, 2),
            "peak_pnl": round(self._peak_pnl, 2),
        }
