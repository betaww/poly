"""
Strategy Base Class — defines the interface all strategies must implement.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..config import Config
from ..market_scanner import MarketInfo

logger = logging.getLogger(__name__)


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Outcome(str, Enum):
    UP = "Up"
    DOWN = "Down"


@dataclass
class Signal:
    """A trading signal produced by a strategy."""
    market: MarketInfo
    outcome: Outcome               # Which outcome to trade
    side: Side                     # BUY or SELL the outcome token
    price: float                   # limit price (0.01 - 0.99)
    size_usd: float                # order size in USD
    confidence: float = 0.5        # 0.0 - 1.0 confidence score
    reason: str = ""               # human-readable rationale
    order_type: str = "GTC"        # GTC, FOK, GTD

    @property
    def token_id(self) -> str:
        if self.outcome == Outcome.UP:
            return self.market.token_id_up
        return self.market.token_id_down

    @property
    def is_maker(self) -> bool:
        return self.order_type == "GTC"


@dataclass
class Quote:
    """A two-sided quote (bid + ask) for market making."""
    market: MarketInfo
    bid_up: Optional[Signal] = None     # Buy "Up" tokens
    ask_up: Optional[Signal] = None     # Sell "Up" tokens  
    bid_down: Optional[Signal] = None   # Buy "Down" tokens
    ask_down: Optional[Signal] = None   # Sell "Down" tokens


@dataclass
class StrategyState:
    """Runtime state tracked per strategy."""
    # P&L tracking
    total_pnl_usd: float = 0.0
    rounds_traded: int = 0
    rounds_won: int = 0
    consecutive_losses: int = 0

    # Current round
    current_position_up: float = 0.0     # shares held
    current_position_down: float = 0.0
    current_cost_basis: float = 0.0

    # Daily
    daily_pnl_usd: float = 0.0
    daily_volume_usd: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.rounds_traded == 0:
            return 0.0
        return self.rounds_won / self.rounds_traded

    def reset_round(self):
        self.current_position_up = 0.0
        self.current_position_down = 0.0
        self.current_cost_basis = 0.0

    def reset_daily(self):
        self.daily_pnl_usd = 0.0
        self.daily_volume_usd = 0.0


class BaseStrategy(ABC):
    """
    Abstract base class for all Polymarket strategies.

    Strategies produce Signals or Quotes that the engine executes.
    The strategy does NOT execute orders directly — that's the engine's job.
    """

    def __init__(self, config: Config, name: str = "BaseStrategy"):
        self.config = config
        self.name = name
        self.state = StrategyState()
        self.is_active = True
        self._logger = logging.getLogger(f"strategy.{name}")

    @abstractmethod
    async def on_market_update(self, market: MarketInfo) -> list[Signal]:
        """
        Called when market data updates (new prices, order book changes).
        Return a list of signals to execute, or empty list for no action.
        """
        ...

    async def on_round_start(self, market: MarketInfo):
        """Called when a new 5-minute round begins."""
        self.state.reset_round()
        self._logger.info(f"Round start: {market.asset.upper()} | {market.slug}")

    async def on_round_end(self, market: MarketInfo, settled_outcome: Optional[str] = None):
        """Called when a round settles."""
        self.state.rounds_traded += 1
        self._logger.info(
            f"Round end: {market.asset.upper()} | "
            f"Outcome={settled_outcome} | "
            f"PnL={self.state.daily_pnl_usd:+.2f}"
        )

    async def on_fill(self, signal: Signal, fill_price: float, fill_size: float):
        """Called when an order is filled."""
        cost = fill_price * fill_size
        if signal.outcome == Outcome.UP:
            if signal.side == Side.BUY:
                self.state.current_position_up += fill_size
            else:
                self.state.current_position_up -= fill_size
        else:
            if signal.side == Side.BUY:
                self.state.current_position_down += fill_size
            else:
                self.state.current_position_down -= fill_size

        self.state.current_cost_basis += cost if signal.side == Side.BUY else -cost
        self.state.daily_volume_usd += cost
        self._logger.info(
            f"Fill: {signal.side.value} {signal.outcome.value} @ {fill_price:.3f} x {fill_size:.1f} "
            f"(cost=${cost:.2f})"
        )

    def should_pause(self) -> bool:
        """Check if strategy should pause due to risk limits."""
        sc = self.config.strategy

        # Daily loss limit
        if self.state.daily_pnl_usd <= -sc.daily_loss_limit_usd:
            self._logger.warning(f"Daily loss limit hit: ${self.state.daily_pnl_usd:.2f}")
            return True

        # Consecutive loss pause
        if self.state.consecutive_losses >= sc.consecutive_loss_pause:
            self._logger.warning(f"Consecutive losses: {self.state.consecutive_losses}")
            return True

        return False
