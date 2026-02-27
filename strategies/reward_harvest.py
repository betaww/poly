"""
Strategy 4: Liquidity Reward Harvester.

The lowest-risk strategy. Simply places qualifying limit orders on markets
that offer daily liquidity rewards, earning the reward pool.

Uses polymarket-apis' get_reward_markets() to find eligible markets.

How it works:
  1. Scan for markets with active rewards (rewardsDailyRate > 0)
  2. Place two-sided limit orders within rewardsMaxSpread of midpoint
  3. Keep orders alive to score reward points
  4. Collect daily reward distributions
"""
import logging
import time
from typing import Optional

from .base import BaseStrategy, Signal, Side, Outcome
from ..config import Config
from ..market_scanner import MarketInfo

logger = logging.getLogger(__name__)


class RewardHarvester(BaseStrategy):
    """
    Places qualifying orders on reward-eligible markets.
    
    Reward program rules:
      - Orders must be limit (maker) orders
      - Orders must be within rewardsMaxSpread of midpoint
      - Orders must meet rewardsMinSize
      - Rewards distributed daily proportional to qualifying order time
    """

    def __init__(self, config: Config):
        super().__init__(config, name="RewardHarvest")
        self._reward_markets: dict = {}   # condition_id -> reward info
        self._last_scan_time: float = 0
        self._scan_interval: float = 3600  # re-scan every hour

    async def scan_reward_markets(self):
        """
        Use polymarket-apis to find markets with active rewards.
        """
        try:
            from polymarket_apis import PolymarketClobClient
            clob = PolymarketClobClient()
            reward_markets = clob.get_reward_markets()
            if reward_markets:
                self._reward_markets = {
                    m.get("conditionId", ""): m
                    for m in reward_markets
                    if m.get("rewardsDailyRate", 0) > 0
                }
                self._logger.info(
                    f"Found {len(self._reward_markets)} markets with active rewards"
                )
        except ImportError:
            self._logger.warning("polymarket-apis not installed, cannot scan rewards")
        except Exception as e:
            self._logger.error(f"Reward scan failed: {e}")

        self._last_scan_time = time.time()

    def _get_reward_params(self, market: MarketInfo) -> Optional[dict]:
        """Check if this market has active rewards."""
        return self._reward_markets.get(market.condition_id)

    async def on_market_update(self, market: MarketInfo) -> list[Signal]:
        """
        Place qualifying orders on reward-eligible markets.
        """
        if self.should_pause():
            return []

        # Periodic re-scan
        if time.time() - self._last_scan_time > self._scan_interval:
            await self.scan_reward_markets()

        # Check if this market has rewards
        reward_info = self._get_reward_params(market)
        if not reward_info:
            return []

        # Get reward parameters
        max_spread = float(reward_info.get("rewardsMaxSpread", 0.05))
        min_size = float(reward_info.get("rewardsMinSize", 5))
        daily_rate = float(reward_info.get("rewardsDailyRate", 0))

        if daily_rate <= 0:
            return []

        # Calculate qualifying prices
        midpoint = market.midpoint
        half_spread = max_spread / 2.0
        tick = market.tick_size

        bid_price = round(midpoint - half_spread + 0.01, 2)  # inside max spread
        ask_price = round(midpoint + half_spread - 0.01, 2)

        signals = []

        # Place qualifying bid on "Up"
        if 0.01 <= bid_price <= 0.99:
            signals.append(Signal(
                market=market,
                outcome=Outcome.UP,
                side=Side.BUY,
                price=bid_price,
                size_usd=min_size,
                confidence=0.5,
                reason=f"Reward harvest: bid Up @ {bid_price:.2f} (daily_rate=${daily_rate:.2f})",
                order_type="GTC",
            ))

        # Place qualifying bid on "Down"
        if 0.01 <= (1.0 - ask_price) <= 0.99:
            signals.append(Signal(
                market=market,
                outcome=Outcome.DOWN,
                side=Side.BUY,
                price=1.0 - ask_price,
                size_usd=min_size,
                confidence=0.5,
                reason=f"Reward harvest: bid Down @ {1.0 - ask_price:.2f} (daily_rate=${daily_rate:.2f})",
                order_type="GTC",
            ))

        if signals:
            self._logger.info(
                f"Reward orders: {market.asset.upper()} | "
                f"rate=${daily_rate:.2f}/day | spread_limit={max_spread:.2f}"
            )

        return signals
