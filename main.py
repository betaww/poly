"""
Polymarket Trading System — Main Orchestrator.

Orchestrates:
  1. Market Scanner → discovers active 5-min markets
  2. Strategy Engine → generates trading signals
  3. Risk Manager → validates signals
  4. Order Executor → places/cancels orders
  5. Lifecycle → manages round transitions + settlement

Usage:
  # Paper mode (default):
  python -m user_data.polymarket.main

  # With specific strategy:
  python -m user_data.polymarket.main --strategy crypto_mm

  # Live mode:
  POLYMARKET_MODE=live POLYMARKET_PRIVATE_KEY=0x... python -m user_data.polymarket.main
"""
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional

from .config import Config
from .market_scanner import MarketScanner, MarketInfo
from .engine.order_executor import OrderExecutor
from .engine.risk_manager import RiskManager
from .strategies.base import BaseStrategy, Signal
from .strategies.crypto_mm import CryptoMarketMaker
from .strategies.oracle_arb import OracleArbStrategy
from .strategies.reward_harvest import RewardHarvester

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main trading engine that orchestrates all components.

    Flow per round:
      1. Scanner discovers current market
      2. Strategy.on_round_start() initializes
      3. Loop: fetch prices → strategy.on_market_update() → risk check → execute
      4. Market settles → strategy.on_round_end()
      5. Auto-transition to next round
    """

    def __init__(self, config: Config, strategy_name: str = "crypto_mm"):
        self.config = config
        self._scanner = MarketScanner(config)
        self._executor = OrderExecutor(config)
        self._risk = RiskManager(config)
        self._strategy = self._create_strategy(strategy_name)
        self._running = False
        self._current_market: Optional[MarketInfo] = None
        self._last_day: int = 0

        # Stats
        self._total_rounds: int = 0
        self._total_signals: int = 0
        self._total_fills: int = 0

    def _create_strategy(self, name: str) -> BaseStrategy:
        """Factory for strategies."""
        strategies = {
            "crypto_mm": lambda: CryptoMarketMaker(self.config),
            "oracle_arb": lambda: OracleArbStrategy(self.config),
            "reward_harvest": lambda: RewardHarvester(self.config),
        }
        if name not in strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
        logger.info(f"Strategy: {name}")
        return strategies[name]()

    async def start(self):
        """Main event loop."""
        self._running = True
        mode_label = "📝 PAPER" if self.config.mode == "paper" else "🔴 LIVE"

        logger.info(f"""
╔═══════════════════════════════════════════════════════╗
║  Polymarket 5-Min Crypto Trading Engine               ║
║  Mode: {mode_label:<46s}  ║
║  Strategy: {self._strategy.name:<42s}  ║
║  Assets: {', '.join(a.upper() for a in self.config.market.assets):<44s}  ║
║  Timeframe: {self.config.market.timeframe:<41s}  ║
╚═══════════════════════════════════════════════════════╝
        """)

        try:
            while self._running:
                await self._run_cycle()
                await asyncio.sleep(1)  # 1s tick
        except asyncio.CancelledError:
            logger.info("Engine stopped via cancellation")
        finally:
            await self._shutdown()

    async def _run_cycle(self):
        """Single engine cycle (runs every ~1 second)."""
        now = time.time()

        # --- Daily reset ---
        today = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
        if today != self._last_day:
            self._last_day = today
            self._risk.reset_daily()
            self._strategy.state.reset_daily()
            logger.info(f"=== New day: {today} ===")

        # --- Discover markets ---
        markets = await self._scanner.discover_current_markets()
        if not markets:
            return

        # Pick best market for primary asset
        primary_asset = self.config.market.assets[0]
        market = None
        for m in markets:
            if m.asset == primary_asset and not m.is_expired:
                if market is None or m.seconds_remaining > market.seconds_remaining:
                    market = m

        if not market:
            return

        # --- Round transition detection ---
        if self._current_market is None or self._current_market.slug != market.slug:
            # New round started
            if self._current_market is not None:
                # Previous round ended
                await self._handle_round_end(self._current_market)

            self._current_market = market
            self._total_rounds += 1
            await self._strategy.on_round_start(market)
            logger.info(
                f"━━━ Round #{self._total_rounds}: {market.asset.upper()} | "
                f"slug={market.slug[-15:]} | "
                f"T-{market.seconds_remaining:.0f}s ━━━"
            )

        # --- Check if current market expired ---
        if market.is_expired:
            await self._handle_round_end(market)
            self._current_market = None
            return

        # --- Generate signals ---
        signals = await self._strategy.on_market_update(market)

        if signals:
            # Cancel stale orders before placing new quotes
            await self._executor.cancel_all()

        # --- Validate and execute ---
        for sig in signals:
            approved, reason = self._risk.check_signal(sig, self._strategy.state)
            if approved:
                record = await self._executor.place_order(sig)
                self._total_signals += 1

                if record.status == "matched":
                    self._total_fills += 1
                    await self._strategy.on_fill(sig, record.fill_price, record.fill_size)
            else:
                logger.debug(f"Signal rejected: {reason}")

    async def _handle_round_end(self, market: MarketInfo):
        """Handle round settlement."""
        # Cancel any remaining open orders
        await self._executor.cancel_all()

        # In paper mode, simulate settlement
        settled = "Up" if market.price_up > 0.5 else "Down"

        await self._strategy.on_round_end(market, settled)
        # Sync strategy P&L to risk manager
        self._risk.state.daily_pnl = self._strategy.state.daily_pnl_usd
        logger.info(
            f"Round complete | Settled: {settled} | "
            f"Total P&L: ${self._strategy.state.total_pnl_usd:+.2f} | "
            f"Win rate: {self._strategy.state.win_rate:.1%}"
        )

    async def _shutdown(self):
        """Cleanup on exit."""
        logger.info("Shutting down...")
        await self._executor.cancel_all()
        await self._scanner.close()
        self._print_summary()

    def _print_summary(self):
        """Print final session summary."""
        s = self._strategy.state
        logger.info(f"""
╔═══════════════ Session Summary ═══════════════╗
║  Rounds:     {self._total_rounds:<35d}║
║  Signals:    {self._total_signals:<35d}║
║  Fills:      {self._total_fills:<35d}║
║  Win Rate:   {s.win_rate:<35.1%}║
║  Total P&L:  ${s.total_pnl_usd:<+34.2f}║
║  Daily P&L:  ${s.daily_pnl_usd:<+34.2f}║
╚═══════════════════════════════════════════════╝
        """)

    def stop(self):
        """Signal the engine to stop."""
        self._running = False


async def run(strategy: str = "crypto_mm"):
    """Entry point for the trading engine."""
    config = Config.from_env()

    engine = TradingEngine(config, strategy_name=strategy)

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()

    def _signal_handler():
        logger.info("Received shutdown signal")
        engine.stop()

    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, _signal_handler)

    await engine.start()


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket 5-Min Crypto Trading Engine")
    parser.add_argument(
        "--strategy", "-s",
        default="crypto_mm",
        choices=["crypto_mm", "oracle_arb", "reward_harvest"],
        help="Strategy to run (default: crypto_mm)",
    )
    parser.add_argument(
        "--assets", "-a",
        nargs="+",
        default=["btc", "eth"],
        help="Assets to trade (default: btc eth)",
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="5m",
        choices=["5m", "15m", "1h", "4h"],
        help="Timeframe (default: 5m)",
    )
    parser.add_argument(
        "--mode", "-m",
        default="paper",
        choices=["paper", "live"],
        help="Trading mode (default: paper)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)-20s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Apply CLI args to env
    import os
    os.environ["POLYMARKET_MODE"] = args.mode

    asyncio.run(run(strategy=args.strategy))


if __name__ == "__main__":
    main()
