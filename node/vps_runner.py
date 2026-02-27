"""
VPS Node Runner — the core trading node deployed closest to Polymarket CLOB.

Runs:
  - Market Scanner (discover 5-min markets)
  - Strategies (crypto_mm, oracle_arb, reward_harvest)
  - Redis Consumer (receive brain predictions)
  - Order Executor (py-clob-client)
  - Risk Manager

This is the main production entry point for VPS deployment.

Usage:
  python -m user_data.polymarket.node.vps_runner

  # With env vars:
  POLY_REDIS_HOST=100.76.x.x POLYMARKET_PRIVATE_KEY=0x... python -m ...
"""
import asyncio
import logging
import signal
import time
from datetime import datetime, timezone

from ..config import Config
from ..market_scanner import MarketScanner, MarketInfo
from ..engine.order_executor import OrderExecutor
from ..engine.risk_manager import RiskManager
from ..strategies.base import BaseStrategy
from ..strategies.crypto_mm import CryptoMarketMaker
from ..strategies.oracle_arb import OracleArbStrategy
from ..strategies.reward_harvest import RewardHarvester
from .redis_consumer import RedisConsumer

logger = logging.getLogger(__name__)


class VPSRunner:
    """
    Production VPS trading engine.
    
    Key difference from standalone main.py:
      - Receives price predictions from Alienware brain via Redis
      - Publishes telemetry to Mac dashboard via Redis
      - Handles control commands (PANIC_STOP, PAUSE, etc.)
    """

    def __init__(self, config: Config, strategy_name: str = "crypto_mm"):
        self.config = config
        self._scanner = MarketScanner(config)
        self._executor = OrderExecutor(config)
        self._risk = RiskManager(config)
        self._redis = RedisConsumer(config)
        self._strategy = self._create_strategy(strategy_name)
        self._strategy_name = strategy_name

        self._running = False
        self._current_market = None
        self._last_day = 0
        self._total_rounds = 0
        self._total_signals = 0
        self._telemetry_interval = 2.0  # publish telemetry every 2s
        self._last_telemetry = 0.0

    def _create_strategy(self, name: str) -> BaseStrategy:
        strategies = {
            "crypto_mm": lambda: CryptoMarketMaker(self.config),
            "oracle_arb": lambda: OracleArbStrategy(self.config),
            "reward_harvest": lambda: RewardHarvester(self.config),
        }
        if name not in strategies:
            raise ValueError(f"Unknown: {name}. Available: {list(strategies.keys())}")
        return strategies[name]()

    def _handle_control(self, command: str, data: dict):
        """Handle control commands from Mac dashboard."""
        if command == "PANIC_STOP":
            logger.warning("🚨 PANIC STOP received from dashboard!")
            self._running = False
        elif command == "PAUSE":
            self._risk.state.is_paused = True
            self._risk.state.pause_reason = "Dashboard pause"
            self._risk.state.pause_until = time.time() + 3600
            logger.warning("⏸️  PAUSED by dashboard")
        elif command == "RESUME":
            self._risk.state.is_paused = False
            self._risk.state.pause_reason = ""
            self._risk.state.pause_until = 0
            logger.info("▶️  RESUMED by dashboard")
        elif command == "UPDATE_CONFIG":
            # Hot-reload strategy params
            if "spread" in data:
                self.config.strategy.base_spread = float(data["spread"])
                logger.info(f"Config updated: spread={data['spread']}")

    def _feed_predictions_to_strategy(self, market: MarketInfo):
        """Push brain predictions into strategy."""
        pred = self._redis.get_prediction(market.asset)
        if pred is None:
            return  # no prediction available — strategy uses market fallback

        # Feed CEX price and volatility to strategy
        if isinstance(self._strategy, CryptoMarketMaker):
            self._strategy.set_cex_price(pred.cex_price)
            self._strategy.set_volatility(pred.volatility)
        elif isinstance(self._strategy, OracleArbStrategy):
            self._strategy.set_cex_price(pred.cex_price)

    def _publish_telemetry(self):
        """Publish current state to Mac dashboard."""
        now = time.time()
        if now - self._last_telemetry < self._telemetry_interval:
            return
        self._last_telemetry = now

        s = self._strategy.state
        risk = self._risk.get_status()

        self._redis.publish_telemetry({
            "strategy": self._strategy_name,
            "mode": self.config.mode,
            "round": self._total_rounds,
            "signals": self._total_signals,
            "daily_pnl": round(s.daily_pnl_usd, 2),
            "total_pnl": round(s.total_pnl_usd, 2),
            "win_rate": round(s.win_rate, 3),
            "rounds_traded": s.rounds_traded,
            "position_up": round(s.current_position_up, 2),
            "position_down": round(s.current_position_down, 2),
            "risk": risk,
            "current_market": self._current_market.slug if self._current_market else None,
            "brain_connected": self._redis.is_connected,
            "timestamp": now,
        })

    async def start(self):
        """Main event loop."""
        self._running = True
        mode_label = "📝 PAPER" if self.config.mode == "paper" else "🔴 LIVE"

        # Start Redis consumer
        self._redis.start(control_callback=self._handle_control)
        brain_status = "✅ Connected" if self._redis.is_connected else "⚠️  Standalone"

        logger.info(f"""
╔═══════════════════════════════════════════════════════════╗
║  Polymarket VPS Node (London)                             ║
║  Mode: {mode_label:<50s}║
║  Strategy: {self._strategy.name:<46s}║
║  Assets: {', '.join(a.upper() for a in self.config.market.assets):<48s}║
║  Brain: {brain_status:<49s}║
║  Redis: {f"{self.config.redis.host}:{self.config.redis.port}":<49s}║
╚═══════════════════════════════════════════════════════════╝
        """)

        try:
            while self._running:
                await self._run_cycle()
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    async def _run_cycle(self):
        """Single engine cycle."""
        # Daily reset
        today = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
        if today != self._last_day:
            self._last_day = today
            self._risk.reset_daily()
            self._strategy.state.reset_daily()

        # Discover markets
        markets = await self._scanner.discover_current_markets()
        if not markets:
            return

        # Pick best market
        primary = self.config.market.assets[0]
        market = None
        for m in markets:
            if m.asset == primary and not m.is_expired:
                if market is None or m.seconds_remaining > market.seconds_remaining:
                    market = m

        if not market:
            return

        # Round transition
        if self._current_market is None or self._current_market.slug != market.slug:
            if self._current_market:
                await self._handle_round_end(self._current_market)
            self._current_market = market
            self._total_rounds += 1
            await self._strategy.on_round_start(market)

        if market.is_expired:
            await self._handle_round_end(market)
            self._current_market = None
            return

        # Feed brain predictions into strategy
        self._feed_predictions_to_strategy(market)

        # Generate signals
        signals = await self._strategy.on_market_update(market)

        if signals:
            # Cancel stale orders before placing new quotes (cancel-before-requote)
            # This prevents order accumulation and reduces adverse selection risk
            await self._executor.cancel_all()

        # Validate and execute
        for sig in signals:
            approved, reason = self._risk.check_signal(sig, self._strategy.state)
            if approved:
                record = await self._executor.place_order(sig)
                self._total_signals += 1
                if record.status == "matched":
                    await self._strategy.on_fill(sig, record.fill_price, record.fill_size)

        # Publish telemetry
        self._publish_telemetry()

    async def _handle_round_end(self, market: MarketInfo):
        await self._executor.cancel_all()
        settled = "Up" if market.price_up > 0.5 else "Down"
        await self._strategy.on_round_end(market, settled)
        # Sync strategy P&L to risk manager
        self._risk.state.daily_pnl = self._strategy.state.daily_pnl_usd

    async def _shutdown(self):
        await self._executor.cancel_all()
        await self._scanner.close()
        self._redis.stop()
        s = self._strategy.state
        logger.info(f"""
╔═══════════ VPS Session Summary ════════════╗
║  Rounds:    {self._total_rounds:<34d}║
║  Signals:   {self._total_signals:<34d}║
║  Win Rate:  {s.win_rate:<34.1%}║
║  Total P&L: ${s.total_pnl_usd:<+33.2f}║
╚════════════════════════════════════════════╝
        """)

    def stop(self):
        self._running = False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket VPS Node")
    parser.add_argument("-s", "--strategy", default="crypto_mm",
                        choices=["crypto_mm", "oracle_arb", "reward_harvest"])
    parser.add_argument("-m", "--mode", default="paper", choices=["paper", "live"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)-25s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    import os
    os.environ["POLYMARKET_MODE"] = args.mode

    config = Config.from_env()
    config.node = "vps"
    runner = VPSRunner(config, strategy_name=args.strategy)

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, runner.stop)
    loop.add_signal_handler(signal.SIGTERM, runner.stop)

    asyncio.run(runner.start())


if __name__ == "__main__":
    main()
