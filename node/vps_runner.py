"""
VPS Node Runner — the core trading node deployed closest to Polymarket CLOB.

Supports running multiple strategies simultaneously with persistent trade recording.

Runs:
  - Market Scanner (discover 5-min markets)
  - Multiple Strategies (crypto_mm + oracle_arb)
  - Redis Consumer (receive brain predictions)
  - Order Executor (py-clob-client)
  - Risk Manager
  - Trade Ledger (SQLite persistence)

Usage:
  python -m user_data.polymarket.node.vps_runner
  python -m user_data.polymarket.node.vps_runner -s crypto_mm,oracle_arb
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
from ..engine.trade_ledger import TradeLedger
from ..strategies.base import BaseStrategy
from ..strategies.crypto_mm import CryptoMarketMaker
from ..strategies.oracle_arb import OracleArbStrategy
from ..strategies.reward_harvest import RewardHarvester
from ..engine.clob_book_feed import CLOBBookFeed
from .redis_consumer import RedisConsumer

logger = logging.getLogger(__name__)


class StrategySlot:
    """Wraps a strategy with its own state tracking."""
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.name = strategy.name
        self.current_market: MarketInfo | None = None
        self.round_signals = 0
        self.round_fills = 0
        self.round_cost = 0.0
        self.predicted_direction = ""
        self.pnl_before_round = 0.0  # snapshot for per-round delta


class VPSRunner:
    """
    Production VPS trading engine — supports multiple strategies.
    """

    STRATEGY_REGISTRY = {
        "crypto_mm": lambda c: CryptoMarketMaker(c),
        "oracle_arb": lambda c: OracleArbStrategy(c),
        "reward_harvest": lambda c: RewardHarvester(c),
    }

    def __init__(self, config: Config, strategy_name: str = "crypto_mm"):
        self.config = config
        self._scanner = MarketScanner(config)
        self._executor = OrderExecutor(config)
        self._risk = RiskManager(config)
        self._redis = RedisConsumer(config)

        # Trade ledger for persistent recording
        db_path = "data/paper_trades.db"
        self._ledger = TradeLedger(db_path)

        # Support comma-separated strategy names: "crypto_mm,oracle_arb"
        strategy_names = [s.strip() for s in strategy_name.split(",")]
        self._slots: list[StrategySlot] = []
        for name in strategy_names:
            if name not in self.STRATEGY_REGISTRY:
                raise ValueError(f"Unknown: {name}. Available: {list(self.STRATEGY_REGISTRY.keys())}")
            strat = self.STRATEGY_REGISTRY[name](config)
            self._slots.append(StrategySlot(strat))

        self._running = False
        self._last_day = 0
        self._total_rounds = 0
        self._total_signals = 0
        self._telemetry_interval = 2.0
        self._last_telemetry = 0.0
        # CLOB orderbook feed for real-time depth
        self._book_feed = CLOBBookFeed()
        self._book_subscribed_tokens: set[str] = set()

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
            if "spread" in data:
                self.config.strategy.base_spread = float(data["spread"])
                logger.info(f"Config updated: spread={data['spread']}")

    def _feed_predictions_to_strategy(self, slot: StrategySlot, market: MarketInfo):
        """Push brain predictions into strategy."""
        pred = self._redis.get_prediction(market.asset)
        if pred is None:
            return

        strat = slot.strategy
        if isinstance(strat, CryptoMarketMaker):
            strat.set_cex_price(pred.cex_price)
            strat.set_volatility(pred.volatility)
            # FIX: CryptoMM needs strike_price for T-10s directional mode
            if market.strike_price > 0:
                strat.set_strike_price(market.strike_price)
            # BLIND SPOT FIX: Feed real CLOB depth to CryptoMM
            book_up = self._book_feed.get_book(market.token_id_up)
            book_down = self._book_feed.get_book(market.token_id_down)
            if book_up and book_down:
                strat.set_book_data(
                    best_bid_up=book_up.best_bid, best_ask_up=book_up.best_ask,
                    best_bid_down=book_down.best_bid, best_ask_down=book_down.best_ask,
                    depth_bid_usd=book_up.total_bid_depth_usd + book_down.total_bid_depth_usd,
                    depth_ask_usd=book_up.total_ask_depth_usd + book_down.total_ask_depth_usd,
                )
        elif isinstance(strat, OracleArbStrategy):
            strat.set_cex_price(pred.cex_price)
            # BLIND SPOT FIX: OracleArb also needs volatility for sizing
            strat.set_volatility(pred.volatility)

        # Track predicted direction for round recording
        if pred.cex_price > 0 and market.strike_price > 0:
            slot.predicted_direction = "Up" if pred.cex_price > market.strike_price else "Down"

    def _publish_telemetry(self):
        """Publish current state to Mac dashboard."""
        now = time.time()
        if now - self._last_telemetry < self._telemetry_interval:
            return
        self._last_telemetry = now

        # Aggregate across strategies
        total_pnl = sum(s.strategy.state.total_pnl_usd for s in self._slots)
        daily_pnl = sum(s.strategy.state.daily_pnl_usd for s in self._slots)
        total_rounds = sum(s.strategy.state.rounds_traded for s in self._slots)
        total_wins = sum(s.strategy.state.rounds_won for s in self._slots)

        strat_info = {s.name: {
            "pnl": round(s.strategy.state.total_pnl_usd, 2),
            "win_rate": round(s.strategy.state.win_rate, 3),
            "rounds": s.strategy.state.rounds_traded,
        } for s in self._slots}

        # L6 FIX: Include all fields the dashboard expects
        self._redis.publish_telemetry({
            "strategies": strat_info,
            "strategy": ", ".join(s.name for s in self._slots),
            "mode": self.config.mode,
            "round": self._total_rounds,
            "rounds_traded": total_rounds,
            "signals": self._total_signals,
            "daily_pnl": round(daily_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(total_wins / max(1, total_rounds), 3),
            "position_up": sum(s.strategy.state.current_position_up for s in self._slots),
            "position_down": sum(s.strategy.state.current_position_down for s in self._slots),
            "risk": {
                "is_paused": self._risk.state.is_paused,
                "pause_reason": self._risk.state.pause_reason,
                "daily_volume": abs(daily_pnl),
            },
            "brain_connected": self._redis.is_connected,
            "timestamp": now,
        })

    async def start(self):
        """Main event loop."""
        self._running = True
        mode_label = "📝 PAPER" if self.config.mode == "paper" else "🔴 LIVE"
        strat_names = ", ".join(s.name for s in self._slots)

        self._redis.start(control_callback=self._handle_control)
        brain_status = "✅ Connected" if self._redis.is_connected else "⚠️  Standalone"

        logger.info(f"""
╔═══════════════════════════════════════════════════════════╗
║  Polymarket VPS Node (London)                             ║
║  Mode: {mode_label:<50s}║
║  Strategies: {strat_names:<44s}║
║  Assets: {', '.join(a.upper() for a in self.config.market.assets):<48s}║
║  Brain: {brain_status:<49s}║
║  Ledger: SQLite (data/paper_trades.db)                    ║
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
        """Single engine cycle — runs all strategies on all markets."""
        # Daily reset
        today = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
        if today != self._last_day:
            self._last_day = today
            self._risk.reset_daily()
            for slot in self._slots:
                slot.strategy.state.reset_daily()

        # Discover markets
        markets = await self._scanner.discover_current_markets()
        if not markets:
            return

        # H6 FIX: Pick best market for EACH configured asset (not just primary)
        best_markets: dict[str, MarketInfo] = {}
        for m in markets:
            if m.is_expired:
                continue
            if m.asset not in best_markets or m.seconds_remaining > best_markets[m.asset].seconds_remaining:
                best_markets[m.asset] = m

        if not best_markets:
            return

        # Run each strategy on its assigned asset's market
        # Each slot tracks one market at a time via current_market.
        # With N slots and M assets, distribute: slot[i] gets asset[i % M]
        assets = self.config.market.assets
        for i, slot in enumerate(self._slots):
            asset = assets[i % len(assets)] if assets else None
            if not asset:
                continue
            market = best_markets.get(asset)
            if not market:
                # Fallback: try any available market
                market = next(iter(best_markets.values()), None)
            if market:
                await self._run_strategy_on_market(slot, market)

        # Publish telemetry
        self._publish_telemetry()

    async def _run_strategy_on_market(self, slot: StrategySlot, market: MarketInfo):
        """Run a single strategy on a market."""
        # Round transition
        if slot.current_market is None or slot.current_market.slug != market.slug:
            if slot.current_market:
                await self._handle_round_end(slot, slot.current_market)
            slot.current_market = market
            slot.round_signals = 0
            slot.round_fills = 0
            slot.round_cost = 0.0
            slot.predicted_direction = ""
            slot.pnl_before_round = slot.strategy.state.total_pnl_usd  # snapshot
            self._total_rounds += 1
            await slot.strategy.on_round_start(market)

            # BLIND SPOT FIX: Subscribe to CLOB orderbook for new market tokens
            for token_id, (asset, outcome) in [
                (market.token_id_up, (market.asset, "Up")),
                (market.token_id_down, (market.asset, "Down")),
            ]:
                if token_id and token_id not in self._book_subscribed_tokens:
                    self._book_subscribed_tokens.add(token_id)
                    if not self._book_feed.is_connected:
                        # First market: start the feed with initial tokens
                        self._book_feed.start({
                            market.token_id_up: (market.asset, "Up"),
                            market.token_id_down: (market.asset, "Down"),
                        })
                    else:
                        # Later market: subscribe additional tokens
                        self._book_feed.subscribe_token(token_id, asset, outcome)

        if market.is_expired:
            await self._handle_round_end(slot, market)
            slot.current_market = None
            return

        # Feed brain predictions
        self._feed_predictions_to_strategy(slot, market)

        # CRIT2 FIX: Poll for live order fills (live mode only)
        if self.config.mode != "paper":
            filled_records = await self._executor.poll_live_fills()
            for record in filled_records:
                sig = record.signal
                await slot.strategy.on_fill(sig, record.fill_price, record.fill_size)
                slot.round_fills += 1
                slot.round_cost += record.fill_price * record.fill_size
                self._risk.record_exposure(market.asset, record.fill_price * record.fill_size)

        # Generate signals
        signals = await slot.strategy.on_market_update(market)

        if signals:
            # CRIT3 FIX: Don't cancel ALL orders — our GTC makers need time to rest.
            # Only cancel orders from PREVIOUS rounds (older than current round start).
            # Round-end already calls cancel_all() for cleanup.
            pass  # removed blanket cancel_all — let GTC orders rest

        # Validate and execute
        for sig in signals:
            approved, reason = self._risk.check_signal(sig, slot.strategy.state)
            if approved:
                record = await self._executor.place_order(sig)
                self._total_signals += 1
                slot.round_signals += 1

                # Record to ledger
                if hasattr(self._executor, '_simulator') and self._executor._simulator:
                    from ..engine.paper_simulator import SimulationResult
                    result = SimulationResult(
                        filled=(record.status == "matched"),
                        fill_price=record.fill_price,
                        fill_size=record.fill_size,
                    )
                    self._ledger.record_order(slot.name, sig, result)

                if record.status == "matched":
                    await slot.strategy.on_fill(sig, record.fill_price, record.fill_size)
                    slot.round_fills += 1
                    slot.round_cost += record.fill_price * record.fill_size
                    # Track per-asset exposure for risk manager
                    self._risk.record_exposure(market.asset, record.fill_price * record.fill_size)

    async def _handle_round_end(self, slot: StrategySlot, market: MarketInfo):
        """Handle end of round for a strategy."""
        await self._executor.cancel_all()

        # C2 FIX: Use CEX price vs strike for settlement, not market.price_up
        pred = self._redis.get_prediction(market.asset)
        cex_price = pred.cex_price if pred else 0
        if cex_price > 0 and market.strike_price > 0:
            settled = "Up" if cex_price > market.strike_price else "Down"
        else:
            # Fallback: use market price if no CEX data
            settled = "Up" if market.price_up > 0.5 else "Down"
            logger.warning(f"No CEX price for settlement, using market price fallback")

        await slot.strategy.on_round_end(market, settled)

        # C3 FIX: Record per-round PnL delta, not cumulative
        round_pnl = slot.strategy.state.total_pnl_usd - slot.pnl_before_round

        self._ledger.record_round(
            strategy=slot.name,
            market=market,
            settled=settled,
            predicted=slot.predicted_direction,
            pnl=round_pnl,
            cost=slot.round_cost,
            signals=slot.round_signals,
            fills=slot.round_fills,
            cex_price=cex_price,
        )

        # Sync to risk manager
        total_daily = sum(s.strategy.state.daily_pnl_usd for s in self._slots)
        self._risk.state.daily_pnl = total_daily
        # Release per-asset exposure since positions are settled
        self._risk.release_exposure(market.asset, slot.round_cost)

    async def _shutdown(self):
        await self._executor.cancel_all()
        await self._scanner.close()
        self._redis.stop()
        self._book_feed.stop()

        # Print per-strategy summary
        for slot in self._slots:
            s = slot.strategy.state
            logger.info(f"""
╔═══════════ {slot.name} Summary ════════════╗
║  Rounds:    {s.rounds_traded:<34d}║
║  Signals:   {slot.round_signals:<34d}║
║  Win Rate:  {s.win_rate:<34.1%}║
║  Total P&L: ${s.total_pnl_usd:<+33.2f}║
╚════════════════════════════════════════════╝
            """)

        # Print ledger stats
        try:
            stats = self._ledger.get_stats(days=1)
            logger.info(
                f"Today: {stats['total_rounds']} rounds, "
                f"{stats['total_wins']} wins ({stats['win_rate']}), "
                f"P&L: {stats['total_pnl']}"
            )
        except Exception:
            pass

        self._ledger.close()

    def stop(self):
        self._running = False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket VPS Node")
    parser.add_argument("-s", "--strategy", default="crypto_mm",
                        help="Strategy name(s), comma-separated: crypto_mm,oracle_arb")
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
