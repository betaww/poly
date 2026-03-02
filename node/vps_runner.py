"""
VPS Node Runner — the core trading node deployed closest to Polymarket CLOB.

Supports running multiple strategies simultaneously with persistent trade recording.

Runs:
  - Market Scanner (discover 5-min markets)
  - Strategy Engine (default: DirectionalSniper v6)
  - Redis Consumer (receive brain predictions)
  - Order Executor (py-clob-client)
  - Risk Manager
  - Trade Ledger (SQLite persistence)

Usage:
  python -m user_data.polymarket.node.vps_runner
  python -m user_data.polymarket.node.vps_runner -s sniper
"""
import asyncio
import logging
import random
import signal
import time
from datetime import datetime, timezone

import httpx

from ..config import Config
from ..market_scanner import MarketScanner, MarketInfo
from ..engine.order_executor import OrderExecutor
from ..engine.risk_manager import RiskManager
from ..engine.trade_ledger import TradeLedger
from ..strategies.base import BaseStrategy
from ..strategies.crypto_mm import CryptoMarketMaker
from ..strategies.oracle_arb import OracleArbStrategy
from ..strategies.reward_harvest import RewardHarvester
from ..strategies.directional_sniper import DirectionalSniper
from ..engine.clob_book_feed import CLOBBookFeed
from ..engine.alerting import AlertManager
from ..engine.analytics import PerformanceAnalytics
from ..engine.settlement_verifier import SettlementVerifier
from ..engine.binance_depth_feed import BinanceDepthFeed
from .redis_consumer import RedisConsumer

logger = logging.getLogger(__name__)


class StrategySlot:
    """Wraps a strategy with its own state tracking."""
    def __init__(self, strategy: BaseStrategy, asset: str = ""):
        self.strategy = strategy
        self.name = strategy.name
        self.asset = asset  # M6 FIX: each slot is bound to one asset
        self.current_market: MarketInfo | None = None
        self.round_signals = 0
        self.round_fills = 0
        self.round_cost = 0.0
        self.predicted_direction = ""
        self.pnl_before_round = 0.0  # snapshot for per-round delta
        self._settlement_cex_snapshot: float = 0.0  # C4 FIX: explicit init
        # v10 FIX #2: Strike freshness TTL
        self._strike_set_at: float = 0.0  # when strike was last set


class VPSRunner:
    """
    Production VPS trading engine — supports multiple strategies.
    """

    STRATEGY_REGISTRY = {
        "sniper": lambda c: DirectionalSniper(c),
        "crypto_mm": lambda c: CryptoMarketMaker(c),
        "oracle_arb": lambda c: OracleArbStrategy(c),
        "reward_harvest": lambda c: RewardHarvester(c),
    }

    def __init__(self, config: Config, strategy_name: str = "sniper"):
        self.config = config
        self._scanner = MarketScanner(config)
        self._executor = OrderExecutor(config)
        self._risk = RiskManager(config)
        self._redis = RedisConsumer(config)

        # Trade ledger for persistent recording
        db_path = "data/paper_trades.db"
        self._ledger = TradeLedger(db_path)

        # M6 FIX: Create separate slots per (strategy, asset) pair.
        # Each slot tracks its own current_market independently,
        # preventing round-thrashing when multiple assets are active.
        strategy_names = [s.strip() for s in strategy_name.split(",")]
        self._slots: list[StrategySlot] = []
        for name in strategy_names:
            if name not in self.STRATEGY_REGISTRY:
                raise ValueError(f"Unknown: {name}. Available: {list(self.STRATEGY_REGISTRY.keys())}")
            for asset in config.market.assets:
                strat = self.STRATEGY_REGISTRY[name](config)
                strat.name = f"{strat.name}-{asset}"  # e.g. "Sniper-btc"
                self._slots.append(StrategySlot(strat, asset=asset))

        self._running = False
        self._last_day = 0
        self._total_rounds = 0
        self._total_signals = 0
        self._telemetry_interval = 2.0
        self._last_telemetry = 0.0
        # CLOB orderbook feed for real-time depth
        self._book_feed = CLOBBookFeed()
        self._book_subscribed_tokens: set[str] = set()
        # E8: Alert manager
        self._alerter = AlertManager(config)
        self._brain_offline_since: float = 0.0
        self._coinflip_count: int = 0
        # E4: Performance analytics
        self._analytics = PerformanceAnalytics()
        # Settlement verifier: compare paper vs Polymarket actual
        self._verifier = SettlementVerifier()
        # D3: Binance L2 depth feed for OFI signal
        self._depth_feed = BinanceDepthFeed()
        self._depth_feed.start(config.market.assets)
        # E3: Wire CLOB feed into executor for real depth
        self._executor._clob_feed = self._book_feed
        # v10 FIX #14: Clock offset tracking (VPS vs Polymarket server)
        self._clock_offset: float = 0.0  # seconds; positive = VPS ahead
        self._last_clock_sync: float = 0.0
        # v11 #15: Cross-asset correlation tracking
        self._slot_signals_this_cycle: dict[str, str] = {}  # {asset: direction}

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

    async def _feed_predictions_to_strategy(self, slot: StrategySlot, market: MarketInfo):
        """Push brain predictions into strategy."""
        pred = self._redis.get_prediction(market.asset)

        # v10 FIX #12: VPS fallback oracle when Brain is offline
        if pred is None or (pred and pred.cex_price <= 0):
            if market.seconds_remaining < 15:
                # Near commitment window with no Brain — try HTTP fallback
                fallback_price = await self._vps_fallback_cex_price(market.asset)
                if fallback_price > 0:
                    strat = slot.strategy
                    if hasattr(strat, 'set_cex_price'):
                        strat.set_cex_price(fallback_price)
                    if hasattr(strat, 'set_strike_price') and market.strike_price > 0:
                        strat.set_strike_price(market.strike_price)
                    logger.info(
                        f"🔄 Brain offline fallback: {market.asset.upper()} "
                        f"CEX=${fallback_price:,.2f} (HTTP)"
                    )
                    # Track prediction direction from fallback
                    if fallback_price > 0 and market.strike_price > 0:
                        slot.predicted_direction = "Up" if fallback_price > market.strike_price else "Down"
                    if market.seconds_remaining < 3:
                        slot._settlement_cex_snapshot = fallback_price
                    # Still try CLOB / OFI below
                else:
                    if market.seconds_remaining < 12:
                        logger.warning(
                            f"⚠️ No prediction for {slot.name} ({market.asset}) | T-{market.seconds_remaining:.0f}s"
                        )
                    return
            else:
                return

        if pred and pred.cex_price > 0:
            strat = slot.strategy
            # Universal setters for all strategies that have them
            if hasattr(strat, 'set_cex_price'):
                strat.set_cex_price(pred.cex_price)
            if hasattr(strat, 'set_volatility'):
                strat.set_volatility(pred.volatility)
            if hasattr(strat, 'set_strike_price') and market.strike_price > 0:
                strat.set_strike_price(market.strike_price)

            # Track predicted direction for round recording
            if pred.cex_price > 0 and market.strike_price > 0:
                slot.predicted_direction = "Up" if pred.cex_price > market.strike_price else "Down"

            # M6 FIX: Snapshot CEX price near round end for settlement accuracy
            if market.seconds_remaining < 3 and pred.cex_price > 0:
                slot._settlement_cex_snapshot = pred.cex_price

        strat = slot.strategy

        # C3 FIX: Update market prices from CLOB book feed in real-time
        book_up = self._book_feed.get_book(market.token_id_up)
        if book_up:
            market.price_up = book_up.midpoint
        book_down = self._book_feed.get_book(market.token_id_down)
        if book_down:
            market.price_down = book_down.midpoint

        # CryptoMM-specific: CLOB book data
        if isinstance(strat, CryptoMarketMaker):
            if book_up and book_down:
                strat.set_book_data(
                    best_bid_up=book_up.best_bid, best_ask_up=book_up.best_ask,
                    best_bid_down=book_down.best_bid, best_ask_down=book_down.best_ask,
                    depth_bid_usd=book_up.total_bid_depth_usd + book_down.total_bid_depth_usd,
                    depth_ask_usd=book_up.total_ask_depth_usd + book_down.total_ask_depth_usd,
                )

        # D2: Pass CLOB Up token midpoint to sniper for Bayesian fusion
        if hasattr(strat, 'set_clob_midpoint') and book_up:
            strat.set_clob_midpoint(book_up.midpoint)

        # D3: Pass Binance OFI signal to sniper
        if hasattr(strat, 'set_ofi'):
            ofi = self._depth_feed.get_ofi(market.asset)
            strat.set_ofi(ofi)

    async def _vps_fallback_cex_price(self, asset: str) -> float:
        """v10 FIX #12: Lightweight VPS-side oracle fallback when Brain is offline.

        Uses direct HTTP call to Binance (available from VPS) to get current price.
        Less accurate than Brain's multi-source Kalman fusion but prevents total
        blackout when Alienware is down.
        """
        symbol = f"{asset.upper()}USDT"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(
                    f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                )
                if resp.status_code == 200:
                    return float(resp.json()["price"])
        except Exception as e:
            logger.debug(f"VPS fallback CEX failed for {asset}: {e}")
        return 0.0

    async def _calibrate_clock(self):
        """v10 FIX #14: Calibrate VPS clock against Polymarket API server time.

        Computes offset between local time and server time to correct
        seconds_remaining calculations. Runs every ~5 minutes.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                t_before = time.time()
                resp = await client.get(f"{self.config.api.gamma_host}/events?limit=1")
                t_after = time.time()
                rtt = t_after - t_before

                if resp.status_code == 200:
                    # Use HTTP Date header as server time reference
                    date_header = resp.headers.get("date", "")
                    if date_header:
                        from email.utils import parsedate_to_datetime
                        server_time = parsedate_to_datetime(date_header).timestamp()
                        local_time = (t_before + t_after) / 2  # midpoint estimate
                        self._clock_offset = local_time - server_time
                        self._last_clock_sync = time.time()
                        if abs(self._clock_offset) > 1.0:
                            logger.warning(
                                f"⏰ Clock offset: {self._clock_offset:+.2f}s "
                                f"(VPS {'ahead' if self._clock_offset > 0 else 'behind'}, RTT={rtt*1000:.0f}ms)"
                            )
                        else:
                            logger.debug(f"Clock sync OK: offset={self._clock_offset:+.3f}s RTT={rtt*1000:.0f}ms")
                    else:
                        self._last_clock_sync = time.time()
        except Exception as e:
            logger.debug(f"Clock calibration failed: {e}")
            self._last_clock_sync = time.time()  # don't retry immediately

    def _publish_telemetry(self):
        """Publish current state to Mac dashboard."""
        now = time.time()
        if now - self._last_telemetry < self._telemetry_interval:
            return
        self._last_telemetry = now

        # Aggregate across strategies
        total_pnl = sum(s.strategy.state.total_pnl_usd for s in self._slots)
        daily_pnl = sum(s.strategy.state.daily_pnl_usd for s in self._slots)
        daily_volume = sum(s.strategy.state.daily_volume_usd for s in self._slots)  # M7 FIX
        total_rounds = sum(s.strategy.state.rounds_traded for s in self._slots)
        total_wins = sum(s.strategy.state.rounds_won for s in self._slots)

        strat_info = {s.name: {
            "pnl": round(s.strategy.state.total_pnl_usd, 2),
            "win_rate": round(s.strategy.state.win_rate, 3),
            "rounds": s.strategy.state.rounds_traded,
        } for s in self._slots}

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
                "daily_volume": round(daily_volume, 2),  # M7 FIX: actual volume not abs(pnl)
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

        # FIX: Pick the NEAREST expiring market per asset (not farthest).
        # Old logic picked market with most time → when current market reached
        # T-10s, scanner already returned next market (300s) → slug changed →
        # round transition → Sniper's commitment window (T-10 to T-5) was NEVER hit.
        # Now: pick market with LEAST remaining time, but still > 0 (not expired).
        best_markets: dict[str, MarketInfo] = {}
        for m in markets:
            if m.is_expired:
                continue
            if m.asset not in best_markets or m.seconds_remaining < best_markets[m.asset].seconds_remaining:
                best_markets[m.asset] = m

        if not best_markets:
            return

        # v10 FIX #14: Periodic clock offset calibration (~every 5 min)
        now = time.time()
        if now - self._last_clock_sync > 300:
            await self._calibrate_clock()

        # v11 #15: Reset cross-asset correlation tracking each scan cycle
        self._slot_signals_this_cycle = {}

        # M6 FIX: Dispatch each market to asset-specific slots only.
        # Each slot is bound to one asset, preventing round-thrashing.
        for slot in self._slots:
            market = best_markets.get(slot.asset)
            if market:
                # v10 FIX #14: Adjust market timing with clock offset
                if abs(self._clock_offset) > 0.5:
                    market.end_time = int(market.end_time + self._clock_offset)
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

            # DYNAMIC STRIKE: Polymarket 5-min markets resolve
            # "Up if price at END ≥ price at START". There is NO fixed strike
            # in the question text. We use CEX price at round start as strike.
            if market.strike_price <= 0:
                pred = self._redis.get_prediction(market.asset)
                if pred and pred.cex_price > 0:
                    market.strike_price = pred.cex_price
                    slot._strike_set_at = time.time()  # v10 FIX #2
                    logger.info(
                        f"Dynamic strike set: {market.asset.upper()} "
                        f"${market.strike_price:,.2f} (CEX at round start)"
                    )
                else:
                    # Fallback: use strategy's cached CEX price
                    strat = slot.strategy
                    if hasattr(strat, '_cex_price') and strat._cex_price and strat._cex_price > 0:
                        market.strike_price = strat._cex_price
                        slot._strike_set_at = time.time()  # v10 FIX #2
                        logger.info(
                            f"Dynamic strike set (fallback): {market.asset.upper()} "
                            f"${market.strike_price:,.2f}"
                        )

            await slot.strategy.on_round_start(market)

        # FIX: Propagate dynamic strike to fresh market objects.
        # discover_current_markets() returns new MarketInfo each cycle with strike=0
        # because the "Up or Down" title has no dollar amount. The dynamic strike
        # was set on slot.current_market during round transition — pass it through.
        if market.strike_price <= 0 and slot.current_market and slot.current_market.strike_price > 0:
            market.strike_price = slot.current_market.strike_price

        # v10 FIX #2: Strike freshness TTL — expire strikes older than 120s
        # Prevents stale-strike trades after Brain reconnection
        if slot._strike_set_at > 0 and (time.time() - slot._strike_set_at) > 120:
            # Try to refresh strike from current prediction
            pred = self._redis.get_prediction(market.asset)
            if pred and pred.cex_price > 0:
                old_strike = market.strike_price
                market.strike_price = pred.cex_price
                slot._strike_set_at = time.time()
                if slot.current_market:
                    slot.current_market.strike_price = market.strike_price
                logger.info(
                    f"Strike refreshed (TTL): {market.asset.upper()} "
                    f"${old_strike:,.2f} → ${market.strike_price:,.2f}"
                )

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
        await self._feed_predictions_to_strategy(slot, market)

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
            # v11 #2: Handle CANCEL signals from direction re-check
            if any(sig.order_type == "CANCEL" for sig in signals):
                canceled = await self._executor.cancel_all()
                logger.info(f"v11 Direction-reversal cancel: {canceled} orders canceled for {slot.name}")
                return  # don't process further signals this cycle

            # CRIT3 FIX: Don't cancel ALL orders — let GTC orders rest.
            pass

        # v11 #15: Cross-asset correlation penalty
        # If both BTC and ETH have same-direction signals, penalize the second one
        for sig in signals:
            if sig.confidence > 0 and sig.size_usd > 0:
                direction = "Up" if sig.outcome.value == "Up" else "Down"
                other_dirs = [d for a, d in self._slot_signals_this_cycle.items() if a != slot.asset]
                if other_dirs and direction in other_dirs:
                    # Same direction as another asset → apply 50% correlation penalty
                    strat = slot.strategy
                    if hasattr(strat, 'set_correlation_penalty'):
                        strat.set_correlation_penalty(0.5)
                        logger.info(f"v11 Correlation penalty: {slot.asset.upper()} {direction} same as other asset")
                self._slot_signals_this_cycle[slot.asset] = direction

        # v11 #17: Pass CLOB best bid to strategy for maker price guard
        strat = slot.strategy
        if hasattr(strat, 'set_clob_best_bid') and hasattr(strat, '_trade_direction'):
            trade_dir = strat._trade_direction or ("Down" if strat._cex_price < strat._strike_price else "Up")
            token_id = market.token_id_down if trade_dir == "Down" else market.token_id_up
            book = self._book_feed.get_book(token_id)
            if book and book.best_bid > 0:
                strat.set_clob_best_bid(book.best_bid)

        # Validate and execute (P0 FIX: cap fills per round for realism)
        MAX_FILLS_PER_ROUND = 3  # real 5-min markets: 1-3 maker fills max
        for sig in signals:
            if sig.order_type == "CANCEL":  # v11: skip CANCEL pseudo-signals
                continue
            if slot.round_fills >= MAX_FILLS_PER_ROUND:
                break  # already filled enough this round
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

    async def _fetch_settlement_price_http(self, asset: str) -> float:
        """C6 FIX: Last-resort HTTP REST fallback for settlement price.
        Used when Brain is offline and strategy cache is empty.
        """
        symbol = f"{asset.upper()}USDT"
        urls = [
            f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
            f"https://api.coingecko.com/api/v3/simple/price?ids={'bitcoin' if asset == 'btc' else 'ethereum'}&vs_currencies=usd",
        ]
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try Binance first
                try:
                    resp = await client.get(urls[0])
                    if resp.status_code == 200:
                        price = float(resp.json()["price"])
                        logger.info(f"Settlement HTTP fallback (Binance): {asset.upper()} ${price:,.2f}")
                        return price
                except Exception as e:
                    logger.debug(f"Binance HTTP fallback failed: {e}")

                # Try CoinGecko
                try:
                    resp = await client.get(urls[1])
                    if resp.status_code == 200:
                        data = resp.json()
                        key = "bitcoin" if asset == "btc" else "ethereum"
                        price = float(data[key]["usd"])
                        logger.info(f"Settlement HTTP fallback (CoinGecko): {asset.upper()} ${price:,.2f}")
                        return price
                except Exception as e:
                    logger.debug(f"CoinGecko HTTP fallback failed: {e}")
        except Exception as e:
            logger.error(f"HTTP settlement fallback failed entirely: {e}")
        return 0.0

    async def _handle_round_end(self, slot: StrategySlot, market: MarketInfo):
        """Handle end of round for a strategy."""
        await self._executor.cancel_all()

        # M6 FIX: Sync risk state BEFORE round-end processing
        total_daily = sum(s.strategy.state.daily_pnl_usd for s in self._slots)
        self._risk.state.daily_pnl = total_daily

        # Settlement: determine if Up or Down won
        # Priority: 1) M6 CEX snapshot near T-0, 2) Brain prediction,
        #           3) Strategy cache, 4) HTTP REST fallback, 5) coin flip
        cex_price = 0.0

        # M6 FIX: prefer snapshot taken near T-0s
        if hasattr(slot, '_settlement_cex_snapshot') and slot._settlement_cex_snapshot > 0:
            cex_price = slot._settlement_cex_snapshot
            logger.debug(f"Settlement: using T-0 snapshot: ${cex_price:.2f}")

        # Fallback 1: current Brain prediction
        if cex_price <= 0:
            pred = self._redis.get_prediction(market.asset)
            if pred and pred.cex_price > 0:
                cex_price = pred.cex_price

        # Fallback 2: strategy's cached CEX price
        if cex_price <= 0:
            strat = slot.strategy
            if hasattr(strat, '_cex_price') and strat._cex_price and strat._cex_price > 0:
                cex_price = strat._cex_price
                logger.debug(f"Settlement: using strategy cached CEX price: ${cex_price:.2f}")

        # C6 FIX: Fallback 3: HTTP REST call before coin flip
        if cex_price <= 0:
            cex_price = await self._fetch_settlement_price_http(market.asset)

        strike = market.strike_price

        if cex_price > 0 and strike > 0:
            # C1 FIX: Polymarket rule is "Up if END >= START", use >= not >
            settled = "Up" if cex_price >= strike else "Down"
            logger.info(f"Settlement: CEX=${cex_price:.2f} vs strike=${strike:.2f} → {settled}")
        else:
            # Absolute last resort: 50/50 coin flip
            settled = random.choice(["Up", "Down"])
            logger.warning(
                f"Settlement fallback 50/50 (all sources failed): "
                f"cex=${cex_price:.2f}, strike=${strike:.2f}"
            )
            # E8: Alert on coin-flip settlement
            await self._alerter.check_coinflip_settlement(was_coinflip=True)

        await slot.strategy.on_round_end(market, settled)

        # Record per-round PnL delta, not cumulative
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

        # Sync risk state after settlement
        total_daily = sum(s.strategy.state.daily_pnl_usd for s in self._slots)
        self._risk.state.daily_pnl = total_daily
        # Release per-asset exposure since positions are settled
        self._risk.release_exposure(market.asset, slot.round_cost)

        # Clean up settlement snapshot
        if hasattr(slot, '_settlement_cex_snapshot'):
            slot._settlement_cex_snapshot = 0.0

        # E4: Record round in analytics engine
        was_fill = slot.round_fills > 0
        direction = getattr(slot, '_trade_direction', slot.predicted_direction) or ""
        # Try to get sniper's trade_direction
        if hasattr(slot.strategy, '_trade_direction'):
            direction = slot.strategy._trade_direction
        self._analytics.record_round(
            pnl=round_pnl,
            was_fill=was_fill,
            direction=direction,
            confidence=getattr(slot.strategy, '_last_confidence', 0.0),
            maker_price=getattr(slot.strategy, '_last_maker_price', 0.0),
            order_size=slot.round_cost,
        )

        # Schedule settlement verification against Polymarket actual outcome
        if market.slug and cex_price > 0 and strike > 0:
            self._verifier.schedule_verification(
                slug=market.slug,
                asset=market.asset,
                paper_settlement=settled,
                paper_cex_price=cex_price,
                paper_strike=strike,
                strategy=slot.name,
                confidence=getattr(slot.strategy, '_last_confidence', 0.0),  # v10 FIX #10
            )

    async def _shutdown(self):
        await self._executor.cancel_all()
        await self._scanner.close()
        self._redis.stop()
        self._book_feed.stop()
        self._depth_feed.stop()  # D3: Binance OFI feed
        await self._alerter.close()  # E8
        await self._verifier.close()

        # Settlement verification summary
        logger.info(f"\n{self._verifier.get_summary()}")

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
    parser.add_argument("-s", "--strategy", default="sniper",
                        help="Strategy name(s), comma-separated: sniper,crypto_mm")
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

    # v11 #18: Graceful shutdown handler
    def _graceful_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning(f"🛑 Received {sig_name} — initiating graceful shutdown...")
        runner.stop()

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    # M7 FIX: Use asyncio.run() directly
    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        runner.stop()


if __name__ == "__main__":
    main()
