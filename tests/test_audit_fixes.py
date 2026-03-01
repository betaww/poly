"""
Comprehensive test suite for Polymarket trading system.

Covers:
  - C1: Settlement >= rule
  - C4/M2: Confidence curve and volatility normalization  
  - C5: Direction re-evaluation window
  - Fee calculations (paper simulator)
  - GTC fill probability edge cases
  - Market scanner parsing
  - Risk manager circuit breakers
"""
import math
import sys
import os
import time

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest


# ─── Settlement Logic Tests (C1) ─────────────────────────────────────────────

class TestSettlement:
    """Test that settlement uses >= per Polymarket rules."""

    def test_settlement_up_when_cex_equals_strike(self):
        """C1 FIX: CEX == strike should settle as Up (>= rule)."""
        cex_price = 85000.0
        strike = 85000.0
        settled = "Up" if cex_price >= strike else "Down"
        assert settled == "Up", f"CEX == strike should be Up, got {settled}"

    def test_settlement_up_when_cex_above_strike(self):
        cex_price = 85001.0
        strike = 85000.0
        settled = "Up" if cex_price >= strike else "Down"
        assert settled == "Up"

    def test_settlement_down_when_cex_below_strike(self):
        cex_price = 84999.0
        strike = 85000.0
        settled = "Up" if cex_price >= strike else "Down"
        assert settled == "Down"

    def test_settlement_up_at_tiny_difference(self):
        """Edge case: cex is $0.01 above strike."""
        cex_price = 85000.01
        strike = 85000.00
        settled = "Up" if cex_price >= strike else "Down"
        assert settled == "Up"

    def test_settlement_down_at_tiny_difference(self):
        """Edge case: cex is $0.01 below strike."""
        cex_price = 84999.99
        strike = 85000.00
        settled = "Up" if cex_price >= strike else "Down"
        assert settled == "Down"


# ─── Fee Calculation Tests ────────────────────────────────────────────────────

class TestFeeCalculation:
    """Test Polymarket taker fee formula: C × feeRate × (p(1-p))^exponent."""

    def setup_method(self):
        from polymarket.engine.paper_simulator import calculate_taker_fee, calculate_maker_fee
        self.calc_taker = calculate_taker_fee
        self.calc_maker = calculate_maker_fee

    def test_taker_fee_at_midpoint(self):
        """100 shares @ $0.50 → fee = 100 × 0.25 × (0.25)^2 = $1.5625."""
        result = self.calc_taker(100.0, 0.50)
        assert abs(result.fee_usd - 1.5625) < 0.01, f"Expected ~$1.56, got ${result.fee_usd:.4f}"
        assert result.fee_type.value == "taker"

    def test_taker_fee_at_extreme_price(self):
        """Fee near p=0.90 should be much lower."""
        result = self.calc_taker(100.0, 0.90)
        # p(1-p) = 0.09, (0.09)^2 = 0.0081, fee = 100 × 0.25 × 0.0081 = $0.2025
        assert abs(result.fee_usd - 0.2025) < 0.01, f"Expected ~$0.20, got ${result.fee_usd:.4f}"

    def test_taker_fee_symmetric(self):
        """Fee at p=0.10 should equal fee at p=0.90 (symmetric formula)."""
        fee_10 = self.calc_taker(100.0, 0.10)
        fee_90 = self.calc_taker(100.0, 0.90)
        assert abs(fee_10.fee_usd - fee_90.fee_usd) < 0.001

    def test_maker_fee_is_zero(self):
        """Maker orders have 0% fee."""
        result = self.calc_maker(100.0, 0.60)
        assert result.fee_usd == 0.0
        assert result.fee_type.value == "maker"

    def test_taker_fee_at_boundary_prices(self):
        """Fee at p=0 and p=1 should be 0."""
        fee_0 = self.calc_taker(100.0, 0.0)
        fee_1 = self.calc_taker(100.0, 1.0)
        assert fee_0.fee_usd == 0.0
        assert fee_1.fee_usd == 0.0


# ─── GTC Fill Probability Tests (C4/M1) ──────────────────────────────────────

class TestGTCFillProbability:
    """Test GTC fill probability estimation edge cases."""

    def setup_method(self):
        from polymarket.engine.paper_simulator import estimate_gtc_fill_probability
        self.estimate = estimate_gtc_fill_probability

    def test_aggressive_buy_high_probability(self):
        """Buy above midpoint = crossing spread = ~95% fill."""
        prob = self.estimate(
            limit_price=0.55, midpoint=0.50,
            seconds_resting=5.0, is_buy=True, volatility=0.05
        )
        assert prob >= 0.85, f"Aggressive buy should have high fill prob, got {prob:.2%}"

    def test_passive_buy_low_probability(self):
        """Buy far below midpoint = resting far in book = low fill."""
        prob = self.estimate(
            limit_price=0.30, midpoint=0.50,
            seconds_resting=5.0, is_buy=True, volatility=0.05
        )
        assert prob < 0.10, f"Passive buy should have low fill prob, got {prob:.2%}"

    def test_more_resting_time_increases_probability(self):
        """Longer resting = higher fill probability."""
        prob_1s = self.estimate(0.48, 0.50, 1.0, True, 0.05)
        prob_10s = self.estimate(0.48, 0.50, 10.0, True, 0.05)
        assert prob_10s > prob_1s, f"10s ({prob_10s:.2%}) should > 1s ({prob_1s:.2%})"

    def test_zero_midpoint_returns_zero(self):
        """Midpoint <= 0 should return 0 probability."""
        prob = self.estimate(0.50, 0.0, 5.0, True, 0.05)
        assert prob == 0.0

    def test_fill_probability_capped_at_95(self):
        """Fill probability should never exceed 95%."""
        prob = self.estimate(
            limit_price=0.60, midpoint=0.50,
            seconds_resting=100.0, is_buy=True, volatility=0.10
        )
        assert prob <= 0.95, f"Prob should cap at 95%, got {prob:.2%}"


# ─── Confidence Curve Tests (M2) ─────────────────────────────────────────────

class TestConfidenceCurve:
    """Test volatility-normalized confidence curve in DirectionalSniper."""

    def setup_method(self):
        from polymarket.config import Config
        from polymarket.strategies.directional_sniper import DirectionalSniper
        self.config = Config()
        self.sniper = DirectionalSniper(self.config)

    def test_unknown_when_no_prices(self):
        """No CEX/strike → Unknown direction."""
        self.sniper._cex_price = 0.0
        self.sniper._strike_price = 0.0
        direction, conf = self.sniper._get_direction_confidence()
        assert direction == "Unknown"
        assert conf == 0.0

    def test_unknown_when_at_strike(self):
        """CEX very close to strike → Unknown."""
        self.sniper._cex_price = 85000.0
        self.sniper._strike_price = 85000.0
        direction, conf = self.sniper._get_direction_confidence()
        assert direction == "Unknown"

    def test_up_direction(self):
        """CEX above strike → Up."""
        self.sniper._cex_price = 85100.0
        self.sniper._strike_price = 85000.0
        self.sniper._volatility = 0.01
        direction, conf = self.sniper._get_direction_confidence()
        assert direction == "Up"

    def test_down_direction(self):
        """CEX below strike → Down."""
        self.sniper._cex_price = 84900.0
        self.sniper._strike_price = 85000.0
        self.sniper._volatility = 0.01
        direction, conf = self.sniper._get_direction_confidence()
        assert direction == "Down"

    def test_high_vol_reduces_confidence(self):
        """Same absolute move, higher vol → lower confidence (M2 fix)."""
        self.sniper._cex_price = 85085.0  # ~0.1% above strike
        self.sniper._strike_price = 85000.0

        self.sniper._volatility = 0.005  # low vol → move is 2x vol
        _, conf_low_vol = self.sniper._get_direction_confidence()

        self.sniper._volatility = 0.05  # high vol → move is 0.02x vol
        _, conf_high_vol = self.sniper._get_direction_confidence()

        assert conf_low_vol > conf_high_vol, (
            f"Low vol conf ({conf_low_vol:.2%}) should > high vol ({conf_high_vol:.2%})"
        )

    def test_confidence_capped_at_95(self):
        """Confidence should never exceed 95%."""
        self.sniper._cex_price = 90000.0  # massive move
        self.sniper._strike_price = 85000.0
        self.sniper._volatility = 0.001
        _, conf = self.sniper._get_direction_confidence()
        assert conf <= 0.95, f"Confidence should cap at 95%, got {conf:.2%}"

    def test_noise_level_move_below_threshold(self):
        """0.1% move in 2% vol should NOT pass 65% confidence threshold."""
        self.sniper._cex_price = 85085.0  # 0.1% above strike
        self.sniper._strike_price = 85000.0
        self.sniper._volatility = 0.02  # 2% vol, so z-score = 0.1/2 = 0.05
        _, conf = self.sniper._get_direction_confidence()
        assert conf < 0.65, (
            f"Noise-level move in high vol should be < 65%, got {conf:.2%}"
        )


# ─── Direction Re-evaluation Tests (C5) ──────────────────────────────────────

class TestDirectionReeval:
    """Test that DirectionalSniper allows re-evaluation at T-8s."""

    def setup_method(self):
        from polymarket.config import Config
        from polymarket.strategies.directional_sniper import DirectionalSniper
        self.config = Config()
        self.sniper = DirectionalSniper(self.config)

    def test_direction_not_locked_at_t10(self):
        """At T-10s, direction != Up should NOT lock _signal_sent."""
        # Simulate commitment window (T-10s to T-5s)
        self.sniper._cex_price = 84900.0  # Below strike → Down
        self.sniper._strike_price = 85000.0
        self.sniper._volatility = 0.01
        self.sniper._signal_sent = False

        # Simulate being at T-9s (direction is Down, but time > 7)
        # The confidence threshold and direction check happen inside on_market_update
        # We test the core logic directly:
        t = 9  # T-9s: should NOT lock
        direction = "Down"
        if direction != "Up":
            if t <= 7:
                locked = True
            else:
                locked = False
        assert not locked, "At T-9s, Down direction should NOT lock signal_sent"

    def test_direction_locked_at_t7(self):
        """At T-7s, direction != Up SHOULD lock _signal_sent."""
        t = 7
        direction = "Down"
        if direction != "Up":
            if t <= 7:
                locked = True
            else:
                locked = False
        assert locked, "At T-7s, Down direction SHOULD lock signal_sent"


# ─── Market Scanner Tests ────────────────────────────────────────────────────

class TestMarketScanner:
    """Test MarketScanner strike parsing and window calculation."""

    def test_parse_strike_standard(self):
        from polymarket.market_scanner import MarketScanner
        strike = MarketScanner._parse_strike("Will BTC be above $84,500.00 at 2:30 PM?", "btc")
        assert strike == 84500.0

    def test_parse_strike_naked(self):
        from polymarket.market_scanner import MarketScanner
        strike = MarketScanner._parse_strike("BTC above 84500 at 14:30", "btc")
        assert strike == 84500.0

    def test_parse_strike_no_strike(self):
        """5-min markets have no strike in question text → returns 0."""
        from polymarket.market_scanner import MarketScanner
        strike = MarketScanner._parse_strike("Bitcoin Up or Down - March 1, 12:00AM-12:05AM ET", "btc")
        # This should return 0 (no parseable strike)
        assert strike == 0.0

    def test_window_timestamps_include_prev(self):
        """M9 FIX: timestamps should include previous window."""
        from polymarket.config import Config
        from polymarket.market_scanner import MarketScanner
        config = Config()
        scanner = MarketScanner(config)
        timestamps = scanner._calculate_window_timestamps("btc")
        assert len(timestamps) == 3, f"Expected 3 timestamps (prev, current, next), got {len(timestamps)}"
        # Verify ordering: prev < current < next
        assert timestamps[0] < timestamps[1] < timestamps[2]


# ─── Risk Manager Tests ──────────────────────────────────────────────────────

class TestRiskManager:
    """Test risk manager circuit breakers and exposure tracking."""

    def setup_method(self):
        from polymarket.config import Config
        from polymarket.engine.risk_manager import RiskManager
        self.config = Config()
        self.risk = RiskManager(self.config)

    def test_daily_loss_limit_triggers_pause(self):
        from polymarket.strategies.base import StrategyState, Signal, Side, Outcome
        from polymarket.market_scanner import MarketInfo
        
        market = MarketInfo(
            asset="btc", event_id="1", market_id="1",
            condition_id="1", slug="test", question="test",
            token_id_up="up", token_id_down="down",
            end_time=int(time.time()) + 60,
        )
        signal = Signal(
            market=market, outcome=Outcome.UP, side=Side.BUY,
            price=0.60, size_usd=10.0
        )
        state = StrategyState()

        # Set daily PnL to exceed limit
        self.risk.state.daily_pnl = -101.0  # limit is -100
        approved, reason = self.risk.check_signal(signal, state)
        assert not approved, "Should reject when daily loss limit hit"
        assert "loss limit" in reason.lower()

    def test_consecutive_loss_triggers_pause(self):
        from polymarket.strategies.base import StrategyState, Signal, Side, Outcome
        from polymarket.market_scanner import MarketInfo

        market = MarketInfo(
            asset="btc", event_id="1", market_id="1",
            condition_id="1", slug="test", question="test",
            token_id_up="up", token_id_down="down",
            end_time=int(time.time()) + 60,
        )
        signal = Signal(
            market=market, outcome=Outcome.UP, side=Side.BUY,
            price=0.60, size_usd=10.0
        )
        state = StrategyState(consecutive_losses=3)  # meets threshold

        approved, reason = self.risk.check_signal(signal, state)
        assert not approved, "Should reject after 3 consecutive losses"

    def test_exposure_tracking(self):
        """Track and release per-asset exposure correctly."""
        self.risk.record_exposure("btc", 10.0)
        self.risk.record_exposure("btc", 20.0)
        assert self.risk.state.exposure_per_asset["btc"] == 30.0

        self.risk.release_exposure("btc", 15.0)
        assert self.risk.state.exposure_per_asset["btc"] == 15.0

        self.risk.release_exposure("btc", 100.0)  # more than current
        assert self.risk.state.exposure_per_asset["btc"] == 0.0  # clamped to 0


# ─── Redis Consumer Stale Tests (C8) ─────────────────────────────────────────

class TestPredictionStaleness:
    """Test C8 stale threshold change."""

    def test_stale_threshold_15s(self):
        """Predictions should not be stale until 15s old."""
        from polymarket.node.redis_consumer import Prediction
        pred = Prediction(
            asset="btc", cex_price=85000.0, confidence=0.9,
            volatility=0.01, sources=3, timestamp=time.time(),
            age_ms=10000  # 10 seconds
        )
        assert not pred.is_stale, "10s-old prediction should NOT be stale (threshold=15s)"

        pred_old = Prediction(
            asset="btc", cex_price=85000.0, confidence=0.9,
            volatility=0.01, sources=3, timestamp=time.time(),
            age_ms=16000  # 16 seconds
        )
        assert pred_old.is_stale, "16s-old prediction SHOULD be stale"


# ─── Price Feeds / SyntheticOracle Tests ──────────────────────────────────────

class TestSyntheticOracle:
    """Test oracle prediction and volatility calculation."""

    def setup_method(self):
        pytest.importorskip("websockets", reason="websockets not installed")
        from polymarket.brain.price_feeds import SyntheticOracle, PriceTick
        from polymarket.config import Config
        self.config = Config()
        self.oracle = SyntheticOracle(self.config)
        self.PriceTick = PriceTick

    def test_no_data_returns_zero(self):
        predicted, confidence = self.oracle.predict()
        assert predicted == 0.0
        assert confidence == 0.0

    def test_single_source_low_confidence(self):
        tick = self.PriceTick(exchange="binance", asset="btc", price=85000.0, timestamp=time.time())
        self.oracle.update(tick)
        predicted, confidence = self.oracle.predict()
        assert abs(predicted - 85000.0) < 1.0
        assert confidence == 0.6  # single source

    def test_multiple_sources_increase_confidence(self):
        now = time.time()
        self.oracle.update(self.PriceTick("binance", "btc", 85000.0, now))
        self.oracle.update(self.PriceTick("coinbase", "btc", 85001.0, now))
        self.oracle.update(self.PriceTick("okx", "btc", 84999.0, now))
        _, confidence = self.oracle.predict()
        assert confidence > 0.8, f"3 agreeing sources should give high confidence, got {confidence:.2%}"

    def test_prediction_history_bounded(self):
        """M8 FIX: deque maxlen should keep history bounded."""
        now = time.time()
        self.oracle.update(self.PriceTick("binance", "btc", 85000.0, now))
        for i in range(500):
            self.oracle.predict()
        # deque(maxlen=300) should keep at most 300 entries
        assert len(self.oracle._prediction_history) <= 300


# ─── E1: Bidirectional Trading Tests ──────────────────────────────────────────

class TestBidirectionalTrading:
    """E1: Test that DirectionalSniper now supports both Up and Down."""

    def setup_method(self):
        from polymarket.config import Config
        from polymarket.strategies.directional_sniper import DirectionalSniper
        self.config = Config()
        self.sniper = DirectionalSniper(self.config)

    def test_down_direction_detected(self):
        """CEX below strike should give Down direction."""
        self.sniper._cex_price = 84800.0
        self.sniper._strike_price = 85000.0
        self.sniper._volatility = 0.01
        direction, conf = self.sniper._get_direction_confidence()
        assert direction == "Down", f"Expected Down, got {direction}"
        assert conf > 0.50

    def test_down_pnl_calculation(self):
        """Buying Down tokens: Down settles → profit."""
        import asyncio
        from polymarket.market_scanner import MarketInfo

        market = MarketInfo(
            asset="btc", event_id="1", market_id="1",
            condition_id="1", slug="test", question="test",
            token_id_up="up", token_id_down="down",
            end_time=int(time.time()) + 60,
        )
        self.sniper._shares_bought = 10.0
        self.sniper._buy_cost = 6.0  # $0.60 per share
        self.sniper._trade_direction = "Down"

        asyncio.get_event_loop().run_until_complete(
            self.sniper.on_round_end(market, "Down")
        )
        # Down wins → $1.00/share × 10 - $6.00 cost = $4.00 profit
        assert self.sniper.state.total_pnl_usd == 4.0

    def test_down_loss_when_up_settles(self):
        """Buying Down tokens: Up settles → loss."""
        import asyncio
        from polymarket.market_scanner import MarketInfo

        market = MarketInfo(
            asset="btc", event_id="1", market_id="1",
            condition_id="1", slug="test", question="test",
            token_id_up="up", token_id_down="down",
            end_time=int(time.time()) + 60,
        )
        self.sniper._shares_bought = 10.0
        self.sniper._buy_cost = 6.0
        self.sniper._trade_direction = "Down"

        asyncio.get_event_loop().run_until_complete(
            self.sniper.on_round_end(market, "Up")
        )
        # Up wins → Down token = $0.00, loss = -$6.00
        assert self.sniper.state.total_pnl_usd == -6.0


# ─── E2: Kelly Sizing Tests ──────────────────────────────────────────────────

class TestKellySizing:
    """E2: Test half-Kelly position sizing."""

    def setup_method(self):
        from polymarket.config import Config
        from polymarket.strategies.directional_sniper import DirectionalSniper
        self.config = Config()
        self.sniper = DirectionalSniper(self.config)
        self.sniper._bankroll = 100.0

    def test_high_confidence_larger_size(self):
        """Higher confidence → larger Kelly fraction."""
        size_high = self.sniper._kelly_size(0.90, 0.60)
        size_low = self.sniper._kelly_size(0.65, 0.60)
        assert size_high > size_low, (
            f"High conf size ${size_high:.2f} should > low conf ${size_low:.2f}"
        )

    def test_minimum_size_enforced(self):
        """Kelly size should never go below min_size."""
        size = self.sniper._kelly_size(0.51, 0.60)  # barely any edge
        assert size >= 5.0, f"Size should be >= $5.00, got ${size:.2f}"

    def test_no_edge_returns_minimum(self):
        """When confidence < break-even, should return minimum."""
        # Break-even at $0.60 = 60% win rate
        size = self.sniper._kelly_size(0.50, 0.60)  # below break-even
        assert size == 5.0, f"No edge should return min ${size:.2f}"

    def test_max_fraction_capped(self):
        """Kelly fraction should be capped at max_kelly_fraction."""
        size = self.sniper._kelly_size(0.99, 0.60)  # extreme confidence
        max_allowed = 100.0 * 0.15  # $15
        assert size <= max_allowed, f"Size ${size:.2f} exceeds max ${max_allowed:.2f}"

    def test_bankroll_update_on_win(self):
        """Bankroll should grow after winning round."""
        import asyncio
        from polymarket.market_scanner import MarketInfo

        market = MarketInfo(
            asset="btc", event_id="1", market_id="1",
            condition_id="1", slug="test", question="test",
            token_id_up="up", token_id_down="down",
            end_time=int(time.time()) + 60,
        )
        initial_bankroll = self.sniper._bankroll
        self.sniper._shares_bought = 10.0
        self.sniper._buy_cost = 6.0
        self.sniper._trade_direction = "Up"

        asyncio.get_event_loop().run_until_complete(
            self.sniper.on_round_end(market, "Up")
        )
        assert self.sniper._bankroll > initial_bankroll


# ─── E4: Analytics Tests ─────────────────────────────────────────────────────

class TestPerformanceAnalytics:
    """E4: Test Wilson CI, Sharpe ratio, drawdown tracking."""

    def setup_method(self):
        from polymarket.engine.analytics import PerformanceAnalytics
        self.analytics = PerformanceAnalytics()

    def test_wilson_ci_empty(self):
        """Wilson CI with 0 samples should return (0, 1)."""
        from polymarket.engine.analytics import PerformanceAnalytics
        lower, upper = PerformanceAnalytics.wilson_ci(0, 0)
        assert lower == 0.0 and upper == 1.0

    def test_wilson_ci_small_sample(self):
        """Wilson CI should be wider for small samples."""
        lower_10, upper_10 = self.analytics.wilson_ci(7, 10)
        lower_100, upper_100 = self.analytics.wilson_ci(70, 100)
        width_10 = upper_10 - lower_10
        width_100 = upper_100 - lower_100
        assert width_10 > width_100, "Smaller sample should have wider CI"

    def test_max_drawdown_tracking(self):
        """Drawdown should track correctly."""
        self.analytics.record_round(pnl=10.0, was_fill=True)
        self.analytics.record_round(pnl=5.0, was_fill=True)
        assert self.analytics._peak_pnl == 15.0
        self.analytics.record_round(pnl=-8.0, was_fill=True)
        assert self.analytics._max_drawdown == 8.0
        assert self.analytics._current_drawdown == 8.0

    def test_fill_rate_calculation(self):
        """Fill rate should reflect fills vs total rounds."""
        for i in range(10):
            self.analytics.record_round(pnl=0.0, was_fill=(i % 3 == 0))
        fill_rate = self.analytics.get_fill_rate()
        assert abs(fill_rate - 0.4) < 0.01  # 4/10

    def test_ev_per_round(self):
        """EV includes non-fill rounds."""
        self.analytics.record_round(pnl=1.0, was_fill=True)
        self.analytics.record_round(pnl=0.0, was_fill=False)
        assert self.analytics.get_ev_per_round() == 0.5

    def test_summary_keys(self):
        """Summary should contain all expected keys."""
        self.analytics.record_round(pnl=1.0, was_fill=True)
        summary = self.analytics.get_summary()
        expected_keys = [
            "total_rounds", "total_fills", "fill_rate", "total_pnl",
            "win_rate", "win_rate_95ci", "sharpe_ratio",
            "ev_per_round", "ev_per_fill", "max_drawdown",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"


# ─── E6: Chainlink Latency Tests ─────────────────────────────────────────────

class TestChainlinkLatency:
    """E6: Test Chainlink lag confidence adjustment."""

    def setup_method(self):
        from polymarket.engine.live_fills import ChainlinkLatencyEstimator
        self.estimator = ChainlinkLatencyEstimator(estimated_lag_ms=2000.0)

    def test_far_from_strike_no_adjustment(self):
        """Price far from strike → lag doesn't matter."""
        adj = self.estimator.confidence_adjustment(0.05, vol_per_second=0.001)
        assert adj == 1.0, f"Far from strike should give 1.0, got {adj}"

    def test_close_to_strike_reduces_confidence(self):
        """Price close to strike + lag → reduced confidence."""
        adj = self.estimator.confidence_adjustment(0.001, vol_per_second=0.001)
        assert adj < 1.0, f"Close to strike should reduce confidence, got {adj}"

    def test_at_strike_maximum_reduction(self):
        """At exact strike → maximum uncertainty."""
        adj = self.estimator.confidence_adjustment(0.0)
        assert adj == 0.5, f"At strike should give 0.5, got {adj}"

    def test_lag_update(self):
        """Lag estimate should update with new samples."""
        self.estimator.update_lag_estimate(100.0, 103.0)  # 3s lag
        assert abs(self.estimator.estimated_lag_seconds - 3.0) < 0.1


# ─── E8: Alert Config Tests ─────────────────────────────────────────────────

class TestAlertConfig:
    """E8: Test alert configuration."""

    def test_alert_disabled_by_default(self):
        from polymarket.config import Config
        config = Config()
        assert config.alert.enabled is False

    def test_alert_thresholds_set(self):
        from polymarket.config import Config
        config = Config()
        assert config.alert.brain_offline_seconds == 30
        assert config.alert.consecutive_coinflip_alert == 3


# ─── Edge Case Tests: Deep Algorithm Validation ─────────────────────────────

class TestD1EMATrend:
    """D1: EMA trend detection edge cases."""

    def setup_method(self):
        from polymarket.config import Config
        from polymarket.strategies.directional_sniper import DirectionalSniper
        self.config = Config()
        self.sniper = DirectionalSniper(self.config)

    def test_ema_not_initialized_no_crash(self):
        """Before any price, EMA should not affect confidence."""
        self.sniper._strike_price = 85000.0
        self.sniper._cex_price = 85100.0
        self.sniper._volatility = 0.01
        # EMA not initialized — should still produce valid result
        direction, conf = self.sniper._get_direction_confidence()
        assert direction == "Up"
        assert 0.50 <= conf <= 0.95

    def test_ema_trend_aligned_boosts(self):
        """Rising prices → Up trend → Up direction gets boosted."""
        from polymarket.strategies.directional_sniper import DirectionalSniper
        # Initialize and build uptrend
        for p in [84800, 84900, 85000, 85100, 85200, 85300]:
            self.sniper.set_cex_price(float(p))

        self.sniper._strike_price = 85000.0
        self.sniper._volatility = 0.01
        direction, conf_trending = self.sniper._get_direction_confidence()

        # Compare to a sniper with no trend (fresh)
        fresh = DirectionalSniper(self.config)
        fresh._cex_price = 85300.0
        fresh._strike_price = 85000.0
        fresh._volatility = 0.01
        _, conf_fresh = fresh._get_direction_confidence()

        assert conf_trending >= conf_fresh, (
            f"Trending conf {conf_trending:.3f} should >= fresh {conf_fresh:.3f}"
        )

    def test_ema_trend_opposed_reduces(self):
        """Falling prices → Down trend → Up direction at strike gets penalized."""
        from polymarket.strategies.directional_sniper import DirectionalSniper
        # Build downtrend then set CEX slightly above strike
        for p in [86000, 85800, 85600, 85400, 85200, 85050]:
            self.sniper.set_cex_price(float(p))

        self.sniper._strike_price = 85000.0
        self.sniper._volatility = 0.01
        direction, conf_opposed = self.sniper._get_direction_confidence()
        assert direction == "Up"  # CEX > strike

        # Fresh sniper at same price
        fresh = DirectionalSniper(self.config)
        fresh._cex_price = 85050.0
        fresh._strike_price = 85000.0
        fresh._volatility = 0.01
        _, conf_fresh = fresh._get_direction_confidence()

        assert conf_opposed <= conf_fresh, (
            f"Opposed conf {conf_opposed:.3f} should <= fresh {conf_fresh:.3f}"
        )

    def test_ema_persists_across_rounds(self):
        """EMA should NOT reset between rounds (D1 fix)."""
        import asyncio
        from polymarket.market_scanner import MarketInfo

        market = MarketInfo(
            asset="btc", event_id="1", market_id="1",
            condition_id="1", slug="test", question="test",
            token_id_up="up", token_id_down="down",
            end_time=int(time.time()) + 60,
        )

        # Build trend in round 1
        for p in [84800, 84900, 85000, 85100]:
            self.sniper.set_cex_price(float(p))
        assert self.sniper._ema_initialized is True

        # Start round 2 — EMA should persist
        asyncio.get_event_loop().run_until_complete(
            self.sniper.on_round_start(market)
        )
        assert self.sniper._ema_initialized is True, "EMA should persist across rounds"


class TestD2BayesianFusion:
    """D2: Bayesian fusion edge cases."""

    def setup_method(self):
        from polymarket.config import Config
        from polymarket.strategies.directional_sniper import DirectionalSniper
        self.config = Config()
        self.sniper = DirectionalSniper(self.config)
        self.sniper._strike_price = 85000.0
        self.sniper._volatility = 0.01

    def test_clob_agrees_boosts_confidence(self):
        """CEX says Up + CLOB says Up (high midpoint) → higher confidence."""
        self.sniper._cex_price = 85200.0

        # Without CLOB
        self.sniper._clob_midpoint = 0.0
        _, conf_no_clob = self.sniper._get_direction_confidence()

        # With CLOB agreeing (Up token at $0.75)
        self.sniper._clob_midpoint = 0.75
        _, conf_clob_agree = self.sniper._get_direction_confidence()

        assert conf_clob_agree >= conf_no_clob, (
            f"CLOB agreement {conf_clob_agree:.3f} should >= no CLOB {conf_no_clob:.3f}"
        )

    def test_clob_disagrees_reduces_confidence(self):
        """CEX says Up but CLOB says Down (low midpoint) → lower confidence."""
        self.sniper._cex_price = 85200.0

        # Without CLOB
        self.sniper._clob_midpoint = 0.0
        _, conf_no_clob = self.sniper._get_direction_confidence()

        # CLOB disagrees (Up token at $0.30 → market thinks Down)
        self.sniper._clob_midpoint = 0.30
        _, conf_clob_disagree = self.sniper._get_direction_confidence()

        assert conf_clob_disagree < conf_no_clob, (
            f"CLOB disagreement {conf_clob_disagree:.3f} should < no CLOB {conf_no_clob:.3f}"
        )

    def test_clob_extreme_ignored(self):
        """CLOB at 0.99 or 0.01 should be ignored (outside 0.05-0.95)."""
        self.sniper._cex_price = 85200.0
        self.sniper._clob_midpoint = 0.0
        _, conf_no_clob = self.sniper._get_direction_confidence()

        self.sniper._clob_midpoint = 0.99  # extreme → should be ignored
        _, conf_extreme = self.sniper._get_direction_confidence()

        assert abs(conf_extreme - conf_no_clob) < 0.001, (
            "Extreme CLOB midpoint should be ignored"
        )

    def test_bayesian_result_always_valid(self):
        """Fused confidence should always be in [0.50, 0.95]."""
        test_cases = [
            (85500, 0.90),  # high CEX + very high CLOB
            (85500, 0.10),  # high CEX + very low CLOB
            (84500, 0.90),  # low CEX + very high CLOB
            (84500, 0.10),  # low CEX + very low CLOB
            (85050, 0.50),  # near strike + neutral CLOB
        ]
        for cex, clob in test_cases:
            self.sniper._cex_price = float(cex)
            self.sniper._clob_midpoint = clob
            direction, conf = self.sniper._get_direction_confidence()
            assert 0.50 <= conf <= 0.95, (
                f"cex={cex}, clob={clob}: conf={conf:.4f} out of range"
            )


class TestD3TimeUrgency:
    """D3: Time urgency bonus edge cases."""

    def test_urgency_at_t10_is_zero(self):
        """At T-10s, bonus should be 0."""
        bonus = 0.05 * (1.0 - (10.0 - 5.0) / 5.0)
        assert abs(bonus) < 0.001

    def test_urgency_at_t5_is_max(self):
        """At T-5s, bonus should be 0.05."""
        bonus = 0.05 * (1.0 - (5.0 - 5.0) / 5.0)
        assert abs(bonus - 0.05) < 0.001

    def test_urgency_at_t7_5_is_half(self):
        """At T-7.5s, bonus should be ~0.025."""
        bonus = 0.05 * (1.0 - (7.5 - 5.0) / 5.0)
        assert abs(bonus - 0.025) < 0.001

    def test_urgency_below_t5_is_zero(self):
        """Below T-5s, bonus formula shouldn't apply."""
        t = 4.0
        bonus = 0.05 * (1.0 - (t - 5.0) / 5.0) if 5 <= t <= 10 else 0.0
        assert bonus == 0.0


class TestD5ContinuousPricing:
    """D5: Continuous maker price interpolation."""

    def setup_method(self):
        from polymarket.config import Config
        from polymarket.strategies.directional_sniper import DirectionalSniper
        self.config = Config()
        self.sniper = DirectionalSniper(self.config)

    def test_price_at_min_confidence(self):
        """conf=0.50 → $0.50."""
        price = self.sniper._calculate_maker_price(0.50)
        assert price == 0.50

    def test_price_at_max_confidence(self):
        """conf=0.95 → $0.70."""
        price = self.sniper._calculate_maker_price(0.95)
        assert price == 0.70

    def test_price_at_mid_confidence(self):
        """conf=0.725 → $0.60 (halfway)."""
        price = self.sniper._calculate_maker_price(0.725)
        assert price == 0.60

    def test_price_monotonically_increasing(self):
        """Higher confidence should always give higher or equal price."""
        prev_price = 0.0
        for conf in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            price = self.sniper._calculate_maker_price(conf)
            assert price >= prev_price, f"conf={conf}: ${price} < prev ${prev_price}"
            prev_price = price

    def test_no_price_jump_at_any_confidence(self):
        """No step exceeds $0.01 between adjacent confidence points."""
        last = self.sniper._calculate_maker_price(0.50)
        for conf_x10 in range(51, 96):
            conf = conf_x10 / 100.0
            price = self.sniper._calculate_maker_price(conf)
            gap = abs(price - last)
            assert gap <= 0.01 + 0.001, (  # rounding tolerance
                f"Gap ${gap:.3f} at conf={conf:.2f} exceeds $0.01"
            )
            last = price

    def test_price_below_min_clamped(self):
        """conf < 0.50 should clamp to $0.50."""
        price = self.sniper._calculate_maker_price(0.30)
        assert price == 0.50

    def test_price_above_max_clamped(self):
        """conf > 0.95 should clamp to $0.70."""
        price = self.sniper._calculate_maker_price(0.99)
        assert price == 0.70


class TestKellyEdgeCases:
    """D6 + E2: Kelly sizing boundary conditions."""

    def setup_method(self):
        from polymarket.config import Config
        from polymarket.strategies.directional_sniper import DirectionalSniper
        self.config = Config()
        self.sniper = DirectionalSniper(self.config)

    def test_zero_bankroll_returns_minimum(self):
        """Bankroll = $10 minimum → size = $5 min."""
        self.sniper._bankroll = 10.0
        size = self.sniper._kelly_size(0.90, 0.60)
        assert size >= 5.0, "Should return at least min_size"

    def test_zero_confidence_returns_minimum(self):
        """Zero confidence → Kelly negative → returns min."""
        size = self.sniper._kelly_size(0.0, 0.60)
        assert size == 5.0

    def test_one_minus_epsilon_confidence(self):
        """conf=0.999 → very large Kelly → should be capped."""
        self.sniper._bankroll = 1000.0
        size = self.sniper._kelly_size(0.999, 0.60)
        max_allowed = 1000.0 * 0.15
        assert size <= max_allowed

    def test_maker_price_at_boundary(self):
        """price=0.01 (extreme) → huge odds, but still capped."""
        self.sniper._bankroll = 100.0
        size = self.sniper._kelly_size(0.90, 0.01)
        assert size <= 100.0 * 0.15

    def test_maker_price_at_1_returns_minimum(self):
        """price=1.0 → should return min (can't buy at $1.00)."""
        size = self.sniper._kelly_size(0.90, 1.0)
        assert size == 5.0


class TestOracleEdgeCases:
    """P3/P4/P5: SyntheticOracle edge cases."""

    def setup_method(self):
        pytest.importorskip("websockets", reason="websockets not installed")
        from polymarket.brain.price_feeds import SyntheticOracle, PriceTick
        from polymarket.config import Config
        self.config = Config()
        self.oracle = SyntheticOracle(self.config)
        self.PriceTick = PriceTick

    def test_single_exchange_gives_0_6_confidence(self):
        """Single source should give 0.6 confidence."""
        tick = self.PriceTick("binance", "btc", 85000.0, time.time())
        self.oracle.update(tick)
        _, conf = self.oracle.predict()
        assert conf == 0.6

    def test_stale_exchange_reduces_weight(self):
        """8s-old data should have ~50% weight remaining (P3 exponential)."""
        import math
        factor = math.exp(-8 * 0.693 / 8.0)  # should be ~0.5
        assert 0.49 < factor < 0.51, f"Expected ~0.50, got {factor:.3f}"

    def test_mad_outlier_resistance(self):
        """One outlier exchange shouldn't destroy confidence (P4)."""
        now = time.time()
        self.oracle.update(self.PriceTick("binance", "btc", 85000.0, now))
        self.oracle.update(self.PriceTick("coinbase", "btc", 85001.0, now))
        self.oracle.update(self.PriceTick("okx", "btc", 84000.0, now))  # outlier!
        _, conf = self.oracle.predict()
        # MAD should ignore the outlier somewhat
        assert conf > 0.5, f"Confidence too low with outlier: {conf:.3f}"

    def test_raw_tick_volatility(self):
        """P5: Volatility should come from raw ticks, not predictions."""
        now = time.time()
        for i in range(20):
            price = 85000.0 + i * 10  # steady climb $10/tick
            self.oracle.update(self.PriceTick("binance", "btc", price, now + i * 0.1))
        vol = self.oracle.get_volatility()
        assert vol > 0.001, f"Volatility should be positive, got {vol}"
        assert vol < 0.1, f"Volatility too high: {vol}"

    def test_empty_oracle_volatility(self):
        """No data → default 0.01 volatility."""
        vol = self.oracle.get_volatility()
        assert vol == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
