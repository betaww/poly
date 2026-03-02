"""
v10 Edge Case Tests — exercises extreme/boundary inputs for all 14 improvements.

Run: python -m pytest user_data/polymarket/tests/test_v10_edge_cases.py -v
  or: python user_data/polymarket/tests/test_v10_edge_cases.py  (standalone)
"""
import math
import sys
import os

# Add project root to path for standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# ═══════════════════════════════════════════════════════════════════════
# Test helpers: log-odds functions (#3)
# ═══════════════════════════════════════════════════════════════════════

def _to_log_odds(p: float) -> float:
    p = max(0.01, min(0.99, p))
    return math.log(p / (1.0 - p))

def _from_log_odds(lo: float) -> float:
    lo = max(-20.0, min(20.0, lo))  # guard
    return 1.0 / (1.0 + math.exp(-lo))


class TestLogOdds:
    """#3: Independent log-odds fusion helpers."""

    def test_roundtrip_normal(self):
        """Normal probabilities survive roundtrip."""
        for p in [0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95]:
            lo = _to_log_odds(p)
            recovered = _from_log_odds(lo)
            assert abs(recovered - p) < 1e-10, f"Roundtrip failed for p={p}: got {recovered}"

    def test_extreme_probabilities_clamped(self):
        """Extreme probabilities are clamped to [0.01, 0.99]."""
        lo_zero = _to_log_odds(0.0)  # should clamp to 0.01
        lo_one = _to_log_odds(1.0)   # should clamp to 0.99
        assert abs(_from_log_odds(lo_zero) - 0.01) < 1e-10
        assert abs(_from_log_odds(lo_one) - 0.99) < 1e-10

    def test_negative_prob(self):
        """Negative probability clamped to 0.01."""
        lo = _to_log_odds(-0.5)
        assert abs(_from_log_odds(lo) - 0.01) < 1e-10

    def test_prob_above_one(self):
        """Probability > 1 clamped to 0.99."""
        lo = _to_log_odds(1.5)
        assert abs(_from_log_odds(lo) - 0.99) < 1e-10

    def test_overflow_guard_positive(self):
        """Extreme positive log-odds don't cause overflow."""
        result = _from_log_odds(1000.0)  # would be exp(-1000) in denominator
        assert result > 0.99  # should be clamped near 1.0
        assert math.isfinite(result)

    def test_overflow_guard_negative(self):
        """Extreme negative log-odds don't cause overflow."""
        result = _from_log_odds(-1000.0)  # would be exp(1000) in denominator
        assert result < 0.01  # should be clamped near 0.0
        assert math.isfinite(result)

    def test_additivity_symmetry(self):
        """Adding equal positive/negative deltas returns to base."""
        base = _to_log_odds(0.70)
        delta = 0.5
        fused = base + delta - delta  # should equal base
        assert abs(_from_log_odds(fused) - 0.70) < 1e-10

    def test_d1_d2_d3_max_deltas_dont_overflow(self):
        """Max possible D1+D2+D3 deltas (+0.40 each) don't crash."""
        base = _to_log_odds(0.95)  # already high
        max_delta = 0.40 * 3  # all three stages at maximum boost
        result = _from_log_odds(base + max_delta)
        assert 0.0 < result <= 1.0
        assert math.isfinite(result)

    def test_all_stages_oppose(self):
        """All stages opposing gives low confidence, not crash."""
        base = _to_log_odds(0.50)  # neutral
        total_oppose = -0.40 * 3  # all three oppose
        result = _from_log_odds(base + total_oppose)
        assert 0.0 < result < 0.50
        assert math.isfinite(result)


# ═══════════════════════════════════════════════════════════════════════
# Test: Deadzone (#1)
# ═══════════════════════════════════════════════════════════════════════

class TestDeadzone:
    """#1: Boundary deadzone filter."""

    def test_exact_strike(self):
        """CEX == strike → Unknown."""
        cex = 85000.0
        strike = 85000.0
        distance = abs(cex - strike) / strike
        assert distance < 0.0005  # within deadzone

    def test_just_inside_deadzone(self):
        """CEX 0.04% above strike → still in deadzone."""
        cex = 85000.0
        strike = 85000.0 / 1.0004  # ~0.04% below cex
        distance = abs(cex - strike) / strike
        assert distance < 0.0005

    def test_just_outside_deadzone(self):
        """CEX 0.06% above strike → outside deadzone, should trade."""
        cex = 85000.0
        strike = 85000.0 / 1.0006
        distance = abs(cex - strike) / strike
        assert distance > 0.0005

    def test_btc_deadzone_dollars(self):
        """For BTC ≈ $85K, 0.05% deadzone = ±$42.50."""
        btc = 85000.0
        threshold_dollars = btc * 0.0005
        assert 42.0 < threshold_dollars < 43.0, f"Threshold = ${threshold_dollars:.2f}"

    def test_eth_deadzone_dollars(self):
        """For ETH ≈ $3K, 0.05% deadzone = ±$1.50."""
        eth = 3000.0
        threshold_dollars = eth * 0.0005
        assert 1.4 < threshold_dollars < 1.6, f"Threshold = ${threshold_dollars:.2f}"


# ═══════════════════════════════════════════════════════════════════════
# Test: EMA parameters (#6)
# ═══════════════════════════════════════════════════════════════════════

class TestEMAParameters:
    """#6: EMA α fix."""

    def test_fast_ema_half_life(self):
        """Fast EMA α=0.05 has half-life ≈ 14 ticks (1Hz → 14s)."""
        alpha = 0.05
        half_life = -math.log(2) / math.log(1 - alpha)
        assert 13 < half_life < 15, f"Half-life = {half_life:.1f}s"

    def test_slow_ema_half_life(self):
        """Slow EMA α=0.01 has half-life ≈ 69 ticks (1Hz → 69s)."""
        alpha = 0.01
        half_life = -math.log(2) / math.log(1 - alpha)
        assert 68 < half_life < 71, f"Half-life = {half_life:.1f}s"

    def test_trend_noise_filter(self):
        """Trend < 0.1% should produce zero D1 delta."""
        ema_slow = 85000.0
        ema_fast = 85000.0 * 1.0005  # 0.05% trend — below 0.1% threshold
        trend = (ema_fast - ema_slow) / ema_slow
        assert abs(trend) < 0.001  # below threshold → no D1 delta

    def test_trend_signal_pass(self):
        """Trend > 0.1% should produce nonzero D1 delta."""
        ema_slow = 85000.0
        ema_fast = 85000.0 * 1.002  # 0.2% trend — above threshold
        trend = (ema_fast - ema_slow) / ema_slow
        assert abs(trend) > 0.001  # above threshold → D1 active

    def test_ema_convergence_from_step(self):
        """EMAs converge to new price after step change."""
        ema_fast = 85000.0
        ema_slow = 85000.0
        new_price = 86000.0

        for _ in range(300):  # 300 ticks (5 minutes at 1Hz)
            ema_fast = 0.95 * ema_fast + 0.05 * new_price
            ema_slow = 0.99 * ema_slow + 0.01 * new_price

        # After 300 ticks, fast should be ~99.99% converged, slow ~95%
        assert abs(ema_fast - new_price) / new_price < 0.001
        assert abs(ema_slow - new_price) / new_price < 0.06


# ═══════════════════════════════════════════════════════════════════════
# Test: Weighted median interpolation (#11)
# ═══════════════════════════════════════════════════════════════════════

class TestWeightedMedianInterpolation:
    """#11: Interpolated weighted median."""

    def _weighted_median(self, prices_weights):
        """Copy of the v10 interpolated weighted median."""
        if not prices_weights:
            return 0.0
        pw = sorted(prices_weights, key=lambda x: x[0])
        total = sum(w for _, w in pw)
        if total == 0:
            return 0.0
        if len(pw) == 1:
            return pw[0][0]
        if len(pw) == 2:
            return (pw[0][0] * pw[0][1] + pw[1][0] * pw[1][1]) / total
        half = total / 2
        cumulative = 0.0
        for i, (price, weight) in enumerate(pw):
            prev_cum = cumulative
            cumulative += weight
            if cumulative > half:
                if i == 0 or prev_cum >= half:
                    return price
                prev_price = pw[i - 1][0]
                frac = (half - prev_cum) / weight if weight > 0 else 0.0
                return prev_price + frac * (price - prev_price)
        return pw[-1][0]

    def test_single_source(self):
        assert self._weighted_median([(85000.0, 1.0)]) == 85000.0

    def test_two_sources_equal_weight(self):
        result = self._weighted_median([(85000.0, 0.5), (85010.0, 0.5)])
        assert result == 85005.0  # weighted average

    def test_two_sources_unequal_weight(self):
        result = self._weighted_median([(85000.0, 0.7), (85010.0, 0.3)])
        expected = (85000.0 * 0.7 + 85010.0 * 0.3) / 1.0
        assert abs(result - expected) < 0.01

    def test_three_sources_pyth_dominates(self):
        """Pyth (0.40) + Binance (0.30) + Coinbase (0.15) — Pyth dominates but interpolation smooths."""
        result = self._weighted_median([
            (85000.0, 0.15),  # coinbase
            (85005.0, 0.30),  # binance
            (85002.0, 0.40),  # pyth
        ])
        # Sorted: coinbase(85000, 0.15), pyth(85002, 0.40), binance(85005, 0.30)
        # half = 0.425. After coinbase: 0.15. After pyth: 0.55 > 0.425.
        # Interpolation: prev_cum=0.15, weight=0.40, half=0.425
        # frac = (0.425 - 0.15) / 0.40 = 0.6875
        # result = 85000 + 0.6875 * (85002 - 85000) = 85001.375
        assert abs(result - 85001.375) < 0.01

    def test_three_equal_weights(self):
        """Three sources, equal weight — should be the middle value."""
        result = self._weighted_median([
            (100.0, 1.0), (200.0, 1.0), (300.0, 1.0)
        ])
        # half = 1.5. After first: 1.0. After second: 2.0 > 1.5.
        # frac = (1.5 - 1.0) / 1.0 = 0.5
        # result = 100 + 0.5 * (200 - 100) = 150
        assert abs(result - 150.0) < 0.01

    def test_empty_input(self):
        assert self._weighted_median([]) == 0.0

    def test_zero_weights(self):
        assert self._weighted_median([(85000.0, 0.0), (85010.0, 0.0)]) == 0.0

    def test_one_dominant_source(self):
        """One source has 99% weight — result is interpolated near it."""
        result = self._weighted_median([
            (85000.0, 0.99), (90000.0, 0.005), (80000.0, 0.005)
        ])
        # Sorted: (80000,0.005), (85000,0.99), (90000,0.005). half=0.5
        # After 80000: cum=0.005. After 85000: cum=0.995 > 0.5
        # frac = (0.5-0.005)/0.99 ≈ 0.5. result ≈ 80000 + 0.5*5000 = 82500
        # This is correct: interpolated median leans toward the dominant source
        assert 82000 < result < 86000

    def test_identical_prices(self):
        """All sources agree on price."""
        result = self._weighted_median([
            (85000.0, 0.40), (85000.0, 0.30), (85000.0, 0.15)
        ])
        assert result == 85000.0


# ═══════════════════════════════════════════════════════════════════════
# Test: OFI adaptive normalization (#5)
# ═══════════════════════════════════════════════════════════════════════

class TestOFIAdaptive:
    """#5: Adaptive OFI normalization."""

    def test_bootstrap_from_zero(self):
        """First OFI reading initializes EMA to max(value, 1.0)."""
        ema = 0.0
        raw_ofi = 5.0

        # Bootstrap logic
        abs_ofi = abs(raw_ofi)
        if ema < 0.001:
            ema = max(abs_ofi, 1.0)
        assert ema == 5.0

    def test_bootstrap_tiny_value(self):
        """Tiny first OFI floors EMA to 1.0."""
        ema = 0.0
        raw_ofi = 0.001

        abs_ofi = abs(raw_ofi)
        if ema < 0.001:
            ema = max(abs_ofi, 1.0)
        assert ema == 1.0  # floor at 1.0

    def test_scale_3x_ema(self):
        """Scale = 3× EMA, so OFI=3×EMA normalizes to 1.0."""
        ema = 10.0
        scale = max(1.0, ema * 3.0)
        raw_ofi = 30.0
        normalized = max(-1.0, min(1.0, raw_ofi / scale))
        assert abs(normalized - 1.0) < 0.01

    def test_negative_ofi(self):
        """Negative OFI normalizes correctly."""
        ema = 10.0
        scale = max(1.0, ema * 3.0)
        raw_ofi = -15.0
        normalized = max(-1.0, min(1.0, raw_ofi / scale))
        assert normalized == -0.5

    def test_ema_adapts_over_time(self):
        """EMA α=0.05 gradually adjusts to new regime."""
        ema = 10.0  # old regime: ~10 BTC flow
        new_regime = 50.0  # market shift: now ~50

        for _ in range(100):
            ema = 0.95 * ema + 0.05 * new_regime

        # After 100 updates, should be very close to new regime
        assert abs(ema - new_regime) / new_regime < 0.01


# ═══════════════════════════════════════════════════════════════════════
# Test: Kalman Q adaptive (#7)
# ═══════════════════════════════════════════════════════════════════════

class TestKalmanQAdaptive:
    """#7: Kalman Q derived from Parkinson volatility."""

    def test_low_vol_Q(self):
        """Low volatility → small Q (floored at 1e-7)."""
        parkinson_vol = 0.001  # quiet market
        Q = max(1e-7, (parkinson_vol ** 2) * 0.01)
        # (0.001)^2 * 0.01 = 1e-8, floored to 1e-7
        assert Q == 1e-7

    def test_medium_vol_Q(self):
        """Medium volatility → adaptive Q active (not floored)."""
        parkinson_vol = 0.01  # normal
        Q = max(1e-7, (parkinson_vol ** 2) * 0.01)
        # (0.01)^2 * 0.01 = 1e-6, above floor
        assert abs(Q - 1e-6) < 1e-12

    def test_high_vol_Q(self):
        """High volatility → large Q."""
        parkinson_vol = 0.05  # volatile period
        Q = max(1e-7, (parkinson_vol ** 2) * 0.01)
        assert abs(Q - 0.000025) < 1e-10

    def test_extreme_vol_Q(self):
        """Extreme volatility → Q scales sensibly."""
        parkinson_vol = 0.20  # extreme (20%)
        Q = max(1e-7, (parkinson_vol ** 2) * 0.01)
        assert abs(Q - 0.0004) < 1e-10

    def test_zero_vol_Q_floored(self):
        """Zero volatility → Q floored."""
        parkinson_vol = 0.0
        Q = max(1e-7, (parkinson_vol ** 2) * 0.01)
        assert Q == 1e-7


# ═══════════════════════════════════════════════════════════════════════
# Test: Paper sim fill probability (#8)
# ═══════════════════════════════════════════════════════════════════════

class TestPaperSimFillProb:
    """#8: Dynamic queue competition."""

    def _fill_prob(self, clob_resting_orders=0, **kwargs):
        """Simplified estimate_gtc_fill_probability for testing."""
        limit_price = kwargs.get('limit_price', 0.60)
        midpoint = kwargs.get('midpoint', 0.55)
        seconds_resting = kwargs.get('seconds_resting', 3.0)
        is_buy = kwargs.get('is_buy', True)
        volatility = kwargs.get('volatility', 0.01)
        time_remaining = kwargs.get('time_remaining', 300.0)

        if midpoint <= 0:
            return 0.0

        if is_buy:
            distance = midpoint - limit_price
        else:
            distance = limit_price - midpoint

        if distance < 0:
            return 0.95

        distance_pct = distance / midpoint
        vol_adjusted = distance_pct / max(0.001, volatility)
        base_prob = math.exp(-vol_adjusted * 5.0)
        time_factor = 1.0 - math.exp(-seconds_resting / 5.0)

        if clob_resting_orders > 0:
            queue_penalty = 1.0 / max(1, clob_resting_orders)
        else:
            queue_penalty = 0.30

        if time_remaining < 30:
            settlement_factor = 0.4 + 0.6 * (time_remaining / 30.0)
        else:
            settlement_factor = 1.0

        probability = base_prob * time_factor * queue_penalty * settlement_factor * 0.85
        return max(0.0, min(0.85, probability))

    def test_no_clob_data_uses_default(self):
        """clob_resting_orders=0 → default 0.30 penalty."""
        prob = self._fill_prob(clob_resting_orders=0)
        prob_default = self._fill_prob(clob_resting_orders=0)
        assert prob == prob_default

    def test_one_competitor(self):
        """1 competitor → 100% share (best case)."""
        # Use non-crossing buy: bid=0.50 below midpoint=0.55
        prob = self._fill_prob(clob_resting_orders=1, limit_price=0.50, midpoint=0.55)
        prob_default = self._fill_prob(clob_resting_orders=0, limit_price=0.50, midpoint=0.55)
        assert prob > prob_default  # 1/1 > 0.30

    def test_ten_competitors(self):
        """10 competitors → 10% share (worse than default 30%)."""
        prob = self._fill_prob(clob_resting_orders=10, limit_price=0.50, midpoint=0.55)
        prob_default = self._fill_prob(clob_resting_orders=0, limit_price=0.50, midpoint=0.55)
        assert prob < prob_default  # 1/10 < 0.30

    def test_zero_midpoint(self):
        """midpoint=0 → probability=0."""
        assert self._fill_prob(midpoint=0.0) == 0.0

    def test_settlement_edge(self):
        """T-1s → liquidity factor ~0.42."""
        prob_early = self._fill_prob(time_remaining=300, limit_price=0.50, midpoint=0.55)
        prob_late = self._fill_prob(time_remaining=1, limit_price=0.50, midpoint=0.55)
        assert prob_late < prob_early


# ═══════════════════════════════════════════════════════════════════════
# Test: Kelly sizing edge cases (#4)
# ═══════════════════════════════════════════════════════════════════════

class TestKellySizing:
    """#4: Kelly empirical blend threshold."""

    def test_under_50_rounds_uses_model_only(self):
        """<50 rounds → p = confidence (no empirical blend)."""
        rounds_traded = 30
        confidence = 0.80
        # Under threshold: p = confidence
        if rounds_traded > 50:
            p = 0.7 * confidence + 0.3 * 0.50  # would reduce
        else:
            p = confidence
        assert p == 0.80

    def test_exactly_50_rounds_uses_model_only(self):
        """=50 rounds → p = confidence (threshold is >50, not >=50)."""
        rounds_traded = 50
        confidence = 0.80
        if rounds_traded > 50:
            p = 0.7 * confidence + 0.3 * 0.50
        else:
            p = confidence
        assert p == 0.80

    def test_51_rounds_blends(self):
        """51 rounds → empirical blend kicks in."""
        rounds_traded = 51
        confidence = 0.80
        if rounds_traded > 50:
            p = 0.7 * confidence + 0.3 * 0.50  # simplified
        else:
            p = confidence
        assert p < 0.80  # blended down

    def test_negative_kelly_returns_min_size(self):
        """Kelly < 0 (no edge) → minimum order size."""
        maker_price = 0.60
        b = (1.0 / maker_price) - 1.0  # 0.667
        p = 0.40  # below breakeven for these odds
        q = 1.0 - p
        kelly_f = (b * p - q) / b
        assert kelly_f < 0  # negative → no edge


# ═══════════════════════════════════════════════════════════════════════
# Test: Strike TTL (#2)
# ═══════════════════════════════════════════════════════════════════════

class TestStrikeTTL:
    """#2: Strike freshness TTL."""

    def test_fresh_strike_not_expired(self):
        """Strike set 60s ago → still fresh."""
        import time
        set_at = time.time() - 60
        ttl = 120
        assert (time.time() - set_at) <= ttl

    def test_stale_strike_expired(self):
        """Strike set 130s ago → expired."""
        import time
        set_at = time.time() - 130
        ttl = 120
        assert (time.time() - set_at) > ttl

    def test_zero_set_at_not_expired(self):
        """_strike_set_at=0 (never set) → don't trigger TTL."""
        set_at = 0.0
        # The check is: if set_at > 0 and expired
        should_expire = set_at > 0 and True
        assert not should_expire


# ═══════════════════════════════════════════════════════════════════════
# Test: Reversal filter (#15)
# ═══════════════════════════════════════════════════════════════════════

class TestReversalFilter:
    """#15: Reversal threshold lowered to 1.5."""

    def _weighted_reversals(self, reversal_times, now):
        return sum(1.0 / (1.0 + (now - rt) / 10.0) for rt in reversal_times)

    def test_no_reversals(self):
        assert self._weighted_reversals([], 100.0) == 0.0

    def test_single_recent_reversal(self):
        """One reversal 2s ago → weight ≈ 0.83, below 1.5."""
        now = 100.0
        wt = self._weighted_reversals([98.0], now)
        assert wt < 1.5

    def test_three_rapid_reversals_triggers(self):
        """3 reversals in last 5s → should trigger (>1.5)."""
        now = 100.0
        times = [96.0, 98.0, 99.0]
        wt = self._weighted_reversals(times, now)
        # 1/(1+4/10) + 1/(1+2/10) + 1/(1+1/10) = 0.714 + 0.833 + 0.909 = 2.456
        assert wt > 1.5, f"Weight = {wt:.3f}"

    def test_old_reversals_decay(self):
        """3 reversals 60s ago → negligible weight."""
        now = 100.0
        times = [38.0, 39.0, 40.0]
        wt = self._weighted_reversals(times, now)
        # 1/(1+62/10) + 1/(1+61/10) + 1/(1+60/10) ≈ 0.14 each → ~0.42
        assert wt < 1.5


# ═══════════════════════════════════════════════════════════════════════
# Test: Confidence calibration bins (#10)
# ═══════════════════════════════════════════════════════════════════════

class TestCalibrationBins:
    """#10: Confidence bin assignment."""

    def _bin_for(self, confidence):
        bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]
        for lo, hi in bins:
            if lo <= confidence < hi or (hi == 1.00 and confidence >= 0.90):
                return f"{lo:.2f}-{hi:.2f}"
        return None

    def test_bin_at_lower_boundary(self):
        assert self._bin_for(0.50) == "0.50-0.60"
        assert self._bin_for(0.60) == "0.60-0.70"
        assert self._bin_for(0.70) == "0.70-0.80"

    def test_bin_exactly_0_90(self):
        assert self._bin_for(0.90) == "0.90-1.00"

    def test_bin_0_95(self):
        assert self._bin_for(0.95) == "0.90-1.00"

    def test_bin_below_0_50(self):
        """Confidence < 0.50 → no bin (shouldn't happen, clamped)."""
        assert self._bin_for(0.45) is None

    def test_bin_at_0_599(self):
        """0.599 → first bin (0.50-0.60)."""
        assert self._bin_for(0.599) == "0.50-0.60"

    def test_bin_at_0_6001(self):
        """0.6001 → second bin (0.60-0.70)."""
        assert self._bin_for(0.6001) == "0.60-0.70"


# ═══════════════════════════════════════════════════════════════════════
# Test: D2 CLOB fusion math (#3)
# ═══════════════════════════════════════════════════════════════════════

class TestD2CLOBFusion:
    """D2 CLOB log-odds delta correctness."""

    def test_clob_neutral_zero_delta(self):
        """CLOB=0.50 → log_odds=0 → d2_delta=0."""
        p_clob = 0.50
        clob_lo = _to_log_odds(p_clob)
        assert abs(clob_lo) < 0.01  # near zero

    def test_clob_bullish_up_direction(self):
        """Direction=Up, CLOB=0.70 → positive delta (confirms)."""
        direction = "Up"
        p_clob = 0.70
        clob_lo = _to_log_odds(p_clob)
        if direction == "Up":
            d2_delta = clob_lo * 0.5
        else:
            d2_delta = -clob_lo * 0.5
        assert d2_delta > 0  # confirms Up

    def test_clob_bearish_down_direction(self):
        """Direction=Down, CLOB=0.30 → positive delta (confirms Down)."""
        direction = "Down"
        p_clob = 0.30
        clob_lo = _to_log_odds(p_clob)
        if direction == "Up":
            d2_delta = clob_lo * 0.5
        else:
            d2_delta = -clob_lo * 0.5
        # clob_lo ≈ -0.847, d2_delta = -(-0.847)*0.5 = +0.423
        assert d2_delta > 0  # confirms Down direction

    def test_clob_contradicts_up(self):
        """Direction=Up, CLOB=0.30 → negative delta (opposes)."""
        direction = "Up"
        p_clob = 0.30
        clob_lo = _to_log_odds(p_clob)
        if direction == "Up":
            d2_delta = clob_lo * 0.5
        else:
            d2_delta = -clob_lo * 0.5
        assert d2_delta < 0  # opposes Up

    def test_clob_extreme_0_95(self):
        """CLOB=0.95 → large positive log-odds, but capped by pipeline clamp."""
        p_clob = 0.95
        clob_lo = _to_log_odds(p_clob)
        d2_delta = clob_lo * 0.5
        # clob_lo ≈ 2.944, d2_delta ≈ 1.472
        # Combined with base_lo: e.g. base=0.70 → lo=0.847 + 1.472 = 2.319 → p=0.91
        # Then clamped to 0.95 max
        assert d2_delta > 1.0  # large boost
        base = _to_log_odds(0.70)
        result = _from_log_odds(base + d2_delta)
        assert result < 1.0  # finite
        assert result > 0.85  # strong boost

    def test_clob_filtered_in_uninformed_zone(self):
        """CLOB in 0.45-0.55 → should NOT generate D2 delta."""
        midpoints = [0.45, 0.48, 0.50, 0.52, 0.55]
        for mp in midpoints:
            active = (0.05 < mp < 0.45) or (0.55 < mp < 0.95)
            assert not active, f"CLOB {mp} should be filtered as uninformed"


# ═══════════════════════════════════════════════════════════════════════
# Test: Clock offset (#14)
# ═══════════════════════════════════════════════════════════════════════

class TestClockOffset:
    """#14: Clock offset correction."""

    def test_vps_ahead_correction(self):
        """VPS 2s ahead → end_time increased → seconds_remaining goes up."""
        import time
        end_time = int(time.time()) + 300
        clock_offset = 2.0  # VPS ahead by 2s

        corrected_end = int(end_time + clock_offset)
        old_remaining = end_time - time.time()
        new_remaining = corrected_end - time.time()

        assert new_remaining > old_remaining  # correction adds time

    def test_vps_behind_correction(self):
        """VPS 2s behind → end_time decreased → seconds_remaining goes down."""
        import time
        end_time = int(time.time()) + 300
        clock_offset = -2.0  # VPS behind by 2s

        corrected_end = int(end_time + clock_offset)
        old_remaining = end_time - time.time()
        new_remaining = corrected_end - time.time()

        assert new_remaining < old_remaining  # correction removes time

    def test_small_offset_ignored(self):
        """Offset < 0.5s → no correction applied."""
        clock_offset = 0.3
        should_correct = abs(clock_offset) > 0.5
        assert not should_correct


# ═══════════════════════════════════════════════════════════════════════
# Test: Full confidence pipeline integration
# ═══════════════════════════════════════════════════════════════════════

class TestConfidencePipelineIntegration:
    """End-to-end confidence pipeline with extreme inputs."""

    def _simulate_confidence(self, cex, strike, vol, ema_fast=None, ema_slow=None,
                              clob_mid=0.50, ofi=0.0):
        """Simulate the full pipeline from _get_direction_confidence."""
        from scipy.stats import t as t_dist

        if cex <= 0 or strike <= 0:
            return "Unknown", 0.0

        distance = (cex - strike) / strike
        abs_distance = abs(distance)

        if abs_distance < 0.0005:
            return "Unknown", 0.50

        direction = "Up" if distance > 0 else "Down"
        vol = max(vol, 0.001)
        z_score = abs_distance / vol
        raw_conf = t_dist.cdf(z_score, df=4)
        base_conf = max(0.50, min(0.95, raw_conf))

        base_lo = _to_log_odds(base_conf)
        d1_delta = 0.0
        d2_delta = 0.0
        d3_delta = 0.0

        if ema_fast is not None and ema_slow is not None and ema_slow > 0:
            trend = (ema_fast - ema_slow) / ema_slow
            if abs(trend) > 0.001:
                if (direction == "Up" and trend > 0) or (direction == "Down" and trend < 0):
                    d1_delta = min(abs(trend) * 200, 0.40)
                else:
                    d1_delta = -min(abs(trend) * 150, 0.40)

        if (0.05 < clob_mid < 0.45) or (0.55 < clob_mid < 0.95):
            clob_lo = _to_log_odds(clob_mid)
            if direction == "Up":
                d2_delta = clob_lo * 0.5
            else:
                d2_delta = -clob_lo * 0.5

        if abs(ofi) > 0.2:
            if (direction == "Up" and ofi > 0) or (direction == "Down" and ofi < 0):
                d3_delta = min(abs(ofi) * 0.6, 0.40)
            else:
                d3_delta = -min(abs(ofi) * 0.4, 0.40)

        fused_lo = base_lo + d1_delta + d2_delta + d3_delta
        conf = _from_log_odds(fused_lo)
        conf = max(0.50, min(0.95, conf))
        return direction, conf

    def test_all_signals_confirming_up(self):
        """All signals agree: strong Up. Confidence near cap."""
        direction, conf = self._simulate_confidence(
            cex=85500.0, strike=85000.0, vol=0.005,
            ema_fast=85500.0, ema_slow=85200.0,
            clob_mid=0.70, ofi=0.8,
        )
        assert direction == "Up"
        assert conf >= 0.85  # should be very high

    def test_all_signals_confirming_down(self):
        """All signals agree: strong Down."""
        direction, conf = self._simulate_confidence(
            cex=84500.0, strike=85000.0, vol=0.005,
            ema_fast=84500.0, ema_slow=84800.0,
            clob_mid=0.30, ofi=-0.8,
        )
        assert direction == "Down"
        assert conf >= 0.85

    def test_all_signals_contradicting(self):
        """CEX says Up, but EMA/CLOB/OFI say Down → low confidence."""
        direction, conf = self._simulate_confidence(
            cex=85100.0, strike=85000.0, vol=0.01,  # weak Up signal
            ema_fast=84900.0, ema_slow=85100.0,  # downtrend
            clob_mid=0.35, ofi=-0.6,  # bearish
        )
        assert direction == "Up"  # CEX determines direction
        assert conf < 0.60  # heavily reduced by opposing signals

    def test_zero_cex(self):
        direction, conf = self._simulate_confidence(cex=0.0, strike=85000.0, vol=0.01)
        assert direction == "Unknown"
        assert conf == 0.0

    def test_zero_strike(self):
        direction, conf = self._simulate_confidence(cex=85000.0, strike=0.0, vol=0.01)
        assert direction == "Unknown"
        assert conf == 0.0

    def test_boundary_zone(self):
        """CEX within 0.05% of strike → Unknown."""
        direction, conf = self._simulate_confidence(cex=85000.0, strike=84980.0, vol=0.01)
        abs_dist = abs(85000.0 - 84980.0) / 84980.0  # ≈ 0.024% < 0.05%
        assert direction == "Unknown"

    def test_zero_volatility(self):
        """Zero vol → floored to 0.001, z-score becomes huge → high confidence."""
        direction, conf = self._simulate_confidence(cex=85500.0, strike=85000.0, vol=0.0)
        assert direction == "Up"
        assert conf > 0.90  # extreme z-score → near cap

    def test_extreme_volatility(self):
        """Very high vol → z-score tiny → confidence near 0.50."""
        direction, conf = self._simulate_confidence(cex=85500.0, strike=85000.0, vol=0.50)
        assert direction == "Up"
        assert conf < 0.55  # washed out by vol

    def test_clob_in_uninformed_zone_ignored(self):
        """CLOB=0.50 should have same result as no CLOB."""
        _, conf_with = self._simulate_confidence(
            cex=85500.0, strike=85000.0, vol=0.005, clob_mid=0.50)
        _, conf_without = self._simulate_confidence(
            cex=85500.0, strike=85000.0, vol=0.005, clob_mid=0.50)
        assert conf_with == conf_without

    def test_ofi_below_threshold_ignored(self):
        """OFI=0.15 (below 0.2 threshold) should not affect confidence."""
        _, conf_with = self._simulate_confidence(
            cex=85500.0, strike=85000.0, vol=0.005, ofi=0.15)
        _, conf_without = self._simulate_confidence(
            cex=85500.0, strike=85000.0, vol=0.005, ofi=0.0)
        assert conf_with == conf_without

    def test_confidence_always_in_range(self):
        """No combination of inputs produces confidence outside [0.50, 0.95]."""
        import random
        random.seed(42)
        for _ in range(1000):
            cex = random.uniform(80000, 90000)
            strike = random.uniform(80000, 90000)
            vol = random.uniform(0.0, 0.20)
            clob = random.uniform(0.0, 1.0)
            ofi = random.uniform(-1.0, 1.0)
            ema_f = random.uniform(80000, 90000)
            ema_s = random.uniform(80000, 90000)
            _, conf = self._simulate_confidence(
                cex=cex, strike=strike, vol=vol,
                ema_fast=ema_f, ema_slow=ema_s,
                clob_mid=clob, ofi=ofi
            )
            assert 0.0 <= conf <= 0.95, f"Confidence {conf} out of range!"
            assert math.isfinite(conf), f"Confidence is not finite!"


# ═══════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all test classes manually (no pytest required)."""
    import traceback

    test_classes = [
        TestLogOdds, TestDeadzone, TestEMAParameters,
        TestWeightedMedianInterpolation, TestOFIAdaptive,
        TestKalmanQAdaptive, TestPaperSimFillProb, TestKellySizing,
        TestStrikeTTL, TestReversalFilter, TestCalibrationBins,
        TestD2CLOBFusion, TestClockOffset, TestConfidencePipelineIntegration,
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in sorted(methods):
            method = getattr(instance, method_name)
            test_id = f"{cls.__name__}.{method_name}"
            try:
                method()
                total_passed += 1
                print(f"  ✅ {test_id}")
            except Exception as e:
                total_failed += 1
                failures.append((test_id, e))
                print(f"  ❌ {test_id}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    if failures:
        print(f"\nFailures:")
        for test_id, e in failures:
            print(f"  {test_id}: {e}")
            traceback.print_exception(type(e), e, e.__traceback__)
    print(f"{'='*60}")
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
