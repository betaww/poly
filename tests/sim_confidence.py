"""Simulate confidence pipeline to find zero-signal bottleneck."""
import random
random.seed(42)

def base_confidence(dist, vol):
    z = abs(dist) / max(vol, 0.001)
    if z > 3.0: c = 0.90 + min(z - 3.0, 0.05)
    elif z > 2.0: c = 0.80 + (z - 2.0) * 0.10
    elif z > 1.0: c = 0.65 + (z - 1.0) * 0.15
    elif z > 0.5: c = 0.55 + (z - 0.5) * 0.20
    else: c = 0.50 + z * 0.10
    return min(c, 0.95), z

def d1_ema(c, trend, up=True):
    if (up and trend > 0) or (not up and trend < 0):
        return c * (1.0 + min(abs(trend) * 500, 0.10))
    else:
        return c * max(0.80, 1.0 - abs(trend) * 500)

def d2_bayesian(c, clob, up=True):
    if not (0.05 < clob < 0.95): return c
    p = c if up else (1 - c)
    p = max(0.02, min(0.98, p))
    oc = p / (1 - p); ol = clob / (1 - clob)
    f = (oc * ol) / (1 + oc * ol)
    r = f if up else (1 - f)
    return max(0.50, min(0.95, r))

def d3(c, t):
    if 5 <= t <= 10: return c + 0.05 * (1 - (t - 5) / 5)
    return c

print("=" * 90)
print("PIPELINE TRACE — BTC $85,000 typical scenarios")
print(f"{'Scenario':<40} {'dist%':>6} {'vol':>6} {'z':>5} {'base':>5} {'D1':>5} {'D2':>5} {'D3':>5} {'OK?':>5}")
print("-" * 90)
for name, dist, vol, trend, clob in [
    ("$50 move (0.059%)", 0.00059, 0.001, 0, 0.50),
    ("$100 move (0.118%)", 0.00118, 0.001, 0, 0.50),
    ("$100 + realistic vol 0.002", 0.00118, 0.002, 0, 0.50),
    ("$200 move (0.235%)", 0.00235, 0.001, 0, 0.50),
    ("$100 + EMA opposing", 0.00118, 0.001, -0.0005, 0.50),
    ("$100 + CLOB=0.47", 0.00118, 0.001, 0, 0.47),
    ("$100 + CLOB=0.45", 0.00118, 0.001, 0, 0.45),
    ("$100 + EMA opp + CLOB=0.47", 0.00118, 0.001, -0.0005, 0.47),
    ("$100 + real vol + CLOB=0.47", 0.00118, 0.002, 0, 0.47),
    ("$200 + real vol + CLOB=0.47", 0.00235, 0.002, 0, 0.47),
    ("$500 move + real vol", 0.00588, 0.002, 0, 0.50),
    ("$100 + CLOB=0.55 (agrees)", 0.00118, 0.001, 0, 0.55),
    ("$100 + EMA aligned + CLOB=0.55", 0.00118, 0.001, 0.0005, 0.55),
]:
    c, z = base_confidence(dist, vol)
    c1 = d1_ema(c, trend)
    c2 = d2_bayesian(c1, clob)
    c3 = d3(c2, 7)
    ok = "PASS" if c3 >= 0.65 else "FAIL"
    print(f"{name:<40} {dist*100:>5.3f}% {vol:>6.4f} {z:>5.2f} {c:>5.3f} {c1:>5.3f} {c2:>5.3f} {c3:>5.3f} {ok:>5}")

print("\n" + "=" * 90)
print("MONTE CARLO: 10k random 5-min rounds — signal pass rate by threshold")
print("=" * 90)
n = 10000
for thr in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]:
    cnt = 0
    for _ in range(n):
        dist = abs(random.gauss(0, 0.002))
        vol = max(abs(random.gauss(0.0015, 0.0005)), 0.001)
        trend = random.gauss(0, 0.001)
        clob = max(0.06, min(0.94, random.gauss(0.50, 0.03)))
        c, _ = base_confidence(dist, vol)
        c = d1_ema(c, trend)
        c = d2_bayesian(c, clob)
        c = d3(c, 7)
        if c >= thr: cnt += 1
    print(f"  threshold={thr:.2f}: {cnt:>5}/{n} ({cnt/n*100:>5.1f}%) signals")

print("\n" + "=" * 90)
print("FUNNEL: where do signals die? (threshold=0.65)")
print("=" * 90)
s = {'base': 0, 'D1': 0, 'D2': 0, 'D3': 0}
for _ in range(n):
    dist = abs(random.gauss(0, 0.002))
    vol = max(abs(random.gauss(0.0015, 0.0005)), 0.001)
    trend = random.gauss(0, 0.001)
    clob = max(0.06, min(0.94, random.gauss(0.50, 0.03)))
    c, _ = base_confidence(dist, vol)
    if c >= 0.65: s['base'] += 1
    c = d1_ema(c, trend)
    if c >= 0.65: s['D1'] += 1
    c = d2_bayesian(c, clob)
    if c >= 0.65: s['D2'] += 1
    c = d3(c, 7)
    if c >= 0.65: s['D3'] += 1
for k in ['base', 'D1', 'D2', 'D3']:
    print(f"  After {k:>4}: {s[k]:>5}/{n} ({s[k]/n*100:>5.1f}%)")

print("\n" + "=" * 90)
print("EV ANALYSIS: What if we traded at different thresholds?")
print("=" * 90)
print("Assuming 88.4% win rate (from 43-round data), maker price $0.60:")
for thr in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
    wr = 0.884
    ev = wr * (1 - 0.60) - (1 - wr) * 0.60
    print(f"  threshold={thr:.2f}: EV/share = +${ev:.3f} (+{ev/0.60*100:.1f}%)")
print("  Note: EV is SAME regardless of threshold because win rate is oracle-dependent,")
print("  not confidence-dependent. Lower threshold = more trades = more total profit.")
