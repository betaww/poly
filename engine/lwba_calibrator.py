"""
LWBA Calibrator — compare our LWBA shadow price against Chainlink settlement.

Captures per-round LWBA snapshots, verifies against Polymarket actual
settlement via Gamma API, and produces accuracy statistics + tuning
suggestions for the LWBA engine parameters.

Runs alongside SettlementVerifier but focuses specifically on:
1. LWBA Mid vs CEX spot — which better predicts Chainlink direction?
2. Per-exchange breakdown — which exchange is most aligned with Chainlink?
3. Boundary rounds — where |price - strike| < 0.05%, accuracy breakdown.
4. Parameter tuning — max_levels, exchange weighting, deadzone width.
"""
import asyncio
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field

import httpx

from .lwba_engine import LWBAResult

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
VERIFICATION_DELAY_S = 60
MAX_RETRIES = 3
RETRY_DELAY_S = 30


@dataclass
class CalibrationSnapshot:
    """Per-round LWBA calibration data."""
    slug: str
    asset: str
    timestamp: float
    # Our LWBA
    lwba_mid: float
    lwba_bid: float
    lwba_ask: float
    lwba_spread_bps: float
    lwba_sources: int
    # Per-exchange LWBA mids
    binance_mid: float
    okx_mid: float
    coinbase_mid: float
    # Reference prices
    cex_price: float
    strike: float
    confidence: float
    # Direction calls
    lwba_direction: str  # "Up"/"Down" based on LWBA vs strike
    cex_direction: str   # "Up"/"Down" based on CEX vs strike
    # Actual (filled after verification)
    actual_direction: str = "Unknown"
    lwba_match: bool = False
    cex_match: bool = False


class LWBACalibrator:
    """Compare LWBA shadow price against Chainlink settlement outcomes.

    Usage (from vps_runner):
        calibrator = LWBACalibrator()
        # At round end:
        calibrator.snapshot_round(slug, asset, strike, lwba_result, cex_price, conf)
        # Periodically:
        report = calibrator.get_accuracy_report()
        logger.info(report)
    """

    def __init__(self, db_path: str = "data/paper_trades.db"):
        self._client = httpx.AsyncClient(timeout=10, follow_redirects=True)
        self._db_path = db_path
        self._pending_tasks: set[asyncio.Task] = set()
        self._snapshots: list[CalibrationSnapshot] = []
        self._init_db()
        self._load_stats()

    # ── SQLite ──────────────────────────────────────────────────────────────

    def _init_db(self):
        """Create lwba_calibration table if not exists."""
        try:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lwba_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT,
                    asset TEXT,
                    timestamp REAL,
                    lwba_mid REAL,
                    lwba_bid REAL,
                    lwba_ask REAL,
                    lwba_spread_bps REAL,
                    lwba_sources INTEGER,
                    cex_price REAL,
                    strike REAL,
                    confidence REAL,
                    binance_mid REAL DEFAULT 0,
                    okx_mid REAL DEFAULT 0,
                    coinbase_mid REAL DEFAULT 0,
                    lwba_direction TEXT,
                    cex_direction TEXT,
                    actual_direction TEXT DEFAULT 'Unknown',
                    lwba_match INTEGER DEFAULT 0,
                    cex_match INTEGER DEFAULT 0
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"LWBA Calibrator DB init failed: {e}")

    def _persist(self, snap: CalibrationSnapshot):
        """Write a calibration record."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO lwba_calibration "
                "(slug, asset, timestamp, lwba_mid, lwba_bid, lwba_ask, "
                "lwba_spread_bps, lwba_sources, cex_price, strike, confidence, "
                "binance_mid, okx_mid, coinbase_mid, "
                "lwba_direction, cex_direction, actual_direction, "
                "lwba_match, cex_match) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (snap.slug, snap.asset, snap.timestamp,
                 snap.lwba_mid, snap.lwba_bid, snap.lwba_ask,
                 snap.lwba_spread_bps, snap.lwba_sources,
                 snap.cex_price, snap.strike, snap.confidence,
                 snap.binance_mid, snap.okx_mid, snap.coinbase_mid,
                 snap.lwba_direction, snap.cex_direction,
                 snap.actual_direction,
                 int(snap.lwba_match), int(snap.cex_match))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"LWBA Calibrator DB write failed: {e}")

    def _update_actual(self, slug: str, actual: str, lwba_match: bool, cex_match: bool):
        """Update a pending record with actual settlement."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "UPDATE lwba_calibration SET actual_direction=?, lwba_match=?, cex_match=? "
                "WHERE slug=? AND actual_direction='Unknown'",
                (actual, int(lwba_match), int(cex_match), slug)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"LWBA Calibrator DB update failed: {e}")

    def _load_stats(self):
        """Load historical snapshots for reporting."""
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                "SELECT slug, asset, timestamp, lwba_mid, lwba_bid, lwba_ask, "
                "lwba_spread_bps, lwba_sources, cex_price, strike, confidence, "
                "binance_mid, okx_mid, coinbase_mid, "
                "lwba_direction, cex_direction, actual_direction, "
                "lwba_match, cex_match "
                "FROM lwba_calibration WHERE actual_direction != 'Unknown' "
                "ORDER BY timestamp DESC LIMIT 500"
            ).fetchall()
            conn.close()
            self._snapshots = [
                CalibrationSnapshot(
                    slug=r[0], asset=r[1], timestamp=r[2],
                    lwba_mid=r[3], lwba_bid=r[4], lwba_ask=r[5],
                    lwba_spread_bps=r[6], lwba_sources=r[7],
                    cex_price=r[8], strike=r[9], confidence=r[10],
                    binance_mid=r[11], okx_mid=r[12], coinbase_mid=r[13],
                    lwba_direction=r[14], cex_direction=r[15],
                    actual_direction=r[16],
                    lwba_match=bool(r[17]), cex_match=bool(r[18]),
                ) for r in rows
            ]
        except Exception:
            self._snapshots = []

    # ── Core API ────────────────────────────────────────────────────────────

    def snapshot_round(
        self,
        slug: str,
        asset: str,
        strike: float,
        lwba_result: LWBAResult | None,
        cex_price: float,
        confidence: float = 0.0,
    ):
        """Capture a LWBA snapshot at round end and schedule verification.

        Called by vps_runner at the end of each round.
        """
        if strike <= 0:
            return

        lwba_mid = lwba_result.mid if lwba_result and lwba_result.is_valid else 0.0
        lwba_bid = lwba_result.bid if lwba_result else 0.0
        lwba_ask = lwba_result.ask if lwba_result else 0.0
        lwba_spread = lwba_result.spread_bps if lwba_result else 0.0
        lwba_sources = lwba_result.n_sources if lwba_result else 0
        exchange_mids = lwba_result.exchange_mids if lwba_result else {}

        # Direction calls
        lwba_dir = "Up" if lwba_mid > strike else "Down" if lwba_mid > 0 else "Unknown"
        cex_dir = "Up" if cex_price > strike else "Down" if cex_price > 0 else "Unknown"

        snap = CalibrationSnapshot(
            slug=slug, asset=asset, timestamp=time.time(),
            lwba_mid=lwba_mid, lwba_bid=lwba_bid, lwba_ask=lwba_ask,
            lwba_spread_bps=lwba_spread, lwba_sources=lwba_sources,
            binance_mid=exchange_mids.get("binance", 0.0),
            okx_mid=exchange_mids.get("okx", 0.0),
            coinbase_mid=exchange_mids.get("coinbase", 0.0),
            cex_price=cex_price, strike=strike, confidence=confidence,
            lwba_direction=lwba_dir, cex_direction=cex_dir,
        )

        self._persist(snap)

        # Schedule delayed verification
        task = asyncio.create_task(
            self._verify_after_delay(snap)
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _verify_after_delay(self, snap: CalibrationSnapshot):
        """Wait for settlement, then compare."""
        await asyncio.sleep(VERIFICATION_DELAY_S)

        actual = "Unknown"
        for attempt in range(MAX_RETRIES):
            try:
                actual = await self._fetch_actual(snap.slug)
                if actual != "Unknown":
                    break
            except Exception as e:
                logger.debug(f"LWBA Cal: fetch failed for {snap.slug}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY_S)

        if actual == "Unknown":
            return

        snap.actual_direction = actual
        snap.lwba_match = (snap.lwba_direction == actual) if snap.lwba_direction != "Unknown" else False
        snap.cex_match = (snap.cex_direction == actual) if snap.cex_direction != "Unknown" else False

        self._update_actual(snap.slug, actual, snap.lwba_match, snap.cex_match)
        self._snapshots.append(snap)

        # Log individual result
        lwba_icon = "✅" if snap.lwba_match else "❌"
        cex_icon = "✅" if snap.cex_match else "❌"
        dist_bps = abs(snap.lwba_mid - snap.strike) / snap.strike * 10000 if snap.strike > 0 and snap.lwba_mid > 0 else 0
        logger.info(
            f"🔬 LWBA Cal: {snap.asset.upper()} actual={actual} | "
            f"LWBA={snap.lwba_direction}{lwba_icon} CEX={snap.cex_direction}{cex_icon} | "
            f"lwba=${snap.lwba_mid:,.2f} cex=${snap.cex_price:,.2f} strike=${snap.strike:,.2f} | "
            f"dist={dist_bps:.1f}bps spread={snap.lwba_spread_bps:.1f}bps"
        )

    async def _fetch_actual(self, slug: str) -> str:
        """Fetch actual settlement direction from Gamma API."""
        try:
            resp = await self._client.get(
                f"{GAMMA_API}/markets",
                params={"slug": slug, "closed": "true"},
            )
            if resp.status_code != 200:
                return "Unknown"
            markets = resp.json()
            if not markets:
                return "Unknown"

            market = markets[0]
            outcomes = market.get("outcomePrices", "")
            if isinstance(outcomes, str):
                import json
                try:
                    outcomes = json.loads(outcomes)
                except Exception:
                    return "Unknown"

            if len(outcomes) >= 2:
                up_price = float(outcomes[0])
                down_price = float(outcomes[1])
                if up_price > 0.9:
                    return "Up"
                elif down_price > 0.9:
                    return "Down"
            return "Unknown"
        except Exception:
            return "Unknown"

    # ── Reporting ───────────────────────────────────────────────────────────

    def get_accuracy_report(self) -> str:
        """Generate accuracy comparison report."""
        resolved = [s for s in self._snapshots if s.actual_direction != "Unknown"]
        if not resolved:
            return "📊 LWBA Calibration: no data yet"

        n = len(resolved)
        lwba_correct = sum(1 for s in resolved if s.lwba_match)
        cex_correct = sum(1 for s in resolved if s.cex_match)
        lwba_pct = lwba_correct / n * 100
        cex_pct = cex_correct / n * 100
        advantage = lwba_pct - cex_pct

        lines = [
            f"📊 LWBA Calibration ({n} rounds):",
            f"  Overall: LWBA方向 {lwba_pct:.1f}% | CEX方向 {cex_pct:.1f}% | "
            f"LWBA {advantage:+.1f}% {'优势' if advantage > 0 else '劣势'}",
        ]

        # Boundary rounds (|price - strike| < 0.05%)
        boundary = [s for s in resolved
                     if s.lwba_mid > 0 and s.strike > 0
                     and abs(s.lwba_mid - s.strike) / s.strike < 0.0005]
        if boundary:
            b_n = len(boundary)
            b_lwba = sum(1 for s in boundary if s.lwba_match) / b_n * 100
            b_cex = sum(1 for s in boundary if s.cex_match) / b_n * 100
            lines.append(
                f"  边界轮(<0.05%): LWBA {b_lwba:.0f}% vs CEX {b_cex:.0f}% "
                f"({b_n} rounds)"
            )

        # Per exchange
        lines.append("  ─── 各交易所独立准确率 ───")
        for ex_name, ex_attr in [("Binance", "binance_mid"),
                                  ("OKX", "okx_mid"),
                                  ("Coinbase", "coinbase_mid")]:
            ex_data = [(getattr(s, ex_attr), s.strike, s.actual_direction)
                       for s in resolved if getattr(s, ex_attr) > 0 and s.strike > 0]
            if ex_data:
                ex_correct = sum(
                    1 for mid, strike, actual in ex_data
                    if ("Up" if mid > strike else "Down") == actual
                )
                ex_n = len(ex_data)
                lines.append(f"  {ex_name:10s}: {ex_correct/ex_n*100:.0f}% ({ex_correct}/{ex_n})")

        # Spread vs accuracy correlation
        narrow = [s for s in resolved if s.lwba_spread_bps < 5]
        wide = [s for s in resolved if s.lwba_spread_bps >= 5]
        if narrow and wide:
            narrow_acc = sum(1 for s in narrow if s.lwba_match) / len(narrow) * 100
            wide_acc = sum(1 for s in wide if s.lwba_match) / len(wide) * 100
            lines.append(f"  ─── Spread 与准确率 ───")
            lines.append(f"  Spread<5bps: {narrow_acc:.0f}% ({len(narrow)} rounds)")
            lines.append(f"  Spread≥5bps: {wide_acc:.0f}% ({len(wide)} rounds)")

        return "\n".join(lines)

    def get_tuning_suggestions(self) -> str:
        """Generate parameter tuning suggestions based on data."""
        resolved = [s for s in self._snapshots if s.actual_direction != "Unknown"]
        if len(resolved) < 20:
            return "⚙️ 需要 20+ 轮数据才能给出调优建议"

        suggestions = ["⚙️ LWBA 调优建议:"]

        # 1. Exchange recommendation
        exchange_acc = {}
        for ex_name, ex_attr in [("binance", "binance_mid"),
                                  ("okx", "okx_mid"),
                                  ("coinbase", "coinbase_mid")]:
            data = [(getattr(s, ex_attr), s.strike, s.actual_direction)
                    for s in resolved if getattr(s, ex_attr) > 0 and s.strike > 0]
            if data:
                correct = sum(
                    1 for mid, strike, actual in data
                    if ("Up" if mid > strike else "Down") == actual
                )
                exchange_acc[ex_name] = correct / len(data)

        if exchange_acc:
            best = max(exchange_acc, key=exchange_acc.get)
            worst = min(exchange_acc, key=exchange_acc.get)
            spread = (exchange_acc[best] - exchange_acc[worst]) * 100
            if spread > 10:
                suggestions.append(
                    f"  ⚠️ {worst} 准确率最低 ({exchange_acc[worst]:.0%})，"
                    f"考虑降权或排除"
                )
            else:
                suggestions.append(f"  ✅ 各交易所准确率接近 (差距<{spread:.0f}pp)")

        # 2. LWBA vs CEX
        n = len(resolved)
        lwba_pct = sum(1 for s in resolved if s.lwba_match) / n
        cex_pct = sum(1 for s in resolved if s.cex_match) / n
        if lwba_pct > cex_pct + 0.03:
            suggestions.append(
                f"  ✅ LWBA 方向优于 CEX ({lwba_pct:.0%} vs {cex_pct:.0%})，继续使用 LWBA"
            )
        elif cex_pct > lwba_pct + 0.03:
            suggestions.append(
                f"  ⚠️ CEX 方向优于 LWBA ({cex_pct:.0%} vs {lwba_pct:.0%})，"
                f"考虑回退到 CEX spot"
            )
        else:
            suggestions.append(
                f"  ℹ️ LWBA 和 CEX 准确率接近 ({lwba_pct:.0%} vs {cex_pct:.0%})"
            )

        # 3. Deadzone recommendation
        boundary = [s for s in resolved
                     if s.lwba_mid > 0 and s.strike > 0
                     and abs(s.lwba_mid - s.strike) / s.strike < 0.0005]
        if boundary:
            boundary_acc = sum(1 for s in boundary if s.lwba_match) / len(boundary)
            if boundary_acc < 0.6:
                suggestions.append(
                    f"  ⚠️ 边界轮准确率低 ({boundary_acc:.0%})，"
                    f"考虑加宽 deadzone (当前 0.05% → 建议 0.08%)"
                )
            elif boundary_acc > 0.8:
                suggestions.append(
                    f"  ✅ 边界轮准确率高 ({boundary_acc:.0%})，"
                    f"可缩窄 deadzone 捕获更多机会"
                )

        return "\n".join(suggestions)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def close(self):
        """Cancel pending tasks and close HTTP client."""
        for task in self._pending_tasks:
            task.cancel()
        self._pending_tasks.clear()
        await self._client.aclose()
