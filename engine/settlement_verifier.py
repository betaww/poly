"""
Settlement Verifier — compares paper settlements against Polymarket actual results.

After each round ends, schedules a delayed check against the Gamma API to get
the real on-chain resolution. Tracks accuracy metrics and logs discrepancies.

This directly addresses the self-verification bias: our paper settlements use
the same CEX price source as predictions, which may differ from Chainlink's
actual resolution.
"""
import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
VERIFICATION_DELAY_S = 60   # wait 60s after expiry for resolution to propagate
MAX_RETRIES = 3
RETRY_DELAY_S = 30


@dataclass
class VerificationRecord:
    """A single settlement verification result."""
    slug: str
    asset: str
    paper_settlement: str       # "Up" or "Down" — what WE said
    actual_settlement: str      # "Up" or "Down" — what Polymarket says (or "Unknown")
    match: bool                 # paper == actual
    paper_cex_price: float
    paper_strike: float
    actual_up_price: float      # Polymarket's final outcomePrices[0]
    actual_down_price: float    # Polymarket's final outcomePrices[1]
    timestamp: float
    strategy: str = ""
    confidence: float = 0.0     # v10 FIX #10: confidence at signal time


@dataclass
class VerificationStats:
    """Cumulative settlement verification statistics."""
    total_verified: int = 0
    matches: int = 0
    mismatches: int = 0
    unresolved: int = 0         # Polymarket didn't resolve yet

    # Per-asset breakdown
    per_asset: dict = field(default_factory=lambda: {
        "btc": {"verified": 0, "matches": 0, "mismatches": 0},
        "eth": {"verified": 0, "matches": 0, "mismatches": 0},
    })

    # v10 FIX #10: Confidence calibration bins
    # Key = bin label (e.g. "0.55-0.65"), value = {"total": N, "correct": M}
    calibration_bins: dict = field(default_factory=lambda: {
        "0.50-0.60": {"total": 0, "correct": 0},
        "0.60-0.70": {"total": 0, "correct": 0},
        "0.70-0.80": {"total": 0, "correct": 0},
        "0.80-0.90": {"total": 0, "correct": 0},
        "0.90-1.00": {"total": 0, "correct": 0},
    })

    @property
    def accuracy(self) -> float:
        if self.total_verified == 0:
            return 0.0
        return self.matches / self.total_verified

    @property
    def mismatch_rate(self) -> float:
        if self.total_verified == 0:
            return 0.0
        return self.mismatches / self.total_verified


class SettlementVerifier:
    """Asynchronous settlement verifier that checks Polymarket actual outcomes."""

    def __init__(self, db_path: str = "data/paper_trades.db"):
        self._client = httpx.AsyncClient(timeout=10, follow_redirects=True)
        self._stats = VerificationStats()
        self._records: list[VerificationRecord] = []
        self._pending_tasks: set[asyncio.Task] = set()
        # v10 FIX #13: SQLite persistence
        self._db_path = db_path
        self._init_db()
        self._load_from_db()

    def _init_db(self):
        """v10 FIX #13: Create verification table if not exists."""
        try:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settlement_verifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT, asset TEXT, strategy TEXT,
                    paper_settlement TEXT, actual_settlement TEXT,
                    match BOOLEAN,
                    paper_cex REAL, paper_strike REAL,
                    actual_up_price REAL, actual_down_price REAL,
                    confidence REAL,
                    timestamp REAL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Verifier DB init failed: {e}")

    def _load_from_db(self):
        """v10 FIX #13: Reload historical verification stats from DB."""
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                "SELECT asset, paper_settlement, actual_settlement, match, confidence "
                "FROM settlement_verifications WHERE actual_settlement != 'Unknown'"
            ).fetchall()
            conn.close()
            for asset, paper, actual, matched, conf in rows:
                self._stats.total_verified += 1
                asset_stats = self._stats.per_asset.setdefault(
                    asset, {"verified": 0, "matches": 0, "mismatches": 0}
                )
                asset_stats["verified"] += 1
                if matched:
                    self._stats.matches += 1
                    asset_stats["matches"] += 1
                else:
                    self._stats.mismatches += 1
                    asset_stats["mismatches"] += 1
                # Calibration
                if conf and conf > 0:
                    self._record_calibration(conf, bool(matched))
            if rows:
                logger.info(
                    f"Loaded {len(rows)} historical verifications from DB: "
                    f"{self._stats.matches}/{self._stats.total_verified} correct ({self._stats.accuracy:.1%})"
                )
        except Exception as e:
            logger.debug(f"Verifier DB load failed (first run?): {e}")

    def _persist_record(self, record: VerificationRecord):
        """v10 FIX #13: Write a verification record to SQLite."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO settlement_verifications "
                "(slug, asset, strategy, paper_settlement, actual_settlement, "
                "match, paper_cex, paper_strike, actual_up_price, actual_down_price, "
                "confidence, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.slug, record.asset, record.strategy,
                 record.paper_settlement, record.actual_settlement,
                 record.match, record.paper_cex_price, record.paper_strike,
                 record.actual_up_price, record.actual_down_price,
                 record.confidence, record.timestamp)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Verifier DB write failed: {e}")

    def _record_calibration(self, confidence: float, correct: bool):
        """v10 FIX #10: Record into calibration bins."""
        bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]
        for lo, hi in bins:
            if lo <= confidence < hi or (hi == 1.00 and confidence >= 0.90):
                key = f"{lo:.2f}-{hi:.2f}"
                bin_data = self._stats.calibration_bins.setdefault(
                    key, {"total": 0, "correct": 0}
                )
                bin_data["total"] += 1
                if correct:
                    bin_data["correct"] += 1
                break

    @property
    def stats(self) -> VerificationStats:
        return self._stats

    @property
    def records(self) -> list[VerificationRecord]:
        return self._records

    def schedule_verification(
        self,
        slug: str,
        asset: str,
        paper_settlement: str,
        paper_cex_price: float,
        paper_strike: float,
        strategy: str = "",
        confidence: float = 0.0,  # v10 FIX #10
    ):
        """Schedule a delayed verification check after round ends."""
        task = asyncio.create_task(
            self._verify_after_delay(
                slug=slug,
                asset=asset,
                paper_settlement=paper_settlement,
                paper_cex_price=paper_cex_price,
                paper_strike=paper_strike,
                strategy=strategy,
                confidence=confidence,
            )
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        logger.debug(f"Scheduled verification for {slug} in {VERIFICATION_DELAY_S}s")

    async def _verify_after_delay(
        self,
        slug: str,
        asset: str,
        paper_settlement: str,
        paper_cex_price: float,
        paper_strike: float,
        strategy: str,
        confidence: float = 0.0,
    ):
        """Wait, then fetch and compare actual settlement."""
        await asyncio.sleep(VERIFICATION_DELAY_S)

        actual_settlement = "Unknown"
        actual_up = 0.0
        actual_down = 0.0

        for attempt in range(MAX_RETRIES):
            try:
                actual_settlement, actual_up, actual_down = await self._fetch_actual_outcome(slug)
                if actual_settlement != "Unknown":
                    break
            except Exception as e:
                logger.warning(f"Verification fetch failed for {slug} (attempt {attempt+1}): {e}")

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY_S)

        # Record result
        match = (actual_settlement == paper_settlement) if actual_settlement != "Unknown" else False
        record = VerificationRecord(
            slug=slug,
            asset=asset,
            paper_settlement=paper_settlement,
            actual_settlement=actual_settlement,
            match=match,
            paper_cex_price=paper_cex_price,
            paper_strike=paper_strike,
            actual_up_price=actual_up,
            actual_down_price=actual_down,
            timestamp=time.time(),
            strategy=strategy,
            confidence=confidence,
        )
        self._records.append(record)
        # v10 FIX #13: Persist to SQLite
        self._persist_record(record)

        # Update stats
        if actual_settlement == "Unknown":
            self._stats.unresolved += 1
            logger.warning(f"🔍 UNRESOLVED: {slug} — Polymarket didn't resolve after {MAX_RETRIES} retries")
        else:
            self._stats.total_verified += 1
            asset_stats = self._stats.per_asset.setdefault(
                asset, {"verified": 0, "matches": 0, "mismatches": 0}
            )
            asset_stats["verified"] += 1

            if match:
                self._stats.matches += 1
                asset_stats["matches"] += 1
                logger.info(
                    f"✅ VERIFIED: {slug} | paper={paper_settlement} actual={actual_settlement} | "
                    f"CEX=${paper_cex_price:.2f} strike=${paper_strike:.2f} | conf={confidence:.1%} | "
                    f"PM prices: Up={actual_up:.3f} Down={actual_down:.3f}"
                )
            else:
                self._stats.mismatches += 1
                asset_stats["mismatches"] += 1
                logger.warning(
                    f"❌ MISMATCH: {slug} | paper={paper_settlement} actual={actual_settlement} | "
                    f"CEX=${paper_cex_price:.2f} strike=${paper_strike:.2f} | conf={confidence:.1%} | "
                    f"PM prices: Up={actual_up:.3f} Down={actual_down:.3f}"
                )

            # v10 FIX #10: Record calibration data
            if confidence > 0:
                self._record_calibration(confidence, match)

            # Periodic summary
            s = self._stats
            if s.total_verified % 10 == 0:
                logger.info(
                    f"🔍 Verification summary: {s.matches}/{s.total_verified} correct "
                    f"({s.accuracy:.1%}) | mismatches={s.mismatches} | unresolved={s.unresolved}"
                )

    async def _fetch_actual_outcome(self, slug: str) -> tuple[str, float, float]:
        """Fetch the actual settlement from Polymarket Gamma API.
        
        Returns: (settlement_direction, up_price, down_price)
        
        After resolution:
        - Up wins:   outcomePrices = ["1", "0"] or prices close to 1.0/0.0
        - Down wins:  outcomePrices = ["0", "1"] 
        - Not resolved yet: prices still near 0.5/0.5
        """
        url = f"{GAMMA_API}/events"
        params = {"slug": slug}

        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or len(data) == 0:
            return "Unknown", 0.0, 0.0

        event = data[0]
        markets = event.get("markets", [])
        if not markets:
            return "Unknown", 0.0, 0.0

        market = markets[0]
        outcome_prices = market.get("outcomePrices", "")
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, TypeError):
                return "Unknown", 0.0, 0.0

        if len(outcome_prices) < 2:
            return "Unknown", 0.0, 0.0

        up_price = float(outcome_prices[0])
        down_price = float(outcome_prices[1])

        # Determine resolution: resolved markets have one price ≥ 0.95
        if up_price >= 0.95:
            return "Up", up_price, down_price
        elif down_price >= 0.95:
            return "Down", up_price, down_price
        else:
            # Not yet resolved — prices still near 0.5
            return "Unknown", up_price, down_price

    def get_summary(self) -> str:
        """Get a formatted summary string."""
        s = self._stats
        lines = [
            f"Settlement Verification: {s.matches}/{s.total_verified} correct ({s.accuracy:.1%})",
            f"  Mismatches: {s.mismatches} | Unresolved: {s.unresolved}",
        ]
        for asset, data in s.per_asset.items():
            if data["verified"] > 0:
                acc = data["matches"] / data["verified"] * 100
                lines.append(
                    f"  {asset.upper()}: {data['matches']}/{data['verified']} ({acc:.0f}%) "
                    f"| mismatches={data['mismatches']}"
                )
        # v10 FIX #10: Calibration summary
        if any(v["total"] > 0 for v in s.calibration_bins.values()):
            lines.append("  Confidence Calibration:")
            for bin_label, data in sorted(s.calibration_bins.items()):
                if data["total"] > 0:
                    actual_wr = data["correct"] / data["total"] * 100
                    lines.append(f"    {bin_label}: {data['correct']}/{data['total']} ({actual_wr:.0f}% actual WR)")
        return "\n".join(lines)

    async def close(self):
        """Cancel pending tasks and close HTTP client."""
        for task in self._pending_tasks:
            task.cancel()
        await self._client.aclose()
