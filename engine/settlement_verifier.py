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

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=10, follow_redirects=True)
        self._stats = VerificationStats()
        self._records: list[VerificationRecord] = []
        self._pending_tasks: set[asyncio.Task] = set()

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
    ):
        """Schedule a delayed verification check after round ends.
        
        Non-blocking: creates a background asyncio task that runs after
        VERIFICATION_DELAY_S seconds.
        """
        task = asyncio.create_task(
            self._verify_after_delay(
                slug=slug,
                asset=asset,
                paper_settlement=paper_settlement,
                paper_cex_price=paper_cex_price,
                paper_strike=paper_strike,
                strategy=strategy,
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
        )
        self._records.append(record)

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
                    f"CEX=${paper_cex_price:.2f} strike=${paper_strike:.2f} | "
                    f"PM prices: Up={actual_up:.3f} Down={actual_down:.3f}"
                )
            else:
                self._stats.mismatches += 1
                asset_stats["mismatches"] += 1
                logger.warning(
                    f"❌ MISMATCH: {slug} | paper={paper_settlement} actual={actual_settlement} | "
                    f"CEX=${paper_cex_price:.2f} strike=${paper_strike:.2f} | "
                    f"PM prices: Up={actual_up:.3f} Down={actual_down:.3f}"
                )

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
        return "\n".join(lines)

    async def close(self):
        """Cancel pending tasks and close HTTP client."""
        for task in self._pending_tasks:
            task.cancel()
        await self._client.aclose()
