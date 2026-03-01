"""
Market Scanner — auto-discovers 5-minute crypto markets on Polymarket.

Uses deterministic slug calculation: {asset}-updown-{tf}-{unix_timestamp}
Verified working pattern: events?slug=btc-updown-5m-{ts}
"""
import time
import re
import json
import math
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from .config import Config, MarketConfig

logger = logging.getLogger(__name__)


@dataclass
class MarketInfo:
    """Discovered market data."""
    asset: str                    # "btc", "eth", etc.
    event_id: str                 # Polymarket event ID
    market_id: str                # Polymarket market ID
    condition_id: str             # On-chain condition ID
    slug: str                     # Full slug
    question: str                 # Market question text

    # CLOB token IDs for trading
    token_id_up: str              # Token ID for "Up" outcome
    token_id_down: str            # Token ID for "Down" outcome

    # Current prices
    price_up: float = 0.5
    price_down: float = 0.5

    # Timing
    start_time: int = 0           # Unix timestamp
    end_time: int = 0             # Unix timestamp
    duration_seconds: int = 300   # 5 minutes

    # Market parameters
    tick_size: str = "0.01"
    min_order_size: float = 5.0
    neg_risk: bool = False
    # Polymarket dynamic fee: fee = C × feeRate × (p(1-p))^exponent
    # For 5m/15m crypto: feeRate=0.25, exponent=2, max ~1.56% at p=0.50
    # Maker orders: 0% fee + rebate
    fee_rate: float = 0.25       # Polymarket feeRate parameter
    fee_exponent: int = 2        # Polymarket exponent parameter

    # Strike price parsed from question (e.g. "BTC above $84,500.00" → 84500.0)
    strike_price: float = 0.0

    @property
    def seconds_remaining(self) -> float:
        return max(0, self.end_time - time.time())

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.end_time

    @property
    def midpoint(self) -> float:
        """Midpoint of the Up outcome token.

        CRIT1 FIX: On Polymarket, price_up + price_down ≈ $1.00 always,
        so (price_up + price_down)/2 ≈ 0.50 which is useless.
        The correct midpoint for strategy pricing is the current market
        consensus probability for Up (i.e. price_up itself).
        """
        return self.price_up


class MarketScanner:
    """
    Discovers active 5-minute crypto markets using deterministic slug calculation.

    The slug pattern is: {asset}-updown-{tf}-{unix_timestamp}
    where unix_timestamp = the END time of the current window, rounded to the interval.
    """

    def __init__(self, config: Config):
        self.config = config
        self.gamma_host = config.api.gamma_host
        self._client = httpx.AsyncClient(timeout=10.0)
        self._active_markets: dict[str, MarketInfo] = {}

    async def close(self):
        await self._client.aclose()

    def _calculate_window_timestamps(self, asset: str) -> list[int]:
        """
        Calculate the current and next window end timestamps.
        
        Windows are aligned to unix epoch:
          - 5m windows end at multiples of 300
          - 15m windows end at multiples of 900
          - etc.
        """
        now = time.time()
        tf = self.config.market.timeframe
        duration = self.config.market.DURATIONS[tf]

        # Current window end = next multiple of duration
        current_end = int(math.ceil(now / duration) * duration)
        # Next window end
        next_end = current_end + duration
        # Previous window (might still be settling)
        prev_end = current_end - duration

        return [prev_end, current_end, next_end]  # M9 FIX: include prev window

    @staticmethod
    def _parse_strike(question: str, asset: str = "") -> float:
        """Parse strike price from market question text with multi-pattern fallback.
        
        Examples:
          "Will BTC be above $84,500.00 at 2:30 PM?" → 84500.0
          "Will ETH be above $2,250.50 at 3:00 PM?"  → 2250.5
          "BTC above 84500 at 14:30"                  → 84500.0
        """
        # Pattern 1: $XX,XXX.XX (standard format)
        match = re.search(r'\$([0-9,]+(?:\.[0-9]+)?)', question)
        if match:
            strike = float(match.group(1).replace(',', ''))
        else:
            # Pattern 2: naked number after "above" or "below"
            match = re.search(r'(?:above|below)\s+([0-9,]+(?:\.[0-9]+)?)', question, re.IGNORECASE)
            if match:
                strike = float(match.group(1).replace(',', ''))
            else:
                # Pattern 3: any large number in the string (last resort)
                numbers = re.findall(r'[0-9,]+\.?[0-9]*', question)
                candidates = [float(n.replace(',', '')) for n in numbers if float(n.replace(',', '')) > 100]
                if candidates:
                    strike = max(candidates)  # largest number is likely the strike
                    logger.warning(f"Strike parsed via fallback (largest number): ${strike:,.2f} from '{question}'")
                else:
                    logger.warning(f"❌ Strike parse FAILED: '{question}'")
                    return 0.0

        # Sanity check: strike must be in realistic range for the asset
        SANITY = {
            "btc": (10_000, 500_000),
            "eth": (100, 50_000),
            "sol": (1, 5_000),
        }
        lo, hi = SANITY.get(asset.lower(), (0, 1_000_000))
        if not (lo <= strike <= hi):
            logger.warning(
                f"⚠️ Strike ${strike:,.2f} out of range [{lo:,}-{hi:,}] for {asset.upper()}, "
                f"using anyway but may be wrong. Question: '{question}'"
            )

        return strike

    def _make_slug(self, asset: str, window_end: int) -> str:
        """Build deterministic slug."""
        tf = self.config.market.timeframe
        pattern = self.config.market.SLUG_PATTERNS[tf]
        return pattern.format(asset=asset, ts=window_end)

    async def _fetch_event_by_slug(self, slug: str) -> Optional[dict]:
        """Query Gamma API with exact slug (verified working pattern)."""
        url = f"{self.gamma_host}/events"
        params = {"slug": slug}
        try:
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch event for slug {slug}: {e}")
            return None

    def _parse_event(self, event: dict, asset: str) -> Optional[MarketInfo]:
        """Parse Gamma API event response into MarketInfo."""
        try:
            markets = event.get("markets", [])
            if not markets:
                logger.warning(f"No markets in event {event.get('id')}")
                return None

            market = markets[0]  # Binary market has one market entry

            # Extract CLOB token IDs from clobTokenIds JSON string
            clob_token_ids = market.get("clobTokenIds", "")
            if isinstance(clob_token_ids, str):
                clob_token_ids = json.loads(clob_token_ids)

            # Extract outcome prices
            outcome_prices = market.get("outcomePrices", "")
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)

            # Parse timestamps
            end_date = market.get("endDate", "")
            start_date = market.get("startDate", "")

            # Convert ISO dates to unix timestamps
            from datetime import datetime, timezone
            end_ts = 0
            start_ts = 0
            if end_date:
                try:
                    dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    end_ts = int(dt.timestamp())
                except (ValueError, TypeError):
                    pass
            if start_date:
                try:
                    dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                    start_ts = int(dt.timestamp())
                except (ValueError, TypeError):
                    pass

            # Fee info — 5m/15m crypto markets use dynamic fee formula
            # fee = C × feeRate × (p(1-p))^exponent
            # feeRate=0.25, exponent=2 for crypto markets
            # API may return feeRateBps but we use the dynamic formula instead

            return MarketInfo(
                asset=asset,
                event_id=str(event.get("id", "")),
                market_id=str(market.get("id", "")),
                condition_id=market.get("conditionId", ""),
                slug=event.get("slug", ""),
                question=market.get("question", ""),
                token_id_up=clob_token_ids[0] if len(clob_token_ids) > 0 else "",
                token_id_down=clob_token_ids[1] if len(clob_token_ids) > 1 else "",
                price_up=float(outcome_prices[0]) if len(outcome_prices) > 0 else 0.5,
                price_down=float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5,
                start_time=start_ts,
                end_time=end_ts,
                duration_seconds=self.config.market.DURATIONS[self.config.market.timeframe],
                tick_size=market.get("minimumTickSize", "0.01"),
                min_order_size=float(market.get("minimumOrderSize", 5)),
                neg_risk=market.get("negRisk", False),
                fee_rate=0.25,       # fixed for 5m/15m crypto
                fee_exponent=2,      # fixed for 5m/15m crypto
                strike_price=self._parse_strike(market.get("question", ""), asset),
            )

        except Exception as e:
            logger.error(f"Failed to parse event: {e}", exc_info=True)
            return None

    async def discover_current_markets(self) -> list[MarketInfo]:
        """
        Discover all current 5-minute markets for configured assets.
        Returns list of active (non-expired) MarketInfo objects.
        """
        discovered = []

        for asset in self.config.market.assets:
            timestamps = self._calculate_window_timestamps(asset)

            for ts in timestamps:
                slug = self._make_slug(asset, ts)

                # Skip if already cached and not expired
                if slug in self._active_markets:
                    market = self._active_markets[slug]
                    if not market.is_expired:
                        discovered.append(market)
                        continue

                # Fetch from API
                event = await self._fetch_event_by_slug(slug)
                if event:
                    market_info = self._parse_event(event, asset)
                    if market_info and not market_info.is_expired:
                        self._active_markets[slug] = market_info
                        discovered.append(market_info)
                        logger.info(
                            f"Discovered: {asset.upper()} {self.config.market.timeframe} | "
                            f"market_id={market_info.market_id} | "
                            f"Up={market_info.price_up:.3f} Down={market_info.price_down:.3f} | "
                            f"Remaining={market_info.seconds_remaining:.0f}s"
                        )
                    else:
                        logger.debug(f"Slug {slug}: expired or unparseable")
                else:
                    logger.debug(f"Slug {slug}: not found (market may not exist yet)")

        # Cleanup expired entries
        expired = [k for k, v in self._active_markets.items() if v.is_expired]
        for k in expired:
            del self._active_markets[k]

        return discovered

    async def get_best_market(self, asset: str) -> Optional[MarketInfo]:
        """Get the currently active market for a specific asset with the most time remaining."""
        markets = await self.discover_current_markets()
        asset_markets = [m for m in markets if m.asset == asset and not m.is_expired]
        if not asset_markets:
            return None
        # Return the one with the most time remaining
        return max(asset_markets, key=lambda m: m.seconds_remaining)

    async def watch_markets(self, callback, interval: float = 5.0):
        """
        Continuously monitor markets, calling callback with new/updated markets.
        Automatically transitions to next window when current one expires.
        """
        logger.info(f"Starting market watcher for {self.config.market.assets} @ {self.config.market.timeframe}")

        while True:
            try:
                markets = await self.discover_current_markets()
                if markets:
                    await callback(markets)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market watcher error: {e}", exc_info=True)
                await asyncio.sleep(interval)


# --- CLI test ---
async def _test():
    """Quick test: discover current BTC/ETH 5-minute markets."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = Config()
    scanner = MarketScanner(config)

    print("\n=== Scanning for active 5-minute crypto markets ===\n")
    markets = await scanner.discover_current_markets()

    if not markets:
        print("No active markets found. This is normal if between windows.")
    else:
        for m in markets:
            print(f"  {m.asset.upper()} | ID={m.market_id} | Up={m.price_up:.3f} Down={m.price_down:.3f} | "
                  f"Remaining={m.seconds_remaining:.0f}s | slug={m.slug}")
            print(f"    token_up={m.token_id_up[:20]}... token_down={m.token_id_down[:20]}...")
            print()

    await scanner.close()


if __name__ == "__main__":
    asyncio.run(_test())
