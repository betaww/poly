"""
E8: Telegram Alert System for Polymarket trading.

Provides real-time alerts for critical system events:
  - Brain node disconnection
  - Consecutive coin-flip settlements (price source failure)
  - Risk manager pauses
  - CLOB WebSocket disconnection
  - Fill rate anomalies

Usage:
    alerter = AlertManager(config)
    await alerter.check_and_alert(state)
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class AlertState:
    """Tracks alert state to prevent spam."""
    last_brain_alert: float = 0.0
    last_clob_alert: float = 0.0
    last_risk_alert: float = 0.0
    consecutive_coinflips: int = 0
    total_rounds: int = 0
    total_fills: int = 0
    # Cooldown between repeated alerts (seconds)
    cooldown: float = 300.0  # 5 minutes


class AlertManager:
    """
    E8: Manages Telegram alerts for critical trading events.

    Integrates with vps_runner to monitor system health and
    send time-critical notifications.
    """

    def __init__(self, config: Config):
        self.config = config
        self._alert_config = config.alert
        self._state = AlertState()
        self._enabled = config.alert.enabled
        self._client: Optional[httpx.AsyncClient] = None

        if self._enabled:
            logger.info("Telegram alerting enabled")
        else:
            logger.info("Telegram alerting disabled (no token/chat_id)")

    async def _send_telegram(self, message: str, level: str = "warning"):
        """Send a message via Telegram Bot API."""
        if not self._enabled:
            return

        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨", "success": "✅"}.get(level, "📢")
        text = f"{emoji} *Polymarket {level.upper()}*\n\n{message}"

        try:
            if not self._client:
                self._client = httpx.AsyncClient(timeout=10.0)
            url = f"https://api.telegram.org/bot{self._alert_config.telegram_bot_token}/sendMessage"
            await self._client.post(url, json={
                "chat_id": self._alert_config.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown",
            })
            logger.debug(f"Telegram alert sent: {message[:80]}...")
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")

    # ─── Alert Checks ─────────────────────────────────────────────────────

    async def check_brain_offline(self, is_connected: bool, offline_since: float):
        """Alert when Brain node has been offline too long."""
        now = time.time()
        threshold = self._alert_config.brain_offline_seconds

        if not is_connected and offline_since > 0:
            duration = now - offline_since
            if (duration > threshold and
                    now - self._state.last_brain_alert > self._state.cooldown):
                await self._send_telegram(
                    f"Brain node offline for *{duration:.0f}s* "
                    f"(threshold: {threshold}s)\n"
                    f"Settlement will use HTTP fallback → coin flip",
                    level="critical",
                )
                self._state.last_brain_alert = now

    async def check_coinflip_settlement(self, was_coinflip: bool):
        """Alert on consecutive coin-flip settlements."""
        if was_coinflip:
            self._state.consecutive_coinflips += 1
            threshold = self._alert_config.consecutive_coinflip_alert
            if self._state.consecutive_coinflips >= threshold:
                await self._send_telegram(
                    f"*{self._state.consecutive_coinflips}* consecutive coin-flip settlements!\n"
                    f"All price sources failed. Check:\n"
                    f"• Brain node connection\n"
                    f"• Exchange API access\n"
                    f"• HTTP fallback endpoints",
                    level="critical",
                )
        else:
            self._state.consecutive_coinflips = 0

    async def check_risk_pause(self, is_paused: bool, reason: str):
        """Alert when risk manager triggers a pause."""
        now = time.time()
        if is_paused and now - self._state.last_risk_alert > self._state.cooldown:
            await self._send_telegram(
                f"Risk manager *PAUSED* trading\n"
                f"Reason: {reason}",
                level="warning",
            )
            self._state.last_risk_alert = now

    async def check_clob_disconnect(self, ws_connected: bool, disconnect_since: float):
        """Alert when CLOB WebSocket has been disconnected too long."""
        now = time.time()
        threshold = self._alert_config.clob_disconnect_seconds

        if not ws_connected and disconnect_since > 0:
            duration = now - disconnect_since
            if (duration > threshold and
                    now - self._state.last_clob_alert > self._state.cooldown):
                await self._send_telegram(
                    f"CLOB WebSocket disconnected for *{duration:.0f}s*\n"
                    f"Midpoint data is stale. Paper sim accuracy degraded.",
                    level="warning",
                )
                self._state.last_clob_alert = now

    async def check_fill_rate(self, total_rounds: int, total_fills: int):
        """Alert if fill rate is anomalous."""
        if total_rounds < 20:  # need minimum sample size
            return

        fill_rate = total_fills / max(1, total_rounds)
        if fill_rate < self._alert_config.fill_rate_min:
            await self._send_telegram(
                f"Fill rate *{fill_rate:.1%}* is below threshold "
                f"({self._alert_config.fill_rate_min:.0%})\n"
                f"Rounds: {total_rounds}, Fills: {total_fills}\n"
                f"Check: maker price, CLOB depth, strategy parameters",
                level="warning",
            )
        elif fill_rate > self._alert_config.fill_rate_max:
            await self._send_telegram(
                f"Fill rate *{fill_rate:.1%}* is suspiciously high "
                f"({self._alert_config.fill_rate_max:.0%} max)\n"
                f"Paper sim might be too optimistic.",
                level="warning",
            )

    async def send_daily_summary(self, stats: dict):
        """Send end-of-day performance summary."""
        await self._send_telegram(
            f"📊 *Daily Summary*\n"
            f"PnL: ${stats.get('daily_pnl', 0):+.2f}\n"
            f"Rounds: {stats.get('rounds', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.1%}\n"
            f"Volume: ${stats.get('volume', 0):.0f}\n"
            f"Bankroll: ${stats.get('bankroll', 0):.0f}",
            level="info",
        )

    async def close(self):
        """Clean up HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
