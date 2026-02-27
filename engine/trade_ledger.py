"""
Trade Ledger — SQLite-backed persistent recording of all paper/live trades.

Tables:
  rounds  — one row per 5-minute round (start, end, pnl, outcome)
  orders  — one row per order/signal (fill, fee, slippage)
  daily   — one row per day, aggregated stats

Usage:
  ledger = TradeLedger("/opt/polymarket/data/paper_trades.db")
  ledger.record_order(strategy, signal, result)
  ledger.record_round(strategy, market, settled, pnl)
  stats = ledger.get_stats(days=7)
"""
import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class TradeLedger:
    """Persistent trade recorder backed by SQLite."""

    def __init__(self, db_path: str = "data/paper_trades.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"TradeLedger opened: {db_path}")

    def _create_tables(self):
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS rounds (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL NOT NULL,
            strategy    TEXT NOT NULL,
            asset       TEXT NOT NULL,
            slug        TEXT NOT NULL,
            strike      REAL DEFAULT 0,
            cex_price   REAL DEFAULT 0,
            predicted   TEXT DEFAULT '',
            settled     TEXT DEFAULT '',
            correct     INTEGER DEFAULT 0,
            pnl         REAL DEFAULT 0,
            cost        REAL DEFAULT 0,
            signals     INTEGER DEFAULT 0,
            fills       INTEGER DEFAULT 0,
            duration_s  REAL DEFAULT 300
        );

        CREATE TABLE IF NOT EXISTS orders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL NOT NULL,
            strategy    TEXT NOT NULL,
            asset       TEXT NOT NULL,
            slug        TEXT NOT NULL,
            side        TEXT NOT NULL,
            outcome     TEXT NOT NULL,
            order_type  TEXT NOT NULL,
            price       REAL NOT NULL,
            size_usd    REAL NOT NULL,
            confidence  REAL DEFAULT 0,
            reason      TEXT DEFAULT '',
            filled      INTEGER DEFAULT 0,
            fill_price  REAL DEFAULT 0,
            fill_size   REAL DEFAULT 0,
            fee_usd     REAL DEFAULT 0,
            slippage    REAL DEFAULT 0,
            latency_ms  REAL DEFAULT 0,
            reject_reason TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS daily (
            date        TEXT PRIMARY KEY,
            rounds      INTEGER DEFAULT 0,
            wins        INTEGER DEFAULT 0,
            losses      INTEGER DEFAULT 0,
            signals     INTEGER DEFAULT 0,
            fills       INTEGER DEFAULT 0,
            pnl         REAL DEFAULT 0,
            volume      REAL DEFAULT 0,
            fees        REAL DEFAULT 0,
            max_drawdown REAL DEFAULT 0,
            strategies  TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_rounds_ts ON rounds(ts);
        CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts);
        CREATE INDEX IF NOT EXISTS idx_rounds_strategy ON rounds(strategy);
        """)
        self._db.commit()

    def record_order(self, strategy: str, signal, result) -> int:
        """Record an order attempt (filled or rejected).
        
        Args:
            strategy: strategy name (e.g. 'CryptoMM', 'OracleArb')
            signal: Signal object from strategy
            result: SimulationResult from paper_simulator
        
        Returns:
            row id
        """
        cur = self._db.execute("""
        INSERT INTO orders (ts, strategy, asset, slug, side, outcome, order_type,
                           price, size_usd, confidence, reason,
                           filled, fill_price, fill_size, fee_usd,
                           slippage, latency_ms, reject_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            strategy,
            signal.market.asset,
            signal.market.slug,
            signal.side.value,
            signal.outcome.value,
            signal.order_type,
            signal.price,
            signal.size_usd,
            signal.confidence,
            signal.reason,
            1 if result.filled else 0,
            result.fill_price,
            result.fill_size,
            result.fee_usd,
            result.slippage_ticks,
            result.latency_ms,
            result.reject_reason,
        ))
        self._db.commit()
        return cur.lastrowid

    def record_round(self, strategy: str, market, settled: str,
                     predicted: str, pnl: float, cost: float,
                     signals: int, fills: int, cex_price: float = 0):
        """Record the end of a 5-minute round."""
        correct = 1 if predicted == settled and predicted != "" else 0

        self._db.execute("""
        INSERT INTO rounds (ts, strategy, asset, slug, strike, cex_price,
                           predicted, settled, correct, pnl, cost,
                           signals, fills, duration_s)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            strategy,
            market.asset,
            market.slug,
            market.strike_price,
            cex_price,
            predicted,
            settled,
            correct,
            pnl,
            cost,
            signals,
            fills,
            market.duration_seconds,
        ))
        self._db.commit()

        # Update daily aggregation
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._update_daily(today, correct, pnl, cost, signals, fills, strategy)

    def _update_daily(self, date: str, correct: int, pnl: float,
                      cost: float, signals: int, fills: int, strategy: str):
        """Update daily stats (upsert)."""
        row = self._db.execute("SELECT * FROM daily WHERE date = ?", (date,)).fetchone()
        if row:
            strategies = set(row["strategies"].split(",")) if row["strategies"] else set()
            strategies.add(strategy)
            self._db.execute("""
            UPDATE daily SET
                rounds = rounds + 1,
                wins = wins + ?,
                losses = losses + ?,
                signals = signals + ?,
                fills = fills + ?,
                pnl = pnl + ?,
                volume = volume + ?,
                fees = fees + (SELECT COALESCE(SUM(fee_usd), 0) FROM orders 
                              WHERE ts > ? AND ts < ?),
                strategies = ?
            WHERE date = ?
            """, (
                correct,
                1 - correct,
                signals,
                fills,
                pnl,
                cost,
                time.time() - 300, time.time(),
                ",".join(strategies),
                date,
            ))
        else:
            self._db.execute("""
            INSERT INTO daily (date, rounds, wins, losses, signals, fills, pnl, volume, strategies)
            VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
            """, (date, correct, 1 - correct, signals, fills, pnl, cost, strategy))
        self._db.commit()

    def get_stats(self, days: int = 7) -> dict:
        """Get aggregated stats for the last N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        daily_rows = self._db.execute(
            "SELECT * FROM daily WHERE date >= ? ORDER BY date", (cutoff,)
        ).fetchall()

        total_rounds = sum(r["rounds"] for r in daily_rows)
        total_wins = sum(r["wins"] for r in daily_rows)
        total_pnl = sum(r["pnl"] for r in daily_rows)
        total_signals = sum(r["signals"] for r in daily_rows)
        total_fills = sum(r["fills"] for r in daily_rows)
        total_volume = sum(r["volume"] for r in daily_rows)

        # Per-strategy breakdown
        strategy_stats = {}
        for row in self._db.execute("""
            SELECT strategy, COUNT(*) as rounds, SUM(correct) as wins,
                   SUM(pnl) as pnl, SUM(signals) as signals, SUM(fills) as fills
            FROM rounds WHERE ts > ?
            GROUP BY strategy
        """, (time.time() - days * 86400,)).fetchall():
            strategy_stats[row["strategy"]] = {
                "rounds": row["rounds"],
                "wins": row["wins"] or 0,
                "win_rate": f"{(row['wins'] or 0) / max(row['rounds'], 1):.1%}",
                "pnl": round(row["pnl"] or 0, 2),
                "signals": row["signals"] or 0,
                "fills": row["fills"] or 0,
            }

        # Recent rounds
        recent = []
        for row in self._db.execute("""
            SELECT strategy, asset, predicted, settled, correct, pnl, 
                   strike, cex_price, ts
            FROM rounds ORDER BY ts DESC LIMIT 20
        """).fetchall():
            recent.append({
                "time": datetime.fromtimestamp(row["ts"], tz=timezone.utc).strftime("%H:%M"),
                "strategy": row["strategy"],
                "asset": row["asset"].upper(),
                "predicted": row["predicted"],
                "settled": row["settled"],
                "correct": "✅" if row["correct"] else "❌",
                "pnl": f"${row['pnl']:+.2f}",
            })

        return {
            "period": f"Last {days} days",
            "total_rounds": total_rounds,
            "total_wins": total_wins,
            "win_rate": f"{total_wins / max(total_rounds, 1):.1%}",
            "total_pnl": f"${total_pnl:+.2f}",
            "total_signals": total_signals,
            "total_fills": total_fills,
            "fill_rate": f"{total_fills / max(total_signals, 1):.1%}",
            "total_volume": f"${total_volume:,.2f}",
            "strategies": strategy_stats,
            "daily": [{
                "date": r["date"],
                "rounds": r["rounds"],
                "wins": r["wins"],
                "pnl": f"${r['pnl']:+.2f}",
            } for r in daily_rows],
            "recent_rounds": recent,
        }

    def close(self):
        self._db.close()
