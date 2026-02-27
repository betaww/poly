"""
Performance Report CLI — queries the trade ledger and prints formatted stats.

Usage:
  python -m polymarket.engine.report --days 7
  python -m polymarket.engine.report --days 1 --strategy oracle_arb

Can be run from VPS via SSH:
  ssh root@192.248.167.26 "cd /opt && source polymarket/venv/bin/activate && \
    python -m polymarket.engine.report --days 1"
"""
import argparse
import json
import sys
import os

sys.path.insert(0, "/opt")

from polymarket.engine.trade_ledger import TradeLedger


def print_report(stats: dict):
    """Print a nicely formatted performance report."""
    print()
    print("=" * 60)
    print(f"  Polymarket Paper Trading Report — {stats['period']}")
    print("=" * 60)
    print()

    print(f"  Rounds:    {stats['total_rounds']}")
    print(f"  Wins:      {stats['total_wins']} ({stats['win_rate']})")
    print(f"  Total P&L: {stats['total_pnl']}")
    print(f"  Signals:   {stats['total_signals']}")
    print(f"  Fills:     {stats['total_fills']} ({stats['fill_rate']})")
    print(f"  Volume:    {stats['total_volume']}")
    print()

    # Per-strategy breakdown
    if stats["strategies"]:
        print("─── Per Strategy ───")
        for name, s in stats["strategies"].items():
            print(f"  {name:15s}  Rounds={s['rounds']:3d}  "
                  f"Win={s['win_rate']:>5s}  "
                  f"P&L={s['pnl']:>+8.2f}  "
                  f"Signals={s['signals']:3d}  "
                  f"Fills={s['fills']:3d}")
        print()

    # Daily breakdown
    if stats["daily"]:
        print("─── Daily ───")
        for d in stats["daily"]:
            bar = "█" * max(1, int(abs(d["pnl"].replace("$", "").replace("+", "")) * 2))
            color = "+" if not d["pnl"].startswith("$-") else "-"
            print(f"  {d['date']}  Rounds={d['rounds']:3d}  "
                  f"Wins={d['wins']:3d}  P&L={d['pnl']:>8s}  {bar}")
        print()

    # Recent rounds
    if stats["recent_rounds"]:
        print("─── Last 20 Rounds ───")
        for r in stats["recent_rounds"]:
            print(f"  {r['time']}  {r['strategy']:12s}  {r['asset']:3s}  "
                  f"Pred={r['predicted']:4s}  Settled={r['settled']:4s}  "
                  f"{r['correct']}  {r['pnl']}")
        print()

    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="Polymarket Paper Trading Report")
    parser.add_argument("--days", type=int, default=7, help="Report period (days)")
    parser.add_argument("--db", default="/opt/polymarket/data/paper_trades.db",
                        help="Path to SQLite database")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        print("No trades recorded yet. Let the system run for a few rounds.")
        sys.exit(1)

    ledger = TradeLedger(args.db)
    stats = ledger.get_stats(days=args.days)
    ledger.close()

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_report(stats)


if __name__ == "__main__":
    main()
