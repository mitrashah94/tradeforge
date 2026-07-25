#!/usr/bin/env python3
"""scripts/paper_trade_replay.py — MULTI-SESSION REAL PAPER TRADING (Phase 1 / P5).

Generalizes the one-session ``scripts/paper_dry_run.py`` into a MANY-session
driver: it boots the P3 event core ONCE and replays every session's bars through
the IDENTICAL event path (reconcile-first boot -> fast loop -> order gateway ->
paper broker -> F2 lifecycle -> POSITION_CLOSED), accumulating real paper fills
in a PERSISTENT ledger and compounding equity trade-to-trade. This is how we get
from "empty ledgers" to "N real paper trades scored against the paper->live gate"
(MASTER_PLAN §8) — the data the gate needs that has never existed.

PAPER ONLY + OFFLINE: reads ``data/duckdb/market.duckdb`` via
``orchestrator.tools.market_data.ReplayFeed``; all order flow is the PaperBroker;
no live broker is constructed, no network, no LLM/MCP in the hot path. The
research firewall holds (nothing live is written/sent).

HONESTY: the PaperBroker fills at the signal close with ``slippage_bps`` (default
0) and does NOT model adverse stop gaps, so its PnL is OPTIMISTIC vs the realistic
backtest cost model — the realistic edge read remains the backtest's. The point
of this run is the WORKING PIPELINE + persistent ledger + analyst/journal scoring
+ paper-vs-backtest trade-count reconciliation, not a new edge verdict. Real
slippage captured here later recalibrates the cost model (the sim-to-real loop).

Run:
    PYTHONPATH=. .venv/bin/python scripts/paper_trade_replay.py
    PYTHONPATH=. .venv/bin/python scripts/paper_trade_replay.py --start 2024-07-01 --end 2026-06-12
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def enforce_firewall() -> None:
    """Pin safe defaults: file notify sink, no recipient, refuse the live surface."""
    os.environ["TRADEFORGE_NOTIFY_CHANNEL"] = "file"
    os.environ.pop("TRADEFORGE_NOTIFY_TO", None)
    if os.environ.get("TRADEFORGE_LIVE") == "1":
        raise SystemExit("FIREWALL: TRADEFORGE_LIVE=1 set; refusing paper replay.")


def _rm(path: str) -> None:
    for p in (path, path + ".wal"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    import duckdb

    from backtest.stats.metrics import max_drawdown  # equity-curve max DD (fraction)
    from data.schema import DEFAULT_DB_PATH
    from orchestrator.agents.performance_analyst import alpha_vs_spy, equity_curve
    from orchestrator.events import EventType
    from orchestrator.fast_loop import ArmedStrategy
    from orchestrator.main import boot, build_system
    from orchestrator.tools.market_data import ReplayFeed
    from risk.config import load_limits
    from risk.sizing import resolve_ri
    from strategies.breakout_retest.strategy import BreakoutRetestStrategy, load_params

    p = argparse.ArgumentParser(description="TradeForge multi-session paper replay (Phase 1)")
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--variant", default="v0_atr_stop", help="breakout_retest variant")
    p.add_argument("--grade", default="A", help="conviction grade (-> RI cap)")
    p.add_argument("--start", default=None, help="first session date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="last session date (YYYY-MM-DD)")
    p.add_argument("--equity", type=float, default=100_000.0,
                   help="paper starting equity (measurement basis; sizing-robust PF)")
    p.add_argument("--slippage-bps", type=float, default=0.0,
                   help="paper fill slippage in bps (0 = optimistic; disclose)")
    p.add_argument("--market-db", default=DEFAULT_DB_PATH)
    p.add_argument("--paper-ledger-db", default="paper/ledger.duckdb")
    p.add_argument("--events-db", default="paper/replay_events.duckdb")
    p.add_argument("--orderbook-db", default="paper/replay_orderbook.duckdb")
    p.add_argument("--resume", action="store_true",
                   help="keep existing ledger DBs (default: fresh run, wipe them)")
    args = p.parse_args(argv)

    enforce_firewall()
    os.makedirs("paper", exist_ok=True)
    if not args.resume:
        for db in (args.paper_ledger_db, args.events_db, args.orderbook_db):
            _rm(db)

    limits = load_limits()
    ri = resolve_ri(args.grade, limits)

    feed = ReplayFeed(args.symbol, args.timeframe, start=args.start, end=args.end,
                      db_path=args.market_db)
    if len(feed) == 0:
        raise SystemExit(f"no bars for {args.symbol} {args.timeframe} in {args.market_db}")

    armed = ArmedStrategy(
        name="breakout_retest",
        strategy=BreakoutRetestStrategy(params=load_params(args.variant)),
        symbol=args.symbol, asset_class="equity", tick=0.01,
        levels={}, grade=args.grade, route="paper", ri=ri,
    )
    system = build_system(
        armed=[armed], equity=args.equity, ri=ri,
        events_db=args.events_db, orderbook_db=args.orderbook_db,
        paper_ledger_db=args.paper_ledger_db, paper_slippage_bps=args.slippage_bps,
    )
    boot(system)
    if not system.started:
        raise SystemExit(f"boot HALTED (not trading): {system.gateway._halt_reason}")

    # --- collect realized trades off the event path + compound equity ---
    realized: list[float] = []          # realized_pnl per closed position
    state = {"equity": float(args.equity)}

    def on_closed(ev) -> None:
        d = ev.data or {}
        pnl = d.get("realized_pnl")
        if pnl is None:
            pnl = d.get("pnl", 0.0)
        pnl = float(pnl or 0.0)
        realized.append(pnl)
        state["equity"] += pnl
        system.equity_source.set(state["equity"])   # compound the next trade's size

    system.bus.subscribe(EventType.POSITION_CLOSED, on_closed)

    # --- replay every session: re-arm levels, feed bars through the bus ---
    n_bars = 0
    for s in feed.iter_sessions():
        # PREMARKET DAILY RESET (this logic belongs in workflows/premarket.py):
        # an intraday cooldown / scoped daily halt stands trading down for the
        # rest of that session; the new session resumes — EXCEPT through a hard
        # program halt (-35% from peak), which requires a manual restart.
        system.breakers.clear_cooldown()
        if not system.breakers.is_program_halted():
            system.gateway.resume()

        armed.levels = feed.levels.get(s.session_date, {}) or {}
        armed.session_started = False     # force on_session_start with fresh levels
        for payload in s.bars:
            system.mark_price(args.symbol, payload["close"])
            system.feed_bar(payload)
            n_bars += 1

    # Close the system FIRST so the PaperBroker releases the ledger file lock
    # (DuckDB forbids a second connection with a different config). We already
    # collected every realized trade off the bus during the feed.
    latency = system.fast_loop.latency_summary()
    system.close()

    # --- score the realized ledger (the Phase-1 deliverable) ---
    import numpy as np
    arr = np.asarray(realized, dtype="float64")
    n = int(arr.size)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else float("nan"))
    win_rate = float((arr > 0).mean()) if n else float("nan")
    net = float(arr.sum())
    curve = equity_curve(realized, args.equity)
    mdd = max_drawdown(curve) if n else 0.0

    # alpha vs SPY over the traded window (SPY closes from the same DB)
    con = duckdb.connect(args.market_db, read_only=True)
    try:
        spy_bars = con.execute(
            "SELECT ts_utc, close FROM bars WHERE symbol='SPY' AND timeframe=? ORDER BY ts_utc",
            [args.timeframe],
        ).fetchall()
    finally:
        con.close()
    alpha = alpha_vs_spy(curve, spy_bars) if (n and spy_bars) else None

    # persistent-ledger evidence (the PaperBroker's own venue view)
    lcon = duckdb.connect(args.paper_ledger_db, read_only=True)
    try:
        n_fills = lcon.execute("SELECT COUNT(*) FROM ledger_fills").fetchone()[0]
        n_orders = lcon.execute("SELECT COUNT(*) FROM ledger_orders").fetchone()[0]
    finally:
        lcon.close()

    SEP = "=" * 72
    print(SEP)
    print("TradeForge — MULTI-SESSION PAPER REPLAY (Phase 1 / P5)  ·  PAPER + OFFLINE")
    print(SEP)
    print(f"  strategy        : breakout_retest/{args.variant}  ({args.symbol} {args.timeframe})")
    print(f"  window          : {feed.session_dates[0]} -> {feed.session_dates[-1]}  "
          f"({len(feed)} sessions, {n_bars} bars fed)")
    print(f"  paper fills      : {n_fills} fills / {n_orders} orders in {args.paper_ledger_db}")
    print(f"  slippage_bps    : {args.slippage_bps} (0 = optimistic; realistic read = backtest)")
    print()
    print(f"  CLOSED TRADES   : {n}")
    print(f"  profit_factor   : {pf:.3f}" if pf == pf else "  profit_factor   : n/a")
    print(f"  win_rate        : {win_rate*100:.1f}%" if win_rate == win_rate else "  win_rate        : n/a")
    print(f"  net P&L         : ${net:,.2f}  (gross +${gp:,.2f} / -${gl:,.2f})")
    print(f"  end equity      : ${state['equity']:,.2f}  (start ${args.equity:,.0f})")
    print(f"  max_drawdown    : {mdd*100:.2f}%")
    if alpha is not None:
        print(f"  ALPHA vs SPY    : strat {alpha['strategy_return']*100:+.2f}%  -  "
              f"SPY {alpha['spy_return']*100:+.2f}%  =  {alpha['alpha']*100:+.2f}%")
    print()
    # paper->live gate (MASTER_PLAN §8): >=30 trades, PF>=1.3, maxDD<=10%, net of costs
    gate_n = n >= 30
    gate_pf = pf == pf and pf >= 1.3
    gate_dd = mdd <= 0.10
    print("  paper->live gate (>=30 trades, PF>=1.3, maxDD<=10%):")
    print(f"    trades>=30 : {'PASS' if gate_n else 'FAIL'} ({n})")
    print(f"    PF>=1.3    : {'PASS' if gate_pf else 'FAIL'} ({pf:.3f})" if pf==pf
          else "    PF>=1.3    : FAIL (n/a)")
    print(f"    maxDD<=10% : {'PASS' if gate_dd else 'FAIL'} ({mdd*100:.2f}%)")
    verdict = "MET" if (gate_n and gate_pf and gate_dd) else "NOT MET"
    print(f"  GATE: {verdict}  (NOTE: optimistic paper fills — confirm on realistic costs)")
    print(f"  hot-path latency: {latency['count']} events, avg {latency['avg_ms']:.3f}ms, "
          f"max {latency['max_ms']:.3f}ms")
    print(SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
