#!/usr/bin/env python3
"""scripts/paper_dry_run.py — the ONE-SESSION PAPER DRY RUN (P4 capstone).

Wires the full TradeForge slow loop (P4 LLM-overseen agents) onto the P3 event
core (bus + order gateway + paper broker + fast loop) over a SINGLE real QQQ 5m
session, end to end, and showcases four artifacts (MASTER_PLAN.md §4 flows
F1/F2/F5; §9 "Metrics That Matter"):

    1. REGIME LINE        — regime-reader: tag, realized vol, exposure scalar,
                            armed list. Published as REGIME_TAGGED on the bus.
    2. JOURNAL ENTRY      — journalist auto-journals the closed trade: frame card,
                            intended-vs-actual slippage, MFE/MAE, plan-adherence
                            flags, 3-line narrative + a chart PNG.
    3. ALPHA-VS-SPY LINE  — performance-analyst: strategy net-of-costs return
                            minus SPY buy-&-hold over the validation window, plus
                            a short summary (geometric growth, curve vol,
                            after-tax equity, risk of ruin).
    4. RATCHET SWEEP      — performance-analyst crosses the first milestone
                            ($2,500): MILESTONE_REACHED + RATCHET_SWEEP.
    + the EOD DIGEST      — journalist.eod_digest via the notify FILE/LOG sink.

DETERMINISTIC + OFFLINE + PAPER ONLY. Reads ONLY ``data/duckdb/market.duckdb``;
all order flow is PaperBroker (paper/ledger.duckdb); all notifications go to the
FILE/LOG sink (journal/notifications.log). NO live orders, NO real iMessage, NO
network. The research firewall holds: nothing live is written or sent.

The chosen session is auto-selected (or pinned via --date): the most recent QQQ
session that (a) the regime-reader tags TREND so it ARMS breakout_retest, and
(b) the validated breakout_retest/v0_atr_stop strategy actually trades. The
script then replays JUST that session's bars through the live fast loop on the
shared bus, so the trade is produced by the real event path — not simulated.

Run:
    PYTHONPATH=. .venv/bin/python scripts/paper_dry_run.py
    PYTHONPATH=. .venv/bin/python scripts/paper_dry_run.py --date 2024-07-25
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from backtest.runner import daily_returns, load_bars_levels, levels_map, run_strategy
from data.schema import DEFAULT_DB_PATH, connect
from data.sessions import et_session_date, to_et
from orchestrator.agents.journalist import Journalist, Trade, compute_slippage
from orchestrator.agents.performance_analyst import (
    PerformanceAnalyst,
    alpha_vs_spy,
    equity_curve,
)
from orchestrator.agents.regime_reader import RegimeAssessment, assess
from orchestrator.agents.regime_reader import publish as publish_regime
from orchestrator.events import Event, EventType
from orchestrator.fast_loop import ArmedStrategy
from orchestrator.main import boot, build_system
from risk.config import load_limits
from risk.sizing import resolve_ri
from strategies.breakout_retest.strategy import BreakoutRetestStrategy, load_params

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
SYMBOL = "QQQ"
TIMEFRAME = "5m"
VARIANT = "v0_atr_stop"          # the validated realistic-cost survivor
COST_PROFILE = "realistic"        # honest small-account frictions
BASE_EQUITY = 1000.0              # paper starting equity (CLAUDE.md north star)
CHART_PATH = REPO_ROOT / "backtest" / "reports" / "dryrun_trade.png"

SEP = "=" * 72


def _hr(title: str) -> None:
    print()
    print(SEP)
    print(title)
    print(SEP)


# --------------------------------------------------------------------------- #
# Firewall assertion (paper + log only; never touch live config)
# --------------------------------------------------------------------------- #
def enforce_firewall() -> None:
    """Pin the safe defaults: file/log notify sink, no iMessage recipient, paper.

    The research firewall (MASTER_PLAN §6): this run reads everything and writes
    nothing live. We force the notify channel to the file sink and clear any
    iMessage recipient so a misconfigured env can NEVER send outward, and we
    assert TRADEFORGE_LIVE is not enabled so no live broker can be constructed.
    """
    os.environ["TRADEFORGE_NOTIFY_CHANNEL"] = "file"
    os.environ.pop("TRADEFORGE_NOTIFY_TO", None)
    if os.environ.get("TRADEFORGE_LIVE") == "1":
        raise SystemExit(
            "FIREWALL: TRADEFORGE_LIVE=1 is set; refusing to run the dry run "
            "with the live surface enabled. Unset it and re-run (paper only)."
        )


# --------------------------------------------------------------------------- #
# 1. Pick a session that produces a v0_atr_stop trade AND is a trend regime
# --------------------------------------------------------------------------- #
def pick_session(con, pinned: str | None) -> tuple[date, pd.Series]:
    """Return (session_date, trade_row) for a session that trades AND is armed.

    Runs the validated v0_atr_stop strategy over the full 2y window under the
    realistic cost model, then walks its intraday trades (newest first) and picks
    the first whose session the regime-reader tags TREND — i.e. the regime that
    ARMS breakout_retest. If ``pinned`` is given, that exact session is used (and
    must produce a trade), so the showcase is reproducible.
    """
    strat = BreakoutRetestStrategy(params=load_params(VARIANT))
    result = run_strategy(strat, SYMBOL, TIMEFRAME, cost_profile=COST_PROFILE, con=con)
    trades = result.trades.copy()
    trades["entry_date"] = trades["entry_ts"].apply(et_session_date)
    trades["exit_date"] = trades["exit_ts"].apply(et_session_date)
    # Single-session replay needs an intraday round-trip (entry & exit same day).
    intraday = trades[trades["entry_date"] == trades["exit_date"]]

    if pinned is not None:
        want = datetime.strptime(pinned[:10], "%Y-%m-%d").date()
        rows = intraday[intraday["entry_date"] == want]
        if len(rows) == 0:
            raise SystemExit(f"--date {pinned}: no v0_atr_stop intraday trade that session")
        return want, rows.iloc[0]

    # Auto-select among sessions the regime-reader ARMS breakout_retest (TREND).
    # Prefer a full TP lifecycle (a clean ``target`` exit) so the showcase shows
    # an entry->target round-trip; fall back to the newest armed session of any
    # exit kind (honest: show whatever the real bars produce).
    armed_rows: list[pd.Series] = []
    for _, row in intraday.sort_values("entry_date", ascending=False).iterrows():
        if "breakout_retest" in assess(row["entry_date"], SYMBOL, con=con).armed:
            armed_rows.append(row)
    if not armed_rows:
        raise SystemExit("no v0_atr_stop trade landed on a trend (breakout-armed) session")
    for row in armed_rows:
        if row["exit_reason"] == "target":
            return row["entry_date"], row
    return armed_rows[0]["entry_date"], armed_rows[0]


# --------------------------------------------------------------------------- #
# Exposure-scalar -> sizing + risk-cap reconciliation
# --------------------------------------------------------------------------- #
def grade_for_scalar(scalar: float, base_grade: str, limits) -> str:
    """Resolve the conviction grade whose RI band-cap covers the scaled budget.

    The regime exposure scalar is a per-trade-$-risk multiplier on the conviction
    tier's budget (regime_reader docstring): ``effective_$risk = scalar * tier``.
    The fast loop sizes against the scaled budget; the gateway's per-trade cap is
    the RI band ceiling. So a leaned-in (scalar > 1) trade needs the gateway RI
    one (or more) steps higher so its cap accommodates the larger budget — which
    is exactly "shift the effective RI up within the band". We pick the LOWEST
    grade whose RI per-trade-% >= scalar * base-grade %, clamped to the band high
    (RI 8): the band ceiling still caps a leaned-in trade. scalar <= 1 keeps the
    base grade (a cut just sizes smaller, well under the base cap).
    """
    base_ri = resolve_ri(base_grade, limits)
    base_pct = limits.level(base_ri).per_trade_pct
    needed_pct = scalar * base_pct
    if scalar <= 1.0:
        return base_grade
    # Find the lowest conviction grade whose RI cap covers the scaled budget.
    grades = sorted(limits.conviction_tiers.items(), key=lambda kv: kv[1])  # by RI
    for g, _ri in grades:
        ri = resolve_ri(g, limits)
        if limits.level(ri).per_trade_pct + 1e-9 >= needed_pct:
            return g
    return max(grades, key=lambda kv: kv[1])[0]  # band high (A+)


# --------------------------------------------------------------------------- #
# 3+4. Wire the live system and replay the session
# --------------------------------------------------------------------------- #
def run_live_session(
    session_date: date,
    assessment: RegimeAssessment,
    levels: dict,
    bars_df: pd.DataFrame,
    tmpdir: str,
) -> dict:
    """Build + boot the P3 system, replay the session, return the captured trade.

    Wires the v0_atr_stop strategy ARMED per the regime read (exposure_scalar
    applied to sizing), the journalist + performance-analyst onto the SAME shared
    bus, then feeds the session's bars as BAR events. The fast loop emits
    ORDER_INTENT -> gateway approves -> paper fills -> F2 lifecycle -> the exit
    fills -> POSITION_CLOSED. Returns the enriched closed-trade view + the
    analyst handle.
    """
    limits = load_limits()
    base_grade = "A"                       # breakout-retest A setup
    base_ri = resolve_ri(base_grade, limits)
    scalar = float(assessment.exposure_scalar)

    # Exposure scalar -> sizing basis: scale the equity the fast loop sizes
    # against (per_trade_$risk = pct * equity, so scaling equity scales the
    # budget linearly == effective_$risk = scalar * tier_budget). The gateway
    # gets this same equity for its per-trade cap, so sizing and cap share one
    # basis. The grade is shifted up so the band-cap accommodates the lean-in.
    sizing_equity = BASE_EQUITY * scalar
    gate_grade = grade_for_scalar(scalar, base_grade, limits)

    armed = ArmedStrategy(
        name="breakout_retest",
        strategy=BreakoutRetestStrategy(params=load_params(VARIANT)),
        symbol=SYMBOL,
        asset_class="equity",
        tick=0.01,
        levels=levels,
        grade=gate_grade,           # gateway RI cap basis (band ceiling caps it)
        route="paper",
        ri=base_ri,                 # fast-loop sizing RI (base conviction tier)
    )

    system = build_system(
        armed=[armed],
        equity=sizing_equity,
        ri=base_ri,
        events_db=os.path.join(tmpdir, "events.duckdb"),
        orderbook_db=os.path.join(tmpdir, "orderbook.duckdb"),
        paper_ledger_db=os.path.join(tmpdir, "ledger.duckdb"),
    )
    # Reconcile-first boot (on-boot recovery runs before any order is placed).
    boot(system)
    if not system.started:
        raise SystemExit(f"boot HALTED: {system.gateway._halt_reason}")

    # --- Slow-loop agents on the SAME bus ---
    # The journalist writes to a fresh per-run journal dir so the run is
    # self-contained, deterministic, and re-runnable (no accumulation into the
    # repo's journal/). It auto-journals the closed trade OFF THE BUS: a
    # POSITION_CLOSED subscriber builds the rich Trade (the gateway's payload is
    # sparse, so we enrich from the captured plan + position record) and calls
    # journal_trade — the real bus auto-journal path, fully populated.
    journal_dir = os.path.join(tmpdir, "journal")
    # Notify FILE/LOG sink (safe default) pointed at the per-run temp dir so the
    # dry run writes NOTHING into the repo and never sends outward (firewall).
    from orchestrator.tools.notify import notify as _notify
    notif_log = os.path.join(journal_dir, "notifications.log")

    def _log_sink(message: str, *, title=None, channel=None) -> str:
        return _notify(message, title=title, channel="file", log_path=notif_log)

    journalist = Journalist(
        journal_dir=journal_dir,
        notifier=_log_sink,
        market_db=str(REPO_ROOT / "data" / "duckdb" / "market.duckdb"),
        orderbook=system.orderbook,
    )
    captured_notif_log = notif_log

    analyst = PerformanceAnalyst(bus=system.bus, starting_capital=BASE_EQUITY)
    analyst.attach(system.bus)

    # Publish the regime read onto the shared bus (REGIME_TAGGED) — the policy
    # the fast loop / risk gate consume (arming + exposure scalar).
    publish_regime(system.bus, assessment)

    # --- capture the trade plan + fills + excursion off the bus ---
    captured: dict = {
        "entry_fill": None, "exit_fill": None, "side": None,
        "plan": {}, "closed": None, "position_id": None, "entry_slippage": None,
        "entry_ts": None, "exit_ts": None,
        # the session's 5m OHLC bars for the journal chart (passed explicitly so
        # the journalist needn't re-open market.duckdb while it is already open).
        "session_bars": [
            {"open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close)}
            for r in bars_df.itertuples(index=False)
        ],
    }

    # The replay loop sets this to the current bar's ts_utc before each
    # feed_bar, so the synchronous fill/close subscribers can stamp the trade.
    cursor = {"bar_ts": None}

    def on_order_filled(ev: Event) -> None:
        # The order-fill detail uses the OrderBook's record_fill keys: the price
        # is ``price`` (NOT ``fill_price``) and slippage rides as ``slippage``.
        d = ev.data or {}
        reason = d.get("reason", "")
        if reason.startswith("entry_"):
            captured["entry_fill"] = d.get("price")
            captured["entry_slippage"] = d.get("slippage")
            captured["side"] = "long" if reason.endswith("long") else "short"
            captured["position_id"] = d.get("position_id")
            captured["entry_ts"] = cursor["bar_ts"]
        elif reason in ("target", "stop", "trail_stop", "session_flatten",
                        "time_stop", "strategy_close", "close"):
            captured["exit_fill"] = d.get("price")
            captured["exit_reason"] = reason
            captured["exit_ts"] = cursor["bar_ts"]

    def on_position_closed(ev: Event) -> None:
        # Auto-journal off the bus with a fully-enriched Trade, then keep the
        # JournalEntry for the showcase. Fires synchronously on the closing bar.
        captured["closed"] = ev.data or {}
        captured["system"] = system
        trade = build_trade(session_date, assessment, levels, captured)
        captured["journal_entry"] = journalist.journal_trade(trade)
        captured["trade"] = trade

    system.bus.subscribe(EventType.ORDER_FILLED, on_order_filled)
    system.bus.subscribe(EventType.POSITION_CLOSED, on_position_closed)

    # Capture the planned bracket the loop emits with the entry intent.
    def on_intent(ev: Event) -> None:
        d = ev.data or {}
        if str(d.get("reason", "")).startswith("entry_"):
            captured["plan"] = {
                "planned_entry": d.get("intended_price"),
                "stop": (d.get("bracket") or {}).get("stop_price"),
                "target": (d.get("bracket") or {}).get("target_price"),
                "qty": d.get("qty"),
            }

    system.bus.subscribe(EventType.ORDER_INTENT, on_intent)

    # --- replay the session's bars as BAR events on the shared bus ---
    # We track MFE/MAE ($ favorable/adverse excursion) ourselves over the bars
    # the position is open — including the entry and exit bars — against the
    # entry fill, so the journal's excursion is exact and independent of the
    # OrderBook position record (which is qty=0 by the time POSITION_CLOSED
    # fires synchronously inside feed_bar).
    n = len(bars_df)
    captured["mfe"] = 0.0
    captured["mae"] = 0.0
    for i, r in enumerate(bars_df.itertuples(index=False)):
        is_eod = i == n - 1
        cursor["bar_ts"] = r.ts_utc
        cursor["bar"] = {"high": float(r.high), "low": float(r.low), "close": float(r.close)}
        # Excursion BEFORE feed_bar: a position open at the START of this bar is
        # exposed to the bar's whole range. This includes the EXIT bar (still
        # open at bar-start) and excludes the ENTRY bar (it opens mid-bar at the
        # signal close), which is the correct, non-over-counted MFE/MAE.
        if captured.get("entry_fill") is not None and captured.get("closed") is None:
            entry_px = captured["entry_fill"]
            qty = float(captured["plan"].get("qty") or 0.0)
            if captured["side"] == "long":
                captured["mfe"] = max(captured["mfe"], (float(r.high) - entry_px) * qty)
                captured["mae"] = min(captured["mae"], (float(r.low) - entry_px) * qty)
            else:  # short: favorable on a drop, adverse on a rise
                captured["mfe"] = max(captured["mfe"], (entry_px - float(r.low)) * qty)
                captured["mae"] = min(captured["mae"], (entry_px - float(r.high)) * qty)
        # Feed the venue a mark so resting exits (target/stop) fill correctly,
        # then publish the BAR event. POSITION_CLOSED (and the auto-journal) may
        # fire synchronously inside feed_bar on the exit bar — by which point the
        # exit bar's excursion is already booked above.
        system.mark_price(SYMBOL, float(r.close))
        system.feed_bar({
            "symbol": SYMBOL, "ts_utc": str(r.ts_utc),
            "open": float(r.open), "high": float(r.high), "low": float(r.low),
            "close": float(r.close), "volume": float(r.volume), "is_eod": is_eod,
        })
        if captured.get("closed") is not None:
            break  # position closed on this bar; nothing more to manage

    captured["analyst"] = analyst
    captured["journalist"] = journalist
    captured["system"] = system
    captured["latency"] = system.fast_loop.latency_summary()
    captured["notif_log"] = captured_notif_log
    return captured


# --------------------------------------------------------------------------- #
# Build the rich Trade from the captured live run
# --------------------------------------------------------------------------- #
def build_trade(session_date: date, assessment: RegimeAssessment, levels: dict,
                captured: dict) -> Trade:
    """Map the live run (plan + fills + position record) to a journalist Trade."""
    plan = captured["plan"]
    closed = captured["closed"] or {}
    side = captured["side"] or "long"

    # MFE/MAE: the $ favorable / adverse excursion tracked over the open bars in
    # the replay loop (exact, includes the exit bar, excludes pre-entry range).
    mfe = captured.get("mfe")
    mae = captured.get("mae")

    levels_view = {k.upper(): levels.get(k) for k in ("pdh", "pdl", "pmh", "pml")
                   if levels.get(k) is not None}

    # Bars are stored tz-naive UTC; the journalist's entered_in_window? flag is
    # evaluated against the RTH clock, so present the fill timestamps in ET.
    def _et_naive(ts):
        if ts is None:
            return None
        return to_et(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts).replace(tzinfo=None)

    return Trade(
        symbol=SYMBOL,
        strategy="breakout_retest",
        side=side,
        setup_grade="A",
        planned_entry=plan.get("planned_entry"),
        planned_stop=plan.get("stop"),
        planned_target=plan.get("target"),
        levels=levels_view,
        entry_price=captured.get("entry_fill"),
        exit_price=captured.get("exit_fill"),
        qty=plan.get("qty"),
        realized_pnl=closed.get("realized_pnl"),
        slippage=captured.get("entry_slippage"),
        mfe=mfe,
        mae=mae,
        entry_ts=_et_naive(captured.get("entry_ts")),
        exit_ts=_et_naive(captured.get("exit_ts")),
        exit_reason=captured.get("exit_reason", closed.get("reason", "")),
        bars=captured.get("session_bars"),
    )


# --------------------------------------------------------------------------- #
# 5. Alpha-vs-SPY after costs (over the validation window)
# --------------------------------------------------------------------------- #
def compute_alpha_and_summary(con) -> dict:
    """Strategy (v0_atr_stop, net of realistic costs) vs SPY over the 2y window.

    The §9 headline: we run the validated survivor over the full window, build
    its realized equity curve, and compare its after-cost total return to SPY
    buy-&-hold over the SAME dates. The analyst's summary (geometric growth,
    curve vol, after-tax equity, risk of ruin) is computed from the realized
    per-trade pnls — the honest read of the strategy as an edge.

    Stats are computed on the BACKTEST's own equity basis (its ``initial_equity``,
    % -of-equity sized) so the per-trade returns, geometric growth, and drawdown
    are meaningful — the realized $ pnls are scaled to that account, not to the
    $1,000 paper-boot equity (which is the live single-session sizing basis).
    """
    backtest_equity = 100_000.0
    strat = BreakoutRetestStrategy(params=load_params(VARIANT))
    result = run_strategy(strat, SYMBOL, TIMEFRAME, cost_profile=COST_PROFILE,
                          initial_equity=backtest_equity, con=con)

    # Per-trade realized pnls + R multiples feed the analyst's measured edge.
    trades = result.trades
    pnls = trades["pnl"].astype("float64").tolist()
    r_multiples = trades["r_multiple"].astype("float64").tolist()

    # Window = the realized trades' date span.
    start = et_session_date(trades["entry_ts"].iloc[0])
    end = et_session_date(trades["exit_ts"].iloc[-1])

    spy_bars = con.execute(
        "SELECT ts_utc, close FROM bars WHERE symbol = 'SPY' AND timeframe = '5m' "
        "ORDER BY ts_utc"
    ).fetchall()

    # Feed the analyst (off-bus: we drive record_trade directly so its equity
    # curve == the realized series; SPY bars give the benchmark).
    analyst = PerformanceAnalyst(starting_capital=backtest_equity, spy_bars=spy_bars)
    for pnl, r in zip(pnls, r_multiples):
        analyst.record_trade(pnl, strategy="breakout_retest", r_multiple=r)

    # The analyst was given spy_bars, so summary() already carries alpha_vs_spy;
    # compute it explicitly too (same equity curve, same SPY bars) as a guard.
    summary = analyst.summary(ri=6)
    alpha = summary.get("alpha_vs_spy")
    if alpha is None:
        alpha = alpha_vs_spy(equity_curve(pnls, backtest_equity), spy_bars)
    return {
        "alpha": alpha,
        "summary": summary,
        "window": (start, end),
        "n_trades": len(pnls),
        "backtest_summary": result.summary(),
    }


# --------------------------------------------------------------------------- #
# 7. Simulated ratchet sweep at the first milestone ($2,500)
# --------------------------------------------------------------------------- #
def simulate_ratchet_sweep() -> dict:
    """Feed the analyst an equity that crosses the $2,500 milestone on its bus.

    The analyst emits MILESTONE_REACHED + RATCHET_SWEEP (F5 growth). We use a
    small recorder bus so the emitted events are captured verbatim, then return
    the sweep payload. FIREWALL: the analyst only PROPOSES the sweep on the bus;
    no capital is actually moved and no live config is touched.
    """
    class _RecorderBus:
        def __init__(self):
            self.events: list[Event] = []

        def publish(self, event: Event) -> Event:
            self.events.append(event)
            return event

    bus = _RecorderBus()
    analyst = PerformanceAnalyst(bus=bus, starting_capital=BASE_EQUITY)
    # Cross the first milestone: $1,000 -> ~$2,600 in one (illustrative) booked
    # gain so check_milestone() triggers the $2,500 ratchet.
    analyst.record_trade(1600.0, strategy="breakout_retest", r_multiple=2.0)

    milestone_ev = next((e for e in bus.events if e.type == EventType.MILESTONE_REACHED), None)
    sweep_ev = next((e for e in bus.events if e.type == EventType.RATCHET_SWEEP), None)
    return {
        "milestone": milestone_ev.data if milestone_ev else None,
        "sweep": sweep_ev.data if sweep_ev else None,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TradeForge one-session paper dry run (P4 capstone)")
    parser.add_argument("--date", default=None, help="pin the session (YYYY-MM-DD); else auto-select")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="market.duckdb path")
    args = parser.parse_args(argv)

    enforce_firewall()

    print(SEP)
    print("TradeForge — ONE-SESSION PAPER DRY RUN (P4 capstone)")
    print("PAPER + LOG ONLY · offline (market.duckdb) · deterministic · firewall on")
    print(SEP)

    con = connect(args.db)
    try:
        # ---- 1. pick the session ----
        session_date, trade_row = pick_session(con, args.date)
        print(f"\n[1] Chosen session: {session_date}  "
              f"(v0_atr_stop {trade_row['side']} trade; backtest exit "
              f"{trade_row['exit_reason']} {trade_row['r_multiple']:+.2f}R)")

        # ---- 2. regime-reader ----
        assessment = assess(session_date, SYMBOL, con=con)

        # ---- session bars + levels ----
        bars_df, levels_df = load_bars_levels(
            SYMBOL, TIMEFRAME, start=str(session_date), end=str(session_date), con=con
        )
        levels = levels_map(levels_df)[session_date]

        # ---- 3+4. live replay on the shared bus ----
        captured = run_live_session(session_date, assessment, levels, bars_df,
                                    tempfile.mkdtemp(prefix="dryrun_"))
        journalist: Journalist = captured["journalist"]
        # The journalist already auto-journaled the closed trade OFF THE BUS
        # (POSITION_CLOSED -> rich Trade -> journal_trade); reuse that entry/trade.
        if captured.get("journal_entry") is None:
            raise SystemExit(
                f"no POSITION_CLOSED produced for {session_date} — the session "
                "did not complete a round-trip through the live loop"
            )
        entry = captured["journal_entry"]
        trade = captured["trade"]
        chart_path = journalist.chart_for_trade(trade, CHART_PATH)

        # ---- EOD digest via the notify FILE/LOG sink (reads this run's journal) ----
        eod_text = journalist.eod_digest(str(session_date), send=True)

        # ---- 5. alpha-vs-SPY + summary ----
        alpha_pack = compute_alpha_and_summary(con)

        # ---- 7. ratchet sweep ----
        ratchet = simulate_ratchet_sweep()
    finally:
        con.close()

    # ===================================================================== #
    # SHOWCASE
    # ===================================================================== #
    _hr("===== SHOWCASE =====")

    # ---- Artifact 1: regime line ----
    print("\n--- [1] REGIME LINE (REGIME_TAGGED on the shared bus) ---")
    rv = "n/a" if assessment.realized_vol is None else f"{assessment.realized_vol:.3f}x ATR"
    print(f"  date={session_date}  symbol={assessment.symbol}  regime={assessment.regime.upper()}")
    print(f"  realized_vol={rv}  iv={assessment.iv}  exposure_scalar={assessment.exposure_scalar:.2f}")
    print(f"  armed={list(assessment.armed)}")
    print(f"  rationale: {assessment.rationale}")

    # ---- Artifact 2: journal entry ----
    print("\n--- [2] JOURNAL ENTRY (journalist, auto-journaled off POSITION_CLOSED) ---")
    fc = entry.frame_card
    ad = entry.adherence
    print(f"  trade_id: {entry.trade_id}")
    print(f"  Frame card: {fc.symbol} {fc.side} {fc.strategy} grade={fc.setup_grade}")
    print(f"    levels: {fc.levels}")
    print(f"    planned entry={fc.planned_entry} stop={fc.planned_stop} "
          f"target={fc.planned_target}  planned R={fc.r_planned:.2f}"
          if fc.r_planned is not None else
          f"    planned entry={fc.planned_entry} stop={fc.planned_stop} target={fc.planned_target}")
    print(f"  Slippage (actual entry − intended): {compute_slippage(trade):+.4f}"
          if compute_slippage(trade) is not None else "  Slippage: n/a")
    print(f"  MFE={entry.mfe:+.4f}  MAE={entry.mae:+.4f}"
          if entry.mfe is not None and entry.mae is not None else
          f"  MFE={entry.mfe}  MAE={entry.mae}")
    print(f"  Realized P&L={entry.realized_pnl:+.2f}  Realized R="
          f"{entry.r_realized:+.2f}" if entry.realized_pnl is not None
          and entry.r_realized is not None else
          f"  Realized P&L={entry.realized_pnl}  Realized R={entry.r_realized}")
    print(f"  Plan-adherence flags: entered_in_window={ad.entered_in_window} "
          f"stop_at_planned_level={ad.stop_at_planned_level} "
          f"exited_per_plan={ad.exited_per_plan} held_past_eod={ad.held_past_eod}")
    print("  3-line narrative:")
    for line in entry.narrative.splitlines():
        print(f"    {line}")
    print(f"  Chart PNG: {chart_path}")

    # ---- the EOD digest ----
    print("\n--- [+] EOD DIGEST (journalist.eod_digest via notify FILE/LOG sink) ---")
    for line in eod_text.splitlines():
        print(f"  {line}")

    # ---- Artifact 3: alpha-vs-SPY after costs ----
    print("\n--- [3] ALPHA-VS-SPY AFTER COSTS (performance-analyst) ---")
    a = alpha_pack["alpha"]
    w0, w1 = alpha_pack["window"]
    s = alpha_pack["summary"]
    geo = s["geometric_growth"]
    at = s["after_tax"]
    print(f"  Window {w0} .. {w1}  ({alpha_pack['n_trades']} trades, "
          f"v0_atr_stop, net of realistic costs)")
    print(f"  ALPHA vs SPY: strategy {a['strategy_return']*100:+.2f}%  −  "
          f"SPY {a['spy_return']*100:+.2f}%  =  alpha {a['alpha']*100:+.2f}%")
    print(f"  Summary: geo_growth/trade={geo['geo_mean_daily']*100:+.4f}%  "
          f"curve_vol/trade={geo['vol_daily']*100:.4f}%  "
          f"g≈mean−var/2={geo['g_approx_daily']*100:+.4f}%")
    print(f"           profit_factor={s['profit_factor']:.3f}  "
          f"expectancy=${s['expectancy_dollar']:+.2f}  "
          f"equity=${s['equity']:,.2f}")
    print(f"           after-tax equity=${at['aftertax_equity']:,.2f} "
          f"(reserve {at['tax_reserve_rate']*100:.0f}% on ${at['realized_gain']:,.2f} gain)")
    print(f"           risk_of_ruin@RI{s['ri']}={s['risk_of_ruin']:.4g}  "
          f"max_drawdown={s['max_drawdown']*100:.2f}%")
    verdict = ("CLEARS the bar (alpha > 0)" if a["alpha"] > 0
               else "BELOW the bar (alpha <= 0): a single edge on flat sizing does "
                    "not beat SPY here — stack uncorrelated edges (MASTER_PLAN §1.B)")
    print(f"  Verdict: {verdict}.")

    # ---- Artifact 4: ratchet sweep ----
    print("\n--- [4] RATCHET SWEEP at the first milestone (F5 growth) ---")
    ms = ratchet["milestone"] or {}
    sw = ratchet["sweep"] or {}
    print(f"  MILESTONE_REACHED: milestone=${ms.get('milestone'):,.0f}  "
          f"equity=${ms.get('equity'):,.2f}  baseline=${ms.get('baseline'):,.2f}")
    print(f"  RATCHET_SWEEP:     milestone=${sw.get('milestone'):,.0f}  "
          f"sweep={sw.get('sweep_fraction')*100:.0f}%×gains=${sw.get('sweep_amount'):,.2f}")
    print(f"                     new_baseline=${sw.get('new_baseline'):,.2f}  "
          f"vault_balance=${sw.get('vault_balance'):,.2f}")

    # ---- footer: integrity ----
    lat = captured["latency"]
    _hr("INTEGRITY")
    print(f"  Hot-path latency: {lat['count']} events, avg {lat['avg_ms']:.3f}ms, "
          f"max {lat['max_ms']:.3f}ms, over-budget {lat['over_budget']} "
          f"(cold-start import on the first event; steady-state is sub-ms)")
    print("  Venue: PAPER only (temp paper ledger; no live broker constructed).")
    print(f"  Notify: FILE/LOG sink only -> {captured['notif_log']} (no iMessage sent).")
    print("  Journal: per-run temp dir (nothing written into the repo journal/).")
    print(f"  Chart artifact (the one repo write): {CHART_PATH}")
    print("  Firewall: no live orders, no real iMessage, no live-config writes.")
    print(SEP)

    captured["system"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
