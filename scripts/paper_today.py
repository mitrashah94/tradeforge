#!/usr/bin/env python3
"""scripts/paper_today.py — paper-trade the VALIDATED edge on a real session.

Bridges a live/real market session (pulled out-of-band, e.g. via the Robinhood
market-data tools) into the SAME validated event path the gate uses, so a single
day can be paper-traded exactly as it was validated — no re-implementation of the
signal.

It runs the gate-passed **breakout_retest / v0_atr_stop** strategy (PDH/PDL
break->retest, 0.25*ATR14 role-reversal stop, QQQ 5m — status PAPER in
``strategies/registry.yaml``) on one session's bars via the real
``BacktestEngine`` under the ``realistic`` cost profile, then sizes any trade on
a small ($1k) account with **fractional** vol-target sizing and journals it.

Why a bridge (not the DB): ``scripts/paper_dry_run.py`` runs this same strategy
from ``market.duckdb`` (historical). This script instead consumes a session that
was pulled LIVE — two JSON files in Robinhood ``get_equity_historicals`` shape:

  --intraday  5-minute bars covering the trade session (+ enough prior context)
  --daily     >=15 daily bars ending at/after the prior session (for the Wilder
              ATR14 and PDH/PDL, computed with the REAL ``data.levels`` code)

Honest small-account note it surfaces: for a ~$720 underlying the vol-target
share count that would spend the 1%-ish risk budget is far more than $1k of
buying power can hold, so the size is **buying-power-capped** and the realized
risk is a small fraction of the budget. The script reports which constraint binds
and the actual risk %. PAPER / RESEARCH only — it places no orders.

Usage
-----
    python3 scripts/paper_today.py --symbol QQQ \
        --intraday session_5m.json --daily qqq_daily.json --date 2026-07-06
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from backtest.engine.cost import CostModel
from backtest.engine.engine import Bar, BacktestEngine
from data.levels import atr14_wilder
from risk.config import load_limits
from risk.sizing import resolve_ri
from strategies.breakout_retest.strategy import BreakoutRetestStrategy, load_params
from orchestrator.fast_loop.sizing import vol_target_qty

VARIANT = "v0_atr_stop"        # the validated realistic-cost survivor
COST_PROFILE = "realistic"     # honest small-account frictions
ET_OFFSET_H = 4                # EDT (US summer); RTH bars only


# --------------------------------------------------------------------------- #
# Parsing (Robinhood get_equity_historicals shape)
# --------------------------------------------------------------------------- #
def _rows(path: str, symbol: str) -> list[dict]:
    raw = Path(path).read_text()
    obj = json.loads(raw[raw.find('{"data"'):]) if '{"data"' in raw else json.loads(raw)
    for res in obj["data"]["results"]:
        if res["symbol"] == symbol:
            return res["bars"]
    raise SystemExit(f"symbol {symbol} not found in {path}")


def _utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)


def _et_date(ts: str) -> date:
    return (_utc(ts) - timedelta(hours=ET_OFFSET_H)).date()


def _daily_df(path: str, symbol: str) -> pd.DataFrame:
    rows = _rows(path, symbol)
    return pd.DataFrame([{
        "session_date": date.fromisoformat(r["begins_at"][:10]),
        "high": float(r["high_price"]), "low": float(r["low_price"]),
        "close": float(r["close_price"]),
    } for r in rows]).sort_values("session_date").reset_index(drop=True)


def _session_bars(path: str, symbol: str, sess: date) -> list[Bar]:
    out = []
    for r in _rows(path, symbol):
        if _et_date(r["begins_at"]) != sess:
            continue
        out.append(Bar(ts=_utc(r["begins_at"]), open=float(r["open_price"]),
                       high=float(r["high_price"]), low=float(r["low_price"]),
                       close=float(r["close_price"]),
                       volume=float(r.get("volume", 0) or 0)))
    return out


# --------------------------------------------------------------------------- #
# Levels (via the REAL data.levels code, for parity with the gate)
# --------------------------------------------------------------------------- #
def build_levels(daily: pd.DataFrame, sess: date) -> dict:
    prior = daily[daily["session_date"] < sess]
    if prior.empty:
        raise SystemExit(f"no daily bars before {sess} to build levels")
    last = prior.iloc[-1]
    atr = atr14_wilder(prior)  # Wilder RMA ATR(14) over daily up to prior session
    return {"pdh": float(last["high"]), "pdl": float(last["low"]),
            "pmh": None, "pml": None, "ntz_low": None, "ntz_high": None,
            "ntz_valid": False, "atr14": None if atr is None else float(atr)}


# --------------------------------------------------------------------------- #
# Small-account fractional sizing (vol-target, buying-power-capped)
# --------------------------------------------------------------------------- #
def size_trade(equity: float, entry: float, stop: float, limits, grade="B") -> dict:
    ri = resolve_ri(grade, limits)
    lvl = limits.level(ri)
    sr = vol_target_qty(equity=equity, ri=ri, limits=limits, entry_price=entry,
                        stop_price=stop, allow_fractional=True)
    ideal = sr.qty
    leverage = lvl.leverage_max or 1.0
    bp_cap = equity * leverage / entry           # fractional shares affordable
    shares = min(ideal, bp_cap)
    binding = "risk_budget" if ideal <= bp_cap else "buying_power"
    risk_dollars = shares * abs(entry - stop)
    return {"ri": ri, "risk_budget": round(sr.dollar_risk, 2),
            "ideal_shares": round(ideal, 4), "bp_cap_shares": round(bp_cap, 4),
            "shares": round(shares, 4), "binding": binding,
            "actual_risk": round(risk_dollars, 2),
            "actual_risk_pct": round(risk_dollars / equity, 4)}


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(symbol: str, intraday: str, daily_path: str, sess: date | None,
        equity: float) -> dict:
    daily = _daily_df(daily_path, symbol)
    if sess is None:
        # last intraday session present
        sess = max(_et_date(r["begins_at"]) for r in _rows(intraday, symbol))
    bars = _session_bars(intraday, symbol, sess)
    if not bars:
        raise SystemExit(f"no intraday bars for {symbol} on {sess}")
    levels = build_levels(daily, sess)

    strat = BreakoutRetestStrategy(params=load_params(VARIANT))
    cost = CostModel.from_profile(COST_PROFILE)
    eng = BacktestEngine(strat, cost, symbol=symbol, asset_class="equity",
                         tick=0.01, initial_equity=equity, percent_of_equity=1.0)
    res = eng.run(bars, {sess: levels}, lambda b: sess)

    limits = load_limits()
    trades = []
    for t in res.trades.itertuples(index=False):
        sizing = size_trade(equity, float(t.entry_price), float(t.stop), limits)
        sign = 1.0 if t.side == "long" else -1.0
        pnl = sizing["shares"] * (float(t.exit_price) - float(t.entry_price)) * sign
        trades.append({
            "side": t.side, "entry_ts": str(t.entry_ts), "exit_ts": str(t.exit_ts),
            "entry": round(float(t.entry_price), 2), "exit": round(float(t.exit_price), 2),
            "stop": round(float(t.stop), 2),
            "target": None if t.target is None else round(float(t.target), 2),
            "exit_reason": t.exit_reason, "r_multiple": round(float(t.r_multiple), 3),
            "sizing": sizing, "pnl": round(pnl, 2), "return_pct": round(pnl / equity, 4),
        })

    return {"symbol": symbol, "session": str(sess), "equity": equity,
            "variant": VARIANT, "cost_profile": COST_PROFILE,
            "levels": {k: (round(v, 2) if isinstance(v, float) else v)
                       for k, v in levels.items()},
            "n_trades": len(trades), "trades": trades}


# --------------------------------------------------------------------------- #
# Journal / digest
# --------------------------------------------------------------------------- #
def journal(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{report['session']}_{report['symbol']}.json"
    p.write_text(json.dumps(report, indent=2))
    return p


def digest(report: dict) -> str:
    L = report["levels"]
    lines = [
        f"── PAPER {report['symbol']} {report['session']}  "
        f"[{report['variant']} · {report['cost_profile']} · ${report['equity']:,.0f}]",
        f"   levels: PDH={L['pdh']} PDL={L['pdl']} ATR14(Wilder,daily)={L['atr14']}",
    ]
    if report["n_trades"] == 0:
        lines.append("   → NO TRADE — no PDH/PDL break→retest this session (correctly flat)")
        return "\n".join(lines)
    tot = 0.0
    for t in report["trades"]:
        s = t["sizing"]
        tot += t["pnl"]
        lines += [
            f"   {t['side'].upper()} entry {t['entry']} → exit {t['exit']} "
            f"({t['exit_reason']}, {t['r_multiple']:+.2f}R)",
            f"     stop {t['stop']}  target {t['target']}",
            f"     size: {s['shares']} sh  (ideal {s['ideal_shares']} vs "
            f"bp-cap {s['bp_cap_shares']} → binding: {s['binding']})",
            f"     risk ${s['actual_risk']} ({s['actual_risk_pct']*100:.2f}% "
            f"of ${report['equity']:,.0f}; budget ${s['risk_budget']})",
            f"     >>> paper P&L ${t['pnl']:+.2f} ({t['return_pct']*100:+.2f}%)",
        ]
    if report["n_trades"] > 1:
        lines.append(f"   TOTAL: ${tot:+.2f} ({tot/report['equity']*100:+.2f}%)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paper-trade breakout_retest on one real session.")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--intraday", required=True, help="5m bars JSON (RH get_equity_historicals shape)")
    ap.add_argument("--daily", required=True, help=">=15 daily bars JSON (same shape)")
    ap.add_argument("--date", default=None, help="trade session YYYY-MM-DD (default: last in intraday)")
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--journal-dir", default="paper/paper_today")
    args = ap.parse_args(argv)

    sess = date.fromisoformat(args.date) if args.date else None
    report = run(args.symbol, args.intraday, args.daily, sess, args.equity)
    print(digest(report))
    p = journal(report, Path(args.journal_dir))
    print(f"\n   journaled → {p}   (PAPER/RESEARCH only — no orders placed)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
