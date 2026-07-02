"""scripts/kronos_probe.py — the honest Kronos probe: forecast, verify skill, deploy-test.

Runs the REAL Kronos-mini (MPS/CPU) over the post-cutoff window and answers, in
order, the only questions that matter:

  1. BATCH — forecast every session (cutoff → data end) for a focused symbol set
     (the meanrev universe + NVDA/TSLA/GLD/IBIT), 5-day horizon, 16 sampled paths,
     seeded per (symbol, session) → a point-in-time ``kronos_forecasts`` table in
     a scratch DuckDB (never mutating market.duckdb).
  2. SKILL — is there any predictive signal? Per-session Spearman rank IC of
     ``exp_return`` vs the realized 5-day forward return, mean IC + t-stat, and
     the directional hit rate of ``prob_up``. No skill here → the overlay is dead
     on arrival, whatever the backtest says.
  3. DEPLOY — run the eval-sim fitness on the post-cutoff window (~10.5 months —
     UNDERPOWERED by construction; every number is labeled as such) for:
       (a) meanrev_strict baseline (no Kronos),
       (b) meanrev_strict + Kronos VETO (negative exp_return blocks the entry),
       (c) KronosTopN — hold the top-3 positive-exp_return names, equal weight
           (the pure "Kronos as the signal" sleeve, risk-managed by the engine).

Honesty: the leakage guard enforces post-cutoff TARGETS on every forecast; the
window is one short regime; and Kronos accrues a FORWARD track record — nothing
here validates it through the normal gate.

CLI:
  KRONOS_REPO_PATH=<clone> PYTHONPATH=. .venv/bin/python scripts/kronos_probe.py [scratch_db]
"""

from __future__ import annotations

import sys
import time
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from backtest.daily.portfolio_backtester import run_portfolio
from backtest.daily.validation import returns_metrics
from data.schema import DEFAULT_DB_PATH, connect
from forecast.kronos.leakage import KRONOS_TRAINING_CUTOFF, LeakageError
from forecast.kronos.predictor import KronosForecaster
from forecast.kronos.store import read_forecasts_asof, write_forecast
from portfolio.config import KronosOverlay, PortfolioConfig, SyntheticStop
from portfolio.model import SleeveSpec
from prop.campaign import run_campaign
from prop.rules import load_firm
from prop.simulate import expected_value

SYMBOLS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU",
           "NVDA", "TSLA", "GLD", "IBIT"]
HORIZON = 5
N_PATHS = 16
CONTEXT = 128
END = "2026-06-26"


def _load_ohlcv(con, symbols):
    """{symbol: OHLCV df (ts_utc asc)} for the probe set (one DB read)."""
    out = {}
    for sym in symbols:
        df = con.execute(
            "SELECT ts_utc, open, high, low, close, volume FROM bars "
            "WHERE symbol=? AND timeframe='1d' ORDER BY ts_utc", [sym],
        ).df()
        if len(df):
            df["d"] = pd.to_datetime(df["ts_utc"]).dt.date
            out[sym] = df
    return out


def _seed(sym: str, d: date) -> int:
    return (abs(hash(sym)) ^ d.toordinal()) % (2**31)


def batch_forecast(bars_by_sym, fdb) -> int:
    """Forecast every post-cutoff session for every symbol → the scratch table."""
    fc = KronosForecaster(device="mps", context=CONTEXT)
    sessions = sorted({d for df in bars_by_sym.values() for d in df["d"]
                       if d >= KRONOS_TRAINING_CUTOFF})
    written = 0
    t0 = time.time()
    for i, asof in enumerate(sessions):
        for sym, df in bars_by_sym.items():
            hist = df[df["d"] <= asof].tail(CONTEXT)
            if len(hist) < CONTEXT:
                continue
            try:
                f = fc.forecast_distribution(sym, asof, hist, horizon=HORIZON,
                                             n_paths=N_PATHS, seed=_seed(sym, asof))
            except LeakageError:
                continue
            write_forecast(fdb, sym, asof, f, horizon=HORIZON, n_paths=N_PATHS)
            written += 1
        if (i + 1) % 40 == 0:
            print(f"  ... {i+1}/{len(sessions)} sessions, {written} forecasts, "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"  batch done: {written} forecasts over {len(sessions)} sessions "
          f"in {time.time()-t0:.0f}s", flush=True)
    return written


def skill_analysis(bars_by_sym, fdb) -> dict:
    """Spearman rank IC of exp_return vs realized 5d forward return + hit rates."""
    closes = {s: df.set_index("d")["close"].astype(float) for s, df in bars_by_sym.items()}
    fwd = {}
    for s, c in closes.items():
        arr = c.to_numpy()
        f = np.full(arr.size, np.nan)
        f[:-HORIZON] = arr[HORIZON:] / arr[:-HORIZON] - 1.0
        fwd[s] = pd.Series(f, index=c.index)

    rows = fdb.execute(
        "SELECT symbol, session_date, exp_return, prob_up FROM kronos_forecasts "
        "WHERE horizon=? ORDER BY session_date", [HORIZON],
    ).fetchall()
    by_day: dict = {}
    for sym, d, er, pu in rows:
        by_day.setdefault(d, []).append((sym, er, pu))

    ics, hits, n_pairs = [], [], 0
    for d, entries in by_day.items():
        pred, real, pus = [], [], []
        for sym, er, pu in entries:
            r = fwd.get(sym)
            if r is None or d not in r.index:
                continue
            rv = r.loc[d]
            if rv != rv or er is None:
                continue
            pred.append(er)
            real.append(float(rv))
            pus.append(pu)
        if len(pred) >= 5:
            pr = pd.Series(pred).rank()
            rr = pd.Series(real).rank()
            ic = float(np.corrcoef(pr, rr)[0, 1])
            if ic == ic:
                ics.append(ic)
        for er, rv, pu in zip(pred, real, pus):
            if pu is not None:
                hits.append(1.0 if ((pu > 0.5) == (rv > 0)) else 0.0)
                n_pairs += 1
    ics = np.asarray(ics)
    mean_ic = float(ics.mean()) if ics.size else float("nan")
    t_stat = (mean_ic / (float(ics.std(ddof=1)) / np.sqrt(ics.size))
              if ics.size > 2 and ics.std(ddof=1) > 0 else float("nan"))
    return {"n_days": int(ics.size), "mean_ic": mean_ic, "t_stat": t_stat,
            "hit_rate": float(np.mean(hits)) if hits else float("nan"),
            "n_pairs": n_pairs}


# ---- deploy-test strategies -------------------------------------------------
class KronosTopN:
    """Hold the top-N positive-exp_return names, equal weight (pure Kronos signal).

    Reads the point-in-time forecast table (a plain dict lookup — deterministic,
    no torch at decision time: the sanctioned slow-loop/fast-loop split).
    """

    def __init__(self, table: dict, top_n: int = 3, weight_cap: float = 0.75):
        self.table = table            # {date: {symbol: exp_return}}
        self.top_n = top_n
        self.weight_cap = weight_cap

    def target_weights(self, asof, history):
        fc = self.table.get(asof, {})
        pos = sorted(((er, s) for s, er in fc.items() if er is not None and er > 0),
                     reverse=True)[: self.top_n]
        if not pos:
            return {}
        w = self.weight_cap / len(pos)
        return {s: w for _er, s in pos}


def deploy_test(con, fdb, start, end) -> list:
    """Eval-sim fitness for baseline / veto / top-N on the post-cutoff window."""
    from strategies.swing_meanrev.strategy import SwingMeanRevStrategy, load_params

    rules = load_firm("equities_25k")
    provider_table: dict = {}
    for (d,) in fdb.execute(
        "SELECT DISTINCT session_date FROM kronos_forecasts").fetchall():
        provider_table[d] = {
            s: f["exp_return"] for s, f in read_forecasts_asof(fdb, d, horizon=HORIZON).items()
        }

    def provider(asof, symbol):
        er = provider_table.get(asof, {}).get(symbol)
        return None if er is None else {"exp_return": er}

    mr = SwingMeanRevStrategy(load_params("STRICT"))
    uni = sorted(set(mr.universe) | set(SYMBOLS) | {"SPY"})
    base_spec = SleeveSpec(name="swing_meanrev", strategy=mr, kind="weight", grade="B",
                           family="meanrev", benchmark="SPY", allocation=1.0)
    kron_pcfg = PortfolioConfig(
        synthetic_stop=SyntheticStop(),
        kronos=KronosOverlay(use_kronos=True, veto_negative_return=True),
    )
    topn = KronosTopN(provider_table, top_n=3)
    topn_spec = SleeveSpec(name="kronos_topn", strategy=topn, kind="weight", grade="B",
                           family="kronos", benchmark="SPY", allocation=1.0)

    runs = [
        ("meanrev_strict (baseline)", [base_spec], None, None),
        ("meanrev_strict + kronos veto", [base_spec], kron_pcfg, provider),
        ("kronos_top3 (pure signal)", [topn_spec], None, None),
    ]
    out = []
    for name, sleeves, pcfg, prov in runs:
        res = run_portfolio(sleeves, uni, start=start, end=end, initial_equity=1000.0,
                            con=con, pcfg=pcfg, forecast_provider=prov)
        rets = res.twr_returns
        rm = returns_metrics(rets)
        ev = expected_value(rets, rules, leverage=12.0, horizon=60,
                            n_paths=1500, block=10, seed=0)
        camp = run_campaign(rets, rules, leverage=12.0, initial_cash=1000.0,
                            horizon_days=378, max_concurrent=10,
                            checkpoints=(252, 378), n_paths=400, block=10, seed=5)
        out.append({
            "name": name, "ann": rm["ann_return"], "maxdd": rm["max_drawdown"],
            "sharpe": rm["sharpe"], "p_pass": ev["p_pass"],
            "survival": ev["funded_survival_rate"],
            "payout": ev["mean_annual_payout_per_account"],
            "ev": ev["ev_one_attempt"], "p100k_18m": camp["p_target_378d"],
        })
    return out


def main(argv) -> int:
    scratch = argv[1] if len(argv) > 1 else "/tmp/kronos_forecasts.duckdb"
    fdb = duckdb.connect(scratch)
    con = connect(DEFAULT_DB_PATH)
    try:
        print(f"Kronos probe — cutoff {KRONOS_TRAINING_CUTOFF}, window -> {END}, "
              f"{len(SYMBOLS)} symbols, h={HORIZON}, paths={N_PATHS}")
        bars = _load_ohlcv(con, SYMBOLS)

        print("\n[1] BATCH forecast (Kronos-mini, MPS)")
        existing = 0
        try:
            existing = fdb.execute("SELECT COUNT(*) FROM kronos_forecasts").fetchone()[0]
        except Exception:  # noqa: BLE001
            pass
        if existing:
            print(f"  reusing {existing} existing forecasts in {scratch}")
        else:
            batch_forecast(bars, fdb)

        print("\n[2] SKILL — does exp_return predict the realized 5d return?")
        sk = skill_analysis(bars, fdb)
        print(f"  sessions with IC: {sk['n_days']}   mean rank-IC: {sk['mean_ic']:+.4f}   "
              f"t-stat: {sk['t_stat']:+.2f}")
        print(f"  prob_up directional hit rate: {sk['hit_rate']:.1%}  "
              f"({sk['n_pairs']} symbol-days)")

        print(f"\n[3] DEPLOY — eval fitness on the post-cutoff window "
              f"({KRONOS_TRAINING_CUTOFF} -> {END}; ~10.5 months — UNDERPOWERED)")
        rows = deploy_test(con, fdb, str(KRONOS_TRAINING_CUTOFF), END)
        hdr = (f"  {'config':<30s} {'ann':>7s} {'maxDD':>6s} {'sharpe':>7s} "
               f"{'P(pass)':>8s} {'surv':>6s} {'$/acct':>8s} {'EV':>7s} {'P100k@18m':>10s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in rows:
            print(f"  {r['name']:<30s} {r['ann']:>7.1%} {r['maxdd']:>6.1%} "
                  f"{r['sharpe']:>7.2f} {r['p_pass']:>8.1%} {r['survival']:>6.1%} "
                  f"${r['payout']:>6,.0f} ${r['ev']:>5,.0f} {r['p100k_18m']:>10.1%}")
        print("\nHONESTY: one ~10.5-month regime, forecasts from a 4.1M-param mini "
              "model, deploy metrics at 12x leverage — directionally informative, "
              "not validation. Kronos earns trust only through a forward record.")
    finally:
        con.close()
        fdb.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
