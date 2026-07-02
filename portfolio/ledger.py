"""portfolio/ledger.py — deposits vs edge: TWR (excludes flows) vs MWR/IRR. (#7)

The north star is "$1,000 + $50/week → $100,000", so the book is fed a steady DCA
deposit stream. Two performance questions must NOT be conflated:

  * Is the EDGE working?  → TIME-WEIGHTED RETURN. TWR chains daily returns on
    ``(nav_end - flow_today) / nav_prev`` so the deposit is excluded from the
    return numerator — a $50 deposit must never read as a $50 "gain". This is the
    series the validators (``validation.haircut_verdict``) judge.
  * What did the INVESTOR experience?  → MONEY-WEIGHTED RETURN (IRR) over the
    actual flow stream + terminal NAV — it credits the deposits as the capital
    that did the compounding.

This module is the calculator; the backtester feeds it the daily NAV + the flow
on each day. PURE — no scipy (IRR by bracketed bisection), no I/O.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Optional

import numpy as np
import pandas as pd


def _as_date(d):
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    if hasattr(d, "date"):
        return d.date()
    raise TypeError(f"cannot coerce {d!r} to a date")


class ContributionLedger:
    """Records the deposit/withdrawal flow stream and computes TWR vs MWR.

    Construct with ``{date: cash_flow}`` (a positive flow is a deposit INTO the
    book). ``total_deposited`` is the summed external capital; the TWR / MWR
    helpers take the realized daily NAV curve the backtester builds.
    """

    def __init__(self, contributions: Optional[Mapping] = None):
        self.contributions: dict = {}
        for d, v in (contributions or {}).items():
            self.contributions[_as_date(d)] = float(v)

    # ------------------------------------------------------------------ flows
    def flow_on(self, d) -> float:
        """The external cash flow on ``d`` (0.0 if none)."""
        return float(self.contributions.get(_as_date(d), 0.0))

    def total_deposited(self) -> float:
        """Summed external capital contributed over the program."""
        return float(sum(self.contributions.values()))

    # ------------------------------------------------------------------ TWR
    @staticmethod
    def twr_daily_returns(nav: pd.Series, flows: Mapping) -> pd.Series:
        """Per-day TIME-WEIGHTED return excluding the day's flow.

        ``r_t = (nav_t - flow_t) / nav_{t-1} - 1``. The deposit added to ``nav_t``
        is removed from the numerator so the return reflects only market/edge P&L,
        not the inflow. The first day has no prior NAV and is dropped.
        """
        if nav is None or len(nav) < 2:
            return pd.Series(dtype="float64", name="twr_return")
        idx = [_as_date(d) for d in nav.index]
        vals = nav.astype("float64").to_numpy()
        out_idx = []
        out_val = []
        for i in range(1, len(vals)):
            prev = vals[i - 1]
            if prev <= 0:
                continue
            flow = float(flows.get(idx[i], 0.0)) if flows else 0.0
            out_idx.append(idx[i])
            out_val.append((vals[i] - flow) / prev - 1.0)
        s = pd.Series(out_val, index=out_idx, name="twr_return")
        s.index.name = "date"
        return s

    @staticmethod
    def twr_curve(nav: pd.Series, flows: Mapping, base: float = 1.0) -> pd.Series:
        """The flow-free TWR index (``base * cumprod(1 + twr_returns)``)."""
        rets = ContributionLedger.twr_daily_returns(nav, flows)
        if len(rets) == 0:
            return pd.Series(dtype="float64", name="twr_index")
        curve = base * (1.0 + rets).cumprod()
        curve.name = "twr_index"
        return curve

    @staticmethod
    def twr_cagr(nav: pd.Series, flows: Mapping, periods_per_year: int = 252) -> float:
        """Annualized TWR (geometric) from the flow-free daily return stream."""
        rets = ContributionLedger.twr_daily_returns(nav, flows)
        n = int(len(rets))
        if n < 1:
            return float("nan")
        growth = float((1.0 + rets).prod())
        years = n / float(periods_per_year)
        if years <= 0 or growth <= 0:
            return float("nan")
        return growth ** (1.0 / years) - 1.0

    # ------------------------------------------------------------------ MWR / IRR
    def mwr_irr(self, nav: pd.Series, initial_equity: float) -> float:
        """Money-weighted (IRR) annual return over the flow stream + terminal NAV.

        Investor-perspective cash flows: the starting equity and every deposit are
        capital PUT IN (negative), the terminal NAV is the value TAKEN OUT
        (positive). Solves ``Σ CF_t / (1+r)^{years_t} = 0`` for the annual rate
        ``r`` by bracketed bisection (NPV is monotone decreasing in ``r`` for this
        sign pattern). ``nan`` when the stream cannot be solved (e.g. no terminal
        value or no bracket).
        """
        if nav is None or len(nav) == 0:
            return float("nan")
        idx = [_as_date(d) for d in nav.index]
        t0 = idx[0]
        terminal_date = idx[-1]
        terminal_nav = float(nav.astype("float64").iloc[-1])

        # Build (years_from_t0, cashflow) — initial + deposits negative, terminal +.
        flows: list[tuple[float, float]] = []
        init = float(initial_equity)
        if init != 0:
            flows.append((0.0, -init))
        for d, amt in sorted(self.contributions.items()):
            yrs = (d - t0).days / 365.25
            flows.append((yrs, -float(amt)))
        term_yrs = (terminal_date - t0).days / 365.25
        flows.append((term_yrs, terminal_nav))

        def npv(r: float) -> float:
            acc = 0.0
            for yrs, cf in flows:
                acc += cf / ((1.0 + r) ** yrs)
            return acc

        lo, hi = -0.9999, 100.0
        f_lo, f_hi = npv(lo), npv(hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
            # No sign change in the bracket -> unsolvable / degenerate.
            return float("nan")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = npv(mid)
            if abs(f_mid) < 1e-9:
                return float(mid)
            if f_lo * f_mid < 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return float(0.5 * (lo + hi))
