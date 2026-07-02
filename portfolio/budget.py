"""portfolio/budget.py — book-level sizing + limits. (#4 size, #5 limits)

This is where the book becomes ONE risk-budgeted account. Every limit is read
from a single ``risk/limits.yaml`` row (the BOOK RI = ``limits.default_ri``,
conviction-flexed per candidate) — not from per-strategy config — so heat,
concurrency, the daily / weekly halts, and the always-on program-abort drawdown
halts all apply to the whole book at once.

Two halves:

  SIZE (#4)
    * SCORE candidate → ``shares = dollar_risk / (entry - stop)`` with
      ``dollar_risk = per_trade_dollar_risk(equity, resolve_ri(grade)) * scalar``;
      its heat contribution IS that ``dollar_risk``.
    * WEIGHT candidate → ``notional = target_weight * equity * scalar``,
      ``shares = notional / mark``; its heat is the SYNTHETIC-STOP band risk
      ``shares * (mark - (mark - k*ATR)) = notional * (k*ATR / mark)`` — the one
      new modeling knob, so a stop-less weight position contributes a comparable
      dollar-risk to the heat budget (band falls back to ``fallback_band_frac`` of
      the mark when ATR is unavailable).

  LIMITS (#5)
    * program-abort drawdown halts (−35% from peak, −20% in the month) — always
      on, never on the risk dial;
    * daily / weekly loss halts (the RI row);
    * the heat-admission walk: in priority order, admit candidates while the
      running book heat stays ≤ ``portfolio_heat_pct`` AND each family stays ≤ its
      optional family cap AND the book stays within ``max_concurrent`` and 100%
      cash (no leverage).

Exits / risk-reducers are NEVER gated here — the engine routes them around the
budget entirely (the F2 bypass). PURE — reads the validated ``Limits`` +
``PortfolioConfig``, no I/O.
"""

from __future__ import annotations

from typing import Optional

from portfolio.model import Candidate, PlannedOpen
from risk.config import Limits
from risk.sizing import per_trade_dollar_risk, resolve_ri


# --------------------------------------------------------------------------- #
# Synthetic-stop band (the one new knob) + per-candidate dollar-risk
# --------------------------------------------------------------------------- #
def synthetic_stop(mark: float, atr: Optional[float], k: float, fallback_frac: float) -> float:
    """Synthetic trailing-vol stop ``mark - k*ATR`` (band falls back to a fraction).

    Used ONLY to give a stop-less weight position a comparable dollar-risk for the
    heat budget — it never triggers an exit. When ATR is missing/degenerate the
    band is ``fallback_frac * mark`` so the risk is finite and positive.
    """
    if atr is not None and atr > 0:
        return float(mark) - float(k) * float(atr)
    return float(mark) * (1.0 - float(fallback_frac))


def score_dollar_risk(
    equity: float, grade: str, limits: Limits, scalar: float = 1.0, floor: Optional[int] = None
) -> float:
    """Per-trade $ risk for a SCORE candidate = RI%·equity·exposure_scalar.

    Conviction-flexes the dial: ``resolve_ri(grade)`` maps the grade to an RI in
    ``[floor, band_high]`` and ``per_trade_dollar_risk`` reads its ``per_trade_pct``
    against CURRENT equity. The exposure scalar (regime read) modulates it.
    """
    ri = resolve_ri(grade, limits, floor)
    return per_trade_dollar_risk(float(equity), ri, limits) * float(scalar)


# --------------------------------------------------------------------------- #
# Size one candidate -> a PlannedOpen (with its heat = dollar_risk)
# --------------------------------------------------------------------------- #
def size_candidate(
    c: Candidate,
    equity: float,
    limits: Limits,
    *,
    scalar: float = 1.0,
    synthetic_k: float = 2.5,
    fallback_frac: float = 0.15,
    floor: Optional[int] = None,
    available_cash: Optional[float] = None,
) -> Optional[PlannedOpen]:
    """Size ``c`` into a :class:`PlannedOpen` (or ``None`` if it can't be sized).

    ``available_cash`` (when given) caps the notional — the book takes no leverage,
    so a position is clipped to the cash it can actually fund. Returns ``None`` for
    a degenerate stop / non-positive size / unpriced name.
    """
    if c.entry_price is None or c.entry_price <= 0:
        return None
    mark = float(c.entry_price)

    if c.kind == "score":
        rps = c.risk_per_share()
        if rps is None:
            return None
        dollar_risk = score_dollar_risk(equity, c.grade, limits, scalar=scalar, floor=floor)
        if dollar_risk <= 0:
            return None
        shares = dollar_risk / rps
        stop = float(c.stop)
        tp1_price = c.tp1_price
        hard_target = c.hard_target
        target_weight = None
    elif c.kind == "weight":
        if c.target_weight is None or c.target_weight <= 0:
            return None
        notional = float(c.target_weight) * float(equity) * float(scalar)
        if notional <= 0:
            return None
        shares = notional / mark
        stop = synthetic_stop(mark, c.atr, synthetic_k, fallback_frac)
        dollar_risk = shares * max(mark - stop, 0.0)
        tp1_price = None
        hard_target = None
        target_weight = float(c.target_weight)
    else:
        return None

    if shares <= 0:
        return None

    # No-leverage cash clip (when a cash budget is supplied).
    if available_cash is not None:
        notional = shares * mark
        if notional > available_cash:
            if available_cash <= 0:
                return None
            shares = available_cash / mark
            # rescale the heat with the clipped size.
            if c.kind == "score":
                dollar_risk = shares * c.risk_per_share()
            else:
                dollar_risk = shares * max(mark - stop, 0.0)
            if shares <= 0:
                return None

    return PlannedOpen(
        sleeve=c.sleeve,
        symbol=c.symbol,
        kind=c.kind,
        side=c.side,
        shares=float(shares),
        entry_price=mark,
        grade=c.grade,
        family=c.family,
        dollar_risk=float(dollar_risk),
        stop=float(stop),
        atr=c.atr,
        tp1_price=tp1_price,
        hard_target=hard_target,
        target_weight=target_weight,
    )


# --------------------------------------------------------------------------- #
# Kronos veto (the "filter weak signals" overlay; off unless use_kronos)
# --------------------------------------------------------------------------- #
def kronos_veto(candidate, kronos_cfg) -> Optional[str]:
    """Return a veto reason if Kronos rejects ``candidate``, else None.

    Off unless ``kronos_cfg.use_kronos`` AND the candidate carries a ``forecast``.
    Vetoes a candidate whose forecast ``exp_return`` is negative (when
    ``veto_negative_return``) or whose ``downside_cvar`` exceeds
    ``max_downside_cvar`` — the "filter weak strategy signals" use of the overlay.
    A candidate with no forecast is never vetoed (the engine stays runnable without
    a forecast table).
    """
    if not getattr(kronos_cfg, "use_kronos", False):
        return None
    f = getattr(candidate, "forecast", None)
    if not f:
        return None
    if kronos_cfg.veto_negative_return:
        er = f.get("exp_return")
        if er is not None and float(er) < 0:
            return f"kronos veto: exp_return {float(er):+.4f} < 0"
    cap = kronos_cfg.max_downside_cvar
    if cap is not None:
        dc = f.get("downside_cvar")
        if dc is not None and float(dc) > float(cap):
            return f"kronos veto: downside_cvar {float(dc):.4f} > {float(cap):.4f}"
    return None


# --------------------------------------------------------------------------- #
# Always-on program-abort + dial halts (#5)
# --------------------------------------------------------------------------- #
def program_abort_halt(equity: float, book, limits: Limits) -> Optional[str]:
    """Return a program-abort reason if equity breached a drawdown limit, else None.

    −35% from the all-time peak → halt + manual restart; −20% inside the calendar
    month → mandatory review. Both suppress new opens (closes still run). These are
    insurance against bugs/death-spirals, NOT risk appetite, and are never on the
    dial.
    """
    pa = limits.program_abort
    peak = float(getattr(book, "peak_equity", 0.0) or 0.0)
    if peak > 0:
        dd = equity / peak - 1.0
        if dd <= -float(pa.peak_halt_drawdown_pct) / 100.0:
            return (f"program_abort: -{abs(dd):.1%} from peak "
                    f"(>= {pa.peak_halt_drawdown_pct:.0f}% peak-halt)")
    ms = float(getattr(book, "month_start_equity", 0.0) or 0.0)
    if ms > 0:
        mdd = equity / ms - 1.0
        if mdd <= -float(pa.monthly_review_drawdown_pct) / 100.0:
            return (f"program_abort: -{abs(mdd):.1%} in month "
                    f"(>= {pa.monthly_review_drawdown_pct:.0f}% monthly-review)")
    return None


def dial_halt(equity: float, book, limits: Limits, ri: Optional[int] = None) -> Optional[str]:
    """Return a daily/weekly halt reason if the RI loss limit was hit, else None.

    Reads the BOOK RI row (``limits.default_ri`` by default): a same-day drop vs
    the prior close ≥ ``daily_halt_pct`` halts the day; a drop vs the week-start
    NAV ≥ ``weekly_halt_pct`` halts the week. Opens are suppressed; closes run.
    """
    ri = limits.default_ri if ri is None else ri
    row = limits.level(ri)
    prev = float(getattr(book, "prev_nav", 0.0) or 0.0)
    if prev > 0:
        day_ret = equity / prev - 1.0
        if day_ret <= -float(row.daily_halt_pct) / 100.0:
            return f"daily_halt: -{abs(day_ret):.1%} vs prior close (>= {row.daily_halt_pct:.1f}%)"
    wk = float(getattr(book, "week_start_equity", 0.0) or 0.0)
    if wk > 0:
        wk_ret = equity / wk - 1.0
        if wk_ret <= -float(row.weekly_halt_pct) / 100.0:
            return f"weekly_halt: -{abs(wk_ret):.1%} vs week start (>= {row.weekly_halt_pct:.1f}%)"
    return None


# --------------------------------------------------------------------------- #
# Heat-admission walk (#5) — the book budget gate
# --------------------------------------------------------------------------- #
def admit_candidates(
    ranked: list,
    equity: float,
    limits: Limits,
    pcfg,
    *,
    scalar: float = 1.0,
    sleeve_scalars: Optional[dict] = None,
    current_book_heat: float = 0.0,
    current_family_heat: Optional[dict] = None,
    available_cash: float = 0.0,
    n_held: int = 0,
    floor: Optional[int] = None,
) -> tuple:
    """Walk ranked candidates admitting until a book limit binds → (opens, rejected).

    In priority order, size each candidate and admit it while ALL hold:
      * book heat: ``running_heat + cand_heat ≤ portfolio_heat_pct * equity``;
      * family heat: ``family_running + cand_heat ≤ family_caps[family] * equity``
        (a family without a cap is bounded only by the book cap);
      * concurrency: ``n_held + n_admitted < max_concurrent`` (the RI row);
      * cash: the cumulative admitted notional stays within available cash
        (no leverage) — each open is also individually cash-clipped.

    A candidate that fails a binding limit is rejected (with the binding reason)
    and the walk CONTINUES (a smaller later candidate may still fit under heat —
    but once concurrency is full the walk stops). Returns
    ``(list[PlannedOpen], list[(Candidate, reason)])``.
    """
    ri = limits.default_ri if floor is None else floor
    row = limits.level(limits.default_ri)
    heat_cap = float(row.portfolio_heat_pct) / 100.0 * float(equity)
    max_concurrent = int(row.max_concurrent)
    family_caps = dict(pcfg.family_caps or {})
    ss = pcfg.synthetic_stop

    book_heat = float(current_book_heat)
    fam_heat = dict(current_family_heat or {})
    cash_left = float(available_cash)

    opens: list = []
    rejected: list = []
    n_open = int(n_held)

    sleeve_scalars = dict(sleeve_scalars or {})
    for c in ranked:
        if n_open >= max_concurrent:
            rejected.append((c, f"concurrency cap {max_concurrent} reached"))
            continue
        eff_scalar = float(sleeve_scalars.get(c.sleeve, scalar))
        planned = size_candidate(
            c, equity, limits,
            scalar=eff_scalar,
            synthetic_k=ss.k,
            fallback_frac=ss.fallback_band_frac,
            floor=floor,
            available_cash=cash_left,
        )
        if planned is None or planned.shares <= 0:
            rejected.append((c, "unsizable (degenerate stop / no cash / no price)"))
            continue

        cand_heat = float(planned.dollar_risk)
        # book heat cap
        if book_heat + cand_heat > heat_cap + 1e-9:
            rejected.append((c, f"book heat cap {heat_cap:.2f} would be exceeded"))
            continue
        # per-family cap
        fam = planned.family
        if fam in family_caps:
            cap = float(family_caps[fam]) * float(equity)
            if fam_heat.get(fam, 0.0) + cand_heat > cap + 1e-9:
                rejected.append((c, f"family '{fam}' heat cap {cap:.2f} would be exceeded"))
                continue

        # admit
        opens.append(planned)
        book_heat += cand_heat
        fam_heat[fam] = fam_heat.get(fam, 0.0) + cand_heat
        cash_left -= planned.shares * planned.entry_price
        n_open += 1

    return opens, rejected
