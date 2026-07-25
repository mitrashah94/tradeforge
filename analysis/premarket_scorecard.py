#!/usr/bin/env python3
"""premarket_scorecard.py -- implements strategy.md sec 9 (premarket scorecard).

WHAT THIS IS -- AND IS NOT
--------------------------
This TOOLS AN EXISTING RULE. strategy.md sec 9 already defines an eight-point
premarket scorecard that advances a primary + backup before 08:45 CT. It has
never been executable, so it was done by feel. This module makes it scored,
logged and auditable. **No strategy change. strategy.md is not modified.**

It selects **WHAT TO WATCH, never WHAT TO TRADE.** A shortlisted ticker still
has to produce a QUALIFIED from the live engine before anything is tradable.
The scorecard cannot create an entry, and a high score is not a signal.

WHY IT BECAME LOAD-BEARING (2026-07-24)
---------------------------------------
The Pine `qualifiedToday` lockout is a plain `var bool`, so it is PER SYMBOL --
it silences the other three tracks on that chart only. With QUALIFIED alerts now
armed on all five universe tickers, up to FIVE QUALIFIED can land in one day,
while strategy.md sec 1 permits ONE live trade per day. Something has to decide
which one, in advance, or the choice gets made by whichever chart looks most
exciting at 09:10 -- the exact failure mode of 2026-07-13.

RULES ENCODED (user decisions, 2026-07-24)
------------------------------------------
- **Catalysts are EXCLUSION-ONLY.** News/earnings/macro can remove a ticker and
  can never promote one. Prevents "catalyst" becoming a discretion backdoor.
- **Shortlist-only.** Only primary or backup are tradable. An off-list QUALIFIED
  is logged and watched, never taken (sec 9: "No rotating after the window opens").
- The shortlist is LOCKED with a timestamp. Re-running after 08:45 warns.

HARD RULES (AGENTS.md): Python 3.9 stdlib only. No network. No brokerage access.
Live data is fetched OUT OF BAND by the orchestrator via read-only tools and
saved as `premarket_snapshot.json`; this module only reads and scores it.

HONEST LIMITS
-------------
- `room_r` here is a PROXY (distance to next obstacle / an ATR-fraction stop).
  The real room_r comes from the engine at QUALIFIED time and can differ.
- "clear daily S/R" is scored from levels the human marked premarket; the module
  cannot see a chart.
- Scoring is a prior about where to LOOK. It has no demonstrated edge and must
  not be treated as one.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import os

SCORE_MAX = 8
ENTRY_WINDOW_CT = ("08:45", "10:30")     # strategy.md sec 3
NEWS_BUFFER_MIN = 10                     # sec 3


# --------------------------------------------------------------------------
# thresholds (documented, not magic)
# --------------------------------------------------------------------------
@dataclass
class Thresholds:
    min_gap_pct: float = 0.15            # "clear gap/direction"
    min_pm_rvol: float = 1.2             # "elevated premarket volume" (matches sec 4 RVOL)
    near_level_atr_frac: float = 0.50    # "near an important level"
    stop_proxy_atr_frac: float = 0.30    # PROXY risk for premarket room_r only
    min_room_r: float = 3.0              # sec 9 ">=3R open space"
    max_option_spread: float = 0.05      # sec 5 "wide spread"
    min_open_interest: int = 500
    min_option_volume: int = 100
    max_premium_usd: Optional[float] = None   # settled cash; None = skip the check


@dataclass
class Criterion:
    name: str
    earned: bool
    detail: str


@dataclass
class TickerScore:
    ticker: str
    score: int
    criteria: List[Criterion] = field(default_factory=list)
    excluded: bool = False
    exclusions: List[str] = field(default_factory=list)
    bias: Optional[str] = None           # CALL / PUT / None
    room_r_proxy: Optional[float] = None

    def as_dict(self) -> Dict:
        return {"ticker": self.ticker, "score": self.score,
                "max_score": SCORE_MAX, "bias": self.bias,
                "excluded": self.excluded, "exclusions": self.exclusions,
                "room_r_proxy": self.room_r_proxy,
                "criteria": [{"name": c.name, "earned": c.earned,
                              "detail": c.detail} for c in self.criteria]}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _hhmm_in_window(t: str, lo: str, hi: str) -> bool:
    return lo <= t <= hi


def _pct(a: float, b: float) -> float:
    return 0.0 if not b else (a - b) / b * 100.0


def _direction(gap_pct: float, thr: float) -> Optional[str]:
    if gap_pct >= thr:
        return "CALL"
    if gap_pct <= -thr:
        return "PUT"
    return None


# --------------------------------------------------------------------------
# exclusions (sec 2 / sec 3 / sec 5 / sec 7) -- catalysts can ONLY remove
# --------------------------------------------------------------------------
def hard_exclusions(t: Dict, thr: Thresholds) -> List[str]:
    """Only conditions strategy.md states as EXCLUSIONS belong here.

    Deliberately NOT here: a macro release inside the entry window. sec 3 defines
    a 10-minute news BUFFER around a release, not a day-long ban -- so a macro
    event costs criterion 8 and produces a blackout window (see blackout_windows)
    rather than removing the ticker. Encoding it as an exclusion would be
    stricter than the written plan, and the plan is authoritative.
    """
    out: List[str] = []
    if t.get("earnings_in_hold_window"):
        out.append("earnings_play (sec 2): {}".format(
            t.get("earnings_note", "reports inside the hold window")))
    opt = t.get("option") or {}
    prem = opt.get("premium_usd")
    if thr.max_premium_usd is not None and prem is not None:
        if prem > thr.max_premium_usd:
            out.append("premium_unaffordable (sec 7): ${:.0f} > ${:.0f} cash"
                       .format(prem, thr.max_premium_usd))
    if opt.get("only_far_otm_affordable"):
        out.append("affordable_contract_only_far_otm (sec 5)")
    if t.get("manual_exclusion"):
        out.append("manual: {}".format(t["manual_exclusion"]))
    return out


# --------------------------------------------------------------------------
# the eight criteria (strategy.md sec 9)
# --------------------------------------------------------------------------
def score_ticker(ticker: str, t: Dict, market: Dict,
                 thr: Thresholds = Thresholds()) -> TickerScore:
    crits: List[Criterion] = []
    atr = t.get("atr14") or 0.0
    pc = t.get("prior_close") or 0.0
    last = t.get("premarket_last") or pc
    gap = _pct(last, pc)
    bias = _direction(gap, thr.min_gap_pct)

    # 1. clear gap / direction
    crits.append(Criterion(
        "clear_gap_direction", bias is not None,
        "gap {:+.2f}% vs {:.2f}% threshold -> {}".format(
            gap, thr.min_gap_pct, bias or "no clear bias")))

    # 2. elevated premarket volume
    pv, pav = t.get("premarket_volume"), t.get("premarket_volume_avg")
    pm_rvol = (pv / pav) if (pv and pav) else None
    crits.append(Criterion(
        "elevated_premarket_volume",
        bool(pm_rvol and pm_rvol >= thr.min_pm_rvol),
        "pm RVOL {} vs {}".format(
            "n/a" if pm_rvol is None else "{:.2f}x".format(pm_rvol),
            thr.min_pm_rvol)))

    # 3. near an important level (the seven computed levels)
    levels = t.get("computed_levels") or {}
    near, near_name = None, None
    for nm, price in levels.items():
        if price is None:
            continue
        d = abs(last - price)
        if near is None or d < near:
            near, near_name = d, nm
    ok3 = bool(atr and near is not None and near <= thr.near_level_atr_frac * atr)
    crits.append(Criterion(
        "near_important_level", ok3,
        "nearest {} at {:.2f} ATR away".format(
            near_name or "n/a", (near / atr) if (atr and near is not None) else float("nan"))))

    # 4. clear daily S/R marked premarket
    daily = t.get("daily_levels") or []
    crits.append(Criterion(
        "clear_daily_sr", len(daily) > 0,
        "{} daily level(s) marked".format(len(daily))))

    # 5. strong option liquidity
    opt = t.get("option") or {}
    spr, oi, ovol = opt.get("spread"), opt.get("open_interest"), opt.get("volume")
    ok5 = bool(spr is not None and spr <= thr.max_option_spread
               and (oi or 0) >= thr.min_open_interest
               and (ovol or 0) >= thr.min_option_volume)
    crits.append(Criterion(
        "strong_option_liquidity", ok5,
        "spread ${} OI {} vol {}".format(spr, oi, ovol)))

    # 6. >= 3R open space (PROXY -- real room_r comes from the engine)
    room_r = None
    nxt = t.get("next_obstacle")
    risk = thr.stop_proxy_atr_frac * atr if atr else None
    if nxt is not None and risk:
        room_r = round(abs(nxt - last) / risk, 2)
    crits.append(Criterion(
        "room_3r_proxy", bool(room_r is not None and room_r >= thr.min_room_r),
        "proxy room {}R to {} (stop proxy {:.2f})".format(
            room_r, nxt, risk or 0.0)))

    # 7. SPY/QQQ alignment
    mkt = [market.get("SPY"), market.get("QQQ")]
    ok7 = bool(bias and all(m == bias for m in mkt if m))
    crits.append(Criterion(
        "spy_qqq_alignment", ok7,
        "SPY {} / QQQ {} vs bias {}".format(
            market.get("SPY"), market.get("QQQ"), bias)))

    # 8. no imminent event risk
    evs = t.get("macro_events_ct") or []
    ok8 = (not t.get("earnings_in_hold_window")) and not any(
        _hhmm_in_window(e.get("time_ct", ""), ENTRY_WINDOW_CT[0], ENTRY_WINDOW_CT[1])
        for e in evs)
    crits.append(Criterion(
        "no_imminent_event_risk", ok8,
        "earnings={} events={}".format(
            bool(t.get("earnings_in_hold_window")),
            [e.get("name") for e in evs] or "none")))

    ts = TickerScore(ticker=ticker, score=sum(1 for c in crits if c.earned),
                     criteria=crits, bias=bias, room_r_proxy=room_r)
    ts.exclusions = hard_exclusions(t, thr)
    ts.excluded = bool(ts.exclusions)
    return ts


# --------------------------------------------------------------------------
# shortlist
# --------------------------------------------------------------------------
def _minus_minutes(hhmm: str, mins: int) -> str:
    try:
        h, m = (int(x) for x in hhmm.split(":"))
    except Exception:
        return hhmm
    tot = max(0, h * 60 + m - mins)
    return "{:02d}:{:02d}".format(tot // 60, tot % 60)


def blackout_windows(snapshot: Dict) -> List[Dict]:
    """strategy.md sec 3: no entry within 10 minutes before a scheduled release.
    Returns de-duplicated [buffer_start, event_time] windows for the day."""
    seen, out = set(), []
    for tk, t in sorted((snapshot.get("tickers") or {}).items()):
        for ev in (t.get("macro_events_ct") or []):
            when, name = ev.get("time_ct", ""), ev.get("name", "event")
            key = (when, name)
            if not when or key in seen:
                continue
            seen.add(key)
            out.append({
                "event": name, "time_ct": when,
                "no_entry_from_ct": _minus_minutes(when, NEWS_BUFFER_MIN),
                "no_entry_to_ct": when,
                "inside_entry_window": _hhmm_in_window(
                    when, ENTRY_WINDOW_CT[0], ENTRY_WINDOW_CT[1]),
            })
    return sorted(out, key=lambda w: w["time_ct"])


def build_card(snapshot: Dict, thr: Thresholds = Thresholds()) -> Dict:
    """Score every ticker, apply exclusions, lock primary + backup.
    Deterministic ordering: score desc, then option OI desc, then ticker asc."""
    market = snapshot.get("market_bias") or {}
    tickers = snapshot.get("tickers") or {}
    scored = [score_ticker(k, v, market, thr) for k, v in sorted(tickers.items())]

    def rank_key(s: TickerScore):
        oi = ((tickers.get(s.ticker, {}).get("option") or {}).get("open_interest") or 0)
        return (-s.score, -oi, s.ticker)

    eligible = sorted([s for s in scored if not s.excluded], key=rank_key)
    primary = eligible[0].ticker if eligible else None
    backup = eligible[1].ticker if len(eligible) > 1 else None

    return {
        "session_date_ct": snapshot.get("session_date_ct"),
        "captured_ct": snapshot.get("captured_ct"),
        "locked": True,
        "rule": ("SHORTLIST-ONLY: only primary or backup are tradable. "
                 "An off-list QUALIFIED is logged and watched, never taken "
                 "(strategy.md sec 9). This card selects WHAT TO WATCH; a "
                 "QUALIFIED from the live engine is still required."),
        "catalyst_policy": "EXCLUSION-ONLY (can remove a ticker, never promote one)",
        "primary": primary,
        "backup": backup,
        "blackout_windows": blackout_windows(snapshot),
        "context_only": [s.ticker for s in scored if s.excluded],
        "scores": [s.as_dict() for s in scored],
        "thresholds": thr.__dict__.copy(),
        "caveat": ("room_r is a premarket PROXY; scoring is a prior about where "
                   "to look and has no demonstrated edge."),
    }


def evaluate_qualified(card: Dict, ticker: str) -> Dict:
    """Enforcement hook: given the LOCKED card, is this QUALIFIED tradable?
    Makes sec 9's shortlist rule executable instead of aspirational."""
    if ticker == card.get("primary"):
        return {"tradable": True, "role": "primary",
                "note": "pre-committed primary; proceed to the contract checklist"}
    if ticker == card.get("backup"):
        return {"tradable": True, "role": "backup",
                "note": "pre-committed backup; proceed to the contract checklist"}
    excluded = ticker in (card.get("context_only") or [])
    return {"tradable": False, "role": "off_list",
            "note": ("EXCLUDED premarket ({}) -- do not trade"
                     .format("; ".join(
                         next((s["exclusions"] for s in card.get("scores", [])
                               if s["ticker"] == ticker), [])) or "see card")
                     if excluded else
                     "not on the pre-committed shortlist -- log and watch, "
                     "do not trade (strategy.md sec 9: no rotating after the "
                     "window opens)")}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def render(card: Dict) -> str:
    L = []
    L.append("=" * 74)
    L.append("PREMARKET SCORECARD  {}   captured {}".format(
        card.get("session_date_ct"), card.get("captured_ct")))
    L.append("strategy.md sec 9 | catalysts: {}".format(card["catalyst_policy"]))
    L.append("=" * 74)
    L.append("{:<7}{:>7}  {:<6} {}".format("ticker", "score", "bias", "status"))
    L.append("-" * 74)
    for s in card["scores"]:
        if s["ticker"] == card.get("primary"):
            status = "** PRIMARY **"
        elif s["ticker"] == card.get("backup"):
            status = "** BACKUP **"
        elif s["excluded"]:
            status = "EXCLUDED: " + "; ".join(s["exclusions"])
        else:
            status = "watch only"
        L.append("{:<7}{:>4}/{:<2}  {:<6} {}".format(
            s["ticker"], s["score"], s["max_score"], s["bias"] or "-", status))
    L.append("-" * 74)
    for s in card["scores"]:
        L.append("{}  ({}/{}){}".format(
            s["ticker"], s["score"], s["max_score"],
            "  EXCLUDED" if s["excluded"] else ""))
        for c in s["criteria"]:
            L.append("   [{}] {:<28} {}".format(
                "x" if c["earned"] else " ", c["name"], c["detail"]))
    if card.get("blackout_windows"):
        L.append("-" * 74)
        L.append("NEWS BUFFER (sec 3) -- no entry inside these windows:")
        for w in card["blackout_windows"]:
            L.append("   {}-{} CT  {}{}".format(
                w["no_entry_from_ct"], w["no_entry_to_ct"], w["event"],
                "   <-- INSIDE THE ENTRY WINDOW" if w["inside_entry_window"] else ""))
    L.append("-" * 74)
    L.append(card["rule"])
    L.append("CAVEAT: " + card["caveat"])
    L.append("NO BROKERAGE ACTION -- information only; the click is yours.")
    L.append("=" * 74)
    return "\n".join(L)


def run(snapshot_path: str, out_path: Optional[str] = None,
        thr: Thresholds = Thresholds()) -> Optional[Dict]:
    if not os.path.exists(snapshot_path):
        print("[premarket] missing snapshot:", snapshot_path)
        return None
    with open(snapshot_path) as fh:
        snap = json.load(fh)
    card = build_card(snap, thr)
    print(render(card))
    out = out_path or os.path.join(os.path.dirname(snapshot_path),
                                   "premarket_card.json")
    with open(out, "w") as fh:
        json.dump(card, fh, indent=2)
    print("[premarket] wrote", out)
    return card


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="strategy.md sec 9 premarket scorecard.")
    ap.add_argument("snapshot", nargs="?", help="premarket_snapshot.json")
    ap.add_argument("--out")
    ap.add_argument("--cash", type=float, default=None,
                    help="settled cash for the affordability exclusion")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        import test_premarket_scorecard
        test_premarket_scorecard.run()
    elif a.snapshot:
        run(a.snapshot, a.out, Thresholds(max_premium_usd=a.cash))
    else:
        ap.print_help()
