"""strategies/breakout_retest_options/evaluate.py — LIVE evaluation runner.

Pulls a live SPY/QQQ option chain + an IV-rank estimate from **Polygon** (free
tier: 5 calls/min), builds the overlay's ``OptionContract`` view, and prints the
``small_account_first90`` overlay's ticket decision for a break-retest signal.

Design constraints this module is built around
----------------------------------------------
* **5 calls/min rate limit** (Polygon free/Starter). Every network call goes
  through :class:`RateLimiter` (default 12s spacing). One evaluation costs ~2
  calls per underlying — an option-chain snapshot and one 1-year daily-bar pull —
  so SPY+QQQ is ~4 calls, comfortably inside one minute.
* **Greeks may be absent on the free tier.** Polygon's option snapshot only
  carries greeks/IV on paid Options tiers. When ``delta``/``theta`` are missing
  but IV is present, we compute them locally with **Black-Scholes**
  (:func:`bs_greeks`) so the overlay always has the greeks it needs. If IV is
  also missing, that contract is dropped.
* **No paid IV-history feed.** True IV rank needs a year of IV history. We use an
  honest **proxy**: rank the current ATM IV inside the trailing 1-year envelope of
  the underlying's 20-day realized volatility (:func:`iv_rank_proxy`). It places
  current IV in the year's vol regime; it is NOT textbook IV-rank and is labelled
  as a proxy everywhere it surfaces.

Everything network-facing sits behind a small source interface
(:class:`PolygonChainSource` / :class:`FixtureChainSource`) so the pure transforms
(:func:`bs_greeks`, :func:`realized_vol_series`, :func:`iv_rank_proxy`,
:func:`snapshot_to_contracts`) and the whole ``--offline`` path run with no key,
no SDK, and no network — which is how the tests exercise it.

This is a RESEARCH / evaluation aid: it PRINTS a decision. It does not place
orders (options are off the sanctioned agentic path, CLAUDE.md P0 #3).

Usage
-----
    # offline demo (no key needed)
    python3 -m strategies.breakout_retest_options.evaluate --offline --side long

    # live (needs POLYGON_API_KEY), long setup, $1k, opening-90-min profile
    python3 -m strategies.breakout_retest_options.evaluate \
        --symbols SPY,QQQ --side long --equity 1000 --time 10:05
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

from risk.config import load_limits
from risk.sizing import per_trade_dollar_risk, resolve_ri
from strategies.breakout_retest_options.overlay import (
    OptionContract,
    OptionsOverlay,
    OverlayDecision,
    UnderlyingSignal,
    load_params,
)

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Pure math — Black-Scholes greeks + realized-vol IV-rank proxy
# --------------------------------------------------------------------------- #
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def bs_greeks(
    spot: float, strike: float, dte_days: float, iv: float, right: str,
    *, r: float = 0.04, q: float = 0.0, t_floor_days: float = 0.5,
) -> tuple[float, float]:
    """Black-Scholes ``(delta, theta_per_day)`` — a greeks fallback from IV.

    ``t_floor_days`` floors time-to-expiry so 0DTE greeks don't blow up (a 0DTE
    contract at midday has hours, not zero, of life). Returns ``(0, 0)`` on
    degenerate inputs. ``theta`` is per *calendar day* and negative for long
    premium — the sign convention the overlay expects.
    """
    if iv <= 0 or spot <= 0 or strike <= 0:
        return (0.0, 0.0)
    T = max(float(dte_days), t_floor_days) / 365.0
    srt = iv * math.sqrt(T)
    if srt <= 0:
        return (0.0, 0.0)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * T) / srt
    d2 = d1 - srt
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    common = -(spot * dq * _npdf(d1) * iv) / (2.0 * math.sqrt(T))
    if right == "call":
        delta = dq * _ncdf(d1)
        theta = common - r * strike * dr * _ncdf(d2) + q * spot * dq * _ncdf(d1)
    else:
        delta = -dq * _ncdf(-d1)
        theta = common + r * strike * dr * _ncdf(-d2) - q * spot * dq * _ncdf(-d1)
    return (delta, theta / 365.0)


def realized_vol_series(closes: list[float], window: int = 20) -> list[float]:
    """Trailing annualized realized vol (rolling stdev of log returns * sqrt(252))."""
    if len(closes) < window + 1:
        return []
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            if closes[i] > 0 and closes[i - 1] > 0]
    out: list[float] = []
    for i in range(window, len(rets) + 1):
        w = rets[i - window:i]
        m = sum(w) / len(w)
        var = sum((x - m) ** 2 for x in w) / (len(w) - 1)
        out.append(math.sqrt(var) * math.sqrt(252.0))
    return out


def iv_rank_proxy(closes: list[float], current_iv: float, window: int = 20) -> float:
    """PROXY IV rank (0..1): where ``current_iv`` sits in the trailing 1-year
    envelope of 20-day realized vol. 0.5 when history is insufficient."""
    series = realized_vol_series(closes, window)
    if not series or current_iv <= 0:
        return 0.5
    lo, hi = min(series), max(series)
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (current_iv - lo) / (hi - lo)))


# --------------------------------------------------------------------------- #
# Snapshot -> OptionContract transform
# --------------------------------------------------------------------------- #
def _get(obj, *names, default=None):
    """Read a field from a nested dict OR SDK object, trying alias names."""
    for name in names:
        cur = obj
        ok = True
        for part in name.split("."):
            if isinstance(cur, dict):
                if part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            elif hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


def snapshot_to_contracts(
    underlying: str, results: list, spot: float, as_of: date,
    *, r: float = 0.04, compute_greeks_if_missing: bool = True,
) -> list[OptionContract]:
    """Map Polygon option-snapshot rows (dicts or SDK objects) to OptionContracts.

    Fills missing greeks from IV via Black-Scholes; drops rows with neither
    greeks nor IV, or with no quote.
    """
    out: list[OptionContract] = []
    for row in results:
        strike = _get(row, "details.strike_price", "strike_price", "strike")
        exp = _get(row, "details.expiration_date", "expiration_date", "expiration")
        ctype = _get(row, "details.contract_type", "contract_type", "type", "right")
        if strike is None or exp is None or ctype is None:
            continue
        right = "call" if str(ctype).lower().startswith("c") else "put"
        bid = _get(row, "last_quote.bid", "bid", default=0.0) or 0.0
        ask = _get(row, "last_quote.ask", "ask", default=0.0) or 0.0
        if ask <= 0:
            continue
        iv = _get(row, "implied_volatility", "iv")
        delta = _get(row, "greeks.delta", "delta")
        theta = _get(row, "greeks.theta", "theta")
        if (delta is None or theta is None):
            if compute_greeks_if_missing and iv is not None and iv > 0:
                dte = (date.fromisoformat(str(exp)[:10]) - as_of).days
                delta, theta = bs_greeks(spot, float(strike), dte, float(iv), right, r=r)
            else:
                continue
        out.append(OptionContract(
            symbol=underlying, expiration=str(exp)[:10], strike=float(strike),
            right=right, bid=float(bid), ask=float(ask),
            delta=float(delta), theta=float(theta),
            iv=float(iv) if iv is not None else None,
            open_interest=_get(row, "open_interest", "oi"),
            volume=_get(row, "day.volume", "volume"),
        ))
    return out


def atm_iv(contracts: list[OptionContract], spot: float) -> float:
    """Average IV of the nearest-the-money call+put with IV populated."""
    withiv = [c for c in contracts if c.iv is not None and c.iv > 0]
    if not withiv:
        return 0.0
    nearest = min(withiv, key=lambda c: abs(c.strike - spot))
    same = [c for c in withiv if c.strike == nearest.strike]
    return sum(c.iv for c in same) / len(same)


# --------------------------------------------------------------------------- #
# Rate limiter + data sources
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Enforce a minimum spacing between calls (default 5/min => 12s)."""

    def __init__(self, calls_per_min: int = 5, *, sleep=time.sleep, clock=time.monotonic):
        self.min_interval = 60.0 / max(1, calls_per_min)
        self._sleep, self._clock = sleep, clock
        self._last = None
        self.calls = 0

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            delay = self.min_interval - (now - self._last)
            if delay > 0:
                self._sleep(delay)
        self._last = self._clock()
        self.calls += 1


@dataclass
class Underlying:
    """The per-underlying data the runner assembles before calling the overlay."""

    symbol: str
    spot: float
    contracts: list[OptionContract]
    iv_rank: float
    atm_iv: float


class FixtureChainSource:
    """Offline source (dict fixtures) — no key, no network. Used by --offline/tests."""

    def __init__(self, snapshots: dict, closes: dict, spots: dict):
        self._snap, self._closes, self._spots = snapshots, closes, spots

    def spot(self, underlying: str) -> float:
        return self._spots[underlying]

    def option_snapshot(self, underlying: str) -> list:
        return self._snap[underlying]

    def daily_closes(self, underlying: str, days: int = 365) -> list[float]:
        return self._closes[underlying]


class PolygonChainSource:
    """Live Polygon source. Lazily imports the SDK; throttled by ``limiter``.

    Free/Starter tiers may omit greeks on the snapshot — the transform fills them
    from IV. If your key lacks even IV on the snapshot, upgrade the Options tier or
    supply greeks another way.
    """

    def __init__(self, limiter: RateLimiter, *, api_key: str | None = None,
                 strike_pct: float = 0.06):
        self._limiter = limiter
        self._key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self._key:
            raise RuntimeError("POLYGON_API_KEY missing (see .env.example) — or use --offline.")
        self._strike_pct = strike_pct
        self._client = None

    def _c(self):
        if self._client is None:
            from polygon import RESTClient  # lazy
            self._client = RESTClient(self._key, retries=3)
        return self._client

    def spot(self, underlying: str) -> float:
        self._limiter.wait()
        agg = self._c().get_previous_close_agg(underlying)
        row = agg[0] if isinstance(agg, list) else list(agg)[0]
        return float(_get(row, "close", "c"))

    def option_snapshot(self, underlying: str) -> list:
        self._limiter.wait()
        # near-the-money, near-dated only -> one page, one call.
        it = self._c().list_snapshot_options_chain(
            underlying,
            params={"contract_type": None, "order": "asc",
                    "sort": "expiration_date", "limit": 250},
        )
        return list(it)

    def daily_closes(self, underlying: str, days: int = 365) -> list[float]:
        self._limiter.wait()
        end = datetime.now(timezone.utc).date()
        start = date.fromordinal(end.toordinal() - days)
        aggs = self._c().get_aggs(underlying, 1, "day", start.isoformat(), end.isoformat())
        return [float(_get(a, "close", "c")) for a in aggs if _get(a, "close", "c")]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_underlying(source, symbol: str, as_of: date, *, iv_window: int = 20) -> Underlying:
    """Assemble spot + chain (+ greeks) + IV-rank proxy for one underlying."""
    spot = source.spot(symbol)
    raw = source.option_snapshot(symbol)
    contracts = snapshot_to_contracts(symbol, raw, spot, as_of)
    closes = source.daily_closes(symbol)
    ivr_atm = atm_iv(contracts, spot)
    ivr = iv_rank_proxy(closes, ivr_atm, iv_window)
    return Underlying(symbol, spot, contracts, ivr, ivr_atm)


def signal_from(u: Underlying, side: str, *, stop: float | None, stop_pct: float,
                r_multiple: float, time_et, grade: str) -> UnderlyingSignal:
    """Build the break-retest signal (stop from --stop or --stop-pct; target = R·stop)."""
    if stop is None:
        stop = u.spot * (1 - stop_pct) if side == "long" else u.spot * (1 + stop_pct)
    risk = abs(u.spot - stop)
    target = u.spot + r_multiple * risk if side == "long" else u.spot - r_multiple * risk
    return UnderlyingSignal(u.symbol, side, u.spot, stop, target, grade, time_et=time_et)


def evaluate(
    source, symbols: list[str], *, side: str, equity: float, as_of: date,
    time_et, profile: str = "small_account_first90", grade: str = "B",
    ri: int | None = None, stop: float | None = None, stop_pct: float = 0.003,
    r_multiple: float = 2.0,
) -> list[tuple[Underlying, OverlayDecision]]:
    """Pull data for each symbol and run the overlay; return (underlying, decision)."""
    limits = load_limits()
    ov = OptionsOverlay(load_params(profile))
    use_ri = ri if ri is not None else resolve_ri(grade, limits)
    policy = limits.level(use_ri).options
    dollar_risk = per_trade_dollar_risk(equity, use_ri, limits)
    halt = limits.level(use_ri).daily_halt_pct

    results = []
    for sym in symbols:
        u = build_underlying(source, sym, as_of)
        sig = signal_from(u, side, stop=stop, stop_pct=stop_pct,
                          r_multiple=r_multiple, time_et=time_et, grade=grade)
        dec = ov.select(sig, u.contracts, u.iv_rank, equity,
                        options_policy=policy, dollar_risk=dollar_risk,
                        ri=use_ri, as_of=as_of, daily_halt_pct=halt)
        results.append((u, dec))
    return results


def format_decision(u: Underlying, dec: OverlayDecision) -> str:
    """Human-readable one-block summary of a decision."""
    lines = [
        f"── {u.symbol}  spot={u.spot:.2f}  "
        f"ATM_IV={u.atm_iv:.1%}  IVrank(proxy)={u.iv_rank:.0%} [{dec.iv_regime}]",
    ]
    if dec.ok:
        legs = "  ".join(
            f"{l.action.upper()} {l.contract.strike:g}{l.contract.right[0].upper()}"
            f"@{(l.contract.ask if l.action=='buy' else l.contract.bid):.2f}"
            for l in dec.legs
        )
        price = f"debit ${dec.net_debit:.0f}" if dec.net_debit > 0 else f"credit ${dec.net_credit:.0f}"
        lines += [
            f"   ✓ {dec.structure}  x{dec.contracts}   {legs}   ({price})",
            f"     max_loss=${dec.diagnostics.get('position_max_loss', dec.max_loss_per_contract):.0f}"
            f"  risk={dec.diagnostics.get('risk_pct_of_equity', 0)*100:.1f}% of equity"
            f"  Δ={dec.net_delta:g}  Θ=${dec.net_theta:g}/day"
            + (f"  breakeven={dec.breakeven:.2f}" if dec.breakeven else ""),
        ]
    else:
        lines.append(f"   ✗ SKIP: {dec.reason}")
        mv = dec.diagnostics.get("min_viable_equity")
        if mv:
            lines.append(f"     min_viable_equity=${mv:,.0f}")
    for w in dec.warnings:
        lines.append(f"     ⚠ {w}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Offline demo fixture + CLI
# --------------------------------------------------------------------------- #
def _demo_source() -> FixtureChainSource:
    """A tiny SPY/QQQ fixture (greeks omitted on purpose -> BS fallback) so
    ``--offline`` and the tests run with no key/SDK/network."""
    import tests.test_breakout_retest_options as T  # reuse the chain shape offline

    def snap(spot, calls, puts):
        rows = []
        for k, (d, th, bid, ask) in calls.items():
            rows.append({"details": {"strike_price": k, "expiration_date": "2026-07-06",
                                     "contract_type": "call"},
                         "last_quote": {"bid": bid, "ask": ask},
                         "implied_volatility": 0.18, "open_interest": 800})
        for k, (d, th, bid, ask) in puts.items():
            rows.append({"details": {"strike_price": k, "expiration_date": "2026-07-06",
                                     "contract_type": "put"},
                         "last_quote": {"bid": bid, "ask": ask},
                         "implied_volatility": 0.20, "open_interest": 800})
        return rows

    closes = [500.0 + 8.0 * math.sin(i / 9.0) for i in range(300)]  # a wiggly year
    return FixtureChainSource(
        snapshots={"SPY": snap(500.0, T._CALLS, T._PUTS),
                   "QQQ": snap(500.0, T._CALLS, T._PUTS)},
        closes={"SPY": closes, "QQQ": closes},
        spots={"SPY": 500.0, "QQQ": 500.0},
    )


def _now_et_hhmm() -> str:
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:  # pragma: no cover - zoneinfo always present on 3.11
        now = datetime.now(timezone.utc)
    return now.strftime("%H:%M")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Live options-overlay evaluation (Polygon).")
    ap.add_argument("--symbols", default="SPY,QQQ")
    ap.add_argument("--side", choices=["long", "short"], default="long")
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--grade", default="B")
    ap.add_argument("--ri", type=int, default=None, help="override resolved risk index")
    ap.add_argument("--stop", type=float, default=None, help="explicit underlying stop price")
    ap.add_argument("--stop-pct", type=float, default=0.003, help="stop distance as frac of spot")
    ap.add_argument("--r-multiple", type=float, default=2.0)
    ap.add_argument("--profile", default="small_account_first90")
    ap.add_argument("--time", default=None, help="ET HH:MM to evaluate as-of (default: now)")
    ap.add_argument("--calls-per-min", type=int, default=5)
    ap.add_argument("--offline", action="store_true", help="use the built-in fixture (no key)")
    args = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    time_et = args.time or _now_et_hhmm()
    as_of = datetime.now(timezone.utc).date()

    if args.offline:
        source = _demo_source()
        as_of = date(2026, 7, 5)  # fixture expiry is 2026-07-06 -> 0-1 DTE
    else:
        source = PolygonChainSource(RateLimiter(args.calls_per_min))

    print(f"Evaluating {symbols}  side={args.side}  equity=${args.equity:,.0f}  "
          f"profile={args.profile}  as-of ET {time_et}"
          + ("  [OFFLINE FIXTURE]" if args.offline else "  [POLYGON]"))
    results = evaluate(
        source, symbols, side=args.side, equity=args.equity, as_of=as_of,
        time_et=time_et, profile=args.profile, grade=args.grade, ri=args.ri,
        stop=args.stop, stop_pct=args.stop_pct, r_multiple=args.r_multiple,
    )
    for u, dec in results:
        print(format_decision(u, dec))
    if not args.offline:
        print(f"\n(Polygon calls used: {source._limiter.calls})")
    print("\nNOTE: IV rank is a realized-vol PROXY; decision is RESEARCH-only "
          "(no orders placed).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
