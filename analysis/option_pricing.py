#!/usr/bin/env python3
"""option_pricing.py -- stdlib Black-Scholes engine for the Asymmetric campaign.

WHY THIS EXISTS
---------------
memory sec 3b parks spreads with an explicit precondition: "the backtester cannot
evaluate it -- it models the UNDERLYING only (no delta/theta/IV/spread), so it
literally cannot compare a debit spread to a long call."

`option_model.py` closed half of that by reading SAVED option bars. But saved bars
only exist for contracts we happened to record. To evaluate spread WIDTHS and
strikes across accumulated sessions we must price legs we never observed. That is
what this module does: calibrate an implied vol from one observed quote, then
price that contract anywhere along the underlying path.

HARD RULES (AGENTS.md)
----------------------
- Python 3.9 STDLIB ONLY (math, datetime, zoneinfo). No third-party deps.
- No network. No brokerage access. Pure arithmetic on saved data.

MODEL LIMITS -- READ BEFORE TRUSTING A NUMBER
---------------------------------------------
1. Constant vol per leg. Each contract carries its OWN calibrated IV, so static
   skew across strikes is captured; smile EVOLUTION and IV crush are NOT.
2. Calendar-time decay (standard BS). Real options decay closer to trading-time,
   so intraday theta here is slightly overstated on weekends/overnight.
3. European pricing on American ETF options. For short-dated near-ATM contracts
   with no dividend in the hold window the error is small; it is not zero.
4. Mid-price world. Spreads/slippage are applied by the caller, not here.
Always run `validate_against_bars()` before believing a modeled leg.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math

try:
    from zoneinfo import ZoneInfo   # stdlib 3.9+
    _ET = ZoneInfo("America/New_York")
except Exception:                    # pragma: no cover - fallback, fixed EDT
    _ET = None

SECONDS_PER_YEAR = 365.0 * 24 * 3600
DEFAULT_RATE = 0.04          # risk-free proxy; impact is tiny at <=21 DTE
MIN_T = 1.0 / SECONDS_PER_YEAR


# --------------------------------------------------------------------------
# time
# --------------------------------------------------------------------------
def expiry_epoch(expiry_date: str, hour_et: int = 16) -> int:
    """Epoch seconds of expiration (16:00 ET on the expiry date by default).
    Uses zoneinfo so DST is handled; falls back to a fixed -04:00 (EDT)."""
    y, m, d = (int(x) for x in expiry_date.split("-"))
    if _ET is not None:
        dt = datetime(y, m, d, hour_et, 0, tzinfo=_ET)
    else:                            # pragma: no cover
        from datetime import timezone
        dt = datetime(y, m, d, hour_et, 0, tzinfo=timezone(timedelta(hours=-4)))
    return int(dt.timestamp())


def year_fraction(now_epoch: int, exp_epoch: int) -> float:
    """Time to expiry in years, floored so pricing never divides by zero."""
    return max((exp_epoch - now_epoch) / SECONDS_PER_YEAR, MIN_T)


# --------------------------------------------------------------------------
# Black-Scholes
# --------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S, K, T, r, sigma):
    v = sigma * math.sqrt(T)
    if v <= 0:
        v = 1e-12
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return d1, d1 - v


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             right: str) -> float:
    """European Black-Scholes price. right = 'C' or 'P'."""
    if S <= 0 or K <= 0:
        return 0.0
    T = max(T, MIN_T)
    sigma = max(sigma, 1e-9)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)
    if right.upper().startswith("C"):
        val = S * norm_cdf(d1) - K * disc * norm_cdf(d2)
    else:
        val = K * disc * norm_cdf(-d2) - S * norm_cdf(-d1)
    return max(val, 0.0)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              right: str) -> Dict[str, float]:
    """delta, gamma, vega (per 1 vol point), theta (per calendar day)."""
    T = max(T, MIN_T)
    sigma = max(sigma, 1e-9)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)
    sqrtT = math.sqrt(T)
    call = right.upper().startswith("C")
    delta = norm_cdf(d1) if call else norm_cdf(d1) - 1.0
    gamma = norm_pdf(d1) / (S * sigma * sqrtT)
    vega = S * norm_pdf(d1) * sqrtT / 100.0
    term = -(S * norm_pdf(d1) * sigma) / (2.0 * sqrtT)
    if call:
        theta = (term - r * K * disc * norm_cdf(d2)) / 365.0
    else:
        theta = (term + r * K * disc * norm_cdf(-d2)) / 365.0
    return {"delta": round(delta, 6), "gamma": round(gamma, 6),
            "vega": round(vega, 6), "theta": round(theta, 6)}


def intrinsic(S: float, K: float, right: str) -> float:
    return max(S - K, 0.0) if right.upper().startswith("C") else max(K - S, 0.0)


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                right: str, lo: float = 1e-4, hi: float = 5.0,
                tol: float = 1e-7, max_iter: int = 200) -> Optional[float]:
    """Solve IV by bisection (robust; no derivative blowups near expiry).
    Returns None when the price is unattainable (below intrinsic / above cap)."""
    if price is None or price <= 0:
        return None
    if price < intrinsic(S, K, right) - 1e-9:
        return None                      # arbitrage / stale mark, refuse to fit
    p_hi = bs_price(S, K, T, r, hi, right)
    if price > p_hi:
        return None
    p_lo = bs_price(S, K, T, r, lo, right)
    if price < p_lo:
        return lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pm = bs_price(S, K, T, r, mid, right)
        if abs(pm - price) < tol:
            return mid
        if pm < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# calibrated contract
# --------------------------------------------------------------------------
@dataclass
class PricedLeg:
    """A contract with an IV calibrated from one observed mark, so it can be
    repriced anywhere along the path."""
    label: str
    right: str
    strike: float
    exp_epoch: int
    iv: float
    qty: int = 1                 # +1 long, -1 short
    rate: float = DEFAULT_RATE

    def value(self, S: float, now_epoch: int) -> float:
        T = year_fraction(now_epoch, self.exp_epoch)
        return bs_price(S, self.strike, T, self.rate, self.iv, self.right)

    def greeks(self, S: float, now_epoch: int) -> Dict[str, float]:
        T = year_fraction(now_epoch, self.exp_epoch)
        return bs_greeks(S, self.strike, T, self.rate, self.iv, self.right)


def calibrate_leg(label: str, right: str, strike: float, expiry_date: str,
                  observed_mark: float, S: float, now_epoch: int,
                  qty: int = 1, rate: float = DEFAULT_RATE) -> Optional[PricedLeg]:
    """Fit IV to one observed mark. Returns None if the mark cannot be fit
    (never silently substitutes a guessed vol)."""
    exp_e = expiry_epoch(expiry_date)
    T = year_fraction(now_epoch, exp_e)
    iv = implied_vol(observed_mark, S, strike, T, rate, right)
    if iv is None:
        return None
    return PricedLeg(label, right, strike, exp_e, iv, qty, rate)


def structure_value(legs: List[PricedLeg], S: float, now_epoch: int) -> float:
    """Net value of a multi-leg structure (long legs +, short legs -)."""
    return round(sum(l.qty * l.value(S, now_epoch) for l in legs), 6)


def structure_greeks(legs: List[PricedLeg], S: float,
                     now_epoch: int) -> Dict[str, float]:
    out = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for l in legs:
        g = l.greeks(S, now_epoch)
        for k in out:
            out[k] += l.qty * g[k]
    return {k: round(v, 6) for k, v in out.items()}


# --------------------------------------------------------------------------
# model validation against observed bars
# --------------------------------------------------------------------------
def validate_against_bars(leg: PricedLeg, underlying_bars, option_bars,
                          start_epoch: int) -> Dict:
    """Compare modeled prices to ACTUALLY OBSERVED option bars over the same
    path. This is the honesty check: if mean abs error is large, do not trust
    modeled legs for strikes we never recorded.

    bars are any objects with .epoch and .close.
    """
    u_by_epoch = {b.epoch: b.close for b in underlying_bars}
    errs, rows = [], []
    for ob in option_bars:
        if ob.epoch < start_epoch or ob.epoch not in u_by_epoch:
            continue
        S = u_by_epoch[ob.epoch]
        model = leg.value(S, ob.epoch)
        err = model - ob.close
        errs.append(abs(err))
        rows.append({"epoch": ob.epoch, "S": S,
                     "observed": round(ob.close, 4),
                     "modeled": round(model, 4), "error": round(err, 4)})
    if not errs:
        return {"n": 0, "mae": None, "max_abs_error": None, "rows": [],
                "verdict": "no overlapping bars"}
    mae = sum(errs) / len(errs)
    return {"n": len(errs), "mae": round(mae, 4),
            "max_abs_error": round(max(errs), 4),
            "iv_used": round(leg.iv, 4), "rows": rows,
            "verdict": ("good (<=$0.03 MAE)" if mae <= 0.03 else
                        "acceptable (<=$0.06 MAE)" if mae <= 0.06 else
                        "POOR -- do not trust modeled legs")}


if __name__ == "__main__":
    import test_option_pricing
    test_option_pricing.run()
