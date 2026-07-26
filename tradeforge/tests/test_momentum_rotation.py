"""tests/test_momentum_rotation.py — momentum_rotation signal unit tests.

Deterministic, OFFLINE (no DB): every test builds a tiny synthetic ADJUSTED
daily-close frame/panel with KNOWN behavior and asserts the EXACT output of the
pure signal functions (total-return / blended momentum / 200d SMA gate / sector
ranking / realized vol / vol scalar / GEM selection) and the weight-construction
+ growth-lever behavior of :class:`MomentumRotationStrategy`.

The point of testing the free functions directly (not just the end-to-end NAV)
is that each economic decision — which asset momentum picks, whether the trend
gate is satisfied, how the toggles re-route a slot — is verified in isolation, so
a regression in any one rule fails loudly and locally.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_rotation import (
    MomentumRotationStrategy,
    above_sma,
    blended_momentum,
    gem_select,
    load_params,
    rank_sectors,
    realized_vol,
    total_return,
    vol_scalar,
)


# --------------------------------------------------------------------------- #
# Synthetic frame helpers
# --------------------------------------------------------------------------- #
def _dates(n: int, start=date(2020, 1, 1)) -> list:
    """N consecutive Mon-Fri business dates (a daily trading calendar)."""
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _ramp(n: int, start: float, daily: float) -> list[float]:
    """A geometric price path: ``start * (1+daily)**i`` for i in 0..n-1."""
    return [start * (1.0 + daily) ** i for i in range(n)]


def _panel(series: dict[str, list[float]]) -> pd.DataFrame:
    """Wide ADJUSTED-close panel from {symbol: [closes]} on a business calendar."""
    n = len(next(iter(series.values())))
    return pd.DataFrame(series, index=_dates(n))


class _History:
    """Minimal DailyHistory stand-in: ``prices()`` returns the whole panel.

    The real engine slices to ``<= asof_date`` before the strategy sees it; for a
    unit test we pass an already-clipped panel and only need the ``prices()``
    accessor the strategy calls.
    """

    def __init__(self, panel: pd.DataFrame):
        self._panel = panel

    def prices(self, symbols=None, lookback=None):
        df = self._panel if symbols is None else self._panel[list(symbols)]
        if lookback is not None:
            df = df.iloc[-int(lookback):]
        return df.copy()


# --------------------------------------------------------------------------- #
# 1. total_return / blended_momentum
# --------------------------------------------------------------------------- #
def test_total_return_exact():
    # 21 points, +1%/day -> 20 steps of compounding over a 20-day lookback.
    px = pd.Series(_ramp(21, 100.0, 0.01))
    assert total_return(px, 20) == pytest.approx(1.01 ** 20 - 1.0)


def test_total_return_insufficient_history_is_nan():
    px = pd.Series([100.0, 101.0, 102.0])
    assert np.isnan(total_return(px, 20))      # need lookback+1 = 21 points
    assert np.isnan(total_return(px, 0))       # degenerate lookback


def test_blended_momentum_weighted_average():
    # Construct a path whose 3m/6m/12m returns are known, then blend.
    n = 260
    px = pd.Series(_ramp(n, 100.0, 0.002))
    r3 = 1.002 ** 63 - 1.0
    r6 = 1.002 ** 126 - 1.0
    r12 = 1.002 ** 252 - 1.0
    out = blended_momentum(px, (63, 126, 252), (0.0, 0.5, 0.5))
    assert out == pytest.approx((0.5 * r6 + 0.5 * r12) / 1.0)
    # equal blend across all three horizons
    out2 = blended_momentum(px, (63, 126, 252), (1.0, 1.0, 1.0))
    assert out2 == pytest.approx((r3 + r6 + r12) / 3.0)


def test_blended_momentum_renormalizes_when_horizon_missing():
    # Only ~100 points: 12m (252) horizon is unavailable -> drops out, the 3m/6m
    # weights renormalize. Result must equal the 3m/6m-only blend, not NaN.
    n = 130
    px = pd.Series(_ramp(n, 100.0, 0.001))
    r3 = 1.001 ** 63 - 1.0
    r6 = 1.001 ** 126 - 1.0
    out = blended_momentum(px, (63, 126, 252), (1.0, 1.0, 1.0))
    assert out == pytest.approx((r3 + r6) / 2.0)


# --------------------------------------------------------------------------- #
# 2. above_sma (the 200d trend gate)
# --------------------------------------------------------------------------- #
def test_above_sma_uptrend_true_downtrend_false():
    up = pd.Series(_ramp(210, 100.0, 0.01))    # rising -> last close >> SMA
    assert above_sma(up, 200) is True
    down = pd.Series(_ramp(210, 100.0, -0.01))  # falling -> last close << SMA
    assert above_sma(down, 200) is False


def test_above_sma_insufficient_history_false():
    # Fewer than `window` points -> conservative False (untrusted gate).
    px = pd.Series(_ramp(50, 100.0, 0.01))
    assert above_sma(px, 200) is False


def test_above_sma_equality_counts_as_above():
    # Flat series: last == SMA exactly -> >= -> True.
    px = pd.Series([100.0] * 210)
    assert above_sma(px, 200) is True


# --------------------------------------------------------------------------- #
# 3. rank_sectors (cross-sectional ordering)
# --------------------------------------------------------------------------- #
def test_rank_sectors_orders_by_blended_momentum():
    n = 260
    panel = _panel({
        "FAST": _ramp(n, 100.0, 0.003),   # strongest
        "MID": _ramp(n, 100.0, 0.001),
        "SLOW": _ramp(n, 100.0, 0.0001),  # weakest
    })
    ranked = rank_sectors(panel, ["FAST", "MID", "SLOW"], (63, 126, 252), (0.0, 0.5, 0.5))
    assert [s for s, _ in ranked] == ["FAST", "MID", "SLOW"]
    assert ranked[0][1] > ranked[1][1] > ranked[2][1]


def test_rank_sectors_drops_unrankable_and_missing():
    n = 260
    panel = _panel({
        "GOOD": _ramp(n, 100.0, 0.002),
        "SHORT": _ramp(n, 100.0, 0.002),
    })
    # Truncate SHORT's history so its 6m/12m horizons are NaN -> only 3m. With a
    # (0,0.5,0.5) blend (no 3m weight) SHORT has no usable horizon -> dropped.
    panel.loc[panel.index[:200], "SHORT"] = np.nan
    ranked = rank_sectors(panel, ["GOOD", "SHORT", "ABSENT"], (63, 126, 252), (0.0, 0.5, 0.5))
    assert [s for s, _ in ranked] == ["GOOD"]   # SHORT unrankable, ABSENT missing


def test_rank_sectors_tie_breaks_by_symbol():
    n = 260
    same = _ramp(n, 100.0, 0.002)
    panel = _panel({"BBB": list(same), "AAA": list(same)})
    ranked = rank_sectors(panel, ["BBB", "AAA"], (63, 126, 252), (0.0, 0.5, 0.5))
    assert [s for s, _ in ranked] == ["AAA", "BBB"]   # equal score -> alpha order


# --------------------------------------------------------------------------- #
# 4. realized_vol / vol_scalar
# --------------------------------------------------------------------------- #
def test_realized_vol_constant_return_is_zero():
    # A perfectly geometric path has identical daily returns -> zero std.
    px = pd.Series(_ramp(60, 100.0, 0.01))
    assert realized_vol(px, 20) == pytest.approx(0.0)


def test_realized_vol_annualizes():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, 5000)
    px = pd.Series(100.0 * np.cumprod(1.0 + rets))
    rv = realized_vol(px, 5000)
    # ~1% daily std -> ~0.01*sqrt(252) ≈ 0.1587 annualized.
    assert rv == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_vol_scalar_clamps_and_inverts():
    # realized == target -> scalar 1.
    assert vol_scalar(0.12, 0.12, 0.1, 1.0) == pytest.approx(1.0)
    # realized 2x target -> 0.5 (de-risk).
    assert vol_scalar(0.24, 0.12, 0.1, 1.0) == pytest.approx(0.5)
    # realized 0.5x target -> would be 2.0 but max_gross=1.0 clamps to 1.0.
    assert vol_scalar(0.06, 0.12, 0.1, 1.0) == pytest.approx(1.0)
    # very high vol clamps UP to the min_gross floor.
    assert vol_scalar(10.0, 0.12, 0.1, 1.0) == pytest.approx(0.1)
    # missing / non-positive vol -> neutral min(1, max_gross), never /0.
    assert vol_scalar(float("nan"), 0.12, 0.1, 1.0) == pytest.approx(1.0)
    assert vol_scalar(0.0, 0.12, 0.1, 2.0) == pytest.approx(1.0)
    # with leverage allowed, a quiet market scales UP to max_gross.
    assert vol_scalar(0.06, 0.12, 0.1, 2.0) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 5. gem_select (dual momentum + absolute gate)
# --------------------------------------------------------------------------- #
def _gem_panel(us_daily, ex_daily, cash_daily=0.0001, n=260, **extra):
    series = {
        "SPY": _ramp(n, 100.0, us_daily),
        "VXUS": _ramp(n, 100.0, ex_daily),
        "BIL": _ramp(n, 100.0, cash_daily),
        "BND": _ramp(n, 100.0, 0.0002),
    }
    series.update({k: _ramp(n, 100.0, v) for k, v in extra.items()})
    return _panel(series)


def test_gem_picks_stronger_leg_and_holds_when_risk_on():
    # US clearly stronger and rising -> winner SPY, risk_on True, holds SPY.
    panel = _gem_panel(0.003, 0.001)
    sel = gem_select(panel, "SPY", "VXUS", "BIL", "BND", 252, 200, True)
    assert sel["winner"] == "SPY"
    assert sel["side"] == "us"
    assert sel["risk_on"] is True
    assert sel["asset"] == "SPY"
    assert sel["strong_down"] is False


def test_gem_picks_exus_when_stronger():
    panel = _gem_panel(0.0005, 0.003)   # ex-US stronger
    sel = gem_select(panel, "SPY", "VXUS", "BIL", "BND", 252, 200, True)
    assert sel["winner"] == "VXUS"
    assert sel["side"] == "exus"
    assert sel["asset"] == "VXUS"


def test_gem_routes_to_bonds_when_absolute_gate_fails():
    # Both legs DOWN: winner is the less-bad, but below SMA and below cash ->
    # absolute gate fails -> hold bonds, strong_down True.
    panel = _gem_panel(-0.001, -0.003)  # SPY less negative -> winner
    sel = gem_select(panel, "SPY", "VXUS", "BIL", "BND", 252, 200, True)
    assert sel["winner"] == "SPY"
    assert sel["risk_on"] is False
    assert sel["asset"] == "BND"
    assert sel["strong_down"] is True   # negative 12m AND below SMA


def test_gem_use_sma_or_vs_and():
    # Winner ABOVE its SMA (recent recovery) but 12m return BELOW cash.
    # Path: a long decline then a modest recovery so the latest close climbs back
    # above the 200d SMA, yet the 252-day point-to-point return is still NEGATIVE
    # (below the small positive cash hurdle). This is exactly the case where the
    # OR gate (above-SMA alone) and the AND gate (needs the cash hurdle too) split.
    n = 260
    down = _ramp(180, 100.0, -0.003)
    up = _ramp(80, down[-1], 0.003)
    spy = down + up
    panel = _panel({
        "SPY": spy,
        "VXUS": _ramp(n, 100.0, -0.01),     # clearly worse -> SPY wins
        "BIL": _ramp(n, 100.0, 0.0003),     # positive cash hurdle
        "BND": _ramp(n, 100.0, 0.0002),
    })
    assert above_sma(pd.Series(spy), 200) is True       # above trend
    mom12 = total_return(pd.Series(spy), 252)
    cash12 = total_return(panel["BIL"], 252)
    assert mom12 < cash12                                # fails the cash hurdle
    # OR mode: above-SMA alone passes -> risk_on True, holds SPY.
    sel_or = gem_select(panel, "SPY", "VXUS", "BIL", "BND", 252, 200, True)
    assert sel_or["risk_on"] is True and sel_or["asset"] == "SPY"
    # AND mode: needs BOTH -> cash hurdle fails -> risk_off -> bonds.
    sel_and = gem_select(panel, "SPY", "VXUS", "BIL", "BND", 252, 200, False)
    assert sel_and["risk_on"] is False and sel_and["asset"] == "BND"


# --------------------------------------------------------------------------- #
# 6. weight construction (the full target_weights assembly)
# --------------------------------------------------------------------------- #
def _full_panel(n=260, spy_daily=0.002, sector_daily=0.002, bond_daily=0.0002):
    """A complete-universe panel so target_weights can run end to end."""
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
    series = {
        "SPY": _ramp(n, 100.0, spy_daily),
        "VXUS": _ramp(n, 100.0, spy_daily * 0.5),
        "BIL": _ramp(n, 100.0, 0.0001),
        "BND": _ramp(n, 100.0, bond_daily),
        "SH": _ramp(n, 100.0, -spy_daily),
        "PSQ": _ramp(n, 100.0, -spy_daily),
        "RWM": _ramp(n, 100.0, -spy_daily),
        "QLD": _ramp(n, 100.0, spy_daily * 2),
        "TQQQ": _ramp(n, 100.0, spy_daily * 3),
    }
    # Give sectors a spread of momenta so ranking is non-trivial.
    for i, s in enumerate(sectors):
        series[s] = _ramp(n, 100.0, sector_daily * (1.0 + 0.1 * i))
    return _panel(series)


def test_target_weights_long_only_and_sums_to_at_most_one():
    panel = _full_panel()
    strat = MomentumRotationStrategy(load_params("DEFAULT"))
    w = strat.target_weights(panel.index[-1], _History(panel))
    assert all(v >= 0 for v in w.values())            # long-only
    assert sum(w.values()) <= 1.0 + 1e-9              # cash remainder allowed
    # Default has vol_target 0.12 and these paths are smooth (low vol) -> gross
    # clamps to 1.0, so the book is fully invested (sum ≈ 1).
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_default_holds_bonds_not_inverse_in_downtrend():
    # Everything DOWN: GEM gate fails and every sector is below trend. With the
    # DEFAULT (levers OFF) the whole book must sit in the SAFE asset (bonds),
    # never an inverse ETF.
    panel = _full_panel(spy_daily=-0.002, sector_daily=-0.002)
    strat = MomentumRotationStrategy(load_params("DEFAULT"))
    w = strat.target_weights(panel.index[-1], _History(panel))
    assert "BND" in w and w["BND"] > 0
    for inv in ("SH", "PSQ", "RWM"):
        assert inv not in w                            # no inverse with levers off
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_offensive_lever_routes_risk_off_into_inverse():
    # Same downtrend, but offensive_risk_off ON -> the GEM risk-off slot routes
    # into the WINNING-side inverse ETF, and below-trend sectors route to PSQ.
    # In _full_panel both equity legs fall, with VXUS (= half the SPY decline)
    # falling LESS, so VXUS wins the relative leg -> side 'exus' -> its inverse is
    # the small-cap RWM (no -1x ex-US fund in the universe). Sectors -> PSQ.
    panel = _full_panel(spy_daily=-0.002, sector_daily=-0.002)
    strat = MomentumRotationStrategy(load_params("OFFENSIVE_OFF_ON"))
    w = strat.target_weights(panel.index[-1], _History(panel))
    # GEM risk-off slot in an exus-side strong downtrend -> RWM.
    assert w.get("RWM", 0.0) > 0
    # below-trend sectors -> PSQ (the configured sector inverse).
    assert w.get("PSQ", 0.0) > 0
    # No bonds and no leverage with only the offensive lever on in a full downtrend.
    assert "BND" not in w
    assert "QLD" not in w and "TQQQ" not in w


def test_leveraged_lever_tilts_strong_uptrend_into_2x():
    # Strong uptrend: SPY well above SMA and 12m momentum > strong_threshold ->
    # leveraged_long ON tilts lev_fraction of the GEM slot into QLD (2x).
    panel = _full_panel(spy_daily=0.004, sector_daily=0.002)
    strat = MomentumRotationStrategy(load_params("LEVERAGED_ON"))
    w = strat.target_weights(panel.index[-1], _History(panel))
    assert w.get("QLD", 0.0) > 0                       # 2x tilt present
    assert w.get("SPY", 0.0) > 0                       # remainder stays in SPY 1x
    # The GEM slot split: QLD weight ≈ lev_fraction * SPY weight (pre vol-scale).
    # Both share the same gross scalar so their ratio is lev_fraction/(1-frac).
    assert w["QLD"] / (w["QLD"] + w["SPY"]) == pytest.approx(0.5, abs=1e-6)


def test_leveraged_lever_does_not_tilt_weak_uptrend():
    # Mild uptrend BELOW the strong_mom_threshold -> no leverage even with the
    # lever on (the trend filter that mitigates leverage decay).
    p = load_params("LEVERAGED_ON")
    # 12m momentum here ≈ 1.0005**252-1 ≈ 13.4%? keep it under threshold:
    panel = _full_panel(spy_daily=0.0003, sector_daily=0.0003)
    mom12 = total_return(panel["SPY"], 252)
    assert mom12 < p["strong_mom_threshold"]           # confirm it's "weak"
    strat = MomentumRotationStrategy(p)
    w = strat.target_weights(panel.index[-1], _History(panel))
    assert "QLD" not in w and "TQQQ" not in w
    assert w.get("SPY", 0.0) > 0


def test_top_n_sectors_held_equal_weight():
    panel = _full_panel(spy_daily=0.002, sector_daily=0.002)
    p = load_params("SECTOR_ONLY")                     # pure sector sleeve
    strat = MomentumRotationStrategy(p)
    w = strat.target_weights(panel.index[-1], _History(panel))
    held_sectors = [s for s in p["sectors"] if s in w]
    assert len(held_sectors) == p["top_n"]             # exactly top_n slots
    # equal-weight: every held sector has the same weight (vol scalar is common).
    vals = [w[s] for s in held_sectors]
    assert max(vals) - min(vals) < 1e-9
    # the strongest-ramp sectors (highest index -> highest daily) are the ones held.
    assert set(held_sectors) == {"XLC", "XLRE", "XLB"}  # top 3 by construction


def test_sector_below_trend_forfeits_slot_to_bonds():
    # All sectors DOWN -> none above trend -> the whole sector sleeve -> bonds.
    panel = _full_panel(spy_daily=-0.002, sector_daily=-0.002)
    p = load_params("SECTOR_ONLY")
    strat = MomentumRotationStrategy(p)
    w = strat.target_weights(panel.index[-1], _History(panel))
    assert w.get("BND", 0.0) > 0
    for s in p["sectors"]:
        assert s not in w                               # no below-trend sector held


def test_vol_overlay_derisks_when_vol_above_target():
    # Inject high realized vol on a still-rising book -> gross < 1 -> sum < 1
    # (some equity forfeited to cash by the overlay).
    n = 260
    # SPY on a DETERMINISTIC high-vol sawtooth (+6% / -4.5% alternating) with a
    # net upward drift: it ends ~5x its start, stays above its 200d SMA, and has a
    # huge 12m momentum, so it clears the risk-on gate and is actually HELD. Its
    # realized vol (~85% annualized) is far above the 12% target. VXUS falls so
    # SPY wins the relative leg. The overlay must therefore de-risk gross < 1.
    spy = [100.0]
    for i in range(n - 1):
        spy.append(spy[-1] * (1.06 if i % 2 == 0 else 0.955))
    series = {"SPY": spy, "VXUS": _ramp(n, 100.0, -0.001),
              "BIL": _ramp(n, 100.0, 0.0001), "BND": _ramp(n, 100.0, 0.0002)}
    panel = _panel(series)
    strat = MomentumRotationStrategy(load_params("GEM_ONLY"))   # isolate the overlay
    w = strat.target_weights(panel.index[-1], _History(panel))
    assert w.get("SPY", 0.0) > 0                         # the high-vol asset is held
    # realized vol >> 12% target -> gross well below 1 -> book not fully invested.
    assert 0 < sum(w.values()) < 1.0


def test_empty_history_returns_empty():
    strat = MomentumRotationStrategy(load_params("DEFAULT"))
    assert strat.target_weights(date(2020, 1, 1), _History(pd.DataFrame())) == {}


def test_variants_load_and_toggle_flags():
    assert load_params("DEFAULT")["offensive_risk_off"] is False
    assert load_params("DEFAULT")["leveraged_long"] is False
    assert load_params("OFFENSIVE_OFF_ON")["offensive_risk_off"] is True
    assert load_params("LEVERAGED_ON")["leveraged_long"] is True
    both = load_params("BOTH_LEVERS_ON")
    assert both["offensive_risk_off"] is True and both["leveraged_long"] is True
    assert load_params("GEM_ONLY")["sector_weight"] == 0.0
    assert load_params("SECTOR_ONLY")["gem_weight"] == 0.0
    with pytest.raises(KeyError):
        load_params("NOPE")
