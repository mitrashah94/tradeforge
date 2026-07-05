"""tests/test_breakout_retest_options.py — the IV-rank options overlay.

Deterministic, offline: a synthetic SPY option chain with hand-assigned greeks so
we can assert the structure choice (IV rank), the strike choice (delta), the
policy gate (incl. RI-4 'minimal'), the theta gate, and the honest $1,000
sub-one-contract sizing skip.
"""

from __future__ import annotations

from datetime import date

import pytest

from risk.config import load_limits
from risk.sizing import per_trade_dollar_risk
from strategies.breakout_retest_options.overlay import (
    STRUCT_CREDIT,
    STRUCT_DEBIT,
    STRUCT_LONG,
    OptionContract,
    OptionsOverlay,
    UnderlyingSignal,
    choose_structure,
    classify_iv_regime,
    load_params,
    nearest_delta,
)

AS_OF = date(2026, 7, 5)
EXP = "2026-07-06"  # dte = 1

# Hand-assigned call/put greeks around spot 500 so strike picks are unambiguous.
_CALLS = {
    #  strike: (delta, theta, bid, ask)
    498: (0.62, -0.16, 4.40, 4.55),
    499: (0.58, -0.16, 3.70, 3.85),
    500: (0.54, -0.15, 3.00, 3.10),
    501: (0.48, -0.14, 2.45, 2.55),
    502: (0.42, -0.12, 1.95, 2.05),
    503: (0.35, -0.10, 1.50, 1.58),
    504: (0.30, -0.09, 1.15, 1.22),
    505: (0.24, -0.07, 0.85, 0.91),
    506: (0.18, -0.06, 0.60, 0.66),
}
_PUTS = {
    500: (-0.46, -0.15, 2.95, 3.05),
    499: (-0.42, -0.14, 2.45, 2.55),
    498: (-0.38, -0.13, 2.00, 2.08),
    497: (-0.32, -0.11, 1.55, 1.62),
    496: (-0.27, -0.10, 1.20, 1.27),
    495: (-0.22, -0.08, 0.90, 0.96),
    494: (-0.16, -0.06, 0.62, 0.68),
    493: (-0.12, -0.05, 0.42, 0.47),
}


def _chain(oi: int = 500) -> list[OptionContract]:
    out = []
    for strike, (d, th, bid, ask) in _CALLS.items():
        out.append(OptionContract("SPY", EXP, strike, "call", bid, ask, d, th,
                                   open_interest=oi))
    for strike, (d, th, bid, ask) in _PUTS.items():
        out.append(OptionContract("SPY", EXP, strike, "put", bid, ask, d, th,
                                   open_interest=oi))
    return out


def _long_signal(spot=500.0, stop=499.2, target=501.6, grade="B"):
    return UnderlyingSignal("SPY", "long", spot, stop, target, grade)


# --------------------------------------------------------------------- params
def test_params_load_defaults():
    p = load_params()
    assert p["iv_rank_low"] == 0.30
    assert p["iv_rank_high"] == 0.60
    assert p["long_delta_target"] == 0.55
    assert p["min_dte"] == 1  # 0DTE off by default


# ---------------------------------------------------------------- pure helpers
def test_classify_iv_regime():
    assert classify_iv_regime(0.10, 0.30, 0.60) == "low"
    assert classify_iv_regime(0.45, 0.30, 0.60) == "mid"
    assert classify_iv_regime(0.60, 0.30, 0.60) == "high"
    assert classify_iv_regime(0.80, 0.30, 0.60) == "high"


def test_choose_structure_policy_matrix():
    # 'none' blocks everything.
    assert choose_structure("low", "none") is None
    assert choose_structure("high", "none") is None
    # 'minimal' (RI-4) forces long options ONLY, regardless of IV rank.
    assert choose_structure("low", "minimal") == STRUCT_LONG
    assert choose_structure("high", "minimal") == STRUCT_LONG
    # 'defined_risk_small' and 'ok' let IV rank drive the structure.
    for pol in ("defined_risk_small", "ok"):
        assert choose_structure("low", pol) == STRUCT_LONG
        assert choose_structure("mid", pol) == STRUCT_DEBIT
        assert choose_structure("high", pol) == STRUCT_CREDIT


def test_nearest_delta_picks_closest():
    calls = [c for c in _chain() if c.right == "call"]
    assert nearest_delta(calls, 0.55).strike == 500   # 0.54 is closest
    assert nearest_delta(calls, 0.30).strike == 504   # exact
    assert nearest_delta([], 0.5) is None


# ------------------------------------------------------------ structure choice
def test_low_iv_buys_a_long_call_near_target_delta():
    ov = OptionsOverlay()
    dec = ov.select(_long_signal(), _chain(), iv_rank=0.10, equity=100_000.0,
                    options_policy="ok", dollar_risk=1250.0, ri=6, as_of=AS_OF)
    assert dec.ok
    assert dec.structure == STRUCT_LONG
    assert len(dec.legs) == 1
    leg = dec.legs[0]
    assert leg.action == "buy" and leg.contract.right == "call"
    assert leg.contract.strike == 500          # ~0.55 delta long leg
    assert dec.max_loss_per_contract == pytest.approx(310.0)  # ask 3.10 * 100


def test_mid_iv_builds_a_debit_vertical():
    ov = OptionsOverlay()
    dec = ov.select(_long_signal(), _chain(), iv_rank=0.45, equity=100_000.0,
                    options_policy="ok", dollar_risk=1250.0, ri=6, as_of=AS_OF)
    assert dec.ok
    assert dec.structure == STRUCT_DEBIT
    strikes = sorted(l.contract.strike for l in dec.legs)
    assert strikes == [500.0, 504.0]           # buy 0.55d, sell 0.30d call above
    # net debit = (buy ask 3.10 - sell bid 1.15) * 100
    assert dec.net_debit == pytest.approx((3.10 - 1.15) * 100)
    assert dec.net_debit < 310.0               # cheaper than the outright long


def test_high_iv_builds_a_theta_positive_credit_spread():
    ov = OptionsOverlay()
    dec = ov.select(_long_signal(), _chain(), iv_rank=0.75, equity=100_000.0,
                    options_policy="ok", dollar_risk=1250.0, ri=6, as_of=AS_OF)
    assert dec.ok
    assert dec.structure == STRUCT_CREDIT
    # bullish view -> bull PUT credit spread: sell higher put, buy lower put.
    sell = [l for l in dec.legs if l.action == "sell"][0].contract
    buy = [l for l in dec.legs if l.action == "buy"][0].contract
    assert sell.right == "put" and buy.right == "put"
    assert buy.strike < sell.strike
    assert dec.net_credit > 0
    assert dec.net_theta > 0                    # theta is a TAILWIND here


# ------------------------------------------------------------------ policy gate
def test_policy_none_blocks():
    ov = OptionsOverlay()
    dec = ov.select(_long_signal(), _chain(), iv_rank=0.10, equity=100_000.0,
                    options_policy="none", dollar_risk=1000.0, ri=1, as_of=AS_OF)
    assert not dec.ok
    assert dec.reason.startswith("options_blocked_by_policy")


def test_ri4_minimal_forces_long_even_at_high_iv():
    """RI-4 -> options: minimal. Long options only; flags the costly high-IV buy."""
    limits = load_limits()
    assert limits.level(4).options == "minimal"
    equity = 100_000.0
    dollar_risk = per_trade_dollar_risk(equity, 4, limits)  # 0.75% at RI-4
    ov = OptionsOverlay()
    dec = ov.select(_long_signal(), _chain(), iv_rank=0.80, equity=equity,
                    options_policy="minimal", dollar_risk=dollar_risk, ri=4, as_of=AS_OF)
    assert dec.structure == STRUCT_LONG        # NOT a credit spread
    assert all(l.action == "buy" for l in dec.legs)
    assert "minimal_policy_forces_long_premium_at_high_iv" in dec.warnings


# --------------------------------------------------------------- $1,000 sizing
def test_thousand_dollar_account_skips_and_reports_min_viable_equity():
    limits = load_limits()
    equity = 1000.0
    dollar_risk = per_trade_dollar_risk(equity, 6, limits)  # 1.25% -> $12.50
    assert dollar_risk == pytest.approx(12.5)
    ov = OptionsOverlay()
    dec = ov.select(_long_signal(), _chain(), iv_rank=0.10, equity=equity,
                    options_policy="ok", dollar_risk=dollar_risk, ri=6, as_of=AS_OF)
    assert not dec.ok
    assert dec.reason == "sub_one_contract_within_risk_budget"
    assert dec.diagnostics["min_viable_equity"] > equity
    assert dec.diagnostics["binding_constraint"] in {"max_loss", "vol_target", "premium"}


def test_allow_min_ticket_takes_one_over_budget_contract():
    p = load_params()
    p["allow_min_ticket"] = True
    ov = OptionsOverlay(p)
    dec = ov.select(_long_signal(), _chain(), iv_rank=0.10, equity=1000.0,
                    options_policy="ok", dollar_risk=12.5, ri=6, as_of=AS_OF)
    assert dec.ok
    assert dec.contracts == 1
    assert any(w.startswith("over_budget_min_ticket") for w in dec.warnings)


def test_large_account_sizes_multiple_contracts():
    ov = OptionsOverlay()
    dec = ov.select(_long_signal(), _chain(), iv_rank=0.10, equity=100_000.0,
                    options_policy="ok", dollar_risk=1250.0, ri=6, as_of=AS_OF)
    assert dec.ok
    assert dec.contracts >= 1
    assert dec.net_delta > 0            # long calls -> positive position delta


# ------------------------------------------------------------------ theta gate
def test_theta_gate_rejects_theta_heavy_long_ticket():
    # Tiny move-to-target vs a long call's theta -> theta eats the edge.
    ov = OptionsOverlay()
    sig = _long_signal(spot=500.0, stop=499.7, target=500.3)  # 0.3 move edge
    dec = ov.select(sig, _chain(), iv_rank=0.10, equity=100_000.0,
                    options_policy="ok", dollar_risk=1250.0, ri=6, as_of=AS_OF)
    assert not dec.ok
    assert dec.reason == "theta_exceeds_edge_budget"
    assert dec.diagnostics["theta_to_edge"] > 0.35


def test_theta_gate_ignored_for_credit_spreads():
    ov = OptionsOverlay()
    sig = _long_signal(spot=500.0, stop=499.7, target=500.3)
    dec = ov.select(sig, _chain(), iv_rank=0.80, equity=100_000.0,
                    options_policy="ok", dollar_risk=1250.0, ri=6, as_of=AS_OF)
    assert dec.structure == STRUCT_CREDIT
    assert dec.ok                       # theta positive -> gate does not bite
    assert dec.diagnostics["theta_to_edge"] == 0.0


# ------------------------------------------------------------------ dte window
def test_dte_window_excludes_out_of_range_expirations():
    ov = OptionsOverlay()
    far = [OptionContract("SPY", "2026-08-20", 500, "call", 3.0, 3.1, 0.54, -0.05,
                          open_interest=500)]  # ~46 dte, outside [1,7]
    dec = ov.select(_long_signal(), far, iv_rank=0.10, equity=100_000.0,
                    options_policy="ok", dollar_risk=1250.0, ri=6, as_of=AS_OF)
    assert not dec.ok
    assert dec.reason == "no_expiration_in_dte_window"


# =========================================================================== #
# small_account_first90 profile — $1,000, opening-90-minutes options trading
# =========================================================================== #
def _first90():
    return OptionsOverlay(load_params("small_account_first90"))


def _timed_signal(time_et, spot=500.0, stop=499.2, target=501.6):
    return UnderlyingSignal("SPY", "long", spot, stop, target, "B", time_et=time_et)


def test_profile_loads_small_account_first90():
    p = load_params("small_account_first90")
    assert p["_profile"] == "small_account_first90"
    assert p["session_first_n_minutes"] == 90
    assert p["sizing_mode"] == "premium_risk"
    assert p["min_dte"] == 0                    # 0DTE allowed intraday
    assert p["enable_structure_downgrade"] is True


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        load_params("nope")


def test_session_gate_blocks_outside_first_90_minutes():
    ov = _first90()
    dec = ov.select(_timed_signal("13:00"), _chain(), iv_rank=0.10, equity=1000.0,
                    options_policy="ok", dollar_risk=12.5, ri=6, as_of=AS_OF)
    assert not dec.ok
    assert dec.reason == "outside_first_90min_window"


def test_session_gate_allows_inside_window():
    ov = _first90()
    dec = ov.select(_timed_signal("09:31"), _chain(), iv_rank=0.10, equity=1000.0,
                    options_policy="ok", dollar_risk=12.5, ri=6, as_of=AS_OF)
    assert dec.reason != "outside_first_90min_window"


def test_session_gate_warns_when_no_time_given():
    ov = _first90()
    dec = ov.select(_timed_signal(None), _chain(), iv_rank=0.10, equity=1000.0,
                    options_policy="ok", dollar_risk=12.5, ri=6, as_of=AS_OF)
    assert "session_gate_enabled_but_no_signal_time" in dec.warnings


def test_thousand_dollar_first90_downgrades_to_affordable_defined_risk_ticket():
    """The heart of it: $1k CAN trade — the ATM outright ($310, 31%) is
    downgraded to a cheap 1-wide debit vertical sized to premium-at-risk."""
    limits = load_limits()
    dec = _first90().select(
        _timed_signal("10:05"), _chain(), iv_rank=0.10, equity=1000.0,
        options_policy="ok", dollar_risk=per_trade_dollar_risk(1000.0, 6, limits),
        ri=6, as_of=AS_OF, daily_halt_pct=limits.level(6).daily_halt_pct,
    )
    assert dec.ok
    assert dec.structure == STRUCT_DEBIT
    assert dec.contracts == 1
    assert dec.max_loss_per_contract <= 0.12 * 1000.0     # within the hard ceiling
    assert dec.diagnostics["sizing_mode"] == "premium_risk"
    assert any(w.startswith("downgraded_to") for w in dec.warnings)
    # honest: the ticket risks MORE than the RI daily halt, and it is flagged.
    assert dec.diagnostics["risk_pct_of_equity"] > 0.025
    assert any(w.startswith("ticket_risk_exceeds_daily_halt") for w in dec.warnings)


def test_first90_skips_when_nothing_fits_hard_ceiling():
    # A punishing 2% hard ceiling ($20 on $1k) — no SPY ticket fits -> honest skip.
    p = load_params("small_account_first90")
    p["hard_max_trade_risk_pct"] = 0.02
    dec = OptionsOverlay(p).select(
        _timed_signal("10:05"), _chain(), iv_rank=0.10, equity=1000.0,
        options_policy="ok", dollar_risk=12.5, ri=6, as_of=AS_OF,
    )
    assert not dec.ok
    assert dec.reason == "no_defined_risk_ticket_under_ceiling"
    assert dec.diagnostics["min_viable_equity"] > 1000.0


def test_first90_premium_risk_sizes_multiple_contracts_when_affordable():
    # Enough equity that the 6% premium-risk budget clears 2x the ATM debit.
    dec = _first90().select(
        _timed_signal("10:05"), _chain(), iv_rank=0.10, equity=12_000.0,
        options_policy="ok", dollar_risk=150.0, ri=6, as_of=AS_OF,
    )
    assert dec.ok
    assert dec.contracts >= 2                   # budget 6%*$12k=$720 vs ~$310 debit
    assert dec.net_delta > 0
