"""tests/test_cost_model.py — CostModel friction direction + profile loading."""

from __future__ import annotations

import pytest

from backtest.engine.cost import FILL_LIMIT, FILL_MARKET, FILL_STOP, CostModel


def test_tv_style_profile_loads():
    cm = CostModel.from_profile("tv_style")
    # $1/order commission, 1-tick slippage, no half-spread on equity.
    assert cm.commission(100, 50_000, "equity") == pytest.approx(1.0)
    ref = 500.0
    # Long market entry slips UP by 1 tick.
    assert cm.apply_entry("long", ref, "equity") == pytest.approx(500.01)
    # Short market entry slips DOWN by 1 tick.
    assert cm.apply_entry("short", ref, "equity") == pytest.approx(499.99)


def test_realistic_profile_half_spread_and_slippage():
    cm = CostModel.from_profile("realistic")
    ref = 500.0
    # Equity: half_spread 0.005 + slippage 0.01 adverse on a market entry.
    assert cm.apply_entry("long", ref, "equity") == pytest.approx(500.015)
    assert cm.apply_entry("short", ref, "equity") == pytest.approx(499.985)
    # Zero commission on realistic equity.
    assert cm.commission(100, 50_000, "equity") == pytest.approx(0.0)


def test_limit_exit_pays_half_spread_no_slippage():
    cm = CostModel.from_profile("realistic")
    ref = 500.0
    # Closing a long via a resting target limit SELLS at ref - half_spread.
    px = cm.apply_exit("long", ref, "equity", fill_kind=FILL_LIMIT)
    assert px == pytest.approx(500.0 - 0.005)
    # Closing a short via a resting target limit BUYS at ref + half_spread.
    px = cm.apply_exit("short", ref, "equity", fill_kind=FILL_LIMIT)
    assert px == pytest.approx(500.0 + 0.005)


def test_stop_exit_is_adverse():
    cm = CostModel.from_profile("realistic")
    ref = 500.0
    # Stopping out of a long SELLS adversely (lower); out of a short BUYS higher.
    long_stop = cm.apply_exit("long", ref, "equity", fill_kind=FILL_STOP)
    short_stop = cm.apply_exit("short", ref, "equity", fill_kind=FILL_STOP)
    assert long_stop == pytest.approx(500.0 - 0.015)
    assert short_stop == pytest.approx(500.0 + 0.015)


def test_crypto_bps_costs():
    cm = CostModel.from_profile("realistic")
    ref = 100_000.0  # BTC-ish
    # 25 bps half_spread + 5 bps slippage = 30 bps adverse on a market entry.
    expected = ref * (30.0 / 10_000.0)
    assert cm.apply_entry("long", ref, "crypto") == pytest.approx(ref + expected)


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        CostModel.from_profile("does_not_exist")
