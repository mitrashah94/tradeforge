"""tests/test_performance_analyst.py — the performance-analyst compute + events.

Deterministic and OFFLINE: synthetic series only, a fake recorder bus, no DB and
no network. Pins the §9 headline metrics (alpha-vs-SPY, geometric growth + curve
vol, after-tax equity, risk-of-ruin) and the event contracts (STRATEGY_DEMOTED,
MILESTONE_REACHED + RATCHET_SWEEP), and asserts the research firewall: the
analyst EMITS events / returns values and never writes live config.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from orchestrator.events import Event, EventType
from orchestrator.agents.performance_analyst import (
    DEFAULT_PF_WINDOW,
    PerformanceAnalyst,
    after_tax_equity,
    after_tax_equity_curve,
    alpha_vs_spy,
    equity_curve,
    geometric_growth,
    reconcile,
    risk_of_ruin,
    rolling_profit_factor,
    spy_buy_and_hold_return,
)
from risk.config import Ratchet, load_limits


def _limits_no_sweep_gate(starting_capital: float = 1000.0):
    """Real ``Limits`` with the early-game no-sweep gate OFF (sweep_threshold=0).

    The operator may activate the no-sweep-below-$10k gate in risk/limits.yaml;
    the milestone-MATH tests pin the unconditional ``0.25 × gain`` sweep, so they
    swap in a ratchet whose ``sweep_threshold`` is 0 to stay independent of that
    operator tuning.
    """
    base = load_limits()
    ratchet = Ratchet(
        starting_capital=starting_capital,
        sweep_fraction=0.25,
        milestones=[2500, 5000, 10000, 25000, 50000, 100000],
        vault_sleeve="vault",
    )
    return base.model_copy(update={"ratchet": ratchet})


# --------------------------------------------------------------------------- #
# Fake bus (records published events; duck-typed publish/subscribe)
# --------------------------------------------------------------------------- #
class FakeBus:
    def __init__(self):
        self.published: list[Event] = []
        self.subs: list = []

    def publish(self, event: Event) -> Event:
        if event.seq is None:
            event.seq = len(self.published) + 1
        self.published.append(event)
        return event

    def subscribe(self, types, handler) -> None:
        self.subs.append((types, handler))

    def of_type(self, t: EventType) -> list[Event]:
        return [e for e in self.published if e.type == t]


# --------------------------------------------------------------------------- #
# 1. Equity curve
# --------------------------------------------------------------------------- #
def test_equity_curve_has_n_plus_one_points_and_compounds():
    curve = equity_curve([100.0, -50.0, 25.0], starting_capital=1000.0)
    assert list(curve) == [1000.0, 1100.0, 1050.0, 1075.0]


def test_equity_curve_empty_is_just_the_start():
    assert list(equity_curve([], starting_capital=1000.0)) == [1000.0]


# --------------------------------------------------------------------------- #
# 2. ALPHA VS SPY — strategy − SPY, on synthetic series
# --------------------------------------------------------------------------- #
def test_spy_buy_and_hold_close_to_close():
    bars = [
        {"ts_utc": "2026-01-01", "close": 100.0},
        {"ts_utc": "2026-02-01", "close": 110.0},
        {"ts_utc": "2026-03-01", "close": 120.0},
    ]
    # 100 -> 120 over the full window = +20%.
    assert spy_buy_and_hold_return(bars) == pytest.approx(0.20)


def test_alpha_vs_spy_from_equity_levels():
    # Strategy: 1000 -> 1300 = +30%. SPY: 100 -> 110 = +10%. Alpha = +20%.
    eq = equity_curve([300.0], starting_capital=1000.0)  # [1000, 1300]
    bars = [
        {"ts_utc": "2026-01-01", "close": 100.0},
        {"ts_utc": "2026-06-01", "close": 110.0},
    ]
    out = alpha_vs_spy(eq, bars, start="2026-01-01", end="2026-06-01")
    assert out["strategy_return"] == pytest.approx(0.30)
    assert out["spy_return"] == pytest.approx(0.10)
    assert out["alpha"] == pytest.approx(0.20)


def test_alpha_vs_spy_from_returns_series():
    # A return series (|x|<1) is compounded: (1.1)(1.1) - 1 = 0.21.
    rets = [0.10, 0.10]
    bars = [{"close": 100.0}, {"close": 105.0}]  # SPY +5%
    out = alpha_vs_spy(rets, bars)
    assert out["strategy_return"] == pytest.approx(0.21)
    assert out["spy_return"] == pytest.approx(0.05)
    assert out["alpha"] == pytest.approx(0.16)


def test_alpha_negative_when_spy_wins():
    eq = equity_curve([50.0], starting_capital=1000.0)  # +5%
    bars = [{"close": 100.0}, {"close": 120.0}]  # SPY +20%
    out = alpha_vs_spy(eq, bars)
    assert out["alpha"] < 0  # the "stop if <=0 over 6 months" bar (§9)


def test_alpha_windowing_respects_start_end():
    bars = [
        {"ts_utc": "2026-01-01", "close": 100.0},
        {"ts_utc": "2026-02-01", "close": 200.0},  # inside window
        {"ts_utc": "2026-03-01", "close": 300.0},  # outside window
    ]
    # Window clips to [Jan, Feb]: 100 -> 200 = +100%.
    assert spy_buy_and_hold_return(bars, start="2026-01-01", end="2026-02-15") == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 3. GEOMETRIC growth + curve vol
# --------------------------------------------------------------------------- #
def test_geometric_growth_matches_definition():
    rets = [0.01, -0.02, 0.03, 0.00, 0.015]
    out = geometric_growth(rets)

    arr = np.asarray(rets)
    exp_arith = float(arr.mean())
    exp_vol = float(np.std(arr, ddof=1))
    exp_geo = float(np.exp(np.mean(np.log1p(arr))) - 1.0)

    assert out["arith_mean_daily"] == pytest.approx(exp_arith)
    assert out["vol_daily"] == pytest.approx(exp_vol)
    assert out["geo_mean_daily"] == pytest.approx(exp_geo)
    # g ≈ mean − var/2 (the §0 maximization target).
    assert out["g_approx_daily"] == pytest.approx(exp_arith - 0.5 * exp_vol**2)
    assert out["n"] == 5


def test_geometric_mean_le_arithmetic_mean():
    # Geometric mean is always <= arithmetic mean for a varying series.
    rets = [0.10, -0.05, 0.08, -0.03]
    out = geometric_growth(rets)
    assert out["geo_mean_daily"] <= out["arith_mean_daily"]


def test_geometric_growth_empty_is_zero():
    out = geometric_growth([])
    assert out["n"] == 0
    assert out["geo_mean_daily"] == 0.0
    assert out["vol_daily"] == 0.0


# --------------------------------------------------------------------------- #
# 4. After-tax equity — reserve reduces realized gains by the rate
# --------------------------------------------------------------------------- #
def test_after_tax_equity_reduces_gains_by_reserve_rate():
    # Start 1000, +400 realized. Reserve 30% of the 400 gain = 120.
    out = after_tax_equity([300.0, 200.0, -100.0], starting_capital=1000.0, tax_reserve_rate=0.30)
    assert out["pretax_equity"] == pytest.approx(1400.0)
    assert out["realized_gain"] == pytest.approx(400.0)
    assert out["tax_reserve"] == pytest.approx(120.0)
    assert out["aftertax_equity"] == pytest.approx(1280.0)


def test_after_tax_no_reserve_on_a_loss():
    out = after_tax_equity([-100.0, -50.0], starting_capital=1000.0, tax_reserve_rate=0.30)
    assert out["realized_gain"] == pytest.approx(-150.0)
    assert out["tax_reserve"] == 0.0
    assert out["aftertax_equity"] == pytest.approx(850.0)


def test_after_tax_equity_curve_reserves_running_gain():
    curve = after_tax_equity_curve([200.0, -100.0], starting_capital=1000.0, tax_reserve_rate=0.25)
    # pre: [1000, 1200, 1100]; reserve 25% of gain over 1000.
    assert curve[0] == pytest.approx(1000.0)
    assert curve[1] == pytest.approx(1200.0 - 0.25 * 200.0)  # 1150
    assert curve[2] == pytest.approx(1100.0 - 0.25 * 100.0)  # 1075


def test_tax_reserve_contract_matches_daily_engine():
    """Pin the documented cross-module after-tax contract (engine vs analyst).

    Both reservers must share the same default rate and the same gains-only,
    floored-at-0 sign rule. See the ``after_tax_equity`` docstring CONTRACT note.
    """
    from orchestrator.agents.performance_analyst import DEFAULT_TAX_RESERVE_RATE
    from backtest.daily.engine import DEFAULT_SHORT_TERM_TAX_RATE

    # Same blunt default reserve rate on both sides.
    assert DEFAULT_TAX_RESERVE_RATE == DEFAULT_SHORT_TERM_TAX_RATE == 0.30

    # Gains-only: a net-up period reserves rate * net gain.
    up = after_tax_equity([300.0, -100.0], starting_capital=1000.0, tax_reserve_rate=0.30)
    assert up["realized_gain"] == pytest.approx(200.0)
    assert up["tax_reserve"] == pytest.approx(60.0)  # 0.30 * 200

    # Floored at 0: a net-down period reserves nothing and never refunds.
    down = after_tax_equity([100.0, -300.0], starting_capital=1000.0, tax_reserve_rate=0.30)
    assert down["realized_gain"] == pytest.approx(-200.0)
    assert down["tax_reserve"] == 0.0
    assert down["aftertax_equity"] == pytest.approx(down["pretax_equity"])


# --------------------------------------------------------------------------- #
# 5. Paper-vs-backtest reconciliation — flags drift
# --------------------------------------------------------------------------- #
def test_reconcile_flags_pf_drift():
    # Live PF well below the backtest expectation -> drift flagged.
    live = [10.0, -10.0, 9.0, -10.0, 8.0, -10.0]  # PF = 27/30 = 0.9
    out = reconcile(live, {"profit_factor": 2.0, "expectancy_dollar": 1.0})
    assert out["live_pf"] == pytest.approx(27.0 / 30.0)
    assert out["pf_drift_flag"] is True
    assert out["drift"] is True


def test_reconcile_no_drift_when_live_matches():
    live = [20.0, -10.0, 20.0, -10.0]  # PF = 40/20 = 2.0, exp = 5.0
    out = reconcile(live, {"profit_factor": 2.0, "expectancy_dollar": 5.0})
    assert out["pf_drift_flag"] is False
    assert out["expectancy_drift_flag"] is False
    assert out["drift"] is False


def test_reconcile_missing_expectation_leg_is_none():
    out = reconcile([1.0, -1.0], {"profit_factor": 2.0})
    assert out["expectancy_drift"] is None
    assert out["expectancy_drift_flag"] is False


# --------------------------------------------------------------------------- #
# 6. Rolling PF(30) decay -> STRATEGY_DEMOTED (exactly one, on a fake bus)
# --------------------------------------------------------------------------- #
def test_rolling_profit_factor_window():
    pnls = [1.0] * 29 + [-1.0]  # 30 trades
    out = rolling_profit_factor(pnls, window=30)
    assert len(out) == 1
    assert out[0] == pytest.approx(29.0 / 1.0)


def test_rolling_pf_empty_below_window():
    assert rolling_profit_factor([1.0, 2.0], window=30) == []


def test_decay_emits_exactly_one_strategy_demoted():
    bus = FakeBus()
    pa = PerformanceAnalyst(bus=bus, starting_capital=1000.0, pf_window=DEFAULT_PF_WINDOW,
                            pf_demote_threshold=1.0)

    # 30 trades on one strategy whose rolling PF < 1.0 (net losers): mix so the
    # last-30 window PF is below threshold. 10 wins of +1, 20 losses of -1.
    pnls = [1.0] * 10 + [-1.0] * 20  # PF over the 30 = 10/20 = 0.5 < 1.0
    for p in pnls:
        pa.record_trade(p, strategy="breakout_retest")

    demoted = bus.of_type(EventType.STRATEGY_DEMOTED)
    assert len(demoted) == 1
    ev = demoted[0]
    assert ev.data["strategy"] == "breakout_retest"
    assert ev.data["to_status"] == "PAPER"
    assert ev.data["rolling_pf"] < 1.0
    assert ev.source == "performance_analyst"

    # Continue feeding losers — still only ONE demotion event (fires once).
    for p in [-1.0] * 10:
        pa.record_trade(p, strategy="breakout_retest")
    assert len(bus.of_type(EventType.STRATEGY_DEMOTED)) == 1


def test_no_demotion_when_pf_healthy():
    bus = FakeBus()
    pa = PerformanceAnalyst(bus=bus, starting_capital=1000.0, pf_demote_threshold=1.0)
    # Healthy strategy: 20 wins +2, 10 losses -1 -> PF = 40/10 = 4.0.
    for p in [2.0] * 20 + [-1.0] * 10:
        pa.record_trade(p, strategy="good")
    assert bus.of_type(EventType.STRATEGY_DEMOTED) == []


# --------------------------------------------------------------------------- #
# 7. Milestone crossing -> MILESTONE_REACHED + RATCHET_SWEEP (0.25 × gains)
# --------------------------------------------------------------------------- #
def test_milestone_emits_reached_and_sweep_with_correct_math():
    bus = FakeBus()
    limits = _limits_no_sweep_gate(starting_capital=1000.0)
    pa = PerformanceAnalyst(bus=bus, limits=limits, starting_capital=1000.0)
    # Drive equity 1000 -> 2600 (crosses the 2500 milestone) in one trade.
    pa.record_trade(1600.0, strategy="breakout_retest")

    reached = bus.of_type(EventType.MILESTONE_REACHED)
    swept = bus.of_type(EventType.RATCHET_SWEEP)
    assert len(reached) == 1
    assert len(swept) == 1
    assert reached[0].data["milestone"] == 2500

    s = swept[0].data
    # sweep = 0.25 * (2600 - 1000) = 400; baseline advances to the milestone.
    assert s["milestone"] == 2500
    assert s["sweep_amount"] == pytest.approx(400.0)
    assert s["new_baseline"] == 2500
    assert s["vault_balance"] == pytest.approx(400.0)
    # MILESTONE_REACHED is emitted BEFORE RATCHET_SWEEP (seq order).
    assert reached[0].seq < swept[0].seq


def test_milestone_does_not_double_sweep():
    bus = FakeBus()
    limits = _limits_no_sweep_gate(starting_capital=1000.0)
    pa = PerformanceAnalyst(bus=bus, limits=limits, starting_capital=1000.0)
    pa.record_trade(1600.0, strategy="s")  # 1000 -> 2600, crosses 2500
    # Another gain that stays below the NEXT milestone (5000) must not re-sweep.
    pa.record_trade(300.0, strategy="s")  # 2600 -> 2900
    assert len(bus.of_type(EventType.RATCHET_SWEEP)) == 1
    assert pa.baseline == 2500


def test_milestone_via_position_closed_event():
    bus = FakeBus()
    limits = _limits_no_sweep_gate(starting_capital=1000.0)
    pa = PerformanceAnalyst(bus=bus, limits=limits, starting_capital=1000.0)
    pa.on_position_closed(
        Event(type=EventType.POSITION_CLOSED,
              data={"symbol": "SPY", "realized_pnl": 1600.0, "strategy": "s"})
    )
    assert len(bus.of_type(EventType.RATCHET_SWEEP)) == 1
    assert bus.of_type(EventType.RATCHET_SWEEP)[0].data["sweep_amount"] == pytest.approx(400.0)


def test_milestone_below_sweep_threshold_checkpoints_zero_sweep():
    """With the early-game gate ON, crossing a milestone BELOW the threshold is a
    checkpoint: MILESTONE_REACHED fires, RATCHET_SWEEP carries sweep_amount 0, the
    baseline advances, and the vault stays empty (early gains fully compound)."""
    bus = FakeBus()
    base = load_limits()
    gated = Ratchet(
        starting_capital=1000.0,
        sweep_fraction=0.25,
        milestones=[2500, 5000, 10000, 25000, 50000, 100000],
        vault_sleeve="vault",
        sweep_threshold=10000,
        vault_below_threshold=0,
    )
    limits = base.model_copy(update={"ratchet": gated})
    pa = PerformanceAnalyst(bus=bus, limits=limits, starting_capital=1000.0)
    pa.record_trade(1600.0, strategy="s")  # 1000 -> 2600, crosses 2500 (< 10k)

    reached = bus.of_type(EventType.MILESTONE_REACHED)
    swept = bus.of_type(EventType.RATCHET_SWEEP)
    assert len(reached) == 1 and len(swept) == 1
    assert swept[0].data["milestone"] == 2500
    assert swept[0].data["sweep_amount"] == pytest.approx(0.0)
    assert swept[0].data["new_baseline"] == 2500
    assert swept[0].data["vault_balance"] == pytest.approx(0.0)
    assert pa.baseline == 2500
    assert pa.vault_balance == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 8. Risk of ruin — bounded, ->1 for negative edge, ->0 for strong edge
# --------------------------------------------------------------------------- #
def test_risk_of_ruin_is_a_probability():
    p = risk_of_ruin(0.5, 1.5, 1.0, 0.02)
    assert 0.0 <= p <= 1.0


def test_risk_of_ruin_certain_for_negative_edge():
    # Losing edge: 40% win rate at 1:1 R -> negative drift -> ruin certain.
    assert risk_of_ruin(0.40, 1.0, 1.0, 0.02) == pytest.approx(1.0)


def test_risk_of_ruin_low_for_strong_edge_small_risk():
    # Strong edge (60% win at 2:1) + tiny risk fraction -> ~0 ruin.
    p = risk_of_ruin(0.60, 2.0, 1.0, 0.01)
    assert p < 0.05


def test_risk_of_ruin_zero_when_no_risk_taken():
    assert risk_of_ruin(0.55, 1.5, 1.0, 0.0) == 0.0


def test_risk_of_ruin_monotone_in_risk_fraction():
    # More risk per trade -> higher ruin, all else equal.
    low = risk_of_ruin(0.55, 1.5, 1.0, 0.01)
    high = risk_of_ruin(0.55, 1.5, 1.0, 0.05)
    assert high >= low


# --------------------------------------------------------------------------- #
# 9. Summary / EOD report assembles the headline lines
# --------------------------------------------------------------------------- #
def test_summary_assembles_headline_metrics():
    bus = FakeBus()
    spy = [{"ts_utc": "2026-01-01", "close": 100.0},
           {"ts_utc": "2026-06-01", "close": 105.0}]
    pa = PerformanceAnalyst(bus=bus, starting_capital=1000.0, spy_bars=spy,
                            tax_reserve_rate=0.30)
    for p, r in [(50.0, 2.0), (-25.0, -1.0), (40.0, 1.6), (-25.0, -1.0)]:
        pa.record_trade(p, strategy="breakout_retest", r_multiple=r)

    s = pa.summary()
    # Headline keys the journalist's EOD digest consumes.
    for key in ("alpha_vs_spy", "geometric_growth", "curve_vol_daily",
                "after_tax", "current_drawdown", "risk_of_ruin",
                "abort_peak_halt_pct", "ri", "risk_fraction"):
        assert key in s

    assert s["equity"] == pytest.approx(1040.0)
    assert 0.0 <= s["risk_of_ruin"] <= 1.0
    # risk_fraction tracks the YAML's default RI row's per_trade_pct/100,
    # whatever the operator has the floor set to (RI 5 -> 0.01, RI 6 -> 0.0125).
    limits = load_limits()
    expected_rf = limits.level(limits.default_ri).per_trade_pct / 100.0
    assert s["risk_fraction"] == pytest.approx(expected_rf)
    assert s["ri"] == limits.default_ri
    # after-tax equity is below pretax (a net gain was made).
    assert s["after_tax"]["aftertax_equity"] < s["after_tax"]["pretax_equity"]
    # eod_report is the alias the journalist calls.
    assert pa.eod_report()["equity"] == pytest.approx(1040.0)


def test_summary_without_spy_bars_omits_alpha():
    pa = PerformanceAnalyst(bus=FakeBus(), starting_capital=1000.0)
    pa.record_trade(10.0, strategy="s")
    assert "alpha_vs_spy" not in pa.summary()


# --------------------------------------------------------------------------- #
# FIREWALL (§6): the analyst writes NOTHING live — only emits events / returns.
# --------------------------------------------------------------------------- #
def test_firewall_no_writes_to_limits_or_registry(tmp_path, monkeypatch):
    """Run a full trade stream that triggers demotion + ratchet, and assert the
    live config files are byte-for-byte unchanged: the analyst emits events and
    returns values; a human/gate applies them."""
    import os

    limits_path = "risk/limits.yaml"
    registry_path = "strategies/registry.yaml"
    before_limits = open(limits_path, "rb").read()
    before_registry = open(registry_path, "rb").read()
    before_mtime_limits = os.path.getmtime(limits_path)
    before_mtime_registry = os.path.getmtime(registry_path)

    bus = FakeBus()
    pa = PerformanceAnalyst(bus=bus, starting_capital=1000.0, pf_demote_threshold=1.0)
    # Trigger a demotion (decayed strategy) ...
    for p in [1.0] * 10 + [-1.0] * 20:
        pa.record_trade(p, strategy="decayed")
    # ... and a milestone sweep (separate strategy that pushes equity up).
    pa.record_trade(2600.0, strategy="winner")
    _ = pa.summary()

    # Events were emitted (the analyst's only "output" to the system).
    assert bus.of_type(EventType.STRATEGY_DEMOTED)
    assert bus.of_type(EventType.RATCHET_SWEEP)

    # Live config is untouched: same bytes, same mtime.
    assert open(limits_path, "rb").read() == before_limits
    assert open(registry_path, "rb").read() == before_registry
    assert os.path.getmtime(limits_path) == before_mtime_limits
    assert os.path.getmtime(registry_path) == before_mtime_registry


def test_analyst_uses_real_ratchet_config():
    # The ratchet baseline/milestones/sweep come from the real limits.yaml (P0
    # reuse). The starting capital is operator-tuned, so assert the analyst's
    # baseline tracks the YAML rather than a hardcoded amount.
    limits = load_limits()
    pa = PerformanceAnalyst(bus=FakeBus())
    assert pa.baseline == limits.ratchet.starting_capital
    assert limits.ratchet.sweep_fraction == 0.25
