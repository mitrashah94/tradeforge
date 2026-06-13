"""tests/test_fast_loop_lifecycle.py — F2 open-position lifecycle on a fake bus.

Feeds a synthetic bar stream + simulated fills through an injected fake in-memory
bus and asserts the deterministic state machine (MASTER_PLAN §4 flow F2):

  * an entry ORDER_INTENT is emitted on the armed signal (and only when armed);
  * a TP1 partial-exit intent fires at +1R;
  * the stop moves to BREAKEVEN after the TP1 partial fills;
  * the runner trails (prior-bar low/high);
  * a TIME-STOP fires after N bars;
  * SESSION-FLATTEN closes the position at EOD;
  * stand-down (CIRCUIT_BREAKER_TRIPPED / NO_TRADE_WINDOW) suppresses entries.

No LLM/MCP/network — everything runs against the fake bus, synchronously.
"""

import pytest

from risk.config import load_limits
from backtest.engine.engine import PartialPlan, Strategy
from orchestrator.fast_loop import (
    ArmedStrategy,
    FastLoop,
    LifecycleState,
)


# --------------------------------------------------------------------------- #
# Fake in-memory bus (duck-typed to the contract: subscribe/publish).
# --------------------------------------------------------------------------- #
class FakeBus:
    """Minimal synchronous in-memory bus matching the duck-typed interface."""

    def __init__(self):
        self._subs = {}          # event-name -> list[handler]
        self.published = []      # all events published (the record)

    def subscribe(self, types, handler):
        for t in types:
            self._subs.setdefault(str(t), []).append(handler)

    def publish(self, event):
        self.published.append(event)
        for h in self._subs.get(str(event.type), []):
            h(event)

    # test helpers --------------------------------------------------------
    def intents(self):
        return [e for e in self.published if str(e.type) == "ORDER_INTENT"]

    def of_type(self, t):
        return [e for e in self.published if str(e.type) == t]

    def clear(self):
        self.published = []


class _SimpleEvent:
    def __init__(self, type, data, ts_utc=None, seq=0, source="test"):
        self.type = type
        self.data = data
        self.ts_utc = ts_utc
        self.seq = seq
        self.source = source


class _Bar:
    """A tiny OHLCV bar for synthetic streams."""

    def __init__(self, o, h, l, c, is_eod=False, ts=0):
        self.open, self.high, self.low, self.close = o, h, l, c
        self.is_eod = is_eod
        self.ts = ts
        self.volume = 0.0

    def as_data(self, symbol="TEST"):
        return {
            "symbol": symbol,
            "ts": self.ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "is_eod": self.is_eod,
        }


# --------------------------------------------------------------------------- #
# A deterministic synthetic strategy: arm a long on a chosen bar index with a
# partial+runner plan. Lets us drive the lifecycle precisely without rebuilding
# a full break/retest sequence (that path is covered separately below).
# --------------------------------------------------------------------------- #
class ArmOnceLong(Strategy):
    """Arms a long with a partial+runner plan.

    By default fires once at ``fire_at``. With ``repeat=True`` it re-attempts on
    every bar at/after ``fire_at`` while flat — used to verify a stand-down
    window can OPEN and then CLOSE (the strategy keeps offering the setup).
    """

    def __init__(self, *, fire_at, stop, tp1, fraction=0.5, repeat=False):
        self.fire_at = fire_at
        self.stop = stop
        self.tp1 = tp1
        self.fraction = fraction
        self.repeat = repeat
        self._fired = False

    def on_session_start(self, ctx):
        self._fired = False

    def on_bar(self, ctx):
        if ctx.position is not None:
            return
        if not self.repeat and self._fired:
            return
        if ctx.bar_index >= self.fire_at:
            plan = PartialPlan(tp1=self.tp1, tp1_r=1.0, fraction=self.fraction,
                               trail_mode="prior_bar")
            ctx.enter_long(stop=self.stop, target=None, partial=plan)
            self._fired = True


def _make_loop(bus, armed, equity=10_000.0, ri=5, budget=50.0):
    return FastLoop(
        bus=bus,
        armed=armed,
        equity_source=lambda: equity,
        limits=load_limits(),
        ri=ri,
        clock=lambda: 0,
        latency_budget_ms=budget,
        event_factory=_SimpleEvent,
    )


def _feed(bus, bar, symbol="TEST"):
    bus.publish(_SimpleEvent("BAR", bar.as_data(symbol)))


def _fill_entry(bus, loop, symbol="TEST"):
    """Simulate the broker filling the pending entry intent."""
    intent = loop._pending_entry
    bus.publish(_SimpleEvent("ORDER_FILLED", {
        "client_id": intent["client_id"], "symbol": symbol,
        "qty": intent["qty"], "fill_price": intent["bracket"]["stop_price"],
    }))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_entry_intent_emitted_on_armed_signal():
    bus = FakeBus()
    strat = ArmOnceLong(fire_at=1, stop=99.0, tp1=102.0)
    armed = ArmedStrategy(name="armtest", strategy=strat, symbol="TEST",
                          tp1_fraction=0.5, time_stop_bars=None)
    loop = _make_loop(bus, [armed])
    loop.start()

    _feed(bus, _Bar(100, 100, 100, 100, ts=0))   # bar_index 0: no fire
    assert bus.intents() == []
    _feed(bus, _Bar(100, 101, 100, 101, ts=1))   # bar_index 1: fire long
    intents = bus.intents()
    assert len(intents) == 1
    d = intents[0].data
    assert d["side"] == "buy"
    assert d["order_type"] == "market"
    assert d["symbol"] == "TEST"
    assert d["bracket"]["stop_price"] == pytest.approx(99.0)
    # vol-target: $100 risk / $2 stop distance (101-99) -> 50 shares.
    assert d["qty"] == pytest.approx(50.0)
    assert loop._managed.state is LifecycleState.OPEN_PENDING


def test_tp1_partial_then_breakeven_then_trail_then_close():
    bus = FakeBus()
    # Entry at bar 0 close=100, stop=99 (risk 1.0/share), tp1 at +1R = 101.
    strat = ArmOnceLong(fire_at=0, stop=99.0, tp1=101.0, fraction=0.5)
    armed = ArmedStrategy(name="be", strategy=strat, symbol="TEST", tp1_fraction=0.5)
    loop = _make_loop(bus, [armed])
    loop.start()

    # bar 0: arm + entry intent (managed pos OPEN_PENDING).
    _feed(bus, _Bar(100, 100, 100, 100, ts=0))
    assert len(bus.intents()) == 1
    initial_qty = loop._managed.initial_qty
    assert initial_qty == pytest.approx(100.0)   # $100 / $1 stop dist

    # entry fills -> OPEN.
    _fill_entry(bus, loop)
    assert loop._managed.state is LifecycleState.OPEN

    bus.clear()
    # bar 1: high reaches TP1 (101) -> partial-exit intent + TP1_HIT marker.
    _feed(bus, _Bar(100, 101, 100, 100.5, ts=1))
    tp1_intents = [e for e in bus.intents() if e.data["reason"] == "tp1_partial_exit"]
    assert len(tp1_intents) == 1
    assert tp1_intents[0].data["order_type"] == "limit"
    assert tp1_intents[0].data["qty"] == pytest.approx(50.0)   # 50% of 100
    assert len(bus.of_type("TP1_HIT")) == 1
    assert loop._managed.state is LifecycleState.TP1_PENDING

    bus.clear()
    # partial fills -> stop moves to BREAKEVEN (entry 100) + RUNNER state.
    bus.publish(_SimpleEvent("ORDER_PARTIAL", {
        "client_id": loop._managed.client_id + ":tp1", "qty": 50.0,
    }))
    assert loop._managed.state is LifecycleState.RUNNER
    assert loop._managed.stop_price == pytest.approx(100.0)   # breakeven
    assert loop._managed.qty == pytest.approx(50.0)           # runner only
    be_moves = [e for e in bus.intents() if e.data["reason"] == "tp1_breakeven"]
    assert len(be_moves) == 1
    assert be_moves[0].data["order_type"] == "stop"
    assert be_moves[0].data["stop_price"] == pytest.approx(100.0)

    bus.clear()
    # bar 2: price runs up; prior-bar low (101) becomes the trailing floor on
    # the NEXT bar. First record this bar's low as prior extreme.
    _feed(bus, _Bar(101, 103, 101, 102.5, ts=2))
    # bar 3: trail should tighten stop up toward prior-bar low (101).
    _feed(bus, _Bar(102, 104, 102, 103.5, ts=3))
    assert loop._managed.stop_price == pytest.approx(101.0)   # trailed up
    trail_moves = [e for e in bus.intents() if e.data["reason"] == "trail"]
    assert len(trail_moves) >= 1

    bus.clear()
    # bar 4: drops through the trailed stop (101) -> close (trail_stop) + STOP_HIT.
    _feed(bus, _Bar(103, 103, 100, 100.5, ts=4))
    closes = [e for e in bus.intents() if e.data["reason"] == "trail_stop"]
    assert len(closes) == 1
    assert closes[0].data["order_type"] == "market"
    assert len(bus.of_type("STOP_HIT")) == 1
    assert loop._managed.state is LifecycleState.CLOSING

    # POSITION_CLOSED -> terminal, slot freed.
    bus.publish(_SimpleEvent("POSITION_CLOSED", {"client_id": loop._managed.client_id}))
    assert loop._managed is None


def test_time_stop_fires():
    bus = FakeBus()
    strat = ArmOnceLong(fire_at=0, stop=99.0, tp1=200.0)   # tp1 unreachable
    armed = ArmedStrategy(name="ts", strategy=strat, symbol="TEST",
                          time_stop_bars=3)
    loop = _make_loop(bus, [armed])
    loop.start()

    _feed(bus, _Bar(100, 100, 100, 100, ts=0))   # arm + entry
    _fill_entry(bus, loop)
    bus.clear()
    # Drift sideways (never hits stop or tp1). bars_held increments each bar.
    _feed(bus, _Bar(100, 100.5, 99.5, 100, ts=1))   # held 1
    _feed(bus, _Bar(100, 100.5, 99.5, 100, ts=2))   # held 2
    assert not bus.of_type("ORDER_INTENT")          # no close yet
    _feed(bus, _Bar(100, 100.5, 99.5, 100, ts=3))   # held 3 -> time-stop
    closes = [e for e in bus.intents() if e.data["reason"] == "time_stop"]
    assert len(closes) == 1
    assert loop._managed.state is LifecycleState.CLOSING


def test_session_flatten_at_eod():
    bus = FakeBus()
    strat = ArmOnceLong(fire_at=0, stop=99.0, tp1=200.0)
    armed = ArmedStrategy(name="eod", strategy=strat, symbol="TEST")
    loop = _make_loop(bus, [armed])
    loop.start()

    _feed(bus, _Bar(100, 100, 100, 100, ts=0))   # arm + entry
    _fill_entry(bus, loop)
    bus.clear()
    _feed(bus, _Bar(100, 100.5, 99.5, 100, ts=1))
    assert not [e for e in bus.intents() if e.data["reason"] == "session_flatten"]
    # EOD bar -> session-flatten close.
    _feed(bus, _Bar(100, 100.5, 99.5, 100, is_eod=True, ts=2))
    flat = [e for e in bus.intents() if e.data["reason"] == "session_flatten"]
    assert len(flat) == 1
    assert flat[0].data["order_type"] == "market"
    assert loop._managed.state is LifecycleState.CLOSING


def test_standdown_circuit_breaker_suppresses_entry():
    bus = FakeBus()
    strat = ArmOnceLong(fire_at=0, stop=99.0, tp1=101.0)
    armed = ArmedStrategy(name="cb", strategy=strat, symbol="TEST")
    loop = _make_loop(bus, [armed])
    loop.start()

    bus.publish(_SimpleEvent("CIRCUIT_BREAKER_TRIPPED", {}))
    assert loop.halted is True
    _feed(bus, _Bar(100, 101, 100, 100, ts=0))   # strategy WOULD fire
    assert bus.intents() == []                    # but no entry emitted
    assert loop._managed is None


def test_standdown_no_trade_window_suppresses_entry():
    bus = FakeBus()
    strat = ArmOnceLong(fire_at=0, stop=99.0, tp1=101.0, repeat=True)
    armed = ArmedStrategy(name="ntz", strategy=strat, symbol="TEST")
    loop = _make_loop(bus, [armed])
    loop.start()

    bus.publish(_SimpleEvent("NO_TRADE_WINDOW", {"active": True}))
    assert loop.standing_down is True
    _feed(bus, _Bar(100, 101, 100, 100, ts=0))
    assert bus.intents() == []

    # Window closes -> entries allowed again.
    bus.publish(_SimpleEvent("NO_TRADE_WINDOW", {"active": False}))
    assert loop.standing_down is False
    _feed(bus, _Bar(100, 101, 100, 100, ts=1))
    assert len(bus.intents()) == 1


def test_not_armed_strategy_does_not_fire():
    bus = FakeBus()
    strat = ArmOnceLong(fire_at=0, stop=99.0, tp1=101.0)
    armed = ArmedStrategy(name="off", strategy=strat, symbol="TEST", armed=False)
    loop = _make_loop(bus, [armed])
    loop.start()
    _feed(bus, _Bar(100, 101, 100, 100, ts=0))
    assert bus.intents() == []


def test_real_breakout_strategy_emits_entry_through_live_context():
    """The live Context adapter correctly drives the REAL Strategy interface.

    Reconstructs a PDH break->retest on the real BreakoutRetestStrategy (V0) and
    asserts the fast loop publishes a single entry ORDER_INTENT, sized vol-target.
    """
    from strategies.breakout_retest.strategy import BreakoutRetestStrategy, load_params

    bus = FakeBus()
    params = load_params("V0")
    strat = BreakoutRetestStrategy(params=params)
    armed = ArmedStrategy(
        name="breakout", strategy=strat, symbol="QQQ", tick=0.01,
        levels={"pdh": 100.0, "pdl": 90.0},
    )
    loop = _make_loop(bus, [armed])
    loop.start()

    # Bar 0: CLOSE above pdh=100 -> break detected (bars_since_break=1).
    _feed(bus, _Bar(99.5, 100.6, 99.5, 100.5, ts=0), symbol="QQQ")
    # Bars 1..: drift; on a retest bar (bars_since_break in [2,7]) where low<=100
    # and close>100, a long retest fires.
    _feed(bus, _Bar(100.5, 100.9, 100.4, 100.7, ts=1), symbol="QQQ")  # bsb=2
    # Bar 2: retest touch of the level then close back above -> entry.
    _feed(bus, _Bar(100.7, 100.8, 99.9, 100.4, ts=2), symbol="QQQ")   # low<=100, close>100
    intents = bus.intents()
    assert len(intents) == 1
    d = intents[0].data
    assert d["side"] == "buy"
    assert d["symbol"] == "QQQ"
    assert d["qty"] > 0
    # Stop = pdh - 1 tick = 99.99; entry ref = signal close 100.4.
    assert d["bracket"]["stop_price"] == pytest.approx(99.99)
