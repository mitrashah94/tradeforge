"""tests/test_journalist.py — the JOURNALIST agent: journal a closed trade.

Deterministic, offline. A synthetic closed trade yields a frame card, correct
slippage (actual − intended), MFE/MAE passthrough, deterministic plan-adherence
flags (a plan-violating trade flags False), and a 3-line narrative; the entry is
written as md + json and is findable via search(). A bus subscriber auto-journals
on POSITION_CLOSED.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from orchestrator.agents.journalist import (
    JournalEntry,
    Journalist,
    Trade,
)
from orchestrator.events import Event, EventType


# --------------------------------------------------------------------------- #
# fixtures: a clean (plan-followed) winner and a plan-violating loser          #
# --------------------------------------------------------------------------- #
def _clean_winner() -> Trade:
    """A long that entered in-window, used its planned stop, hit target."""
    return Trade(
        symbol="QQQ",
        strategy="breakout_retest",
        side="long",
        setup_grade="A",
        planned_entry=100.00,
        planned_stop=99.00,
        planned_target=102.00,   # 2R plan
        levels={"PDH": 101.20, "PDL": 98.40, "PMH": 100.90, "PML": 99.30},
        entry_price=100.05,      # +0.05 slippage vs intended
        exit_price=102.00,
        qty=100,
        realized_pnl=195.0,
        mfe=210.0,
        mae=-30.0,
        entry_ts=datetime(2026, 6, 1, 14, 35),   # 14:35 -> within RTH window
        exit_ts=datetime(2026, 6, 1, 15, 50),
        exit_reason="target",
    )


def _plan_violator() -> Trade:
    """A long that entered OUTSIDE the window and was held past EOD (deviations)."""
    return Trade(
        symbol="SPY",
        strategy="momentum_thrust",
        side="long",
        setup_grade="B",
        planned_entry=500.0,
        planned_stop=498.0,
        planned_target=504.0,
        levels={"PDH": 503.0, "PDL": 497.0},
        entry_price=500.20,
        exit_price=499.0,
        qty=10,
        realized_pnl=-12.0,
        mfe=15.0,
        mae=-25.0,
        entry_ts=datetime(2026, 6, 1, 3, 0),     # 03:00 -> before premarket/RTH
        exit_ts=datetime(2026, 6, 2, 10, 0),     # next day -> held past EOD
        exit_reason="discretionary_close",       # not a planned exit
    )


@pytest.fixture
def journalist(tmp_path):
    # market_db points at a nonexistent path so chart/digest degrade gracefully.
    return Journalist(journal_dir=tmp_path / "journal",
                      market_db=tmp_path / "nope.duckdb",
                      notifier=lambda *a, **k: a[0] if a else "")


# --------------------------------------------------------------------------- #
# frame card + slippage + MFE/MAE + narrative                                 #
# --------------------------------------------------------------------------- #
def test_journal_trade_produces_frame_card(journalist):
    entry = journalist.journal_trade(_clean_winner())
    assert isinstance(entry, JournalEntry)
    fc = entry.frame_card
    assert fc.symbol == "QQQ"
    assert fc.strategy == "breakout_retest"
    assert fc.setup_grade == "A"
    assert fc.levels["PDH"] == 101.20 and fc.levels["PDL"] == 98.40
    assert fc.planned_entry == 100.0 and fc.planned_stop == 99.0
    assert fc.planned_target == 102.0
    assert fc.r_planned == pytest.approx(2.0)  # (102-100)/(100-99)


def test_slippage_is_actual_minus_intended(journalist):
    entry = journalist.journal_trade(_clean_winner())
    # actual entry 100.05 − intended 100.00 = +0.05
    assert entry.slippage == pytest.approx(0.05)


def test_explicit_slippage_field_is_honored(journalist):
    t = _clean_winner()
    t.slippage = -0.10  # straight from the fill record overrides the derivation
    entry = journalist.journal_trade(t)
    assert entry.slippage == pytest.approx(-0.10)


def test_mfe_mae_passthrough(journalist):
    entry = journalist.journal_trade(_clean_winner())
    assert entry.mfe == pytest.approx(210.0)
    assert entry.mae == pytest.approx(-30.0)


def test_narrative_is_three_lines(journalist):
    entry = journalist.journal_trade(_clean_winner())
    lines = entry.narrative.split("\n")
    assert len(lines) == 3
    assert "QQQ" in lines[0]
    assert "slippage" in lines[1].lower()
    assert "verdict" in lines[2].lower()


# --------------------------------------------------------------------------- #
# deterministic plan-adherence flags                                          #
# --------------------------------------------------------------------------- #
def test_clean_trade_adherence_all_true(journalist):
    entry = journalist.journal_trade(_clean_winner())
    ad = entry.adherence
    assert ad.entered_in_window is True
    assert ad.stop_at_planned_level is True
    assert ad.exited_per_plan is True
    assert ad.held_past_eod is False


def test_plan_violator_flags_false(journalist):
    entry = journalist.journal_trade(_plan_violator())
    ad = entry.adherence
    # Entered at 03:00 (outside RTH window) -> False.
    assert ad.entered_in_window is False
    # Discretionary close is not a planned exit -> False.
    assert ad.exited_per_plan is False
    # Carried into the next day -> held past EOD True.
    assert ad.held_past_eod is True
    # The narrative records the deviation verdict.
    assert "PLAN DEVIATION" in entry.narrative


# --------------------------------------------------------------------------- #
# archived as md + json, findable via search                                  #
# --------------------------------------------------------------------------- #
def test_entry_written_as_md_and_json(journalist, tmp_path):
    entry = journalist.journal_trade(_clean_winner())
    date_dir = tmp_path / "journal" / entry.date
    md = date_dir / f"{entry.trade_id}.md"
    js = date_dir / f"{entry.trade_id}.json"
    assert md.exists() and js.exists()
    # JSON round-trips and carries the frame card + adherence.
    import json
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["frame_card"]["symbol"] == "QQQ"
    assert payload["adherence"]["entered_in_window"] is True
    # Markdown is human-readable with the headline sections.
    text = md.read_text(encoding="utf-8")
    assert "# QQQ" in text and "## Frame card" in text and "## Narrative" in text


def test_search_finds_entry(journalist):
    journalist.journal_trade(_clean_winner())
    journalist.journal_trade(_plan_violator())

    hits = journalist.search("QQQ")
    assert len(hits) == 1
    assert hits[0]["symbol"] == "QQQ"

    # search by strategy
    assert any(h["strategy"] == "momentum_thrust" for h in journalist.search("momentum"))
    # empty query returns all (newest-first)
    allrecs = journalist.search("")
    assert len(allrecs) == 2
    assert allrecs[0]["symbol"] == "SPY"  # the violator was journaled last


# --------------------------------------------------------------------------- #
# bus subscription: POSITION_CLOSED auto-journals                             #
# --------------------------------------------------------------------------- #
def test_attach_to_bus_auto_journals_on_close(tmp_path):
    from orchestrator.bus import EventBus

    bus = EventBus(db_path=":memory:")
    j = Journalist(journal_dir=tmp_path / "journal",
                   market_db=tmp_path / "nope.duckdb",
                   notifier=lambda *a, **k: "")
    j.attach_to_bus(bus)

    # The gateway emits POSITION_CLOSED as {symbol, **exit_intent, **closed}.
    bus.publish(Event(
        type=EventType.POSITION_CLOSED,
        data={
            "symbol": "QQQ",
            "strategy": "breakout_retest",
            "side": "long",
            "setup_grade": "A",
            "planned_entry": 100.0,
            "planned_stop": 99.0,
            "planned_target": 102.0,
            "entry_price": 100.05,
            "exit_price": 102.0,
            "realized_pnl": 195.0,
            "mfe": 210.0,
            "mae": -30.0,
            "entry_ts": "2026-06-01T14:35:00",
            "exit_ts": "2026-06-01T15:50:00",
            "exit_reason": "target",
        },
        source="order_gateway",
    ))

    hits = j.search("QQQ")
    assert len(hits) == 1
    assert hits[0]["symbol"] == "QQQ"
    assert hits[0]["realized_pnl"] == pytest.approx(195.0)
    bus.close()
