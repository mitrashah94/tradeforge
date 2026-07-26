#!/usr/bin/env python3
"""test_premarket_scorecard.py -- plain-assert tests for premarket_scorecard.py.

The anchor case is 2026-07-14: bank earnings premarket, CPI 07:30 CT, Fed Chair
testimony 09:00 CT (inside the entry window). The trader WANTED XLF and the
written rules said no. The scorecard must reach that same answer mechanically.
Run: python3 analysis/test_premarket_scorecard.py   (must print OK). Stdlib only."""
import premarket_scorecard as ps


def _base(**kw):
    """A generically attractive ticker; override fields per test."""
    d = {
        "prior_close": 100.0, "premarket_last": 100.5,      # +0.5% gap -> CALL
        "premarket_volume": 200000, "premarket_volume_avg": 100000,   # 2.0x
        "atr14": 1.0,
        "computed_levels": {"PDH": 100.4},                   # 0.1 ATR away
        "daily_levels": [98.0],
        "option": {"spread": 0.04, "open_interest": 3000,
                   "volume": 5000, "premium_usd": 120.0},
        "next_obstacle": 102.0,                              # 1.5/0.3 = 5.0R proxy
        "earnings_in_hold_window": False,
        "macro_events_ct": [],
    }
    d.update(kw)
    return d


ALIGNED = {"SPY": "CALL", "QQQ": "CALL"}


def test_perfect_ticker_scores_eight():
    s = ps.score_ticker("XLE", _base(), ALIGNED)
    assert s.score == 8, [(c.name, c.earned, c.detail) for c in s.criteria]
    assert s.bias == "CALL" and not s.excluded
    assert s.room_r_proxy == 5.0, s.room_r_proxy


def test_each_criterion_can_fail_independently():
    cases = {
        "clear_gap_direction": _base(premarket_last=100.0),      # no gap
        "elevated_premarket_volume": _base(premarket_volume=50000),
        "near_important_level": _base(computed_levels={"PDH": 110.0}),
        "clear_daily_sr": _base(daily_levels=[]),
        "room_3r_proxy": _base(next_obstacle=100.55),
    }
    for crit, data in cases.items():
        s = ps.score_ticker("X", data, ALIGNED)
        got = {c.name: c.earned for c in s.criteria}
        assert got[crit] is False, (crit, got)
        assert s.score == 7 or crit == "clear_gap_direction", (crit, s.score)


def test_wide_spread_or_thin_oi_kills_liquidity_point():
    for opt in ({"spread": 0.12, "open_interest": 3000, "volume": 5000},
                {"spread": 0.02, "open_interest": 100, "volume": 5000},
                {"spread": 0.02, "open_interest": 3000, "volume": 5}):
        s = ps.score_ticker("X", _base(option=opt), ALIGNED)
        got = {c.name: c.earned for c in s.criteria}
        assert got["strong_option_liquidity"] is False, opt


def test_market_misalignment_costs_the_point():
    s = ps.score_ticker("X", _base(), {"SPY": "PUT", "QQQ": "CALL"})
    got = {c.name: c.earned for c in s.criteria}
    assert got["spy_qqq_alignment"] is False


# ---------------- exclusions: catalysts can only REMOVE ----------------
def test_earnings_excludes_even_a_perfect_score():
    """The 2026-07-14 XLF case: everything looked great, bank earnings premarket."""
    t = _base(earnings_in_hold_window=True, earnings_note="all 5 major banks")
    s = ps.score_ticker("XLF", t, ALIGNED)
    assert s.score == 7, s.score            # loses only the event-risk point
    assert s.excluded is True
    assert any("earnings_play" in e for e in s.exclusions), s.exclusions


def test_unaffordable_premium_excludes():
    thr = ps.Thresholds(max_premium_usd=1085.0)     # settled cash
    cheap = ps.score_ticker("XLE", _base(), ALIGNED, thr)
    rich = ps.score_ticker("SPY", _base(option={"spread": 0.02, "open_interest": 9000,
                                                "volume": 9000, "premium_usd": 2400.0}),
                           ALIGNED, thr)
    assert not cheap.excluded
    assert rich.excluded and any("premium_unaffordable" in e for e in rich.exclusions)


def test_macro_event_is_a_buffer_not_an_exclusion():
    """strategy.md sec 3 defines a 10-min buffer, NOT a day-long ban. A macro
    event must cost criterion 8 and create a blackout window -- and must NOT
    remove the ticker, which would be stricter than the written plan."""
    t = _base(macro_events_ct=[{"time_ct": "09:00", "name": "Fed testimony"}])
    s = ps.score_ticker("XLE", t, ALIGNED)
    got = {c.name: c.earned for c in s.criteria}
    assert got["no_imminent_event_risk"] is False
    assert s.excluded is False, s.exclusions
    card = ps.build_card({"tickers": {"XLE": t}})
    w = card["blackout_windows"][0]
    assert w["no_entry_from_ct"] == "08:50" and w["no_entry_to_ct"] == "09:00", w
    assert w["inside_entry_window"] is True


def test_catalyst_cannot_promote_only_demote():
    """A dull ticker with a loud catalyst must NOT outrank a better-scoring one."""
    dull = _base(premarket_last=100.0, premarket_volume=10000,
                 daily_levels=[], next_obstacle=100.4)
    dull["headline"] = "huge bullish news"          # narrative field is ignored
    snap = {"tickers": {"AAA": dull, "BBB": _base()},
            "market_bias": ALIGNED}
    card = ps.build_card(snap)
    assert card["primary"] == "BBB", card["primary"]


# ---------------- shortlist + enforcement ----------------
def test_shortlist_picks_top_two_and_is_deterministic():
    snap = {"tickers": {"AAA": _base(daily_levels=[]),          # 7
                        "BBB": _base(),                          # 8
                        "CCC": _base()},                         # 8, tie
            "market_bias": ALIGNED}
    card = ps.build_card(snap)
    assert card["primary"] == "BBB" and card["backup"] == "CCC", card
    assert ps.build_card(snap) == card                            # deterministic


def test_excluded_ticker_never_becomes_primary():
    snap = {"tickers": {"XLF": _base(earnings_in_hold_window=True),
                        "XLE": _base(daily_levels=[])},
            "market_bias": ALIGNED}
    card = ps.build_card(snap)
    assert card["primary"] == "XLE"
    assert "XLF" in card["context_only"]


def test_all_excluded_yields_a_no_trade_card():
    snap = {"tickers": {"XLF": _base(earnings_in_hold_window=True)},
            "market_bias": ALIGNED}
    card = ps.build_card(snap)
    assert card["primary"] is None and card["backup"] is None


def test_off_list_qualified_is_not_tradable():
    """The rule that makes the 5-alert selection problem safe."""
    snap = {"tickers": {"XLE": _base(), "IWM": _base(daily_levels=[]),
                        "XLF": _base(earnings_in_hold_window=True)},
            "market_bias": ALIGNED}
    card = ps.build_card(snap)
    assert ps.evaluate_qualified(card, card["primary"])["tradable"] is True
    assert ps.evaluate_qualified(card, card["backup"])["tradable"] is True
    off = ps.evaluate_qualified(card, "QQQ")
    assert off["tradable"] is False and off["role"] == "off_list"
    exc = ps.evaluate_qualified(card, "XLF")
    assert exc["tradable"] is False and "EXCLUDED" in exc["note"]


def test_render_carries_the_safety_footer():
    card = ps.build_card({"tickers": {"XLE": _base()}, "market_bias": ALIGNED})
    txt = ps.render(card)
    assert "NO BROKERAGE ACTION" in txt
    assert "SHORTLIST-ONLY" in txt and "EXCLUSION-ONLY" in txt
    assert "CAVEAT" in txt


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print("OK  ({} premarket_scorecard tests passed)".format(len(tests)))


if __name__ == "__main__":
    run()
