"""tests/test_portfolio_ledger.py — TWR excludes flows, MWR/IRR includes them. (#7)"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from portfolio.ledger import ContributionLedger


def test_twr_excludes_the_deposit():
    # Day 1 NAV 100 -> day 2 a $10 deposit lands AND the book is flat: NAV 110.
    # The naive return (110/100-1 = +10%) is ALL deposit; TWR must read 0%.
    nav = pd.Series([100.0, 110.0], index=[date(2024, 1, 1), date(2024, 1, 2)])
    flows = {date(2024, 1, 2): 10.0}
    twr = ContributionLedger.twr_daily_returns(nav, flows)
    assert twr.iloc[0] == pytest.approx(0.0)


def test_twr_captures_edge_on_a_deposit_day():
    # NAV 100 -> deposit 10 AND the book gains 5 -> NAV 115. TWR = (115-10)/100-1 = 5%.
    nav = pd.Series([100.0, 115.0], index=[date(2024, 1, 1), date(2024, 1, 2)])
    flows = {date(2024, 1, 2): 10.0}
    twr = ContributionLedger.twr_daily_returns(nav, flows)
    assert twr.iloc[0] == pytest.approx(0.05)


def test_mwr_irr_positive_when_terminal_exceeds_contributions():
    # Invest 100 at t0, deposit 100 at +1y, end at 230 at +2y -> positive IRR.
    led = ContributionLedger({date(2025, 1, 1): 100.0})
    nav = pd.Series([100.0, 230.0], index=[date(2024, 1, 1), date(2026, 1, 1)])
    irr = led.mwr_irr(nav, initial_equity=100.0)
    assert irr > 0.0


def test_mwr_irr_zero_when_no_growth():
    # Invest 100, deposit 50, terminal exactly 150 -> ~0% money-weighted return.
    led = ContributionLedger({date(2025, 1, 1): 50.0})
    nav = pd.Series([100.0, 150.0], index=[date(2024, 1, 1), date(2026, 1, 1)])
    irr = led.mwr_irr(nav, initial_equity=100.0)
    assert irr == pytest.approx(0.0, abs=1e-6)


def test_total_deposited():
    led = ContributionLedger({date(2025, 1, 1): 50.0, date(2025, 1, 8): 50.0})
    assert led.total_deposited() == pytest.approx(100.0)
