"""Tests for stock-split back-adjustment continuity."""

from datetime import datetime

import pandas as pd
import pytest

from data.pipelines.corporate_actions import apply_split_adjustment


def _build_df():
    # Six daily bars d0..d5. A 2:1 split is effective on d3, so the raw close
    # drops from 100 (d0..d2) to 50 (d3..d5) and volume doubles.
    dates = [datetime(2024, 1, 1 + i) for i in range(6)]
    return pd.DataFrame(
        {
            "ts_utc": dates,
            "open": [100, 100, 100, 50, 50, 50],
            "high": [100, 100, 100, 50, 50, 50],
            "low": [100, 100, 100, 50, 50, 50],
            "close": [100, 100, 100, 50, 50, 50],
            "volume": [10, 10, 10, 20, 20, 20],
        }
    )


def test_raw_has_discontinuity():
    df = _build_df()
    # Pre-adjustment: a real 50-point gap exists in close.
    assert df["close"].max() - df["close"].min() == 50


def test_split_adjust_restores_continuity():
    df = _build_df()
    split_date = datetime(2024, 1, 4)  # d3
    adj = apply_split_adjustment(df, [(split_date, 2.0)])

    # All closes collapse to ~50 (continuity restored).
    assert adj["close"].max() - adj["close"].min() == pytest.approx(0.0, abs=1e-9)
    for c in adj["close"]:
        assert c == pytest.approx(50.0)

    # Pre-split volume is scaled up by the ratio (10 -> 20).
    assert adj["volume"].iloc[0] == pytest.approx(20.0)
    assert adj["volume"].iloc[2] == pytest.approx(20.0)
    # On/after the split, volume is untouched.
    assert adj["volume"].iloc[3] == pytest.approx(20.0)


def test_split_adjust_does_not_mutate_input():
    df = _build_df()
    original_close = df["close"].tolist()
    original_volume = df["volume"].tolist()
    apply_split_adjustment(df, [(datetime(2024, 1, 4), 2.0)])
    assert df["close"].tolist() == original_close
    assert df["volume"].tolist() == original_volume


def test_multiple_splits_compound():
    df = _build_df()
    # Two 2:1 splits before d0's later bars compound to 4x on the earliest bars.
    adj = apply_split_adjustment(
        df, [(datetime(2024, 1, 2), 2.0), (datetime(2024, 1, 4), 2.0)]
    )
    # d0 is before both splits -> close 100 / 4 = 25; volume 10 * 4 = 40.
    assert adj["close"].iloc[0] == pytest.approx(25.0)
    assert adj["volume"].iloc[0] == pytest.approx(40.0)
