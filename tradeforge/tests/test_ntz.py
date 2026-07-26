"""Tests for the No-Trade Zone (NTZ) overlap math."""

from data.levels import compute_ntz


def test_ntz_overlapping():
    # PDH/PDL = 400/395, PMH/PML = 398/396 -> overlap [396, 398].
    assert compute_ntz(400, 395, 398, 396) == (396, 398, True)


def test_ntz_contained():
    # Premarket range fully inside prior-day range -> overlap = premarket.
    assert compute_ntz(400, 390, 398, 392) == (392, 398, True)


def test_ntz_non_overlapping():
    # Premarket entirely below the prior-day range -> no overlap.
    assert compute_ntz(400, 398, 396, 394) == (None, None, False)


def test_ntz_degenerate_no_premarket():
    # No premarket (None inputs) -> invalid.
    assert compute_ntz(400, 395, None, None) == (None, None, False)


def test_ntz_degenerate_range():
    # Zero-width prior-day range (pdh == pdl) -> invalid.
    assert compute_ntz(400, 400, 398, 396) == (None, None, False)
