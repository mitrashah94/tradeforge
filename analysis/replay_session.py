#!/usr/bin/env python3
"""
Offline replay harness for the Asymmetric Live Signal v2 engine
(pine/asymmetric_live_signal.pine).

Reruns the EXACT four-track break -> retest -> confirm engine over a day of
5-minute OHLCV bars (CSV) and prints every event it would have emitted, plus
writes the events as canonical-schema JSON (see AGENTS.md) so they can be
piped straight into signals/receiver.py.

This is pure offline math over OHLCV bars: no brokerage access, no order
placement, no TradingView MCP. Python 3.9, stdlib only.

Usage:
    python3 analysis/replay_session.py --session backtests/session_2026-07-13

Mirrors, deliberately, from the Pine source:
  - Four tracks: ORH-long (CALL), ORL-short (PUT), PDH-long (CALL),
    PDL-short (PUT). ORB blocks are evaluated BEFORE PD blocks on every
    bar -- a same-bar tie-break where ORB wins (see pine header comment
    "SAME-BAR PRIORITY").
  - One QUALIFIED per day per ticker (qualifiedToday lockout silences the
    other three tracks; a silenced broken-but-unresolved track emits no
    further REJECT/EXPIRED of its own -- dangling WATCH is expected).
  - Sessions: regular 08:30-15:00 CT, opening range 08:30-08:44 CT
    (3 x 5-min bars), entry window 08:45-10:30 CT (08:45-11:30 ET).
  - ATR(14) is Wilder's RMA of true range (NOT a simple average).
  - VWAP is a manual cumulative sum reset at the regular-session open.
  - EMA trend filter is OFF by default (useEmaFilter=false in the Pine
    inputs), so it always passes.
  - riskValid gate: if the computed stop distance fails the ATR/tick
    bounds, the track emits NOTHING on that bar -- it is not "fixed", it
    stays open and may qualify/reject/expire on a later bar.
"""

import argparse
import copy
import csv
import json
import os
import sys
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# Constants mirroring the Pine inputs (defaults, asymmetric_live_signal.pine)
# -----------------------------------------------------------------------------
CT_OFFSET_HOURS = 5          # America/Chicago is UTC-5 (CDT) for this session;
                              # no DST transition occurs intraday so this is a
                              # fixed offset for the whole replay.
MINTICK = 0.01
MIN_STOP_TICKS = 3
MAX_STOP_ATR = 1.0
ATR_STOP_BUFFER = 0.10
RETEST_MIN_BARS = 1
RETEST_MAX_BARS = 6
SIGNAL_TTL_BARS = 3
MIN_REL_VOL = 1.2
VOL_AVG_LEN = 20
EMA_FAST = 9
EMA_SLOW = 20
ATR_LEN = 14
USE_VWAP_FILTER = True
USE_VOLUME_FILTER = True
USE_EMA_FILTER = False        # Pine default: off -- always passes.
REQUIRE_CONFIRM_COLOR = True

REGULAR_SESSION_SECONDS = int(6.5 * 3600)   # 08:30-15:00 CT
OPENING_RANGE_SECONDS = 900                 # 08:30-08:44 CT -> 3 x 5m bars

LEVEL_NAMES_ORDER = ("PDH", "PDL", "PDC", "PMH", "PML", "ORH", "ORL")


# -----------------------------------------------------------------------------
# Time helpers (America/Chicago, fixed UTC-5 offset for this session)
# -----------------------------------------------------------------------------
def ct_dt(epoch):
    return datetime.utcfromtimestamp(epoch) - timedelta(hours=CT_OFFSET_HOURS)


def ct_str(epoch, fmt="%Y-%m-%d %H:%M"):
    return ct_dt(epoch).strftime(fmt)


def ct_yyyymmdd(epoch):
    return ct_dt(epoch).strftime("%Y%m%d")


def round2(x):
    if x is None:
        return None
    return round(x + 0.0, 2)


# -----------------------------------------------------------------------------
# CSV / levels loading
# -----------------------------------------------------------------------------
def read_bars(path):
    bars = []
    with open(path, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bars.append({
                "t": int(row["time"]),
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
                "v": float(row["volume"]),
            })
    bars.sort(key=lambda b: b["t"])
    return bars


def load_levels(path):
    with open(path, "r") as fh:
        return json.load(fh)


# -----------------------------------------------------------------------------
# Session predicates
# -----------------------------------------------------------------------------
def in_regular_session(t, rth_open_epoch):
    return rth_open_epoch <= t < rth_open_epoch + REGULAR_SESSION_SECONDS


def in_opening_range(t, rth_open_epoch):
    return rth_open_epoch <= t < rth_open_epoch + OPENING_RANGE_SECONDS


def in_entry_window(t, entry_start_epoch, entry_end_epoch):
    return entry_start_epoch <= t < entry_end_epoch


def in_premarket(t, rth_open_epoch):
    return t < rth_open_epoch


# -----------------------------------------------------------------------------
# Indicator computation, bar-by-bar over the FULL array (mirrors Pine ta.*
# running continuously over the whole chart, independent of session labels).
# -----------------------------------------------------------------------------
def compute_ema(closes, length):
    """Standard TradingView ta.ema: seed = source[0], then EMA recursion."""
    alpha = 2.0 / (length + 1)
    out = [None] * len(closes)
    ema = None
    for i, c in enumerate(closes):
        ema = c if ema is None else (alpha * c + (1 - alpha) * ema)
        out[i] = ema
    return out


def compute_true_range(bars):
    tr = [None] * len(bars)
    for i, b in enumerate(bars):
        if i == 0:
            tr[i] = b["h"] - b["l"]
        else:
            pc = bars[i - 1]["c"]
            tr[i] = max(b["h"] - b["l"], abs(b["h"] - pc), abs(b["l"] - pc))
    return tr


def compute_rma(src, length):
    """Wilder's RMA, matching Pine's ta.rma / ta.atr exactly:
    na until `length` values are available; seeded with the SMA of the
    first `length` values; thereafter the standard alpha=1/length EMA
    recursion. This is NOT a plain moving average.
    """
    out = [None] * len(src)
    if len(src) >= length:
        seed = sum(src[0:length]) / float(length)
        out[length - 1] = seed
        alpha = 1.0 / length
        prev = seed
        for i in range(length, len(src)):
            prev = alpha * src[i] + (1 - alpha) * prev
            out[i] = prev
    return out


def compute_sma(src, length):
    out = [None] * len(src)
    for i in range(len(src)):
        if i + 1 >= length:
            window = src[i - length + 1:i + 1]
            out[i] = sum(window) / float(length)
    return out


def compute_pmh_pml(bars, rth_open_epoch):
    """Premarket high/low: running max/min while inPremarket, frozen
    (retains last value) once the regular session begins -- mirrors the
    Pine `var float pmh/pml` that only mutates `if inPremarket`."""
    pmh = [None] * len(bars)
    pml = [None] * len(bars)
    cur_h = None
    cur_l = None
    for i, b in enumerate(bars):
        if in_premarket(b["t"], rth_open_epoch):
            cur_h = b["h"] if cur_h is None else max(cur_h, b["h"])
            cur_l = b["l"] if cur_l is None else min(cur_l, b["l"])
        pmh[i] = cur_h
        pml[i] = cur_l
    return pmh, pml


def compute_orh_orl(bars, rth_open_epoch):
    """Opening-range high/low: running max/min over the opening-range bars
    ONLY, frozen thereafter -- mirrors the Pine `var float orh/orl`."""
    orh = [None] * len(bars)
    orl = [None] * len(bars)
    cur_h = None
    cur_l = None
    for i, b in enumerate(bars):
        if in_opening_range(b["t"], rth_open_epoch):
            cur_h = b["h"] if cur_h is None else max(cur_h, b["h"])
            cur_l = b["l"] if cur_l is None else min(cur_l, b["l"])
        orh[i] = cur_h
        orl[i] = cur_l
    return orh, orl


def compute_vwap(bars, rth_open_epoch):
    """Manual cumulative-sum VWAP, reset at regular-session start,
    accumulated only while inRegularSession -- mirrors the Pine block
    exactly (including that the *exposed* value is not session-gated,
    only the accumulation is; this does not matter for this dataset
    since it never runs past the regular-session end)."""
    vwap = [None] * len(bars)
    cum_pv = 0.0
    cum_vol = 0.0
    for i, b in enumerate(bars):
        t = b["t"]
        is_reg = in_regular_session(t, rth_open_epoch)
        prev_reg = i > 0 and in_regular_session(bars[i - 1]["t"], rth_open_epoch)
        if is_reg and not prev_reg:
            cum_pv = 0.0
            cum_vol = 0.0
        if is_reg:
            hlc3 = (b["h"] + b["l"] + b["c"]) / 3.0
            cum_pv += hlc3 * b["v"]
            cum_vol += b["v"]
        vwap[i] = (cum_pv / cum_vol) if cum_vol > 0 else None
    return vwap


# -----------------------------------------------------------------------------
# f_score / f_nextObstacle ports
# -----------------------------------------------------------------------------
def f_next_obstacle(entry_price, is_call, broken_level, levels_now):
    """Nearest of the seven computed levels strictly beyond entry in the
    trade direction, excluding the broken level BY NAME (never by price
    equality)."""
    best = None
    for name in LEVEL_NAMES_ORDER:
        if name == broken_level:
            continue
        val = levels_now.get(name)
        if val is None:
            continue
        if is_call and val > entry_price and (best is None or val < best):
            best = val
        if not is_call and val < entry_price and (best is None or val > best):
            best = val
    return best


def f_score(is_call, room_r, next_obstacle, bars_since_break,
            rel_vol, ema9, ema20, vwap_now, atr, close):
    sc = 0
    if rel_vol is not None and rel_vol >= MIN_REL_VOL:
        sc += 1
    if next_obstacle is None or (room_r is not None and room_r >= 3.0):
        sc += 1
    if ema9 is not None and ema20 is not None:
        if (is_call and ema9 > ema20) or ((not is_call) and ema9 < ema20):
            sc += 1
    if vwap_now is not None and atr is not None:
        if is_call and close > vwap_now + 0.05 * atr:
            sc += 1
        elif (not is_call) and close < vwap_now - 0.05 * atr:
            sc += 1
    if bars_since_break <= 3:
        sc += 1
    return sc


# -----------------------------------------------------------------------------
# Per-track state
# -----------------------------------------------------------------------------
class Track(object):
    def __init__(self):
        self.broken = False
        self.break_bar = None
        self.done = False


# -----------------------------------------------------------------------------
# Core replay: one ticker, one day.
# -----------------------------------------------------------------------------
def run_ticker(ticker, bars, static_levels, session_epochs, pre_window_reject_rearms=False):
    """static_levels: dict with 'pdh', 'pdl', 'pdc' (constants for the day).
    session_epochs: dict with 'rth_open', 'entry_start', 'entry_end'.
    pre_window_reject_rearms: if True, a PD track (PDH/PDL only -- ORB tracks
        are structurally immune) whose REJECT condition fires BEFORE the
        entry window opens re-arms (waits for a fresh break) instead of being
        burned for the day. Off by default; tests the pre-window-REJECT
        hypothesis (see CLAUDE.md "Open question -- the pre-window REJECT").
    Returns (events, computed_levels_summary).
    """
    rth_open_epoch = session_epochs["rth_open"]
    entry_start = session_epochs["entry_start"]
    entry_end = session_epochs["entry_end"]

    pdh = static_levels.get("pdh")
    pdl = static_levels.get("pdl")
    pdc = static_levels.get("pdc")

    closes = [b["c"] for b in bars]
    volumes = [b["v"] for b in bars]

    ema9_arr = compute_ema(closes, EMA_FAST)
    ema20_arr = compute_ema(closes, EMA_SLOW)
    tr_arr = compute_true_range(bars)
    atr_arr = compute_rma(tr_arr, ATR_LEN)
    vol_sma_arr = compute_sma(volumes, VOL_AVG_LEN)
    pmh_arr, pml_arr = compute_pmh_pml(bars, rth_open_epoch)
    orh_arr, orl_arr = compute_orh_orl(bars, rth_open_epoch)
    vwap_arr = compute_vwap(bars, rth_open_epoch)

    orh_track = Track()
    orl_track = Track()
    pdh_track = Track()
    pdl_track = Track()

    qualified_today = False
    q_live = False
    q_stop = None
    q_direction = 0
    q_expire_bar_index = None
    q_level_name = None
    q_level_price = None

    events = []

    def emit(event_name, direction, level_name, level_price, bar_i, extra=None):
        t = bars[bar_i]["t"]
        setup_type = "A_break_retest" if direction == "CALL" else "B_breakdown_bounce"
        event_id = "{}-{}-{}-{}-{}-{}".format(
            ticker, ct_yyyymmdd(t), direction, level_name, event_name, bar_i)
        payload = {
            "event": event_name,
            "event_id": event_id,
            "ticker": ticker,
            "timeframe": "5",
            "setup_type": setup_type,
            "direction": direction,
            "level": level_name,
            "level_price": round2(level_price),
            "signal_time_ct": ct_str(t),
            "vwap": round2(vwap_arr[bar_i]),
            "rvol": round2(_rel_vol(bar_i)),
        }
        if extra:
            payload.update(extra)
        events.append(payload)
        return payload

    def _rel_vol(i):
        va = vol_sma_arr[i]
        if va is None or va <= 0:
            return None
        return volumes[i] / va

    def vwap_filter_ok(is_call, i):
        if not USE_VWAP_FILTER:
            return True
        vw = vwap_arr[i]
        if vw is None:
            return False
        return (closes[i] > vw) if is_call else (closes[i] < vw)

    def volume_filter_ok(i):
        if not USE_VOLUME_FILTER:
            return True
        rv = _rel_vol(i)
        return rv is not None and rv >= MIN_REL_VOL

    def ema_filter_ok(is_call, i):
        if not USE_EMA_FILTER:
            return True
        e9, e20 = ema9_arr[i], ema20_arr[i]
        if e9 is None or e20 is None:
            return False
        if is_call:
            return e9 > e20 and closes[i] > e9
        return e9 < e20 and closes[i] < e9

    def candle_ok(is_call, i):
        if not REQUIRE_CONFIRM_COLOR:
            return True
        b = bars[i]
        return (b["c"] > b["o"]) if is_call else (b["c"] < b["o"])

    def levels_now(i):
        return {
            "PDH": pdh, "PDL": pdl, "PDC": pdc,
            "PMH": pmh_arr[i], "PML": pml_arr[i],
            "ORH": orh_arr[i], "ORL": orl_arr[i],
        }

    n = len(bars)
    for i in range(n):
        b = bars[i]
        t, o, h, l, c, v = b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]
        prev_close = bars[i - 1]["c"] if i > 0 else None
        is_regular = in_regular_session(t, rth_open_epoch)
        is_or = in_opening_range(t, rth_open_epoch)
        is_entry = in_entry_window(t, entry_start, entry_end)
        atr = atr_arr[i]

        # ---------------------------------------------------------------
        # INVALIDATED check (runs first every bar, mirrors Pine order)
        # ---------------------------------------------------------------
        if q_live:
            if i > q_expire_bar_index:
                q_live = False
            else:
                stop_violated = (c < q_stop) if q_direction == 1 else (c > q_stop)
                if stop_violated:
                    direction = "CALL" if q_direction == 1 else "PUT"
                    emit("INVALIDATED", direction, q_level_name, q_level_price, i,
                         {"reason": "closed_through_stop_after_qualified"})
                    q_live = False
                    q_expire_bar_index = None
                    q_stop = None
                    q_direction = 0
                    q_level_name = None
                    q_level_price = None

        # =================================================================
        # ORB blocks FIRST (deliberate tie-break: ORB wins). Do not reorder.
        # =================================================================

        # ---- ORH-long / CALL ----
        orh_now = orh_arr[i]
        if (not qualified_today and not orh_track.done and not orh_track.broken
                and is_regular and not is_or and orh_now is not None
                and prev_close is not None and c > orh_now and prev_close <= orh_now):
            orh_track.broken = True
            orh_track.break_bar = i
            emit("WATCH", "CALL", "ORH", orh_now, i)

        if not qualified_today and not orh_track.done and orh_track.broken:
            bars_since = i - orh_track.break_bar
            in_window = RETEST_MIN_BARS <= bars_since <= RETEST_MAX_BARS
            orh_lvl = orh_arr[i]
            if (in_window and is_entry and l <= orh_lvl and c > orh_lvl
                    and vwap_filter_ok(True, i) and volume_filter_ok(i)
                    and ema_filter_ok(True, i) and candle_ok(True, i)):
                entry_price = c
                if atr is not None:
                    stop_price = min(l, orh_lvl) - atr * ATR_STOP_BUFFER
                    risk = entry_price - stop_price
                    risk_valid = (risk >= MINTICK * MIN_STOP_TICKS
                                  and risk <= atr * MAX_STOP_ATR)
                else:
                    risk_valid = False
                if risk_valid:
                    r1 = entry_price + 1.0 * risk
                    r2 = entry_price + 2.0 * risk
                    r3 = entry_price + 3.0 * risk
                    r4 = entry_price + 4.0 * risk
                    r5 = entry_price + 5.0 * risk
                    lv = levels_now(i)
                    nobs = f_next_obstacle(entry_price, True, "ORH", lv)
                    room_r = None if nobs is None else (nobs - entry_price) / risk
                    sc = f_score(True, room_r, nobs, bars_since, _rel_vol(i),
                                 ema9_arr[i], ema20_arr[i], vwap_arr[i], atr, c)
                    expiration_bar_time = t + SIGNAL_TTL_BARS * 5 * 60
                    entry_high = h
                    expiration_price = entry_high + 0.5 * risk
                    q_stop = stop_price
                    q_direction = 1
                    q_live = True
                    q_expire_bar_index = i + SIGNAL_TTL_BARS
                    q_level_name = "ORH"
                    q_level_price = orh_lvl
                    orh_track.done = True
                    qualified_today = True
                    emit("QUALIFIED", "CALL", "ORH", orh_lvl, i, {
                        "entry_low": round2(entry_price),
                        "entry_high": round2(entry_high),
                        "stop": round2(stop_price),
                        "r1": round2(r1), "r2": round2(r2), "r3": round2(r3),
                        "r4": round2(r4), "r5": round2(r5),
                        "next_obstacle": round2(nobs),
                        "room_r": round2(room_r),
                        "score": sc,
                        "expiration_time_ct": ct_str(expiration_bar_time),
                        "expiration_price": round2(expiration_price),
                    })
            if (not orh_track.done and bars_since >= 1 and bars_since <= RETEST_MAX_BARS
                    and atr is not None and c < orh_lvl - atr * ATR_STOP_BUFFER):
                orh_track.done = True
                emit("REJECT", "CALL", "ORH", orh_lvl, i,
                     {"reason": "failed_hold_below_level"})
            if not orh_track.done and bars_since > RETEST_MAX_BARS:
                orh_track.done = True
                emit("EXPIRED", "CALL", "ORH", orh_lvl, i,
                     {"reason": "retest_window_elapsed"})

        # ---- ORL-short / PUT ----
        orl_now = orl_arr[i]
        if (not qualified_today and not orl_track.done and not orl_track.broken
                and is_regular and not is_or and orl_now is not None
                and prev_close is not None and c < orl_now and prev_close >= orl_now):
            orl_track.broken = True
            orl_track.break_bar = i
            emit("WATCH", "PUT", "ORL", orl_now, i)

        if not qualified_today and not orl_track.done and orl_track.broken:
            bars_since = i - orl_track.break_bar
            in_window = RETEST_MIN_BARS <= bars_since <= RETEST_MAX_BARS
            orl_lvl = orl_arr[i]
            if (in_window and is_entry and h >= orl_lvl and c < orl_lvl
                    and vwap_filter_ok(False, i) and volume_filter_ok(i)
                    and ema_filter_ok(False, i) and candle_ok(False, i)):
                entry_price = c
                if atr is not None:
                    stop_price = max(h, orl_lvl) + atr * ATR_STOP_BUFFER
                    risk = stop_price - entry_price
                    risk_valid = (risk >= MINTICK * MIN_STOP_TICKS
                                  and risk <= atr * MAX_STOP_ATR)
                else:
                    risk_valid = False
                if risk_valid:
                    r1 = entry_price - 1.0 * risk
                    r2 = entry_price - 2.0 * risk
                    r3 = entry_price - 3.0 * risk
                    r4 = entry_price - 4.0 * risk
                    r5 = entry_price - 5.0 * risk
                    lv = levels_now(i)
                    nobs = f_next_obstacle(entry_price, False, "ORL", lv)
                    room_r = None if nobs is None else (entry_price - nobs) / risk
                    sc = f_score(False, room_r, nobs, bars_since, _rel_vol(i),
                                 ema9_arr[i], ema20_arr[i], vwap_arr[i], atr, c)
                    expiration_bar_time = t + SIGNAL_TTL_BARS * 5 * 60
                    entry_low_field = l  # unused directly; entry_low is close per spec
                    expiration_price = entry_price - 0.5 * risk
                    q_stop = stop_price
                    q_direction = -1
                    q_live = True
                    q_expire_bar_index = i + SIGNAL_TTL_BARS
                    q_level_name = "ORL"
                    q_level_price = orl_lvl
                    orl_track.done = True
                    qualified_today = True
                    emit("QUALIFIED", "PUT", "ORL", orl_lvl, i, {
                        "entry_low": round2(entry_price),
                        "entry_high": round2(h),
                        "stop": round2(stop_price),
                        "r1": round2(r1), "r2": round2(r2), "r3": round2(r3),
                        "r4": round2(r4), "r5": round2(r5),
                        "next_obstacle": round2(nobs),
                        "room_r": round2(room_r),
                        "score": sc,
                        "expiration_time_ct": ct_str(expiration_bar_time),
                        "expiration_price": round2(expiration_price),
                    })
            if (not orl_track.done and bars_since >= 1 and bars_since <= RETEST_MAX_BARS
                    and atr is not None and c > orl_lvl + atr * ATR_STOP_BUFFER):
                orl_track.done = True
                emit("REJECT", "PUT", "ORL", orl_lvl, i,
                     {"reason": "failed_hold_above_level"})
            if not orl_track.done and bars_since > RETEST_MAX_BARS:
                orl_track.done = True
                emit("EXPIRED", "PUT", "ORL", orl_lvl, i,
                     {"reason": "retest_window_elapsed"})

        # ---- PDH-long / CALL ----
        if (not qualified_today and not pdh_track.done and not pdh_track.broken
                and is_regular and pdh is not None
                and prev_close is not None and c > pdh and prev_close <= pdh):
            pdh_track.broken = True
            pdh_track.break_bar = i
            emit("WATCH", "CALL", "PDH", pdh, i)

        if not qualified_today and not pdh_track.done and pdh_track.broken:
            bars_since = i - pdh_track.break_bar
            in_window = RETEST_MIN_BARS <= bars_since <= RETEST_MAX_BARS
            if (in_window and is_entry and l <= pdh and c > pdh
                    and vwap_filter_ok(True, i) and volume_filter_ok(i)
                    and ema_filter_ok(True, i) and candle_ok(True, i)):
                entry_price = c
                if atr is not None:
                    stop_price = min(l, pdh) - atr * ATR_STOP_BUFFER
                    risk = entry_price - stop_price
                    risk_valid = (risk >= MINTICK * MIN_STOP_TICKS
                                  and risk <= atr * MAX_STOP_ATR)
                else:
                    risk_valid = False
                if risk_valid:
                    r1 = entry_price + 1.0 * risk
                    r2 = entry_price + 2.0 * risk
                    r3 = entry_price + 3.0 * risk
                    r4 = entry_price + 4.0 * risk
                    r5 = entry_price + 5.0 * risk
                    lv = levels_now(i)
                    nobs = f_next_obstacle(entry_price, True, "PDH", lv)
                    room_r = None if nobs is None else (nobs - entry_price) / risk
                    sc = f_score(True, room_r, nobs, bars_since, _rel_vol(i),
                                 ema9_arr[i], ema20_arr[i], vwap_arr[i], atr, c)
                    expiration_bar_time = t + SIGNAL_TTL_BARS * 5 * 60
                    entry_high = h
                    expiration_price = entry_high + 0.5 * risk
                    q_stop = stop_price
                    q_direction = 1
                    q_live = True
                    q_expire_bar_index = i + SIGNAL_TTL_BARS
                    q_level_name = "PDH"
                    q_level_price = pdh
                    pdh_track.done = True
                    qualified_today = True
                    emit("QUALIFIED", "CALL", "PDH", pdh, i, {
                        "entry_low": round2(entry_price),
                        "entry_high": round2(entry_high),
                        "stop": round2(stop_price),
                        "r1": round2(r1), "r2": round2(r2), "r3": round2(r3),
                        "r4": round2(r4), "r5": round2(r5),
                        "next_obstacle": round2(nobs),
                        "room_r": round2(room_r),
                        "score": sc,
                        "expiration_time_ct": ct_str(expiration_bar_time),
                        "expiration_price": round2(expiration_price),
                    })
            if (not pdh_track.done and bars_since >= 1 and bars_since <= RETEST_MAX_BARS
                    and atr is not None and c < pdh - atr * ATR_STOP_BUFFER):
                if pre_window_reject_rearms and t < entry_start:
                    pdh_track.broken = False
                    pdh_track.break_bar = None
                else:
                    pdh_track.done = True
                    emit("REJECT", "CALL", "PDH", pdh, i,
                         {"reason": "failed_hold_below_level"})
            if not pdh_track.done and bars_since > RETEST_MAX_BARS:
                pdh_track.done = True
                emit("EXPIRED", "CALL", "PDH", pdh, i,
                     {"reason": "retest_window_elapsed"})

        # ---- PDL-short / PUT ----
        if (not qualified_today and not pdl_track.done and not pdl_track.broken
                and is_regular and pdl is not None
                and prev_close is not None and c < pdl and prev_close >= pdl):
            pdl_track.broken = True
            pdl_track.break_bar = i
            emit("WATCH", "PUT", "PDL", pdl, i)

        if not qualified_today and not pdl_track.done and pdl_track.broken:
            bars_since = i - pdl_track.break_bar
            in_window = RETEST_MIN_BARS <= bars_since <= RETEST_MAX_BARS
            if (in_window and is_entry and h >= pdl and c < pdl
                    and vwap_filter_ok(False, i) and volume_filter_ok(i)
                    and ema_filter_ok(False, i) and candle_ok(False, i)):
                entry_price = c
                if atr is not None:
                    stop_price = max(h, pdl) + atr * ATR_STOP_BUFFER
                    risk = stop_price - entry_price
                    risk_valid = (risk >= MINTICK * MIN_STOP_TICKS
                                  and risk <= atr * MAX_STOP_ATR)
                else:
                    risk_valid = False
                if risk_valid:
                    r1 = entry_price - 1.0 * risk
                    r2 = entry_price - 2.0 * risk
                    r3 = entry_price - 3.0 * risk
                    r4 = entry_price - 4.0 * risk
                    r5 = entry_price - 5.0 * risk
                    lv = levels_now(i)
                    nobs = f_next_obstacle(entry_price, False, "PDL", lv)
                    room_r = None if nobs is None else (entry_price - nobs) / risk
                    sc = f_score(False, room_r, nobs, bars_since, _rel_vol(i),
                                 ema9_arr[i], ema20_arr[i], vwap_arr[i], atr, c)
                    expiration_bar_time = t + SIGNAL_TTL_BARS * 5 * 60
                    expiration_price = entry_price - 0.5 * risk
                    q_stop = stop_price
                    q_direction = -1
                    q_live = True
                    q_expire_bar_index = i + SIGNAL_TTL_BARS
                    q_level_name = "PDL"
                    q_level_price = pdl
                    pdl_track.done = True
                    qualified_today = True
                    emit("QUALIFIED", "PUT", "PDL", pdl, i, {
                        "entry_low": round2(entry_price),
                        "entry_high": round2(h),
                        "stop": round2(stop_price),
                        "r1": round2(r1), "r2": round2(r2), "r3": round2(r3),
                        "r4": round2(r4), "r5": round2(r5),
                        "next_obstacle": round2(nobs),
                        "room_r": round2(room_r),
                        "score": sc,
                        "expiration_time_ct": ct_str(expiration_bar_time),
                        "expiration_price": round2(expiration_price),
                    })
            if (not pdl_track.done and bars_since >= 1 and bars_since <= RETEST_MAX_BARS
                    and atr is not None and c > pdl + atr * ATR_STOP_BUFFER):
                if pre_window_reject_rearms and t < entry_start:
                    pdl_track.broken = False
                    pdl_track.break_bar = None
                else:
                    pdl_track.done = True
                    emit("REJECT", "PUT", "PDL", pdl, i,
                         {"reason": "failed_hold_above_level"})
            if not pdl_track.done and bars_since > RETEST_MAX_BARS:
                pdl_track.done = True
                emit("EXPIRED", "PUT", "PDL", pdl, i,
                     {"reason": "retest_window_elapsed"})

    computed_levels = {
        "PDH": pdh, "PDL": pdl, "PDC": pdc,
        "PMH": pmh_arr[-1] if bars else None,
        "PML": pml_arr[-1] if bars else None,
        "ORH": orh_arr[-1] if bars else None,
        "ORL": orl_arr[-1] if bars else None,
    }
    return events, computed_levels


# -----------------------------------------------------------------------------
# CLI / reporting
# -----------------------------------------------------------------------------
QUALIFIED_PRINT_KEYS = (
    "entry_low", "entry_high", "stop", "r1", "r2", "r3", "r4", "r5",
    "next_obstacle", "room_r", "score", "expiration_time_ct", "expiration_price",
)


def print_ticker_report(ticker, events, computed_levels):
    print("=" * 70)
    print("{}  --  computed levels".format(ticker))
    print("=" * 70)
    for name in ("PDH", "PDL", "PDC", "PMH", "PML", "ORH", "ORL"):
        val = computed_levels.get(name)
        print("  {:4s}: {}".format(name, "n/a" if val is None else "{:.2f}".format(val)))
    print("-" * 70)
    if not events:
        print("  (no events)")
    for ev in events:
        line = "  {}  {:11s} {:4s} {:4s} @ {:.2f}  vwap={}  rvol={}".format(
            ev["signal_time_ct"], ev["event"], ev["direction"], ev["level"],
            ev["level_price"],
            "n/a" if ev["vwap"] is None else "{:.2f}".format(ev["vwap"]),
            "n/a" if ev["rvol"] is None else "{:.2f}".format(ev["rvol"]))
        print(line)
        if ev["event"] == "QUALIFIED":
            for k in QUALIFIED_PRINT_KEYS:
                print("      {:18s}: {}".format(k, ev.get(k)))
        if "reason" in ev:
            print("      reason            : {}".format(ev["reason"]))
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay the Asymmetric Live Signal v2 engine offline over "
                    "a day of 5-minute bars.")
    parser.add_argument("--session", default="backtests/session_2026-07-13",
                        help="path to the session directory containing "
                             "<TICKER>_5m.csv files and levels.json")
    parser.add_argument("--pre-window-reject-rearms", action="store_true",
                        default=False,
                        help="PD tracks that REJECT before the entry window "
                             "opens re-arm (wait for a fresh break) instead "
                             "of being burned for the day. Off by default; "
                             "used to test the pre-window-REJECT hypothesis. "
                             "RESEARCH ONLY: this can emit two WATCH events "
                             "for one track/day, which violates the "
                             "AGENTS.md one-event-per-type-per-track-per-day "
                             "invariant, so it is never written to the "
                             "canonical events.json filename (see --out).")
    parser.add_argument("--out", default=None,
                        help="override the output events JSON path. Default "
                             "is <session>/events.json, EXCEPT when "
                             "--pre-window-reject-rearms is set, where the "
                             "default becomes <session>/events_rearm.json "
                             "so a research run can never silently poison "
                             "the canonical record analysis/record_session.py "
                             "consumes.")
    args = parser.parse_args(argv)

    session_dir = args.session
    levels_path = os.path.join(session_dir, "levels.json")
    levels_data = load_levels(levels_path)

    rth_open_epoch = levels_data["rth_open_epoch"]
    entry_start = levels_data["entry_window_start_epoch"]
    entry_end = levels_data["entry_window_end_epoch"]
    session_epochs = {
        "rth_open": rth_open_epoch,
        "entry_start": entry_start,
        "entry_end": entry_end,
    }

    tickers = sorted(levels_data["tickers"].keys())
    all_events = []
    for ticker in tickers:
        csv_path = os.path.join(session_dir, "{}_5m.csv".format(ticker))
        if not os.path.exists(csv_path):
            print("WARNING: no CSV for {} at {}".format(ticker, csv_path),
                  file=sys.stderr)
            continue
        bars = read_bars(csv_path)
        static_levels = levels_data["tickers"][ticker]
        events, computed_levels = run_ticker(
            ticker, bars, static_levels, session_epochs,
            pre_window_reject_rearms=args.pre_window_reject_rearms)
        print_ticker_report(ticker, events, computed_levels)
        all_events.extend(events)

    if args.out:
        out_path = args.out
    elif args.pre_window_reject_rearms:
        out_path = os.path.join(session_dir, "events_rearm.json")
    else:
        out_path = os.path.join(session_dir, "events.json")

    with open(out_path, "w") as fh:
        json.dump(all_events, fh, indent=2)

    if args.pre_window_reject_rearms:
        print("Wrote {} RESEARCH-ONLY events (--pre-window-reject-rearms is "
              "ON) to {}".format(len(all_events), out_path))
        print("This is NOT canonical-schema data (may contain more than one "
              "WATCH per track/day) and must never be consumed by "
              "analysis/record_session.py or sit under the events.json "
              "filename.")
    else:
        print("Wrote {} events to {}".format(len(all_events), out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
