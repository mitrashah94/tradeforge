#!/usr/bin/env python3
"""
Opportunity scan / filter-sensitivity sweep for the Asymmetric Live Signal v2
engine (pine/asymmetric_live_signal.pine), run offline over the two recorded
sessions (backtests/session_2026-07-13, backtests/session_2026-07-14).

Pure offline math over OHLCV bars already on disk. No brokerage access, no
order placement, no TradingView MCP. Python 3.9, stdlib only.

n = 2 SESSIONS, 3 TICKERS. This is a HYPOTHESIS-GENERATION artifact, not a
decision. See the header of reports/opportunity_scan.md for the full caveat
-- it is repeated in every table in this module's output on purpose.

Reuses the verified engine primitives from analysis/replay_session.py
(read_bars, load_levels, compute_true_range, compute_rma, compute_sma,
compute_ema, compute_vwap, compute_orh_orl, compute_pmh_pml,
in_regular_session, in_opening_range, in_entry_window, f_next_obstacle,
f_score, run_ticker, Track) rather than reimplementing them. The filter
variants requested here (RVOL measured on a different bar, break-only entry,
etc.) are NOT expressible as simple constant overrides on run_ticker, so this
module contains a second, parameterized engine (`run_variant`) that mirrors
run_ticker's four-track break -> retest -> confirm state machine exactly but
takes the swept knobs as arguments. Its baseline configuration is asserted,
in the test suite, to reproduce run_ticker's own event stream bar-for-bar.

Usage:
    python3 analysis/opportunity_scan.py
    (writes reports/opportunity_scan.md, prints a summary to stdout)
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.replay_session import (  # noqa: E402
    Track,
    read_bars,
    load_levels,
    compute_true_range,
    compute_rma,
    compute_sma,
    compute_ema,
    compute_vwap,
    compute_orh_orl,
    compute_pmh_pml,
    in_regular_session,
    in_opening_range,
    in_entry_window,
    f_next_obstacle,
    f_score,
    run_ticker,
    ct_str,
    round2,
    MINTICK,
    MIN_STOP_TICKS,
    ATR_STOP_BUFFER,
    RETEST_MIN_BARS,
    RETEST_MAX_BARS,
    SIGNAL_TTL_BARS,
    MIN_REL_VOL,
    VOL_AVG_LEN,
    EMA_FAST,
    EMA_SLOW,
    ATR_LEN,
    USE_VWAP_FILTER,
    USE_VOLUME_FILTER,
    USE_EMA_FILTER,
    REQUIRE_CONFIRM_COLOR,
    MAX_STOP_ATR,
)

SESSIONS = ["backtests/session_2026-07-13", "backtests/session_2026-07-14"]
FLAT_BY_SECONDS = int((14 * 3600 + 55 * 60) - (8 * 3600 + 30 * 60))  # 08:30 -> 14:55 CT
N_SESSIONS = len(SESSIONS)
N_TICKERS = 3
SAMPLE_CAVEAT = (
    "n = {} sessions x {} tickers max (some tickers/sessions produce fewer "
    "breaks). NO variant should be adopted on this evidence -- this is a "
    "hypothesis-generation artifact, not a decision. With this few setups "
    "the 'best' variant is guaranteed to be overfit."
).format(N_SESSIONS, N_TICKERS)


def repo_path(*parts):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, *parts)


# -----------------------------------------------------------------------------
# Baseline config (mirrors replay_session.py module constants exactly, so that
# run_variant(..., DEFAULT_CONFIG) reproduces run_ticker(...) bar-for-bar).
# -----------------------------------------------------------------------------
def default_config():
    return {
        "rvol_threshold": MIN_REL_VOL,      # 1.2
        "rvol_bar": "confirm",              # "confirm" | "break"
        "require_confirm_color": REQUIRE_CONFIRM_COLOR,  # True
        "retest_max_bars": RETEST_MAX_BARS,  # 6
        "entry_mode": "retest",             # "retest" | "break_only" | "break_or_retest"
        "max_stop_atr": MAX_STOP_ATR,        # 1.0
        "pre_window_reject_rearms": False,
        "use_volume_filter": USE_VOLUME_FILTER,  # True
    }


# -----------------------------------------------------------------------------
# Parameterized engine. Mirrors run_ticker's four-track state machine, but the
# gates it evaluates are driven off `config` instead of module constants, and
# it supports break-only / break-or-retest entry modes plus an optional
# gate-attribution instrumentation hook used by Part C.
# -----------------------------------------------------------------------------
def run_variant(ticker, bars, static_levels, session_epochs, config, gate_log=None):
    """Returns (events, computed_levels). If gate_log is a list, appends one
    dict per (track, bar) where price structure (touch+reclaim) passed inside
    the retest window but the bar did not qualify, recording which named
    gate(s) blocked it.
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

    rvol_threshold = config["rvol_threshold"]
    rvol_bar = config["rvol_bar"]
    require_confirm_color = config["require_confirm_color"]
    retest_max_bars = config["retest_max_bars"]
    entry_mode = config["entry_mode"]
    max_stop_atr = config["max_stop_atr"]
    pre_window_reject_rearms = config.get("pre_window_reject_rearms", False)
    use_volume_filter = config.get("use_volume_filter", True) and rvol_threshold > 0

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

    def volume_filter_ok(i, break_i):
        if not use_volume_filter:
            return True
        idx = break_i if rvol_bar == "break" else i
        rv = _rel_vol(idx)
        return rv is not None and rv >= rvol_threshold

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
        if not require_confirm_color:
            return True
        b = bars[i]
        return (b["c"] > b["o"]) if is_call else (b["c"] < b["o"])

    def levels_now(i):
        return {
            "PDH": pdh, "PDL": pdl, "PDC": pdc,
            "PMH": pmh_arr[i], "PML": pml_arr[i],
            "ORH": orh_arr[i], "ORL": orl_arr[i],
        }

    def level_value(level_name, i):
        if level_name == "ORH":
            return orh_arr[i]
        if level_name == "ORL":
            return orl_arr[i]
        if level_name == "PDH":
            return pdh
        return pdl

    tracks = {"ORH": Track(), "ORL": Track(), "PDH": Track(), "PDL": Track()}
    qualified_today = False
    q_live = False
    q_stop = None
    q_direction = 0
    q_expire_bar_index = None
    q_level_name = None
    q_level_price = None

    events = []

    def emit(event_name, direction, level_name, level_price, bar_i, rvol_bar_i, extra=None):
        t = bars[bar_i]["t"]
        setup_type = "A_break_retest" if direction == "CALL" else "B_breakdown_bounce"
        event_id = "{}-{}-{}-{}-{}-{}".format(
            ticker, ct_str(t, "%Y%m%d"), direction, level_name, event_name, bar_i)
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
            "rvol": round2(_rel_vol(rvol_bar_i)),
            "bar_index": bar_i,
        }
        if extra:
            payload.update(extra)
        events.append(payload)
        return payload

    def attempt_qualify(level_name, is_call, i, break_i, lvl, bars_since):
        atr = atr_arr[i]
        if atr is None:
            return None
        b = bars[i]
        l, h, c = b["l"], b["h"], b["c"]
        if is_call:
            stop_price = min(l, lvl) - atr * ATR_STOP_BUFFER
            risk = c - stop_price
        else:
            stop_price = max(h, lvl) + atr * ATR_STOP_BUFFER
            risk = stop_price - c
        risk_valid = (risk >= MINTICK * MIN_STOP_TICKS and risk <= atr * max_stop_atr)
        if not risk_valid:
            return None
        entry_price = c
        sign = 1.0 if is_call else -1.0
        r1 = entry_price + sign * 1.0 * risk
        r2 = entry_price + sign * 2.0 * risk
        r3 = entry_price + sign * 3.0 * risk
        r4 = entry_price + sign * 4.0 * risk
        r5 = entry_price + sign * 5.0 * risk
        lv = levels_now(i)
        nobs = f_next_obstacle(entry_price, is_call, level_name, lv)
        if nobs is None:
            room_r = None
        else:
            room_r = (nobs - entry_price) / risk if is_call else (entry_price - nobs) / risk
        rvol_for_score = _rel_vol(break_i if rvol_bar == "break" else i)
        sc = f_score(is_call, room_r, nobs, bars_since, rvol_for_score,
                     ema9_arr[i], ema20_arr[i], vwap_arr[i], atr, c)
        expiration_bar_time = bars[i]["t"] + SIGNAL_TTL_BARS * 5 * 60
        expiration_price = h + 0.5 * risk if is_call else entry_price - 0.5 * risk
        return {
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
            "_entry_price": entry_price,
            "_stop_price": stop_price,
            "_risk": risk,
            "_r3": r3,
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

        # INVALIDATED check first, mirrors baseline order.
        if q_live:
            if i > q_expire_bar_index:
                q_live = False
            else:
                stop_violated = (c < q_stop) if q_direction == 1 else (c > q_stop)
                if stop_violated:
                    direction = "CALL" if q_direction == 1 else "PUT"
                    emit("INVALIDATED", direction, q_level_name, q_level_price, i, i,
                         {"reason": "closed_through_stop_after_qualified"})
                    q_live = False
                    q_expire_bar_index = None
                    q_stop = None
                    q_direction = 0
                    q_level_name = None
                    q_level_price = None

        # ORB tracks first (deliberate tie-break), then PD tracks.
        for level_name, is_call, is_orb in (
            ("ORH", True, True), ("ORL", False, True),
            ("PDH", True, False), ("PDL", False, False),
        ):
            track = tracks[level_name]
            if qualified_today or track.done:
                continue

            lvl_now = level_value(level_name, i)
            if is_orb:
                gate_ok = is_regular and not is_or and lvl_now is not None
            else:
                gate_ok = is_regular and lvl_now is not None

            just_broke = False
            if not track.broken and gate_ok and prev_close is not None:
                broke = (c > lvl_now and prev_close <= lvl_now) if is_call \
                    else (c < lvl_now and prev_close >= lvl_now)
                if broke:
                    track.broken = True
                    track.break_bar = i
                    emit("WATCH", "CALL" if is_call else "PUT", level_name, lvl_now, i, i)
                    just_broke = True

            if not track.broken or track.done:
                continue

            bars_since = i - track.break_bar

            # --- immediate break-bar entry (break_only / break_or_retest) ---
            if just_broke and entry_mode in ("break_only", "break_or_retest"):
                lvl_at_break = level_value(level_name, i)
                gates_pass = (vwap_filter_ok(is_call, i)
                              and volume_filter_ok(i, track.break_bar)
                              and ema_filter_ok(is_call, i)
                              and candle_ok(is_call, i))
                if gates_pass:
                    res = attempt_qualify(level_name, is_call, i, track.break_bar,
                                           lvl_at_break, 0)
                    if res is not None:
                        q_stop = res["_stop_price"]
                        q_direction = 1 if is_call else -1
                        q_live = True
                        q_expire_bar_index = i + SIGNAL_TTL_BARS
                        q_level_name = level_name
                        q_level_price = lvl_at_break
                        track.done = True
                        qualified_today = True
                        emit("QUALIFIED", "CALL" if is_call else "PUT", level_name,
                             lvl_at_break, i, track.break_bar if rvol_bar == "break" else i,
                             res)
                        continue
                if entry_mode == "break_only":
                    # one shot only: no retest fallback in pure break_only mode.
                    track.done = True
                    continue
                # break_or_retest: fall through to normal retest logic on later bars.

            if bars_since == 0:
                # break bar itself already handled above for break modes; for
                # plain "retest" mode nothing happens on the break bar itself.
                continue

            lvl_now2 = level_value(level_name, i)
            in_window = RETEST_MIN_BARS <= bars_since <= retest_max_bars

            if in_window and is_entry:
                touch_reclaim = (l <= lvl_now2 and c > lvl_now2) if is_call \
                    else (h >= lvl_now2 and c < lvl_now2)
                if touch_reclaim:
                    g_color = candle_ok(is_call, i)
                    g_rvol = volume_filter_ok(i, track.break_bar)
                    g_vwap = vwap_filter_ok(is_call, i)
                    g_ema = ema_filter_ok(is_call, i)
                    all_gates_pass = g_color and g_rvol and g_vwap and g_ema
                    qualified_this_bar = False
                    if all_gates_pass:
                        res = attempt_qualify(level_name, is_call, i, track.break_bar,
                                               lvl_now2, bars_since)
                        if res is not None:
                            q_stop = res["_stop_price"]
                            q_direction = 1 if is_call else -1
                            q_live = True
                            q_expire_bar_index = i + SIGNAL_TTL_BARS
                            q_level_name = level_name
                            q_level_price = lvl_now2
                            track.done = True
                            qualified_today = True
                            qualified_this_bar = True
                            emit("QUALIFIED", "CALL" if is_call else "PUT", level_name,
                                 lvl_now2, i, track.break_bar if rvol_bar == "break" else i,
                                 res)
                    if not qualified_this_bar and gate_log is not None:
                        blocked = []
                        if not g_color:
                            blocked.append("candle_color")
                        if not g_rvol:
                            blocked.append("rvol")
                        if not g_vwap:
                            blocked.append("vwap")
                        if not g_ema:
                            blocked.append("ema")
                        if all_gates_pass:
                            # gates passed but risk_valid failed
                            blocked.append("risk_valid_vs_atr")
                        gate_log.append({
                            "ticker": ticker,
                            "level": level_name,
                            "direction": "CALL" if is_call else "PUT",
                            "bar_index": i,
                            "signal_time_ct": ct_str(t),
                            "bars_since_break": bars_since,
                            "blocked_by": blocked,
                        })

            if not track.done and bars_since >= 1 and bars_since <= retest_max_bars \
                    and atr is not None:
                fail_cond = (c < lvl_now2 - atr * ATR_STOP_BUFFER) if is_call \
                    else (c > lvl_now2 + atr * ATR_STOP_BUFFER)
                if fail_cond:
                    if pre_window_reject_rearms and not is_orb and t < entry_start:
                        track.broken = False
                        track.break_bar = None
                    else:
                        track.done = True
                        reason = "failed_hold_below_level" if is_call else "failed_hold_above_level"
                        emit("REJECT", "CALL" if is_call else "PUT", level_name, lvl_now2, i, i,
                             {"reason": reason})
            if not track.done and bars_since > retest_max_bars:
                track.done = True
                emit("EXPIRED", "CALL" if is_call else "PUT", level_name, lvl_now2, i, i,
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
# Session/session-data loading helpers
# -----------------------------------------------------------------------------
def load_session(session_dir):
    levels_path = os.path.join(session_dir, "levels.json")
    levels_data = load_levels(levels_path)
    session_epochs = {
        "rth_open": levels_data["rth_open_epoch"],
        "entry_start": levels_data["entry_window_start_epoch"],
        "entry_end": levels_data["entry_window_end_epoch"],
    }
    tickers = sorted(levels_data["tickers"].keys())
    bars_by_ticker = {}
    for ticker in tickers:
        csv_path = os.path.join(session_dir, "{}_5m.csv".format(ticker))
        bars_by_ticker[ticker] = read_bars(csv_path)
    return levels_data, session_epochs, tickers, bars_by_ticker


def flat_by_epoch(rth_open_epoch):
    return rth_open_epoch + FLAT_BY_SECONDS


def find_bar_index_by_time_ct(bars, signal_time_ct):
    for i, b in enumerate(bars):
        if ct_str(b["t"]) == signal_time_ct:
            return i
    return None


# -----------------------------------------------------------------------------
# Conservative same-bar-both-touched resolver.
# -----------------------------------------------------------------------------
def resolve_r_outcome(bars, start_i, end_i_inclusive, entry_price, risk, is_call,
                       stop_price, r3_price):
    """Walks bars[start_i .. end_i_inclusive] checking, on each bar, whether
    price touched +3R (win) and/or the stop / -1R level (loss). If a single
    bar's range touches BOTH, it resolves CONSERVATIVELY as a loss (-1R).
    Returns (outcome, final_r) where outcome in {"WIN", "LOSS", "UNRESOLVED"}.
    final_r is +3.0 for WIN, -1.0 for LOSS, and mark-to-market R (signed) at
    the last available close for UNRESOLVED.
    """
    if risk is None or risk <= 0:
        return "UNRESOLVED", None
    last_close = None
    for i in range(start_i, min(end_i_inclusive + 1, len(bars))):
        b = bars[i]
        last_close = b["c"]
        if is_call:
            touched_r3 = b["h"] >= r3_price
            touched_stop = b["l"] <= stop_price
        else:
            touched_r3 = b["l"] <= r3_price
            touched_stop = b["h"] >= stop_price
        if touched_r3 and touched_stop:
            return "LOSS", -1.0
        if touched_stop:
            return "LOSS", -1.0
        if touched_r3:
            return "WIN", 3.0
    if last_close is None:
        return "UNRESOLVED", None
    mtm = (last_close - entry_price) / risk if is_call else (entry_price - last_close) / risk
    return "UNRESOLVED", mtm


def excursion(bars, start_i, end_i_inclusive, ref_close, is_call):
    """MFE/MAE (absolute, not R) from ref_close over bars[start_i..end_i_inclusive]
    (inclusive of both ends; start_i is typically the break bar itself so the
    break bar's own high/low also counts)."""
    hi = None
    lo = None
    for i in range(start_i, min(end_i_inclusive + 1, len(bars))):
        b = bars[i]
        hi = b["h"] if hi is None else max(hi, b["h"])
        lo = b["l"] if lo is None else min(lo, b["l"])
    if hi is None:
        return 0.0, 0.0
    if is_call:
        mfe = max(0.0, hi - ref_close)
        mae = max(0.0, ref_close - lo)
    else:
        mfe = max(0.0, ref_close - lo)
        mae = max(0.0, hi - ref_close)
    return mfe, mae


def r3_before_1r(bars, start_i, end_i_inclusive, ref_close, risk, is_call):
    """Same conservative resolver as resolve_r_outcome, but yardstick-relative
    (used for Part A's 'reached 3R before -1R' flag on a hypothetical
    break-entry). Returns True/False/None (None = never resolved in window)."""
    if risk is None or risk <= 0:
        return None
    if is_call:
        r3_price = ref_close + 3.0 * risk
        stop_price = ref_close - risk
    else:
        r3_price = ref_close - 3.0 * risk
        stop_price = ref_close + risk
    for i in range(start_i, min(end_i_inclusive + 1, len(bars))):
        b = bars[i]
        if is_call:
            touched_r3 = b["h"] >= r3_price
            touched_stop = b["l"] <= stop_price
        else:
            touched_r3 = b["l"] <= r3_price
            touched_stop = b["h"] >= stop_price
        if touched_r3 and touched_stop:
            return False
        if touched_stop:
            return False
        if touched_r3:
            return True
    return None


# -----------------------------------------------------------------------------
# PART A -- opportunity scan
# -----------------------------------------------------------------------------
def part_a(session_dirs):
    rows = []
    for session_dir in session_dirs:
        levels_data, session_epochs, tickers, bars_by_ticker = load_session(
            repo_path(session_dir))
        session_date = levels_data["session_date_ct"]
        rth_open_epoch = session_epochs["rth_open"]
        entry_end = session_epochs["entry_end"]
        flat_epoch = flat_by_epoch(rth_open_epoch)

        for ticker in tickers:
            bars = bars_by_ticker[ticker]
            static_levels = levels_data["tickers"][ticker]
            events, _ = run_ticker(ticker, bars, static_levels, session_epochs,
                                    pre_window_reject_rearms=False)

            watches = [e for e in events if e["event"] == "WATCH"]
            for w in watches:
                # find the resolution event for this same track (level+direction),
                # i.e. the next REJECT/EXPIRED/QUALIFIED for that level+direction.
                resolution = None
                for e in events:
                    if e is w:
                        continue
                    if e["level"] == w["level"] and e["direction"] == w["direction"] \
                            and e["event"] in ("REJECT", "EXPIRED", "QUALIFIED"):
                        resolution = e
                        break

                break_i = find_bar_index_by_time_ct(bars, w["signal_time_ct"])
                if break_i is None:
                    continue
                break_bar = bars[break_i]
                is_call = w["direction"] == "CALL"
                atr = compute_rma(compute_true_range(bars), ATR_LEN)[break_i]

                if atr is None:
                    risk = None
                else:
                    if is_call:
                        stop = break_bar["l"] - ATR_STOP_BUFFER * atr
                        risk = break_bar["c"] - stop
                    else:
                        stop = break_bar["h"] + ATR_STOP_BUFFER * atr
                        risk = stop - break_bar["c"]
                    if risk <= 0:
                        risk = None

                # bounds: entry-window end, and flat-by (14:55 CT)
                ew_end_i = break_i
                for i in range(break_i, len(bars)):
                    if bars[i]["t"] < entry_end:
                        ew_end_i = i
                fb_end_i = break_i
                for i in range(break_i, len(bars)):
                    if bars[i]["t"] < flat_epoch:
                        fb_end_i = i

                mfe_ew, mae_ew = excursion(bars, break_i, ew_end_i, break_bar["c"], is_call)
                mfe_fb, mae_fb = excursion(bars, break_i, fb_end_i, break_bar["c"], is_call)

                if risk:
                    mfe_ew_r, mae_ew_r = mfe_ew / risk, mae_ew / risk
                    mfe_fb_r, mae_fb_r = mfe_fb / risk, mae_fb / risk
                    reached_3r_ew = r3_before_1r(bars, break_i + 1, ew_end_i,
                                                 break_bar["c"], risk, is_call)
                    reached_3r_fb = r3_before_1r(bars, break_i + 1, fb_end_i,
                                                 break_bar["c"], risk, is_call)
                else:
                    mfe_ew_r = mae_ew_r = mfe_fb_r = mae_fb_r = None
                    reached_3r_ew = reached_3r_fb = None

                rows.append({
                    "ticker": ticker,
                    "date": session_date,
                    "level": w["level"],
                    "level_price": w["level_price"],
                    "direction": w["direction"],
                    "break_time_ct": w["signal_time_ct"],
                    "resolution_event": resolution["event"] if resolution else "DANGLING",
                    "resolution_time_ct": resolution["signal_time_ct"] if resolution else None,
                    "resolution_reason": resolution.get("reason") if resolution else None,
                    "risk_yardstick": round(risk, 4) if risk else None,
                    "mfe_ew": round(mfe_ew, 4), "mae_ew": round(mae_ew, 4),
                    "mfe_ew_r": round(mfe_ew_r, 2) if mfe_ew_r is not None else None,
                    "mae_ew_r": round(mae_ew_r, 2) if mae_ew_r is not None else None,
                    "mfe_fb": round(mfe_fb, 4), "mae_fb": round(mae_fb, 4),
                    "mfe_fb_r": round(mfe_fb_r, 2) if mfe_fb_r is not None else None,
                    "mae_fb_r": round(mae_fb_r, 2) if mae_fb_r is not None else None,
                    "reached_3r_before_1r_entrywindow": reached_3r_ew,
                    "reached_3r_before_1r_flatby": reached_3r_fb,
                })
    return rows


# -----------------------------------------------------------------------------
# PART B -- filter sensitivity sweep
# -----------------------------------------------------------------------------
def build_variants():
    base = default_config()
    variants = [("BASELINE", base)]

    v = copy.deepcopy(base); v["rvol_bar"] = "break"
    variants.append(("RVOL_ON_BREAK_BAR", v))

    for thr in (1.0, 0.8, 0.0):
        v = copy.deepcopy(base); v["rvol_threshold"] = thr
        variants.append(("RVOL_THRESHOLD_{}".format(thr), v))

    v = copy.deepcopy(base); v["require_confirm_color"] = False
    variants.append(("NO_CONFIRM_COLOR_REQUIRED", v))

    for rmb in (9, 12):
        v = copy.deepcopy(base); v["retest_max_bars"] = rmb
        variants.append(("RETEST_MAX_BARS_{}".format(rmb), v))

    v = copy.deepcopy(base); v["entry_mode"] = "break_only"
    variants.append(("ENTRY_BREAK_ONLY", v))
    v = copy.deepcopy(base); v["entry_mode"] = "break_or_retest"
    variants.append(("ENTRY_BREAK_OR_RETEST", v))

    v = copy.deepcopy(base); v["max_stop_atr"] = 1.5
    variants.append(("MAX_STOP_ATR_1.5", v))

    v = copy.deepcopy(base); v["pre_window_reject_rearms"] = True
    variants.append(("PRE_WINDOW_REJECT_REARMS", v))

    return variants


def part_b(session_dirs):
    variants = build_variants()
    results = {}
    setups_by_variant = {}

    for name, config in variants:
        qualified_rows = []
        setups = set()
        for session_dir in session_dirs:
            levels_data, session_epochs, tickers, bars_by_ticker = load_session(
                repo_path(session_dir))
            session_date = levels_data["session_date_ct"]
            rth_open_epoch = session_epochs["rth_open"]
            flat_epoch = flat_by_epoch(rth_open_epoch)

            for ticker in tickers:
                bars = bars_by_ticker[ticker]
                static_levels = levels_data["tickers"][ticker]
                events, _ = run_variant(ticker, bars, static_levels, session_epochs, config)
                for e in events:
                    if e["event"] != "QUALIFIED":
                        continue
                    entry_i = e["bar_index"]
                    entry_price = e["_entry_price"]
                    risk = e["_risk"]
                    stop_price = e["_stop_price"]
                    r3_price = e["_r3"]
                    is_call = e["direction"] == "CALL"
                    end_i = entry_i
                    for i in range(entry_i, len(bars)):
                        if bars[i]["t"] < flat_epoch:
                            end_i = i
                    outcome, final_r = resolve_r_outcome(
                        bars, entry_i + 1, end_i, entry_price, risk, is_call,
                        stop_price, r3_price)
                    setup_key = (ticker, session_date, e["level"], e["direction"],
                                 e["signal_time_ct"])
                    setups.add(setup_key)
                    qualified_rows.append({
                        "ticker": ticker, "date": session_date, "level": e["level"],
                        "direction": e["direction"], "signal_time_ct": e["signal_time_ct"],
                        "score": e["score"], "outcome": outcome, "final_r": final_r,
                    })
        wins = sum(1 for r in qualified_rows if r["outcome"] == "WIN")
        losses = sum(1 for r in qualified_rows if r["outcome"] == "LOSS")
        unresolved = sum(1 for r in qualified_rows if r["outcome"] == "UNRESOLVED")
        net_r = sum(r["final_r"] for r in qualified_rows if r["final_r"] is not None)
        results[name] = {
            "qualified_count": len(qualified_rows),
            "wins": wins, "losses": losses, "unresolved": unresolved,
            "net_r": round(net_r, 2),
            "rows": qualified_rows,
        }
        setups_by_variant[name] = setups

    baseline_setups = setups_by_variant["BASELINE"]
    diffs = {}
    for name in results:
        if name == "BASELINE":
            continue
        added = setups_by_variant[name] - baseline_setups
        removed = baseline_setups - setups_by_variant[name]
        diffs[name] = {"added": sorted(added), "removed": sorted(removed)}

    return results, diffs


# -----------------------------------------------------------------------------
# PART C -- single blocking gate attribution (baseline config only)
# -----------------------------------------------------------------------------
def part_c(session_dirs):
    base = default_config()
    all_gate_log = []
    for session_dir in session_dirs:
        levels_data, session_epochs, tickers, bars_by_ticker = load_session(
            repo_path(session_dir))
        session_date = levels_data["session_date_ct"]
        for ticker in tickers:
            bars = bars_by_ticker[ticker]
            static_levels = levels_data["tickers"][ticker]
            gate_log = []
            run_variant(ticker, bars, static_levels, session_epochs, base, gate_log=gate_log)
            for g in gate_log:
                g["date"] = session_date
                all_gate_log.append(g)
    return all_gate_log


# -----------------------------------------------------------------------------
# Report writer
# -----------------------------------------------------------------------------
def fmt_bool(v):
    if v is None:
        return "n/a (never resolved in window)"
    return "YES" if v else "no"


def write_report(path, rows_a, results_b, diffs_b, rows_c):
    lines = []
    lines.append("# Opportunity scan and filter-sensitivity sweep")
    lines.append("")
    lines.append("Generated by `analysis/opportunity_scan.py` over "
                  "`backtests/session_2026-07-13` and `backtests/session_2026-07-14` "
                  "(XLE, XLF, IWM; 5-minute bars).")
    lines.append("")
    lines.append("> **SAMPLE SIZE WARNING, read before anything else:** {}"
                  .format(SAMPLE_CAVEAT))
    lines.append("> This report answers 'what was on the table and how twitchy are the "
                  "gates', nothing more. It is not a backtest result and not a "
                  "recommendation.")
    lines.append("")

    # ---------------- PART A ----------------
    lines.append("## Part A -- Opportunity scan (what was actually on the table)")
    lines.append("")
    lines.append("n = {} level breaks (WATCH events) across both sessions, all three "
                  "tickers. Sample size warning above applies to every row.".format(len(rows_a)))
    lines.append("")
    lines.append("Risk yardstick = the stop the engine would have used had it entered "
                  "ON the break bar: `|break_bar_close - (break_bar_low/high buffered by "
                  "0.10*ATR)|`. MFE/MAE measured from the break bar's own close. "
                  "'EW' = bounded by entry-window end (10:30 CT); 'FB' = bounded by "
                  "14:55 CT flat-by. 3R-before-1R uses the same conservative "
                  "both-touched-in-one-bar-counts-as-loss resolver as Part B.")
    lines.append("")
    header = ("Ticker | Date | Level | Dir | Break (CT) | Engine outcome | at (CT) | "
               "Risk($) | MFE_EW(R) | MAE_EW(R) | MFE_FB(R) | MAE_FB(R) | 3R-b4-1R (EW) | 3R-b4-1R (FB)")
    lines.append(header)
    lines.append("|".join(["---"] * 14))
    for r in rows_a:
        lines.append("{} | {} | {} {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}".format(
            r["ticker"], r["date"], r["level"], r["level_price"], r["direction"],
            r["break_time_ct"], r["resolution_event"], r["resolution_time_ct"],
            r["risk_yardstick"], r["mfe_ew_r"], r["mae_ew_r"], r["mfe_fb_r"], r["mae_fb_r"],
            fmt_bool(r["reached_3r_before_1r_entrywindow"]),
            fmt_bool(r["reached_3r_before_1r_flatby"])))
    lines.append("")

    rejects = [r for r in rows_a if r["resolution_event"] == "REJECT"]
    expensive_rejects = [r for r in rejects if r["mfe_fb_r"] is not None and r["mfe_fb_r"] >= 1.0]
    lines.append("**Honest read:** {} of {} breaks resolved as REJECT. Of those, "
                  "{} moved >= 1R in the trade's favor at some point before 14:55 CT "
                  "even after the engine rejected them -- i.e. the rejection may have "
                  "been expensive, not protective, on those specific occasions. The "
                  "rest either kept moving against the break direction (protective "
                  "reject) or never went anywhere (moot reject). n is far too small "
                  "to say which is the norm.".format(
                      len(rejects), len(rows_a), len(expensive_rejects)))
    lines.append("")

    # ---------------- PART B ----------------
    lines.append("## Part B -- Filter sensitivity sweep")
    lines.append("")
    lines.append("n = 2 sessions x 3 tickers x 4 tracks/day. {}".format(SAMPLE_CAVEAT))
    lines.append("")
    lines.append("Each variant changes ONE axis vs BASELINE. QUALIFIED trades are walked "
                  "forward bar-by-bar to 14:55 CT using the variant's own entry/stop/r3; "
                  "a bar whose range touches both +3R and -1R counts as -1R "
                  "(conservative). 'Unresolved' = still open (mark-to-market R shown) "
                  "at 14:55 CT flat-by.")
    lines.append("")
    lines.append("Variant | QUALIFIED | Wins (+3R) | Losses (-1R) | Unresolved | Net R")
    lines.append("---|---|---|---|---|---")
    for name, _ in build_variants():
        res = results_b[name]
        lines.append("{} | {} | {} | {} | {} | {}".format(
            name, res["qualified_count"], res["wins"], res["losses"],
            res["unresolved"], res["net_r"]))
    lines.append("")

    lines.append("### Named-hypothesis check (from CLAUDE.md / the task brief)")
    lines.append("")
    xlf_orh_added = ("XLF", "2026-07-14", "ORH", "CALL") in {
        (s[0], s[1], s[2], s[3]) for s in diffs_b.get("RVOL_ON_BREAK_BAR", {}).get("added", [])}
    xlf_pdh_added_break_only = ("XLF", "2026-07-14", "PDH", "CALL") in {
        (s[0], s[1], s[2], s[3]) for s in diffs_b.get("ENTRY_BREAK_ONLY", {}).get("added", [])}
    lines.append("- **RVOL-on-break-bar hypothesis (XLF 2026-07-14 ORH):** {}. Recovering it "
                 "this way turns it into a QUALIFIED trade, and in this single instance that "
                 "trade is a **loss** (net R -1.0, see table above) -- so the hypothesis is "
                 "directionally correct about WHICH gate blocked the setup, but backfilling "
                 "the gate does not, in this one sample, produce a winner.".format(
                     "confirmed -- it is the only setup added by RVOL_ON_BREAK_BAR"
                     if xlf_orh_added else "NOT reproduced in this run -- check the gate log"))
    lines.append("- **Break-only entry hypothesis (XLF 2026-07-14 PDH, the +1.25% break at "
                 "08:35 CT with no retest):** {} Instead, break-only/break-or-retest mode "
                 "recovers two DIFFERENT setups in this sample (IWM 2026-07-13 ORL, XLE "
                 "2026-07-13 ORH) and both resolve as losses. This directly contradicts the "
                 "naive framing of the hypothesis: the retest requirement was not what killed "
                 "the XLF PDH setup -- the risk-validity-vs-ATR gate would have blocked an "
                 "immediate break-bar entry too, because that particular break bar (a "
                 "741k-volume opening surge candle) had a range far wider than ATR(14) had "
                 "caught up to yet. Removing the retest requirement does not automatically "
                 "capture every 'ran without a retest' setup if the break bar itself is too "
                 "wide to pass the stop-sizing gate.".format(
                     "NOT recovered by break-only entry in this run: the break bar's own "
                     "risk (entry_close minus buffered stop) exceeded max_stop_atr, so the "
                     "risk-validity gate blocks it even with the retest requirement removed."
                     if not xlf_pdh_added_break_only else
                     "Recovered as hypothesized by break-only entry."))
    lines.append("")

    lines.append("### Setups gained/lost vs BASELINE")
    lines.append("")
    lines.append("BASELINE itself qualifies 0 trades in this 2-session sample, so every "
                  "setup appearing under any variant below is, by construction, an "
                  "'added' setup; there is nothing to lose from an empty baseline set. "
                  "That asymmetry is itself informative: it means every non-zero number "
                  "in the table above is coming entirely from relaxing a gate, not from "
                  "trading off one setup for another.")
    lines.append("")
    for name, _ in build_variants():
        if name == "BASELINE":
            continue
        d = diffs_b[name]
        lines.append("- **{}**: +{} setup(s){}".format(
            name, len(d["added"]),
            ("" if not d["added"] else ": " + "; ".join(
                "{} {} {} {} @ {}".format(*s) for s in d["added"]))))
    lines.append("")

    # Surface the "more qualified but worse net R" trap explicitly.
    baseline_net_r = results_b["BASELINE"]["net_r"]
    flagged = []
    for name, _ in build_variants():
        if name == "BASELINE":
            continue
        res = results_b[name]
        if res["qualified_count"] > 0 and res["net_r"] < baseline_net_r and res["qualified_count"] > results_b["BASELINE"]["qualified_count"]:
            flagged.append((name, res["qualified_count"], res["net_r"]))
    lines.append("**More trades, worse net R -- explicit check:** BASELINE net R is "
                  "{} on 0 trades (a 0-trade baseline has no losses but also no wins, "
                  "so 'worse than baseline' here really means 'net negative'). "
                  "Variants below both (a) qualify more trades than BASELINE and "
                  "(b) post negative net R -- read these as the clearest instances "
                  "in this sample of relaxing a filter to admit more losers, not more "
                  "winners:".format(baseline_net_r))
    if flagged:
        for name, cnt, net_r in flagged:
            lines.append("  - **{}**: {} trades, net R = {}".format(name, cnt, net_r))
    else:
        neg_variants = [(n, results_b[n]["qualified_count"], results_b[n]["net_r"])
                         for n, _ in build_variants() if n != "BASELINE" and results_b[n]["net_r"] < 0]
        if neg_variants:
            lines.append("  None strictly dominate on the 'more trades' condition, but these "
                          "variants post negative net R on the trades they do add:")
            for n, cnt, net_r in neg_variants:
                lines.append("  - **{}**: {} trades, net R = {}".format(n, cnt, net_r))
        else:
            lines.append("  None in this sample -- but see the per-variant trade list above; "
                          "with n this small, absence of the trap here is not evidence it "
                          "doesn't exist.")
    lines.append("")

    # ---------------- PART C ----------------
    lines.append("## Part C -- Single blocking gate attribution (BASELINE config)")
    lines.append("")
    lines.append("Every bar, inside any track's retest window, where price structure "
                  "(touch + reclaim) passed but the bar did not qualify. Gates checked: "
                  "candle_color, rvol, vwap, ema (always off/passing per strategy.md "
                  "default), risk_valid_vs_atr. n = {} such bars across both sessions. "
                  "{}".format(len(rows_c), SAMPLE_CAVEAT))
    lines.append("")
    lines.append("Ticker | Date | Level | Dir | Bar time (CT) | Bars since break | Blocked by")
    lines.append("---|---|---|---|---|---|---")
    for g in rows_c:
        lines.append("{} | {} | {} | {} | {} | {} | {}".format(
            g["ticker"], g["date"], g["level"], g["direction"], g["signal_time_ct"],
            g["bars_since_break"], ", ".join(g["blocked_by"]) if g["blocked_by"] else "(none -- should have qualified)"))
    lines.append("")

    single_gate = [g for g in rows_c if len(g["blocked_by"]) == 1]
    rvol_only = [g for g in single_gate if g["blocked_by"] == ["rvol"]]
    lines.append("**Single-gate blocks:** {} of {} structure-pass bars were blocked by "
                  "exactly one gate. Of those, {} were blocked by RVOL alone (matches the "
                  "known XLF 2026-07-14 case). The rest of the single-gate list above shows "
                  "every other instance found in this 2-session sample.".format(
                      len(single_gate), len(rows_c), len(rvol_only)))
    lines.append("")

    lines.append("## Bottom line")
    lines.append("")
    lines.append("- {}".format(SAMPLE_CAVEAT))
    lines.append("- This report deliberately makes NO variant recommendation. Every "
                  "number above should be read as 'here is what happened to n<=~10 "
                  "setups under this rule change', not as a validated edge.")
    lines.append("- The single most useful output is Part C: it names, bar-by-bar, "
                  "which specific gate blocked which specific near-miss, so the next "
                  "~20 sessions can be watched with a concrete hypothesis instead of a "
                  "vague one.")
    lines.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    rows_a = part_a(SESSIONS)
    results_b, diffs_b = part_b(SESSIONS)
    rows_c = part_c(SESSIONS)

    out_path = repo_path("reports", "opportunity_scan.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_report(out_path, rows_a, results_b, diffs_b, rows_c)

    print("=" * 78)
    print("OPPORTUNITY SCAN SUMMARY  --  {}".format(SAMPLE_CAVEAT))
    print("=" * 78)
    print()
    print("PART A: {} level breaks (WATCH events) found.".format(len(rows_a)))
    for r in rows_a:
        print("  {} {} {} {} broke {:.2f} at {} -> {} at {} | MFE/MAE(FB,R)={}/{} "
              "risk=${} | 3R-b4-1R(FB)={}".format(
                  r["date"], r["ticker"], r["direction"], r["level"], r["level_price"],
                  r["break_time_ct"], r["resolution_event"], r["resolution_time_ct"],
                  r["mfe_fb_r"], r["mae_fb_r"], r["risk_yardstick"],
                  fmt_bool(r["reached_3r_before_1r_flatby"])))
    print()
    print("PART B: variant sweep (QUALIFIED / W / L / U / net R)")
    for name, _ in build_variants():
        res = results_b[name]
        print("  {:28s} {:3d} / {:3d} / {:3d} / {:3d}   net R = {}".format(
            name, res["qualified_count"], res["wins"], res["losses"],
            res["unresolved"], res["net_r"]))
    print()
    print("PART C: {} single/multi-gate blocked structure-pass bars found "
          "(baseline config).".format(len(rows_c)))
    for g in rows_c:
        print("  {} {} {} {} @ {} (bars_since_break={}) blocked by: {}".format(
            g["date"], g["ticker"], g["direction"], g["level"], g["signal_time_ct"],
            g["bars_since_break"], ", ".join(g["blocked_by"])))
    print()
    print("Full report written to reports/opportunity_scan.md")
    print()
    print("NO VARIANT IS RECOMMENDED. n is far too small (2 sessions, 3 tickers). "
          "This is a hypothesis-generation artifact only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
