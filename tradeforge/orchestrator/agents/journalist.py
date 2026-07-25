"""orchestrator/agents/journalist.py — the JOURNALIST agent (MASTER_PLAN.md §4).

Auto-journals every closed trade, renders a per-trade chart PNG, writes
premarket/EOD digests, and pushes them through the NOTIFY tool. Model tier:
HAIKU (routine, cost-tracked) — but the implementation is fully DETERMINISTIC:
it produces a solid templated journal entry and narrative WITHOUT any LLM call,
so it works offline. The .md notes an LLM *may* enrich the narrative later; the
code never depends on one.

FIREWALL (MASTER_PLAN §6): the journalist reads everything and writes ONLY under
``journal/`` (per-trade md+json artifacts, the search index, digests, charts,
the notifications log). It never touches live config or ``limits.yaml`` and emits
no live events — it only writes journal artifacts.

Per-trade artifact, for a closed trade, contains:
  * frame card        — symbol, strategy, setup grade, levels (PDH/PDL/...),
                        planned entry/stop/target, R (reward:risk).
  * slippage          — intended-vs-actual: actual fill price − intended.
  * MFE / MAE         — passed through from the position record.
  * plan-adherence    — DETERMINISTIC boolean flags (entered_in_window?,
                        stop_at_planned_level?, exited_per_plan?, held_past_eod?).
  * 3-line narrative   — a deterministic template summary.

Public API
----------
    Trade                          — the input dataclass (the closed-trade view).
    JournalEntry                   — the produced entry (frame card + flags + ...).
    Journalist(journal_dir=..., notifier=..., market_db=...)
        journal_trade(trade) -> JournalEntry
        chart_for_trade(trade, out_path) -> str
        premarket_digest(date, *, send=False) -> str
        eod_digest(date, *, send=False) -> str
        search(query) -> list[dict]
        attach_to_bus(bus)         — subscribe POSITION_CLOSED -> auto-journal.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any, Callable, Optional

# Default journal root (write-only zone for this agent).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_JOURNAL_DIR = _REPO_ROOT / "journal"
DEFAULT_MARKET_DB = _REPO_ROOT / "data" / "duckdb" / "market.duckdb"

# Equity RTH entry window (ET), for the entered_in_window? adherence flag.
_RTH_START = time(9, 30)
_RTH_END = time(16, 0)


# --------------------------------------------------------------------------- #
# Input + output shapes                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Trade:
    """The closed-trade view the journalist journals.

    Built by the bus subscriber from POSITION_CLOSED (+ the order/fill/position
    records), or constructed directly in tests. Everything is plain JSON-able.

    Plan side (what we INTENDED):
        symbol, strategy, side ("long"/"short"), setup_grade ("B"/"A"/"A+"),
        planned_entry, planned_stop, planned_target, levels (PDH/PDL/...).
    Actual side (what HAPPENED):
        entry_price, exit_price, qty, realized_pnl, mfe, mae, slippage,
        entry_ts, exit_ts, exit_reason (e.g. "target"/"stop"/"trail"/"time"/
        "eod"/"strategy"), held_past_eod (optional explicit flag).
    """

    symbol: str
    strategy: str = ""
    side: str = "long"
    setup_grade: str = ""
    trade_id: str = ""

    # plan
    planned_entry: Optional[float] = None
    planned_stop: Optional[float] = None
    planned_target: Optional[float] = None
    levels: dict = field(default_factory=dict)
    entry_window: Optional[tuple] = None  # (start, end) ET times or None -> RTH

    # actual
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    qty: Optional[float] = None
    realized_pnl: Optional[float] = None
    mfe: Optional[float] = None
    mae: Optional[float] = None
    slippage: Optional[float] = None  # actual entry − intended (signed); derived if None
    entry_ts: Optional[Any] = None
    exit_ts: Optional[Any] = None
    exit_reason: str = ""
    held_past_eod: Optional[bool] = None

    # optional: precomputed session bars for the chart (list of OHLC dicts)
    bars: Optional[Any] = None

    def __post_init__(self) -> None:
        if not self.trade_id:
            base = self.entry_ts or self.exit_ts or datetime.utcnow()
            stamp = _ts_compact(base)
            self.trade_id = f"{self.symbol}_{self.strategy or 'na'}_{stamp}".replace(" ", "")


@dataclass
class FrameCard:
    """The frame card: the at-a-glance plan + structure of the trade."""

    symbol: str
    strategy: str
    side: str
    setup_grade: str
    levels: dict
    planned_entry: Optional[float]
    planned_stop: Optional[float]
    planned_target: Optional[float]
    r_planned: Optional[float]  # planned reward:risk


@dataclass
class PlanAdherence:
    """Deterministic plan-adherence flags (computed, never LLM)."""

    entered_in_window: bool
    stop_at_planned_level: bool
    exited_per_plan: bool
    held_past_eod: bool


@dataclass
class JournalEntry:
    """One archived journal entry for a closed trade."""

    trade_id: str
    date: str
    symbol: str
    strategy: str
    frame_card: FrameCard
    slippage: Optional[float]
    mfe: Optional[float]
    mae: Optional[float]
    realized_pnl: Optional[float]
    r_realized: Optional[float]
    adherence: PlanAdherence
    narrative: str  # the 3-line narrative

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# small helpers                                                                #
# --------------------------------------------------------------------------- #
def _as_dt(value) -> Optional[datetime]:
    """Best-effort coerce a value to a datetime (ISO string / datetime / None)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _ts_compact(value) -> str:
    dt = _as_dt(value) or datetime.utcnow()
    return dt.strftime("%Y%m%dT%H%M%S")


def _date_str(value) -> str:
    dt = _as_dt(value)
    return (dt or datetime.utcnow()).strftime("%Y-%m-%d")


def _fmt_num(x, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _signed(x, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):+.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


# --------------------------------------------------------------------------- #
# deterministic computations                                                   #
# --------------------------------------------------------------------------- #
def compute_slippage(trade: Trade) -> Optional[float]:
    """Intended-vs-actual entry slippage = actual entry − intended (signed).

    Uses the explicit ``trade.slippage`` if provided (e.g. straight from the
    fill record); otherwise derives it from entry_price − planned_entry.
    """
    if trade.slippage is not None:
        return float(trade.slippage)
    if trade.entry_price is None or trade.planned_entry is None:
        return None
    return float(trade.entry_price) - float(trade.planned_entry)


def planned_r(trade: Trade) -> Optional[float]:
    """Planned reward:risk = |target − entry| / |entry − stop| (None if undef)."""
    e, s, t = trade.planned_entry, trade.planned_stop, trade.planned_target
    if e is None or s is None or t is None:
        return None
    risk = abs(float(e) - float(s))
    if risk <= 1e-12:
        return None
    return abs(float(t) - float(e)) / risk


def realized_r(trade: Trade) -> Optional[float]:
    """Realized R = (exit − entry)/risk, signed by side (None if undefined)."""
    e, s = trade.planned_entry, trade.planned_stop
    actual_entry = trade.entry_price if trade.entry_price is not None else e
    if actual_entry is None or s is None or trade.exit_price is None:
        return None
    risk = abs(float(actual_entry) - float(s))
    if risk <= 1e-12:
        return None
    direction = 1.0 if str(trade.side).lower() in ("buy", "long") else -1.0
    return direction * (float(trade.exit_price) - float(actual_entry)) / risk


def _in_window(ts, window) -> bool:
    """True if ``ts`` (ET-naive ok) falls within the entry window."""
    dt = _as_dt(ts)
    if dt is None:
        return False
    start, end = (window or (_RTH_START, _RTH_END))
    if isinstance(start, str):
        start = time.fromisoformat(start)
    if isinstance(end, str):
        end = time.fromisoformat(end)
    return start <= dt.time() < end


def compute_adherence(trade: Trade) -> PlanAdherence:
    """Deterministic plan-adherence flags (no LLM).

    - entered_in_window?    entry_ts inside the entry window (default RTH).
    - stop_at_planned_level? the order's stop sat at the planned stop level
      (within a small tolerance) — i.e. we did not move/skip the stop.
    - exited_per_plan?       exit_reason is a planned exit (target/stop/trail/
      time/eod) AND not a discretionary override / no exit recorded.
    - held_past_eod?         the position was carried past session close.
    """
    # entered_in_window
    entered_in_window = _in_window(trade.entry_ts, trade.entry_window) \
        if trade.entry_ts is not None else False

    # stop_at_planned_level: compare the actual stop the position used against
    # the planned stop. We treat the planned stop as authoritative; a missing
    # planned stop or a clearly different protective level flags False.
    stop_ok = False
    if trade.planned_stop is not None:
        # If the trade carries no separate "actual stop", the planned stop is
        # assumed used UNLESS the exit was a stop/trail hit at a materially
        # different price (which would reveal a relocated/violated stop).
        tol = max(abs(float(trade.planned_stop)) * 0.005, 0.01)
        if trade.exit_reason.lower() in ("stop",) and trade.exit_price is not None:
            stop_ok = abs(float(trade.exit_price) - float(trade.planned_stop)) <= tol
        else:
            # Non-stop exit: the planned stop was in force and never breached.
            stop_ok = True

    # exited_per_plan
    planned_exits = {"target", "stop", "trail", "trailing", "time", "time_stop",
                     "eod", "session_flatten", "tp1", "be", "breakeven"}
    exited_per_plan = trade.exit_reason.lower() in planned_exits

    # held_past_eod
    if trade.held_past_eod is not None:
        held_past_eod = bool(trade.held_past_eod)
    else:
        entry_dt, exit_dt = _as_dt(trade.entry_ts), _as_dt(trade.exit_ts)
        if entry_dt is not None and exit_dt is not None:
            held_past_eod = exit_dt.date() > entry_dt.date()
        else:
            held_past_eod = False

    return PlanAdherence(
        entered_in_window=bool(entered_in_window),
        stop_at_planned_level=bool(stop_ok),
        exited_per_plan=bool(exited_per_plan),
        held_past_eod=bool(held_past_eod),
    )


def build_frame_card(trade: Trade) -> FrameCard:
    return FrameCard(
        symbol=trade.symbol,
        strategy=trade.strategy,
        side=trade.side,
        setup_grade=trade.setup_grade or "n/a",
        levels=dict(trade.levels or {}),
        planned_entry=trade.planned_entry,
        planned_stop=trade.planned_stop,
        planned_target=trade.planned_target,
        r_planned=planned_r(trade),
    )


def build_narrative(trade: Trade, frame: FrameCard, adherence: PlanAdherence,
                    slippage, r_real) -> str:
    """A deterministic 3-line narrative (template; an LLM MAY enrich later).

    Line 1: the setup — what was armed and the plan.
    Line 2: the execution — entry/exit, slippage, MFE/MAE, R.
    Line 3: the verdict — plan adherence + outcome.
    """
    grade = frame.setup_grade
    side = str(trade.side).lower()
    pnl = trade.realized_pnl
    outcome = "scratch"
    if pnl is not None:
        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "scratch")

    line1 = (
        f"{trade.symbol} {side} on {trade.strategy or 'n/a'} "
        f"(grade {grade}): planned entry {_fmt_num(frame.planned_entry)}, "
        f"stop {_fmt_num(frame.planned_stop)}, target {_fmt_num(frame.planned_target)}"
        f" ({_fmt_num(frame.r_planned, 2)}R)."
    )
    line2 = (
        f"Filled {_fmt_num(trade.entry_price)} (slippage {_signed(slippage)}), "
        f"exited {_fmt_num(trade.exit_price)} via {trade.exit_reason or 'n/a'}; "
        f"MFE {_signed(trade.mfe)} / MAE {_signed(trade.mae)}, realized "
        f"{_signed(pnl)} ({_signed(r_real, 2)}R)."
    )
    flags = []
    flags.append("in-window" if adherence.entered_in_window else "OUT-OF-WINDOW")
    flags.append("stop@plan" if adherence.stop_at_planned_level else "STOP-OFF-PLAN")
    flags.append("exit-per-plan" if adherence.exited_per_plan else "EXIT-OFF-PLAN")
    if adherence.held_past_eod:
        flags.append("HELD-PAST-EOD")
    adhered = all([
        adherence.entered_in_window,
        adherence.stop_at_planned_level,
        adherence.exited_per_plan,
        not adherence.held_past_eod,
    ])
    verdict = "plan followed" if adhered else "PLAN DEVIATION"
    line3 = f"Verdict: {outcome}, {verdict} [{', '.join(flags)}]."
    return "\n".join([line1, line2, line3])


# --------------------------------------------------------------------------- #
# Journalist                                                                   #
# --------------------------------------------------------------------------- #
class Journalist:
    """The journalist agent: journal trades, render charts, write digests.

    Args:
        journal_dir: write-only journal root (default ``journal/``).
        notifier: a callable ``notify(message, *, title=None, channel=None)``
            used to deliver digests. Defaults to :func:`orchestrator.tools.notify.notify`.
        market_db: path to ``market.duckdb`` for chart bars + digest levels.
            Optional; chart/digest degrade gracefully if it is absent.
        orderbook: optional OrderBook for the bus subscriber to enrich a closed
            trade with the fill/position records. Injectable for tests.
    """

    INDEX_NAME = "index.jsonl"

    def __init__(
        self,
        journal_dir: str | Path | None = None,
        notifier: Optional[Callable[..., str]] = None,
        market_db: str | Path | None = None,
        orderbook=None,
    ):
        self.journal_dir = Path(journal_dir) if journal_dir else DEFAULT_JOURNAL_DIR
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.market_db = Path(market_db) if market_db else DEFAULT_MARKET_DB
        self.orderbook = orderbook
        if notifier is None:
            from orchestrator.tools.notify import notify as _notify
            notifier = _notify
        self._notify = notifier

    # ------------------------------------------------------------------ #
    # core: journal one trade                                            #
    # ------------------------------------------------------------------ #
    def journal_trade(self, trade: Trade) -> JournalEntry:
        """Produce + archive a :class:`JournalEntry` for a closed trade.

        Computes the frame card, slippage, MFE/MAE passthrough, deterministic
        plan-adherence flags, and the 3-line narrative; writes the entry as BOTH
        Markdown and JSON under ``journal/<date>/<trade_id>.{md,json}`` and adds
        it to the searchable index. Returns the entry.
        """
        if not isinstance(trade, Trade):
            trade = Trade(**trade) if isinstance(trade, dict) else trade

        frame = build_frame_card(trade)
        slippage = compute_slippage(trade)
        adherence = compute_adherence(trade)
        r_real = realized_r(trade)
        narrative = build_narrative(trade, frame, adherence, slippage, r_real)

        entry = JournalEntry(
            trade_id=trade.trade_id,
            date=_date_str(trade.exit_ts or trade.entry_ts),
            symbol=trade.symbol,
            strategy=trade.strategy,
            frame_card=frame,
            slippage=slippage,
            mfe=trade.mfe,
            mae=trade.mae,
            realized_pnl=trade.realized_pnl,
            r_realized=r_real,
            adherence=adherence,
            narrative=narrative,
        )

        self._archive(entry, trade)
        return entry

    # ------------------------------------------------------------------ #
    # archive: md + json + index                                        #
    # ------------------------------------------------------------------ #
    def _entry_dir(self, date: str) -> Path:
        d = self.journal_dir / date
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _archive(self, entry: JournalEntry, trade: Trade) -> dict:
        """Write the JSON + Markdown artifacts and append to the index."""
        d = self._entry_dir(entry.date)
        json_path = d / f"{entry.trade_id}.json"
        md_path = d / f"{entry.trade_id}.md"

        payload = entry.to_dict()
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self._render_markdown(entry), encoding="utf-8")

        self._append_index(entry, json_path, md_path)
        return {"json": str(json_path), "md": str(md_path)}

    def _render_markdown(self, entry: JournalEntry) -> str:
        fc = entry.frame_card
        ad = entry.adherence

        def chk(b: bool) -> str:
            return "[x]" if b else "[ ]"

        levels_md = "\n".join(
            f"- **{k}**: {_fmt_num(v)}" for k, v in (fc.levels or {}).items()
        ) or "- (none)"

        return "\n".join([
            f"# {entry.symbol} — {entry.strategy or 'n/a'} — {entry.date}",
            "",
            f"_Trade ID: `{entry.trade_id}`_",
            "",
            "## Frame card",
            f"- **Symbol**: {fc.symbol}",
            f"- **Strategy**: {fc.strategy or 'n/a'}",
            f"- **Side**: {fc.side}",
            f"- **Setup grade**: {fc.setup_grade}",
            f"- **Planned entry**: {_fmt_num(fc.planned_entry)}",
            f"- **Planned stop**: {_fmt_num(fc.planned_stop)}",
            f"- **Planned target**: {_fmt_num(fc.planned_target)}",
            f"- **Planned R**: {_fmt_num(fc.r_planned)}",
            "",
            "### Levels",
            levels_md,
            "",
            "## Execution",
            f"- **Slippage (actual − intended)**: {_signed(entry.slippage)}",
            f"- **MFE**: {_signed(entry.mfe)}",
            f"- **MAE**: {_signed(entry.mae)}",
            f"- **Realized P&L**: {_signed(entry.realized_pnl)}",
            f"- **Realized R**: {_fmt_num(entry.r_realized)}",
            "",
            "## Plan adherence",
            f"- {chk(ad.entered_in_window)} entered in window",
            f"- {chk(ad.stop_at_planned_level)} stop at planned level",
            f"- {chk(ad.exited_per_plan)} exited per plan",
            f"- {chk(not ad.held_past_eod)} did not hold past EOD",
            "",
            "## Narrative",
            entry.narrative,
            "",
        ])

    def _append_index(self, entry: JournalEntry, json_path: Path, md_path: Path) -> None:
        """Append a compact searchable record to ``journal/index.jsonl``."""
        rec = {
            "trade_id": entry.trade_id,
            "date": entry.date,
            "symbol": entry.symbol,
            "strategy": entry.strategy,
            "setup_grade": entry.frame_card.setup_grade,
            "side": entry.frame_card.side,
            "realized_pnl": entry.realized_pnl,
            "r_realized": entry.r_realized,
            "narrative": entry.narrative,
            "json_path": str(json_path),
            "md_path": str(md_path),
        }
        idx = self.journal_dir / self.INDEX_NAME
        with idx.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    # ------------------------------------------------------------------ #
    # search                                                             #
    # ------------------------------------------------------------------ #
    def search(self, query: str) -> list[dict]:
        """Search the archived journal for ``query`` (case-insensitive substring).

        Reads the ``journal/index.jsonl`` index first (fast). Matches across the
        trade_id, symbol, strategy, side, grade, date and narrative fields. An
        empty/whitespace query returns ALL entries. Newest-first.
        """
        idx = self.journal_dir / self.INDEX_NAME
        if not idx.exists():
            return []
        q = (query or "").strip().lower()
        rows: list[dict] = []
        for line in idx.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not q:
                rows.append(rec)
                continue
            hay = " ".join(str(rec.get(k, "")) for k in (
                "trade_id", "symbol", "strategy", "side", "setup_grade",
                "date", "narrative",
            )).lower()
            if q in hay:
                rows.append(rec)
        rows.reverse()  # newest-first (index is append-order)
        return rows

    # ------------------------------------------------------------------ #
    # per-trade chart                                                    #
    # ------------------------------------------------------------------ #
    def chart_for_trade(self, trade: Trade, out_path: str | Path) -> str:
        """Render the per-trade chart PNG (candles + levels + fills). Returns path.

        Uses ``trade.bars`` if present; otherwise tries to pull the session's
        bars from ``market.duckdb`` for the symbol/date. If no bars are
        available the chart still renders (levels + fill markers only).
        """
        from reporting.charts import trade_chart

        if not isinstance(trade, Trade):
            trade = Trade(**trade) if isinstance(trade, dict) else trade

        bars = trade.bars
        if bars is None:
            bars = self._load_session_bars(trade)

        title = (
            f"{trade.symbol} {trade.strategy or ''} "
            f"{_date_str(trade.exit_ts or trade.entry_ts)}"
        ).strip()
        entry = {"price": trade.entry_price} if trade.entry_price is not None else None
        exit_ = {"price": trade.exit_price} if trade.exit_price is not None else None

        return trade_chart(
            bars, out_path, title=title,
            levels=dict(trade.levels or {}),
            entry=entry, exit=exit_,
            stop=trade.planned_stop, target=trade.planned_target,
        )

    def _load_session_bars(self, trade: Trade):
        """Best-effort: load the trade's session bars from market.duckdb.

        Returns a list of OHLC dicts or None. Never raises — a missing DB / table
        just yields None so the chart degrades gracefully.
        """
        if not self.market_db or not Path(self.market_db).exists():
            return None
        date = _date_str(trade.exit_ts or trade.entry_ts)
        try:
            import duckdb  # lazy

            con = duckdb.connect(str(self.market_db), read_only=True)
            try:
                rows = con.execute(
                    "SELECT open, high, low, close FROM bars "
                    "WHERE symbol = ? AND CAST(ts_utc AS DATE) = CAST(? AS DATE) "
                    "ORDER BY ts_utc ASC",
                    [trade.symbol, date],
                ).fetchall()
            finally:
                con.close()
        except Exception:  # noqa: BLE001 — chart degrades to levels-only
            return None
        if not rows:
            return None
        return [{"open": r[0], "high": r[1], "low": r[2], "close": r[3]} for r in rows]

    # ------------------------------------------------------------------ #
    # digests                                                            #
    # ------------------------------------------------------------------ #
    def premarket_digest(self, date, *, send: bool = False,
                         armed: Optional[list] = None) -> str:
        """A concise premarket digest: armed strategies, levels, watchlist focus.

        ``armed`` is an optional list of dicts ``{"symbol", "strategy", "levels"}``
        describing what is armed for the session; when omitted the digest reports
        the levels available in ``market.duckdb`` for the date. Suitable for
        iMessage. Sends through the notifier when ``send=True``.
        """
        d = _date_str(date)
        lines = [f"PREMARKET {d}"]

        if armed:
            lines.append(f"Armed ({len(armed)}):")
            for a in armed:
                lv = a.get("levels") or {}
                lv_s = " ".join(f"{k} {_fmt_num(v)}" for k, v in lv.items()) or "—"
                lines.append(
                    f"  {a.get('symbol','?')} {a.get('strategy','')}: {lv_s}"
                )
        else:
            lv_rows = self._levels_for_date(d)
            if lv_rows:
                lines.append("Levels:")
                for r in lv_rows:
                    lines.append(
                        f"  {r['symbol']}: PDH {_fmt_num(r['pdh'])} / "
                        f"PDL {_fmt_num(r['pdl'])}"
                        + (f" NTZ [{_fmt_num(r['ntz_low'])},{_fmt_num(r['ntz_high'])}]"
                           if r.get("ntz_valid") else "")
                    )
            else:
                lines.append("No armed strategies / levels on file.")

        text = "\n".join(lines)
        if send:
            self._notify(text, title=f"Premarket {d}")
        return text

    def eod_digest(self, date, *, send: bool = False) -> str:
        """A concise EOD digest: trades, P&L, plan-adherence summary, tomorrow's
        levels. Built from the day's archived journal entries. Suitable for
        iMessage. Sends through the notifier when ``send=True``.
        """
        d = _date_str(date)
        entries = [r for r in self.search("") if r.get("date") == d]

        n = len(entries)
        total_pnl = sum(
            float(r["realized_pnl"]) for r in entries
            if r.get("realized_pnl") is not None
        )
        wins = sum(
            1 for r in entries
            if r.get("realized_pnl") is not None and float(r["realized_pnl"]) > 0
        )
        # Plan-adherence summary: count deviations flagged in the narratives.
        deviations = sum(1 for r in entries if "PLAN DEVIATION" in str(r.get("narrative", "")))

        lines = [f"EOD {d}"]
        lines.append(f"Trades: {n} | Wins: {wins} | Net P&L: {_signed(total_pnl)}")
        lines.append(
            f"Plan adherence: {n - deviations}/{n} clean"
            if n else "Plan adherence: no trades"
        )
        if entries:
            lines.append("By trade:")
            for r in entries:
                lines.append(
                    f"  {r['symbol']} {r.get('strategy','')}: "
                    f"{_signed(r.get('realized_pnl'))} "
                    f"({_fmt_num(r.get('r_realized'))}R)"
                )

        # Tomorrow's levels (today's level rows roll forward as the watch set).
        lv_rows = self._levels_for_date(d)
        if lv_rows:
            lines.append("Tomorrow's levels:")
            for r in lv_rows:
                lines.append(
                    f"  {r['symbol']}: PDH {_fmt_num(r['pdh'])} / PDL {_fmt_num(r['pdl'])}"
                )

        text = "\n".join(lines)
        if send:
            self._notify(text, title=f"EOD {d}")
        return text

    def _levels_for_date(self, date: str) -> list[dict]:
        """Pull level rows for ``date`` from market.duckdb (best-effort, [] if none)."""
        if not self.market_db or not Path(self.market_db).exists():
            return []
        try:
            import duckdb  # lazy

            con = duckdb.connect(str(self.market_db), read_only=True)
            try:
                rows = con.execute(
                    "SELECT symbol, pdh, pdl, pmh, pml, ntz_low, ntz_high, "
                    "ntz_valid FROM levels WHERE CAST(session_date AS DATE) = "
                    "CAST(? AS DATE) ORDER BY symbol",
                    [date],
                ).fetchall()
            finally:
                con.close()
        except Exception:  # noqa: BLE001
            return []
        cols = ["symbol", "pdh", "pdl", "pmh", "pml", "ntz_low", "ntz_high", "ntz_valid"]
        return [dict(zip(cols, r)) for r in rows]

    # ------------------------------------------------------------------ #
    # bus subscription: POSITION_CLOSED -> auto-journal                  #
    # ------------------------------------------------------------------ #
    def attach_to_bus(self, bus) -> None:
        """Subscribe to POSITION_CLOSED on ``bus`` to auto-journal each trade.

        ``bus`` is any object exposing ``subscribe(types, handler)`` (the real
        :class:`EventBus` or a fake). The handler builds a :class:`Trade` from
        the event payload (enriching from the injected OrderBook when available)
        and calls :meth:`journal_trade`. Writes journal artifacts only — emits
        nothing back onto the bus (FIREWALL).
        """
        from orchestrator.events import EventType

        bus.subscribe([EventType.POSITION_CLOSED], self._on_position_closed)

    def _on_position_closed(self, event) -> None:
        """Bus handler: turn a POSITION_CLOSED event into a journal entry."""
        data = getattr(event, "data", None) or {}
        trade = self._trade_from_event(data, getattr(event, "ts_utc", None))
        if trade is not None:
            self.journal_trade(trade)

    def _trade_from_event(self, data: dict, ts) -> Optional[Trade]:
        """Map a POSITION_CLOSED payload (+ OrderBook records) to a Trade.

        The gateway emits POSITION_CLOSED as ``{"symbol", **exit_intent,
        **closed}`` where ``closed`` has ``realized_pnl``. We pull MFE/MAE and
        avg_price from the OrderBook position when available; the exit reason and
        plan levels ride on the exit-intent dict the fast loop attached.
        """
        symbol = data.get("symbol")
        if not symbol:
            return None

        mfe = data.get("mfe")
        mae = data.get("mae")
        entry_price = data.get("avg_price") or data.get("entry_price")

        # Enrich from the OrderBook position record if we have one.
        if self.orderbook is not None:
            try:
                pos = self.orderbook.position_for_symbol(symbol)
            except Exception:  # noqa: BLE001
                pos = None
            if pos is not None:
                mfe = mfe if mfe is not None else pos.mfe
                mae = mae if mae is not None else pos.mae
                entry_price = entry_price if entry_price is not None else pos.avg_price

        return Trade(
            symbol=symbol,
            strategy=data.get("strategy", ""),
            side=data.get("side", "long"),
            setup_grade=data.get("setup_grade", ""),
            planned_entry=data.get("planned_entry", data.get("intended_price")),
            planned_stop=data.get("planned_stop", data.get("stop_price")),
            planned_target=data.get("planned_target", data.get("target")),
            levels=data.get("levels", {}) or {},
            entry_price=entry_price,
            exit_price=data.get("exit_price", data.get("fill_price", data.get("price"))),
            qty=data.get("qty"),
            realized_pnl=data.get("realized_pnl", data.get("pnl")),
            mfe=mfe,
            mae=mae,
            slippage=data.get("slippage"),
            entry_ts=data.get("entry_ts"),
            exit_ts=data.get("exit_ts", ts),
            exit_reason=data.get("exit_reason", data.get("reason", "")),
            held_past_eod=data.get("held_past_eod"),
        )
