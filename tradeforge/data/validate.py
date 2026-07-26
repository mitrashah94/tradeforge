"""TradeForge levels validation printer (P1 gate helper).

A runnable, read-only utility that prints the most recent computed levels for a
symbol so they can be EYEBALL-VALIDATED against TradingView (MASTER_PLAN §5 / P1
gate: "eyeball-validate 10 sessions vs TradingView").

This module contains NO trading or strategy logic. It only reads the ``levels``
table (built by ``data.levels.build_all`` from ingested bars) and renders it.

Usage::

    python -m data.validate --symbol QQQ --n 10
    python -m data.validate --symbol SPY --n 10 --db data/duckdb/market.duckdb

It is safe to run before the DB is populated: it prints a clear "populate first"
message and returns without raising.
"""

from __future__ import annotations

import argparse

import data.schema as schema

# ``rich`` gives a nice boxed table; fall back to plain text if unavailable so
# this utility works in a bare environment (another agent manages venv deps).
try:
    from rich.console import Console
    from rich.table import Table

    _HAVE_RICH = True
except ImportError:  # pragma: no cover - exercised only without rich installed
    _HAVE_RICH = False


# Columns selected from the levels table, in display order (oldest -> newest is
# applied to the *rows*; this is the per-row column layout).
_QUERY = """
    SELECT
        session_date,
        pdh,
        pdl,
        pmh,
        pml,
        ntz_low,
        ntz_high,
        ntz_valid,
        atr14
    FROM levels
    WHERE symbol = ?
      AND session_type = ?
    ORDER BY session_date DESC
    LIMIT ?
"""


def _fmt_price(value) -> str:
    """Round a price-like value to 2dp, or em-dash if missing."""
    if value is None:
        return "—"
    try:
        return "{:.2f}".format(float(value))
    except (TypeError, ValueError):
        return "—"


def _fmt_ntz(low, high, valid) -> str:
    """Format the no-trade zone as 'lo–hi' when valid, else em-dash."""
    if not valid or low is None or high is None:
        return "—"
    return "{}–{}".format(_fmt_price(low), _fmt_price(high))


def _table_missing_message(db_path: str, symbol: str) -> str:
    """Human-readable guidance when the levels table is empty or absent."""
    return (
        "No levels found for symbol '{symbol}' in '{db}'.\n"
        "The market database has not been populated yet. To populate it:\n"
        "  1. Ingest bars:   python -m data.pipelines.alpaca_ingest --days 5\n"
        "  2. Build levels:  python -m data.levels\n"
        "Then re-run:        python -m data.validate --symbol {symbol}\n"
    ).format(symbol=symbol, db=db_path)


def print_levels(
    symbol: str = "QQQ",
    n: int = 10,
    db_path: str = schema.DEFAULT_DB_PATH,
) -> None:
    """Print the ``n`` most recent RTH levels rows for ``symbol``.

    Rows are queried newest-first then displayed oldest->newest so the table
    reads chronologically (top = older session, bottom = most recent), which
    matches scrolling a TradingView chart left-to-right.

    If the ``levels`` table is missing or has no rows for ``symbol``, a clear
    message is printed explaining how to populate the DB and the function
    returns without raising.
    """
    symbol = symbol.upper()
    # Equities are evaluated on the regular-session calendar; crypto would use
    # 'crypto_utc', but the eyeball-vs-TradingView flow targets equities (RTH).
    session_type = "rth"

    con = schema.connect(db_path)
    try:
        try:
            rows = con.execute(_QUERY, [symbol, session_type, n]).fetchall()
        except Exception:
            # Table likely does not exist yet (DB not initialized/populated).
            print(_table_missing_message(db_path, symbol))
            return
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not rows:
        print(_table_missing_message(db_path, symbol))
        return

    # Query is DESC (newest first); reverse to display oldest -> newest.
    rows = list(reversed(rows))

    headers = ["session_date", "PDH", "PDL", "PMH", "PML", "NTZ", "ATR14"]

    display_rows = []
    for r in rows:
        session_date, pdh, pdl, pmh, pml, ntz_low, ntz_high, ntz_valid, atr14 = r
        display_rows.append(
            [
                str(session_date),
                _fmt_price(pdh),
                _fmt_price(pdl),
                _fmt_price(pmh),
                _fmt_price(pml),
                _fmt_ntz(ntz_low, ntz_high, ntz_valid),
                _fmt_price(atr14),
            ]
        )

    if _HAVE_RICH:
        console = Console()
        table = Table(
            title="{sym} levels — last {n} RTH sessions (oldest → newest)".format(
                sym=symbol, n=len(display_rows)
            ),
            header_style="bold",
        )
        table.add_column(headers[0], no_wrap=True)
        for h in headers[1:]:
            table.add_column(h, justify="right")
        for dr in display_rows:
            table.add_row(*dr)
        console.print(table)
    else:
        # Plain-text fallback: fixed-width columns.
        widths = [len(h) for h in headers]
        for dr in display_rows:
            for i, cell in enumerate(dr):
                widths[i] = max(widths[i], len(cell))

        def _fmt_line(cells):
            parts = []
            for i, cell in enumerate(cells):
                if i == 0:
                    parts.append(cell.ljust(widths[i]))
                else:
                    parts.append(cell.rjust(widths[i]))
            return "  ".join(parts)

        title = "{sym} levels — last {n} RTH sessions (oldest -> newest)".format(
            sym=symbol, n=len(display_rows)
        )
        print(title)
        print(_fmt_line(headers))
        print("  ".join("-" * w for w in widths))
        for dr in display_rows:
            print(_fmt_line(dr))


def eyeball_check_explainer() -> str:
    """Return a multi-line explainer for validating levels against TradingView."""
    return (
        "\n"
        "How to eyeball-validate these levels against TradingView\n"
        "========================================================\n"
        "Open the symbol on TradingView with **Extended Hours ON** (the clock\n"
        "icon / chart settings → Session → Extended hours). Then check each\n"
        "row:\n"
        "\n"
        "  * PDH / PDL = the PRIOR regular-session (09:30-16:00 ET) HIGH / LOW.\n"
        "    For a row dated D, these come from the previous trading day's RTH\n"
        "    range. Drop a horizontal ray on that day's RTH high and low and\n"
        "    compare to our pdh / pdl.\n"
        "\n"
        "  * PMH / PML = the HIGH / LOW of the **premarket window 04:00-09:30\n"
        "    ET** of session D itself. Zoom to that window on a 2m or 5m chart\n"
        "    and read the extreme of the premarket bars; it should match our\n"
        "    pmh / pml within ~1 tick.\n"
        "\n"
        "  * NTZ = the visual OVERLAP band between [PDL, PDH] and [PML, PMH].\n"
        "    It is the intersection of the prior-day range and the premarket\n"
        "    range. If those two bands do NOT overlap, there is no no-trade\n"
        "    zone and NTZ shows '—'.\n"
        "\n"
        "  * Adjustment caveat: TradingView's 'adjust for dividends' (and\n"
        "    split adjustment) setting MUST match our data. We store split- and\n"
        "    dividend-ADJUSTED bars, so set TradingView to adjusted too —\n"
        "    otherwise older levels will look offset by exactly the dividend /\n"
        "    split amount and you'll chase a phantom discrepancy.\n"
        "\n"
        "Spot-check 2-3 of the 10 sessions rather than all of them — and make\n"
        "sure your sample includes one session with a GAP (premarket trades\n"
        "away from the prior close) and one QUIET day (tight, low-range\n"
        "session). Those two cases exercise the NTZ overlap logic and the ATR\n"
        "the hardest.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m data.validate",
        description=(
            "Print recent computed levels for a symbol and explain how to "
            "eyeball-validate them against TradingView (read-only)."
        ),
    )
    parser.add_argument(
        "--symbol",
        default="QQQ",
        help="Ticker to validate (default: QQQ).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of most-recent sessions to show (default: 10).",
    )
    parser.add_argument(
        "--db",
        default=schema.DEFAULT_DB_PATH,
        help="Path to the DuckDB market database (default: %(default)s).",
    )
    args = parser.parse_args()

    print_levels(symbol=args.symbol, n=args.n, db_path=args.db)
    print(eyeball_check_explainer())


if __name__ == "__main__":
    main()
