"""orchestrator/tools/brokers.py — venue adapters behind the ONE event path.

Paper and live both flow through the identical ORDER_INTENT -> ... -> ORDER_FILLED
sequence in :mod:`orchestrator.tools.order_gateway`; only the venue adapter
differs (MASTER_PLAN.md §4, CLAUDE.md "one event path").

Adapter protocol (duck-typed; the gateway depends only on these methods):

    submit(order) -> dict
        Submit a single-leg order. ``order`` is an :class:`orderbook.state_machine.Order`.
        Returns a venue-result dict:
          {"status": "filled"|"working"|"rejected"|"partial",
           "fill_price": float|None, "filled_qty": float|None,
           "venue_order_id": str, "reason": str}
    cancel(venue_order_id) -> dict
    open_orders() -> list[dict]    # venue's view of working orders
    positions() -> list[dict]      # venue's view of open positions

PaperBroker simulates fills against fed prices and writes to the paper ledger
(``paper/ledger.duckdb``) — the same event path, just a deterministic fill model.

LiveBroker is a DOCUMENTED STUB. The Robinhood *MCP* is a Claude tool, NOT
callable from running Python; a later prompt wires this to Robinhood's REST API
(e.g. robin_stocks). It raises NotImplementedError unless explicitly enabled,
and can only ever be reached behind the P0 hook + a LIVE flag (enforced by the
gateway). NO MCP / NO LLM is ever called here.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

PAPER_LEDGER_PATH = "paper/ledger.duckdb"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _vid() -> str:
    return f"venue_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# PaperBroker                                                                  #
# --------------------------------------------------------------------------- #
class PaperBroker:
    """Deterministic paper venue. Fills against the price fed to it.

    The gateway feeds a reference price per intent (via :meth:`set_price` or the
    ``ref_price`` kwarg on :meth:`submit`). Fill model:
      - market order  -> fills immediately at ref_price (+ optional slippage_bps)
      - limit/stop     -> fills immediately if ref_price satisfies the trigger,
                          else returns "working" (the fast loop drives the rest)

    Writes every fill + the venue's order/position snapshot to
    ``paper/ledger.duckdb`` so the reconciler can query the venue independently.
    """

    route = "paper"

    def __init__(self, db_path: str = PAPER_LEDGER_PATH, slippage_bps: float = 0.0):
        import duckdb  # lazy

        self.db_path = db_path
        self.slippage_bps = slippage_bps
        if db_path != ":memory:":
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._con = duckdb.connect(db_path)
        self._init_schema()
        self._prices: dict[str, float] = {}

    def _init_schema(self) -> None:
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_orders (
                venue_order_id VARCHAR PRIMARY KEY,
                client_order_id VARCHAR,
                symbol VARCHAR,
                side VARCHAR,
                qty DOUBLE,
                order_type VARCHAR,
                limit_price DOUBLE,
                stop_price DOUBLE,
                status VARCHAR,
                ts TIMESTAMP
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_fills (
                fill_id VARCHAR PRIMARY KEY,
                venue_order_id VARCHAR,
                symbol VARCHAR,
                side VARCHAR,
                qty DOUBLE,
                price DOUBLE,
                ts TIMESTAMP
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_positions (
                symbol VARCHAR PRIMARY KEY,
                side VARCHAR,
                qty DOUBLE,
                avg_price DOUBLE,
                ts TIMESTAMP
            )
            """
        )

    # ---- price feed ----
    def set_price(self, symbol: str, price: float) -> None:
        """Feed the current reference price for ``symbol`` (used to fill)."""
        self._prices[symbol] = price

    def _ref(self, symbol: str, override: float | None) -> float | None:
        return override if override is not None else self._prices.get(symbol)

    def _apply_slippage(self, side: str, price: float) -> float:
        if not self.slippage_bps:
            return price
        adj = price * (self.slippage_bps / 10000.0)
        return price + adj if side.lower() in ("buy", "long") else price - adj

    # ---- protocol ----
    def submit(self, order, ref_price: float | None = None) -> dict:
        """Submit a single-leg order; simulate a fill against the ref price."""
        ref = self._ref(order.symbol, ref_price if ref_price is not None else order.intended_price)
        vid = _vid()
        self._con.execute(
            """
            INSERT OR REPLACE INTO ledger_orders
                (venue_order_id, client_order_id, symbol, side, qty, order_type,
                 limit_price, stop_price, status, ts)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [
                vid, order.order_id, order.symbol, order.side, order.qty,
                order.order_type, order.limit_price, order.stop_price,
                "working", _utcnow(),
            ],
        )

        if ref is None:
            # No price to fill against -> leave working for the fast loop.
            return {"status": "working", "fill_price": None, "filled_qty": None,
                    "venue_order_id": vid, "reason": "no reference price"}

        otype = (order.order_type or "market").lower()
        triggered = otype == "market"
        if otype in ("limit",) and order.limit_price is not None:
            triggered = (
                ref <= order.limit_price if order.side.lower() in ("buy", "long")
                else ref >= order.limit_price
            )
        elif otype in ("stop", "stop_market", "stop_limit") and order.stop_price is not None:
            triggered = (
                ref >= order.stop_price if order.side.lower() in ("buy", "long")
                else ref <= order.stop_price
            )

        if not triggered:
            return {"status": "working", "fill_price": None, "filled_qty": None,
                    "venue_order_id": vid, "reason": "trigger not met"}

        fill_price = self._apply_slippage(order.side, ref)
        self._book_fill(vid, order, fill_price)
        return {"status": "filled", "fill_price": fill_price,
                "filled_qty": order.qty, "venue_order_id": vid, "reason": "paper fill"}

    def _book_fill(self, vid: str, order, fill_price: float) -> None:
        fid = f"lfil_{uuid.uuid4().hex[:12]}"
        now = _utcnow()
        self._con.execute(
            "INSERT INTO ledger_fills (fill_id, venue_order_id, symbol, side, "
            "qty, price, ts) VALUES (?,?,?,?,?,?,?)",
            [fid, vid, order.symbol, order.side, order.qty, fill_price, now],
        )
        self._con.execute(
            "UPDATE ledger_orders SET status = 'filled' WHERE venue_order_id = ?",
            [vid],
        )
        # Maintain a simple net position snapshot keyed by symbol.
        existing = self._con.execute(
            "SELECT qty, avg_price, side FROM ledger_positions WHERE symbol = ?",
            [order.symbol],
        ).fetchone()
        signed = order.qty if order.side.lower() in ("buy", "long") else -order.qty
        if existing is None:
            net = signed
            avg = fill_price
        else:
            prev_signed = existing[0] if existing[2].lower() in ("buy", "long") else -existing[0]
            net = prev_signed + signed
            avg = fill_price if prev_signed == 0 else existing[1]
        if abs(net) < 1e-12:
            self._con.execute("DELETE FROM ledger_positions WHERE symbol = ?", [order.symbol])
        else:
            self._con.execute(
                "INSERT OR REPLACE INTO ledger_positions (symbol, side, qty, "
                "avg_price, ts) VALUES (?,?,?,?,?)",
                [order.symbol, "buy" if net > 0 else "sell", abs(net), avg, now],
            )

    def cancel(self, venue_order_id: str) -> dict:
        self._con.execute(
            "UPDATE ledger_orders SET status = 'cancelled' WHERE venue_order_id = ?",
            [venue_order_id],
        )
        return {"status": "cancelled", "venue_order_id": venue_order_id}

    def open_orders(self) -> list[dict]:
        rows = self._con.execute(
            "SELECT venue_order_id, client_order_id, symbol, side, qty, "
            "order_type, status FROM ledger_orders WHERE status = 'working'"
        ).fetchall()
        cols = ["venue_order_id", "client_order_id", "symbol", "side", "qty",
                "order_type", "status"]
        return [dict(zip(cols, r)) for r in rows]

    def positions(self) -> list[dict]:
        rows = self._con.execute(
            "SELECT symbol, side, qty, avg_price FROM ledger_positions"
        ).fetchall()
        cols = ["symbol", "side", "qty", "avg_price"]
        return [dict(zip(cols, r)) for r in rows]

    # Test/recon helper: plant a venue-side order/position directly.
    def _plant_order(self, symbol, side, qty, order_type="market",
                     client_order_id=None, status="working") -> str:
        vid = _vid()
        self._con.execute(
            "INSERT OR REPLACE INTO ledger_orders (venue_order_id, "
            "client_order_id, symbol, side, qty, order_type, limit_price, "
            "stop_price, status, ts) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [vid, client_order_id, symbol, side, qty, order_type, None, None,
             status, _utcnow()],
        )
        return vid

    def _plant_position(self, symbol, side, qty, avg_price) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO ledger_positions (symbol, side, qty, "
            "avg_price, ts) VALUES (?,?,?,?,?)",
            [symbol, side, qty, avg_price, _utcnow()],
        )

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# LiveBroker (documented stub — gated behind hook + LIVE flag)                 #
# --------------------------------------------------------------------------- #
class LiveBroker:
    """Robinhood live venue — DOCUMENTED INTERFACE STUB. NOT YET WIRED.

    REALITY (CLAUDE.md P0 decision #3, MASTER_PLAN §3): the Robinhood *MCP* is a
    Claude tool, NOT callable from running Python. A later prompt will wire this
    adapter to Robinhood's REST API (e.g. ``robin_stocks``): the live order flow
    is review_equity_order -> confirm -> place_equity_order on the agentic_allowed
    account ONLY, single-leg (market/limit/stop_market/stop_limit), equities-only.

    Until then every method raises NotImplementedError unless ``enabled=True`` is
    explicitly passed (so tests can construct it without it ever placing). The
    gateway guarantees this adapter is reached ONLY behind the P0 hook
    (orchestrator.hooks.evaluate) AND a LIVE flag — never by default.

    NO MCP and NO LLM calls live in this class; the hot path stays deterministic.
    """

    route = "live"

    _GUIDANCE = (
        "LiveBroker is a stub. Wire it to Robinhood's REST API (robin_stocks) in "
        "a later prompt: review_equity_order -> confirm -> place_equity_order, "
        "single-leg, equities-only, agentic_allowed account, behind the P0 hook + "
        "LIVE flag. The Robinhood MCP is a Claude tool and cannot be called from "
        "running Python."
    )

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def _guard(self):
        if not self.enabled:
            raise NotImplementedError(self._GUIDANCE)

    def submit(self, order, ref_price: float | None = None) -> dict:
        self._guard()
        raise NotImplementedError(self._GUIDANCE)

    def cancel(self, venue_order_id: str) -> dict:
        self._guard()
        raise NotImplementedError(self._GUIDANCE)

    def open_orders(self) -> list[dict]:
        self._guard()
        raise NotImplementedError(self._GUIDANCE)

    def positions(self) -> list[dict]:
        self._guard()
        raise NotImplementedError(self._GUIDANCE)
