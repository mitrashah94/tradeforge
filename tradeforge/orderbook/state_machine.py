"""orderbook/state_machine.py — the deterministic order FSM + order ledger.

The single source of truth for order lifecycle (MASTER_PLAN.md §4). Pure,
deterministic Python — NO LLM, NO MCP. Persists to ``orderbook/orderbook.duckdb``.

FSM (illegal transitions raise :class:`IllegalTransition`):

    STAGED -> APPROVED -> SUBMITTED -> WORKING -> { FILLED | PARTIAL
                                                    | CANCELLED | REJECTED
                                                    | EXPIRED }
    PARTIAL -> { FILLED | CANCELLED | EXPIRED }
    STAGED  -> VETOED                       (risk/halt rejection, terminal)
    (APPROVED|SUBMITTED|WORKING) -> CANCELLED / REJECTED / EXPIRED

OCO brackets (HARDENED LOCAL — see warning below):
    An entry order plus a stop leg and a target leg form an OCO group. When one
    protective leg FILLS, the sibling leg is CANCELLED automatically. This is a
    *locally simulated* bracket because Robinhood exposes no native server-side
    OCO (CLAUDE.md P0 decision #3).

    # REQUIRES dead-man's switch (watchdog.py, P6)
    Local brackets only protect while THIS process is alive and connected. If
    the process dies or loses the broker/data feed with an open position, the
    stop will not be enforced. The dead-man's switch (watchdog.py, built in P6)
    is therefore MANDATORY in production: on lost connection it must flatten or
    alert-and-halt within a timeout. ``OrderBook`` exposes
    :attr:`requires_dead_mans_switch` (True while any local bracket is live) and
    :meth:`assert_dead_mans_switch_armed` so the boot/run path can refuse to go
    live with local brackets unless the switch is armed.

Tables (orderbook.duckdb):
    orders(order_id PK, parent_id, bracket_id, leg, symbol, side, qty,
           order_type, intended_price, limit_price, stop_price, state,
           strategy, route, ts_created, ts_updated)
    brackets(bracket_id PK, symbol, entry_id, stop_id, target_id, state,
             ts_created, ts_updated)
    positions(position_id PK, symbol, side, qty, avg_price, realized_pnl,
              mfe, mae, state, opened_ts, closed_ts)
    fills(fill_id PK, order_id, symbol, side, qty, price, intended_price,
          slippage, ts)            -- slippage = price - intended_price
    recon(recon_id PK, ts, kind, detail, resolution)
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

DEFAULT_DB_PATH = "orderbook/orderbook.duckdb"


# --------------------------------------------------------------------------- #
# States & transition table                                                   #
# --------------------------------------------------------------------------- #
class OrderState:
    """Canonical order states (string constants; persisted verbatim)."""

    STAGED = "STAGED"
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    WORKING = "WORKING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    VETOED = "VETOED"


# Terminal states have no outbound transitions.
TERMINAL_STATES = frozenset(
    {
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.VETOED,
    }
)

# Legal transitions. Anything not listed raises IllegalTransition.
_LEGAL: dict[str, frozenset] = {
    OrderState.STAGED: frozenset(
        {OrderState.APPROVED, OrderState.VETOED, OrderState.REJECTED, OrderState.CANCELLED}
    ),
    OrderState.APPROVED: frozenset(
        {OrderState.SUBMITTED, OrderState.REJECTED, OrderState.CANCELLED}
    ),
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.WORKING,
            OrderState.FILLED,
            OrderState.PARTIAL,
            OrderState.REJECTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.WORKING: frozenset(
        {
            OrderState.FILLED,
            OrderState.PARTIAL,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.PARTIAL: frozenset(
        {OrderState.FILLED, OrderState.PARTIAL, OrderState.CANCELLED, OrderState.EXPIRED}
    ),
}


class IllegalTransition(Exception):
    """Raised when an order is moved between states the FSM forbids."""


def can_transition(src: str, dst: str) -> bool:
    """True if ``src -> dst`` is a legal FSM edge."""
    return dst in _LEGAL.get(src, frozenset())


def validate_transition(src: str, dst: str) -> None:
    """Raise :class:`IllegalTransition` unless ``src -> dst`` is legal."""
    if not can_transition(src, dst):
        raise IllegalTransition(f"illegal order transition {src} -> {dst}")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Lightweight record dataclasses (read views)                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Order:
    order_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    state: str
    intended_price: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    bracket_id: str | None = None
    leg: str | None = None  # "entry" | "stop" | "target" | None
    parent_id: str | None = None
    strategy: str = ""
    route: str = "paper"


@dataclass
class Position:
    position_id: str
    symbol: str
    side: str
    qty: float
    avg_price: float
    realized_pnl: float
    mfe: float
    mae: float
    state: str


# --------------------------------------------------------------------------- #
# OrderBook                                                                    #
# --------------------------------------------------------------------------- #
class OrderBook:
    """Persistent order/bracket/position/fill ledger + the order FSM.

    Construct with an optional ``bus`` (any object exposing ``publish(Event)``);
    when provided, lifecycle changes can be mirrored onto the bus by the caller
    (the gateway does this). The OrderBook itself stays deterministic and does
    not require the bus to function — it is injectable for tests.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, bus=None):
        import duckdb  # lazy

        self.db_path = db_path
        self.bus = bus
        if db_path != ":memory:":
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._con = duckdb.connect(db_path)
        self._init_schema()

    # ------------------------------------------------------------------ #
    # schema                                                             #
    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id       VARCHAR PRIMARY KEY,
                parent_id      VARCHAR,
                bracket_id     VARCHAR,
                leg            VARCHAR,
                symbol         VARCHAR,
                side           VARCHAR,
                qty            DOUBLE,
                order_type     VARCHAR,
                intended_price DOUBLE,
                limit_price    DOUBLE,
                stop_price     DOUBLE,
                state          VARCHAR,
                strategy       VARCHAR,
                route          VARCHAR,
                ts_created     TIMESTAMP,
                ts_updated     TIMESTAMP
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS brackets (
                bracket_id VARCHAR PRIMARY KEY,
                symbol     VARCHAR,
                entry_id   VARCHAR,
                stop_id    VARCHAR,
                target_id  VARCHAR,
                state      VARCHAR,
                ts_created TIMESTAMP,
                ts_updated TIMESTAMP
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                position_id  VARCHAR PRIMARY KEY,
                symbol       VARCHAR,
                side         VARCHAR,
                qty          DOUBLE,
                avg_price    DOUBLE,
                realized_pnl DOUBLE,
                mfe          DOUBLE,
                mae          DOUBLE,
                state        VARCHAR,
                opened_ts    TIMESTAMP,
                closed_ts    TIMESTAMP
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                fill_id        VARCHAR PRIMARY KEY,
                order_id       VARCHAR,
                symbol         VARCHAR,
                side           VARCHAR,
                qty            DOUBLE,
                price          DOUBLE,
                intended_price DOUBLE,
                slippage       DOUBLE,
                ts             TIMESTAMP
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS recon (
                recon_id   VARCHAR PRIMARY KEY,
                ts         TIMESTAMP,
                kind       VARCHAR,
                detail     VARCHAR,
                resolution VARCHAR
            )
            """
        )

    # ------------------------------------------------------------------ #
    # dead-man's switch flag (see module docstring)                      #
    # ------------------------------------------------------------------ #
    @property
    def requires_dead_mans_switch(self) -> bool:
        """True while any OCO bracket is live (local brackets need the switch).

        # REQUIRES dead-man's switch (watchdog.py, P6)
        """
        row = self._con.execute(
            "SELECT COUNT(*) FROM brackets WHERE state IN ('PENDING','ACTIVE')"
        ).fetchone()
        return int(row[0]) > 0

    def assert_dead_mans_switch_armed(self, armed: bool) -> None:
        """Refuse to proceed with live local brackets unless the switch is armed.

        Production boot/run code calls this before going live: if local brackets
        are in force and ``armed`` is False, raise. Paper paths may pass
        armed=True (the switch is a live-trading protection).
        """
        if self.requires_dead_mans_switch and not armed:
            raise RuntimeError(
                "live local OCO brackets require the dead-man's switch armed "
                "(watchdog.py, P6) — refusing to proceed"
            )

    # ------------------------------------------------------------------ #
    # order creation                                                     #
    # ------------------------------------------------------------------ #
    def create_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        intended_price: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        bracket_id: str | None = None,
        leg: str | None = None,
        parent_id: str | None = None,
        strategy: str = "",
        route: str = "paper",
        order_id: str | None = None,
        state: str = OrderState.STAGED,
    ) -> str:
        """Insert a new order in ``state`` (default STAGED). Returns its id."""
        oid = order_id or _new_id("ord")
        now = _utcnow()
        self._con.execute(
            """
            INSERT INTO orders
                (order_id, parent_id, bracket_id, leg, symbol, side, qty,
                 order_type, intended_price, limit_price, stop_price, state,
                 strategy, route, ts_created, ts_updated)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                oid,
                parent_id,
                bracket_id,
                leg,
                symbol,
                side,
                qty,
                order_type,
                intended_price,
                limit_price,
                stop_price,
                state,
                strategy,
                route,
                now,
                now,
            ],
        )
        return oid

    # ------------------------------------------------------------------ #
    # FSM transition                                                     #
    # ------------------------------------------------------------------ #
    def transition(self, order_id: str, dst: str) -> str:
        """Move ``order_id`` to ``dst`` after validating the FSM edge.

        Raises :class:`IllegalTransition` for a forbidden edge and KeyError if
        the order does not exist. Returns the new state.
        """
        src = self.get_state(order_id)
        if src is None:
            raise KeyError(f"unknown order {order_id}")
        validate_transition(src, dst)
        self._con.execute(
            "UPDATE orders SET state = ?, ts_updated = ? WHERE order_id = ?",
            [dst, _utcnow(), order_id],
        )
        return dst

    def get_state(self, order_id: str) -> str | None:
        row = self._con.execute(
            "SELECT state FROM orders WHERE order_id = ?", [order_id]
        ).fetchone()
        return row[0] if row else None

    def get_order(self, order_id: str) -> Order | None:
        row = self._con.execute(
            "SELECT order_id, symbol, side, qty, order_type, state, "
            "intended_price, limit_price, stop_price, bracket_id, leg, "
            "parent_id, strategy, route FROM orders WHERE order_id = ?",
            [order_id],
        ).fetchone()
        if not row:
            return None
        return Order(
            order_id=row[0], symbol=row[1], side=row[2], qty=row[3],
            order_type=row[4], state=row[5], intended_price=row[6],
            limit_price=row[7], stop_price=row[8], bracket_id=row[9],
            leg=row[10], parent_id=row[11], strategy=row[12], route=row[13],
        )

    def open_orders(self) -> list[Order]:
        """All orders not in a terminal state (working/staged/etc.)."""
        terminals = ",".join(f"'{s}'" for s in TERMINAL_STATES)
        rows = self._con.execute(
            f"SELECT order_id FROM orders WHERE state NOT IN ({terminals})"
        ).fetchall()
        return [self.get_order(r[0]) for r in rows]

    # ------------------------------------------------------------------ #
    # OCO brackets                                                       #
    # ------------------------------------------------------------------ #
    def create_bracket(
        self, symbol: str, entry_id: str, stop_id: str, target_id: str
    ) -> str:
        """Register an OCO group (entry + stop + target). Returns bracket id.

        # REQUIRES dead-man's switch (watchdog.py, P6) — local OCO only.
        Tags all three legs with the bracket id and their leg role.
        """
        bid = _new_id("brk")
        now = _utcnow()
        self._con.execute(
            """
            INSERT INTO brackets
                (bracket_id, symbol, entry_id, stop_id, target_id, state,
                 ts_created, ts_updated)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            [bid, symbol, entry_id, stop_id, target_id, "PENDING", now, now],
        )
        for oid, leg in ((entry_id, "entry"), (stop_id, "stop"), (target_id, "target")):
            self._con.execute(
                "UPDATE orders SET bracket_id = ?, leg = ?, ts_updated = ? "
                "WHERE order_id = ?",
                [bid, leg, now, oid],
            )
        return bid

    def activate_bracket(self, bracket_id: str) -> None:
        """Mark a bracket ACTIVE (entry filled, protective legs now working)."""
        self._con.execute(
            "UPDATE brackets SET state = 'ACTIVE', ts_updated = ? WHERE bracket_id = ?",
            [_utcnow(), bracket_id],
        )

    def active_bracket_for_symbol(self, symbol: str) -> dict | None:
        """Return the most-recent live (PENDING/ACTIVE) bracket for ``symbol``.

        Used by the gateway's F2 management path to locate the protective stop
        leg to relocate (breakeven / trail) for the symbol's open position.
        """
        row = self._con.execute(
            "SELECT bracket_id FROM brackets WHERE symbol = ? "
            "AND state IN ('PENDING','ACTIVE') ORDER BY ts_created DESC LIMIT 1",
            [symbol],
        ).fetchone()
        return self.get_bracket(row[0]) if row else None

    def update_order_stop(self, order_id: str, stop_price: float) -> None:
        """Relocate a (protective) order's stop price in place (BE / trail move).

        A local-OCO stop-move is a price relocation, not a new order — the leg
        stays WORKING. Keeps the ledger's stop leg current for reconciliation.
        """
        self._con.execute(
            "UPDATE orders SET stop_price = ?, intended_price = ?, ts_updated = ? "
            "WHERE order_id = ?",
            [stop_price, stop_price, _utcnow(), order_id],
        )

    def get_bracket(self, bracket_id: str) -> dict | None:
        row = self._con.execute(
            "SELECT bracket_id, symbol, entry_id, stop_id, target_id, state "
            "FROM brackets WHERE bracket_id = ?",
            [bracket_id],
        ).fetchone()
        if not row:
            return None
        return {
            "bracket_id": row[0], "symbol": row[1], "entry_id": row[2],
            "stop_id": row[3], "target_id": row[4], "state": row[5],
        }

    def on_leg_fill(self, filled_order_id: str) -> list[str]:
        """OCO trigger: when one protective leg fills, CANCEL the sibling leg.

        Called after a stop or target leg reaches FILLED. Returns the list of
        sibling order ids that were cancelled (for event emission by the caller).
        A no-op (returns []) for orders not in a bracket or for the entry leg.
        """
        order = self.get_order(filled_order_id)
        if order is None or order.bracket_id is None:
            return []
        bracket = self.get_bracket(order.bracket_id)
        if bracket is None:
            return []

        # Only the protective legs are mutually exclusive. If the entry fills,
        # the bracket activates (handled elsewhere); no sibling cancel here.
        if order.leg not in ("stop", "target"):
            return []

        sibling_id = bracket["target_id"] if order.leg == "stop" else bracket["stop_id"]
        cancelled: list[str] = []
        sib_state = self.get_state(sibling_id)
        if sib_state is not None and sib_state not in TERMINAL_STATES:
            # Drive the sibling to CANCELLED through a legal FSM edge.
            self._force_cancel(sibling_id)
            cancelled.append(sibling_id)

        self._con.execute(
            "UPDATE brackets SET state = 'CLOSED', ts_updated = ? WHERE bracket_id = ?",
            [_utcnow(), order.bracket_id],
        )
        return cancelled

    def _force_cancel(self, order_id: str) -> None:
        """Cancel an order via a legal path to CANCELLED.

        Live protective legs sit in WORKING (directly cancellable); APPROVED and
        STAGED legs are also directly cancellable. This keeps every cancel
        FSM-legal.
        """
        state = self.get_state(order_id)
        if state is None or state in TERMINAL_STATES:
            return
        validate_transition(state, OrderState.CANCELLED)
        self._con.execute(
            "UPDATE orders SET state = ?, ts_updated = ? WHERE order_id = ?",
            [OrderState.CANCELLED, _utcnow(), order_id],
        )

    # ------------------------------------------------------------------ #
    # fills (with slippage capture)                                      #
    # ------------------------------------------------------------------ #
    def record_fill(
        self,
        order_id: str,
        price: float,
        qty: float | None = None,
        intended_price: float | None = None,
        ts: datetime | None = None,
    ) -> dict:
        """Record a fill; slippage = price - intended_price (signed).

        ``intended_price`` defaults to the order's stored intended_price. Returns
        a dict with the fill detail including computed slippage.
        """
        order = self.get_order(order_id)
        if order is None:
            raise KeyError(f"unknown order {order_id}")
        qty = order.qty if qty is None else qty
        intended = intended_price if intended_price is not None else order.intended_price
        slippage = None if intended is None else (price - intended)
        fid = _new_id("fil")
        self._con.execute(
            """
            INSERT INTO fills
                (fill_id, order_id, symbol, side, qty, price, intended_price,
                 slippage, ts)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            [
                fid, order_id, order.symbol, order.side, qty, price,
                intended, slippage, ts or _utcnow(),
            ],
        )
        return {
            "fill_id": fid, "order_id": order_id, "symbol": order.symbol,
            "side": order.side, "qty": qty, "price": price,
            "intended_price": intended, "slippage": slippage,
        }

    def fills_for(self, order_id: str) -> list[dict]:
        rows = self._con.execute(
            "SELECT fill_id, order_id, symbol, side, qty, price, "
            "intended_price, slippage, ts FROM fills WHERE order_id = ? "
            "ORDER BY ts ASC",
            [order_id],
        ).fetchall()
        cols = ["fill_id", "order_id", "symbol", "side", "qty", "price",
                "intended_price", "slippage", "ts"]
        return [dict(zip(cols, r)) for r in rows]

    # ------------------------------------------------------------------ #
    # positions (running MFE / MAE / realized PnL)                       #
    # ------------------------------------------------------------------ #
    def open_position(
        self,
        symbol: str,
        side: str,
        qty: float,
        avg_price: float,
        position_id: str | None = None,
        opened_ts: datetime | None = None,
    ) -> str:
        """Open (or upsert) a position. Returns its id."""
        pid = position_id or _new_id("pos")
        now = opened_ts or _utcnow()
        self._con.execute(
            """
            INSERT OR REPLACE INTO positions
                (position_id, symbol, side, qty, avg_price, realized_pnl,
                 mfe, mae, state, opened_ts, closed_ts)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            [pid, symbol, side, qty, avg_price, 0.0, 0.0, 0.0, "OPEN", now, None],
        )
        return pid

    def update_position_excursion(self, position_id: str, mark_price: float) -> dict:
        """Update running MFE/MAE for an open position given a mark price.

        MFE = max favorable excursion (best unrealized $), MAE = max adverse
        (worst unrealized $, a negative or zero magnitude). Signed by side so
        longs profit on up moves and shorts on down moves.
        """
        pos = self.get_position(position_id)
        if pos is None:
            raise KeyError(f"unknown position {position_id}")
        direction = 1.0 if pos.side.lower() in ("buy", "long") else -1.0
        unrealized = direction * (mark_price - pos.avg_price) * pos.qty
        mfe = max(pos.mfe, unrealized)
        mae = min(pos.mae, unrealized)
        self._con.execute(
            "UPDATE positions SET mfe = ?, mae = ? WHERE position_id = ?",
            [mfe, mae, position_id],
        )
        return {"position_id": position_id, "unrealized": unrealized, "mfe": mfe, "mae": mae}

    def reduce_position(
        self,
        position_id: str,
        exit_price: float,
        qty: float,
        closed_ts: datetime | None = None,
    ) -> dict:
        """Scale OUT ``qty`` of an open position at ``exit_price`` (TP1 partial).

        Books realized PnL on the scaled-out shares and shrinks the position's
        live qty by ``qty`` (clamped at 0). The position stays OPEN (it becomes
        the runner) unless the reduction takes it flat, in which case it is
        CLOSED. Returns the booked partial detail. This is the F2 positive-skew
        TP1 leg (MASTER_PLAN §1.B): scale a partial, let the runner ride.
        """
        pos = self.get_position(position_id)
        if pos is None:
            raise KeyError(f"unknown position {position_id}")
        direction = 1.0 if pos.side.lower() in ("buy", "long") else -1.0
        scale_qty = min(float(qty), pos.qty)
        partial_realized = direction * (exit_price - pos.avg_price) * scale_qty
        new_qty = pos.qty - scale_qty
        total_realized = (pos.realized_pnl or 0.0) + partial_realized
        if new_qty <= 1e-12:
            self._con.execute(
                "UPDATE positions SET qty = 0, realized_pnl = ?, state = 'CLOSED', "
                "closed_ts = ? WHERE position_id = ?",
                [total_realized, closed_ts or _utcnow(), position_id],
            )
            state = "CLOSED"
        else:
            self._con.execute(
                "UPDATE positions SET qty = ?, realized_pnl = ? WHERE position_id = ?",
                [new_qty, total_realized, position_id],
            )
            state = "OPEN"
        return {
            "position_id": position_id, "scaled_qty": scale_qty,
            "partial_realized_pnl": partial_realized, "realized_pnl": total_realized,
            "remaining_qty": new_qty, "state": state,
        }

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        closed_ts: datetime | None = None,
    ) -> dict:
        """Close a position, booking realized PnL on the REMAINING qty.

        Accumulates onto any PnL already booked by a prior :meth:`reduce_position`
        (the TP1 partial), so a runner that scaled out then closes reports the
        TOTAL realized = partial + runner. When no partial was taken,
        ``realized_pnl`` is 0 and this reduces to the simple full-close PnL
        (preserving the original behaviour for single-leg exits / bracket fills).
        """
        pos = self.get_position(position_id)
        if pos is None:
            raise KeyError(f"unknown position {position_id}")
        direction = 1.0 if pos.side.lower() in ("buy", "long") else -1.0
        runner_realized = direction * (exit_price - pos.avg_price) * pos.qty
        total_realized = (pos.realized_pnl or 0.0) + runner_realized
        self._con.execute(
            "UPDATE positions SET qty = 0, realized_pnl = ?, state = 'CLOSED', "
            "closed_ts = ? WHERE position_id = ?",
            [total_realized, closed_ts or _utcnow(), position_id],
        )
        return {"position_id": position_id, "realized_pnl": total_realized,
                "runner_realized_pnl": runner_realized}

    def get_position(self, position_id: str) -> Position | None:
        row = self._con.execute(
            "SELECT position_id, symbol, side, qty, avg_price, realized_pnl, "
            "mfe, mae, state FROM positions WHERE position_id = ?",
            [position_id],
        ).fetchone()
        if not row:
            return None
        return Position(
            position_id=row[0], symbol=row[1], side=row[2], qty=row[3],
            avg_price=row[4], realized_pnl=row[5], mfe=row[6], mae=row[7],
            state=row[8],
        )

    def open_positions(self) -> list[Position]:
        rows = self._con.execute(
            "SELECT position_id FROM positions WHERE state = 'OPEN'"
        ).fetchall()
        return [self.get_position(r[0]) for r in rows]

    def position_for_symbol(self, symbol: str) -> Position | None:
        row = self._con.execute(
            "SELECT position_id FROM positions WHERE symbol = ? AND state = 'OPEN' "
            "ORDER BY opened_ts DESC LIMIT 1",
            [symbol],
        ).fetchone()
        return self.get_position(row[0]) if row else None

    # ------------------------------------------------------------------ #
    # reconciliation snapshots                                           #
    # ------------------------------------------------------------------ #
    def record_recon(self, kind: str, detail: str, resolution: str) -> str:
        """Persist a reconciliation snapshot row. Returns its id."""
        rid = _new_id("rec")
        self._con.execute(
            "INSERT INTO recon (recon_id, ts, kind, detail, resolution) "
            "VALUES (?,?,?,?,?)",
            [rid, _utcnow(), kind, detail, resolution],
        )
        return rid

    def recon_snapshots(self) -> list[dict]:
        rows = self._con.execute(
            "SELECT recon_id, ts, kind, detail, resolution FROM recon ORDER BY ts ASC"
        ).fetchall()
        cols = ["recon_id", "ts", "kind", "detail", "resolution"]
        return [dict(zip(cols, r)) for r in rows]

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:  # noqa: BLE001
            pass
