"""portfolio/model.py — the shared data model for the cross-strategy book.

The whole portfolio engine speaks in these dataclasses. The keystone idea is the
**normalization key** on :class:`Candidate`: a candidate carries an OPTIONAL
``target_weight`` (set by *weight* sleeves like rotation / swing_meanrev) AND an
OPTIONAL ``score`` (set by *score* sleeves like swing_breakout). RANK uses
``score`` (a weight sleeve's ``target_weight`` doubles as its score); SIZE uses
``target_weight * equity`` when present, else the risk formula
``shares = dollar_risk / (entry - stop)``. That single dual-field shape lets a
breakout momentum number and an allocation weight flow through ONE ranking +
budgeting pipeline.

Everything here is a plain, deterministic value object — no behavior beyond a few
derived helpers, no I/O. The engine builds :class:`Candidate` / :class:`Allocation`,
the backtester (or the live seam) consumes :class:`PlannedOpen` /
:class:`PlannedClose` / :class:`PlannedResize`, and :class:`BookState` /
:class:`OpenLot` carry the surviving positions between days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# --------------------------------------------------------------------------- #
# Per-day market view (the engine's ``ohlc_today`` input)
# --------------------------------------------------------------------------- #
@dataclass
class DayBars:
    """Today's OHLC rows + an ATR lookup — the engine's per-day market snapshot.

    Built by the backtester (or the live premarket loop) for each decision day and
    handed to ``PortfolioEngine.step`` as ``ohlc_today``. The rows are pandas
    Series indexed by symbol; ``atr`` maps an ATR window → the Series of ATR values
    as of today (so both the bracket window and the synthetic-stop window can be
    looked up). Keeping this a plain value object keeps the engine PURE — tests
    inject hand-built rows, no DB required.
    """

    asof: date
    open: object        # pd.Series indexed by symbol
    high: object
    low: object
    close: object
    atr: dict = field(default_factory=dict)   # window:int -> pd.Series(by symbol)

    def _val(self, row, symbol):
        try:
            v = row.get(symbol)
        except AttributeError:
            v = row[symbol] if symbol in row else None
        return v

    def close_of(self, symbol) -> Optional[float]:
        """Today's close for ``symbol`` (``None`` if missing / non-finite)."""
        return _finite(self._val(self.close, symbol))

    def open_of(self, symbol) -> Optional[float]:
        return _finite(self._val(self.open, symbol))

    def high_of(self, symbol) -> Optional[float]:
        return _finite(self._val(self.high, symbol))

    def low_of(self, symbol) -> Optional[float]:
        return _finite(self._val(self.low, symbol))

    def atr_of(self, symbol, window: int) -> Optional[float]:
        """ATR(``window``) for ``symbol`` as of today (``None`` if unavailable)."""
        series = self.atr.get(int(window))
        if series is None:
            return None
        return _finite(self._val(series, symbol))


def _finite(v) -> Optional[float]:
    """Coerce to a positive-or-any finite float, else ``None``."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


# --------------------------------------------------------------------------- #
# Sleeve specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SleeveSpec:
    """One sleeve admitted to the book: a named strategy + how to read it.

    Parameters
    ----------
    name
        Short label used for attribution / dedup / the correlation matrix.
    strategy
        The instantiated strategy object. A WEIGHT sleeve implements
        ``target_weights(asof, history) -> {symbol: fraction}``; a SCORE sleeve
        implements ``entry_score(symbol, asof, history) -> float | None`` (+ an
        optional ``exit_signal``).
    kind
        ``"weight"`` or ``"score"`` — selects the adapter and the sizing path.
    grade
        Conviction grade (``"B"`` / ``"A"`` / ``"A+"``) → base RI via
        ``risk.sizing.resolve_ri``. Flexes the per-candidate dollar-risk for SCORE
        sleeves; WEIGHT sleeves size off ``target_weight`` so the grade only tags
        attribution / same-symbol conflict resolution there.
    family
        Risk-family label for the per-family heat cap (e.g. ``"equity"`` /
        ``"bond"`` / ``"trend"``). Defaults to the sleeve ``name``.
    benchmark
        The symbol whose absolute momentum gates this sleeve's arm decision
        (``regime_reader.arm_signals``). ``None`` → the market proxy.
    """

    name: str
    strategy: object
    kind: str = "weight"
    grade: str = "B"
    family: str = ""
    benchmark: Optional[str] = None
    bracket: object = None             # BracketConfig for SCORE sleeves (None for weight)
    allocation: float = 1.0            # sleeve's book share (registry min-variance weight)

    def resolved_family(self) -> str:
        """The family label, defaulting to the sleeve name when unset."""
        return self.family or self.name


# --------------------------------------------------------------------------- #
# Candidate — the normalized cross-sleeve signal
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """A single normalized opportunity from one sleeve, AS OF a decision day.

    The dual ``target_weight`` / ``score`` fields are the normalization key (see
    the module docstring). ``norm_score`` is filled by :mod:`portfolio.rank` (the
    within-sleeve-normalized cross-sleeve ranking key). The bracket basis
    (``entry_price`` / ``stop`` / ``atr``) is set for SCORE candidates at collect
    time and synthesized for WEIGHT candidates (the synthetic-stop band) so every
    admitted position contributes a comparable dollar-risk to the book heat.
    """

    sleeve: str
    symbol: str
    kind: str                          # "weight" | "score"
    side: str = "long"                 # long-only book; field kept for the seam
    grade: str = "B"
    family: str = ""

    # --- the normalization key ---
    score: Optional[float] = None          # set by score sleeves (raw signal)
    target_weight: Optional[float] = None  # set by weight sleeves (fraction)

    # --- bracket / sizing basis ---
    entry_price: Optional[float] = None    # today's close (the fill)
    stop: Optional[float] = None           # score: ATR stop; weight: synthetic band
    atr: Optional[float] = None            # ATR as of asof (trail / synthetic band)
    tp1_price: Optional[float] = None       # score: +tp1_R*R partial target
    hard_target: Optional[float] = None     # score: +hard_target_R*R full cap

    # --- ranking + forecast hooks ---
    norm_score: Optional[float] = None     # within-sleeve-normalized rank key
    forecast: Optional[dict] = None        # Phase-3 Kronos overlay (exp_return, ...)

    def rank_value(self) -> float:
        """The value RANK orders on: ``norm_score`` once set, else the raw key.

        Before :mod:`portfolio.rank` runs, fall back to ``score`` (or
        ``target_weight`` for a weight sleeve) so the candidate is still orderable.
        ``-inf`` when nothing is set (sorts last).
        """
        if self.norm_score is not None:
            return float(self.norm_score)
        if self.score is not None:
            return float(self.score)
        if self.target_weight is not None:
            return float(self.target_weight)
        return float("-inf")

    def risk_per_share(self) -> Optional[float]:
        """``entry - stop`` (the per-share dollar risk), or ``None`` if undefined."""
        if self.entry_price is None or self.stop is None:
            return None
        r = float(self.entry_price) - float(self.stop)
        return r if r > 0 else None


# --------------------------------------------------------------------------- #
# Open lots — the surviving book carried between days
# --------------------------------------------------------------------------- #
@dataclass
class OpenLot:
    """One live position in the book, keyed by SYMBOL (one position per symbol).

    Carries the owning sleeve + management ``kind`` so MANAGE knows whether to run
    the ATR bracket (score lots) or reconcile to a target weight (weight lots).
    The bracket-state fields mirror
    :class:`backtest.daily.bracket_engine.OpenPosition`; the realized-so-far
    accumulators let a partial scale-out fold into the final closed-trade record
    with a size-weighted R.
    """

    sleeve: str
    symbol: str
    kind: str                          # "weight" | "score"
    entry_date: date
    entry_price: float
    shares: float
    grade: str = "B"
    family: str = ""

    # --- score-lot bracket state (None for weight lots) ---
    initial_stop: Optional[float] = None
    initial_shares: Optional[float] = None
    atr_entry: Optional[float] = None
    stop: Optional[float] = None
    tp1_price: Optional[float] = None
    hard_target: Optional[float] = None
    tp1_done: bool = False
    highest_high: float = 0.0
    bars_held: int = 0

    # --- weight-lot standing target (last reconciled fraction) ---
    target_weight: Optional[float] = None

    # --- realized-so-far accumulators (partials fold into the final trade) ---
    realized_pnl: float = 0.0
    realized_gross: float = 0.0
    realized_costs: float = 0.0
    realized_r: float = 0.0

    @property
    def risk_per_share(self) -> float:
        """INITIAL per-share risk ``entry - initial_stop`` for a score lot (R basis).

        ``0.0`` for a weight lot (no fixed initial stop) — its heat contribution is
        computed from the *synthetic* band each day, not this field.
        """
        if self.initial_stop is None:
            return 0.0
        return self.entry_price - self.initial_stop


# --------------------------------------------------------------------------- #
# Planned actions — the engine's output the backtester / live seam executes
# --------------------------------------------------------------------------- #
@dataclass
class PlannedOpen:
    """Open a new position: the engine has decided size, fill, and bracket."""

    sleeve: str
    symbol: str
    kind: str
    side: str
    shares: float
    entry_price: float
    grade: str
    family: str
    dollar_risk: float                 # heat charged to the book by this open
    stop: Optional[float] = None       # score: ATR stop; weight: synthetic band
    atr: Optional[float] = None
    tp1_price: Optional[float] = None
    hard_target: Optional[float] = None
    target_weight: Optional[float] = None


@dataclass
class PlannedClose:
    """Close (all of) a position at ``fill_price`` for ``reason``."""

    sleeve: str
    symbol: str
    shares: float
    fill_price: float
    reason: str


@dataclass
class PlannedResize:
    """Adjust a standing weight lot by ``delta_shares`` (signed) at ``fill_price``.

    ``delta_shares > 0`` buys (increase toward a higher target), ``< 0`` sells
    (a risk-reducing trim). The whole-lot exit is a :class:`PlannedClose`, not a
    full-negative resize, so the closed-trade ledger stays clean.
    """

    sleeve: str
    symbol: str
    delta_shares: float
    fill_price: float
    reason: str


# --------------------------------------------------------------------------- #
# Allocation — the per-day decision bundle
# --------------------------------------------------------------------------- #
@dataclass
class Allocation:
    """Everything the engine decided for one day: opens / closes / resizes / rejects.

    ``rejected`` is a list of ``(Candidate, reason)`` for transparency (heat full,
    cluster dedup, same-symbol lost the conflict, halted, ...). ``halted`` /
    ``halt_reason`` flag a book-level halt day (opens suppressed; closes / trims
    still execute — the F2 risk-reducer bypass).
    """

    opens: list = field(default_factory=list)        # list[PlannedOpen]
    closes: list = field(default_factory=list)        # list[PlannedClose]
    resizes: list = field(default_factory=list)       # list[PlannedResize]
    rejected: list = field(default_factory=list)      # list[(Candidate, str)]
    halted: bool = False
    halt_reason: str = ""


# --------------------------------------------------------------------------- #
# Book state — the mutable account snapshot carried between days
# --------------------------------------------------------------------------- #
@dataclass
class BookState:
    """The account's live state: cash, open lots, and the halt reference levels.

    ``lots`` is keyed by SYMBOL (the book holds at most one position per symbol;
    cross-sleeve duplicates are dedup'd / resolved before they ever open). The
    reference levels (peak / month-start / week-start / prev close NAV) are
    maintained by the backtester (or the live equity source) and READ by the
    engine to evaluate the drawdown / daily / weekly halts — the engine never
    fabricates them.
    """

    cash: float
    lots: dict = field(default_factory=dict)          # symbol -> OpenLot
    peak_equity: float = 0.0
    month_start_equity: float = 0.0
    month_key: Optional[tuple] = None
    week_start_equity: float = 0.0
    week_key: Optional[tuple] = None
    prev_nav: float = 0.0

    # --- accounting accumulators (the engine maintains these as the book evolves) ---
    total_costs: float = 0.0
    tax_reserve: float = 0.0
    realized_gains: float = 0.0
    closed_trades: list = field(default_factory=list)         # list[PortfolioTrade]
    sleeve_realized: dict = field(default_factory=dict)       # sleeve -> net realized PnL
    turnover_today: float = 0.0                               # day's traded notional

    def held_symbols(self) -> set:
        """Every symbol currently held by the book (across all sleeves)."""
        return set(self.lots.keys())

    def held_by_sleeve(self, sleeve: str) -> set:
        """Symbols currently owned by ``sleeve``."""
        return {s for s, lot in self.lots.items() if lot.sleeve == sleeve}
