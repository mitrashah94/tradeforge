"""strategies/breakout_retest_options/overlay.py — IV-rank-driven options overlay.

An **execution overlay**, not an independent edge. It takes a *break-and-retest*
signal on the underlying (the same PDH/PDL continuation trigger the
``breakout_retest`` equity strategy fires — direction, spot, protective stop,
fixed-2R target, conviction grade) and decides **how to express that directional
view in listed options** on SPY / QQQ, driven by:

  * **IV rank** — cheap vs rich premium picks the STRUCTURE:
      low IVR  -> buy premium        (long single option, full theta exposure)
      mid IVR  -> debit vertical      (cut cost + net theta vs the single)
      high IVR -> credit spread       (net-SHORT premium, THETA WORKS FOR YOU)
  * **delta** — picks the STRIKES (target directional delta per leg).
  * **theta** — a gate: reject/flag a long-premium ticket whose decay over the
      intended intraday hold eats too much of the move-to-target edge.
  * **the RI options policy** (``risk/limits.yaml`` -> ``level(ri).options``) —
      gates whether options are permitted at all and caps the ticket size.

Everything here is a PURE, deterministic function of its inputs — no clock, no
network, no I/O, no LLM (CLAUDE.md: "No LLM and no MCP calls in the hot path").
The overlay READS the risk config; it never writes it. It emits a decision an
operator (or a paper driver) inspects; it does NOT place orders — options are
not on the sanctioned agentic order path (CLAUDE.md P0 #3), so this module is a
RESEARCH / paper-evaluation tool.

The $1,000 reality (see ``playbook.md``): at the RI-6 floor a per-trade risk
budget is ~1.25% = ~$12.50, while ONE SPY/QQQ contract's first-order loss at the
underlying stop is typically $30-$60. So a disciplined vol-target size rounds to
ZERO contracts. The overlay reports this honestly (``min_viable_equity`` + the
binding constraint) rather than silently rounding up to an over-risked 1-lot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent / "params.yaml"

# Options-policy strings from risk/limits.yaml -> level(ri).options.
_POLICY_NONE = "none"
_POLICY_MINIMAL = "minimal"
_POLICY_DEFINED_RISK_SMALL = "defined_risk_small"
_POLICY_OK = "ok"

# Structures the overlay can emit.
STRUCT_LONG = "long_option"
STRUCT_DEBIT = "debit_vertical"
STRUCT_CREDIT = "credit_spread"

# IV-rank regimes.
IVR_LOW = "low"
IVR_MID = "mid"
IVR_HIGH = "high"


def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> dict:
    """Load the overlay's ``defaults`` block from params.yaml."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return dict(raw.get("defaults", {}))


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UnderlyingSignal:
    """A break-and-retest signal on the underlying (what breakout_retest fires).

    ``side`` is the directional view: ``long`` -> bullish (calls / bull spreads),
    ``short`` -> bearish (puts / bear spreads). ``spot`` is the signal close,
    ``stop`` the role-reversal protective stop, ``target`` the fixed-2R target
    (may be ``None`` for a trailing exit). ``grade`` maps to the risk index.
    """

    symbol: str
    side: str                    # 'long' | 'short'
    spot: float
    stop: float
    target: float | None = None
    grade: str = "B"

    @property
    def stop_distance(self) -> float:
        return abs(self.spot - self.stop)


@dataclass(frozen=True)
class OptionContract:
    """One listed option with the greeks the overlay needs.

    ``delta`` is signed (calls positive, puts negative). ``theta`` is per share,
    per calendar day (negative for long premium). ``bid``/``ask`` are per share;
    a contract controls 100 shares. ``dte`` is days-to-expiration; if omitted it
    is computed from ``expiration`` vs the ``as_of`` date passed to the overlay.
    """

    symbol: str                  # underlying, e.g. 'SPY'
    expiration: str              # 'YYYY-MM-DD'
    strike: float
    right: str                   # 'call' | 'put'
    bid: float
    ask: float
    delta: float                 # signed
    theta: float                 # per share / day, usually < 0
    gamma: float | None = None
    vega: float | None = None
    iv: float | None = None
    open_interest: int | None = None
    volume: int | None = None
    dte: int | None = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_pct(self) -> float:
        """Bid/ask spread as a fraction of mid (liquidity proxy). inf if no mid."""
        m = self.mid
        if m <= 0:
            return float("inf")
        return (self.ask - self.bid) / m


@dataclass(frozen=True)
class Leg:
    """One leg of the chosen structure."""

    action: str                  # 'buy' | 'sell'
    contract: OptionContract
    ratio: int = 1

    @property
    def sign(self) -> int:
        return 1 if self.action == "buy" else -1


@dataclass
class OverlayDecision:
    """The overlay's verdict for one signal.

    ``ok`` is True only when a tradable, correctly-sized ticket was found. When
    False, ``reason`` says why (policy block, no chain, liquidity, sub-one
    contract, ...) and ``diagnostics`` still carries the useful numbers
    (``min_viable_equity``, the binding sizing constraint, greeks).
    """

    ok: bool
    symbol: str
    side: str
    iv_rank: float
    iv_regime: str
    structure: str                       # STRUCT_* or 'none'
    legs: list[Leg] = field(default_factory=list)
    contracts: int = 0
    ri: int = 0
    options_policy: str = ""
    # pricing (per 1 spread/contract, in dollars unless noted)
    net_debit: float = 0.0               # > 0 you pay; used for debit structures
    net_credit: float = 0.0              # > 0 you receive; used for credit spreads
    max_loss_per_contract: float = 0.0
    max_profit_per_contract: float | None = None
    breakeven: float | None = None
    # position greeks (net, at the chosen size)
    net_delta: float = 0.0               # position delta (shares-equivalent)
    net_theta: float = 0.0               # $/day (positive => theta works for you)
    warnings: list[str] = field(default_factory=list)
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def classify_iv_regime(iv_rank: float, iv_low: float, iv_high: float) -> str:
    """Bucket IV rank (0..1) into low / mid / high."""
    if iv_rank < iv_low:
        return IVR_LOW
    if iv_rank >= iv_high:
        return IVR_HIGH
    return IVR_MID


def choose_structure(iv_regime: str, policy: str) -> str | None:
    """Map (IV regime, RI options policy) -> a structure, or None if blocked.

    ``none``  -> options not permitted at this RI (blocked).
    ``minimal`` -> long single options ONLY (no short legs), regardless of IVR.
    ``defined_risk_small`` / ``ok`` -> IV rank decides the structure. Every
      structure the overlay emits is defined-risk (a long option's max loss is
      its debit; both verticals cap loss), so ``defined_risk_small`` permits all
      three — it only caps SIZE (see the sizing caps).
    """
    if policy == _POLICY_NONE:
        return None
    if policy == _POLICY_MINIMAL:
        return STRUCT_LONG
    # defined_risk_small and ok: IV rank drives it.
    if iv_regime == IVR_LOW:
        return STRUCT_LONG
    if iv_regime == IVR_MID:
        return STRUCT_DEBIT
    return STRUCT_CREDIT


def _right_for(side: str) -> str:
    """Bullish view -> calls; bearish view -> puts."""
    return "call" if side == "long" else "put"


def nearest_delta(contracts: list[OptionContract], target_abs: float) -> OptionContract | None:
    """Pick the contract whose |delta| is closest to ``target_abs``."""
    if not contracts:
        return None
    return min(contracts, key=lambda c: abs(abs(c.delta) - target_abs))


def _pick_expiration(
    chain: list[OptionContract], right: str, as_of: date | None,
    prefer_dte: int, min_dte: int, max_dte: int,
) -> tuple[str | None, list[OptionContract]]:
    """Choose the expiration nearest ``prefer_dte`` within [min_dte, max_dte].

    Returns the chosen expiration string and the contracts of ``right`` at it.
    """
    by_exp: dict[str, list[OptionContract]] = {}
    for c in chain:
        if c.right != right:
            continue
        dte = _dte_of(c, as_of)
        if dte is None or dte < min_dte or dte > max_dte:
            continue
        by_exp.setdefault(c.expiration, []).append(c)
    if not by_exp:
        return None, []
    # nearest expiration to prefer_dte (by any of its contracts' dte).
    def exp_dte(exp: str) -> int:
        return _dte_of(by_exp[exp][0], as_of) or 0
    chosen = min(by_exp, key=lambda e: (abs(exp_dte(e) - prefer_dte), exp_dte(e)))
    return chosen, by_exp[chosen]


def _dte_of(c: OptionContract, as_of: date | None) -> int | None:
    if c.dte is not None:
        return c.dte
    if as_of is None:
        return None
    y, m, d = (int(x) for x in c.expiration.split("-"))
    return (date(y, m, d) - as_of).days


def _liquid(c: OptionContract, max_spread_pct: float, min_oi: int) -> bool:
    if c.spread_pct > max_spread_pct:
        return False
    if min_oi > 0 and c.open_interest is not None and c.open_interest < min_oi:
        return False
    return True


# --------------------------------------------------------------------------- #
# Structure builders — each returns (legs, net_debit, net_credit, max_loss,
# max_profit, breakeven) or None if it cannot be built from the chain.
# --------------------------------------------------------------------------- #
def _build_long(cands: list[OptionContract], p: dict, side: str):
    tgt = nearest_delta(cands, p["long_delta_target"])
    if tgt is None:
        return None
    debit = tgt.ask * 100.0                    # pay the ask (marketable)
    be = tgt.strike + tgt.ask if side == "long" else tgt.strike - tgt.ask
    return (
        [Leg("buy", tgt)],
        debit, 0.0,
        debit,                                 # max loss = premium paid
        None,                                  # unbounded (dir.) upside
        be,
    )


def _build_debit_vertical(cands: list[OptionContract], p: dict, side: str):
    long_leg = nearest_delta(cands, p["long_delta_target"])
    short_leg = nearest_delta(cands, p["debit_short_delta_target"])
    if long_leg is None or short_leg is None or long_leg.strike == short_leg.strike:
        return None
    # correct ordering: bull call -> short strike ABOVE long; bear put -> below.
    if side == "long" and not short_leg.strike > long_leg.strike:
        return None
    if side == "short" and not short_leg.strike < long_leg.strike:
        return None
    net_debit = (long_leg.ask - short_leg.bid) * 100.0
    width = abs(long_leg.strike - short_leg.strike) * 100.0
    if net_debit <= 0:
        return None
    max_profit = width - net_debit
    be = (long_leg.strike + net_debit / 100.0) if side == "long" \
        else (long_leg.strike - net_debit / 100.0)
    return (
        [Leg("buy", long_leg), Leg("sell", short_leg)],
        net_debit, 0.0,
        net_debit,                             # max loss = net debit
        max_profit,
        be,
    )


def _build_credit_spread(cands: list[OptionContract], p: dict, side: str):
    # Same directional bias, but SELL premium: long view -> bull PUT credit
    # spread (sell higher put, buy lower put); short view -> bear CALL credit
    # spread (sell lower call, buy higher call).
    right = "put" if side == "long" else "call"
    legs_pool = [c for c in cands if c.right == right]
    short_leg = nearest_delta(legs_pool, p["credit_short_delta_target"])
    long_leg = nearest_delta(legs_pool, p["credit_long_delta_target"])
    if short_leg is None or long_leg is None or short_leg.strike == long_leg.strike:
        return None
    if right == "put" and not long_leg.strike < short_leg.strike:
        return None
    if right == "call" and not long_leg.strike > short_leg.strike:
        return None
    net_credit = (short_leg.bid - long_leg.ask) * 100.0
    width = abs(short_leg.strike - long_leg.strike) * 100.0
    if net_credit <= 0:
        return None
    max_loss = width - net_credit
    be = (short_leg.strike - net_credit / 100.0) if right == "put" \
        else (short_leg.strike + net_credit / 100.0)
    return (
        [Leg("sell", short_leg), Leg("buy", long_leg)],
        0.0, net_credit,
        max_loss,                              # max loss = width - credit
        net_credit,                            # max profit = credit kept
        be,
    )


# --------------------------------------------------------------------------- #
# The overlay
# --------------------------------------------------------------------------- #
class OptionsOverlay:
    """IV-rank-driven options-expression selector for a break-retest signal."""

    def __init__(self, params: dict | None = None):
        self.p = params if params is not None else load_params()

    # ------------------------------------------------------------------ select
    def select(
        self,
        signal: UnderlyingSignal,
        chain: list[OptionContract],
        iv_rank: float,
        equity: float,
        *,
        options_policy: str,
        dollar_risk: float,
        ri: int = 0,
        as_of: date | None = None,
    ) -> OverlayDecision:
        """Choose a structure + strikes + size for ``signal``.

        ``options_policy`` and ``dollar_risk`` come from the resolved risk index
        (``risk.config.Limits`` -> ``level(ri).options`` and
        ``risk.sizing.per_trade_dollar_risk``); pass them in so this stays a pure
        function decoupled from the risk loader. ``iv_rank`` is 0..1.
        """
        p = self.p
        dec = OverlayDecision(
            ok=False, symbol=signal.symbol, side=signal.side, iv_rank=iv_rank,
            iv_regime=classify_iv_regime(iv_rank, p["iv_rank_low"], p["iv_rank_high"]),
            structure="none", ri=ri, options_policy=options_policy,
        )

        # 1) policy gate
        structure = choose_structure(dec.iv_regime, options_policy)
        if structure is None:
            dec.reason = f"options_blocked_by_policy:{options_policy}"
            return dec
        # minimal policy buying rich premium is allowed but costly — flag it.
        if options_policy == _POLICY_MINIMAL and dec.iv_regime == IVR_HIGH:
            dec.warnings.append("minimal_policy_forces_long_premium_at_high_iv")
        dec.structure = structure

        if signal.stop_distance <= 0:
            dec.reason = "non_positive_stop_distance"
            return dec

        # 2) expiration + candidate contracts of the working right
        right = _right_for(signal.side)
        # credit spreads use the OPPOSITE right; pull that side's expiration too.
        work_right = right if structure != STRUCT_CREDIT else ("put" if signal.side == "long" else "call")
        exp, cands = _pick_expiration(
            chain, work_right, as_of,
            int(p["prefer_dte"]), int(p["min_dte"]), int(p["max_dte"]),
        )
        if not cands:
            dec.reason = "no_expiration_in_dte_window"
            return dec
        cands = [c for c in cands if _liquid(c, p["max_spread_pct"], int(p["min_open_interest"]))]
        if not cands:
            dec.reason = "no_liquid_contracts"
            return dec
        dec.diagnostics["expiration"] = exp
        dec.diagnostics["dte"] = _dte_of(cands[0], as_of)

        # 3) build the structure
        builder = {
            STRUCT_LONG: _build_long,
            STRUCT_DEBIT: _build_debit_vertical,
            STRUCT_CREDIT: _build_credit_spread,
        }[structure]
        built = builder(cands, p, signal.side)
        if built is None:
            dec.reason = f"could_not_build:{structure}"
            return dec
        legs, net_debit, net_credit, max_loss_pc, max_profit_pc, be = built
        dec.legs = legs
        dec.net_debit = net_debit
        dec.net_credit = net_credit
        dec.max_loss_per_contract = max_loss_pc
        dec.max_profit_per_contract = max_profit_pc
        dec.breakeven = be

        # per-spread greeks (per share): sum of signed leg greeks
        delta_ps = sum(l.sign * l.ratio * l.contract.delta for l in legs)
        theta_ps = sum(l.sign * l.ratio * l.contract.theta for l in legs)
        dec.diagnostics["delta_per_spread"] = round(delta_ps, 4)
        dec.diagnostics["theta_per_spread_per_day"] = round(theta_ps * 100.0, 2)

        # 4) sizing — the conservative minimum of three caps
        sizing = self._size(
            equity=equity, dollar_risk=dollar_risk, signal=signal,
            delta_ps=delta_ps, max_loss_pc=max_loss_pc,
            outlay_pc=(net_debit if net_debit > 0 else max_loss_pc),
            policy=options_policy,
        )
        dec.diagnostics.update(sizing["diag"])
        contracts = sizing["contracts"]

        if contracts < 1:
            dec.diagnostics["min_viable_equity"] = round(sizing["min_viable_equity"], 2)
            dec.diagnostics["binding_constraint"] = sizing["binding"]
            if p.get("allow_min_ticket", False):
                contracts = 1
                dec.warnings.append(
                    f"over_budget_min_ticket:risk={sizing['risk_pct_at_1']:.2%}"
                    f"_vs_budget={dollar_risk / equity:.2%}"
                )
            else:
                dec.reason = "sub_one_contract_within_risk_budget"
                return dec

        dec.contracts = contracts
        dec.net_delta = round(delta_ps * 100.0 * contracts, 2)
        dec.net_theta = round(theta_ps * 100.0 * contracts, 2)

        # 5) theta gate (only bites long-premium structures; credit is theta+)
        self._theta_gate(dec, signal, theta_ps, delta_ps, contracts)
        if not dec.ok and dec.reason:
            return dec

        dec.ok = True
        return dec

    # ------------------------------------------------------------------ sizing
    def _size(self, *, equity, dollar_risk, signal, delta_ps, max_loss_pc, outlay_pc, policy):
        p = self.p
        stop_dist = signal.stop_distance
        # first-order loss at the underlying stop, capped by the defined max loss
        loss_at_stop_pc = abs(delta_ps) * stop_dist * 100.0
        loss_at_stop_pc = min(loss_at_stop_pc, max_loss_pc) if max_loss_pc > 0 else loss_at_stop_pc
        loss_at_stop_pc = max(loss_at_stop_pc, 1e-9)

        max_loss_mult = float(p["max_loss_mult"])
        max_prem_pct = float(p["max_premium_pct_of_equity"])
        if policy in (_POLICY_MINIMAL, _POLICY_DEFINED_RISK_SMALL):
            max_prem_pct = min(max_prem_pct, float(p["small_max_premium_pct_of_equity"]))

        vol_target = dollar_risk / loss_at_stop_pc
        maxloss_cap = (max_loss_mult * dollar_risk) / max(max_loss_pc, 1e-9)
        premium_cap = (max_prem_pct * equity) / max(outlay_pc, 1e-9)

        caps = {"vol_target": vol_target, "max_loss": maxloss_cap, "premium": premium_cap}
        binding = min(caps, key=caps.get)
        contracts = int(min(caps.values()))

        # equity that would make the binding cap reach exactly 1 contract
        need = {
            "vol_target": loss_at_stop_pc / (dollar_risk / equity) if equity > 0 and dollar_risk > 0 else float("inf"),
            "max_loss": max_loss_pc / (max_loss_mult * (dollar_risk / equity)) if equity > 0 and dollar_risk > 0 else float("inf"),
            "premium": outlay_pc / max_prem_pct,
        }
        min_viable_equity = max(need.values())

        return {
            "contracts": contracts,
            "binding": binding,
            "min_viable_equity": min_viable_equity,
            "risk_pct_at_1": loss_at_stop_pc / equity if equity > 0 else float("inf"),
            "diag": {
                "loss_at_stop_per_contract": round(loss_at_stop_pc, 2),
                "cap_vol_target": round(vol_target, 3),
                "cap_max_loss": round(maxloss_cap, 3),
                "cap_premium": round(premium_cap, 3),
                "dollar_risk_budget": round(dollar_risk, 2),
            },
        }

    # -------------------------------------------------------------- theta gate
    def _theta_gate(self, dec, signal, theta_ps, delta_ps, contracts):
        p = self.p
        hold = float(p["intraday_hold_fraction"])
        theta_cost = abs(theta_ps * 100.0 * contracts) * hold
        dec.diagnostics["theta_cost_over_hold"] = round(theta_cost, 2)
        # credit spreads / any net-positive-theta ticket: decay is a tailwind.
        if theta_ps >= 0:
            dec.diagnostics["theta_to_edge"] = 0.0
            return
        if signal.target is None:
            dec.diagnostics["theta_to_edge"] = None
            return
        move = abs(signal.target - signal.spot)
        exp_gross = abs(delta_ps) * move * 100.0 * contracts
        ratio = theta_cost / exp_gross if exp_gross > 0 else float("inf")
        dec.diagnostics["theta_to_edge"] = round(ratio, 3)
        if ratio > float(p["max_theta_to_edge"]):
            dec.warnings.append(f"theta_heavy:{ratio:.2f}>{p['max_theta_to_edge']}")
            if bool(p.get("reject_on_theta_gate", True)):
                dec.reason = "theta_exceeds_edge_budget"
                dec.ok = False
