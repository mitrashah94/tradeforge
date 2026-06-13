"""backtest/engine/result.py — trade records, equity curve, and summary stats.

``BacktestResult`` holds the per-trade ledger (a DataFrame), the equity curve
(a pandas Series indexed by bar timestamp), and a :meth:`summary` that returns
the headline metrics the gate and later stages report on:

    profit_factor, expectancy_$, expectancy_R, win_rate, n_trades,
    gross_profit, gross_loss, max_drawdown_pct, net_profit

Profit factor is sizing-robust (gross_profit / gross_loss), so the gate verdict
does not depend on the chosen sizing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

_TRADE_COLUMNS = [
    "symbol",
    "side",
    "entry_ts",
    "exit_ts",
    "entry_price",
    "exit_price",
    "ref_entry_price",
    "signal_close",
    "stop",
    "target",
    "shares",
    "gross_pnl",
    "costs",
    "pnl",
    "r_multiple",
    "exit_reason",
    "bars_held",
    "mfe",
    "mae",
    "equity_after",
]


@dataclass
class TradeRecord:
    """One completed round-trip trade."""

    symbol: str
    side: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    ref_entry_price: float
    signal_close: float
    stop: float
    target: float | None
    shares: float
    gross_pnl: float
    costs: float
    pnl: float
    r_multiple: float
    exit_reason: str
    bars_held: int
    mfe: float
    mae: float
    equity_after: float

    @staticmethod
    def to_frame(trades: list["TradeRecord"]) -> pd.DataFrame:
        """Build the trades DataFrame (empty-but-typed if no trades)."""
        if not trades:
            return pd.DataFrame(columns=_TRADE_COLUMNS)
        return pd.DataFrame([asdict(t) for t in trades])[_TRADE_COLUMNS]


class BacktestResult:
    """Holds trades + equity curve and computes summary metrics."""

    def __init__(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.Series,
        initial_equity: float,
        symbol: str | None = None,
    ):
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_equity = float(initial_equity)
        self.symbol = symbol

    # ------------------------------------------------------------- helpers
    def _max_drawdown_pct(self) -> float:
        """Max peak-to-trough drawdown of the equity curve, as a percent."""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return 0.0
        eq = self.equity_curve.astype(float)
        running_peak = eq.cummax()
        drawdown = (eq - running_peak) / running_peak
        return float(-drawdown.min() * 100.0)

    # ------------------------------------------------------------- summary
    def summary(self) -> dict:
        """Return the headline metrics as a plain dict."""
        t = self.trades
        n = int(len(t))
        if n == 0:
            return {
                "n_trades": 0,
                "profit_factor": float("nan"),
                "expectancy_dollar": 0.0,
                "expectancy_R": 0.0,
                "win_rate": float("nan"),
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "net_profit": 0.0,
                "max_drawdown_pct": 0.0,
                "n_long": 0,
                "n_short": 0,
                "avg_win_R": float("nan"),
                "avg_loss_R": float("nan"),
            }

        pnl = t["pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())  # positive magnitude
        net_profit = float(pnl.sum())

        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )
        win_rate = float((pnl > 0).mean())
        expectancy_dollar = float(pnl.mean())
        expectancy_R = float(t["r_multiple"].astype(float).mean())

        r = t["r_multiple"].astype(float)
        win_R = r[r > 0]
        loss_R = r[r < 0]

        return {
            "n_trades": n,
            "profit_factor": profit_factor,
            "expectancy_dollar": expectancy_dollar,
            "expectancy_R": expectancy_R,
            "win_rate": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_profit": net_profit,
            "max_drawdown_pct": self._max_drawdown_pct(),
            "n_long": int((t["side"] == "long").sum()),
            "n_short": int((t["side"] == "short").sum()),
            "avg_win_R": float(win_R.mean()) if len(win_R) else float("nan"),
            "avg_loss_R": float(loss_R.mean()) if len(loss_R) else float("nan"),
        }

    def date_range(self) -> tuple[datetime | None, datetime | None]:
        """First entry ts and last exit ts across all trades."""
        if len(self.trades) == 0:
            return (None, None)
        return (
            self.trades["entry_ts"].min(),
            self.trades["exit_ts"].max(),
        )
