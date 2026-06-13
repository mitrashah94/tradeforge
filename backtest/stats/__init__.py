"""backtest.stats — validation analytics for TradeForge P2 (MASTER_PLAN §5/§9).

The reusable validation backbone the ablation / edge-portfolio / correlation
stages consume:

  metrics            profit_factor, expectancy_r, expectancy_dollar, win_rate,
                     sharpe, max_drawdown, compute_metrics
  confidence         bootstrap CIs (pf_ci, expectancy_ci) + is_underpowered
  multiple_testing   deflated_sharpe, min_pf_threshold, HypothesisLog (log EVERY
                     hypothesis, not just survivors)
  oos                locked OOS vault: split_is_oos, lock_oos, oos_range,
                     assert_not_tuned_on_oos
  walk_forward       rolling IS/OOS folds: walk_forward, FoldResult, fold_windows
  regime             trend / chop / vol_shock tagging: regime_tags, by_regime

All functions are deterministic (CIs take a seed; the hypothesis log and OOS
vault take caller-supplied timestamps) and depend only on stdlib + numpy/pandas
+ the engine/result/data modules. No scipy.
"""

from backtest.stats.confidence import (
    UNDERPOWERED_THRESHOLD,
    ci_width,
    expectancy_ci,
    is_underpowered,
    pf_ci,
)
from backtest.stats.metrics import (
    TRADING_DAYS_PER_YEAR,
    compute_metrics,
    expectancy_dollar,
    expectancy_r,
    max_drawdown,
    max_drawdown_dollar,
    profit_factor,
    sharpe,
    sharpe_per_period,
    trade_pnls,
    trade_r_multiples,
    win_rate,
)
from backtest.stats.multiple_testing import (
    BASE_PF_THRESHOLD,
    DEFAULT_LOG_PATH,
    HypothesisLog,
    HypothesisRecord,
    deflated_sharpe,
    expected_max_sharpe,
    min_pf_threshold,
    norm_cdf,
    norm_ppf,
)
from backtest.stats.oos import (
    DEFAULT_VAULT_PATH,
    DateRange,
    assert_not_tuned_on_oos,
    lock_oos,
    oos_range,
    read_vault,
    split_is_oos,
    vault_is_locked,
    verify_checksum,
)
from backtest.stats.regime import (
    CHOP,
    REGIMES,
    TREND,
    VOL_SHOCK,
    RegimeConfig,
    by_regime,
    regime_tags,
    regime_tags_from_daily,
)
from backtest.stats.walk_forward import (
    FoldResult,
    aggregate_folds,
    fold_windows,
    walk_forward,
)

__all__ = [
    # metrics
    "profit_factor",
    "expectancy_r",
    "expectancy_dollar",
    "win_rate",
    "sharpe",
    "sharpe_per_period",
    "max_drawdown",
    "max_drawdown_dollar",
    "compute_metrics",
    "trade_pnls",
    "trade_r_multiples",
    "TRADING_DAYS_PER_YEAR",
    # confidence
    "pf_ci",
    "expectancy_ci",
    "is_underpowered",
    "ci_width",
    "UNDERPOWERED_THRESHOLD",
    # multiple testing
    "deflated_sharpe",
    "expected_max_sharpe",
    "min_pf_threshold",
    "HypothesisLog",
    "HypothesisRecord",
    "norm_cdf",
    "norm_ppf",
    "BASE_PF_THRESHOLD",
    "DEFAULT_LOG_PATH",
    # oos
    "split_is_oos",
    "lock_oos",
    "read_vault",
    "oos_range",
    "vault_is_locked",
    "verify_checksum",
    "assert_not_tuned_on_oos",
    "DateRange",
    "DEFAULT_VAULT_PATH",
    # walk forward
    "walk_forward",
    "FoldResult",
    "fold_windows",
    "aggregate_folds",
    # regime
    "regime_tags",
    "regime_tags_from_daily",
    "by_regime",
    "RegimeConfig",
    "TREND",
    "CHOP",
    "VOL_SHOCK",
    "REGIMES",
]
