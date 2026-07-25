# Swing Strategy Proposals (strategy-researcher)

> **Firewall (MASTER_PLAN §6):** these are PROPOSALS only. Nothing reaches PAPER→LIVE
> until it clears the full gauntlet — realistic-cost backtest → locked-OOS → walk-forward
> → by-regime → multiple-testing haircut → correlation/blend → human confirm. Written to
> `research/` (agent-writable); no live config touched.

## Framing — how swing differs from our intraday stack
- **Timeframe:** DAILY bars (some 4h). Holds: days → weeks. No EOD-flat.
- **Data fit:** daily OHLC is feed-robust (liquid-name daily H/L converges across venues),
  so the Alpaca-IEX intraday-wick problem largely vanishes. Swing can start on current data.
- **Sample power comes from BREADTH, not frequency.** A swing setup fires a few times/year
  per name → run it across 50–200 names to get statistically meaningful trade counts.
- **Cost drag is low** (few round-trips) — friendly to the small-account cost reality.
- **New risk = overnight GAPS.** A stop can be jumped on the open → realized loss can exceed
  1R; size for gap exposure. The dead-man's switch can't flatten while the market is closed,
  so swing DMS = reconcile + manage at the next open, not intraday flatten. Program-abort and
  drawdown halts still apply unchanged.
- **Engine change:** a `swing mode` — `eod_flat=False`, daily-bar event stream, multi-day
  position carry, stop/target checked per daily bar with the open-gap fill model.

## The set (designed for low mutual correlation)

### S1 — Trend Pullback (long-momentum continuation) — the core
- **Edge:** in established uptrends, pullbacks to dynamic support are bought; trend resumes
  (momentum + dip-buying behavior).
- **Universe/TF:** liquid equities + ETFs, daily.
- **Regime filter:** name above its 50-SMA AND 200-SMA (uptrend); optionally SPY > 200-SMA.
- **Entry:** pullback that tags the rising 20-EMA (or a higher swing-low) AND a reversal
  trigger — close back above the prior day's high, or RSI(14) turning up from ~40.
- **Stop:** below the pullback swing-low, or 2.0–2.5×ATR(14).
- **Exit:** Chandelier trail (3×ATR off the high) and/or scale 50% at +2R; time fail-safe.
- **Hold:** 3–15 days. **Params:** sma_trend, sma_mid, ema_pullback, atr_stop_k, rsi_floor.
- **Corr role:** long-momentum backbone (carries beta — note the SPY correlation).

### S2 — RSI(2) Oversold Reversion (counter-trend, within uptrend) — the decorrelator
- **Edge:** short-horizon mean reversion — sharp oversold dips in uptrending names snap back
  (Connors-style). BUYS weakness, so it is anti-correlated with S1/S3 (which buy strength).
- **Universe/TF:** liquid large-caps + index ETFs, daily.
- **Filter:** price > 200-SMA (only fade dips inside an uptrend).
- **Entry:** RSI(2) < 5–10 AND close < 5-SMA (optionally a 2–3 down-day streak).
- **Exit:** close > 5-SMA OR RSI(2) > 65; hard time-stop ~5 bars.
- **Stop:** wide catastrophe stop (mean-reversion needs room) ~3×ATR; rely mainly on the
  time/RSI exit. **Hold:** 1–5 days.
- **Corr role:** wins in chop/pullback regimes when momentum stalls → lowers blend variance.

### S3 — Donchian / 52-week-high Breakout (expansion momentum) — fat right tail
- **Edge:** breakouts to new highs continue (Turtle/trend-following); positive skew.
- **Universe/TF:** liquid equities + ETFs + liquid crypto, daily.
- **Entry:** close > 20- or 55-day Donchian high (optionally 52-wk high + volume expansion).
- **Stop/exit:** trailing Donchian lower (10–20 day low) or Chandelier 3×ATR; no fixed target
  (let winners run). **Hold:** weeks → months. **Params:** entry_lookback, exit_lookback, atr_k.
- **Corr role:** harvests the big trends S1 enters earlier/smaller; breakout trigger ≠ pullback
  trigger, but both are long-momentum (expect moderate corr to S1 — track it).

### S4 — Cross-Sectional Relative-Strength Momentum (portfolio ranking) — the smoother
- **Edge:** academic cross-sectional momentum — past 3–12mo winners keep winning over the next
  1–3mo. Decades of evidence; a DIFFERENT mechanism (relative ranking, not a single-name pattern)
  → smoother curve, lower variance.
- **Universe/TF:** broad liquid universe (top ~200 by liquidity), daily, rebalanced WEEKLY/MONTHLY.
- **Signal:** rank by blended 3- and 6-month total return, SKIP the most recent ~5 days (avoid
  short-term reversal); hold top N (10–20), equal- or inverse-vol-weighted.
- **Market filter:** deploy only when SPY > 200-SMA (momentum crashes in bear markets — the known
  tail); stand down / reduce otherwise.
- **Exit:** at rebalance, drop names that left the top quantile. **Hold:** weeks.
- **Corr role:** the portfolio backbone; lowest variance; where swing gets breadth-driven power.

### S5 — Diversifier (choose one; higher infra) — the true uncorrelated sleeve
- **(a) Pairs / market-neutral spread reversion:** long/short a cointegrated pair (e.g.
  AMD/NVDA, or sector ETFs); enter when spread z-score |z|>2, exit z→0, stop |z|>3.5.
  ~Zero beta → genuinely uncorrelated with everything directional. Cost: cointegration testing +
  two-sided execution.
- **(b) Post-Earnings-Announcement Drift (PEAD):** after a large positive earnings surprise +
  gap, hold the multi-week drift. Event-driven → uncorrelated with price-pattern edges. Needs an
  earnings-calendar + surprise feed (the deferred catalyst-watcher).
- **Corr role:** highest diversification value per the maximization thesis; most infra-dependent
  → propose last.

## Correlation design (why this set decorrelates)
- S1/S3 buy STRENGTH; S2 buys WEAKNESS → S2 is the natural anti-correlate.
- S4 is a different MECHANISM (relative ranking) → moderate corr, much lower variance.
- S5 is market-neutral or event-driven → the genuine ~zero-correlation sleeve.
The blend target is the same as MASTER_PLAN §0: maximize g ≈ mean − ½·var by stacking these so
the equity curve smooths and supports larger size at the same drawdown band.

## Validation plan (same rigor as the intraday work)
1. Add engine `swing mode` (no EOD-flat, daily stream, multi-day carry, open-gap fill model).
2. Ingest a broad DAILY universe (Alpaca daily — feed-robust; cheap for 200 names).
3. Per strategy: realistic-cost backtest with bootstrap PF/expectancy CIs + trade counts (breadth
   for power), locked-OOS vault, walk-forward, by-regime (trend/chop/vol-shock), log EVERY variant
   to the hypothesis ledger, apply the rising-PF multiple-testing haircut.
4. Build the live correlation matrix across S1–S5; show a blend beats the best single on
   risk-adjusted g; tear sheets per strategy + blend; REPORT.md per strategy stating what earned
   its place.
5. Only survivors that clear the haircut AND beat SPY after costs are eligible for paper→live.

## Notes / honest caveats
- S1/S3/S4 are all LONG-biased equity momentum → in a 2024–26 bull they will look great and carry
  hidden beta; the SPY > 200-SMA filter and the S2/S5 diversifiers are what keep the blend honest.
- Momentum (S4) has a real left-tail (momentum crashes); the market filter is mandatory, not optional.
- Swing trade counts per name are low → without breadth (many names) the CIs will be too wide to
  conclude anything — the same small-sample trap as the pasted intraday tables.
