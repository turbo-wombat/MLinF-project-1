# Pt2 → Pt3 Recommendations

Scope: improvements to `MomBased_Pt2.ipynb` that fit the *Moderate* change budget for Pt3. Architecture must stay long-only Dow-30 momentum with the XGBoost → Ridge-weighter → regime-filter shape. Tier 3 items are listed for completeness but will **not** be implemented in Pt3.

---

## Tier 1 — highest expected impact, will be implemented in Pt3

### T1.1 — Fix the `vol_60` sign problem by moving volatility out of the predictor set and into the *target*
**Problem.** XGBoost learned that low `vol_60` predicts high forward rank (gain-importance 0.377, the single strongest feature), which is why the final portfolio is a low-vol defensive basket instead of a momentum basket. An earlier Ridge iteration showed the exact same symptom (`vol_60` coefficient had the wrong sign).
**Fix.** Drop `vol_60` from the feature set used for prediction. Replace the raw 5-day-forward-return target with a **risk-adjusted forward return**: `fwd_ret / (ex-ante vol_60)`. This routes volatility into the label as a risk-adjustment denominator rather than as a free-variable predictor the model can abuse.
**Expected benefit.** The ranker stops using vol as a defensiveness filter. Expect feature importance to redistribute onto genuine momentum signals, and expect the portfolio to stop systematically avoiding MSFT/AAPL/UNH/V/HD. Directional expectation: Sharpe up, correlation to benchmark *down* (less closet-indexing).
**Risk if it goes wrong.** Dividing by ex-ante vol can amplify noise for stocks in very quiet periods (small denominator → large rank shift). Mitigate by flooring the denominator at a minimum volatility (e.g., 5% annualised).

### T1.2 — Enrich the feature set so that no single signal dominates
**Problem.** Three features (`mom_12_1`, `vol_60`, `rsi`) is a thin signal vocabulary. When one of them happens to carry a spurious relationship, the model has nothing to balance it against.
**Fix.** Add momentum-flavoured features that capture different horizons and shapes without re-introducing a naked volatility predictor:
  - `mom_6_1` (6-month skip-1 momentum, complementary horizon to 12-1)
  - `mom_3_1` (3-month skip-1 momentum, shorter horizon)
  - `mom_accel = mom_3_1 − mom_12_1` (momentum acceleration — are recent months *faster* than the long run?)
  - `mom_21` (1-month trailing return — captures short-term mean reversion when combined with longer horizons)
  - `ma_cross = SMA_50 / SMA_200 − 1` (trend-following moving-average crossover)
  - `dist_52wk_high = price / trailing_252d_max − 1` (how close to the 52-week high, a classic Jegadeesh-Titman signal)
**Expected benefit.** A richer feature set should let the XGBoost ranker differentiate between stocks more cleanly and reduce the defensiveness tilt. Also spreads feature importance more evenly, which reduces the damage when any single feature is noisy.
**Risk if it goes wrong.** More features → more opportunity to overfit the 2016–2020 training window. Mitigate with (a) stronger L2 regularisation on XGBoost, (b) purged walk-forward CV in Optuna (see T1.4), (c) cross-checking that feature importance is distributed rather than concentrated.

### T1.3 — Calibrate regime thresholds empirically rather than hard-coding them
**Problem.** The four regime thresholds (vol 0.17/0.28, drawdown −0.07/−0.13) are magic numbers. They happen to fire sensibly on this sample but there is no justification for the specific values, and the kill-switch fired only once in the OOS window — a sample size of one proves nothing.
**Fix.** Calibrate each threshold to a target frequency on the *training* sample (2016–2020 only). Specifically:
  - `vol_volatile` = 75th percentile of rolling 20-day vol over train
  - `vol_crash` = 95th percentile of rolling 20-day vol over train
  - `dd_volatile` = 25th percentile of rolling 60-day drawdown over train
  - `dd_crash` = 5th percentile of rolling 60-day drawdown over train
**Expected benefit.** Regime labels now have a consistent frequency interpretation rather than fitting a particular sample by accident. Less degrees of freedom silently burned on the test window.
**Risk if it goes wrong.** Train-sample percentiles might be very different from the test-sample distribution (regime shift), leading to over- or under-firing. Mitigate by reporting the regime-share breakdown on both train and test and flagging large discrepancies.

### T1.4 — Run Optuna with ≥100 trials and purged walk-forward CV inside the objective
**Problem.** 5 trials on a ~180-point discrete grid is essentially random. The objective is also a single-point Sharpe estimate on a fixed 2020–2022 validation slice, which is highly sensitive to that particular window.
**Fix.** Two changes in one:
  1. `N_OPTUNA_TRIALS = 100` (at minimum; 150 preferred if wall-clock allows).
  2. Inside the objective, use **purged walk-forward CV** on the train+validation span: 3–4 folds, each with a gap between train-end and validation-start of at least the forward-return horizon (5 days) to prevent overlapping-label leakage. The objective returns the *mean* validation Sharpe across folds, not a single-fold number.
**Expected benefit.** The Bayesian sampler actually explores. The CV average is a much more stable objective than a single-window Sharpe, which means the "best" trial generalises better OOS.
**Risk if it goes wrong.** 100 trials × 3 folds × one backtest per fold = ~300× the current compute. Will need to be careful about cache reuse (features per RSI period) and about Optuna's pruner (MedianPruner with proper `n_warmup_steps`). If the CV disagrees wildly with the final OOS Sharpe that is itself information — flag it.

---

## Tier 2 — moderate impact, will be implemented in Pt3 if time permits

### T2.1 — Try `rank:pairwise` (LambdaMART) XGBoost with date-keyed query groups
**Problem.** The current objective (`reg:squarederror` on a rank percentile) is a proxy for learning-to-rank. The proper formulation has query groups keyed by date, so the model learns to *order* stocks within a day rather than predict an absolute rank.
**Fix.** Group training rows by date, pass `group` to `xgb.DMatrix`, set `objective="rank:pairwise"`. Same feature set, same hyperparameter search.
**Expected benefit.** More faithful rank learning should tighten the ordering of the top-N picks. Unclear whether this translates to more Sharpe, but it is the technically correct objective.
**Risk if it goes wrong.** LambdaMART is more sensitive to class imbalance per group and to `eta` / `min_child_weight`; Optuna needs to co-tune these. Compute cost roughly doubles.

### T2.2 — Multi-horizon blended forward-return target
**Problem.** The 5-day forward-return target mismatches the 20-day rebalance horizon.
**Fix.** Blend three horizons into the target: `fwd = 0.25 * fwd_5d + 0.5 * fwd_21d + 0.25 * fwd_63d`, then rank. Subsample training rows so labels do not overlap the longest horizon (take every 63rd row to avoid leakage from overlapping 63-day windows).
**Expected benefit.** A training signal that better matches the trading horizon should produce more durable picks.
**Risk if it goes wrong.** Subsampling every 63 rows cuts training data by roughly 63×. May not leave enough training examples. If so, fall back to a single-horizon 21-day target and subsample every 21 rows.

### T2.3 — Explicit turnover penalty in the weighter
**Problem.** The ridge weighter does not consider prior weights, so positions churn freely. At zero modelled cost that is free; in reality it would not be.
**Fix.** Add a turnover penalty to the weighter objective: `max  Σ score_i·w_i − λ·Σw_i² − κ·Σ|w_i − w_i_prev|`. Closed-form is lost, so solve with `scipy.optimize.minimize`.
**Expected benefit.** Lower turnover at a small cost of signal responsiveness. This is useful even though transaction costs are not explicitly modelled, because it makes the strategy more implementable.
**Risk if it goes wrong.** Numerical optimiser may be slower than the current closed form (called ~50 times per backtest). If wall-clock blows up, skip.

---

## Tier 3 — out of scope for Pt3 (documented, deliberately not done)

### T3.1 — Transaction-cost / slippage modelling
Deferred by explicit instruction in PROMPT2.md (*"transaction costs / slippage modelling (that's Tier 3)"*). Would require charging e.g. 5 bps per leg on `weights.diff().abs().sum(axis=1)` and re-tuning. Expected to reduce CAGR by 50–200 bps depending on rebalance frequency; Sharpe roughly half that hit because vol also drops slightly.

### T3.2 — Expand the universe beyond the PIT Dow 30
Deferred by explicit instruction (*"universe (stays PIT Dow 30)"*). A fuller universe (S&P 100 or Russell 1000) would improve the statistical power of the cross-sectional ranking but violates the assignment scope.

### T3.3 — Different model family (LightGBM, CatBoost, transformers, LSTM)
Deferred by explicit instruction (*"NO switching to … a different ML family"*). Noted only because LightGBM's categorical handling and CatBoost's ordered boosting are known to be more robust to overfitting than XGBoost on tabular financial data.

### T3.4 — Long/short or market-neutral construction
Deferred by explicit instruction (*"strategy direction (stays long-only momentum)"*). A long/short version would meaningfully reduce beta and correlation to the benchmark, which is arguably the cleanest way to escape the 0.911 correlation problem, but violates the scope.

### T3.5 — True HMM regime classifier (wire up `hmmlearn`)
Deferred for a softer reason: the Moderate budget permits changing regime *thresholds* but a full HMM adds a new statistical layer with its own hyperparameters (number of states, transition prior, emission distribution) that would need its own calibration and CV. In a larger project it is worth doing; in Pt3 the percentile-calibrated thresholds in T1.3 deliver most of the benefit at a fraction of the complexity.

### T3.6 — Stacked/meta-model (second-stage XGBoost on first-stage residuals)
Deferred. A two-stage stack could squeeze more signal out of the same features, but with only three-to-nine features and ~4 years of training data, the second stage is very likely to overfit. Not worth the risk under the Moderate budget.

---

## Summary — what will change in Pt3

| # | Change | Tier | Rationale |
|---|---|---|---|
| T1.1 | Remove `vol_60` from features; use as denominator in risk-adjusted target | 1 | Kills the defensiveness tilt that caused 2.7 pp/yr underperformance |
| T1.2 | Add `mom_6_1`, `mom_3_1`, `mom_accel`, `mom_21`, `ma_cross`, `dist_52wk_high` | 1 | Richer signal vocabulary so no single feature dominates |
| T1.3 | Calibrate regime thresholds to train-sample percentiles | 1 | Kills four magic numbers; makes regime labels sample-stable |
| T1.4 | 100-trial Optuna with purged walk-forward CV inside the objective | 1 | Actually search the hyperparameter space; prevent CV leakage |
| T2.1 | `rank:pairwise` with date-keyed query groups (if time) | 2 | Technically correct ranking objective |
| T2.2 | Multi-horizon blended target (if time) | 2 | Match training horizon to trading horizon |
| T2.3 | Turnover penalty in the weighter (if time) | 2 | Reduce churn without modelling costs |
| T3.* | See Tier 3 above | — | Deliberately not done |
