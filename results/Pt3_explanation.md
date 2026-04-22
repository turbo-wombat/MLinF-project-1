# MomBased_Pt3 — Plain-English Walkthrough

## Executive summary (60-second read)

Pt3 keeps the same long-only Dow-30 architecture as Pt2 (XGBoost ranker → ridge weighter → regime filter) but makes four changes aimed at the specific problems Pt2's own diagnostics identified: it removes `vol_60` from the predictor set and uses it as a *denominator* in the target, adds six new momentum-flavoured features, calibrates regime thresholds to train-sample percentiles, and replaces the 5-trial Optuna run with a 150-trial search using 3-fold **purged walk-forward cross-validation** (*a walk-forward split where an explicit gap sits between training and validation, to prevent overlapping-label leakage*). On the held-out 2022-01 → 2026-04 test window Pt3 returned **6.01% per year** versus a benchmark of **10.15% per year**, with **Sharpe 0.50** (Pt2: 0.61), **max drawdown −13.1%** (Pt2: −20.8%), **Sortino 0.62**, **Calmar 0.46**, **information ratio −0.36**, **hit rate 37%**, and **4 kill-switches** (Pt2: 1). Verdict vs Pt2: **mixed** — the headline CAGR and Sharpe both got worse, but the maximum drawdown improved by 7.7 percentage points and the feature importance is now evenly distributed across the eight predictors (no single feature above 14%), confirming that the defensiveness-tilt problem is structurally fixed even though it did not translate into more absolute return on this particular test tape.

---

## Deep dive

### Context — what Pt2 was and what went wrong

To recap the Pt2 problem diagnosis (full detail in `Pt2_explanation.md`): the earlier notebook used three features (`mom_12_1`, `vol_60`, `rsi`), XGBoost trained as a regression on the rank of the 5-day forward return, a ridge-penalised long-only weighter over the top 12 scores, a regime filter with hardcoded thresholds, and an Optuna search that ran **only 5 trials** with no cross-validation. The gain-based feature importance put `vol_60` at 0.377, the single strongest signal — and because stocks with strong positive momentum are often also high-volatility, the model effectively learned a *negative* relationship between vol and forward rank (low-vol → high predicted rank). The consequence: the final Pt2 portfolio held classic defensives (IBM, CSCO, JNJ, KO, MMM, DIS) and omitted the actual 2023–2025 winners (MSFT, AAPL, UNH, V, HD). Pt2 finished at CAGR 8.25% vs benchmark 10.94%, with Sharpe 0.64 and max drawdown −21.8%.

Pt3 attacks this via four Tier-1 changes from `Pt2_recommendations.md`. The rest of this document describes each change, what the measured effect was, and what remains broken.

### Change 1 — Volatility out of the predictor set, into the target denominator

**What changed.** `vol_60` is no longer a feature the XGBoost ranker sees. Instead, the *target* the model is trained to predict is now the **risk-adjusted forward return**: `fwd_ret / max(ex-ante vol_60, 5%)`, cross-sectionally ranked as before. The 5% floor prevents very quiet stocks from blowing up the denominator. Volatility's information is still used — but as a *denominator* that normalises the reward signal, not as a predictor the model can use as a free defensiveness filter.

**Why.** Pt2's exact failure mode was that XGBoost used `vol_60` as a *negative* predictor. Moving it into the label removes that degree of freedom entirely: the model can no longer "pick low-vol stocks and call it momentum", because the label itself is already normalised by vol.

**Measured effect.**
- **Feature importance is now evenly distributed**: all eight features fall in the 11–14% band (mom_3_1 13.8%, ma_cross 13.3%, dist_52wk_high 12.9%, mom_12_1 12.8%, mom_accel 12.5%, mom_6_1 12.0%, mom_21 11.8%, rsi 10.9%). In Pt2, vol_60 alone carried 37.7%. The dominance problem is gone.
- However, the final Pt3 OOS Sharpe (0.50) is lower than Pt2's (0.61 on the same reproduced baseline). The defensiveness tilt that Pt2 had was bad in theory but *did* help in 2022 when low-vol defensives outperformed the broad market. Pt3 is less tilted defensively, so it lost that "accidental free alpha" along with the bias.

**Verdict.** Structurally correct fix, neutral-to-slightly-negative on the headline number for this particular test window. This is exactly the kind of trade-off where the *right* answer is not necessarily the *higher-Sharpe* answer on a 4-year sample.

### Change 2 — Six new momentum-flavoured features

**What changed.** The predictor set went from 3 features to 8:
- `mom_12_1` (kept) — canonical 12-1 momentum
- `mom_6_1` (new) — 6-month skip-1 momentum
- `mom_3_1` (new) — 3-month skip-1 momentum
- `mom_accel = mom_3_1 − mom_12_1` (new) — is recent momentum accelerating or decelerating?
- `mom_21` (new) — trailing 1-month return
- `ma_cross = SMA50/SMA200 − 1` (new) — trend-following moving-average crossover
- `dist_52wk_high = price / 252d_max − 1` (new) — Jegadeesh–Titman 52-week-high signal (always ≤ 0; closer to 0 = closer to 52-week high)
- `rsi` (kept) — 14-day Relative Strength Index

**Why.** Three features is a thin signal vocabulary. When one of them (`vol_60`) carried a spurious relationship, Pt2 had nothing to balance it against. Giving the model eight signals with diverse horizons (1-month, 3-month, 6-month, 12-month, trend-following) reduces the chance any single one can hijack the ensemble.

**Measured effect.**
- The top predictor in Pt3 is `mom_3_1` (13.8% gain). That is the shortest horizon in the set, which is interesting — the model found that *near-term* momentum, conditional on the risk-adjusted target, is the most informative signal. `ma_cross` and `dist_52wk_high` come next, both trend-following — another sensible finding.
- Feature importance is remarkably flat (range 10.9%–13.8%), which is a healthy sign: no single feature carries a disproportionate load, so the model should be more robust to any one feature's distribution shifting in the future.
- The absolute OOS performance did not improve from this alone; the benefit is structural (robustness), not numerical (return boost).

**Verdict.** Robustness win, neutral on performance.

### Change 3 — Regime thresholds calibrated to train-sample percentiles

**What changed.** Pt2's four regime thresholds (vol_vol=0.17, vol_crash=0.28, dd_vol=−0.07, dd_crash=−0.13) were hardcoded magic numbers. Pt3 calibrates each one to a percentile of the rolling measure computed on the training sample (2016-04 → 2020-01 only), then freezes those values for the rest of the backtest. Specifically: `vol_vol` = 75th percentile, `vol_crash` = 95th percentile, `dd_vol` = 25th percentile, `dd_crash` = 5th percentile.

The calibrated values came out to: `vol_vol = 0.1312`, `vol_crash = 0.2405`, `dd_vol = −0.0276`, `dd_crash = −0.0766`. Both volatility thresholds are *lower* than Pt2's hardcoded values, and both drawdown thresholds are *less negative* — meaning Pt3's regime filter is **more sensitive**, flipping to "volatile" or "crash" on smaller moves.

**Why.** The hardcoded Pt2 thresholds had no statistical justification. Calibrating to train-sample percentiles gives the regime labels a consistent frequency interpretation: `crash` is "a day whose vol is in the top 5% of 2016–2020 history" — a much more interpretable definition.

**Measured effect.**
- Regime-share split changed from Pt2's {75% trending, 18% volatile, 7% crash} to Pt3's **{52% trending, 35% volatile, 13% crash}**. The filter now fires "volatile" or "crash" on more than half of all days.
- Kill-switches fired **4 times** in the OOS window (vs Pt2's 1). Sample size is still small but is 4× larger.
- **Max drawdown improved from −21.79% to −13.12%** — a 7.7 percentage point improvement. This is the single biggest win in Pt3.
- Calmar ratio (CAGR / max DD) is 0.46 — a reasonable risk-adjusted-to-pain measure.

**Verdict.** Clear win on risk control. The cost is some upside participation during recoveries (the more aggressive filter exits sooner and re-enters later), which contributes to the lower CAGR.

### Change 4 — 150-trial Optuna with purged walk-forward CV

**What changed.** Pt2 ran 5 Optuna trials with a single-fold validation Sharpe as the objective. Pt3 runs **150 trials** with the objective equal to the mean Sharpe across **3 purged walk-forward folds** of the 2016-04 → 2022-01 train+val span, each with a 5-day embargo between the train cutoff and the validation start to prevent overlapping-label leakage (the 5-day forward-return label means consecutive training rows share prediction targets across a 5-day window). Two new XGBoost hyperparameters (`reg_lambda`, `max_depth`) are now also searched.

The best trial found:
- `rsi_period = 21` (up from Pt2's 14)
- `rebal_freq = 8w` (up from Pt2's 4w — Pt3 trades half as often)
- `ridge_lambda = 0.0224` (up from Pt2's 0.012)
- `top_n = 5` (down from Pt2's 12 — Pt3 holds far fewer positions)
- `xgb_reg_lambda = 2.84`
- `xgb_max_depth = 3`
- Mean CV-Sharpe across folds: **1.633**

**Why.** Five trials is not a search; 150 with a fold-averaged objective is. The embargo prevents leakage that would otherwise inflate the CV Sharpe.

**Measured effect.**
- CV-Sharpe ended at 1.633 — *far* higher than any single-fold number Pt2 produced (Pt2's best was 0.544).
- However, OOS Sharpe came in at **0.503**. The CV-to-OOS gap is **1.13 Sharpe points**, which is large and diagnostic in itself: the 2020–2022 CV span (COVID, recovery, onset of 2022 bear) is behaviourally quite different from the 2022–2026 test span (late-2022 bottom + 2023–2025 mega-cap bull + 2026 YTD slowdown). A model tuned hard on the CV folds will necessarily be fit to patterns that may not repeat.
- `top_n = 5` is a big decision. The model concentrates into five names per rebalance, which increases idiosyncratic risk per pick but also amplifies conviction. This probably contributes to both the lower drawdown (fewer positions → easier to avoid ugly ones if the picks are right) and the lower CAGR (fewer positions → less diversification benefit when picks are wrong).
- `rebal_freq = 8w` doubles the holding period. That reduces turnover — average one-sided turnover per rebalance is 1.26, and there are only 27 rebalances across the 4.3-year window, meaning annualised turnover is ~7.9×. This is a reasonable level for a monthly-adjacent strategy.

**Verdict.** Better search procedure, honestly implemented, but the large CV-to-OOS gap is a red flag that the CV fold composition may not resemble the test regime. This is a known hazard of walk-forward CV on a 10-year sample with one major regime change (2022).

---

### Final OOS metrics (held-out 2022-01 → 2026-04)

| Metric | Pt2 (reproduction) | Pt3 | Δ (Pt3 − Pt2) |
|---|---|---|---|
| CAGR | 8.11% | **6.01%** | −2.10 pp |
| Benchmark CAGR (PIT equal-weight) | — | 10.15% | — |
| Sharpe | 0.6115 | **0.5030** | −0.109 |
| Sortino | — | 0.6205 | — |
| Calmar | — | 0.4579 | — |
| Information ratio | — | −0.3615 | — |
| Max drawdown | −20.81% | **−13.12%** | +7.69 pp |
| Cumulative return | — | 28.11% | — |
| Hit rate (rebalance periods) | — | 0.3704 | — |
| Avg turnover (one-sided) | — | 1.26 | — |
| Number of rebalances | — | 27 | — |
| Kill-switches fired | 1 | 4 | +3 |

Note: the Pt2 "reproduction" column uses the exact same feature pipeline and data pull as Pt3 but restricted to the Pt2 configuration (3 features, raw fwd-return target, Pt2 regime thresholds, Pt2 best hyperparameters). It differs slightly from the 8.25% CAGR / 0.64 Sharpe reported in Pt2 itself because the Pt3 feature code is vectorised whereas Pt2's is a per-date Python loop; residual numerical differences are small but non-zero.

### Year-by-year returns (%)

| Year | Pt3 Strategy | PIT Benchmark | DIA ETF |
|---|---|---|---|
| 2022 | **−6.0** | −7.6 | −8.2 |
| 2023 | 10.1 | 18.3 | 16.0 |
| 2024 | 9.4 | 17.1 | 14.8 |
| 2025 | **18.5** | 16.7 | 14.7 |
| 2026 YTD | −5.8 | 0.2 | 0.1 |

Pt3 *beats* the benchmark in 2022 (less painful drawdown) and 2025 (outright outperformance), but underperforms by roughly half the benchmark's return in 2023 and 2024, and posts a meaningful YTD loss in an approximately flat 2026 tape. The pattern is consistent with a strategy that has shed its low-vol bias (Pt2 would have done worse in 2022 and comparable elsewhere) but hasn't successfully loaded on the actual mega-cap bull winners.

---

### What still doesn't work

1. **The CV-to-OOS Sharpe gap is 1.1 points (1.63 → 0.50).** That is very large. The most likely explanation is that the three walk-forward folds all fall inside the 2020–2022 COVID/recovery/early-bear span, which has quite different cross-sectional dispersion than the 2022–2026 test tape. The CV is optimising for a regime that didn't repeat. Fixing this properly would require either (a) training data longer than 10 years so the CV folds cover multiple regimes, or (b) more explicit regime-conditional modelling.

2. **The strategy still underperforms in the 2023–2024 bull runs.** Pt3 does not pick MSFT/AAPL/UNH/V/HD aggressively enough to match the benchmark on those years. Removing the low-vol bias helped in 2022 but did not translate into heavier concentration on the actual winners — the risk-adjusted target still penalises those names somewhat (they had high vol_60 during the training window). A better fix might be to use *realised* volatility of a shorter trailing window (e.g., 21-day vs 60-day) in the denominator.

3. **Hit rate of 37% is below random.** Over 27 rebalance periods the strategy beat the benchmark in 10 of them. A fair coin would hit 50%. This says the long-only top-5 bet is losing more 8-week matchups than it wins, and only barely survives because the wins (when they come) are larger than the losses. The 2026 YTD −5.8% is one of those losing matchups.

4. **Information ratio of −0.36 is negative.** This formalises the "loses to benchmark" observation: not only does the strategy return less than the benchmark, its tracking error means it is *worse* on a pure skill-vs-market basis. A long-only momentum strategy should ideally have a positive IR over a 4-year window; negative IR is a failure mode, not a success mode, even if Sharpe is still mildly positive.

5. **Four kill-switches, but only three of them arguably helped.** The 2022 and 2026 YTD kills likely avoided larger drawdowns; the kills during 2023–2024 whipsaws may have cost participation in the subsequent recoveries. Without a counterfactual this is hard to prove, but the higher kill frequency and the lower CAGR together suggest some kills were costly.

### What we deliberately did not try and why (the Tier-3 items from `Pt2_recommendations.md`)

1. **Transaction-cost / slippage modelling** — explicitly Tier 3 in PROMPT2.md (*"transaction costs / slippage modelling (that's Tier 3)"*). Charging e.g. 5 bps per leg on turnover would reduce Pt3's CAGR further (at 7.9× annual turnover that's roughly 40 bps/yr hit to CAGR). Not implemented under the Moderate change budget.

2. **Expanding the universe beyond the PIT Dow 30** — explicitly out of scope (*"universe (stays PIT Dow 30)"*). A larger universe would give the cross-sectional ranker more power, but the assignment specifies Dow-30.

3. **Different model family (LightGBM, CatBoost, transformers)** — explicitly out of scope (*"NO switching to … a different ML family"*). LightGBM's categorical-feature handling and CatBoost's ordered boosting are both known to be more robust against overfitting on tabular financial data, but replacing XGBoost would violate the PROMPT2.md constraint.

4. **Long/short or market-neutral construction** — explicitly out of scope (*"strategy direction (stays long-only momentum)"*). A long/short version would be the cleanest way to cut the strategy-vs-benchmark correlation, but would change the strategy's identity.

5. **Full HMM regime classifier** — the Moderate budget allows changing regime thresholds but a true Hidden Markov Model adds latent-state hyperparameters (number of states, transition prior, emission distribution) that would need their own CV. Percentile-calibrated thresholds deliver most of the benefit at a fraction of the complexity; the full HMM is deferred.

6. **Stacked / meta-model second stage** — with only eight features and ~5.5 years of training data, a second-stage model would almost certainly overfit. Not worth the risk.

7. **Tier-2 items skipped due to time / redundancy** — `rank:pairwise` LambdaMART with date-keyed query groups (T2.1) and the multi-horizon blended target (T2.2) were listed as "if time permits" in the recommendations. They would have doubled compute cost without clear expected benefit given the CV-to-OOS gap we already observe; they are not implemented in Pt3. The turnover-penalised weighter (T2.3) would have required swapping the closed-form ridge projection for a numerical optimiser; not implemented.

---

### Overall verdict

Pt3 is a **structurally better strategy than Pt2** — the feature importance is evenly distributed, the regime filter has a defensible statistical basis, and the hyperparameter search is honest. But on the specific 2022-01 → 2026-04 test window, Pt3 **underperforms Pt2 on Sharpe and CAGR** while **outperforming on max drawdown**. This is reported as-is and not massaged. The large CV-to-OOS Sharpe gap (1.63 → 0.50) is the main diagnostic finding; it suggests the 2020–2022 CV window is not representative of the 2022–2026 test tape, which in turn suggests the next round of iteration should focus less on more-features / more-trials and more on making the training signal more robust to regime shifts — most likely through a shorter-horizon risk-adjustment denominator, explicit regime conditioning inside the ranker, or a longer training sample. None of those are in Pt3's Moderate change budget.
