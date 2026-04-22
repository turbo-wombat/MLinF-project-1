# MomBased_Pt2 — Plain-English Walkthrough

## Executive summary (60-second read)

This notebook builds a long-only momentum strategy on the Dow Jones 30: every four weeks a machine-learning ranker scores the 30 stocks, picks twelve, and a shrinkage weighter spreads cash across them. A separate regime filter flattens the portfolio to cash when the market looks like it is crashing. On a held-out 2022-to-2026 test window the strategy returned **8.25% per year** versus a **10.94% per year** Dow-30 benchmark, with a **Sharpe ratio of 0.64**, a **maximum drawdown of −21.8%**, and a **0.911 correlation** to the benchmark. Two things explain the underperformance: the ranker's most influential feature is *volatility*, which caused it to systematically pick low-volatility defensives (IBM, CSCO, JNJ, KO, MMM, DIS) instead of the stocks that actually led the 2023–2025 rally (MSFT, AAPL, UNH, V, HD); and the hyperparameter search ran **only 5 Optuna trials**, which is far too few to locate a good region of the parameter space. Net: the pipeline is structurally sound and has no look-ahead leakage, but as configured it is a low-vol basket in a momentum wrapper, and it loses to simply holding the index.

---

## Deep dive

### What this notebook is trying to do

The MIT 15.C51 Project #1 brief (the Proprietary Trading track) asks for a backtested Dow Jones 30 **momentum strategy** (*the empirical pattern that stocks outperforming over the past 3–12 months tend to keep outperforming over the next 1–3 months*), paired with an Investment Committee Memorandum. Pt2 is the current working draft of the momentum leg.

Its architecture is:

1. Pull ten years of daily prices for every stock that has been in the Dow 30 during the study window.
2. Build a handful of numerical features per stock per day.
3. Train an XGBoost model to predict the *cross-sectional rank* of each stock's next 5-day return.
4. At every **rebalance** (*the act of recalculating and resetting portfolio weights on a schedule — here every four weeks*), score all eligible stocks, pick the top twelve, and allocate cash with a ridge-penalised weighter.
5. Overlay a **regime filter** (*a rule that decides whether the market is in a calm, choppy, or crashing state*) that either lets the portfolio run at 100%, sizes it down to 70%, or puts it entirely in cash.
6. Tune the hyperparameters with Optuna, then score the tuned configuration on a held-out **out-of-sample (OOS)** window (*data the model has never seen, used as the final honest test*).

The strategy is **long-only** (*the portfolio only holds positive positions — no short-selling to profit from declines*), cross-sectional (*decisions depend on how stocks compare to each other on a given day, not on each stock's own history in isolation*), and operates on a fixed **point-in-time (PIT) universe** (*the stocks that actually were in the index on each historical date, not just the ones that are still members today*).

The rest of this document walks through the pipeline in the order the notebook runs it, and for each stage describes *what the code does*, *why that choice was made*, and *what could go wrong*.

---

### Stage 1 — Data layer and the point-in-time universe

`fetch_prices()` pulls ten years of daily auto-adjusted closing prices from Yahoo Finance for every ticker that has ever been in the Dow 30 between 2016-04-01 and 2026-04-18. The notebook then reconstructs a daily PIT membership set from a 2016-April baseline (`_BASELINE_APR2016`) plus a hand-coded list of additions and removals (`_CHANGES`).

**Why PIT matters.** Without PIT masking you get **survivorship bias** (*the free-money fantasy of only studying names that are still alive today*), which pretends a 2016 backtest knew in advance that GE would be booted in 2018 and NVDA added in 2024. PIT masking ensures that on any given date the strategy can only consider stocks that were *actually* in the index on that date.

Two tickers (WBA and UTX) are dropped outright: WBA has known data gaps in yfinance; UTX was renamed to RTX before the study window starts.

**What could go wrong.** The `_CHANGES` list is hand-maintained from a Wikipedia lookup. If a date or ticker is wrong, every backtest downstream silently inherits the error. Also, auto-adjusted close prices bake dividends back into the price series — fine for return calculations, but it means the "price" the model sees is not the price that actually traded on that day.

---

### Stage 2 — Feature engineering

`compute_features()` walks day by day through the price panel. For each stock in the PIT universe on each day it computes three **features** (*numerical inputs the model will use to make predictions*):

1. **`mom_12_1`** — the canonical "12-1 momentum": the return from roughly 12 months ago to roughly 1 month ago. The one-month skip is standard because stocks show short-term mean reversion, so including last month's return tends to hurt.
2. **`vol_60`** — the **annualised volatility** (*the standard deviation of daily log returns, scaled by √252 to go from daily to yearly units — i.e., how jumpy the stock has been*) over the last 60 trading days.
3. **`rsi`** — the 14-day **Relative Strength Index** (*a classic technical indicator on a 0–100 scale that ratios recent up-days against recent down-days; values below 30 are called "oversold", above 70 "overbought"*).

Each feature row is paired with a **target/label** (*the thing the model is trying to predict*) called `fwd_rank`: the stock's 5-day forward return converted into a cross-sectional **rank percentile** (*where the stock sits in that day's ranking, 0 = worst in the Dow, 1 = best*). Predicting a rank rather than a raw return is deliberate — a long-only top-N strategy only cares about *relative* outperformance.

**What could go wrong.** Three features is a tiny feature set; real momentum systems often use twenty-plus signals combining different horizons, different volatility regimes, industry tilts, and price/earnings filters. With only three features the model cannot meaningfully differentiate between stocks on a typical day, which shows up later as tightly clustered XGBoost scores and near-equal portfolio weights. The 5-day forward-return horizon also mismatches the 20-day trading horizon (4-week rebalance), which is a known source of degradation: the model is trained to care about next-week performance but the backtest holds the position for four weeks.

---

### Stage 3 — Regime detection

`classify_regimes()` builds a synthetic market index from the equal-weight average of the PIT constituents, then computes two things on it: the rolling 20-day annualised volatility, and the rolling 60-day **drawdown** (*how far the index is below its trailing 60-day high; a 10% drawdown means you are 10% below the recent peak*). Each trading day is then labelled:

- `crash` if rolling vol > 28% OR drawdown < −13%
- `volatile` if rolling vol > 17% OR drawdown < −7% (and not already `crash`)
- `trending` otherwise

A **regime** (*a statistically distinct market state*) is used here as a coarse risk switch rather than as a Bayesian state estimate. The notebook imports `hmmlearn` (for Hidden Markov Models, a principled way to infer latent regimes) but never actually wires one in — this is flagged as a TODO in the notebook itself.

Over the full 2016–2026 sample the classifier labels 1,895 days (75.0%) trending, 451 days (17.9%) volatile, and **180 days (7.1%) crash**, firing around early 2018, late-2018 selloff, Feb–Mar 2020 (COVID), 2022 bear market, and early 2025. Over the OOS test window the kill-switch fired **exactly once**, which is a sample size of one.

**What could go wrong.** The four thresholds (0.17, 0.28, −0.07, −0.13) are hard-coded, not calibrated. They happen to produce sensible-looking regime labels on this particular sample, but on a different window they could be over- or under-sensitive. And a single OOS kill-switch event tells us nothing statistically — we cannot distinguish "the crash filter is protecting the portfolio" from "the crash filter got lucky once".

---

### Stage 4 — XGBoost cross-sectional ranker

`train_xgboost()` fits an XGBoost model on the training features and labels.

**What XGBoost is.** A **decision tree** (*a cascade of if-then rules that splits the feature space into buckets with similar targets*) is the building block. **XGBoost** is a stack of 300 shallow decision trees trained in sequence, where each new tree specialises in fixing the mistakes the previous ones made — picture a team of interns where each intern only reviews the cases the previous intern got wrong, and you add their correction back to the running prediction. Small learning rate (0.05) and shallow trees (max_depth 3) keep any single tree from dominating; **L2 regularisation** (`reg_lambda = 1.0`) is a penalty proportional to the sum of squared leaf weights that discourages the model from assigning extreme values.

The model is trained as a regression (`reg:squarederror`) on the rank percentile. The code comment flags that the more technically correct approach would be `rank:pairwise` (LambdaMART) with query groups keyed by date, but that requires additional plumbing; regression on the rank percentile is the standard simplified workaround.

**Measured feature importance on the final test fit** (*gain-based — how much each feature reduced the loss when used in a split*): `mom_12_1 = 0.320`, `vol_60 = 0.377`, `rsi = 0.303`. Volatility is the single most informative feature.

**What could go wrong — and did.** A long-only momentum strategy wants to *buy* the fastest-moving names. But the single strongest signal in this model is `vol_60`, and because stocks with strong positive momentum are often *also* high-volatility, the ensemble learned an effectively *negative* relationship between vol and forward rank in the training window. An earlier iteration of this project used a Ridge linear regression on the same features, and the `vol_60` coefficient came out with the **wrong sign** (implying lower vol → higher predicted rank) — the code explicitly called this out as a red flag. XGBoost replaces Ridge in Pt2 but inherits the same underlying data relationship: it uses `vol_60` as a *defensiveness filter* rather than a momentum amplifier.

The consequence is visible in the final portfolio. On the last rebalance the twelve long positions are IBM, CSCO, JNJ, CAT, NVDA, KO, AXP, MMM, AMZN, DIS, SHW, GS — six of those are textbook low-vol defensives (IBM, CSCO, JNJ, KO, MMM, DIS). The holdings notably *omit* MSFT, AAPL, UNH, V, and HD, five of the Dow's strongest performers over 2023–2025. The strategy says "momentum" on the tin but behaves like a low-volatility basket in practice.

---

### Stage 5 — Ridge-regularised long-only weighter

`ridge_optimize()` takes the XGBoost scores, picks the top N, and solves the simple quadratic problem:

> maximise  Σ score_i · w_i  −  λ · Σ w_i²     subject to   Σ w = 1,   w ≥ 0

This is **ridge regression** in name only — it is really a ridge-penalised *portfolio weighter*. For the uninitiated, **ridge regression** is the linear-regression variant with an L2 penalty on coefficients; the same idea transferred to portfolio construction pushes weights away from extreme concentrations and toward equal-weight. The closed-form solution is `w_i ∝ max(0, score_i) / (2λ)`, then rescaled so the weights sum to 1 (projection onto the probability simplex).

With the winning λ = 0.0124 and `top_n = 12`, the resulting weights on the final rebalance span IBM 6.5% down to GS 5.5% — a spread of one percentage point across twelve positions. That is essentially equal-weight. The ridge term is almost inactive at this λ; the XGBoost scores are tightly clustered enough that the simplex projection dominates.

**What could go wrong.** A ridge-style weighter over near-identical XGBoost scores cannot differentiate between positions, so the "smart money" allocation collapses to "hold twelve things equally". If the XGBoost scores were more spread out — or if the weighter used something like mean-variance optimisation — the portfolio would express the model's conviction more meaningfully.

---

### Stage 6 — Backtest loop

`run_backtest()` walks through the test dates one day at a time. The split is hard-coded:

- **Train**: 2016-04 → 2020-01 (includes 2018 selloff)
- **Validation** (inside Optuna): 2020-01 → 2022-01 (includes COVID crash and recovery)
- **Test** (final held-out): 2022-01 → 2026-04 (includes 2022 bear market and the 2023–2025 bull run)

On each rebalance day (every `rebal_freq × 5` trading days):
1. Pull today's PIT universe.
2. Score each PIT stock with the *frozen* XGBoost model trained on pre-2020 data.
3. Read the regime label for today. If `crash`: flatten to cash and increment the kill-switch counter. If `volatile`: size at 70% of full exposure. If `trending`: 100% exposure.
4. Run `ridge_optimize()` on the non-crash path.

On non-rebalance days the weights are held constant. The daily strategy return is the weighted average of held positions; the benchmark return is the unweighted average of the PIT universe (i.e., an equal-weight Dow 30 that also respects point-in-time membership). There is no explicit `.shift(1)` in the code — the signal computed on day *t* is only used for trade decisions at rebalance times, and returns are computed with today's prices only, so there is no same-day leakage.

**Final OOS metrics** on the 2022-01 → 2026-04 window:

| Metric | Strategy | Benchmark (PIT EW) |
|---|---|---|
| CAGR (*the constant compound annual rate that would grow the starting cash into the ending cash*) | **8.25%** | 10.94% |
| Sharpe ratio (*return per unit of risk; >1 is considered good, <0 is bad*) | **0.6359** | ~0.76 |
| Annualised volatility | 14.0% | 14.8% |
| Max drawdown | **−21.79%** | — |
| Cumulative return | +40.2% | — |
| Correlation to benchmark | **0.911** | 1.00 |
| Kill-switches fired | **1** | n/a |

**Year by year** (strategy vs PIT benchmark):

- 2022: −12.4% vs −7.6%
- 2023: +14.2% vs +18.3%
- 2024: +8.9% vs +17.1%
- 2025: +17.9% vs +16.7%
- 2026 YTD: +9.1% vs +3.5%

The strategy loses in every full calendar year except 2025, and draws down harder than the benchmark in the 2022 bear. The 0.911 correlation confirms that it is moving with the index most of the time — it is not expressing a meaningfully differentiated view.

**One more wrinkle.** The weight sum on the *final* rebalance snapshot is **0.70**, not 1.00. This is not a bug: the last rebalance date happened to land on a "volatile" day, and the code correctly multiplied every weight by 0.70. But it means the reported final portfolio snapshot is 30% in cash, which is a confusing artefact if you are trying to report "our current positions".

**What could go wrong.** No transaction costs are modelled. Rebalancing twelve positions every four weeks at zero cost is free alpha in any backtest; at 5 bps per side the headline numbers drop meaningfully. The single-OOS-path evaluation is also statistically thin — this is one 4-year window, not a distribution.

---

### Stage 7 — Optuna hyperparameter search

`run_optuna()` runs a Bayesian hyperparameter search. **Optuna** is a Python library that does this; its default sampler is **TPE (Tree-structured Parzen Estimator)** (*a method that models the probability of good hyperparameters given observed trial results, then samples the next trial from the region most likely to improve the objective*). TPE is much more sample-efficient than grid search, but only when you give it enough samples to model from.

**Search space**:
- `rsi_period` ∈ {7, 10, 14, 21, 28}
- `rebal_freq` ∈ {1, 2, 4, 6, 8, 12} weeks
- `ridge_lambda` ∈ log-uniform [0.01, 1.0]
- `top_n` ∈ {3, 5, 7, 9, 12, 15}

That is 5 × 6 × 6 = 180 discrete combinations before counting the continuous λ.

**Number of trials actually run**: **5**.

This is not a typo — `N_OPTUNA_TRIALS = 5` in the code. There is no **cross-validation** (*a standard ML technique that splits the training set into multiple folds and averages performance across them, to reduce overfitting to any single fold*); every trial is a single-point estimate of Sharpe on a fixed 2020–2022 validation window.

**What could go wrong — and did.** Five trials of TPE on a ~180-point discrete grid has essentially no chance of finding a strong region of the hyperparameter space. The "best" trial (RSI=14, rebal=4w, λ=0.012, N=12, validation Sharpe 0.544) is better than the other four by luck, not by optimisation. The final OOS Sharpe of 0.64 is therefore an unreliable estimate of what this architecture is capable of with a proper search. A production sweep would run 100–200 trials with **purged walk-forward cross-validation** (*a walk-forward split where the gap between train and validation is ≥ the forward-label horizon, to prevent leakage through overlapping labels*) inside the objective.

---

### Known issues identified in the notebook itself

The final markdown cells in Pt2 correctly diagnose the root cause:

1. **XGBoost is using `vol_60` as a defensiveness filter.** The feature importance says vol is the strongest signal (0.377 gain), and the resulting portfolio holds classic low-vol defensives (IBM, CSCO, JNJ, KO, MMM, DIS) rather than the 2023–2025 bull-market winners (MSFT, AAPL, UNH, V, HD). This is not a code bug — it is the model doing exactly what the training data told it to do.
2. **Weight sum 0.70 on the final rebalance** is a side-effect of that rebalance landing on a "volatile" day. The reported snapshot is therefore partially in cash, which is correct behaviour but confusing documentation.
3. **Optuna ran only 5 trials**, a sample size that cannot reliably differentiate hyperparameter combinations.

Any honest fix has to address at least two of these three: either (a) remove `vol_60` from the *predictive* feature set and let it enter as a risk-adjustment denominator in the target instead, or (b) enrich the feature set so that volatility no longer dominates the signal; plus (c) run the full ≥100-trial Optuna sweep with purged walk-forward CV. These are exactly the changes proposed in the Tier-1 recommendations in `Pt2_recommendations.md`.
