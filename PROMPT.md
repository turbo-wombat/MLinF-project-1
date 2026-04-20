# Claude Code Prompt — Proprietary Momentum Trading Strategy

> Paste this entire document as the first message to Claude Code in the repo.
> It is also the **source-attribution artifact** the ICM must reference verbatim, so **do not edit it during the project** — if the scope changes, issue a new prompt and log it.

---

## 0. Mission

Build a single notebook, **`ProprietaryMomentumStrategy.ipynb`**, that develops, backtests, iteratively improves, and documents a proprietary **momentum** trading strategy on the Dow Jones 30, over a ~10-year window (Apr 2016 – Apr 2026), then produces a professional **Investment Committee Memorandum (ICM)**.

This is Project #1 for MIT 15.C51 (Spring 2026), "Proprietary Trading" track, **momentum half only**. Mean reversion is a separate notebook — do not implement mean-reversion strategies here, but you may reference the dichotomy briefly in the ICM.

Grading rubric (25 pts each, 100 total):
1. **Completeness of financial analysis** — comparable to a bulge-bracket research report
2. **Novelty of the analysis** — beyond textbook 12-1 momentum; push into residual, regime-gated, ML-stacked variants
3. **Readability and attractiveness** — institutional-grade prose, publication-quality figures
4. **Completeness of source attribution** — every LLM interaction logged (model, prompt, purpose)

Every design decision in this notebook should be justifiable against at least one of those four axes.

---

## 1. Reference Files (read first, before writing anything)

1. **`MomentumBasedStrategy.ipynb`** — contains the data pipeline you will reuse (yfinance pull, FRED DTB3 risk-free rate, DJIA constituent tracking with historical index changes). Read it end-to-end. The data layer is good; the strategy layer (Ridge on stacked features) is the baseline you must beat.
2. **`15.C51-2026-Project1.pdf`** — the assignment. Pay attention to the ICM requirements: strategy description + rationale, profit/loss conditions, statistical properties of cash flows and returns, 10-year historical performance.

After reading both, confirm your understanding in one paragraph before writing code.

---

## 2. Non-Negotiable Constraints

### Data
- Universe: DJIA 30, ~April 2016 – April 2026
- **Must** use the historical constituent map (no survivorship bias). Reuse `BASELINE_APR2016` + `CHANGES` logic from the reference notebook.
- Prices/volume: `yfinance` with `auto_adjust=True`
- Risk-free rate: FRED `DTB3`, converted to daily, ffill business days
- Forward-fill price gaps with `limit=10` trading days; document any stock dropped for excessive gaps

### Backtest Integrity
- **No look-ahead.** Signal at date *t*, trade at *t+1*. Grep for every `.shift(` call and verify sign.
- **Transaction costs:** base case 10 bps per unit turnover (`weights.diff().abs().sum(axis=1) * 0.001`). Sensitivity tests at 5 and 20 bps in Stage 6.
- **Walk-forward expanding window**, ≥ 3 folds. Minimum: train 2016–2020 → test 2021; train 2016–2021 → test 2022; train 2016–2022 → test 2023–2026. Do **not** use a single train/test split — that's what the baseline does and it's not credible.
- No hyperparameter tuning on the test window. CV only within the training window of each fold.

### Benchmarks (compute for every strategy)
- Equal-weighted DJIA buy-and-hold (built from the same price matrix)
- Price-weighted DJIA index (`^DJI` from yfinance)
- SPY (broader US equity)
- Risk-free rate (compounded DTB3)

Any strategy that does not beat EW-DJIA B&H on risk-adjusted terms after costs is not worth shipping, regardless of raw return.

---

## 3. Notebook Structure

Organize into these stages with explicit markdown headers. Each stage ends with a commit.

### Stage 0 — Setup & Reproducibility
- Pin random seeds (numpy, python `random`)
- Print versions of `pandas, numpy, scikit-learn, yfinance, hmmlearn, statsmodels, xgboost, pandas_datareader`
- Define a single `RUN_CONFIG` dict at the top: `START_DATE, END_DATE, COST_BPS, LOOKBACKS, REBALANCE_FREQ, SEED, WALK_FORWARD_FOLDS`. Every downstream cell reads from it — no magic numbers elsewhere.
- Set matplotlib rcParams once (consistent font, figsize, grid style).

### Stage 1 — Data Layer (reuse pipeline)
- Port the data pull, constituent map, and rf pipeline from `MomentumBasedStrategy.ipynb`.
- Data validation cell: shape, date range, avg number of constituents/day, missing-data report per ticker.
- Visualize universe composition over time (stacked line showing #stocks).

### Stage 2 — Strategy Zoo (open exploration, 15 strategies)
Implement each as a pure function `strategy_name(prices, volume, constituent_map, cfg) -> weights_df`. Monthly rebalanced, daily-ffilled weights. Long top 20% / short bottom 20% unless otherwise noted.

**Classical:**
1. Cross-sectional 12-1 momentum (Jegadeesh & Titman 1993)
2. Cross-sectional 6-1 momentum
3. Cross-sectional 3-1 momentum
4. Time-series absolute momentum (long if 12m return > rf, else cash) — Moskowitz/Ooi/Pedersen 2012
5. Dual momentum — absolute filter on relative winners (Antonacci 2012)
6. Volatility-scaled TSMOM — each stock scaled to target vol before aggregation

**Advanced / novel:**
7. Risk-adjusted cross-sectional momentum — signal ÷ rolling 60d vol
8. **Residual momentum** — regress 12-1 returns on market beta (rolling 252d), use residual as signal (Blitz, Huij, Martens 2011)
9. Industry-neutralized momentum — rank within GICS sector, long/short within sector
10. Short-term reversal overlay — long 12-1 winners, short 1-month winners (combines Jegadeesh 1990 reversal with momentum)
11. Acceleration momentum — ∆(12-1 momentum) as signal
12. Volume-confirmed momentum — signal × log(volume / rolling avg volume)
13. **HMM regime-gated momentum** — fit 3-state GaussianHMM on market returns; scale exposure: bull=1.0x, choppy=0.5x, bear=0x (Daniel & Moskowitz 2016 inspiration)
14. **ML-stacked momentum** — XGBoost or LightGBM on feature matrix (all signals above as inputs) predicting next-month cross-sectional rank; walk-forward refit
15. Ensemble of ranks — average percentile rank across strategies 1, 2, 3, 7, 8

For each: produce weight matrix, daily return series (net of costs), and one-row summary. Verify sanity checks: weights sum to 0 (market-neutral), abs weights sum ≤ 2 (no leverage beyond 100%/100%).

### Stage 3 — Evaluation Framework
Build **one** `evaluate(returns, benchmark, rf, label) -> dict` that returns:
- Annualized return, volatility, Sharpe, Sortino, Calmar
- Max drawdown, drawdown duration (days), time underwater %
- Skew, kurtosis (excess), hit rate (% positive months)
- Turnover (avg annualized), average holding period (days)
- Information ratio vs benchmark
- CAPM alpha, beta, R² vs benchmark (OLS)
- **Fama-French 3-factor** regression (alpha, MKT-RF, SMB, HML) — pull factors via `pandas_datareader.famafrench`
- Bootstrap 95% CI on Sharpe (1000 resamples, block bootstrap for serial correlation)
- Newey-West t-stat on mean excess return (lag = 5)

Output a master leaderboard DataFrame: rows = 15 strategies, columns = metrics. Sort by **walk-forward OOS Sharpe**. This is the single most important table in the project.

### Stage 4 — Diagnostic Visuals (top 5 only, to keep the notebook readable)
For each of the top 5 by OOS Sharpe:
- Cumulative return (log scale) vs all 4 benchmarks
- Rolling 12-month Sharpe
- Underwater (drawdown) chart
- Return distribution: histogram + KDE + QQ-plot vs normal
- Monthly return heatmap (rows = years, cols = months)
- FF3 factor exposure bar chart with 95% CIs
- Turnover over time

### Stage 5 — Iterative Improvement Loop
Starting from the **#1 ranked strategy** after Stage 3, iterate up to **5 rounds**. Stop early when **OOS Sharpe improvement between consecutive rounds < 0.05**.

Each iteration must produce a markdown "Iteration Diary" cell and a code cell, structured as:

1. **Diagnose** (markdown) — explicitly name the top 3 weaknesses of the current strategy. Evidence-based, with numbers. Examples: "worst drawdown of -28% concentrated in March 2020", "turnover 340%/yr drives 2% annual cost drag", "negative skew -1.2 indicates tail risk", "Sharpe collapses in 2nd half of test window".
2. **Propose** (markdown) — 2–3 concrete, citation-backed improvements.
3. **Implement & backtest** (code) — add the improvement; rerun walk-forward.
4. **Compare** (code) — head-to-head metrics table vs previous round.
5. **Decide** (markdown) — keep or revert. Log to `iteration_log` DataFrame with columns `[round, change, prev_sharpe, new_sharpe, delta_sharpe, delta_dd, delta_turnover, decision, rationale]`.

Typical improvement levers (not exhaustive — justify each choice):
- Volatility targeting (scale portfolio to 10% annualized vol)
- Regime-conditional position sizing
- Exponential lookback weighting vs flat
- Ledoit-Wolf shrinkage for risk-parity weighting within quintiles
- Trade-threshold rebalancing (trade only if ∆rank > δ) to cut turnover
- Tail-risk overlay (buy OTM puts funded by covered calls, or simple vol-filter cash position)
- Signal denoising (rolling median, exponential smoothing)
- Cross-validated L1/L2 tuning, feature pruning
- Walk-forward hyperparameter refit cadence

The iteration diary is as important as the code — it demonstrates analytical rigor (novelty + readability points).

### Stage 6 — Robustness Tests (winner only)
- **Cost sensitivity:** Sharpe curve at 0, 5, 10, 15, 20 bps
- **Rebalance frequency:** weekly, monthly, quarterly
- **Lookback sensitivity:** ±20% on each lookback
- **Subperiod stability:** metrics in each 2-year window (2016-17, 2018-19, 2020-21, 2022-23, 2024-25)
- **Monte Carlo bootstrap** of daily returns (1000 paths, 2-year horizon): report 5/50/95 percentile equity curves
- **Stress periods:** isolated performance in (a) 2020 COVID crash Feb–Apr, (b) 2022 bear market, (c) March 2023 regional banking crisis
- **Capacity estimate:** notional AUM at which market impact consumes 50% of alpha (rough back-of-envelope using avg daily volume)

### Stage 7 — Investment Committee Memorandum
Pure markdown cells. Formatted like a real IC memo. Length targets in parentheses.

**Header** — Strategy name (give it something memorable), Date, Authors, Classification
**1. Executive Summary** (½ page) — headline Sharpe, max DD, turnover, capacity, one-sentence thesis
**2. Strategy Description & Rationale** (1.5 pages) — intuition (behavioral + risk-based), precise signal math in LaTeX, portfolio construction, risk management, rebalancing protocol
**3. Profit & Loss Conditions** (1 page) — *when it works:* trending markets, persistent dispersion, stable regimes. *When it fails:* momentum crashes (cite 2009 post-GFC reversal, Daniel & Moskowitz 2016), crowded-trade unwinds, regime transitions, vol spikes. Back each claim with a dated example from the 10-year sample.
**4. Statistical Properties** (1 page) — full metric table, return distribution commentary (skew, kurtosis, tails), autocorrelation of returns, factor exposures, cash flow characteristics (monthly P&L distribution, worst-month expected)
**5. Historical Performance — 10-Year Review** (1–2 pages) — cumulative equity curve, year-by-year table, rolling Sharpe, drawdown calendar with narrative
**6. Risk Factors & Limitations** (½ page) — capacity, crowding, data mining, cost sensitivity, regime dependence, model risk
**7. Recommendation** (¼ page) — sizing, fit in a diversified book, governance triggers to pause/reduce
**8. Appendix A — Source Attribution**
 - Full text of this prompt (verbatim)
 - Every LLM interaction: model + version, date, prompt, purpose, what was used from the output
 - Academic citations: Jegadeesh & Titman 1993, Asness 1994, Carhart 1997, Moskowitz/Ooi/Pedersen 2012, Antonacci 2012, Daniel & Moskowitz 2016, Blitz/Huij/Martens 2011, Frazzini/Pedersen 2014 as relevant
 - Data sources with URLs
**9. Appendix B — Iteration Log** — the full `iteration_log` DataFrame rendered as a table

Writing style: first-person plural, declarative, quantitative. Every claim sourced or numbered. No hedge phrases like "it is believed that". No purple prose.

---

## 4. Code Quality Rules

- Every cell re-runnable given prior cells (Kernel → Restart & Run All must pass before you claim done)
- Type hints on all functions
- Docstrings on every strategy and evaluation function (one-line summary + args + returns)
- Functions < 60 lines — refactor if longer
- No print statements inside strategy functions; use a module-level logger
- All parameters in `RUN_CONFIG` — no magic numbers in function bodies
- Save a `results.pkl` after Stage 3 and after Stage 5 so re-running is cheap
- One top-of-notebook "How to run" markdown cell

## 5. Git Hygiene

- Commit after each stage with message `stage-N: <what>` (e.g., `stage-2: implement residual momentum`, `stage-5: iter-2 add vol targeting`)
- Do **not** commit raw data — data is fetched on each run
- Do commit `results.pkl` if it's under ~5 MB, otherwise add to `.gitignore`

## 6. Execution Plan

Follow this order. Do not skip ahead.

1. Read `MomentumBasedStrategy.ipynb` and `15_C512026Project1.pdf`, then write a one-paragraph confirmation of scope.
2. Scaffold `ProprietaryMomentumStrategy.ipynb` with empty Stage 0–7 headers + empty code cells. Commit.
3. Implement Stage 0, 1 fully; run; verify data sanity; commit.
4. Implement Stage 2 **one strategy at a time**. After each, run the single-strategy backtest and spot-check Sharpe/DD look reasonable. Commit after every 3 strategies.
5. Implement Stage 3, produce the leaderboard. Commit.
6. Stage 4 visuals. Commit.
7. Stage 5 iteration loop — **commit after every round**, regardless of keep/revert decision.
8. Stage 6 robustness.
9. Stage 7 ICM.
10. Final: Kernel → Restart & Run All, confirm clean execution, commit `final: all stages complete`.

## 7. Completion Checklist

Before declaring done:
- [ ] Walk-forward validation implemented, not single split
- [ ] `git grep 'shift('` reviewed, no look-ahead anywhere
- [ ] Transaction costs applied to every strategy
- [ ] Leaderboard ranks all 15 strategies
- [ ] Iteration log has ≥ 2 rounds (even if stopping criterion hit early, at least 1 improvement attempt beyond baseline)
- [ ] Every ICM section populated with numbers, not placeholders
- [ ] Appendix A includes this prompt verbatim
- [ ] Appendix A logs every Claude Code interaction during the project
- [ ] FF3 regression present for the winning strategy
- [ ] Bootstrap CI on Sharpe present
- [ ] Notebook runs end-to-end from a fresh kernel
- [ ] Git log reads as a coherent story
- [ ] No TODOs or `#FIXME` left in the code

Begin by reading the two reference files.
