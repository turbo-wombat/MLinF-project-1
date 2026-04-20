# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Coursework for MIT 15.C51 (Machine Learning in Finance) — Project 1. The brief (`15.C51-2026-Project1.pdf`) asks for two backtested DJIA trading strategies (one momentum, one mean-reversion) over the last 10 years, plus an Investment Committee Memorandum covering rationale, win/loss conditions, cash-flow statistics, and historical performance.

All work lives in a single notebook: `MomentumBasedStrategy.ipynb`. The mean-reversion strategy has not been started yet — only momentum exists.

## Running

Open `MomentumBasedStrategy.ipynb` in Jupyter and run top-to-bottom. The first cell runs `pip install hmmlearn` inline; other deps (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `yfinance`, `pandas_datareader`, `hmmlearn`) must be available in the kernel. There is no `requirements.txt`, test suite, or CI.

Data is fetched live each run:
- Prices/volume: `yfinance` (auto-adjusted close)
- Risk-free rate: FRED `DTB3` (3-month T-bill) via `pandas_datareader`

The `data/` directory is gitignored but currently unused — nothing is cached to disk.

## Pipeline architecture

The notebook is organized in labeled stages. The intended 5-stage pipeline is documented in a markdown cell near the top; current code implements stages 1–3 and 5 (regime filter / HMM is planned but not wired in yet, despite `hmmlearn` being imported).

**Stage 1 — Point-in-time constituent map.** `BASELINE_APR2016` + `CHANGES` reconstruct the DJIA membership day-by-day to avoid survivorship bias. `constituent_map` (date × ticker, bool) is used to mask non-members out of signals and weights. `WBA` is explicitly dropped from the download universe.

**Stage 2 — Cross-sectional signals.** Per-stock features built from price panel: `mom_12_1`, `mom_6_1`, `mom_3_1` (skip-one-month momentum at 12/6/3-month horizons), `vol_60` (annualized 60-day vol), `ma_cross` (50/200 SMA ratio). Features are stacked into a long panel, joined to a forward 21-day return target, and **subsampled every 21 rows to avoid overlapping labels** — this is important: shuffled CV on overlapping labels would leak.

**Stage 3 — Signal combination.** `StandardScaler` + `RidgeCV`. Train/test split is the first 70% of unique monthly dates; no shuffling.

**Stage 5 — Backtest.** Scores → monthly rebalance (`resample('ME').last().ffill()`) → cross-sectional rank → long top 20% / short bottom 20%, equal-weighted inside each leg → **`weights.shift(1)`** (signal on t, trade on t+1, no look-ahead) → 10 bps charged on turnover. Metrics: annualized return/vol, Sharpe, Sortino, max DD, Calmar, skew, kurtosis, vs equal-weighted buy-and-hold benchmark.

## Conventions to preserve

- **No look-ahead**: always `.shift(1)` weights before multiplying by returns.
- **Non-overlapping labels**: when building new signals with a forward-return target of horizon H, subsample training rows every H days before fitting.
- **Constituent masking**: apply `constituent_map` before ranking so delisted/non-member names don't receive weight on dates they weren't in the index.
- **Transaction costs**: charge on `weights.diff().abs().sum(axis=1)` — not on gross notional.
