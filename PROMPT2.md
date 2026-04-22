You are working on an MIT 15.C51 momentum trading project. Three files are available in the project directory:
- MomBased_Pt2.ipynb         ← current version of the strategy (DO NOT MODIFY)
- MomentumBasedStrategy.ipynb ← earlier version, reference only
- 15_C512026Project1.pdf      ← the assignment spec + grading rubric

The work is split into four phases. Complete them in order. Do not skip phases. Do not modify MomBased_Pt2.ipynb at any point — it is the frozen baseline.

═══════════════════════════════════════════════════════════════
HARD CONSTRAINTS (apply to ALL phases)
═══════════════════════════════════════════════════════════════
- Strategy MUST remain long-only cross-sectional momentum on the Dow 30.
- Architecture MUST remain: XGBoost cross-sectional ranker → Ridge-regularised long-only weights → regime-scaled exposure (crash = flat, volatile = partial, trending = full).
- Universe MUST stay point-in-time Dow 30 (use the existing _BASELINE_APR2016 + _CHANGES logic).
- NO switching to mean-reversion, pairs trading, short legs, or a different ML family (no LightGBM, no neural nets, no linear-only).
- NO look-ahead: every feature on day t uses only data through t, and trades execute on t+1 prices.
- Data source stays yfinance with the same date range (2016-04-01 through 2026-04-18).

═══════════════════════════════════════════════════════════════
PHASE 1 — Analyse MomBased_Pt2.ipynb
═══════════════════════════════════════════════════════════════
Read Pt2 end-to-end. Read the PDF so you know what the final report is being graded on.

Then produce TWO outputs:
  (a) results/Pt2_explanation.md — deep-dive written for a reader with LIMITED data-science AND LIMITED finance background. Rules for this file:
      • First use of any finance term gets a one-sentence plain-English definition in parentheses. Cover at minimum: momentum, long-only, rebalance, Sharpe ratio, CAGR, drawdown, max drawdown, volatility (annualised), walk-forward, out-of-sample (OOS), point-in-time (PIT) universe, benchmark, RSI.
      • First use of any ML term gets the same treatment. Cover at minimum: feature, target/label, XGBoost, decision tree, ridge regression, L2 regularisation, hyperparameter, Optuna, TPE, regime, cross-validation, rank percentile.
      • Use one concrete analogy per hard concept (e.g., "XGBoost is a team of small decision trees where each new tree specialises in fixing the mistakes the previous ones made").
      • Walk through the pipeline in narrative order: data layer → feature engineering → regime classifier → XGBoost ranker → Ridge optimiser → backtest loop → Optuna search → final metrics.
      • For each stage, explain WHAT the code does, WHY that choice was made, and what COULD go wrong with it.
      • Include the actual numbers Pt2 reported: CAGR 8.25%, benchmark 10.94%, Sharpe 0.6359, max DD -21.79%, correlation 0.911 with benchmark, 180 crash days, 1 kill-switch fired, only 5 Optuna trials.
      • Flag the issues the notebook itself identified (vol_60 coefficient had the wrong sign in the earlier Ridge regression; XGBoost systematically picked low-vol defensives like IBM/CSCO/JNJ/KO/MMM/DIS and avoided the actual winners MSFT/AAPL/UNH/V/HD; weight sum 0.70 on the final rebalance because it landed on a 'volatile' day).
      • No cheerleading. No "this notebook demonstrates...". Direct, honest, technical prose.
      • Target length: 1500–2500 words.
  (b) A 4–6 sentence PLAIN-ENGLISH executive summary at the top of the same .md file, above the deep-dive, that a non-technical reader could understand in 60 seconds.

═══════════════════════════════════════════════════════════════
PHASE 2 — Recommendations
═══════════════════════════════════════════════════════════════
Produce results/Pt2_recommendations.md containing a prioritised, explicit list of what should change. Structure:

  • Tier 1 (highest expected impact, will be implemented in Pt3)
  • Tier 2 (moderate impact, will be implemented in Pt3 if time permits)
  • Tier 3 (out of scope for Pt3 under the "Moderate" change budget — e.g., transaction-cost modelling, richer universe, different model family) — still list them so the user can see what was deliberately NOT done.

For each recommendation give: (1) the problem in one sentence, (2) the proposed fix, (3) why you expect it to help, (4) the risk if it goes wrong.

The Moderate change budget for Pt3 means you MAY touch:
  • the feature set (add new momentum-flavoured features; fix the vol_60 sign issue by changing how volatility enters the model, e.g., as a risk-adjustment denominator rather than a raw predictor)
  • the target/label (e.g., risk-adjusted forward return, multi-horizon blend, residualised return)
  • regime-detector thresholds (calibrate empirically rather than using the current hardcoded 0.17/0.28/-0.07/-0.13 numbers)
  • the Optuna search (more trials — 100+; wider or smarter search space; purged/embargoed walk-forward CV to kill leakage)
  • the XGBoost ranker objective (e.g., keeping reg:squarederror vs trying rank:pairwise with proper query groups by date)

You MAY NOT touch:
  • strategy direction (stays long-only momentum)
  • the broad pipeline shape (XGBoost → Ridge → regime filter)
  • universe (stays PIT Dow 30)
  • transaction costs / slippage modelling (that's Tier 3)

═══════════════════════════════════════════════════════════════
PHASE 3 — Build MomBased_Pt3.ipynb
═══════════════════════════════════════════════════════════════
Create a NEW notebook called MomBased_Pt3.ipynb that implements your Tier 1 (and Tier 2 if feasible) recommendations. Requirements:

  • Must run end-to-end on Colab without manual intervention (!pip install lines at top, fixed random seed).
  • Must produce the same summary metrics Pt2 produces: CAGR, benchmark CAGR, Sharpe, max drawdown, cumulative return, kills.
  • Must ALSO report: information ratio vs PIT benchmark, Sortino ratio, Calmar ratio, hit rate, turnover per rebalance, and year-by-year return table (strategy vs benchmark vs DIA ETF).
  • Must save the same chart outputs Pt2 saves (results/backtest_summary.png, results/optuna_analysis.png, results/best_params.txt) plus a new results/pt3_vs_pt2_comparison.png that overlays the two equity curves on the same OOS window.
  • Must use Optuna with AT LEAST 100 trials.
  • Must use purged walk-forward cross-validation (or at minimum embargoed CV with gap ≥ label horizon) inside the Optuna objective to prevent leakage across overlapping forward-return labels.
  • Must include a short brief-inline markdown summary cell above every major code section (Data Collection, Features, Regime, XGBoost, Ridge, Backtest, Optuna, Results) — 2–4 sentences each, same accessible audience as the .md file, saying what the cell does AND what changed vs Pt2.
  • Actually execute the notebook to completion and save the executed .ipynb with outputs visible.
  • If the final Pt3 result is WORSE than Pt2 on Sharpe or CAGR, do NOT silently cherry-pick the best intermediate run — report the honest final number and note it explicitly in Phase 4. This is a research notebook, not a marketing piece.

═══════════════════════════════════════════════════════════════
PHASE 4 — Explain MomBased_Pt3.ipynb
═══════════════════════════════════════════════════════════════
Produce results/Pt3_explanation.md with the same audience and same structural rules as Pt2_explanation.md. Additional requirements:

  • Open with a 4–6 sentence executive summary including the final Pt3 metrics and a one-line verdict vs Pt2 (improved / mixed / worse).
  • For each change made, describe: what was changed, why (linking back to the specific Pt2 problem), and what the measured effect was (feature importance shifts, Sharpe delta, drawdown delta, turnover delta, hit-rate delta).
  • Include a "what still doesn't work" section — be honest about remaining issues.
  • Include a "what we deliberately didn't try and why" section listing the Tier 3 items from Phase 2.
  • Target length: 1500–2500 words.

═══════════════════════════════════════════════════════════════
DELIVERABLES CHECKLIST
═══════════════════════════════════════════════════════════════
☐ results/Pt2_explanation.md          (Phase 1)
☐ results/Pt2_recommendations.md      (Phase 2)
☐ MomBased_Pt3.ipynb                  (Phase 3, executed with outputs)
☐ results/backtest_summary.png        (Phase 3)
☐ results/optuna_analysis.png         (Phase 3)
☐ results/best_params.txt             (Phase 3)
☐ results/pt3_vs_pt2_comparison.png   (Phase 3)
☐ results/Pt3_explanation.md          (Phase 4)
☐ MomBased_Pt2.ipynb                  UNCHANGED — verify with a diff before you finish.

Begin with Phase 1. Do not jump ahead.