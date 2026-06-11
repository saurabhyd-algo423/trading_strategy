# Exit Strategy: Old vs New Liquidation Modules

---

## Slide 1 — Title
Exit Strategy: From Filter-Based Recycling to Sharpe-Optimized Selection

Speaker notes:
- Presenters: (Your name)
- Audience: Manager & Technical Team
- Objective: Explain what `old_liquidation.py` did, its limitations, and how `new_liquidation_sharpe_v3.py` improves it.

---

## Slide 2 — Agenda
- Motivation & context
- The `old_liquidation.py` module: purpose & flow
- Limitations of the old approach
- The `new_liquidation_sharpe_v3.py` module: design & features
- Technical detail: Sharpe / Markowitz principles used
- What changed, why it matters, and evidence
- Next steps and recommendations

Speaker notes:
- Keep pace: 30–40 minutes total, 10–15 minutes discussion

---

## Slide 3 — Business Motivation
- Problem: Build a simple, repeatable monthly investment process with loss-management (liquidation) and cash recycling.
- Goals:
  - Regularly invest a fixed monthly amount, then recycle liquidation proceeds.
  - Manage losses by liquidating underperformers, subject to filter protection.
  - Track portfolio-level performance over long historical period.

Speaker notes:
- Emphasize real-world need: limited capital and risk control. The initial design prioritized simplicity and interpretability.

---

## Slide 4 — `old_liquidation.py` — What it does (Overview)
- Monthly process:
  - Find first trading day of month
  - Optionally liquidate loss-making holdings (multi-stage filter-protected)
  - Buy equal-dollar into all tickers from a single filter (`Filter 3`) on that day
  - Repeat across months; after 12 months reuse liquidation cash instead of monthly deposit
- Daily mark-to-market and metrics computation
- Outputs written to `output/Prabhav_test/value`

Speaker notes:
- Walk through monthly loop and the buy/liquidation order.
- Mention data sources (prices + filter JSONs + fundamentals).

---

## Slide 5 — `old_liquidation.py` — Outputs
- `portfolio_timeseries.csv` — monthly snapshots
- `portfolio_holdings.csv` — holdings per month
- `daily_portfolio_tracking.csv` — daily equity series
- `monthly_liquidation_log.csv` — liquidation amounts and diagnostics
- `performance_metrics.json` — final metrics (CAGR, Sharpe, drawdown)

Speaker notes:
- These files are the primary artifacts for performance analysis and audit.

---

## Slide 6 — Limitations of `old_liquidation.py`
- Buys are equal-weight across all filter picks — no portfolio-level risk control
- Single filter universe — ignores momentum signal
- Selection ignores covariance / diversification effects
- No explicit optimization objective — only filter membership
- Potential for concentrated, correlated positions and suboptimal risk-adjusted return

Speaker notes:
- Use simple example: two high-return stocks but highly correlated — equal buy can increase portfolio volatility.

---

## Slide 7 — Design Goals for Improvement
- Keep the liquidation logic (loss management & filter protection) intact
- Improve monthly investment decision to be portfolio-aware
- Use both value and momentum signals together
- Target risk-adjusted returns (Sharpe) rather than raw returns
- Keep implementation practical and computationally feasible for monthly backtests

Speaker notes:
- Practical constraint: must run many months; full MIQP solvers may be heavy.

---

## Slide 8 — `new_liquidation_sharpe_v3.py` — What it adds (Overview)
- Combined universe: Value_2_50% ∪ Momentum_2_50% ∪ current portfolio
- Historical return estimation (1-year windows) and covariance (5-year windows)
- Modified covariance stabilisation using cross-sectional avg stddev
- Greedy portfolio Sharpe selection: add stocks one-by-one to maximize Sharpe of `X̄ + Ȳ`
- Max new picks limited to 12; equal dollar allocation to each chosen stock
- Additional logs: `monthly_investment_log.csv`

Speaker notes:
- Emphasize this is a Markowitz / Sharpe-driven selection adapted to discrete, equal-allocation constraints.

---

## Slide 9 — Sharpe / Markowitz intuition (short)
- Input: expected returns vector `μ` and covariance matrix `Σ`
- Markowitz: choose weights `w` to maximize return for a given level of risk (or minimize variance for a return target)
- Sharpe objective: maximize `(μᵀw - r_f) / sqrt(wᵀΣw)`
- In practice: we approximate `μ` & `Σ` from historical windows and evaluate the incremental improvement in Sharpe when adding a stock

Speaker notes:
- Keep formulas light; focus on intuition: covariance creates diversification benefits.

---

## Slide 10 — Greedy Sharpe Selection (how it works)
- For invest month t:
  1. Build candidate set M (value 50% + momentum 50% + portfolio)
  2. Compute mean returns (1y) and cov matrix (5y) over 10-day windows
  3. Start with current portfolio monetary values `X̄`
  4. Allocate `A = invest_amt / n` where `n` ≤ 12
  5. Iteratively add the stock that yields largest increase in Sharpe of `X̄ + Ȳ`
  6. Stop when 12 stocks selected or no candidates left

Speaker notes:
- Explain the use of monetary values rather than normalized weights in selection. This matches real cash allocation.

---

## Slide 11 — Practical enhancements in covariance
- Covariance can be noisy when stocks have short histories
- The script builds a "modified covariance":
  - compute correlations from 5-year windows
  - replace each stock’s variance with the cross-sectional average variance
  - reconstruct covariance as `corr * avg_std^2`
- Result: more stable covariance matrix for selection

Speaker notes:
- Explain why noisy covariance can cause poor optimizer behavior.

---

## Slide 12 — How the new design fixes the old limitations
- Selection is portfolio-aware → reduces concentration of correlated picks
- Uses both value and momentum signals → broader opportunity set
- Explicit Sharpe objective → aligns selection with risk-adjusted performance
- Stability adjustments to covariance → less sensitivity to estimation noise
- Logging and fallback behavior → robust in periods with limited data

Speaker notes:
- Map each limitation from Slide 6 to the specific fix.

---

## Slide 13 — Example results (from latest run)
- Initial Capital: ₹12,000
- Final Capital: ₹150,327.94
- CAGR: 23.95% | Sharpe: 1.08 | Max Drawdown: 44.36%
- Notes:
  - These numbers are illustrative; compare to `old_liquidation.py` run to quantify improvement

Speaker notes:
- Suggest running both scripts with same seed/data and producing a side-by-side metric table.

---

## Slide 14 — Diagnostics & logs to review
- `monthly_liquidation_log.csv` — see how much was sold and which stages contributed
- `monthly_investment_log.csv` — see universe size, optimization used, and stocks selected
- `portfolio_timeseries.csv` / `daily_portfolio_tracking.csv` — raw equity curves for plotting
- `performance_metrics.json` — final numbers for reporting

Speaker notes:
- Explain how to interpret `Achievement_Percent` for liquidation and `Optimization` field for investment.

---

## Slide 15 — Next steps & recommendations
- Validate by A/B test: run both modules on same data and compare metrics and equity curves
- Add swap-improvement step after greedy selection (low-effort, improves results)
- Consider an MIQP solver for formal discrete optimization if compute cost acceptable
- Add unit tests and CI for reproducibility
- Produce a short handover doc for operations (how to run, expected outputs)

Speaker notes:
- Offer to implement swap-improvement and run A/B comparison if they approve.

---

## Slide 16 — Appendix: How to run locally
1. Activate the environment (Windows PowerShell):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\sharpeenv\Scripts\Activate.ps1
```
2. Run the new script:
```powershell
python .\new_liquidation_sharpe_v3.py
```
3. Outputs will be in `output/Prabhav_sharpe_v3/value_momentum_union`

Speaker notes:
- Mention dependency installation steps if env not present.

---

## Slide 17 — Appendix: Files referenced in the deck
- `old_liquidation.py` — original filter-based recycling engine
- `new_liquidation_sharpe_v3.py` — Sharpe-optimized investment upgrade
- `data/*` — prices, filters, fundamentals
- `output/*` — results produced by the scripts

Speaker notes:
- Provide pointers for devs to inspect specific functions: `execute_liquidation`, `greedy_portfolio_sharpe_selection`, `calculate_historical_returns`.

---

## Slide 18 — Closing / Q&A
- Questions? Specific scenarios to validate? Next action choices: (1) run A/B comparison, (2) add swap improvement, (3) export slides to PPTX

Speaker notes:
- Offer to prepare PPTX and speaker notes for the manager delivery.
