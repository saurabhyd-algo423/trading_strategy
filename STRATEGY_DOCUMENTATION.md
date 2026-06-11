# Strategy Documentation — `new_liquidation_sharpe_v3.py`

---

## What We Were Doing Earlier

The previous approach selected stocks by optimising the Sharpe ratio of each stock **individually** — picking the best standalone stocks without considering how they interact with each other or with the existing portfolio.

The investment universe was built from value and momentum filter lists only. The current portfolio holdings were treated as a fixed background — they contributed their existing values to the portfolio but were **not** candidates for re-selection or additional investment.

---

## What We Are Trying to Do Now

We now optimise the Sharpe ratio at the **portfolio level** — meaning we ask: *which combination of 12 stocks, when added to what we already hold, gives the best risk-adjusted return for the entire portfolio?*

The current portfolio holdings are also included as candidates. So every month, the optimiser considers both new stocks from the filter lists and stocks we already own.

---

## Objective — Step by Step

### Step 1: Build the Investment Universe (M)
Every month, we take the union of three sources:
- Stocks from the **Value filter** (`Filter_2_50%`)
- Stocks from the **Momentum filter** (`Filter_2_50%`)
- All stocks **currently held** in the portfolio

This combined set is M — the pool from which we will pick 12 stocks to invest in.

### Step 2: Compute Returns and Risk
For all stocks in M, we look back at historical price data:
- **Expected return** for each stock — estimated from the last **1 year** of data
- **Risk (covariance)** between every pair of stocks — estimated from the last **5 years** of data

We use a **modified covariance matrix** where every stock is treated as having the same volatility. This ensures the optimiser picks stocks based on their return potential and correlation structure, not just because one stock happens to be more volatile than another.

### Step 3: Select 12 Stocks Using Greedy Portfolio Sharpe

#### The Core Idea — X = X̄ + Ȳ

At any point in time the total portfolio has two components:

- **X̄ (existing holdings)** — stocks already held, valued at their current market price. This is fixed. No existing position is sold or changed during the investment step.
- **Ȳ (new investments)** — the money we are about to invest this month, allocated across up to 12 stocks from M.

Together: **X = X̄ + Ȳ** is the portfolio we will hold after investing. Our goal is to choose which 12 stocks to put money into so that X has the **best possible Sharpe ratio**.

---

#### What is the Portfolio Sharpe Ratio?

The Sharpe ratio measures return earned per unit of risk taken:

```
Sharpe = (Portfolio Return − Risk-Free Rate) / Portfolio Volatility
```

- **Portfolio Return** is the weighted average of expected returns of all stocks held, where each stock's weight is its share of total portfolio value.
- **Portfolio Volatility** captures not just how much each stock moves individually, but also how much they move **together** (correlation). Two stocks that always go up and down at the same time give less diversification benefit than two stocks that move independently.
- **Risk-Free Rate** is 6% per year — the return you could earn risk-free. We want to earn meaningfully above this.

A higher Sharpe ratio means we are getting more return for the same amount of risk — which is always preferable.

---

#### Why Not Just Pick the 12 Highest Return Stocks?

Because two stocks with high individual returns might be highly correlated — they tend to rise and fall together. If both are hit by the same market event, the portfolio suffers a large loss. A good portfolio balances high return with **low correlation between holdings** — this is the diversification benefit that the covariance matrix captures.

The portfolio Sharpe ratio naturally rewards this balance. A stock with a decent return but low correlation to everything else in the portfolio can improve the Sharpe more than a stock with a slightly higher return but high correlation to existing holdings.

---

#### Why the Greedy Approach?

The exact mathematical solution — trying every possible combination of 12 stocks from M to find the one with the highest portfolio Sharpe — is computationally impractical. If M has 40 stocks, there are over 500 million possible combinations of 12.

Instead we use a **greedy sequential algorithm**: build the portfolio one stock at a time, always adding the stock that improves the portfolio Sharpe the most at that step. This runs in seconds and produces a good solution.

---

#### The Greedy Algorithm — Step by Step

**Setup:**
- `working_portfolio` = current holdings X̄ (all existing stocks at their current market values)
- `A` = invest_amt / 12 (equal allocation amount per selected stock)
- `candidates` = all stocks in M that have sufficient historical data

**Repeat 12 times:**

1. For each stock still in `candidates`, **tentatively** add A rupees to `working_portfolio`
2. Compute the Sharpe ratio of the full combined portfolio after that tentative addition
3. Pick the stock that gives the **highest portfolio Sharpe** — call it the winner
4. **Lock in** the winner: permanently add A to `working_portfolio` for that stock
5. Remove the winner from `candidates`

After 12 rounds, the 12 locked-in stocks are the selected investments.

---

#### What Each Stock's Value Looks Like in the Portfolio

| Situation | Existing value (X̄) | New investment (Ȳ) | Total X |
|---|---|---|---|
| Brand new stock, selected | 0 | A | A |
| Already held stock, selected again | qty × price | A | qty × price + A |
| Already held stock, not selected | qty × price | 0 | qty × price |
| Stock outside M | — | 0 (not eligible) | X̄ only if held |

In every Sharpe computation, **all current holdings are always visible** — even stocks not being considered for new investment. The optimiser sees the complete picture of what the portfolio will look like after investing.

---

#### Why Including Current Holdings in M Matters

If a stock is already in the portfolio and performing well, the optimiser can choose to add more money into it — reinforcing a position it has already chosen. Without including current holdings in M, the optimiser would be forced to only invest in new stocks regardless of how good the existing ones are.

---

#### Fallback

If there is not enough historical price data to compute returns and covariance for any stock in M, the algorithm falls back to **equal allocation** — spreading the investment amount equally across up to 12 stocks in M.

### Step 4: Invest Equally Across the 12 Selected Stocks
The monthly investment amount is divided equally across the 12 selected stocks.

- Months 1–12: ₹1,000/month from initial capital
- Month 13 onwards: whatever cash was raised from liquidation that month

### Step 5: Liquidation (Runs Before Investment Each Month)
Before investing, we try to raise cash by selling loss-making stocks that no longer pass quality checks. This is done in stages — most important stocks are protected first, weakest protection last. The total amount sold is capped at 8% of the portfolio value per month.

---

## Summary

| | Before | Now |
|---|---|---|
| Stock selection | Individual Sharpe per stock | Portfolio-level Sharpe (combined) |
| Investment universe | Value + Momentum filters only | Value + Momentum filters + current holdings |
| Current portfolio role | Fixed background, not re-selectable | Active candidates for re-investment |
| Covariance | Standard | Modified — equalised volatility |
| Returns lookback | 5 years (same as covariance) | 1 year for returns, 5 years for covariance |
