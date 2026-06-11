# Cross-Entropy Method for 12-Stock Sharpe Selection

## Problem Statement

You have a current portfolio `N`, represented by fixed monetary values `X̄`, and a new investment universe `M` defined as:

- `M = Value_Filter_2_50% ∪ Momentum_Filter_2_50%`

You must select exactly 12 stocks from `M` and invest equally in them.

### Decision variables

- `y_i ∈ {0, 1}` for each stock `i ∈ M`
- `∑_{i ∈ M} y_i = 12`
- `Ȳ_i = A · y_i`, where `A = invest_amt / 12`

### Portfolio after investment

- `X = X̄ + Ȳ`

### Sharpe objective

The portfolio Sharpe is:

```
Sharpe(X)
= (R^T X) / sqrt(X^T Σ X)
= R^T(X̄ + A y) / sqrt((X̄ + A y)^T Σ (X̄ + A y))
```

Where:

- `R` is the vector of expected returns for stocks in `M`
- `Σ` is the covariance matrix for stocks in `M`


## Why this is hard

The objective is:

- binary (`y_i ∈ {0,1}`)
- cardinality constrained (`∑ y_i = 12`)
- fractional and quadratic
- non-additive because covariance induces interactions across stocks

That means the optimal subset is not necessarily discoverable by a simple greedy ranking of individual stocks.


## Cross-Entropy Method Overview

Cross-entropy is a global search algorithm for combinatorial selection problems. Instead of building a subset one stock at a time, it learns a probability distribution over stocks and samples entire 12-stock subsets.

### Main idea

1. Maintain selection probabilities `p_i` for each stock `i ∈ M`
2. Sample many candidate subsets of size 12 according to `p`
3. Score each subset by the Sharpe objective
4. Keep the top-performing subsets (elite set)
5. Update `p` to match the frequency of stocks in elite subsets
6. Repeat until `p` converges or quality stabilizes

This is a search over whole portfolios, not a sequential greedy construction.


## Formal steps

### 1. Define the objective for a subset

For any binary selection vector `y` with `∑ y_i = 12`:

```
S(y) = Sharpe(X̄ + A y)
     = R^T (X̄ + A y) / sqrt((X̄ + A y)^T Σ (X̄ + A y))
```

This is the value we maximize.

### 2. Initialize probabilities

Start with a uniform probability distribution over stocks in `M`:

```
p_i = 12 / |M|
```

or any weak prior that reflects initial belief.

### 3. Sampling

Generate `K` candidate subsets of size 12 using probability weights `p`.

A common sampling strategy is weighted random selection without replacement.

### 4. Evaluation

For each sampled subset `y^{(k)}`, compute the portfolio Sharpe `S^{(k)}`.

### 5. Elite selection

Choose the top `ρ` fraction of samples by Sharpe.

- Example: `ρ = 0.1` or `0.2`
- Elite set `E` contains the best samples

### 6. Probability update

Update each stock probability using elite frequency:

```
p_i ← (1−α) p_i + α · (1/|E|) ∑_{y ∈ E} y_i
```

Where:

- `y_i` is 1 if stock `i` is in elite sample `y`
- `α` is a smoothing factor (e.g. `0.7` or `0.9`)

If you want a more aggressive update, use `α = 1`.

### 7. Repeat

Continue sampling, scoring, and updating until one of these occurs:

- probabilities `p` stabilize
- the best subset stops improving
- a fixed number of iterations is reached


## Why cross-entropy works here

- The method searches entire 12-stock combinations, not incremental picks
- It uses actual portfolio Sharpe as the score
- It learns which stocks appear in the best subsets
- It can recover from misleading early stock rankings

This is especially useful because stock interactions matter through `Σ`.


## Mathematical rationale

The cross-entropy update is equivalent to minimizing the Kullback-Leibler divergence between the current sampling distribution and an empirical distribution concentrated on elite samples.

Let `q(y; p)` be the distribution over subsets induced by probabilities `p`.

The update aims to make `q(y; p)` closer to the elite sample distribution.

This is the core principle of cross-entropy optimization.


## Pseudocode

```python
initialize p_i = 12 / |M| for all i in M
A = invest_amt / 12
best_y = None
best_score = -inf

for iteration in range(T):
    samples = []

    for k in range(K):
        y_k = sample_12_stocks(M, p)
        score_k = sharpe_score(y_k, X_bar, A, R, Sigma)
        samples.append((y_k, score_k))
        if score_k > best_score:
            best_score = score_k
            best_y = y_k

    elites = select_top_fraction(samples, rho)

    for i in M:
        p_i = (1 - alpha) * p_i + alpha * (count(i in elites) / len(elites))

    normalize_or_clip(p)

return best_y, best_score
```

### Sharpe score function

```python
def sharpe_score(y, X_bar, A, R, Sigma):
    X = X_bar.copy()
    for i in M:
        X[i] += A * y[i]

    numerator = R.dot(X)
    denominator = np.sqrt(X.T @ Sigma @ X)
    return numerator / denominator
```

### Sampling subsets

Use weighted random sampling without replacement for 12 stocks.

```python
def sample_12_stocks(M, p):
    return weighted_sample_without_replacement(M, p, 12)
```


## Practical configuration

- `K`: number of samples per iteration (e.g. 100–500)
- `ρ`: elite fraction (e.g. `0.1` or `0.2`)
- `α`: learning rate / smoothing (e.g. `0.7`–`0.95`)
- `T`: number of iterations (e.g. 20–50)

For a 20–60 stock universe, this is usually affordable.


## How to explain it to others

1. We treat stock selection as a probabilistic search instead of a fixed ranking.
2. We sample many full 12-stock portfolios from a probability model.
3. We score those portfolios by the actual Sharpe of the combined portfolio.
4. We keep the best portfolios and update the model to favor stocks that appear there.
5. Over time, the model concentrates on the best selection.

This is a global subset search that captures interactions among stocks via covariance.


## Comparison to greedy

| Approach | What it searches | Strength | Weakness |
|---|---|---|---|
| Greedy | one stock at a time | fast | can miss good joint combinations |
| Swap/local search | neighbors of one candidate set | fixes some mistakes | still local |
| Cross-entropy | many whole 12-stock portfolios | global and adaptive | more compute, but still practical |


## Notes for your strategy

- Use your existing `compute_portfolio_sharpe()` objective for scoring
- Use `M` and `X̄` exactly as in your code
- Keep `A = invest_amt / 12`
- Cross-entropy is a natural fit because the problem is binary subset selection, not a simple linear ranking


## Suggested follow-up

If you want, I can also provide a ready-to-paste implementation outline in Python for this specific script, including functions such as:

- `sample_subset(M, p, k=12)`
- `evaluate_subset(y, current_portfolio_values, mean_returns, cov_matrix)`
- `update_probabilities(p, elites, alpha)`
- `cross_entropy_selection(M, current_portfolio_values, invest_amt, mean_returns, cov_matrix)`

## Manager Summary

- We are solving a portfolio selection problem, not a single-stock ranking problem.
- The goal is to choose 12 stocks from the candidate universe `M` to maximize the combined portfolio Sharpe ratio.
- The objective uses the full portfolio return and covariance structure, so stock interactions are important.
- Cross-entropy is a search method that samples many whole 12-stock portfolios and learns which combinations perform best.
- It is more robust than greedy selection because it evaluates complete portfolios and adapts toward high-quality subsets.

## One-slide explanation

- Input: Current portfolio `X̄`, new universe `M`, equal investment per stock, expected returns `R`, covariance `Σ`.
- Output: Best 12-stock subset that maximizes Sharpe of `X̄ + A y`.
- Algorithm:
  1. Start with a soft probability score for each stock.
  2. Sample many candidate 12-stock portfolios.
  3. Score each by portfolio Sharpe.
  4. Keep top-performing samples.
  5. Update stock probabilities toward those elites.
  6. Repeat until the selection stabilizes.

This makes the optimization data-driven and avoids over-committing to early greedy picks.
