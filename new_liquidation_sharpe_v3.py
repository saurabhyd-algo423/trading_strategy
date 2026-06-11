# ======================================================================
# ROLLING 12-MONTH HOLD & RECYCLE STRATEGY WITH SHARPE OPTIMIZATION
# MOMENTUM + VALUE (50% FILTERS) + PORTFOLIO STOCKS + SHARPE WEIGHTING
# ======================================================================

import json
import time
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from scipy.optimize import minimize

# ======================================================================
# CONFIG
# ======================================================================

STOCKS_CSV = "data/prices.xlsx"
VALUE_FILTER_JSON = "data/value_original2014-2026.json"
MOMENTUM_FILTER_JSON = "data/momentum_original2014-2026.json"

# START_MONTH = "Jul 2014"
START_MONTH = "Jul 2014"
END_DATE = "2026-04-10"

MONTHLY_INVEST = 1000.0
INITIAL_CAPITAL = 12000.0
HOLD_DAYS = 365

MAX_LIQUIDATION_VALUE = 0.08  # Max proportion of portfolio to liquidate per month

# Ranking data paths
PE_DATA_CSV = "data/all_stocks_EPS_data_20y.csv"
PB_DATA_CSV = "data/all_stocks_BV_data_20y.csv"
PS_DATA_CSV = "data/all_stocks_revenue_data_20y.csv"

# Sharpe optimization parameters
LOOKBACK_YEARS = 5
RISK_FREE_RATE = 0.06  # 6% annual risk-free rate

# Optimization toggle: 'greedy' or 'cross_entropy'
OPTIMIZER = "cross_entropy"

# Cross-Entropy parameters (used when OPTIMIZER == 'cross_entropy')
CE_NUM_SAMPLES = 150
CE_ELITE_FRAC = 0.2
CE_ALPHA = 0.7
CE_MAX_ITER = 20
CE_TOL = 1e-4
CE_RANDOM_STATE = 1

# ======================================================================
# OUTPUT PATHS
# ======================================================================

RES = Path("output/Saurabh_sharpe_cross_entropy_v3/value_momentum_union")
RES.mkdir(parents=True, exist_ok=True)

OUT_TS = RES / "portfolio_timeseries.csv"
OUT_HOLD = RES / "portfolio_holdings.csv"
OUT_DAILY = RES / "daily_portfolio_tracking.csv"
OUT_JSON = RES / "monthly_tracking.json"
OUT_TICKER_COUNTS = RES / "monthly_ticker_counts.csv"
OUT_TOP_MONTHLY = RES / "top_weight_stock_per_month.csv"
OUT_TICKER_CHART = RES / "monthly_ticker_counts_chart.png"
OUT_DAILY_CHART = RES / "daily_portfolio_chart.png"
OUT_METRICS_JSON = RES / "performance_metrics.json"
OUT_LIQUIDATION_LOG = RES / "monthly_liquidation_log.csv"
OUT_INVESTMENT_LOG = RES / "monthly_investment_log.csv"

# ======================================================================
# LOADERS
# ======================================================================

def load_prices():
    df = pd.read_excel(STOCKS_CSV, engine="openpyxl")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()

def load_filter():
    """Load both value and momentum filter files for liquidation logic."""
    with open(VALUE_FILTER_JSON, "r") as f:
        return json.load(f)

def load_momentum_filter():
    """Load momentum filter for investment universe."""
    try:
        with open(MOMENTUM_FILTER_JSON, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def load_ranking_data():
    """Load EPS, BV, and revenue data for ranking multiples."""
    def load_csv(path):
        try:
            df = pd.read_csv(path)
            if "Ticker" in df.columns:
                return df.set_index("Ticker")
        except FileNotFoundError:
            pass
        return pd.DataFrame()

    pe_data = load_csv(PE_DATA_CSV)
    pb_data = load_csv(PB_DATA_CSV)
    ps_data = load_csv(PS_DATA_CSV)
    return pe_data, pb_data, ps_data

def get_fundamental_value(df, ticker, column):
    if df.empty or ticker not in df.index or column not in df.columns:
        return None
    value = df.at[ticker, column]
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def get_fundamental_columns(current_date, year):
    suffix = f"Mar {year % 100:02d}"
    return f"EPS_{suffix}", f"BV_{suffix}", f"Revenue_{suffix}"


def get_ranking_multiples(ticker, current_price, current_date, pe_data, pb_data, ps_data):
    """Get PE/PB/PS multiples using fallback years if the current year data are missing."""
    start_year = current_date.year - 2 if current_date.month <= 6 else current_date.year - 1
    eps = bv = revenue = None
    year = start_year
    attempts = 0

    while attempts < 3:
        eps_col, bv_col, revenue_col = get_fundamental_columns(current_date, year)
        eps = get_fundamental_value(pe_data, ticker, eps_col)
        bv = get_fundamental_value(pb_data, ticker, bv_col)
        revenue = get_fundamental_value(ps_data, ticker, revenue_col)

        if any(v is not None and v != 0 for v in [eps, bv, revenue]):
            break

        year -= 1
        attempts += 1

    multiples = {
        "pe": current_price / eps if eps is not None and eps > 0 else None,
        "pb": current_price / bv if bv is not None and bv > 0 else None,
        "ps": current_price / revenue if revenue is not None and revenue > 0 else None,
    }
    return multiples


def get_ranking_score(ticker, current_price, current_date, pe_data, pb_data, ps_data):
    """Return a simple average of available PE/PB/PS multiples."""
    multiples = get_ranking_multiples(ticker, current_price, current_date, pe_data, pb_data, ps_data)
    values = [v for v in multiples.values() if v is not None]
    return sum(values) / len(values) if values else 0.0


def ticker_avg_buy_cost(lots):
    total_qty = sum(lot.get("qty", 0) for lot in lots)
    if total_qty <= 0:
        return None
    total_cost = sum(lot.get("qty", 0) * lot.get("avg_cost", 0) for lot in lots)
    return total_cost / total_qty if total_qty > 0 else None

def is_ticker_in_loss(lots, current_price):
    """Check once if ticker is in loss based on average buy cost."""
    if not lots or current_price is None:
        return False
    avg_cost = ticker_avg_buy_cost(lots)
    if avg_cost is None:
        return False
    return current_price < avg_cost

def is_in_filter(ticker, current_month, filter_dict, filter_name):
    """Check if ticker is in specified filter."""
    filter_tickers = filter_dict.get(current_month, {}).get(filter_name, [])
    return ticker in filter_tickers

def should_liquidate_ticker(ticker, lots, current_price, current_month, filter_dict, filter_name):
    """Determine whether a ticker should be liquidated."""
    # Check in loss only once
    if not is_ticker_in_loss(lots, current_price):
        return False
    # Check if NOT in filter
    return not is_in_filter(ticker, current_month, filter_dict, filter_name)

def execute_liquidation(holdings, prices, current_month, filter_dict, current_date,
                        max_liq_value, pe_data, pb_data, ps_data, filter_type=None):
    """Liquidate loss-making tickers until max liquidation value is reached."""
    liquidation_cash = 0.0
    candidates = []
    
    # Map filter numbers to filter names; None means no filter exclusion
    filter_names = {1: "Momentum", 2: "ROE_ROCE", 3: "Filter_2_50%"}
    filter_name = filter_names.get(filter_type)
    
    # Diagnostic counters
    total_holdings = len(holdings)
    loss_count = 0
    filter_excluded = 0
    holding_period_short = 0
    no_fundamentals = 0

    for ticker, lots in holdings.items():
        current_price = get_price(prices, ticker, current_date)
        
        # Check loss only once per ticker
        if filter_name is not None:
            if not should_liquidate_ticker(ticker, lots, current_price, current_month, filter_dict, filter_name):
                if is_ticker_in_loss(lots, current_price):
                    filter_excluded += 1
                continue
        else:
            # Additional check for final liquidation: holding period > 1 year
            oldest_buy_date = min(lot["buy_date"] for lot in lots)
            if (current_date - oldest_buy_date).days <= 365:
                holding_period_short += 1
                continue

        total_qty = sum(lot["qty"] for lot in lots)
        if total_qty <= 0 or current_price is None:
            continue

        multiples = get_ranking_multiples(ticker, current_price, current_date, pe_data, pb_data, ps_data)
        # Check if we have any fundamentals data
        if not any(multiples.values()):
            no_fundamentals += 1
            continue
            
        candidates.append({
            "ticker": ticker,
            "lots": lots,
            "current_price": current_price,
            "total_qty": total_qty,
            "total_value": total_qty * current_price,
            "multiples": multiples
        })

    # Compute normalized average rank per candidate using available multiples
    def compute_normalized_ranks(candidates, key):
        valid = [(idx, c["multiples"][key]) for idx, c in enumerate(candidates) if c["multiples"].get(key) is not None]
        if not valid:
            return {}
        sorted_valid = sorted(valid, key=lambda x: x[1])
        n = len(sorted_valid)
        return {idx: (rank + 1) / n for rank, (idx, _) in enumerate(sorted_valid)}

    pe_ranks = compute_normalized_ranks(candidates, "pe")
    pb_ranks = compute_normalized_ranks(candidates, "pb")
    ps_ranks = compute_normalized_ranks(candidates, "ps")

    for idx, candidate in enumerate(candidates):
        ranks = [r for r in [pe_ranks.get(idx), pb_ranks.get(idx), ps_ranks.get(idx)] if r is not None]
        candidate["ranking_score"] = sum(ranks) / len(ranks) if ranks else 0.0

    candidates.sort(key=lambda x: (-x["ranking_score"], x["ticker"]))

    for candidate in candidates:
        remaining_target = max_liq_value - liquidation_cash
        if remaining_target <= 0:
            break

        ticker = candidate["ticker"]
        current_price = candidate["current_price"]
        total_value = candidate["total_value"]
        total_qty = candidate["total_qty"]

        if total_value <= remaining_target:
            # Sell the full position
            holdings.pop(ticker, None)
            liquidation_cash += total_value
        else:
            # Sell partially to meet the remaining liquidation target
            qty_to_sell = remaining_target / current_price
            if qty_to_sell <= 0:
                continue

            sell_ratio = qty_to_sell / total_qty
            for lot in list(holdings[ticker]):
                sold_qty = lot["qty"] * sell_ratio
                lot["qty"] -= sold_qty
            holdings[ticker] = [lot for lot in holdings[ticker] if lot["qty"] > 1e-9]
            liquidation_cash += remaining_target
            break

    return liquidation_cash, {
        "total_holdings": total_holdings,
        "candidates_found": len(candidates),
        "loss_count": loss_count,
        "filter_excluded": filter_excluded,
        "holding_period_short": holding_period_short,
        "no_fundamentals": no_fundamentals
    }


def month_iter(start, end):
    s = datetime.strptime(start, "%b %Y")
    e = datetime.strptime(end, "%Y-%m-%d")
    out = []
    while s <= e:
        out.append(s.strftime("%b %Y"))
        s += relativedelta(months=1)
    return out

# ======================================================================
# MARKET HELPERS
# ======================================================================

def get_nse_calendar():
    return mcal.get_calendar("NSE")

def first_trading_day(cal, month):
    d = datetime.strptime(month, "%b %Y")
    sched = cal.schedule(
        start_date=d.replace(day=1),
        end_date=d + relativedelta(months=1, days=-1)
    )
    return None if sched.empty else sched.index[0].date()

def get_price(df, ticker, day):
    if ticker not in df.columns:
        return None
    s = df[ticker].dropna()
    s = s[s.index <= pd.to_datetime(day)]
    return None if s.empty else float(s.iloc[-1])

# ======================================================================
# SHARPE OPTIMIZATION FUNCTIONS
# ======================================================================

def calculate_historical_returns(prices, tickers, invest_day, lookback_years=LOOKBACK_YEARS):
    """
    - Log returns aggregated into non-overlapping 10-day windows
    - Expected returns : mean of 1-year windows, annualised (x 252/10)
    - Covariance       : np.cov of 5-year windows, annualised (x 252/10)
    - Modified cov     : correlations preserved, each stock's std dev replaced
                         by cross-sectional average (mod_cov = corr * avg_std^2)
    Returns: (windows_5y_df, mean_returns, cov_matrix)
    """
    WINDOW = 10
    invest_dt = pd.to_datetime(invest_day)
    start_5y = invest_dt - relativedelta(years=lookback_years)  # covariance
    start_1y = invest_dt - relativedelta(years=1)               # expected returns

    valid = [t for t in tickers if t in prices.columns]
    if not valid:
        return None, None, None

    def make_windows(start):
        px = prices[valid].loc[start:invest_dt].ffill()
        if len(px) < WINDOW + 1:
            return None
        lr = np.log(px / px.shift(1)).dropna()
        lr = lr.dropna(axis=1, how="all")
        if len(lr) < WINDOW:
            return None
        groups = np.arange(len(lr)) // WINDOW
        w = lr.groupby(groups).sum()
        # Drop last incomplete window (fewer than half the window size)
        if lr.groupby(groups).size().iloc[-1] < WINDOW / 2:
            w = w.iloc[:-1]
        w.columns = lr.columns
        return w

    windows_5y = make_windows(start_5y)
    windows_1y = make_windows(start_1y)

    if windows_5y is None or windows_1y is None or len(windows_5y) < 2:
        return None, None, None

    # Align to tickers present in both windows
    common = [t for t in windows_5y.columns if t in windows_1y.columns]
    if not common:
        return None, None, None
    windows_5y = windows_5y[common]
    windows_1y = windows_1y[common]

    ann = 252 / WINDOW  # annualisation factor for 10-day windows

    # Expected returns from 1-year windows
    mean_returns = windows_1y.mean() * ann

    # Covariance from 5-year windows
    raw_cov = np.cov(windows_5y.values, rowvar=False) * ann
    if raw_cov.ndim == 0:
        raw_cov = np.array([[float(raw_cov)]])

    # Modified covariance: replace each stock's std dev with cross-sectional average
    stddevs = np.sqrt(np.diag(raw_cov))
    avg_std = np.sqrt(np.mean(np.diag(raw_cov)))
    with np.errstate(invalid="ignore"):
        corr = raw_cov / np.outer(stddevs, stddevs)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    mod_cov = corr * avg_std * avg_std
    cov_matrix = pd.DataFrame(mod_cov, index=common, columns=common)

    return windows_5y, mean_returns, cov_matrix


def compute_portfolio_sharpe(combined_values, mean_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE):
    """
    Compute the portfolio Sharpe ratio given a dict of {ticker: monetary_value}.

    Corresponds to S(X) = R^T · X̄ / √(X̄^T · Σ · X̄) from the formulation,
    where X̄ are portfolio-value-proportional weights.
    """
    tickers = [
        t for t in combined_values
        if t in mean_returns.index and t in cov_matrix.index and combined_values[t] > 0
    ]
    if not tickers:
        return -np.inf

    total_val = sum(combined_values[t] for t in tickers)
    if total_val <= 0:
        return -np.inf

    weights = np.array([combined_values[t] / total_val for t in tickers])
    ret_vec = mean_returns[tickers].values
    cov_sub = cov_matrix.loc[tickers, tickers].values

    port_return = np.dot(weights, ret_vec)
    port_variance = max(float(np.dot(weights, np.dot(cov_sub, weights))), 1e-12)
    return (port_return - risk_free_rate) / np.sqrt(port_variance)


def greedy_portfolio_sharpe_selection(
    M_tickers, current_portfolio_values, invest_amt,
    mean_returns, cov_matrix, n_stocks=12, risk_free_rate=RISK_FREE_RATE
):
    """
    Select n_stocks from M_tickers (new universe) using greedy portfolio Sharpe
    maximization.

    Formulation (from design doc):
      N  = current portfolio;  X̄ = current portfolio monetary values (fixed)
      M  = new universe = Value_Filter_2_50% ∪ Momentum_Filter_2_50%
      Ȳ  = new investments (only for stocks in M)
      X  = X̄ + Ȳ  (total portfolio after investment)

      max_y  Sharpe(X̄ + Ȳ) = R^T·(X̄+Ȳ) / √((X̄+Ȳ)^T · Σ · (X̄+Ȳ))
      s.t.   Σ_{i∈M} y_i = 12   (select exactly 12 stocks)
             Ȳ_i = A · y_i       (equal allocation A = invest_amt/12)
             y_i ∈ {0, 1}

    Uses a greedy sequential approach: at each step adds the stock from M that
    most improves portfolio Sharpe of the combined position.

    Returns: (allocation dict {ticker: amount}, list of selected tickers)
    """
    # Restrict to candidates with valid historical data
    candidates = [
        t for t in M_tickers
        if t in mean_returns.index and t in cov_matrix.index
    ]

    n = min(n_stocks, len(candidates))
    if n == 0:
        # No data available – equal fallback across all M_tickers
        n_all = max(1, min(n_stocks, len(M_tickers)))
        alloc = invest_amt / n_all
        selected_fallback = M_tickers[:n_all]
        return {t: (alloc if t in selected_fallback else 0.0) for t in M_tickers}, selected_fallback

    alloc_per_stock = invest_amt / n
    selected = []
    remaining = list(candidates)
    # working_values tracks X̄ + Ȳ as we add stocks one by one
    working_values = dict(current_portfolio_values)

    for _ in range(n):
        if not remaining:
            break

        best_ticker = None
        best_sharpe = -np.inf

        for t in remaining:
            test_values = dict(working_values)
            test_values[t] = test_values.get(t, 0.0) + alloc_per_stock
            s = compute_portfolio_sharpe(test_values, mean_returns, cov_matrix, risk_free_rate)
            if s > best_sharpe:
                best_sharpe = s
                best_ticker = t

        if best_ticker is None:
            break

        selected.append(best_ticker)
        remaining.remove(best_ticker)
        working_values[best_ticker] = working_values.get(best_ticker, 0.0) + alloc_per_stock

    allocation = {t: (alloc_per_stock if t in selected else 0.0) for t in M_tickers}
    return allocation, selected


def cross_entropy_portfolio_selection(
    M_tickers, current_portfolio_values, invest_amt,
    mean_returns, cov_matrix,
    n_stocks=12, num_samples=500, elite_frac=0.1, alpha=0.7,
    max_iter=50, tol=1e-4, no_improve_patience=5,
    risk_free_rate=RISK_FREE_RATE, random_state=None
):
    """
    Cross-Entropy (CE) selector for choosing `n_stocks` from `M_tickers`.

    - Maintain sampling probabilities p_i over candidate tickers.
    - At each iteration sample `num_samples` subsets (without replacement)
      according to p, score by Sharpe (using `compute_portfolio_sharpe`).
    - Keep top `elite_frac` fraction, compute frequency of tickers in elites,
      and update p <- alpha*p + (1-alpha)*freq.
    - Stop when p converges or after `max_iter` iterations.

    Returns: (allocation_dict {ticker: amount}, best_subset_list, iterations_used)
    """
    candidates = [t for t in M_tickers if t in mean_returns.index and t in cov_matrix.index]
    n = min(n_stocks, len(candidates))
    if n == 0:
        # fallback equal allocation across M_tickers
        n_all = max(1, min(n_stocks, len(M_tickers)))
        alloc = invest_amt / n_all
        selected_fallback = M_tickers[:n_all]
        return {t: (alloc if t in selected_fallback else 0.0) for t in M_tickers}, selected_fallback, 0

    if random_state is not None:
        np.random.seed(random_state)

    # initialize probabilities uniformly
    p = np.ones(len(candidates)) / len(candidates)
    alloc_per_stock = invest_amt / n
    ticker_to_idx = {t: idx for idx, t in enumerate(candidates)}

    best_subset = None
    best_score = -np.inf
    no_improve_count = 0

    for it in range(max_iter):
        prev_best_score = best_score
        samples = []
        scores = []
        iteration_best_score = -np.inf

        for _ in range(num_samples):
            if len(candidates) <= n:
                subset = list(candidates)
            else:
                p_norm = p / p.sum()
                try:
                    subset = list(np.random.choice(candidates, size=n, replace=False, p=p_norm))
                except ValueError:
                    # numerical issues: fall back to uniform sampling
                    subset = list(np.random.choice(candidates, size=n, replace=False))

            # build combined monetary values
            test_values = dict(current_portfolio_values)
            for t in subset:
                test_values[t] = test_values.get(t, 0.0) + alloc_per_stock

            score = compute_portfolio_sharpe(test_values, mean_returns, cov_matrix, risk_free_rate)
            samples.append(subset)
            scores.append(score)
            iteration_best_score = max(iteration_best_score, score)

            if score > best_score:
                best_score = score
                best_subset = subset

        # determine elite set
        k = max(1, int(np.ceil(elite_frac * len(samples))))
        idxs = np.argsort(scores)[-k:]
        elite_subsets = [samples[i] for i in idxs]

        # compute frequency of each candidate in elite set
        freq = np.zeros(len(candidates), dtype=float)
        for s in elite_subsets:
            for t in s:
                freq[ticker_to_idx[t]] += 1
        freq = freq / k

        p_new = alpha * p + (1 - alpha) * freq
        p_new = np.clip(p_new, 1e-6, 1 - 1e-6)

        # check convergence
        if np.linalg.norm(p_new - p) < tol:
            p = p_new
            break

        p = p_new
        if iteration_best_score > prev_best_score:
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= no_improve_patience:
            break

    # return allocation corresponding to best found subset
    allocation = {t: (alloc_per_stock if t in best_subset else 0.0) for t in M_tickers}
    return allocation, best_subset, it + 1


def get_current_portfolio_values(holdings, prices, invest_day):
    """Return {ticker: current_monetary_value} for the current portfolio (X̄)."""
    portfolio_values = {}
    for ticker, lots in holdings.items():
        qty = sum(lot["qty"] for lot in lots)
        p = get_price(prices, ticker, invest_day)
        if p and qty > 0:
            portfolio_values[ticker] = qty * p
    return portfolio_values


def get_new_universe(current_month, value_filter_dict, momentum_filter_dict, prices, portfolio_stocks=None):
    """
    Return M = Value_Filter_2_50% ∪ Momentum_Filter_2_50% ∪ current portfolio stocks,
    restricted to tickers available in the price data.

    All three sources are selection candidates for the 12-stock Sharpe optimisation.
    Current portfolio stocks (X̄) are still tracked separately so their existing
    monetary values are included when computing the combined portfolio Sharpe.
    """
    value_50 = set(value_filter_dict.get(current_month, {}).get("Filter_2_50%", []))
    momentum_50 = set(momentum_filter_dict.get(current_month, {}).get("Filter_2_50%", []))
    portfolio = set(portfolio_stocks) if portfolio_stocks else set()
    M = value_50 | momentum_50 | portfolio
    return [t for t in M if t in prices.columns]


# ======================================================================
# MONTHLY TRACKER
# ======================================================================

def tracker(prices, value_filter_dict, momentum_filter_dict, months):
    cal = get_nse_calendar()
    pe_data, pb_data, ps_data = load_ranking_data()

    holdings = {}
    monthly_records = []
    holdings_rows = []
    ticker_counts = []
    top_weight_rows = []
    liquidation_log = []
    investment_log = []

    for i, month in enumerate(months):
        invest_day = first_trading_day(cal, month)
        if invest_day is None:
            continue

        sell_cash = 0.0
        max_liq_value = MAX_LIQUIDATION_VALUE * monthly_records[-1]["Portfolio_Value"] if monthly_records else 0.0
        bank_balance = max(
            INITIAL_CAPITAL - min((i + 1) * MONTHLY_INVEST, INITIAL_CAPITAL), 0
        )

        # -------- LIQUIDATE LOSS-MAKING STOCKS NOT IN FILTER 1 --------
        if bank_balance == 0 or i >= 12:
            total_holdings = len(holdings)
            loss_stocks = []
            protected_stocks = []
            eligible_stocks = []
            
            for ticker, lots in holdings.items():
                current_price = get_price(prices, ticker, invest_day)
                if is_ticker_in_loss(lots, current_price):
                    loss_stocks.append(ticker)
                    protected = False
                    for filter_name in ["Momentum", "ROE_ROCE", "Filter_2_50%"]:
                        if is_in_filter(ticker, month, value_filter_dict, filter_name):
                            protected_stocks.append((ticker, filter_name))
                            protected = True
                            break
                    if not protected:
                        eligible_stocks.append(ticker)
            
            monthly_liq = {
                "Month": month,
                "Date": str(invest_day),
                "Max_Liquidation_Target": round(max_liq_value, 2),
                "Total_Holdings": total_holdings,
                "Loss_Stocks_Count": len(loss_stocks),
                "Protected_Stocks_Count": len(protected_stocks),
                "Eligible_Stocks_Count": len(eligible_stocks),
                "Filter1_Liquidated": 0.0,
                "Filter1_Reason": "",
                "Filter2_Liquidated": 0.0,
                "Filter2_Reason": "",
                "Filter3_Liquidated": 0.0,
                "Filter3_Reason": "",
                "Final_Liquidated": 0.0,
                "Final_Reason": "",
                "Profit_Margin_Liquidated": 0.0,
                "Profit_Margin_Reason": "",
                "Total_Liquidated": 0.0,
                "Achievement_Percent": 0.0
            }
            
            liquidation_cash_filter1, info1 = execute_liquidation(
                holdings, prices, month, value_filter_dict, invest_day, max_liq_value,
                pe_data, pb_data, ps_data, filter_type=1
            )
            sell_cash += liquidation_cash_filter1
            monthly_liq["Filter1_Liquidated"] = round(liquidation_cash_filter1, 2)
            
            if sell_cash < max_liq_value:
                liquidation_cash_filter2, info2 = execute_liquidation(
                    holdings, prices, month, value_filter_dict, invest_day, max_liq_value - sell_cash,
                    pe_data, pb_data, ps_data, filter_type=2
                )
                sell_cash += liquidation_cash_filter2
                monthly_liq["Filter2_Liquidated"] = round(liquidation_cash_filter2, 2)
                
                if sell_cash < max_liq_value:
                    liquidation_cash_filter3, info3 = execute_liquidation(
                        holdings, prices, month, value_filter_dict, invest_day, max_liq_value - sell_cash,
                        pe_data, pb_data, ps_data, filter_type=3
                    )
                    sell_cash += liquidation_cash_filter3
                    monthly_liq["Filter3_Liquidated"] = round(liquidation_cash_filter3, 2)
                    
                    if sell_cash < max_liq_value:
                        liquidation_cash_all, info_all = execute_liquidation(
                            holdings, prices, month, value_filter_dict, invest_day, max_liq_value - sell_cash,
                            pe_data, pb_data, ps_data, filter_type=None
                        )
                        sell_cash += liquidation_cash_all
                        monthly_liq["Final_Liquidated"] = round(liquidation_cash_all, 2)
                    
                    if sell_cash < max_liq_value:
                        remaining_target = max_liq_value - sell_cash
                        candidates = []
                        for ticker, lots in holdings.items():
                            current_price = get_price(prices, ticker, invest_day)
                            if current_price is None:
                                continue
                            avg_cost = ticker_avg_buy_cost(lots)
                            if avg_cost is None or avg_cost == 0:
                                continue
                            profit_margin = (current_price / avg_cost) - 1
                            total_qty = sum(lot["qty"] for lot in lots)
                            total_value = total_qty * current_price
                            candidates.append({
                                "ticker": ticker,
                                "lots": lots,
                                "profit_margin": profit_margin,
                                "total_value": total_value,
                                "current_price": current_price,
                                "total_qty": total_qty
                            })
                        candidates.sort(key=lambda x: x["profit_margin"])
                        liquidation_cash_pm = 0.0
                        for candidate in candidates:
                            if remaining_target <= 0:
                                break
                            ticker = candidate["ticker"]
                            total_value = candidate["total_value"]
                            current_price = candidate["current_price"]
                            total_qty = candidate["total_qty"]
                            if total_value <= remaining_target:
                                holdings.pop(ticker, None)
                                liquidation_cash_pm += total_value
                                remaining_target -= total_value
                            else:
                                qty_to_sell = remaining_target / current_price
                                sell_ratio = qty_to_sell / total_qty
                                for lot in list(holdings[ticker]):
                                    sold_qty = lot["qty"] * sell_ratio
                                    lot["qty"] -= sold_qty
                                holdings[ticker] = [lot for lot in holdings[ticker] if lot["qty"] > 1e-9]
                                liquidation_cash_pm += remaining_target
                                remaining_target = 0
                        sell_cash += liquidation_cash_pm
                        monthly_liq["Profit_Margin_Liquidated"] = round(liquidation_cash_pm, 2)
            
            monthly_liq["Total_Liquidated"] = round(sell_cash, 2)
            monthly_liq["Achievement_Percent"] = round((sell_cash / max_liq_value) * 100, 1) if max_liq_value > 0 else 0.0
            liquidation_log.append(monthly_liq)

        # -------- INVEST WITH GREEDY OR CROSS-ENTROPY PORTFOLIO SHARPE OPTIMIZATION --------
        # Formulation: max_Y Sharpe(X̄ + Ȳ) s.t. Σ y_i = 12, Ȳ_i = A·y_i, y_i ∈ {0,1}
        invest_amt = MONTHLY_INVEST if i < 12 else sell_cash
        M_tickers = []  # initialise for ticker_counts below
        opt_label = "No_Investment"

        progress_pct = (i + 1) / len(months) * 100
        print(f"[{i + 1}/{len(months)}] {month} - progress: {progress_pct:.1f}% | invest_amt=₹{invest_amt:.2f}")

        if invest_amt > 0:
            # M = Value_Filter_2_50% ∪ Momentum_Filter_2_50% ∪ current portfolio
            value_50_stocks = set(value_filter_dict.get(month, {}).get("Filter_2_50%", []))
            momentum_50_stocks = set(momentum_filter_dict.get(month, {}).get("Filter_2_50%", []))

            # X̄ = current portfolio monetary values (fixed component of X = X̄ + Ȳ)
            current_portfolio_values = get_current_portfolio_values(holdings, prices, invest_day)

            M_tickers = get_new_universe(
                month, value_filter_dict, momentum_filter_dict, prices,
                portfolio_stocks=list(holdings.keys())
            )

            if M_tickers:
                # M already contains portfolio stocks, so it is the full universe
                _, mean_returns, cov_matrix = calculate_historical_returns(
                    prices, M_tickers, invest_day, lookback_years=LOOKBACK_YEARS
                )

                if mean_returns is not None:
                    ce_runtime = 0.0
                    if OPTIMIZER == "cross_entropy":
                        print(f"    Optimizer: CROSS_ENTROPY | universe={len(M_tickers)} | samples={CE_NUM_SAMPLES} | max_iter={CE_MAX_ITER}")
                        start_time = time.perf_counter()
                        allocations, _, ce_iterations = cross_entropy_portfolio_selection(
                            M_tickers, current_portfolio_values, invest_amt,
                            mean_returns, cov_matrix,
                            n_stocks=12,
                            num_samples=CE_NUM_SAMPLES,
                            elite_frac=CE_ELITE_FRAC,
                            alpha=CE_ALPHA,
                            max_iter=CE_MAX_ITER,
                            tol=CE_TOL,
                            risk_free_rate=RISK_FREE_RATE,
                            random_state=CE_RANDOM_STATE
                        )
                        ce_runtime = time.perf_counter() - start_time
                        opt_label = "CrossEntropy_Portfolio_Sharpe"
                    else:
                        print(f"    Optimizer: GREEDY | universe={len(M_tickers)}")
                        allocations, _ = greedy_portfolio_sharpe_selection(
                            M_tickers, current_portfolio_values, invest_amt,
                            mean_returns, cov_matrix, n_stocks=12,
                            risk_free_rate=RISK_FREE_RATE
                        )
                        ce_iterations = 0
                        opt_label = "Greedy_Portfolio_Sharpe"
                else:
                    # No return data – equal allocation across M
                    n_eq = min(12, len(M_tickers))
                    alloc_eq = invest_amt / n_eq
                    allocations = {t: (alloc_eq if i2 < n_eq else 0.0) for i2, t in enumerate(M_tickers)}
                    ce_iterations = 0
                    opt_label = "Equal_Weight_Fallback"

                # Execute buys
                stocks_invested = 0
                for ticker in M_tickers:
                    alloc_amt = allocations.get(ticker, 0.0)
                    if alloc_amt > 0.01:
                        p = get_price(prices, ticker, invest_day)
                        if p and p > 0:
                            qty = alloc_amt / p
                            holdings.setdefault(ticker, []).append({
                                "buy_date": invest_day,
                                "qty": qty,
                                "avg_cost": p
                            })
                            stocks_invested += 1

                investment_log.append({
                    "Month": month,
                    "Date": str(invest_day),
                    "Investment_Amount": round(invest_amt, 2),
                    "M_Universe_Size": len(M_tickers),
                    "Value_Filter_2_50%_Stocks": len([t for t in M_tickers if t in value_50_stocks]),
                    "Momentum_Filter_2_50%_Stocks": len([t for t in M_tickers if t in momentum_50_stocks]),
                    "Current_Portfolio_Stocks": len(current_portfolio_values),
                    "Stocks_Selected": stocks_invested,
                    "Optimization": opt_label,
                    "Iteration_Count": ce_iterations,
                    "Optimization_Time_Sec": round(ce_runtime if OPTIMIZER == "cross_entropy" else 0.0, 3)
                })
                print(f"   Completed {opt_label}: selected {stocks_invested} stocks in {ce_iterations} iterations, runtime={ce_runtime:.3f}s")

        # ---------------- SNAPSHOT ----------------
        portfolio_value = 0.0
        snapshot = []
        weight_map = {}

        for t, lots in holdings.items():
            qty = sum(l["qty"] for l in lots)
            p = get_price(prices, t, invest_day)
            amt = qty * p if p else 0.0

            portfolio_value += amt
            weight_map[t] = amt

            snapshot.append({"Ticker": t, "Quantity": round(qty, 6)})

            holdings_rows.append({
                "Month": month,
                "Date": str(invest_day),
                "Ticker": t,
                "Quantity": round(qty, 6),
                "Current_Price": round(p or 0, 2),
                "Current_Amount": round(amt, 2)
            })

        # ---------------- TOP WEIGHT STOCK ----------------
        if weight_map:
            top_ticker = max(weight_map, key=weight_map.get)
            top_amt = weight_map[top_ticker]
            if portfolio_value > 0:
                top_weight_rows.append({
                    "Month": month,
                    "Top_Ticker": top_ticker,
                    "Top_Amount": round(top_amt, 2),
                    "Top_Weight_Percent": round((top_amt / portfolio_value) * 100, 2)
                })

        monthly_records.append({
            "Month": month,
            "Date": str(invest_day),
            "Investment": round(invest_amt, 2),
            "Portfolio_Value": round(portfolio_value, 2),
            "Bank": round(bank_balance, 2),
            "Equity": round(portfolio_value + bank_balance, 2),
            "Holdings": snapshot
        })

        ticker_counts.append({
            "Month": month,
            "Holdings_Count": len(snapshot),
            "M_Universe_Size": len(M_tickers),
            "Optimization_Type": opt_label
        })

    return monthly_records, holdings_rows, ticker_counts, top_weight_rows, liquidation_log, investment_log

# ======================================================================
# DAILY MARK-TO-MARKET
# ======================================================================

def daily_tracking(prices, monthly_records):
    cal = get_nse_calendar()
    rows = []

    for i, rec in enumerate(monthly_records):
        start = pd.to_datetime(rec["Date"]).date()
        end = (
            pd.to_datetime(monthly_records[i + 1]["Date"]).date()
            if i < len(monthly_records) - 1
            else pd.to_datetime(END_DATE).date()
        )

        bank = rec["Bank"]

        sched = cal.schedule(start_date=start, end_date=end)
        days = sched.index.date

        for d in days:
            pv = 0.0
            for h in rec["Holdings"]:
                p = get_price(prices, h["Ticker"], d)
                if p:
                    pv += h["Quantity"] * p

            rows.append({
                "Date": d,
                "Portfolio_Value": round(pv, 2),
                "Bank": round(bank, 2),
                "Equity": round(pv + bank, 2)
            })

    return pd.DataFrame(rows).drop_duplicates("Date", keep="last").reset_index(drop=True)

# ======================================================================
# PERFORMANCE METRICS
# ======================================================================

def compute_yearly_metrics(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    results = []

    for year, grp in df.groupby("year"):
        grp = grp.sort_values("Date")

        start_val = grp["Portfolio_Value"].iloc[0]
        end_val = grp["Portfolio_Value"].iloc[-1]
        total_return = end_val / start_val - 1

        days = len(grp)
        ann_return = (1 + total_return) ** (252 / days) - 1

        log_ret = grp["LogRet"].dropna()
        simple_ret = np.exp(log_ret) - 1

        downside = simple_ret[simple_ret < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0

        std = simple_ret.std()
        sharpe = (simple_ret.mean() / std * np.sqrt(252)) if std != 0 else 0

        results.append({
            "year": year,
            "total_return": total_return,
            "annualized_return": ann_return,
            "downside_volatility": downside_vol,
            "sharpe": sharpe,
            "days": days
        })

    return pd.DataFrame(results)


def compute_metrics(daily_df):
    daily_df = daily_df.copy()

    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    daily_df["Equity"] = daily_df["Equity"].astype(float)

    start_date = daily_df["Date"].iloc[0]
    end_date = daily_df["Date"].iloc[-1]
    years = (end_date - start_date).days / 365.25

    daily_df["LogRet"] = np.log(daily_df["Equity"] / daily_df["Equity"].shift(1))
    lr = daily_df["LogRet"].dropna()
    simple_ret = np.exp(daily_df["LogRet"]) - 1
    daily_df["year"] = daily_df["Date"].dt.year
    yearly_df = compute_yearly_metrics(daily_df)

    yearly_df.to_csv(RES / "yearly_performance.csv", index=False)

    final_equity = daily_df["Equity"].iloc[-1]

    cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    volatility = lr.std() * np.sqrt(252) * 100

    downside = lr.copy()
    downside[downside > 0] = 0
    downside_vol = downside.std() * np.sqrt(252) * 100

    roll_max = daily_df["Equity"].cummax()
    drawdown = daily_df["Equity"] / roll_max - 1
    max_drawdown = abs(drawdown.min()) * 100
    
    max_dd_idx = drawdown.idxmin()
    max_dd_date = daily_df.loc[max_dd_idx, "Date"]
    max_dd_value = daily_df.loc[max_dd_idx, "Equity"]
    
    peak_value = roll_max.iloc[max_dd_idx]
    peak_idx = daily_df[daily_df["Equity"] == peak_value].index[0] if (daily_df["Equity"] == peak_value).any() else max_dd_idx
    dd_start_date = daily_df.loc[peak_idx, "Date"]
    
    recovery_date = None
    for idx in range(max_dd_idx + 1, len(daily_df)):
        if daily_df.loc[idx, "Equity"] >= peak_value:
            recovery_date = daily_df.loc[idx, "Date"]
            break
    
    if recovery_date:
        dd_period = (recovery_date - dd_start_date).days
    else:
        dd_period = (daily_df["Date"].iloc[-1] - dd_start_date).days
        
    profits = simple_ret[simple_ret > 0]
    losses = simple_ret[simple_ret < 0]
    accuracy = len(profits) / len(simple_ret) if len(simple_ret) > 0 else 0
    avg_profit = profits.mean() if len(profits) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    sharpe = (cagr / 100) / (volatility / 100) if volatility > 0 else float("inf")
    sortino = (cagr / 100) / (downside_vol / 100) if downside_vol > 0 else float("inf")
    calmar = (cagr / 100) / (max_drawdown / 100) if max_drawdown > 0 else float("inf")
    gain_loss_ratio = (avg_profit / abs(avg_loss)) if avg_loss != 0 else 0

    metrics = {
        "Initial Capital": f"₹{INITIAL_CAPITAL:,.2f}",
        "Final Capital": f"₹{final_equity:,.2f}",
        "Total Return (%)": round(((final_equity / INITIAL_CAPITAL) - 1) * 100, 2),
        "CAGR / Annualized Return (%)": round(cagr, 2),
        "Annualized Volatility (%)": round(volatility, 2),
        "Downside Volatility (%)": round(downside_vol, 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Max Drawdown Start Date": str(dd_start_date.date()),
        "Max Drawdown Trough Date": str(max_dd_date.date()),
        "Max Drawdown Recovery Date": str(recovery_date.date()) if recovery_date else "Not recovered yet",
        "Max Drawdown Period (Days)": dd_period,
        "Calmar Ratio": round(calmar, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Accuracy": round(accuracy, 4),
        "Avg Profit": round(avg_profit, 6),
        "Avg Loss": round(avg_loss, 6),
        "Gain-Loss Ratio": round(gain_loss_ratio, 2),
        "Trading Days": int(len(daily_df)),
        "Years Simulated": round(years, 2),
        "Start Date": str(start_date.date()),
        "End Date": str(end_date.date()),
    }

    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY (SHARPE OPTIMIZED)")
    print("=" * 60)

    for k, v in metrics.items():
        print(f"{k:35}: {v}")

    print("=" * 60 + "\n")

    return metrics

# ======================================================================
# MAIN
# ======================================================================

def main():
    print("\n" + "=" * 70)
    print("GREEDY PORTFOLIO SHARPE OPTIMIZATION")
    print("=" * 70)
    print("Universe : Value_Filter_2_50% union Momentum_Filter_2_50% (M)")
    print("Fixed    : Current portfolio X-bar (not re-selected each month)")
    print("Objective: max Sharpe(X-bar + Y-bar) s.t. sum(y_i)=12, y_i in {0,1}")
    print("=" * 70 + "\n")

    prices = load_prices()
    value_filter_dict = load_filter()
    momentum_filter_dict = load_momentum_filter()
    months = month_iter(START_MONTH, END_DATE)

    records, hold_rows, ticker_counts, top_weight_rows, liquidation_log, investment_log = tracker(
        prices, value_filter_dict, momentum_filter_dict, months
    )

    pd.DataFrame(records).to_csv(OUT_TS, index=False)
    pd.DataFrame(hold_rows).to_csv(OUT_HOLD, index=False)
    pd.DataFrame(ticker_counts).to_csv(OUT_TICKER_COUNTS, index=False)
    pd.DataFrame(top_weight_rows).to_csv(OUT_TOP_MONTHLY, index=False)
    pd.DataFrame(liquidation_log).to_csv(OUT_LIQUIDATION_LOG, index=False)
    pd.DataFrame(investment_log).to_csv(OUT_INVESTMENT_LOG, index=False)

    with open(OUT_JSON, "w") as f:
        json.dump(records, f, indent=2)

    daily_df = daily_tracking(prices, records)
    daily_df.to_csv(OUT_DAILY, index=False)

    metrics = compute_metrics(daily_df)
    with open(OUT_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ MIP SHARPE-OPTIMIZED STRATEGY COMPLETE")
    print("📁 Output folder:", RES.resolve())

if __name__ == "__main__":
    main()
