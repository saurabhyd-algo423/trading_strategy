# ======================================================================
# ROLLING 12-MONTH HOLD & RECYCLE STRATEGY
# FULL SINGLE-FILE ENGINE — DAILY MTM + METRICS + CHARTS
# ======================================================================

import json
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

# ======================================================================
# CONFIG
# ======================================================================

STOCKS_CSV = "data/prices.xlsx"
FILTER_JSON = "data/value_original2014-2026.json"
FILTER_KEY = "Filter 3"

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
# ======================================================================
# OUTPUT PATHS
# ======================================================================

RES = Path("output/Saurabh_test/value")
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

# ======================================================================
# LOADERS
# ======================================================================

def load_prices():
    df = pd.read_excel(STOCKS_CSV, engine="openpyxl")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()

def load_filter():
    with open(FILTER_JSON, "r") as f:
        return json.load(f)

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
            # if not is_ticker_in_loss(lots, current_price):
            #     continue
            # loss_count += 1
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
# MONTHLY TRACKER
# ======================================================================

def tracker(prices, filter_dict, months):
    cal = get_nse_calendar()
    pe_data, pb_data, ps_data = load_ranking_data()

    holdings = {}
    monthly_records = []
    holdings_rows = []
    ticker_counts = []
    top_weight_rows = []
    liquidation_log = []  # New list to store liquidation details

    for i, month in enumerate(months):
        invest_day = first_trading_day(cal, month)
        if invest_day is None:
            continue

        sell_cash = 0.0
        max_liq_value = MAX_LIQUIDATION_VALUE * monthly_records[-1]["Portfolio_Value"] if monthly_records else 0.0
        bank_balance = max(
            INITIAL_CAPITAL - min((i + 1) * MONTHLY_INVEST, INITIAL_CAPITAL), 0
        )

        # -------- SELL EXPIRED LOTS (12+ months) --------
        # for t in list(holdings.keys()):
        #     new_lots = []
        #     for lot in holdings[t]:
        #         if (invest_day - lot["buy_date"]).days >= HOLD_DAYS:
        #             p = get_price(prices, t, invest_day)
        #             if p:
        #                 sell_cash += lot["qty"] * p
        #         else:
        #             new_lots.append(lot)
        #     if new_lots:
        #         holdings[t] = new_lots
        #     else:
        #         holdings.pop(t)

        # -------- LIQUIDATE LOSS-MAKING STOCKS NOT IN FILTER 1 --------
        if bank_balance == 0 or i >= 12:
            # Analyze holdings for comprehensive logging
            total_holdings = len(holdings)
            loss_stocks = []
            protected_stocks = []
            eligible_stocks = []
            
            for ticker, lots in holdings.items():
                current_price = get_price(prices, ticker, invest_day)
                if is_ticker_in_loss(lots, current_price):
                    loss_stocks.append(ticker)
                    # Check if protected by any filter
                    protected = False
                    for filter_name in ["Momentum", "ROE_ROCE", "Filter_2_50%"]:
                        if is_in_filter(ticker, month, filter_dict, filter_name):
                            protected_stocks.append((ticker, filter_name))
                            protected = True
                            break
                    if not protected:
                        eligible_stocks.append(ticker)
            
            # Initialize monthly liquidation record
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
                holdings, prices, month, filter_dict, invest_day, max_liq_value,
                pe_data, pb_data, ps_data, filter_type=1
            )
            sell_cash += liquidation_cash_filter1
            monthly_liq["Filter1_Liquidated"] = round(liquidation_cash_filter1, 2)
            if liquidation_cash_filter1 == 0:
                if info1["filter_excluded"] > 0:
                    monthly_liq["Filter1_Reason"] = f"{info1['filter_excluded']} loss-making stocks protected by Momentum filter"
                elif len(loss_stocks) == 0:
                    monthly_liq["Filter1_Reason"] = "No loss-making stocks in portfolio"
                else:
                    monthly_liq["Filter1_Reason"] = "No eligible stocks found after ranking"
            
            if sell_cash < max_liq_value:
                liquidation_cash_filter2, info2 = execute_liquidation(
                    holdings, prices, month, filter_dict, invest_day, max_liq_value - sell_cash,
                    pe_data, pb_data, ps_data, filter_type=2
                )
                sell_cash += liquidation_cash_filter2
                monthly_liq["Filter2_Liquidated"] = round(liquidation_cash_filter2, 2)
                if liquidation_cash_filter2 == 0:
                    if info2["filter_excluded"] > 0:
                        monthly_liq["Filter2_Reason"] = f"{info2['filter_excluded']} loss-making stocks protected by ROE_ROCE filter"
                    elif len(loss_stocks) == 0:
                        monthly_liq["Filter2_Reason"] = "No loss-making stocks in portfolio"
                    else:
                        monthly_liq["Filter2_Reason"] = "No eligible stocks found after ranking"
                
                if sell_cash < max_liq_value:
                    liquidation_cash_filter3, info3 = execute_liquidation(
                        holdings, prices, month, filter_dict, invest_day, max_liq_value - sell_cash,
                        pe_data, pb_data, ps_data, filter_type=3
                    )
                    sell_cash += liquidation_cash_filter3
                    monthly_liq["Filter3_Liquidated"] = round(liquidation_cash_filter3, 2)
                    if liquidation_cash_filter3 == 0:
                        if info3["filter_excluded"] > 0:
                            monthly_liq["Filter3_Reason"] = f"{info3['filter_excluded']} loss-making stocks protected by Filter_2_50% filter"
                        elif len(loss_stocks) == 0:
                            monthly_liq["Filter3_Reason"] = "No loss-making stocks in portfolio"
                        else:
                            monthly_liq["Filter3_Reason"] = "No eligible stocks found after ranking"
                    
                    if sell_cash < max_liq_value:
                        liquidation_cash_all, info_all = execute_liquidation(
                            holdings, prices, month, filter_dict, invest_day, max_liq_value - sell_cash,
                            pe_data, pb_data, ps_data, filter_type=None
                        )
                        sell_cash += liquidation_cash_all
                        monthly_liq["Final_Liquidated"] = round(liquidation_cash_all, 2)
                        if liquidation_cash_all == 0:
                            reasons = []
                            if info_all["holding_period_short"] > 0:
                                reasons.append(f"{info_all['holding_period_short']} stocks <1yr holding")
                            if info_all["no_fundamentals"] > 0:
                                reasons.append(f"{info_all['no_fundamentals']} stocks no fundamentals data")
                            if len(loss_stocks) == 0:
                                reasons.append("no loss-making stocks")
                            if reasons:
                                monthly_liq["Final_Reason"] = ", ".join(reasons)
                            else:
                                monthly_liq["Final_Reason"] = "No eligible stocks found"
                    
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
                        if liquidation_cash_pm == 0:
                            monthly_liq["Profit_Margin_Reason"] = "No holdings available or target already met"
            
            monthly_liq["Total_Liquidated"] = round(sell_cash, 2)
            monthly_liq["Achievement_Percent"] = round((sell_cash / max_liq_value) * 100, 1) if max_liq_value > 0 else 0.0
            liquidation_log.append(monthly_liq)

        # -------- INVEST --------
        invest_amt = MONTHLY_INVEST if i < 12 else sell_cash

        selected = [
            t for t in filter_dict.get(month, {}).get(FILTER_KEY, [])
            if t in prices.columns
        ]

        if selected and invest_amt > 0:
            per_stock = invest_amt / len(selected)
            for t in selected:
                p = get_price(prices, t, invest_day)
                if p:
                    holdings.setdefault(t, []).append({
                        "buy_date": invest_day,
                        "qty": per_stock / p,
                        "avg_cost": p  # Track average cost for loss calculation
                    })

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
            top_weight_rows.append({
                "Month": month,
                "Top_Ticker": top_ticker,
                "Top_Amount": round(top_amt, 2),
                "Top_Weight_Percent": round((top_amt / portfolio_value) * 100, 2)
            })

        # bank_balance = max(
        #     INITIAL_CAPITAL - min((i + 1) * MONTHLY_INVEST, INITIAL_CAPITAL), 0
        # )

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
            "Selected_Tickers": len(selected)
        })

    return monthly_records, holdings_rows, ticker_counts, top_weight_rows, liquidation_log

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

# ======================================================================
# PERFORMANCE METRICS — FULL VERSION (PRINTS EVERYTHING)
# ======================================================================
def compute_yearly_metrics(df):
    df = df.copy()

    # Ensure proper types
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    results = []

    for year, grp in df.groupby("year"):
        grp = grp.sort_values("Date")

        # --- TOTAL RETURN ---
        start_val = grp["Portfolio_Value"].iloc[0]
        end_val = grp["Portfolio_Value"].iloc[-1]
        total_return = end_val / start_val - 1

        # --- ANNUALIZED RETURN ---
        days = len(grp)
        ann_return = (1 + total_return) ** (252 / days) - 1

        # --- USE LOG RETURNS ---
        log_ret = grp["LogRet"].dropna()

        # Convert log return → normal return (important)
        simple_ret = np.exp(log_ret) - 1

        # --- DOWNSIDE VOL ---
        downside = simple_ret[simple_ret < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0

        # --- SHARPE ---
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

    # ------------------------------------------------------------------
    # Time & returns
    # ------------------------------------------------------------------
    start_date = daily_df["Date"].iloc[0]
    end_date = daily_df["Date"].iloc[-1]
    years = (end_date - start_date).days / 365.25

    daily_df["LogRet"] = np.log(daily_df["Equity"] / daily_df["Equity"].shift(1))
    lr = daily_df["LogRet"].dropna()
    simple_ret = np.exp(daily_df["LogRet"]) - 1
    daily_df["year"] = daily_df["Date"].dt.year
    yearly_df = compute_yearly_metrics(daily_df)

    # Save
    yearly_df.to_csv(RES / "yearly_performance.csv", index=False)
    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------
    final_equity = daily_df["Equity"].iloc[-1]

    cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    volatility = lr.std() * np.sqrt(252) * 100

    downside = lr.copy()
    downside[downside > 0] = 0
    downside_vol = downside.std() * np.sqrt(252) * 100

    roll_max = daily_df["Equity"].cummax()
    drawdown = daily_df["Equity"] / roll_max - 1
    max_drawdown = abs(drawdown.min()) * 100
    
    # Find max drawdown period details
    max_dd_idx = drawdown.idxmin()
    max_dd_date = daily_df.loc[max_dd_idx, "Date"]
    max_dd_value = daily_df.loc[max_dd_idx, "Equity"]
    
    # Find the start of max drawdown (peak before trough)
    peak_value = roll_max.iloc[max_dd_idx]
    peak_idx = daily_df[daily_df["Equity"] == peak_value].index[0] if (daily_df["Equity"] == peak_value).any() else max_dd_idx
    dd_start_date = daily_df.loc[peak_idx, "Date"]
    
    # Find recovery date (when equity returns to peak level after drawdown)
    recovery_date = None
    for idx in range(max_dd_idx + 1, len(daily_df)):
        if daily_df.loc[idx, "Equity"] >= peak_value:
            recovery_date = daily_df.loc[idx, "Date"]
            break
    
    # Calculate max drawdown period
    if recovery_date:
        dd_period = (recovery_date - dd_start_date).days
    else:
        dd_period = (daily_df["Date"].iloc[-1] - dd_start_date).days
    profits = simple_ret[simple_ret > 0]
    losses = simple_ret[simple_ret < 0]
    accuracy = len(profits) / len(simple_ret) if len(simple_ret) > 0 else 0
    avg_profit = profits.mean() if len(profits) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    # ------------------------------------------------------------------
    # Ratios
    # ------------------------------------------------------------------
    sharpe = (cagr / 100) / (volatility / 100) if volatility > 0 else float("inf")
    sortino = (cagr / 100) / (downside_vol / 100) if downside_vol > 0 else float("inf")
    calmar = (cagr / 100) / (max_drawdown / 100) if max_drawdown > 0 else float("inf")
    gain_loss_ratio = (avg_profit / abs(avg_loss)) if avg_loss != 0 else 0
    # ------------------------------------------------------------------
    # Output dict
    # ------------------------------------------------------------------
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
        "accuracy": accuracy,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "gain_loss_ratio": gain_loss_ratio,
        "Trading Days": int(len(daily_df)),
        "Years Simulated": round(years, 2),
        "Start Date": str(start_date.date()),
        "End Date": str(end_date.date()),
    }

    # ------------------------------------------------------------------
    # PRINT SUMMARY (LIKE OLD VERSION)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)

    for k, v in metrics.items():
        print(f"{k:35}: {v}")

    print("=" * 60 + "\n")

    return metrics


# ======================================================================
# CHARTS + EXTRA OUTPUTS
# ======================================================================

# def generate_charts(daily_df, ticker_counts):
#     plt.figure(figsize=(16, 6))
#     plt.plot(pd.to_datetime(daily_df["Date"]), daily_df["Equity"])
#     plt.title("Daily Portfolio Equity (Portfolio + Bank)")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(OUT_DAILY_CHART, dpi=300)
#     plt.close()

#     dfc = pd.DataFrame(ticker_counts)
#     plt.figure(figsize=(16, 6))
#     plt.plot(dfc["Holdings_Count"], label="Holdings")
#     plt.plot(dfc["Selected_Tickers"], label="Selected")

#     if len(dfc) > 12:
#         df_after = dfc.iloc[12:]
#         idx = df_after["Holdings_Count"].idxmin()

#         plt.scatter(idx, dfc.loc[idx, "Holdings_Count"], s=100)
#         plt.annotate(
#             f"Lowest: {dfc.loc[idx, 'Holdings_Count']}",
#             (idx, dfc.loc[idx, "Holdings_Count"]),
#             xytext=(0, 10),
#             textcoords="offset points",
#             ha="center"
#         )

#         with open(OUT_LOWEST_AFTER_12M, "w") as f:
#             json.dump(dfc.loc[idx].to_dict(), f, indent=2)

#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(OUT_TICKER_CHART, dpi=300)
#     plt.close()

# ======================================================================
# MAIN
# ======================================================================

def main():
    prices = load_prices()
    filters = load_filter()
    months = month_iter(START_MONTH, END_DATE)

    records, hold_rows, ticker_counts, top_weight_rows, liquidation_log = tracker(
        prices, filters, months
    )

    pd.DataFrame(records).to_csv(OUT_TS, index=False)
    pd.DataFrame(hold_rows).to_csv(OUT_HOLD, index=False)
    pd.DataFrame(ticker_counts).to_csv(OUT_TICKER_COUNTS, index=False)
    pd.DataFrame(top_weight_rows).to_csv(OUT_TOP_MONTHLY, index=False)
    pd.DataFrame(liquidation_log).to_csv(OUT_LIQUIDATION_LOG, index=False)

    with open(OUT_JSON, "w") as f:
        json.dump(records, f, indent=2)

    daily_df = daily_tracking(prices, records)
    daily_df.to_csv(OUT_DAILY, index=False)

    metrics = compute_metrics(daily_df)
    with open(OUT_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    # generate_charts(daily_df, ticker_counts)

    print("✅ FULL ROLLING 12M STRATEGY COMPLETE")
    print("📁 Output folder:", RES.resolve())

if __name__ == "__main__":
    main()
