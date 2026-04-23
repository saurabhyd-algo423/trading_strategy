# new_liquidation_weekly_v4

import pandas as pd
import numpy as np
import json
import os
import math
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

# ----------------- CONFIG -----------------
STOCKS_CSV = "data/stocks-data-analysis-up-to-date.csv"

# Strategy selector
STRATEGY = "value"  # "momentum" | "value"
STRATEGY = "momentum"  # "momentum" | "value"

VARIANT = "original"  # "original" | "final"
VARIANT = "final"  # "original" | "final"

# ----------------- SCENARIO LOGIC (UPDATED) -----------------
SCENARIO = f"{STRATEGY}-{VARIANT}"

SCENARIOS = {
    "value-original": {
        "invest_json": "data\\value.json",
        "good_json": "data\\filtered_outputs\\Value_Filter_2_50%.json",
        "output_dir": "output\\value\\strategy-original-weekly\\value4",
    },
    "value-final": {
        "invest_json": "data\\value_filter4_modified_with_sharpe_10_drop_20_iter.json",
        "good_json": "data\\filtered_outputs\\Value_Filter_2_50%.json",
        "output_dir": "output\\value\\strategy-final-weekly\\value4",
    },
    "momentum-original": {
        "invest_json": "data\\momentum.json",
        "good_json": "data\\good_stocks_combined.json",
        "output_dir": "output\\momentum\\strategy-original-weekly\\momentum4",
    },
    "momentum-final": {
        "invest_json": "data\\momentum_modified_filter4_with_sharpe_10_drop_20_iter.json",
        "good_json": "data\\good_stocks_combined.json",
        "output_dir": "output\\momentum\\strategy-final-weekly\\momentum4",
    },
}

cfg = SCENARIOS[SCENARIO]

INVEST_JSON = cfg["invest_json"]
GOOD_JSON = cfg["good_json"]

# ----------------- FILTER -----------------
FILTER_KEY = "Filter 3"

# ----------------- SIMULATION RANGE -----------------
START_MONTH = "Jul 2014"
END_DATE = "2025-09-01"

# ----------------- BANK SETTINGS -----------------
INITIAL_BANK_BALANCE = 12000.0
MONTHLY_BANK_WITHDRAW = INITIAL_BANK_BALANCE / 52

# ----------------- LIQUIDATION POLICY -----------------
PARTIAL_ALLOWED = True
HOLDING_PERIOD_THRESHOLD_DAYS = 365  # 1 year

# ----------------- RESULTS FOLDER -----------------
RES = Path(cfg["output_dir"])
RES.mkdir(parents=True, exist_ok=True)


# Output files
OUT_TS = RES / "portfolio_timeseries_filter3.csv"
OUT_HOLD = RES / "portfolio_timeseries_filter3_holdings.csv"
OUT_FIN = RES / "final_portfolio_filter3.csv"
OUT_JSON = RES / f"monthly_tracking_{FILTER_KEY.replace(' ', '_')}.json"
OUT_TICKER_COUNTS = RES / "monthly_ticker_counts.csv"
OUT_CHART = RES / "monthly_ticker_counts_chart.png"
OUT_LIQ = RES / "monthly_liquidation_details.csv"
OUT_BANK_TRACKER = RES / "monthly_bank_tracker.csv"
OUT_BANK_JSON = RES / "monthly_bank_tracker.json"


# ----------------- HELPERS -----------------
def load_stocks_df():
    df = pd.read_csv(STOCKS_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def load_filter(path):
    with open(path, "r") as f:
        return json.load(f)


# def month_iter(start, end):
#     s = datetime.strptime(start, "%b %Y")
#     e = datetime.strptime(end, "%Y-%m-%d")
#     months = []
#     while s <= e:
#         months.append(s.strftime("%b %Y"))
#         s += relativedelta(months=1)
#     return months

# from datetime import datetime, timedelta


def week_iter(start_date, end_date):
    """
    Generate a list of ISO week keys between start_date and end_date.
    Format: 'YYYY-Www'  (e.g., '2024-W05')

    Args:
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'

    Returns:
        List[str]: ordered list of week keys
    """

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Move start to Monday of its ISO week
    start -= timedelta(days=start.weekday())

    weeks = []
    seen = set()
    cur = start

    while cur <= end:
        iso_year, iso_week, _ = cur.isocalendar()
        key = f"{iso_year}-W{iso_week:02d}"

        if key not in seen:
            weeks.append(key)
            seen.add(key)

        cur += timedelta(days=7)

    return weeks


def monthend(month):
    d = datetime.strptime(month, "%b %Y")
    return (d + relativedelta(months=1, days=-1)).date()


# ----------------- TRADING DAY HELPERS (pandas_market_calendars) -----------------
def get_nse_calendar():
    """Return the NSE calendar object (pandas_market_calendars)."""
    return mcal.get_calendar("NSE")


# def get_first_trading_day(nse_calendar, month):
#     """
#     Return the first available trading day (date object) in given month (format 'Jul 2014').
#     Returns None if no trading day found.
#     """
#     d = datetime.strptime(month, "%b %Y")
#     start = d.replace(day=1)
#     end = d + relativedelta(months=1, days=-1)
#     try:
#         schedule = nse_calendar.schedule(start_date=start, end_date=end)
#     except Exception:
#         # defensive: if calendar call fails for some month, return None
#         return None
#     if schedule is None or schedule.empty:
#         return None
#     return schedule.index[0].date()


# def get_last_trading_day(nse_calendar, month):
#     """
#     Return the last available trading day (date object) in given month (format 'Jul 2014').
#     Returns None if no trading day found.
#     """
#     d = datetime.strptime(month, "%b %Y")
#     start = d.replace(day=1)
#     end = d + relativedelta(months=1, days=-1)
#     try:
#         schedule = nse_calendar.schedule(start_date=start, end_date=end)
#     except Exception:
#         return None
#     if schedule is None or schedule.empty:
#         return None
#     return schedule.index[-1].date()


def get_first_trading_day_of_week(nse_calendar, week_key):
    """
    Return the first NSE trading day for a given ISO week.
    Week key format: 'YYYY-Www' (e.g. '2024-W05')

    Returns:
        date or None
    """

    try:
        year_str, week_str = week_key.split("-W")
        year = int(year_str)
        week = int(week_str)
    except Exception:
        raise ValueError(f"Invalid week_key format: {week_key}")

    # ISO week → Monday date
    week_start = datetime.fromisocalendar(year, week, 1).date()
    week_end = week_start + timedelta(days=6)

    try:
        schedule = nse_calendar.schedule(
            start_date=week_start,
            end_date=week_end,
        )
    except Exception:
        return None

    if schedule is None or schedule.empty:
        return None

    return schedule.index[0].date()


# ----------------- PRICE HELPERS -----------------
def safe_div(a, b):
    return a / b if b else 0.0


def compute_lot_return_pct(lot):
    """
    Compute percentage return for a sellable lot.
    """
    lot_price = lot["lot_price"]
    current_price = lot["current_price"]

    if lot_price <= 0:
        return float("inf")

    return (current_price / lot_price) - 1.0


def get_price(df, ticker, on_date):
    """Return last available price for ticker on or before on_date, or None if not available."""
    if ticker not in df.columns:
        return None
    ser = df[ticker].dropna()
    sub = ser.index[ser.index <= pd.to_datetime(on_date)]
    return None if len(sub) == 0 else float(ser.loc[sub[-1]])


# ----------------- LOT REMOVAL (FIFO) -----------------
def remove_qty_from_lots(holdings_entry, qty_to_remove, sell_date):
    """
    FIFO removal of quantity from lots. Updates holdings_entry (lots, qty, invested, avg).

    Args:
        holdings_entry (dict): entry for a stock, containing:
            {
                "lots": [{"month": "Jan 2021", "qty": ..., "price": ...}, ...],
                "qty": total quantity held,
                "invested": total invested amount,
                "avg": average buy price
            }
        qty_to_remove (float): quantity to sell/remove
        sell_date (date): date on which this sell occurs (used for holding period calc)

    Returns:
        cost_removed (float): total invested cost of the sold quantity
        holding_periods (list): [(holding_days, qty_removed), ...] for weighted averaging
    """

    qty_remaining = qty_to_remove
    new_lots = []
    cost_removed = 0.0
    holding_periods = []

    for lot in holdings_entry["lots"]:
        if qty_remaining <= 0:
            # no more to sell; keep the remaining lots
            new_lots.append(lot)
            continue

        if "buy_date" in lot:
            buy_date = datetime.strptime(lot["buy_date"], "%Y-%m-%d").date()
        else:
            buy_date = datetime.strptime(lot["month"], "%b %Y").date()

        holding_days = (sell_date - buy_date).days

        if lot["qty"] <= qty_remaining + 1e-12:
            # full lot sold
            qty_removed = lot["qty"]
            qty_remaining -= qty_removed
            cost_removed += qty_removed * lot["price"]

            # record holding period for this lot
            holding_periods.append((holding_days, qty_removed))
            # do not add this lot (fully sold)
        else:
            # partial lot sold
            qty_removed = qty_remaining
            cost_removed += qty_removed * lot["price"]

            holding_periods.append((holding_days, qty_removed))

            # remaining part of this lot stays
            remaining_qty = lot["qty"] - qty_removed
            new_lots.append(
                {
                    "month": lot["month"],
                    "buy_date": lot.get("buy_date"),  # keep same buy date
                    "qty": remaining_qty,
                    "price": lot["price"],
                    "amount": remaining_qty * lot["price"],
                }
            )

            qty_remaining = 0.0

    # update holdings entry
    holdings_entry["lots"] = new_lots
    holdings_entry["qty"] = sum(l["qty"] for l in new_lots)
    holdings_entry["invested"] = sum(l["qty"] * l["price"] for l in new_lots)
    holdings_entry["avg"] = safe_div(holdings_entry["invested"], holdings_entry["qty"])

    return cost_removed, holding_periods


# ----------------- LOT EVALUATION -----------------
def evaluate_sellable_lots(holdings, good_stocks, liq_date, df, price_cache=None):
    """
    Evaluate sellable lots across holdings using FIFO-scan + stop-on-first-non-sellable rule.

    Args:
        holdings (dict): holdings dict (ticker -> {qty, invested, avg, lots:list})
        good_stocks (list): list of tickers allowed (good list)
        liq_date (date): liquidation date to use for price/P&L
        df (DataFrame): price dataframe
        price_cache (dict): optional cache mapping (ticker, date) -> price

    Returns:
        sellable_lots (dict): mapping ticker -> list of sellable lot dicts:
           { "lot_index": int, "qty": float, "lot_price": float, "current_price": float,
             "current_value": float, "holding_days": int, "buy_month": "Jan 2021", "reason": str }
        per_stock_sellable_value (dict): ticker -> total current_value of sellable lots
        total_possible_liquidation (float)
    """

    if price_cache is None:
        price_cache = {}

    sellable_lots = {}
    per_stock_sellable_value = {}
    total_possible_liquidation = 0.0

    for ticker, holding in holdings.items():

        # Only consider BAD stocks (i.e., not in good_stocks list)
        if ticker in good_stocks:
            continue

        lots = holding.get("lots", [])
        if not lots:
            continue

        stock_sellable_list = []
        stock_sellable_total = 0.0
        stop_scan = False

        # Get price for liquidation date
        cache_key = (ticker, str(liq_date))
        if cache_key in price_cache:
            current_price = price_cache[cache_key]
        else:
            current_price = get_price(df, ticker, liq_date)
            price_cache[cache_key] = current_price

        if current_price is None or (
            isinstance(current_price, float) and math.isnan(current_price)
        ):
            continue

        # ---------- FIFO Scan ----------
        for lot_idx, lot in enumerate(lots):

            if stop_scan:
                break

            if "buy_date" in lot:
                buy_date = datetime.strptime(lot["buy_date"], "%Y-%m-%d").date()
                buy_month = lot["month"]
            else:
                buy_month = lot["month"]
                buy_date = datetime.strptime(buy_month, "%b %Y").date()

            holding_days = (liq_date - buy_date).days
            lot_qty = lot["qty"]
            lot_price = lot["price"]
            lot_amount = lot.get("amount", lot_qty * lot_price)

            # P/L check
            is_loss = current_price < lot_price - 1e-12

            # ---------- Sellability Rules ----------
            if is_loss:
                reason = "loss"
                sellable_flag = True
            else:
                # profit or breakeven
                if holding_days > HOLDING_PERIOD_THRESHOLD_DAYS:
                    reason = "profit_age_gt_1yr"
                    sellable_flag = True
                else:
                    reason = "profit_age_lt_1yr_stop"
                    sellable_flag = False
                    stop_scan = True

            # ---------- Save if sellable ----------
            if sellable_flag and lot_qty > 0:
                current_value = lot_qty * current_price

                stock_sellable_list.append(
                    {
                        "lot_index": lot_idx,
                        "qty": lot_qty,
                        "lot_price": lot_price,
                        "amount": lot_amount,
                        "current_price": current_price,
                        "current_value": current_value,
                        "holding_days": holding_days,
                        "buy_date": str(buy_date),
                        "buy_month": buy_month,
                        "reason": reason,
                    }
                )

                stock_sellable_total += current_value

        # ---------- Per-stock aggregation ----------
        if stock_sellable_list:
            sellable_lots[ticker] = stock_sellable_list
            per_stock_sellable_value[ticker] = stock_sellable_total
            total_possible_liquidation += stock_sellable_total

    return sellable_lots, per_stock_sellable_value, total_possible_liquidation


def flatten_and_sort_sellable_lots(sellable_lots):
    """
    Flatten sellable lots across all stocks and sort them globally.

    Sort priority:
      1) holding_days DESC (older first)
      2) lot_return_pct ASC (worse-performing first)
      3) buy_date ASC (older buy date first)
    """

    flat_lots = []

    for ticker, lots in sellable_lots.items():
        for lot in lots:
            lot_return_pct = compute_lot_return_pct(lot)
            flat_lots.append(
                {
                    **lot,
                    "ticker": ticker,
                    "lot_return_pct": lot_return_pct,
                }
            )

    flat_lots.sort(
        key=lambda x: (
            -x["holding_days"],  # older first
            x["lot_return_pct"],  # worse return first
            x["buy_date"],  # older buy date first
        )
    )

    return flat_lots


# ----------------- TRACKER with updated WEEKLY liquidation logic -----------------
def tracker(
    df,
    invest_flt,
    good_flt,
    weeks,
    initial_bank_balance,
    weekly_withdraw,
    filter_key,
):
    """
    Runs the WEEKLY simulation using:
      - invest_flt: JSON dict mapping week -> filters -> list of tickers
      - good_flt: JSON dict mapping week -> list of good tickers
      - Investment AND liquidation on FIRST trading day of the week
      - Liquidation limit = (bank + portfolio) / 52
    """

    holdings = {}
    liq_events_all = []
    recs = []
    ticker_counts = []
    bank_tracker_records = []

    all_holding_periods = []
    per_stock_holding_periods = {}

    all_returns = []
    per_stock_returns = {}

    bank_balance = initial_bank_balance
    tot_invested_from_bank = 0.0
    tot_liquidated = 0.0

    nse = get_nse_calendar()
    price_cache = {}

    liquidation_limit_hits = []
    liquidation_limit_hit_count = 0
    total_excess_liquidation = 0.0

    for i, w in enumerate(weeks):

        trade_day = get_first_trading_day_of_week(nse, w)

        if trade_day is None:
            print(f"[WARN] {w}: No trading day found. Skipping week.")
            continue

        selected = [
            t for t in invest_flt.get(w, {}).get(filter_key, []) if t in df.columns
        ]
        good_stocks = good_flt.get(w, [])

        bank_before = bank_balance

        # ---------- Portfolio valuation BEFORE liquidation ----------
        portfolio_value_before = 0.0
        holding_summaries = {}

        for t, d in holdings.items():
            cache_key = (t, str(trade_day))
            p = price_cache.get(cache_key)
            if p is None:
                p = get_price(df, t, trade_day)
                price_cache[cache_key] = p

            cur_val = d["qty"] * (p or 0.0)
            portfolio_value_before += cur_val

            holding_summaries[t] = {
                "qty": d["qty"],
                "invested": d["invested"],
                "avg": d["avg"],
                "current_price": p,
                "current_value": cur_val,
                "return_pct": (
                    (float(p) / d["avg"] - 1.0) if (d["avg"] and p) else float("inf")
                ),
            }

        bank_plus_portfolio = bank_before + portfolio_value_before
        allowed_liquidation_limit = bank_plus_portfolio / 52.0

        weekly_liq_total = 0.0
        weekly_liq_events = []

        # ---------- Lot-wise sellable evaluation ----------
        sellable_lots, per_stock_sellable_value, total_possible_liquidation = (
            evaluate_sellable_lots(
                holdings,
                good_stocks,
                trade_day,
                df,
                price_cache=price_cache,
            )
        )

        # ---------- Liquidation logic ----------
        if i >= 1 and portfolio_value_before > 0 and sellable_lots:

            if total_possible_liquidation <= allowed_liquidation_limit + 1e-9:

                for t, lots in sellable_lots.items():
                    for lot in lots:
                        qty = lot["qty"]
                        price = lot["current_price"]
                        amount = qty * price

                        cost_removed, periods = remove_qty_from_lots(
                            holdings[t], qty, trade_day
                        )

                        all_holding_periods.extend(periods)
                        per_stock_holding_periods.setdefault(t, []).extend(periods)

                        if cost_removed > 0:
                            avg_sell_price = price
                            total_qty_sold = sum(q for _, q in periods)
                            avg_buy_price = safe_div(cost_removed, total_qty_sold)
                            ret = safe_div(avg_sell_price, avg_buy_price) - 1.0
                            all_returns.extend([(ret, q) for _, q in periods])
                            per_stock_returns.setdefault(t, []).extend(
                                [(ret, q) for _, q in periods]
                            )

                        if holdings.get(t, {}).get("qty", 0.0) == 0:
                            holdings.pop(t, None)

                        weekly_liq_events.append(
                            {
                                "Week": w,
                                "Date": str(trade_day),
                                "Stock": t,
                                "Liquidated_Qty": round(qty, 6),
                                "Liquidated_Amount": round(amount, 2),
                                "Reason": "Lot-sellable (within limit)",
                                "Remaining_Qty": round(
                                    holdings.get(t, {}).get("qty", 0.0), 6
                                ),
                                "Portfolio_Value_After": None,
                            }
                        )
                        weekly_liq_total += amount

            else:
                if not PARTIAL_ALLOWED:
                    excess = total_possible_liquidation - allowed_liquidation_limit
                    if excess > 0:
                        liquidation_limit_hit_count += 1
                        total_excess_liquidation += excess
                        liquidation_limit_hits.append(
                            {
                                "Week": w,
                                "Allowed_Limit": round(allowed_liquidation_limit, 2),
                                "Possible_Liquidation": round(
                                    total_possible_liquidation, 2
                                ),
                                "Excess": round(excess, 2),
                            }
                        )

                    for t, lots in sellable_lots.items():
                        for lot in lots:
                            qty = lot["qty"]
                            price = lot["current_price"]
                            amount = qty * price

                            cost_removed, periods = remove_qty_from_lots(
                                holdings[t], qty, trade_day
                            )

                            all_holding_periods.extend(periods)
                            per_stock_holding_periods.setdefault(t, []).extend(periods)

                            if holdings.get(t, {}).get("qty", 0.0) == 0:
                                holdings.pop(t, None)

                            weekly_liq_events.append(
                                {
                                    "Week": w,
                                    "Date": str(trade_day),
                                    "Stock": t,
                                    "Liquidated_Qty": round(qty, 6),
                                    "Liquidated_Amount": round(amount, 2),
                                    "Reason": "Lot-sellable (over limit)",
                                    "Remaining_Qty": round(
                                        holdings.get(t, {}).get("qty", 0.0), 6
                                    ),
                                    "Portfolio_Value_After": None,
                                }
                            )
                            weekly_liq_total += amount

                else:
                    remaining_to_liquidate = allowed_liquidation_limit
                    sorted_lots = flatten_and_sort_sellable_lots(sellable_lots)

                    for lot in sorted_lots:
                        if remaining_to_liquidate <= 1e-9:
                            break

                        t = lot["ticker"]
                        price = lot["current_price"]
                        max_qty = lot["qty"]
                        max_value = max_qty * price

                        if max_value <= remaining_to_liquidate:
                            qty_to_sell = max_qty
                            amount = max_value
                        else:
                            qty_to_sell = remaining_to_liquidate / price
                            amount = qty_to_sell * price

                        cost_removed, periods = remove_qty_from_lots(
                            holdings[t], qty_to_sell, trade_day
                        )

                        all_holding_periods.extend(periods)
                        per_stock_holding_periods.setdefault(t, []).extend(periods)

                        if holdings.get(t, {}).get("qty", 0.0) == 0:
                            holdings.pop(t, None)

                        weekly_liq_events.append(
                            {
                                "Week": w,
                                "Date": str(trade_day),
                                "Stock": t,
                                "Liquidated_Qty": round(qty_to_sell, 6),
                                "Liquidated_Amount": round(amount, 2),
                                "Reason": "Lot-based partial liquidation",
                                "Remaining_Qty": round(
                                    holdings.get(t, {}).get("qty", 0.0), 6
                                ),
                                "Portfolio_Value_After": None,
                            }
                        )

                        weekly_liq_total += amount
                        remaining_to_liquidate -= amount

        # ---------- Portfolio after liquidation ----------
        portfolio_value_after_liq = 0.0
        for t, d in holdings.items():
            p = price_cache.get((t, str(trade_day)))
            portfolio_value_after_liq += d["qty"] * (p or 0.0)

        for ev in weekly_liq_events:
            ev["Portfolio_Value_After"] = round(portfolio_value_after_liq, 2)
            liq_events_all.append(ev)

        # ---------- Investment ----------
        # investable_from_bank = min(weekly_withdraw, bank_balance)
        if i < 52:  # first year (52 weeks)
            investable_from_bank = 0.0
        else:
            investable_from_bank = min(weekly_withdraw, bank_balance)

        investable_from_liq = weekly_liq_total
        total_investment = investable_from_bank + investable_from_liq

        bank_balance -= investable_from_bank
        bank_balance = round(bank_balance, 2)

        if selected and total_investment > 0:
            per_stock = total_investment / len(selected)
            for t in selected:
                p = price_cache.get((t, str(trade_day)))
                if p is None:
                    p = get_price(df, t, trade_day)
                    price_cache[(t, str(trade_day))] = p
                if not p:
                    continue

                qty = per_stock / p
                invested = qty * p

                holdings.setdefault(
                    t, {"qty": 0.0, "invested": 0.0, "avg": 0.0, "lots": []}
                )

                holdings[t]["lots"].append(
                    {
                        "week": w,
                        "buy_date": str(trade_day),
                        "qty": qty,
                        "price": p,
                        "amount": invested,
                    }
                )

                holdings[t]["qty"] += qty
                holdings[t]["invested"] += invested
                holdings[t]["avg"] = holdings[t]["invested"] / holdings[t]["qty"]

        tot_invested_from_bank += investable_from_bank
        tot_liquidated += weekly_liq_total

        # ---------- Final record ----------
        recs.append(
            {
                "Week": w,
                "Date": str(trade_day),
                "Investment_Date": str(trade_day),
                "Bank_Balance_Before_Investing": round(bank_before, 2),
                "Portfolio_Value_Before_Liquidity": round(portfolio_value_before, 2),
                "Total_Liquidation_Limit": round(allowed_liquidation_limit, 6),
                "Total_Liquidated": round(weekly_liq_total, 2),
                "Investment": round(total_investment, 2),
                "Bank_Balance_After_Investing": round(bank_balance, 2),
                "Portfolio_Value_After_Investing": round(portfolio_value_after_liq, 2),
                "Holdings": list(holdings.values()),
                "Liquidation_Details": weekly_liq_events,
            }
        )

        print(
            f"{w}: Trade on {trade_day} | "
            f"Portfolio ₹{portfolio_value_after_liq:,.2f} | "
            f"Liquidated ₹{weekly_liq_total:,.2f} | "
            f"Bank ₹{bank_balance:,.2f}"
        )

    # ---------- Average holding period ----------
    if all_holding_periods:
        avg_days = sum(d * q for d, q in all_holding_periods) / sum(
            q for _, q in all_holding_periods
        )
        avg_months = avg_days / 30.44
    else:
        avg_days = avg_months = 0

    return (
        recs,
        holdings,
        tot_invested_from_bank,
        tot_liquidated,
        ticker_counts,
        liq_events_all,
        bank_tracker_records,
        avg_days,
        avg_months,
    )


# ----------------- Utilities -----------------
def generate_holding_vs_return_scatter(csv_path):
    """
    Generate a scatter plot:
      X = Avg_Holding_Days
      Y = Avg_Return_Pct
    Adds:
      - Vertical line at 365 days
      - Different colors for <=365 and >365 holding days
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read {csv_path}: {e}")
        return

    required_cols = {"Avg_Holding_Days", "Avg_Return_Pct"}
    if not required_cols.issubset(df.columns):
        print("[WARN] Required columns not found for scatter plot.")
        return

    # clean data
    df = df.dropna(subset=["Avg_Holding_Days", "Avg_Return_Pct"])

    if df.empty:
        print("[WARN] No valid data for scatter plot.")
        return

    # split by holding-period threshold
    df_short = df[df["Avg_Holding_Days"] <= 365]
    df_long = df[df["Avg_Holding_Days"] > 365]

    plt.figure(figsize=(10, 6))

    # plot points
    plt.scatter(
        df_short["Avg_Holding_Days"],
        df_short["Avg_Return_Pct"],
        alpha=0.7,
        label="≤ 365 days",
    )

    plt.scatter(
        df_long["Avg_Holding_Days"],
        df_long["Avg_Return_Pct"],
        alpha=0.7,
        label="> 365 days",
    )

    # vertical divider
    plt.axvline(
        x=365,
        linestyle="--",
        linewidth=1.5,
        label="365-day threshold",
    )

    plt.xlabel("Average Holding Period (Days)")
    plt.ylabel("Average Return (%)")
    plt.title("Stock-wise Holding Period vs Returns")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = Path(RES) / "holding_vs_return_scatter.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Scatter plot saved to {out_path}")


def generate_holding_period_quartile_summary(csv_path):
    """
    Creates 4 equal-sized buckets based on Avg_Holding_Days
    and computes average return per bucket.
    """

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read {csv_path}: {e}")
        return

    required_cols = {"Ticker", "Avg_Holding_Days", "Avg_Return_Pct"}
    if not required_cols.issubset(df.columns):
        print("[WARN] Required columns not found for quartile analysis.")
        return

    # drop incomplete rows
    df = df.dropna(subset=["Avg_Holding_Days", "Avg_Return_Pct"])

    if len(df) < 4:
        print("[WARN] Not enough stocks to form holding-period quartiles.")
        return

    # sort by holding period
    df = df.sort_values("Avg_Holding_Days").reset_index(drop=True)

    # create equal-count buckets (first 25, next 25, etc.)
    df["Holding_Period_Bucket"] = (df.index // (len(df) // 4 + (len(df) % 4 > 0))) + 1

    df["Holding_Period_Bucket"] = df["Holding_Period_Bucket"].clip(upper=4)

    bucket_labels = {
        1: "Q1 (Shortest Holding)",
        2: "Q2",
        3: "Q3",
        4: "Q4 (Longest Holding)",
    }
    df["Holding_Period_Bucket"] = df["Holding_Period_Bucket"].map(bucket_labels)

    summary = (
        df.groupby("Holding_Period_Bucket")
        .agg(
            Stocks_Count=("Ticker", "count"),
            Start_Holding_Days=("Avg_Holding_Days", "min"),
            End_Holding_Days=("Avg_Holding_Days", "max"),
            Avg_Holding_Days=("Avg_Holding_Days", "mean"),
            Avg_Return_Pct=("Avg_Return_Pct", "mean"),
        )
        .reset_index()
    )

    out_path = Path(RES) / "holding_period_quartile_summary.csv"
    summary.to_csv(out_path, index=False)

    print(f"Holding-period quartile summary saved to {out_path}")


def plot_holding_period_quartile_returns(csv_path):
    """
    Bar plot: Holding Period Quartile vs Average Return %
    """

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read {csv_path}: {e}")
        return

    required_cols = {
        "Holding_Period_Bucket",
        "Avg_Return_Pct",
        "Stocks_Count",
    }
    if not required_cols.issubset(df.columns):
        print("[WARN] Required columns not found for quartile plot.")
        return

    # ensure correct order
    order = [
        "Q1 (Shortest Holding)",
        "Q2",
        "Q3",
        "Q4 (Longest Holding)",
    ]
    df["Holding_Period_Bucket"] = pd.Categorical(
        df["Holding_Period_Bucket"], categories=order, ordered=True
    )
    df = df.sort_values("Holding_Period_Bucket")

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        df["Holding_Period_Bucket"],
        df["Avg_Return_Pct"],
        alpha=0.8,
    )

    plt.xlabel("Holding Period Bucket")
    plt.ylabel("Average Return (%)")
    plt.title("Average Returns vs Holding Period Buckets")
    plt.grid(axis="y", alpha=0.3)

    # annotate values
    for bar, cnt, ret, d_start, d_end in zip(
        bars,
        df["Stocks_Count"],
        df["Avg_Return_Pct"],
        df["Start_Holding_Days"],
        df["End_Holding_Days"],
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ret:.2f}%\n{int(d_start)}–{int(d_end)}d\n({cnt} stocks)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    out_path = Path(RES) / "holding_period_quartile_returns.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Holding-period quartile return plot saved to {out_path}")


def daily_portfolio_tracking(df, holdings_records):
    """
    Create daily portfolio tracking (value over time) from monthly holdings_records.
    Uses the 'Holdings' in each monthly record to compute daily value.
    """
    print("Generating daily tracking...")

    # Create a DataFrame of monthly holdings and corresponding dates
    daily_records = []

    for rec in holdings_records:
        start_date = pd.to_datetime(rec["Investment_Date"])
        end_date = pd.to_datetime(rec["Date"])
        holdings = rec["Holdings"]

        # Create date range of trading days between invest and liquidation date
        nse = get_nse_calendar()
        try:
            schedule = nse.schedule(start_date=start_date, end_date=end_date)
            trading_days = schedule.index.date
        except Exception:
            trading_days = pd.date_range(
                start_date, end_date, freq="B"
            ).date  # fallback

        for day in trading_days:
            total_value = 0.0
            for h in holdings:
                ticker = h["Ticker"]
                qty = h["Quantity"]
                p = get_price(df, ticker, day)
                if p is None or np.isnan(p):
                    continue
                total_value += qty * p

            daily_records.append(
                {
                    "Date": str(day),
                    "Portfolio_Value": round(total_value, 2),
                }
            )

    daily_df = pd.DataFrame(daily_records)
    daily_df = daily_df.groupby("Date", as_index=False).sum().sort_values("Date")

    out_daily = Path(RES) / "daily_portfolio_tracking.csv"
    daily_df.to_csv(out_daily, index=False)
    print(f"Daily portfolio tracking saved to {out_daily}")
    return daily_df


# ----------------- SAVE OUTPUTS (WEEKLY VERSION) -----------------
def save_outputs(
    times, holdings, ticker_counts, df, liq_events_all, bank_tracker_records
):
    # ---------- Output file paths ----------
    OUT_JSON_WEEKLY = RES / f"weekly_tracking_{FILTER_KEY.replace(' ', '_')}.json"
    OUT_TS_WEEKLY = RES / "weekly_portfolio_timeseries.csv"
    OUT_HOLD_WEEKLY = RES / "weekly_portfolio_holdings.csv"
    OUT_FIN_WEEKLY = RES / "final_portfolio_weekly.csv"
    OUT_TICKER_COUNTS_WEEKLY = RES / "weekly_ticker_counts.csv"
    OUT_LIQ_WEEKLY = RES / "weekly_liquidation_details.csv"
    OUT_BANK_TRACKER_WEEKLY = RES / "weekly_bank_tracker.csv"
    OUT_BANK_JSON_WEEKLY = RES / "weekly_bank_tracker.json"

    # ---------- 1) Weekly JSON tracking ----------
    with open(OUT_JSON_WEEKLY, "w") as f:
        json.dump(times, f, indent=2)

    # ---------- 2) Weekly timeseries CSV ----------
    srows = []
    hrows = []

    for t in times:
        srows.append(
            {
                "Week": t["Week"],
                "Date": t["Date"],
                "Bank_Balance_Before_Investing": t.get(
                    "Bank_Balance_Before_Investing", 0.0
                ),
                "Portfolio_Value_Before_Liquidity": t.get(
                    "Portfolio_Value_Before_Liquidity", 0.0
                ),
                "Total_Liquidation_Limit": t.get("Total_Liquidation_Limit", 0.0),
                "Total_Liquidated": t.get("Total_Liquidated", 0.0),
                "Investment": t.get("Investment", 0.0),
                "Bank_Balance_After_Investing": t.get(
                    "Bank_Balance_After_Investing", 0.0
                ),
                "Portfolio_Value_After_Investing": t.get(
                    "Portfolio_Value_After_Investing", 0.0
                ),
            }
        )

        for h in t["Holdings"]:
            row = h.copy()
            row["Week"] = t["Week"]
            row["Date"] = t["Date"]
            hrows.append(row)

    pd.DataFrame(srows).to_csv(OUT_TS_WEEKLY, index=False)
    pd.DataFrame(hrows).to_csv(OUT_HOLD_WEEKLY, index=False)

    # ---------- 3) Top-weight stock(s) per week ----------
    try:
        df_hold = pd.DataFrame(hrows)

        portfolio_totals = (
            df_hold.groupby("Week")["Current_Amount"]
            .sum()
            .reset_index()
            .rename(columns={"Current_Amount": "Portfolio_Total"})
        )

        df_hold = df_hold.merge(portfolio_totals, on="Week", how="left")

        df_hold["Weight_Pct"] = (
            100 * df_hold["Current_Amount"] / df_hold["Portfolio_Total"]
        ).round(3)

        df_hold["Max_Weight_Pct"] = df_hold.groupby("Week")["Weight_Pct"].transform(
            "max"
        )

        top_per_week = df_hold[df_hold["Weight_Pct"] == df_hold["Max_Weight_Pct"]]

        top_per_week = top_per_week[
            [
                "Week",
                "Ticker",
                "Invested_Amount",
                "Current_Amount",
                "Portfolio_Total",
                "Weight_Pct",
            ]
        ]

        OUT_TOP_WEEKLY = RES / "top_weight_stock_per_week.csv"
        top_per_week.to_csv(OUT_TOP_WEEKLY, index=False)

        print(f"[OK] Top-weight stock(s) per week saved to {OUT_TOP_WEEKLY}")

    except Exception as e:
        print(f"[ERROR] Failed to generate top-weight weekly stock CSV: {e}")

    # ---------- 4) Weekly ticker counts ----------
    pd.DataFrame(ticker_counts).to_csv(OUT_TICKER_COUNTS_WEEKLY, index=False)

    # ---------- 5) Weekly liquidation events ----------
    if liq_events_all:
        pd.DataFrame(liq_events_all).to_csv(OUT_LIQ_WEEKLY, index=False)
    else:
        pd.DataFrame(
            columns=[
                "Week",
                "Date",
                "Stock",
                "Liquidated_Qty",
                "Liquidated_Amount",
                "Reason",
                "Remaining_Qty",
                "Portfolio_Value_After",
            ]
        ).to_csv(OUT_LIQ_WEEKLY, index=False)

    # ---------- 6) Final portfolio snapshot ----------
    final_rows = []
    for t, d in holdings.items():
        p = get_price(df, t, datetime.strptime(END_DATE, "%Y-%m-%d").date())
        final_rows.append(
            {
                "Ticker": t,
                "Quantity": round(d["qty"], 6),
                "Avg_Price": round(d["avg"], 2),
                "Invested": round(d["invested"], 2),
                "Current_Price": round(p or 0.0, 2),
                "Current_Amount": round(d["qty"] * (p or 0.0), 2),
            }
        )

    pd.DataFrame(final_rows).to_csv(OUT_FIN_WEEKLY, index=False)

    # ---------- 7) Weekly bank tracker ----------
    pd.DataFrame(bank_tracker_records).to_csv(OUT_BANK_TRACKER_WEEKLY, index=False)
    with open(OUT_BANK_JSON_WEEKLY, "w") as f:
        json.dump(bank_tracker_records, f, indent=2)

    print(
        f"Saved weekly outputs:\n"
        f"- {OUT_JSON_WEEKLY}\n"
        f"- {OUT_TS_WEEKLY}\n"
        f"- {OUT_HOLD_WEEKLY}\n"
        f"- {OUT_FIN_WEEKLY}\n"
        f"- {OUT_TICKER_COUNTS_WEEKLY}\n"
        f"- {OUT_LIQ_WEEKLY}\n"
        f"- {OUT_BANK_TRACKER_WEEKLY}\n"
        f"- {OUT_BANK_JSON_WEEKLY}"
    )


# ----------------- CHART -----------------
def generate_ticker_count_chart(ticker_counts, skip_months=12):
    dfc = pd.DataFrame(ticker_counts)
    if dfc.empty:
        print("[WARN] No ticker count data to plot.")
        return

    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot Holdings Count
    if "Holdings_Count" in dfc.columns:
        ax.plot(
            range(len(dfc)),
            dfc["Holdings_Count"],
            marker="o",
            label="Holdings Count",
            linewidth=2,
        )

        df_eval = dfc.iloc[skip_months:]

        if not df_eval.empty:
            max_val = df_eval["Holdings_Count"].max()
            min_val = df_eval["Holdings_Count"].min()

            max_idx = df_eval["Holdings_Count"].idxmax()
            min_idx = df_eval["Holdings_Count"].idxmin()

            max_month = dfc.loc[max_idx, "Month"]
            min_month = dfc.loc[min_idx, "Month"]

            # Mark max & min on chart
            ax.scatter(max_idx, max_val)
            ax.scatter(min_idx, min_val)

            ax.annotate(
                f"Max (after {skip_months}m): {max_val}\n({max_month})",
                (max_idx, max_val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

            ax.annotate(
                f"Min (after {skip_months}m): {min_val}\n({min_month})",
                (min_idx, min_val),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=9,
            )

            # Log results
            print(
                f"[INFO] After {skip_months} months → "
                f"Max holdings: {max_val} ({max_month}), "
                f"Min holdings: {min_val} ({min_month})"
            )

    # Plot Selected Tickers
    if "Selected_Tickers" in dfc.columns:
        ax.plot(
            range(len(dfc)),
            dfc["Selected_Tickers"],
            marker="s",
            label="Selected Tickers",
            linewidth=2,
        )

    step = max(1, len(dfc) // 20)
    ax.set_xticks(range(0, len(dfc), step))
    ax.set_xticklabels(
        [dfc.iloc[i]["Month"] for i in range(0, len(dfc), step)],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Stocks")
    ax.set_title("Monthly Portfolio Size (Holdings vs Selected)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Chart saved to {OUT_CHART}")


# ----------------- DAILY PORTFOLIO CHART -----------------
def generate_daily_portfolio_chart(daily_csv_path):
    """
    Generate a line chart of daily portfolio value.
    Input: path to 'daily_portfolio_tracking.csv' file.
    """
    try:
        df = pd.read_csv(daily_csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    except Exception as e:
        print(f"[ERROR] Could not read daily portfolio CSV: {e}")
        return

    if df.empty or "Portfolio_Value" not in df.columns:
        print("[WARN] No daily portfolio data to plot.")
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df["Date"], df["Equity"], linewidth=2)
    ax.set_title("Daily Portfolio Value Over Time", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value (₹)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    OUT_DAILY_CHART = Path(RES) / "daily_portfolio_chart.png"
    plt.savefig(OUT_DAILY_CHART, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Daily portfolio chart saved to {OUT_DAILY_CHART}")


def generate_1_rupee_chart(daily_csv_path):
    """
    Generate a normalized chart (1 rupee chart) where Equity starts at 1.
    Uses column: 'Equity' from daily_portfolio_tracking.csv.
    Equity is divided by 12000 so day 1 = 1.
    """
    BASE = 12000  # starting value

    try:
        df = pd.read_csv(daily_csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    except Exception as e:
        print(f"[ERROR] Could not read daily portfolio CSV: {e}")
        return

    if df.empty or "Equity" not in df.columns:
        print("[WARN] No daily Equity data to plot for 1-rupee chart.")
        return

    # Normalization
    df["One_Rupee_Value"] = df["Equity"] / BASE

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df["Date"], df["One_Rupee_Value"], linewidth=2)

    ax.set_title("1 Rupee Normalized Portfolio Chart", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Value of 1 Rupee", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    OUT_1_RUPEE_CHART = Path(RES) / "1_rupee_daily_chart.png"
    plt.savefig(OUT_1_RUPEE_CHART, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"1-Rupee chart saved to {OUT_1_RUPEE_CHART}")


def get_params(save_file_name, initial_capital, years, filter_dict, filter_):
    """
    WEEKLY version of get_params()

    - Builds DAILY equity curve from WEEKLY tracker JSON
    - Uses:
        * Investment_Date == Date (same day)
        * Holdings snapshot per week
        * Bank_Balance_After_Investing (stepwise constant)
    - Computes standard performance metrics
    """

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import json

    # ---------- Load weekly portfolio CSV ----------
    portfolio_weekly = pd.read_csv(save_file_name)

    # Final capital (compatibility)
    if "Portfolio_Value_After_Investing" in portfolio_weekly.columns:
        strategy_final_capital = portfolio_weekly[
            "Portfolio_Value_After_Investing"
        ].iloc[-1]
    else:
        strategy_final_capital = portfolio_weekly.iloc[-1].iloc[-1]

    # ---------- Load weekly tracker JSON ----------
    save_path = Path(save_file_name)
    weekly_json = save_path.parent / f"weekly_tracking_{filter_.replace(' ', '_')}.json"

    if not weekly_json.exists():
        raise FileNotFoundError(f"Weekly tracker JSON not found: {weekly_json}")

    with open(weekly_json, "r") as f:
        weekly_recs = json.load(f)

    # ---------- Load price data ----------
    price_df = load_stocks_df()
    nse = get_nse_calendar()

    daily_records = []

    # ---------- Build DAILY equity ----------
    for rec in weekly_recs:

        trade_date_raw = rec.get("Date")
        holdings = rec.get("Holdings", [])
        bank_after = rec.get("Bank_Balance_After_Investing")

        if trade_date_raw is None:
            continue

        trade_date = pd.to_datetime(trade_date_raw).date()

        # Get trading days UNTIL NEXT WEEK (stepwise equity)
        try:
            sched = nse.schedule(
                start_date=trade_date,
                end_date=trade_date + pd.Timedelta(days=6),
            )
            trading_days = sched.index.date
        except Exception:
            trading_days = [trade_date]

        for day in trading_days:
            portfolio_value = 0.0

            for h in holdings:
                ticker = h.get("Ticker")
                qty = h.get("Quantity", 0.0)

                if not ticker or qty == 0:
                    continue

                p = get_price(price_df, ticker, day)
                if p is None or (isinstance(p, float) and np.isnan(p)):
                    continue

                portfolio_value += float(qty) * float(p)

            daily_records.append(
                {
                    "Date": pd.to_datetime(day),
                    "Portfolio_Value": round(portfolio_value, 2),
                    "Bank": float(bank_after) if bank_after is not None else np.nan,
                    "Week": rec.get("Week"),
                }
            )

    if not daily_records:
        raise RuntimeError("No daily equity records could be generated.")

    daily_df = pd.DataFrame(daily_records)

    # Aggregate in case of overlaps
    daily_df = (
        daily_df.sort_values("Date")
        .groupby("Date", as_index=False)
        .agg(
            {
                "Portfolio_Value": "sum",
                "Bank": "first",
                "Week": "first",
            }
        )
    )

    # Forward-fill bank balance
    daily_df["Bank"] = daily_df["Bank"].ffill().bfill()

    # ---------- Equity ----------
    daily_df["Equity"] = daily_df["Portfolio_Value"] + daily_df["Bank"]

    # ---------- Save daily CSV ----------
    out_daily = save_path.parent / "daily_portfolio_tracking.csv"
    df_to_save = daily_df.copy()
    df_to_save["Date"] = df_to_save["Date"].dt.strftime("%Y-%m-%d")
    df_to_save.to_csv(out_daily, index=False)

    # ---------- Metrics ----------
    daily_df = daily_df.sort_values("Date").reset_index(drop=True)
    daily_df["Equity"] = daily_df["Equity"].astype(float)

    daily_df["LogReturns"] = np.log(daily_df["Equity"] / daily_df["Equity"].shift(1))
    log_returns = daily_df["LogReturns"].dropna()

    final_equity = daily_df["Equity"].iloc[-1]
    initial_equity = initial_capital

    annualized_returns = (((final_equity / initial_equity) ** (1 / years)) - 1) * 100
    annualized_returns = round(annualized_returns, 2)

    annualized_volatility = float(log_returns.std() * np.sqrt(252))

    downside = log_returns.copy()
    downside[downside > 0] = 0
    downside_volatility = float(downside.std() * np.sqrt(252))

    roll_max = daily_df["Equity"].cummax()
    drawdown = (daily_df["Equity"] / roll_max) - 1
    max_drawdown = round(abs(drawdown.min()) * 100, 2)

    ann_ret_decimal = annualized_returns / 100.0
    sharpe_ratio = (
        ann_ret_decimal / annualized_volatility
        if annualized_volatility > 0
        else float("inf")
    )
    sortino_ratio = (
        ann_ret_decimal / downside_volatility
        if downside_volatility > 0
        else float("inf")
    )
    calmar_ratio = (
        ann_ret_decimal / (max_drawdown / 100.0) if max_drawdown > 0 else float("inf")
    )

    sharpe_ratio = round(sharpe_ratio, 2)
    sortino_ratio = round(sortino_ratio, 2)
    calmar_ratio = round(calmar_ratio, 2)

    # ---------- Average selected stocks ----------
    counts = []
    for wk, filters in filter_dict.items():
        if filter_ in filters:
            counts.append(len(filters[filter_]))

    avg_stocks = round(sum(counts) / len(counts), 2) if counts else "N/A"
    min_stocks = min(counts) if counts else "N/A"
    max_stocks = max(counts) if counts else "N/A"

    print("\n================ PERFORMANCE SUMMARY ================\n")
    print(f"{'Initial Capital':30}: ₹{initial_equity:,.2f}")
    print(f"{'Final Capital':30}: ₹{final_equity:,.2f}")
    print(f"{'Total Return (%)':30}: {(final_equity / initial_equity - 1) * 100:.2f}%")
    print(f"{'Annualized Returns (%)':30}: {annualized_returns:.2f}%")
    print(f"{'Annualized Volatility':30}: {annualized_volatility:.6f}")
    print(f"{'Downside Volatility':30}: {downside_volatility:.6f}")
    print(f"{'Max Drawdown (%)':30}: {max_drawdown:.2f}%")
    print(f"{'Calmar Ratio':30}: {calmar_ratio:.2f}")
    print(f"{'Sharpe Ratio':30}: {sharpe_ratio:.2f}")
    print(f"{'Sortino Ratio':30}: {sortino_ratio:.2f}")
    print(f"{'Average Stocks':30}: {avg_stocks}")
    print(f"{'Min Stocks':30}: {min_stocks}")
    print(f"{'Max Stocks':30}: {max_stocks}")
    print(f"{'Years Simulated':30}: {years:.2f}")
    print(f"{'Trading Periods':30}: {len(daily_df)}")
    print("\n=====================================================\n")

    metrics = {
        "Initial Capital": round(initial_equity, 2),
        "Final Capital": round(final_equity, 2),
        "Total Return (%)": round((final_equity / initial_equity - 1) * 100, 2),
        "Annualized Returns (%)": annualized_returns,
        "Annualized Volatility": round(annualized_volatility, 6),
        "Downside Volatility": round(downside_volatility, 6),
        "Calmar Ratio": calmar_ratio,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Average Stocks": avg_stocks,
        "Min Stocks": min_stocks,
        "Max Stocks": max_stocks,
        "Years Simulated": round(years, 2),
        "Trading Periods": len(daily_df),
    }

    return metrics, daily_df


# ----------------- MAIN -----------------
def main():
    print(
        "=== Momentum / Value Portfolio Tracker (Bank-balance + WEEKLY Lot-wise Liquidation) ==="
    )

    # ---------- Load Data ----------
    df = load_stocks_df()
    invest_flt = load_filter(INVEST_JSON)  # weekly investment plan
    good_flt = load_filter(GOOD_JSON)  # weekly good stocks list

    # ---------- Build weekly timeline ----------
    weeks = week_iter(START_DATE, END_DATE)

    if not weeks:
        raise RuntimeError("No weeks generated — check START_DATE / END_DATE")

    # ---------- Run Simulation ----------
    (
        recs,
        holdings,
        tot_from_bank,
        tot_liq,
        ticker_counts,
        liq_events_all,
        bank_tracker,
        avg_days,
        avg_months,
    ) = tracker(
        df,
        invest_flt,
        good_flt,
        weeks,
        INITIAL_BANK_BALANCE,
        WEEKLY_BANK_WITHDRAW,
        FILTER_KEY,
    )

    # ---------- Save Outputs ----------
    save_outputs(recs, holdings, ticker_counts, df, liq_events_all, bank_tracker)

    # ---------- Charts & Daily Tracking ----------
    generate_ticker_count_chart(ticker_counts)
    daily_df = daily_portfolio_tracking(df, recs)

    # ---------- Scatter Plot + Holding-period Quartile Analysis ----------
    scatter_csv = RES / "avg_holding_period_per_stock.csv"

    if scatter_csv.exists():
        generate_holding_vs_return_scatter(scatter_csv)
        generate_holding_period_quartile_summary(scatter_csv)

        quartile_summary_csv = RES / "holding_period_quartile_summary.csv"
        if quartile_summary_csv.exists():
            plot_holding_period_quartile_returns(quartile_summary_csv)
    else:
        print("[WARN] No per-stock holding period CSV found for analysis.")

    # ---------- Performance Metrics ----------
    years = len(weeks) / 52.0 if len(weeks) > 0 else 1.0
    params = get_params(OUT_TS, INITIAL_BANK_BALANCE, years, invest_flt, FILTER_KEY)

    generate_1_rupee_chart(os.path.join(RES, "daily_portfolio_tracking.csv"))
    generate_daily_portfolio_chart(os.path.join(RES, "daily_portfolio_tracking.csv"))

    # ---------- Summary ----------
    print("\n========== FINAL SUMMARY ==========")
    print(f"Total invested from bank: ₹{tot_from_bank:,.2f}")
    print(f"Total liquidated: ₹{tot_liq:,.2f}")
    print(f"Average Holding Period: {avg_days:.1f} days ({avg_months:.2f} months)")
    print("Performance parameters (dict):")
    print(params)

    print("\nRun complete. Outputs and charts saved in:", RES)


if __name__ == "__main__":
    main()
