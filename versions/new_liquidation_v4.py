# new_liquidation_v4

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
        "output_dir": "output\\value\\strategy-original\\value4",
    },
    "value-final": {
        "invest_json": "data\\value_filter4_modified_with_sharpe_10_drop_20_iter.json",
        "good_json": "data\\filtered_outputs\\Value_Filter_2_50%.json",
        "output_dir": "output\\value\\strategy-final\\value4",
    },
    "momentum-original": {
        "invest_json": "data\\momentum.json",
        "good_json": "data\\good_stocks_combined.json",
        "output_dir": "output\\momentum\\strategy-original\\momentum4",
    },
    "momentum-final": {
        "invest_json": "data\\momentum_modified_filter4_with_sharpe_10_drop_20_iter.json",
        "good_json": "data\\good_stocks_combined.json",
        "output_dir": "output\\momentum\\strategy-final\\momentum4",
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
MONTHLY_BANK_WITHDRAW = 1000.0

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


def month_iter(start, end):
    s = datetime.strptime(start, "%b %Y")
    e = datetime.strptime(end, "%Y-%m-%d")
    months = []
    while s <= e:
        months.append(s.strftime("%b %Y"))
        s += relativedelta(months=1)
    return months


def monthend(month):
    d = datetime.strptime(month, "%b %Y")
    return (d + relativedelta(months=1, days=-1)).date()


# ----------------- TRADING DAY HELPERS (pandas_market_calendars) -----------------
def get_nse_calendar():
    """Return the NSE calendar object (pandas_market_calendars)."""
    return mcal.get_calendar("NSE")


def get_first_trading_day(nse_calendar, month):
    """
    Return the first available trading day (date object) in given month (format 'Jul 2014').
    Returns None if no trading day found.
    """
    d = datetime.strptime(month, "%b %Y")
    start = d.replace(day=1)
    end = d + relativedelta(months=1, days=-1)
    try:
        schedule = nse_calendar.schedule(start_date=start, end_date=end)
    except Exception:
        # defensive: if calendar call fails for some month, return None
        return None
    if schedule is None or schedule.empty:
        return None
    return schedule.index[0].date()


def get_last_trading_day(nse_calendar, month):
    """
    Return the last available trading day (date object) in given month (format 'Jul 2014').
    Returns None if no trading day found.
    """
    d = datetime.strptime(month, "%b %Y")
    start = d.replace(day=1)
    end = d + relativedelta(months=1, days=-1)
    try:
        schedule = nse_calendar.schedule(start_date=start, end_date=end)
    except Exception:
        return None
    if schedule is None or schedule.empty:
        return None
    return schedule.index[-1].date()


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


# ----------------- TRACKER with updated liquidation logic -----------------
def tracker(
    df, invest_flt, good_flt, months, initial_bank_balance, monthly_withdraw, filter_key
):
    """
    Runs the monthly simulation using:
      - invest_flt: JSON dict mapping month -> filters -> list of tickers (used for investing)
      - good_flt: JSON dict mapping month -> list of good tickers (used for liquidation checks)
      - bank_balance variables to control monthly investments from the bank
    Uses NSE trading calendar: invests on first trading day of month, liquidates on last trading day.

    Returns:
      recs: monthly records (detailed)
      holdings: final holdings dict
      tot_invested_from_bank: total invested using bank balance
      tot_liquidated: total cash realized from liquidations
      ticker_counts: month-by-month summary
      liq_events_all: flattened list of liquidation events across months
      bank_tracker_records: month-by-month bank tracker rows (for CSV/JSON)
      avg_days: weighted average holding period (days)
      avg_months: weighted average holding period (months)
    """

    holdings = {}  # ticker -> {qty, invested, avg, lots: [{month, qty, price}]}
    liq_events_all = []
    recs = []
    ticker_counts = []
    bank_tracker_records = []

    all_holding_periods = []  # (holding_days, qty_removed)
    per_stock_holding_periods = {}  # ticker -> list of (holding_days, qty_removed)

    all_returns = []  # (return_pct, qty_removed)
    per_stock_returns = {}  # ticker -> list of (return_pct, qty_removed)

    bank_balance = initial_bank_balance
    tot_invested_from_bank = 0.0
    tot_liquidated = 0.0

    nse = get_nse_calendar()

    # A small price cache: key = (ticker, date_str) -> price
    price_cache = {}

    liquidation_limit_hits = []  # list of dicts {Month, Limit, Possible, Excess}
    liquidation_limit_hit_count = 0
    total_excess_liquidation = 0.0

    for i, m in enumerate(months):
        invest_day = get_first_trading_day(nse, m)
        liq_day = get_last_trading_day(nse, m)

        if invest_day is None or liq_day is None:
            print(
                f"[WARN] {m}: No trading day found (invest_day={invest_day}, liq_day={liq_day}). Skipping month."
            )
            continue

        selected = [
            t for t in invest_flt.get(m, {}).get(filter_key, []) if t in df.columns
        ]
        good_stocks = good_flt.get(m, [])

        bank_before = bank_balance

        # ---------- Portfolio valuation BEFORE liquidity ----------
        portfolio_value_before = 0.0
        holding_summaries = {}
        for t, d in holdings.items():
            # Use cached price fetch
            cache_key = (t, str(liq_day))
            if cache_key in price_cache:
                p = price_cache[cache_key]
            else:
                p = get_price(df, t, liq_day)
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
        allowed_liquidation_limit = bank_plus_portfolio / 12.0

        monthly_liq_total = 0.0
        monthly_liq_events = []

        # ---------- Lot-wise sellable evaluation ----------
        sellable_lots, per_stock_sellable_value, total_possible_liquidation = (
            evaluate_sellable_lots(
                holdings, good_stocks, liq_day, df, price_cache=price_cache
            )
        )

        # ---------- Liquidation logic using sellable_lots ----------
        if i >= 1 and portfolio_value_before > 0 and sellable_lots:
            # If possible liquidation is within limit -> sell all sellable lots
            if total_possible_liquidation <= allowed_liquidation_limit + 1e-9:
                # Sell all sellable lots (stock by stock)
                for t, lots in sellable_lots.items():
                    for lot in lots:
                        qty = lot["qty"]
                        price = lot["current_price"]
                        amount = qty * price

                        # Remove qty from holdings using FIFO removal (supports partial)
                        cost_removed, periods = remove_qty_from_lots(
                            holdings[t], qty, liq_day
                        )
                        # record holding periods
                        all_holding_periods.extend(periods)
                        if t not in per_stock_holding_periods:
                            per_stock_holding_periods[t] = []
                        per_stock_holding_periods[t].extend(periods)

                        # --- Return tracking ---
                        if cost_removed > 0:
                            avg_sell_price = price
                            total_qty_sold = sum(q for _, q in periods)
                            avg_buy_price = safe_div(cost_removed, total_qty_sold)
                            ret = safe_div(avg_sell_price, avg_buy_price) - 1.0
                            all_returns.extend([(ret, qty) for _, qty in periods])
                            if t not in per_stock_returns:
                                per_stock_returns[t] = []
                            per_stock_returns[t].extend(
                                [(ret, qty) for _, qty in periods]
                            )

                        if holdings.get(t, {}).get("qty", 0.0) == 0:
                            holdings.pop(t, None)

                        ev = {
                            "Month": m,
                            "Date": str(liq_day),
                            "Stock": t,
                            "Liquidated_Qty": round(qty, 6),
                            "Liquidated_Amount": round(amount, 2),
                            "Reason": "Lot-sellable (within limit)",
                            "Remaining_Qty": round(
                                holdings.get(t, {}).get("qty", 0.0), 6
                            ),
                            "Portfolio_Value_After": None,
                        }
                        monthly_liq_events.append(ev)
                        monthly_liq_total += amount

            else:
                # total_possible_liquidation > allowed_liquidation_limit
                if not PARTIAL_ALLOWED:
                    # Track liquidation limit exceeded
                    excess = total_possible_liquidation - allowed_liquidation_limit
                    if excess > 0:
                        liquidation_limit_hit_count += 1
                        total_excess_liquidation += excess
                        liquidation_limit_hits.append(
                            {
                                "Month": m,
                                "Allowed_Limit": round(allowed_liquidation_limit, 2),
                                "Possible_Liquidation": round(
                                    total_possible_liquidation, 2
                                ),
                                "Excess": round(excess, 2),
                            }
                        )

                    # Current policy: even if exceeds limit, sell everything but mark over_liquidation
                    for t, lots in sellable_lots.items():
                        for lot in lots:
                            qty = lot["qty"]
                            price = lot["current_price"]
                            amount = qty * price

                            cost_removed, periods = remove_qty_from_lots(
                                holdings[t], qty, liq_day
                            )
                            all_holding_periods.extend(periods)
                            if t not in per_stock_holding_periods:
                                per_stock_holding_periods[t] = []
                            per_stock_holding_periods[t].extend(periods)

                            # --- Return tracking ---
                            if cost_removed > 0:
                                avg_sell_price = price
                                total_qty_sold = sum(q for _, q in periods)
                                avg_buy_price = safe_div(cost_removed, total_qty_sold)
                                ret = safe_div(avg_sell_price, avg_buy_price) - 1.0
                                all_returns.extend([(ret, qty) for _, qty in periods])
                                if t not in per_stock_returns:
                                    per_stock_returns[t] = []
                                per_stock_returns[t].extend(
                                    [(ret, qty) for _, qty in periods]
                                )

                            if holdings.get(t, {}).get("qty", 0.0) == 0:
                                holdings.pop(t, None)

                            ev = {
                                "Month": m,
                                "Date": str(liq_day),
                                "Stock": t,
                                "Liquidated_Qty": round(qty, 6),
                                "Liquidated_Amount": round(amount, 2),
                                "Reason": "Lot-sellable (over limit - PARTIAL_ALLOWED=False)",
                                "Remaining_Qty": round(
                                    holdings.get(t, {}).get("qty", 0.0), 6
                                ),
                                "Portfolio_Value_After": None,
                            }
                            monthly_liq_events.append(ev)
                            monthly_liq_total += amount

                else:
                    # PARTIAL_ALLOWED == True
                    # Lot-based partial liquidation (global, FIFO-safe)

                    remaining_to_liquidate = allowed_liquidation_limit

                    # Step 1: flatten + globally sort sellable lots
                    sorted_lots = flatten_and_sort_sellable_lots(sellable_lots)

                    # Step 2: sell lots in that order until limit exhausted
                    for lot in sorted_lots:
                        if remaining_to_liquidate <= 1e-9:
                            break

                        t = lot["ticker"]
                        price = lot["current_price"]
                        max_qty = lot["qty"]
                        max_value = max_qty * price

                        if max_value <= remaining_to_liquidate + 1e-9:
                            qty_to_sell = max_qty
                            amount = max_value
                        else:
                            qty_to_sell = remaining_to_liquidate / price
                            amount = qty_to_sell * price

                        cost_removed, periods = remove_qty_from_lots(
                            holdings[t], qty_to_sell, liq_day
                        )

                        # --- holding period tracking ---
                        all_holding_periods.extend(periods)
                        if t not in per_stock_holding_periods:
                            per_stock_holding_periods[t] = []
                        per_stock_holding_periods[t].extend(periods)

                        # --- return tracking ---
                        if cost_removed > 0:
                            avg_sell_price = price
                            total_qty_sold = sum(q for _, q in periods)
                            avg_buy_price = safe_div(cost_removed, total_qty_sold)
                            ret = safe_div(avg_sell_price, avg_buy_price) - 1.0

                            all_returns.extend(
                                [(ret, qty_to_sell) for _, qty in periods]
                            )
                            if t not in per_stock_returns:
                                per_stock_returns[t] = []
                            per_stock_returns[t].extend(
                                [(ret, qty_to_sell) for _, qty in periods]
                            )

                        if holdings.get(t, {}).get("qty", 0.0) == 0:
                            holdings.pop(t, None)

                        ev = {
                            "Month": m,
                            "Date": str(liq_day),
                            "Stock": t,
                            "Liquidated_Qty": round(qty_to_sell, 6),
                            "Liquidated_Amount": round(amount, 2),
                            "Reason": "Lot-based partial liquidation",
                            "Remaining_Qty": round(
                                holdings.get(t, {}).get("qty", 0.0), 6
                            ),
                            "Portfolio_Value_After": None,
                        }

                        monthly_liq_events.append(ev)
                        monthly_liq_total += amount
                        remaining_to_liquidate -= amount

        # ---------- Portfolio after liquidation ----------
        portfolio_value_after_liq = 0.0
        for t, d in holdings.items():
            cache_key = (t, str(liq_day))
            if cache_key in price_cache:
                p = price_cache[cache_key]
            else:
                p = get_price(df, t, liq_day)
                price_cache[cache_key] = p
            portfolio_value_after_liq += d["qty"] * (p or 0.0)

        for ev in monthly_liq_events:
            ev["Portfolio_Value_After"] = round(portfolio_value_after_liq, 2)
            liq_events_all.append(ev)

        # ---------- Investment step ----------
        investable_from_bank = (
            min(monthly_withdraw, bank_balance) if bank_balance > 0 else 0
        )
        investable_from_liq = monthly_liq_total
        total_investment_this_month = investable_from_bank + investable_from_liq
        bank_balance -= investable_from_bank
        bank_balance = round(bank_balance, 2)

        if selected and total_investment_this_month > 0:
            per_stock = total_investment_this_month / len(selected)
            for t in selected:
                p_cache_key = (t, str(invest_day))
                if p_cache_key in price_cache:
                    p = price_cache[p_cache_key]
                else:
                    p = get_price(df, t, invest_day)
                    price_cache[p_cache_key] = p
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    continue
                qty = per_stock / p
                invested = qty * p
                if t not in holdings:
                    holdings[t] = {"qty": 0.0, "invested": 0.0, "avg": 0.0, "lots": []}
                # holdings[t]["lots"].append({"month": m, "qty": qty, "price": p})
                holdings[t]["lots"].append(
                    {
                        "month": m,  # display string ("Jan 2021")
                        "buy_date": str(invest_day),  # actual trading day
                        "qty": qty,  # quantity purchased
                        "price": p,  # buy price
                        "amount": qty * p,  # invested amount for this lot
                    }
                )
                holdings[t]["qty"] += qty
                holdings[t]["invested"] += invested
                holdings[t]["avg"] = holdings[t]["invested"] / holdings[t]["qty"]

        tot_invested_from_bank += investable_from_bank
        tot_liquidated += monthly_liq_total

        # ---------- Portfolio after investing ----------
        portfolio_value_after_invest = 0.0
        holdings_rows = []
        for t, d in holdings.items():
            cache_key = (t, str(liq_day))
            if cache_key in price_cache:
                p = price_cache[cache_key]
            else:
                p = get_price(df, t, liq_day)
                price_cache[cache_key] = p
            cur = d["qty"] * (p or 0.0)
            holdings_rows.append(
                {
                    "Ticker": t,
                    "Quantity": round(d["qty"], 6),
                    "Avg_Buy_Price": round(d["avg"], 2),
                    "Invested_Amount": round(d["invested"], 2),
                    "Current_Price": round(p or 0.0, 2),
                    "Current_Amount": round(cur, 2),
                }
            )
            portfolio_value_after_invest += cur

        for r in holdings_rows:
            r["Percent_of_Portfolio"] = round(
                100 * safe_div(r["Current_Amount"], portfolio_value_after_invest), 3
            )

        rec = {
            "Month": m,
            "Date": str(liq_day),
            "Investment_Date": str(invest_day),
            "Bank_Balance_Before_Investing": round(bank_before, 2),
            "Portfolio_Value_Before_Liquidity": round(portfolio_value_before, 2),
            "Bank_Plus_Portfolio": round(bank_plus_portfolio, 2),
            "Total_Liquidation_Limit": round(allowed_liquidation_limit, 6),
            "Total_Liquidated": round(monthly_liq_total, 2),
            "Investable_Amount_from_Bank_Balance": round(investable_from_bank, 2),
            "Investable_Amount_from_Liquidation": round(investable_from_liq, 2),
            "Investment": round(total_investment_this_month, 2),
            "Bank_Balance_After_Investing": round(bank_balance, 2),
            "Portfolio_Value_After_Investing": round(portfolio_value_after_invest, 2),
            "Holdings": holdings_rows,
            "Liquidation_Details": monthly_liq_events,
        }

        recs.append(rec)

        ticker_counts.append(
            {
                "Month": m,
                "Date": str(liq_day),
                "Selected_Tickers": len(selected),
                "Holdings_Count": len(holdings),
                "Total_Liquidated": round(monthly_liq_total, 2),
                "Stocks_Affected": len(monthly_liq_events),
            }
        )

        bank_tracker_records.append(
            {
                "Month": m,
                "Date": str(liq_day),
                "Bank_Balance_Before_Investing": round(bank_before, 2),
                "Portfolio_Value_Before_Liquidity": round(portfolio_value_before, 2),
                "Total_Liquidation_Limit": round(allowed_liquidation_limit, 6),
                "Total_Liquidated": round(monthly_liq_total, 2),
                "Investable_Amount_from_Bank_Balance": round(investable_from_bank, 2),
                "Investable_Amount_from_Liquidation": round(investable_from_liq, 2),
                "Investment": round(total_investment_this_month, 2),
                "Bank_Balance_After_Investing": round(bank_balance, 2),
                "Portfolio_Value_After_Investing": round(
                    portfolio_value_after_invest, 2
                ),
                "Stocks_Affected": len(monthly_liq_events),
            }
        )

        print(
            f"{m}: Invest on {invest_day}, Liquidate on {liq_day} "
            f"-> Portfolio ₹{portfolio_value_after_invest:,.2f}, Liquidated ₹{monthly_liq_total:,.2f}, Bank ₹{bank_balance:,.2f}"
        )

    # ---------- Average Return Computation ----------
    if all_returns:
        total_ret = sum(r * q for r, q in all_returns)
        total_qty_r = sum(q for _, q in all_returns)
        avg_return_all = total_ret / total_qty_r
    else:
        avg_return_all = 0.0
    print(f"Average return: {avg_return_all*100:.2f}%\n")

    # ---------- Per-stock holding period and returns ----------
    avg_by_stock = {}
    for t, plist in per_stock_holding_periods.items():
        total_days = sum(days * qty for days, qty in plist)
        total_qty = sum(qty for _, qty in plist)
        if total_qty > 0:
            avg_days_t = total_days / total_qty
            avg_months_t = avg_days_t / 30.44
            avg_by_stock[t] = {
                "Avg_Holding_Days": round(avg_days_t, 1),
                "Avg_Holding_Months": round(avg_months_t, 2),
            }

    for t, rlist in per_stock_returns.items():
        total_ret = sum(r * q for r, q in rlist)
        total_qty_r = sum(q for _, q in rlist)
        if total_qty_r > 0:
            avg_ret_t = total_ret / total_qty_r
            if t not in avg_by_stock:
                avg_by_stock[t] = {}
            avg_by_stock[t]["Avg_Return_Pct"] = round(avg_ret_t * 100, 2)

    # ---------- Overall Average Holding Period (stock-level, Excel-matching) ----------
    if avg_by_stock:
        avg_days = sum(v["Avg_Holding_Days"] for v in avg_by_stock.values()) / len(
            avg_by_stock
        )
        avg_months = avg_days / 30.44
    else:
        avg_days = 0
        avg_months = 0

    print(f"\nAverage holding period: {avg_days:.1f} days ({avg_months:.2f} months)\n")

    # ---------- Save per-stock average to CSV ----------
    if avg_by_stock:
        df_avg = pd.DataFrame([{"Ticker": k, **v} for k, v in avg_by_stock.items()])
        out_csv = RES / "avg_holding_period_per_stock.csv"
        df_avg.to_csv(out_csv, index=False)
        print(f"Per-stock holding period + returns saved to {out_csv}")

    # ---------- Save liquidation limit hits ----------
    if liquidation_limit_hits:
        df_lhits = pd.DataFrame(liquidation_limit_hits)
        out_csv = RES / "liquidation_limit_hits.csv"
        df_lhits.to_csv(out_csv, index=False)
        print(f"Liquidation-limit excess events saved to {out_csv}")

    print(f"\nTotal liquidation-limit hits: {liquidation_limit_hit_count}")
    print(f"Total excess liquidation: ₹{total_excess_liquidation:,.2f}\n")

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


# ----------------- SAVE OUTPUTS -----------------
def save_outputs(
    times, holdings, ticker_counts, df, liq_events_all, bank_tracker_records
):
    # 1) JSON full monthly tracking
    with open(OUT_JSON, "w") as f:
        json.dump(times, f, indent=2)

    # 2) CSV: monthly summary timeseries (from recs)
    srows = []
    hrows = []
    for t in times:
        srows.append(
            {
                "Month": t["Month"],
                "Date": t["Date"],
                "Bank_Balance_Before_Investing": t.get(
                    "Bank_Balance_Before_Investing", 0.0
                ),
                "Portfolio_Value_Before_Liquidity": t.get(
                    "Portfolio_Value_Before_Liquidity", 0.0
                ),
                "Total_Liquidation_Limit": t.get("Total_Liquidation_Limit", 0.0),
                "Available_for_Liquidation": t.get("Available_for_Liquidation", 0.0),
                "Total_Liquidated": t.get("Total_Liquidated", 0.0),
                "Investable_Amount_from_Bank_Balance": t.get(
                    "Investable_Amount_from_Bank_Balance", 0.0
                ),
                "Investable_Amount_from_Liquidation": t.get(
                    "Investable_Amount_from_Liquidation", 0.0
                ),
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
            row["Month"] = t["Month"]
            row["Date"] = t["Date"]
            hrows.append(row)

    pd.DataFrame(srows).to_csv(OUT_TS, index=False)
    pd.DataFrame(hrows).to_csv(OUT_HOLD, index=False)

    # ---------------------------------------------------------
    # TOP PORTFOLIO STOCK(S) PER MONTH (ties included)
    # ---------------------------------------------------------
    try:
        df_hold = pd.DataFrame(hrows)

        # Total portfolio value per month
        portfolio_totals = (
            df_hold.groupby("Month")["Current_Amount"]
            .sum()
            .reset_index()
            .rename(columns={"Current_Amount": "Portfolio_Total"})
        )

        df_hold = df_hold.merge(portfolio_totals, on="Month", how="left")

        # Weight %
        df_hold["Weight_Pct"] = (
            100 * df_hold["Current_Amount"] / df_hold["Portfolio_Total"]
        ).round(3)

        # Max weight per month
        df_hold["Max_Weight_Pct"] = df_hold.groupby("Month")["Weight_Pct"].transform(
            "max"
        )

        # Keep ONLY rows that match the max (ties included)
        top_per_month = df_hold[df_hold["Weight_Pct"] == df_hold["Max_Weight_Pct"]]

        # Final columns exactly as requested
        top_per_month = top_per_month[
            [
                "Month",
                "Ticker",
                "Invested_Amount",
                "Current_Amount",
                "Portfolio_Total",
                "Weight_Pct",
            ]
        ]

        OUT_TOP_MONTHLY = RES / "top_weight_stock_per_month.csv"
        top_per_month.to_csv(OUT_TOP_MONTHLY, index=False)

        print(f"[OK] Top-weight stock(s) per month saved to {OUT_TOP_MONTHLY}")

    except Exception as e:
        print(f"[ERROR] Failed to generate top-weight monthly stock CSV: {e}")

    # 3) Ticker counts CSV
    pd.DataFrame(ticker_counts).to_csv(OUT_TICKER_COUNTS, index=False)

    # 4) Liquidation events flattened CSV
    if liq_events_all:
        pd.DataFrame(liq_events_all).to_csv(OUT_LIQ, index=False)
    else:
        pd.DataFrame(
            columns=[
                "Month",
                "Date",
                "Stock",
                "Liquidated_Qty",
                "Liquidated_Amount",
                "Reason",
                "Remaining_Qty",
                "Portfolio_Value_After",
            ]
        ).to_csv(OUT_LIQ, index=False)

    # 5) Final holdings snapshot
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
    pd.DataFrame(final_rows).to_csv(OUT_FIN, index=False)

    # 6) Bank tracker CSV + JSON
    pd.DataFrame(bank_tracker_records).to_csv(OUT_BANK_TRACKER, index=False)
    with open(OUT_BANK_JSON, "w") as f:
        json.dump(bank_tracker_records, f, indent=2)

    print(
        f"Saved: {OUT_JSON}, {OUT_TS}, {OUT_HOLD}, {OUT_FIN}, {OUT_TICKER_COUNTS}, {OUT_LIQ}, {OUT_BANK_TRACKER}, {OUT_BANK_JSON}"
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
    Rebuilt get_params that:
      - Recreates daily portfolio tracking (Portfolio_Value per trading day)
      - Uses the tracker JSON (monthly_tracking_{filter_.replace(' ','_')}.json)
        to get Investment_Date, Holdings and Bank_Balance_After_Investing per month.
      - Produces daily Bank (stepwise constant between investment days),
        Equity = Portfolio_Value + Bank, then computes log-returns and metrics.
      - Saves daily CSV (daily_portfolio_tracking.csv) next to save_file_name's folder.
      - Returns (metrics_dict, daily_df)  <-- Option B
    """
    from pathlib import Path

    # ----------------- load monthly portfolio CSV (same as before) -----------------
    portfolio_monthly = pd.read_csv(save_file_name)

    # keep compatibility with previous "final capital" logic (for print)
    if "Portfolio_Value_After_Investing" in portfolio_monthly.columns:
        strategy_final_capital = portfolio_monthly[
            "Portfolio_Value_After_Investing"
        ].iloc[-1]
    elif "Portfolio_Value" in portfolio_monthly.columns:
        strategy_final_capital = portfolio_monthly["Portfolio_Value"].iloc[-1]
    else:
        strategy_final_capital = portfolio_monthly.iloc[-1].iloc[-1]

    # ----------------- derive path to monthly tracker JSON (produced by tracker()) -----------------
    save_path = Path(save_file_name)
    monthly_json = (
        save_path.parent / f"monthly_tracking_{filter_.replace(' ', '_')}.json"
    )
    if not monthly_json.exists():
        # fallback: try same folder with different case or name
        raise Exception(f"Monthly tracker JSON not found: {monthly_json}")

    with open(monthly_json, "r") as f:
        monthly_recs = json.load(f)

    # ----------------- load price data (master stocks dataframe) -----------------
    # this uses the module-level helper load_stocks_df() defined elsewhere in this file
    price_df = load_stocks_df()

    # ----------------- build daily records using tracker recs -----------------
    nse = get_nse_calendar()
    daily_records = []

    for rec in monthly_recs:
        # Each rec (from tracker) contains:
        # "Investment_Date" (first trading day), "Date" (liquidation day),
        # "Holdings" (list of holdings rows), "Bank_Balance_After_Investing" (bank after invest)
        invest_date_raw = rec.get("Investment_Date")
        liq_date_raw = rec.get("Date")
        holdings = rec.get("Holdings", [])
        bank_after = rec.get("Bank_Balance_After_Investing", None)

        if invest_date_raw is None or liq_date_raw is None:
            # defensively skip badly formed month
            continue

        invest_date = pd.to_datetime(invest_date_raw).date()
        liq_date = pd.to_datetime(liq_date_raw).date()

        # trading days between investment date and liquidation date inclusive
        try:
            sched = nse.schedule(start_date=invest_date, end_date=liq_date)
            trading_days = sched.index.date
        except Exception:
            # fallback to business days if calendar fails
            trading_days = pd.date_range(invest_date, liq_date, freq="B").date

        # For each trading day, sum holding values using last available price on or before day
        for day in trading_days:
            total_value = 0.0
            for h in holdings:
                # holdings rows in tracker were saved as dicts with keys:
                # "Ticker", "Quantity", "Avg_Buy_Price", "Invested_Amount", ...
                ticker = h.get("Ticker")
                qty = h.get("Quantity", 0.0)
                if ticker is None or qty == 0:
                    continue
                p = get_price(price_df, ticker, day)
                if p is None or (isinstance(p, float) and np.isnan(p)):
                    # price not available -> treat as zero for that day
                    continue
                total_value += float(qty) * float(p)

            daily_records.append(
                {
                    "Date": pd.to_datetime(day),
                    "Portfolio_Value": round(total_value, 2),
                    # Bank is stepwise constant: use bank_after for the whole investment->liquidation span
                    "Bank": float(bank_after) if bank_after is not None else np.nan,
                    "Month": rec.get("Month"),
                }
            )

    if not daily_records:
        raise Exception(
            "No daily records could be constructed from monthly tracker JSON."
        )

    daily_df = pd.DataFrame(daily_records)
    # Ensure sorted by date and if duplicate dates exist (overlapping months), sum portfolio values
    daily_df = (
        daily_df.sort_values("Date")
        .groupby("Date", as_index=False)
        .agg({"Portfolio_Value": "sum", "Bank": "first", "Month": "first"})
    )

    # Forward-fill bank and backfill (defensive)
    daily_df["Bank"] = daily_df["Bank"].ffill().bfill()

    # Equity
    daily_df["Equity"] = daily_df["Portfolio_Value"] + daily_df["Bank"]

    # Save daily CSV next to save_file_name (same behavior as before)
    out_daily = save_path.parent / "daily_portfolio_tracking.csv"
    # write Date as ISO string for compatibility
    df_to_save = daily_df.copy()
    df_to_save["Date"] = df_to_save["Date"].dt.strftime("%Y-%m-%d")
    df_to_save.to_csv(out_daily, index=False)

    # ----------------- compute log returns and metrics -----------------
    daily_df = daily_df.sort_values("Date").reset_index(drop=True)
    # ensure numeric
    daily_df["Equity"] = daily_df["Equity"].astype(float)

    # log returns
    daily_df["LogReturns"] = np.log(daily_df["Equity"] / daily_df["Equity"].shift(1))
    log_returns = daily_df["LogReturns"].dropna()

    # final & initial equity for CAGR
    final_equity = daily_df["Equity"].iloc[-1]
    initial_equity = initial_capital

    annualized_returns = (((final_equity / initial_equity) ** (1 / years)) - 1) * 100
    annualized_returns = round(annualized_returns, 2)

    # annualized volatility
    annualized_volatility = float(log_returns.std() * np.sqrt(252))

    # downside vol
    downside = log_returns.copy()
    downside[downside > 0] = 0
    downside_volatility = float(downside.std() * np.sqrt(252))

    # max drawdown
    roll_max = daily_df["Equity"].cummax()
    drawdown = (daily_df["Equity"] / roll_max) - 1
    max_drawdown = abs(drawdown.min()) * 100
    max_drawdown = round(max_drawdown, 2)

    # ratios
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

    # Average Selected Stocks (same as before)
    counts = []
    for date, filters in filter_dict.items():
        if filter_ in filters:
            counts.append(len(filters[filter_]))
    avg_stocks = round(sum(counts) / len(counts), 2) if counts else "N/A"
    min_stocks = min(counts) if counts else "N/A"
    max_stocks = max(counts) if counts else "N/A"

    # print (same format)
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
    print(f"{'Trading Periods':30}: {len(daily_df.index)}")
    print("\n=====================================================\n")

    # return metrics dict and the daily dataframe (Option B)
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
        "Trading Periods": len(daily_df.index),
    }

    return metrics, daily_df


# ----------------- MAIN -----------------
def main():
    print(
        "=== Momentum / Value Portfolio Tracker (Bank-balance + Monthly Lot-wise Liquidation) ==="
    )

    # ---------- Load Data ----------
    df = load_stocks_df()
    invest_flt = load_filter(INVEST_JSON)  # File A: investment plan
    good_flt = load_filter(GOOD_JSON)  # File B: good stocks for liquidation
    months = month_iter(START_MONTH, END_DATE)

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
        months,
        INITIAL_BANK_BALANCE,
        MONTHLY_BANK_WITHDRAW,
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
    years = len(months) / 12.0 if len(months) > 0 else 1.0
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
