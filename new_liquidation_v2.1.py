# new_liquidation_v1.2

import math
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

# ----------------- CONFIG -----------------
STOCKS_CSV = "data/prices.xlsx"

# Strategy selector
STRATEGY = "value"  # "momentum" | "value"
STRATEGY = "momentum"  # "momentum" | "value"

VARIANT = "original"  # "original" | "final"
VARIANT = "final"  # "original" | "final"

# ----------------- SCENARIO LOGIC (UPDATED) -----------------
SCENARIO = f"{STRATEGY}-{VARIANT}"

SCENARIOS = {
    "value-original": {
        "invest_json": "data\\value_original_v2.2.json",
        "good_json": "data\\filtered_outputs7\\Value_new_data_Filter_2_50%.json",
        "output_dir": "output\\value\\strategy-original\\value1",
    },
    "value-final": {
        "invest_json": "data\\value_final_v2.1.json",
        "good_json": "data\\filtered_outputs8\\Value_new_data_Filter_2_50%.json",
        "output_dir": "output\\value\\strategy-final2\\value1",
    },
    "momentum-original": {
        "invest_json": "data\\momentum_final_v2.1.json",
        "good_json": "data\\MO\\Momentum_new_data_Momentum.json",
        "output_dir": "output\\momentum\\strategy-original\\momentum1",
    },
    "momentum-final": {
        "invest_json": "data\\momentum_final_v2.2.json",
        "good_json": "data\\MF2\\Momentum_new_data_Momentum.json",
        "output_dir": "output\\momentum\\strategy-final\\momentum1",
    },
}

cfg = SCENARIOS[SCENARIO]

INVEST_JSON = cfg["invest_json"]
GOOD_JSON = cfg["good_json"]

# ----------------- FILTER -----------------
FILTER_KEY = "Filter_2_50%"
FILTER_KEY = "Momentum"
FILTER_KEY = "ROE_ROCE"
FILTER_KEY = "TMI"
FILTER_KEY = "Filter 3"

# ----------------- SIMULATION RANGE -----------------
START_MONTH = "Jul 2014"
END_DATE = "2025-09-01"

# ----------------- BANK SETTINGS -----------------
INITIAL_BANK_BALANCE = 12000.0
MONTHLY_BANK_WITHDRAW = 1000.0

# ----------------- LIQUIDATION POLICY -----------------
HOLDING_PERIOD_THRESHOLD_DAYS = 365  # 1 year

# ----------------- RESULTS FOLDER -----------------
RES = Path(cfg["output_dir"])
RES.mkdir(parents=True, exist_ok=True)

# Output files
OUT_TS = RES / "portfolio_timeseries_filter3.csv"
OUT_HOLD = RES / "portfolio_timeseries_filter3_holdings.csv"
OUT_FIN = RES / "final_portfolio_filter3.csv"
OUT_JSON = RES / f"monthly_tracking_{FILTER_KEY.replace(' ','_')}.json"
OUT_TICKER_COUNTS = RES / "monthly_ticker_counts.csv"
OUT_CHART = RES / "monthly_ticker_counts_chart.png"
OUT_LIQ = RES / "monthly_liquidation_details.csv"
OUT_BANK_TRACKER = RES / "monthly_bank_tracker.csv"
OUT_BANK_JSON = RES / "monthly_bank_tracker.json"


def load_stocks_df():
    df = pd.read_excel(STOCKS_CSV, engine="openpyxl")
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


def get_price(df, ticker, on_date):
    """Return last available price for ticker on or before on_date, or None if not available."""
    if ticker not in df.columns:
        return None
    ser = df[ticker].dropna()
    sub = ser.index[ser.index <= pd.to_datetime(on_date)]
    return None if len(sub) == 0 else float(ser.loc[sub[-1]])


def safe_div(a, b):
    return a / b if b else 0.0


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
    holding_periods = []  # NEW: to record holding durations for sold lots

    for lot in holdings_entry["lots"]:
        if qty_remaining <= 0:
            # no more to sell
            new_lots.append(lot)
            continue

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
                    "qty": remaining_qty,
                    "price": lot["price"],
                }
            )

            qty_remaining = 0.0

    # update holdings entry
    holdings_entry["lots"] = new_lots
    holdings_entry["qty"] = sum(l["qty"] for l in new_lots)
    holdings_entry["invested"] = sum(l["qty"] * l["price"] for l in new_lots)
    holdings_entry["avg"] = safe_div(holdings_entry["invested"], holdings_entry["qty"])

    return cost_removed, holding_periods


# ----------------- TRACKER with new bank balance & liquidation logic (uses trading days) -----------------
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
            p = get_price(df, t, liq_day)
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
        allowed_liquidation_limit = bank_plus_portfolio / 1

        monthly_liq_total = 0.0
        monthly_liq_events = []

        # ---------- Liquidation logic ----------
        if i >= 1 and portfolio_value_before > 0:
            bad_stocks = [t for t in holdings.keys() if t not in good_stocks]

            bad_details = []
            for t in bad_stocks:
                hs = holding_summaries.get(t)
                if not hs:
                    continue
                p = hs["current_price"]
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    continue
                bad_details.append(
                    {
                        "ticker": t,
                        "qty": hs["qty"],
                        "current_price": p,
                        "current_value": hs["current_value"],
                        "avg": hs["avg"],
                        "return_pct": hs["return_pct"],
                    }
                )

            total_bad_value = sum(b["current_value"] for b in bad_details)

            # --- FULL liquidation ---
            if total_bad_value <= allowed_liquidation_limit + 1e-9:
                for b in bad_details:
                    t = b["ticker"]
                    qty = b["qty"]
                    price = b["current_price"]
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
                        avg_buy_price = safe_div(
                            cost_removed, sum(q for _, q in periods)
                        )
                        ret = safe_div(avg_sell_price, avg_buy_price) - 1.0
                        all_returns.extend([(ret, qty) for _, qty in periods])
                        if t not in per_stock_returns:
                            per_stock_returns[t] = []
                        per_stock_returns[t].extend([(ret, qty) for _, qty in periods])

                    if holdings.get(t, {}).get("qty", 0.0) == 0:
                        holdings.pop(t, None)

                        ev = {
                            "Month": m,
                            "Date": str(liq_day),
                            "Stock": t,
                            "Liquidated_Qty": round(qty, 6),
                            "Liquidated_Amount": round(amount, 2),
                            "Reason": "Not in good list (full)",
                            "Remaining_Qty": round(
                                holdings.get(t, {}).get("qty", 0.0), 6
                            ),
                            "Portfolio_Value_After": None,
                        }
                    monthly_liq_events.append(ev)
                    monthly_liq_total += amount

            # --- PARTIAL liquidation ---
            else:
                bad_sorted = sorted(
                    bad_details, key=lambda x: x["return_pct"]
                )  # worst first
                remaining_to_liquidate = allowed_liquidation_limit
                for b in bad_sorted:
                    if remaining_to_liquidate <= 1e-9:
                        break
                    t = b["ticker"]
                    qty = b["qty"]
                    price = b["current_price"]
                    total_val = qty * price
                    if total_val <= remaining_to_liquidate + 1e-9:
                        amount = total_val
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
                            avg_buy_price = safe_div(
                                cost_removed, sum(q for _, q in periods)
                            )
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
                            "Reason": "Not in good list (full within limit)",
                            "Remaining_Qty": round(
                                holdings.get(t, {}).get("qty", 0.0), 6
                            ),
                            "Portfolio_Value_After": None,
                        }
                        monthly_liq_events.append(ev)
                        monthly_liq_total += amount
                        remaining_to_liquidate -= amount
                    else:
                        qty_to_sell = remaining_to_liquidate / price
                        if qty_to_sell > 0:
                            amount = qty_to_sell * price
                            cost_removed, periods = remove_qty_from_lots(
                                holdings[t], qty_to_sell, liq_day
                            )
                            all_holding_periods.extend(periods)
                            if t not in per_stock_holding_periods:
                                per_stock_holding_periods[t] = []
                            per_stock_holding_periods[t].extend(periods)

                            # --- Return tracking ---
                            if cost_removed > 0:
                                avg_sell_price = price
                                avg_buy_price = safe_div(
                                    cost_removed, sum(q for _, q in periods)
                                )
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
                                "Liquidated_Qty": round(qty_to_sell, 6),
                                "Liquidated_Amount": round(amount, 2),
                                "Reason": "Not in good list (partial to meet limit)",
                                "Remaining_Qty": round(
                                    holdings.get(t, {}).get("qty", 0.0), 6
                                ),
                                "Portfolio_Value_After": None,
                            }
                            monthly_liq_events.append(ev)
                            monthly_liq_total += amount
                            remaining_to_liquidate -= amount
                            break

            if monthly_liq_total > 0:
                tot_liquidated += monthly_liq_total

        # ---------- Portfolio after liquidation ----------
        portfolio_value_after_liq = 0.0
        for t, d in holdings.items():
            p = get_price(df, t, liq_day)
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
                p = get_price(df, t, invest_day)
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    continue
                qty = per_stock / p
                invested = qty * p
                if t not in holdings:
                    holdings[t] = {"qty": 0.0, "invested": 0.0, "avg": 0.0, "lots": []}
                holdings[t]["lots"].append({"month": m, "qty": qty, "price": p})
                holdings[t]["qty"] += qty
                holdings[t]["invested"] += invested
                holdings[t]["avg"] = holdings[t]["invested"] / holdings[t]["qty"]

        tot_invested_from_bank += investable_from_bank

        # ---------- Portfolio after investing ----------
        portfolio_value_after_invest = 0.0
        holdings_rows = []
        for t, d in holdings.items():
            p = get_price(df, t, liq_day)
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


def generate_holding_vs_return_scatter(csv_path):
    """
    Generate a scatter plot: X = Avg_Holding_Days, Y = Avg_Return_Pct.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read {csv_path}: {e}")
        return

    if "Avg_Holding_Days" not in df.columns or "Avg_Return_Pct" not in df.columns:
        print("[WARN] Required columns not found for scatter plot.")
        return

    df = df.dropna(subset=["Avg_Holding_Days", "Avg_Return_Pct"])

    plt.figure(figsize=(10, 6))
    plt.scatter(df["Avg_Holding_Days"], df["Avg_Return_Pct"], alpha=0.7)
    plt.xlabel("Average Holding Period (Days)")
    plt.ylabel("Average Return (%)")
    plt.title("Stock-wise Holding Period vs Returns")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = Path(RES) / "holding_vs_return_scatter.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved to {out_path}")


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
def generate_ticker_count_chart(ticker_counts):
    dfc = pd.DataFrame(ticker_counts)
    if dfc.empty:
        print("[WARN] No ticker count data to plot.")
        return

    fig, ax = plt.subplots(figsize=(16, 6))

    x = list(range(len(dfc)))

    # ---- Plot lines ----
    if "Holdings_Count" in dfc.columns:
        ax.plot(
            x,
            dfc["Holdings_Count"],
            marker="o",
            label="Holdings Count",
            linewidth=2,
        )

    if "Selected_Tickers" in dfc.columns:
        ax.plot(
            x,
            dfc["Selected_Tickers"],
            marker="s",
            label="Selected Tickers",
            linewidth=2,
        )

    # -------------------------------------------------
    # Highlight LOWEST Holdings Count after 13 months
    # -------------------------------------------------
    if "Holdings_Count" in dfc.columns and len(dfc) > 13:
        df_after_13 = dfc.iloc[13:]              # ignore first 13 months
        min_idx = df_after_13["Holdings_Count"].idxmin()
        min_val = dfc.loc[min_idx, "Holdings_Count"]

        # highlight point
        ax.scatter(
            min_idx,
            min_val,
            color="red",
            s=120,
            zorder=5,
            label="Lowest Holdings (after 13 months)",
        )

        # annotate value
        ax.annotate(
            f"{int(min_val)}",
            (min_idx, min_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="red",
        )

    # ---- X axis formatting ----
    step = max(1, len(dfc) // 20)
    ax.set_xticks(range(0, len(dfc), step))
    ax.set_xticklabels(
        [dfc.iloc[i]["Month"] for i in range(0, len(dfc), step)],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Tickers")
    ax.set_title("Monthly Ticker Counts (Selected vs Holdings)")
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
    ax.plot(df["Date"], df["Portfolio_Value"], linewidth=2)
    ax.set_title("Daily Portfolio Value Over Time", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value (₹)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    OUT_DAILY_CHART = Path(RES) / "daily_portfolio_chart.png"
    plt.savefig(OUT_DAILY_CHART, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Daily portfolio chart saved to {OUT_DAILY_CHART}")


def get_params(save_file_name, initial_capital, filter_dict, filter_):
    """
    Rebuilt get_params that:
      - Recreates daily portfolio tracking (Portfolio_Value per trading day)
      - Builds Equity = Portfolio_Value + Bank
      - Computes metrics using CALENDAR time from equity curve
      - initial_capital is treated as lump-sum baseline (₹12,000 as requested)
    """

    from pathlib import Path

    # ----------------- load monthly portfolio CSV -----------------
    portfolio_monthly = pd.read_csv(save_file_name)

    # ----------------- load monthly tracker JSON -----------------
    save_path = Path(save_file_name)
    monthly_json = (
        save_path.parent / f"monthly_tracking_{filter_.replace(' ', '_')}.json"
    )
    if not monthly_json.exists():
        raise Exception(f"Monthly tracker JSON not found: {monthly_json}")

    with open(monthly_json, "r") as f:
        monthly_recs = json.load(f)

    # ----------------- load price data -----------------
    price_df = load_stocks_df()
    nse = get_nse_calendar()

    # ----------------- reconstruct DAILY equity -----------------
    daily_records = []

    for rec in monthly_recs:
        invest_date_raw = rec.get("Investment_Date")
        liq_date_raw = rec.get("Date")
        holdings = rec.get("Holdings", [])
        bank_after = rec.get("Bank_Balance_After_Investing")

        if not invest_date_raw or not liq_date_raw:
            continue

        invest_date = pd.to_datetime(invest_date_raw).date()
        liq_date = pd.to_datetime(liq_date_raw).date()

        try:
            sched = nse.schedule(start_date=invest_date, end_date=liq_date)
            trading_days = sched.index.date
        except Exception:
            trading_days = pd.date_range(invest_date, liq_date, freq="B").date

        for day in trading_days:
            portfolio_value = 0.0
            for h in holdings:
                ticker = h.get("Ticker")
                qty = h.get("Quantity", 0.0)
                if not ticker or qty == 0:
                    continue
                p = get_price(price_df, ticker, day)
                if p is None or np.isnan(p):
                    continue
                portfolio_value += qty * p

            daily_records.append(
                {
                    "Date": pd.to_datetime(day),
                    "Portfolio_Value": round(portfolio_value, 2),
                    "Bank": float(bank_after) if bank_after is not None else 0.0,
                }
            )

    if not daily_records:
        raise Exception("No daily equity could be reconstructed.")

    daily_df = (
        pd.DataFrame(daily_records)
        .groupby("Date", as_index=False)
        .agg({"Portfolio_Value": "sum", "Bank": "first"})
        .sort_values("Date")
        .reset_index(drop=True)
    )

    daily_df["Equity"] = daily_df["Portfolio_Value"] + daily_df["Bank"]

    # ----------------- SAVE DAILY CSV -----------------
    out_daily = save_path.parent / "daily_portfolio_tracking.csv"
    daily_df.assign(Date=daily_df["Date"].dt.strftime("%Y-%m-%d")).to_csv(
        out_daily, index=False
    )

    # ----------------- METRICS -----------------
    daily_df["LogReturns"] = np.log(
        daily_df["Equity"] / daily_df["Equity"].shift(1)
    )
    log_returns = daily_df["LogReturns"].dropna()

    # CORRECT YEARS (calendar-based)
    start_date = daily_df["Date"].iloc[0]
    end_date = daily_df["Date"].iloc[-1]
    years = (end_date - start_date).days / 365.25
    years = max(years, 1e-6)

    initial_equity = initial_capital
    final_equity = daily_df["Equity"].iloc[-1]

    annualized_returns = ((final_equity / initial_equity) ** (1 / years) - 1) * 100
    annualized_returns = round(annualized_returns, 2)

    annualized_volatility = float(log_returns.std() * np.sqrt(252))

    downside = log_returns.copy()
    downside[downside > 0] = 0
    downside_volatility = float(downside.std() * np.sqrt(252))

    roll_max = daily_df["Equity"].cummax()
    max_drawdown = abs((daily_df["Equity"] / roll_max - 1).min()) * 100
    max_drawdown = round(max_drawdown, 2)

    sharpe = (
        (annualized_returns / 100) / annualized_volatility
        if annualized_volatility > 0
        else float("inf")
    )
    sortino = (
        (annualized_returns / 100) / downside_volatility
        if downside_volatility > 0
        else float("inf")
    )
    calmar = (
        (annualized_returns / 100) / (max_drawdown / 100)
        if max_drawdown > 0
        else float("inf")
    )

    # ----------------- STOCK COUNT STATS -----------------
    counts = [len(v[filter_]) for v in filter_dict.values() if filter_ in v]
    avg_stocks = round(sum(counts) / len(counts), 2) if counts else "N/A"

    # ----------------- PRINT -----------------
    print("\n================ PERFORMANCE SUMMARY ================\n")
    print(f"{'Initial Capital':30}: ₹{initial_equity:,.2f}")
    print(f"{'Final Capital':30}: ₹{final_equity:,.2f}")
    print(f"{'Annualized Returns (%)':30}: {annualized_returns:.2f}%")
    print(f"{'Annualized Volatility':30}: {annualized_volatility:.6f}")
    print(f"{'Downside Volatility':30}: {downside_volatility:.6f}")
    print(f"{'Max Drawdown (%)':30}: {max_drawdown:.2f}%")
    print(f"{'Sharpe Ratio':30}: {round(sharpe,2)}")
    print(f"{'Sortino Ratio':30}: {round(sortino,2)}")
    print(f"{'Calmar Ratio':30}: {round(calmar,2)}")
    print(f"{'Years Simulated':30}: {years:.2f}")
    print("\n=====================================================\n")

    return {
        "Initial Capital": initial_equity,
        "Final Capital": round(final_equity, 2),
        "Annualized Returns (%)": annualized_returns,
        "Annualized Volatility": round(annualized_volatility, 6),
        "Downside Volatility": round(downside_volatility, 6),
        "Max Drawdown (%)": max_drawdown,
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Calmar Ratio": round(calmar, 2),
        "Years Simulated": round(years, 2),
    }, daily_df


# ----------------- MAIN -----------------
def main():
    print(
        "=== Momentum / Value Portfolio Tracker (Bank-balance + Monthly Liquidation) ==="
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
    generate_daily_portfolio_chart(os.path.join(RES, "daily_portfolio_tracking.csv"))

    # ---------- Scatter Plot: Avg Holding Days vs Returns ----------
    scatter_csv = RES / "avg_holding_period_per_stock.csv"
    if scatter_csv.exists():
        generate_holding_vs_return_scatter(scatter_csv)
    else:
        print("[WARN] No per-stock holding period CSV found for scatter plot.")

    # ---------- Performance Metrics ----------
    params, daily_df = get_params(
        OUT_TS,
        INITIAL_BANK_BALANCE,   # ₹12,000 as requested
        invest_flt,
        FILTER_KEY,
    )


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
