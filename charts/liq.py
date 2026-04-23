import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# 1. Read CSV file
# =========================
file_path = r"E:\AA\Liquidation\charts\portfolio_timeseries_filter3_12_limit.csv"
df = pd.read_csv(file_path)

# =========================
# 2. Date parsing & sorting
# =========================
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# =========================
# 3. Create figure
# =========================
plt.figure(figsize=(14, 7))

# =========================
# 4. Plot required lines (FROM FILE ONLY)
# =========================

# Portfolio value before liquidity
plt.plot(
    df["Date"],
    df["Portfolio_Value_Before_Liquidity"],
    label="Portfolio Before Liquidity"
)

# Total liquidation limit (OLD LIMIT FROM FILE)
plt.plot(
    df["Date"],
    df["Total_Liquidation_Limit"],
    label="Total Liquidation Limit"
)

# Total liquidated
plt.plot(
    df["Date"],
    df["Total_Liquidated"],
    label="Total Liquidated"
)

# =========================
# 5. X-axis formatting (MORE MONTHS)
# =========================
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)

# =========================
# 6. Labels, legend, grid
# =========================
plt.xlabel("Month")
plt.ylabel("Amount")
plt.title("Portfolio vs Liquidation & Limit (Month-wise)")
plt.legend()
plt.grid(True, alpha=0.3)

# =========================
# 7. Layout & show
# =========================
plt.tight_layout()
plt.show()
