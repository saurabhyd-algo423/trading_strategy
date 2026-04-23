import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load CSV (Left Table)
# -----------------------------
csv_path = "daily_portfolio_tracking.csv"
df_csv = pd.read_csv(csv_path, parse_dates=["Date"])

# -----------------------------
# 2. Load JSON (Right Table)
# -----------------------------
json_path = "Momentum_strategy_performance_Yash_v1_timeseriesFilter 3.json"
df_json = pd.read_json(json_path)

# Rename for clarity
df_json.rename(columns={"Final Investment": "momentum"}, inplace=True)
df_json["Date"] = pd.to_datetime(df_json["Date"])

# -----------------------------
# 3. Left Merge on Date
# -----------------------------
df_merged = pd.merge(df_csv, df_json[["Date", "momentum"]], on="Date", how="left")

# -----------------------------
# 4. Create New Columns
# -----------------------------
df_merged["ValueMomentum"] = df_merged["Equity"] + df_merged["momentum"]
df_merged["VM"] = df_merged["ValueMomentum"] / 24000

# -----------------------------
# 5. Save New CSV
# -----------------------------
output_csv = "merged_value_momentum.csv"
df_merged.to_csv(output_csv, index=False)

print(f"✅ New CSV saved as: {output_csv}")

# -----------------------------
# 6. Plot 1-Rupee Chart
# -----------------------------
plt.figure(figsize=(14, 7))

plt.plot(
    df_merged["Date"],
    df_merged["VM"],
    linewidth=2.5,
    label="Value Momentum (1 Rupee Chart)",
)

plt.title("Value Momentum – 1 Rupee Growth Chart", fontsize=16, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Value (Base = ₹1)", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()
