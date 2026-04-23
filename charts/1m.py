import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Folder that contains the CSVs
FOLDER = Path(r"E:\AA\Liquidation\charts")

# All 4 versions
FILE_NAMES = [
    "daily_portfolio_tracking_m1.csv",
    "daily_portfolio_tracking_m2.csv",
    "daily_portfolio_tracking_m3.csv",
    "daily_portfolio_tracking_m4.csv",
]


def load_and_prepare(path):
    df = pd.read_csv(path)

    # Strict date parsing (your format = YYYY-MM-DD)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="raise")

    df = df.sort_values("Date")

    if "Equity" not in df.columns:
        raise ValueError(f"Missing 'Equity' column in {path}")

    # Normalize (1 rupee starting point)
    start_value = df["Equity"].iloc[0]
    df["Equity_Normalized"] = df["Equity"] / start_value

    return df


def plot_equity_all_versions(output_path="combined_equity_chart.png"):
    plt.figure(figsize=(16, 6))

    # Light, clean color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red

    for i, file in enumerate(FILE_NAMES):
        csv_path = FOLDER / file

        try:
            df = load_and_prepare(csv_path)
        except Exception as e:
            print(f"[ERROR loading {file}] {e}")
            continue

        label = file.replace("daily_portfolio_tracking_", "").replace(".csv", "")

        # Thin clean line (no markers)
        plt.plot(
            df["Date"],
            df["Equity_Normalized"],
            linewidth=1.3,
            color=colors[i],
            label=label,
        )

    plt.title("1-Rupee Normalized Equity Comparison (v1, v2, v3, v4)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Value of 1 Rupee", fontsize=12)

    plt.grid(True, alpha=0.15)  # light grid
    plt.legend(fontsize=12)

    plt.tight_layout(pad=2)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Combined chart saved: {output_path}")


if __name__ == "__main__":
    OUTPUT = r"E:\AA\Liquidation\charts\equity_comparison_all_versions_Momentum.png"
    plot_equity_all_versions(OUTPUT)
