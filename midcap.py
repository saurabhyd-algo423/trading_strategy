import pandas as pd
import numpy as np

def compute_yearly_full_metrics(csv_file):
    df = pd.read_csv(csv_file)
    df = df
    # --- PREP ---
    df["NIFTY MIDCAP 100"] = pd.to_numeric(df["NIFTY MIDCAP 100"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Log returns
    df["LogRet"] = np.log(df["NIFTY MIDCAP 100"] / df["NIFTY MIDCAP 100"].shift(1))

    df["year"] = df["Date"].dt.year

    results = []

    for year, grp in df.groupby("year"):
        grp = grp["LogRet"].dropna().copy()

        if len(grp) == 0:
            continue

        log_ret = grp

        # --- RETURNS ---
        yearly_log_return = log_ret.sum()
        yearly_return = np.exp(yearly_log_return) - 1

        mean_log = log_ret.mean()
        ann_return = np.exp(mean_log * 252) - 1

        # --- VOLATILITY ---
        vol = log_ret.std() * np.sqrt(252)

        # --- DOWNSIDE VOL ---
        downside = log_ret[log_ret < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0

        # --- SHARPE ---
        std = log_ret.std()
        sharpe = (mean_log / std * np.sqrt(252)) if std != 0 else 0

        # --- MAX DRAWDOWN ---
        cum_returns = np.exp(log_ret.cumsum())  # equity curve
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_dd = drawdown.min()

        results.append({
            "year": year,
            "yearly_log_return": yearly_log_return,
            "yearly_return": yearly_return,
            "annualized_return": ann_return,
            "volatility": vol,
            "downside_volatility": downside_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "days": len(grp)
        })

    return pd.DataFrame(results)
if __name__ == "__main__":
    csv_file = "data/MidCap_and_SmallCap.csv"
    
    yearly_df = compute_yearly_full_metrics(csv_file)

    print(yearly_df)

    # Save output (optional)
    yearly_df.to_csv("output/Prabhav_yearly_performance_midcap_index.csv", index=False)
