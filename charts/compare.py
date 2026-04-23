import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# === Load Data ===
with open(r"E:\AA\Liquidation\data\value.json", "r") as f:
    old_data = json.load(f)

with open(r"E:\AA\Liquidation\data\value_stock_exp_01_om.json", "r") as f:
    new_data = json.load(f)


# === Helper to sort month-year ===
def parse_month_year(s):
    return datetime.strptime(s, "%b %Y")


# === Get all months ===
months = sorted(set(old_data.keys()) | set(new_data.keys()), key=parse_month_year)

# === Get all filters ===
filters = set()
for m in months:
    filters |= set(old_data.get(m, {}).keys())
    filters |= set(new_data.get(m, {}).keys())

filters = sorted(filters)

# === Build dataframe per filter ===
filter_dfs = {}

for f_name in filters:
    old_cnt, new_cnt, union_cnt, inter_cnt = [], [], [], []

    for m in months:
        old_set = set(old_data.get(m, {}).get(f_name, []))
        new_set = set(new_data.get(m, {}).get(f_name, []))

        old_cnt.append(len(old_set))
        new_cnt.append(len(new_set))
        union_cnt.append(len(old_set | new_set))
        inter_cnt.append(len(old_set & new_set))

    filter_dfs[f_name] = pd.DataFrame(
        {
            "Month": months,
            "Old Unique": old_cnt,
            "New Unique": new_cnt,
            "Union": union_cnt,
            "Intersection": inter_cnt,
        }
    )

# === Create Figure ===
fig = go.Figure()

# === Add traces (4 per filter) ===
for i, (f_name, df) in enumerate(filter_dfs.items()):
    visible = i == 0  # show first filter by default

    fig.add_trace(
        go.Scatter(
            x=df["Month"],
            y=df["Old Unique"],
            mode="lines+markers",
            name="Old Unique",
            visible=visible,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Month"],
            y=df["New Unique"],
            mode="lines+markers",
            name="New Unique",
            visible=visible,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Month"],
            y=df["Union"],
            mode="lines+markers",
            name="Union",
            visible=visible,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Month"],
            y=df["Intersection"],
            mode="lines+markers",
            name="Intersection",
            visible=visible,
        )
    )

# === Dropdown buttons ===
buttons = []
trace_per_filter = 4

for i, f_name in enumerate(filters):
    visibility = [False] * (len(filters) * trace_per_filter)
    start = i * trace_per_filter
    visibility[start : start + trace_per_filter] = [True] * trace_per_filter

    buttons.append(
        dict(
            label=f_name,
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"Month-wise Old vs New Comparison – {f_name}"},
            ],
        )
    )

# === Layout ===
fig.update_layout(
    title=f"Month-wise Old vs New Comparison – {filters[0]}",
    xaxis_title="Month",
    yaxis_title="Number of Stocks",
    updatemenus=[
        {
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 1.15,
            "y": 1.1,
        }
    ],
    width=1200,
    height=600,
)

fig.show()
fig.write_html("value_old_vs_new_all_filters.html")
