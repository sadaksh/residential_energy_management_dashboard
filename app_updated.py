import io
from typing import Dict, List, Tuple
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Apartment Energy Dashboard", layout="wide")


CATEGORY_OPTIONS = [
    "Cooling",
    "Water Heating",
    "Lighting",
    "Refrigeration",
    "Cooking",
    "Laundry",
    "Fans",
    "Plug / Miscellaneous",
    "Others",
]


def default_channel_threshold(channel: str) -> float:
    c = str(channel).strip().lower()

    def has_pattern(patterns):
        return any(re.search(p, c) for p in patterns)

    # Laundry first so "washing machine" does not get caught by AC
    if has_pattern([r"\bwashing machine\b", r"\bwasher\b", r"\blaundry\b"]):
        return 0.05

    if has_pattern([r"\bgeyser\b", r"\bwater heater\b", r"\bheater\b"]):
        return 0.30

    if has_pattern([r"\blight\b", r"\blights\b", r"\blighting\b", r"\blamp\b"]):
        return 0.01

    if has_pattern([r"\bfridge\b", r"\brefrigerator\b", r"\bfreezer\b"]):
        return 0.03

    if has_pattern([r"\boven\b", r"\bmicrowave\b", r"\binduction\b", r"\bcooking\b", r"\bkitchen\b"]):
        return 0.10

    if has_pattern([r"\bfan\b", r"\bfans\b"]):
        return 0.03

    if has_pattern([r"\bac\b", r"\bair conditioner\b", r"\bcooling\b"]):
        return 0.15

    if has_pattern([
        r"\bplug\b", r"\bpower\b", r"\bspare\b", r"\bsocket\b", r"\bmisc\b",
        r"\btv\b", r"\bcomputer\b", r"\bstudy plug\b", r"\bunknown\b",
        r"\bwifi\b", r"\bpoint\b",
    ]):
        return 0.02

    return 0.02



def infer_category(channel: str) -> str:
    c = str(channel).strip().lower()

    def has_pattern(patterns):
        return any(re.search(p, c) for p in patterns)

    # Check more specific categories first
    if has_pattern([r"\bwashing machine\b", r"\bwasher\b", r"\blaundry\b"]):
        return "Laundry"

    if has_pattern([r"\bgeyser\b", r"\bwater heater\b", r"\bheater\b"]):
        return "Water Heating"

    if has_pattern([r"\blight\b", r"\blights\b", r"\blighting\b", r"\blamp\b"]):
        return "Lighting"

    if has_pattern([r"\bfridge\b", r"\brefrigerator\b", r"\bfreezer\b"]):
        return "Refrigeration"

    if has_pattern([r"\boven\b", r"\bmicrowave\b", r"\binduction\b", r"\bcooking\b", r"\bkitchen\b"]):
        return "Cooking"

    if has_pattern([r"\bfan\b", r"\bfans\b"]):
        return "Fans"

    # AC should be matched as a standalone word, not inside another word like "machine"
    if has_pattern([r"\bac\b", r"\bair conditioner\b", r"\bcooling\b"]):
        return "Cooling"

    if has_pattern([
        r"\bplug\b", r"\bpower\b", r"\bspare\b", r"\bsocket\b", r"\bmisc\b",
        r"\btv\b", r"\bcomputer\b", r"\bstudy plug\b", r"\bunknown\b",
        r"\bwifi\b", r"\bpoint\b",
    ]):
        return "Plug / Miscellaneous"

    return "Others"



def clean_channel_name(channel: str, apartment_name: str = "") -> str:
    ch = str(channel).strip()

    if apartment_name:
        ap = re.escape(apartment_name.strip())
        ch = re.sub(ap, "", ch, flags=re.IGNORECASE)

    ch = re.sub(r"^\s*[-_:|]+\s*", "", ch)
    ch = re.sub(r"\s*[-_:|]+\s*$", "", ch)
    ch = re.sub(r"\s+", " ", ch).strip()

    return ch



def detect_timestamp_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    return candidates[0] if candidates else df.columns[0]



def auto_detect_interval_minutes(ts: pd.Series) -> float:
    ts = ts.dropna().sort_values().drop_duplicates()
    if len(ts) < 2:
        return 5.0
    diff = ts.diff().dropna().dt.total_seconds() / 60.0
    if diff.empty:
        return 5.0
    val = float(diff.median())
    return 5.0 if val <= 0 else val



def load_raw_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file format. Upload CSV or Excel.")



def preprocess_raw(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, List[str]]:
    df = df.copy()
    ts_col = detect_timestamp_column(df)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    non_ts_cols = [c for c in df.columns if c != ts_col]

    for c in non_ts_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    numeric_cols = [c for c in non_ts_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]

    # Force all numeric values to absolute values
    df[numeric_cols] = df[numeric_cols].abs()

    return df, ts_col, numeric_cols



def build_long_table(
    df: pd.DataFrame,
    ts_col: str,
    load_cols: List[str],
    category_map: pd.DataFrame,
    interval_minutes: float,
    channel_threshold_map: Dict[str, float],
) -> pd.DataFrame:
    long_df = df[[ts_col] + load_cols].melt(
        id_vars=[ts_col],
        value_vars=load_cols,
        var_name="Channel",
        value_name="kW",
    )

    long_df["kW"] = pd.to_numeric(long_df["kW"], errors="coerce").fillna(0).abs()
    long_df["Energy_kWh"] = long_df["kW"] * interval_minutes / 60.0
    long_df["Date"] = long_df[ts_col].dt.date
    long_df["Hour"] = long_df[ts_col].dt.hour
    long_df["YearMonth"] = long_df[ts_col].dt.to_period("M").astype(str)
    long_df["DayType"] = np.where(long_df[ts_col].dt.weekday < 5, "Weekday", "Weekend")

    # category mapping is only for grouping after channel-level calculations
    long_df = long_df.merge(category_map, on="Channel", how="left")
    long_df["Category"] = long_df["Category"].fillna("Others")

    # apply runtime threshold by CHANNEL
    long_df["Runtime_Threshold_kW"] = long_df["Channel"].map(channel_threshold_map).fillna(0.02)

    # channel-level runtime flag
    long_df["Runtime_Flag"] = (long_df["kW"] > long_df["Runtime_Threshold_kW"]).astype(int)

    # runtime in hours for each timestamp row
    long_df["Runtime_hr"] = long_df["Runtime_Flag"] * interval_minutes / 60.0

    return long_df



def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.title("Apartment Energy Consumption Dashboard")
st.caption("All numeric values are converted to absolute values before analysis.")

uploaded_file = st.file_uploader("Upload apartment data file", type=["csv", "xlsx", "xls"])

if not uploaded_file:
    st.info("Upload the apartment CSV/Excel file to begin.")
    st.stop()

raw_df = load_raw_file(uploaded_file)
df, ts_col, numeric_cols = preprocess_raw(raw_df)

if not numeric_cols:
    st.error("No numeric load columns found after cleaning.")
    st.stop()

default_phase_cols = [c for c in numeric_cols if "phase" in c.lower()]

with st.sidebar:
    st.header("Configuration")

    apartment_name = st.text_input("Apartment name", value="Lakeside K 502")
    occupancy = st.number_input("Occupancy", min_value=1, value=7, step=1)

    detected_interval = auto_detect_interval_minutes(df[ts_col])
    interval_minutes = st.number_input("Interval (minutes)", min_value=1.0, value=float(detected_interval), step=1.0)

    phase_cols = st.multiselect(
        "Phase total columns to exclude from end-use analysis",
        options=numeric_cols,
        default=default_phase_cols,
    )

    default_load_cols = [c for c in numeric_cols if c not in phase_cols]
    load_cols = st.multiselect(
        "Load columns for apartment analysis",
        options=numeric_cols,
        default=default_load_cols,
    )

    st.subheader("Benchmarks")
    benchmark_kwh_day = st.number_input("Benchmark: kWh/day", min_value=0.0, value=8.0, step=0.1)
    benchmark_kwh_person_day = st.number_input("Benchmark: kWh/person/day", min_value=0.0, value=1.1, step=0.1)
    benchmark_peak_kw = st.number_input("Benchmark: Peak load (kW)", min_value=0.0, value=5.0, step=0.1)

if not load_cols:
    st.error("Select at least one load column.")
    st.stop()

# Build category map with editable categories
suggested_map = pd.DataFrame({
    "Channel": load_cols,
    "Category": [infer_category(c) for c in load_cols],
}).sort_values("Channel").reset_index(drop=True)

st.subheader("Channel to Category Mapping")
edited_map = st.data_editor(
    suggested_map,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Channel": st.column_config.TextColumn("Channel", disabled=True),
        "Category": st.column_config.SelectboxColumn("Category", options=CATEGORY_OPTIONS, required=True),
    },
    hide_index=True,
)

# Runtime thresholds must be defined per channel BEFORE long_df is built
st.subheader("Runtime Thresholds by Channel (kW)")
channel_threshold_map: Dict[str, float] = {}
with st.expander("Edit channel thresholds", expanded=False):
    for i, ch in enumerate(load_cols):
        channel_threshold_map[ch] = st.number_input(
            label=f"{ch}",
            min_value=0.0,
            value=float(default_channel_threshold(ch)),
            step=0.01,
            key=f"thr_channel_{i}",
        )

st.subheader("Applied Runtime Thresholds by Channel")
threshold_df = pd.DataFrame({
    "Channel": list(channel_threshold_map.keys()),
    "Runtime Threshold (kW)": list(channel_threshold_map.values()),
})
st.dataframe(threshold_df, use_container_width=True)

long_df = build_long_table(
    df=df,
    ts_col=ts_col,
    load_cols=load_cols,
    category_map=edited_map,
    interval_minutes=interval_minutes,
    channel_threshold_map=channel_threshold_map,
)
long_df["Channel_Clean"] = long_df["Channel"].apply(lambda x: clean_channel_name(x, apartment_name))

plot_channels_df = long_df.copy()
plot_channels_df["Power_W"] = plot_channels_df["kW"] * 1000

wide_df = df[[ts_col] + load_cols].copy()
wide_df["Total_Load_kW"] = wide_df[load_cols].sum(axis=1)
wide_df["Total_Energy_kWh"] = wide_df["Total_Load_kW"] * interval_minutes / 60.0
wide_df["Date"] = wide_df[ts_col].dt.date
wide_df["Hour"] = wide_df[ts_col].dt.hour
wide_df["YearMonth"] = wide_df[ts_col].dt.to_period("M").astype(str)
wide_df["DayType"] = np.where(wide_df[ts_col].dt.weekday < 5, "Weekday", "Weekend")

if phase_cols:
    wide_df["Phase_Total_kW"] = df[phase_cols].sum(axis=1)
    wide_df["Residual_kW"] = (wide_df["Phase_Total_kW"] - wide_df["Total_Load_kW"]).abs()

# Summary metrics
daily_energy = wide_df.groupby("Date", as_index=False)["Total_Energy_kWh"].sum()
category_energy = long_df.groupby("Category", as_index=False)["Energy_kWh"].sum().sort_values("Energy_kWh", ascending=False)
hourly_profile = wide_df.groupby(["Hour", "DayType"], as_index=False)["Total_Load_kW"].mean()
monthly_category = long_df.groupby(["YearMonth", "Category"], as_index=False)["Energy_kWh"].sum()
runtime_by_category = long_df.groupby("Category", as_index=False)["Runtime_hr"].sum().sort_values("Runtime_hr", ascending=False)
runtime_by_channel = long_df.groupby("Channel_Clean", as_index=False)["Runtime_hr"].sum().sort_values("Runtime_hr", ascending=False)

actual_total_kwh = float(wide_df["Total_Energy_kWh"].sum())
actual_avg_kwh_day = float(daily_energy["Total_Energy_kWh"].mean()) if not daily_energy.empty else 0.0
actual_avg_load = float(wide_df["Total_Load_kW"].mean()) if not wide_df.empty else 0.0
actual_peak_kw = float(wide_df["Total_Load_kW"].max()) if not wide_df.empty else 0.0
actual_kwh_person_day = actual_avg_kwh_day / occupancy if occupancy > 0 else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Apartment", apartment_name)
col2.metric("Total Energy (kWh)", f"{actual_total_kwh:,.2f}")
col3.metric("Avg Daily Energy (kWh/day)", f"{actual_avg_kwh_day:,.2f}")
col4.metric("Avg Load (kW)", f"{actual_avg_load:,.2f}")
col5.metric("Peak Load (kW)", f"{actual_peak_kw:,.2f}")

st.markdown("---")

st.subheader("Power Consumption for All Channels")
available_channels = sorted(plot_channels_df["Channel_Clean"].dropna().unique().tolist())
selected_channels = st.multiselect(
    "Select channels to display",
    options=available_channels,
    default=available_channels,
)
channel_plot_df = plot_channels_df[
    plot_channels_df["Channel_Clean"].isin(selected_channels)
][[ts_col, "Channel_Clean", "Power_W"]].copy()

fig_all_channels = px.line(
    channel_plot_df,
    x=ts_col,
    y="Power_W",
    color="Channel_Clean",
    title="Power Consumption for All Channels",
)
fig_all_channels.update_layout(
    xaxis_title="Timestamp",
    yaxis_title="Power Consumption (Watts)",
    legend_title="Channels",
)
st.plotly_chart(fig_all_channels, use_container_width=True)

st.subheader("Total Daily Energy Trend")
fig_daily = px.line(
    daily_energy,
    x="Date",
    y="Total_Energy_kWh",
    markers=True,
    title="Daily Energy Consumption",
)
fig_daily.update_layout(xaxis_title="Date", yaxis_title="kWh/day")
st.plotly_chart(fig_daily, use_container_width=True)

st.subheader("End-Use Share of Total Consumption")
c1, c2 = st.columns(2)
fig_enduse_bar = px.bar(
    category_energy,
    x="Energy_kWh",
    y="Category",
    orientation="h",
    title="Energy by End-Use Category",
)
fig_enduse_bar.update_layout(xaxis_title="Energy (kWh)", yaxis_title="")
c1.plotly_chart(fig_enduse_bar, use_container_width=True)

fig_enduse_donut = px.pie(
    category_energy,
    names="Category",
    values="Energy_kWh",
    hole=0.5,
    title="Category Share",
)
c2.plotly_chart(fig_enduse_donut, use_container_width=True)

# 4. Date vs hour heatmap - channel selector
st.subheader("Date vs Hour Heatmap")

if "Channel_Clean" not in long_df.columns:
    long_df["Channel_Clean"] = long_df["Channel"].apply(lambda x: clean_channel_name(x, apartment_name))

available_channels = sorted(long_df["Channel_Clean"].dropna().unique().tolist())

heatmap_metric = st.radio(
    "Heatmap metric",
    ["Energy (kWh)", "Average Power (kW)"],
    horizontal=True
)

channel_option = st.selectbox(
    "Select channel",
    options=["Show all channels"] + available_channels,
    index=0
)

def build_channel_heatmap(ch_df, channel_name, metric):
    if metric == "Energy (kWh)":
        heatmap_df = (
            ch_df.groupby(["Date", "Hour"], as_index=False)["Energy_kWh"]
            .sum()
            .pivot(index="Date", columns="Hour", values="Energy_kWh")
            .fillna(0)
        )
        color_label = "kWh"
        title = f"Date vs Hour Heatmap - {channel_name} (Energy)"
    else:
        heatmap_df = (
            ch_df.groupby(["Date", "Hour"], as_index=False)["kW"]
            .mean()
            .pivot(index="Date", columns="Hour", values="kW")
            .fillna(0)
        )
        color_label = "kW"
        title = f"Date vs Hour Heatmap - {channel_name} (Average Power)"

    fig = px.imshow(
        heatmap_df,
        aspect="auto",
        labels=dict(x="Hour", y="Date", color=color_label),
        title=title,
        color_continuous_scale="YlOrRd",
    )
    st.plotly_chart(fig, use_container_width=True)

if channel_option == "Show all channels":
    for ch in available_channels:
        with st.expander(f"Heatmap - {ch}", expanded=False):
            ch_df = long_df[long_df["Channel_Clean"] == ch].copy()
            build_channel_heatmap(ch_df, ch, heatmap_metric)
else:
    ch_df = long_df[long_df["Channel_Clean"] == channel_option].copy()
    build_channel_heatmap(ch_df, channel_option, heatmap_metric)
    
st.subheader("Average Hourly Load Profile")
fig_hourly = px.line(
    hourly_profile,
    x="Hour",
    y="Total_Load_kW",
    color="DayType",
    markers=True,
    title="Average Hourly Load Profile",
)
fig_hourly.update_layout(xaxis_title="Hour of Day", yaxis_title="Average Load (kW)")
st.plotly_chart(fig_hourly, use_container_width=True)

st.subheader("Top Peak Demand Events")
top_n = st.slider("Number of peak events", min_value=5, max_value=25, value=10, step=1)
peak_df = wide_df[[ts_col] + load_cols + ["Total_Load_kW"]].copy()
peak_df["Dominant_Channel"] = peak_df[load_cols].idxmax(axis=1)
peak_df["Dominant_kW"] = peak_df[load_cols].max(axis=1)
peak_df["Dominant_Channel_Clean"] = peak_df["Dominant_Channel"].apply(lambda x: clean_channel_name(x, apartment_name))
peak_df = peak_df.sort_values("Total_Load_kW", ascending=False).head(top_n).copy()
peak_df["Rank"] = range(1, len(peak_df) + 1)

fig_peaks = px.bar(
    peak_df.sort_values("Rank"),
    x="Rank",
    y="Total_Load_kW",
    hover_data=[ts_col, "Dominant_Channel_Clean", "Dominant_kW"],
    title="Top Peak Demand Events",
)
fig_peaks.update_layout(xaxis_title="Peak Rank", yaxis_title="Total Load (kW)")
st.plotly_chart(fig_peaks, use_container_width=True)
st.dataframe(
    peak_df[[ts_col, "Total_Load_kW", "Dominant_Channel_Clean", "Dominant_kW"]].rename(
        columns={
            ts_col: "Timestamp",
            "Total_Load_kW": "Peak Load (kW)",
            "Dominant_Channel_Clean": "Dominant Channel",
            "Dominant_kW": "Dominant Channel Load (kW)",
        }
    ),
    use_container_width=True,
)

st.subheader("Base Load Trend")
night_start, night_end = st.slider("Night hours for base load", min_value=0, max_value=23, value=(2, 5))
night_df = wide_df[(wide_df["Hour"] >= night_start) & (wide_df["Hour"] <= night_end)].copy()
base_method = st.radio("Base load method", ["Minimum night load", "Average night load"], horizontal=True)

if base_method == "Minimum night load":
    base_load_df = night_df.groupby("Date", as_index=False)["Total_Load_kW"].min()
    base_load_df.rename(columns={"Total_Load_kW": "Base_Load_kW"}, inplace=True)
else:
    base_load_df = night_df.groupby("Date", as_index=False)["Total_Load_kW"].mean()
    base_load_df.rename(columns={"Total_Load_kW": "Base_Load_kW"}, inplace=True)

fig_base = px.line(
    base_load_df,
    x="Date",
    y="Base_Load_kW",
    markers=True,
    title="Base Load Trend",
)
fig_base.update_layout(xaxis_title="Date", yaxis_title="Base Load (kW)")
st.plotly_chart(fig_base, use_container_width=True)

st.subheader("Runtime Pattern (Computed Channel-wise)")
st.caption("Runtime thresholds are applied to each channel first. Category runtime is the sum of constituent channel runtimes.")
runtime_view = st.radio("Runtime view", ["Channel", "Category"], horizontal=True)

if runtime_view == "Category":
    fig_runtime = px.bar(
        runtime_by_category,
        x="Runtime_hr",
        y="Category",
        orientation="h",
        title="Runtime by Category",
    )
else:
    fig_runtime = px.bar(
        runtime_by_channel.head(20),
        x="Runtime_hr",
        y="Channel_Clean",
        orientation="h",
        title="Top 20 Runtime Channels",
    )

fig_runtime.update_layout(xaxis_title="Runtime (hours)", yaxis_title="")
st.plotly_chart(fig_runtime, use_container_width=True)

st.subheader("Monthly Category Trend")
fig_monthly = px.bar(
    monthly_category,
    x="YearMonth",
    y="Energy_kWh",
    color="Category",
    title="Monthly Category Trend",
)
fig_monthly.update_layout(xaxis_title="Month", yaxis_title="Energy (kWh)")
st.plotly_chart(fig_monthly, use_container_width=True)

st.subheader("Comparison Against Benchmark")
benchmark_df = pd.DataFrame(
    {
        "Metric": ["kWh/day", "kWh/person/day", "Peak kW"],
        "Actual": [actual_avg_kwh_day, actual_kwh_person_day, actual_peak_kw],
        "Benchmark": [benchmark_kwh_day, benchmark_kwh_person_day, benchmark_peak_kw],
    }
)
benchmark_df["Variance"] = benchmark_df["Actual"] - benchmark_df["Benchmark"]
benchmark_long = benchmark_df.melt(id_vars="Metric", value_vars=["Actual", "Benchmark"], var_name="Type", value_name="Value")

fig_benchmark = px.bar(
    benchmark_long,
    x="Metric",
    y="Value",
    color="Type",
    barmode="group",
    title="Actual vs Benchmark",
)
fig_benchmark.update_layout(yaxis_title="Value")
st.plotly_chart(fig_benchmark, use_container_width=True)
st.dataframe(benchmark_df, use_container_width=True)

if phase_cols:
    st.subheader("QA / Reconciliation")
    qc_col1, qc_col2 = st.columns(2)

    fig_residual = px.line(
        wide_df,
        x=ts_col,
        y="Residual_kW",
        title="Residual | Phase Total - Monitored Load |",
    )
    fig_residual.update_layout(xaxis_title="Timestamp", yaxis_title="Residual (kW)")
    qc_col1.plotly_chart(fig_residual, use_container_width=True)

    reconciliation = pd.DataFrame(
        {
            "Metric": [
                "Phase energy (kWh)",
                "Monitored energy (kWh)",
                "Absolute difference (kWh)",
            ],
            "Value": [
                (wide_df["Phase_Total_kW"].sum() * interval_minutes / 60.0),
                actual_total_kwh,
                abs((wide_df["Phase_Total_kW"].sum() * interval_minutes / 60.0) - actual_total_kwh),
            ],
        }
    )
    qc_col2.dataframe(reconciliation, use_container_width=True)

st.markdown("---")
st.subheader("Download Processed Data")
d1, d2, d3 = st.columns(3)

d1.download_button(
    "Download processed wide data",
    data=to_csv_bytes(wide_df),
    file_name=f"{apartment_name.replace(' ', '_')}_processed_wide.csv",
    mime="text/csv",
)

d2.download_button(
    "Download processed long data",
    data=to_csv_bytes(long_df),
    file_name=f"{apartment_name.replace(' ', '_')}_processed_long.csv",
    mime="text/csv",
)

d3.download_button(
    "Download channel-category mapping",
    data=to_csv_bytes(edited_map),
    file_name=f"{apartment_name.replace(' ', '_')}_mapping.csv",
    mime="text/csv",
)
