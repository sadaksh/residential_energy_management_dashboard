import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Apartment Operation Schedule", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def infer_timestamp_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if ("time" in c.lower()) or ("date" in c.lower())]
    return candidates[0] if candidates else df.columns[0]


def load_input_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file. Upload CSV or Excel.")


def auto_detect_interval_minutes(ts: pd.Series) -> float:
    ts = pd.to_datetime(ts, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(ts) < 2:
        return 5.0
    diff = ts.diff().dropna().dt.total_seconds() / 60.0
    if diff.empty:
        return 5.0
    interval = float(diff.median())
    return 5.0 if interval <= 0 else interval


def clean_channel_name(channel: str, apartment_name: str = "") -> str:
    ch = str(channel).strip()

    if apartment_name:
        ap = re.escape(apartment_name.strip())
        ch = re.sub(ap, "", ch, flags=re.IGNORECASE)

    ch = re.sub(r"^\s*[-_:|]+\s*", "", ch)
    ch = re.sub(r"\s*[-_:|]+\s*$", "", ch)
    ch = re.sub(r"\s+", " ", ch).strip()

    return ch


def default_channel_threshold(channel: str) -> float:
    c = str(channel).strip().lower()

    def has_pattern(patterns):
        return any(re.search(p, c) for p in patterns)

    # Specific first
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
        r"\bwifi\b", r"\bpoint\b"
    ]):
        return 0.02

    return 0.02


def preprocess_raw(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, List[str]]:
    df = df.copy()

    ts_col = infer_timestamp_column(df)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    non_ts_cols = [c for c in df.columns if c != ts_col]

    # Convert possible numeric columns
    for c in non_ts_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    numeric_cols = [c for c in non_ts_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]

    # Convert all numeric values to absolute
    df[numeric_cols] = df[numeric_cols].abs()

    return df, ts_col, numeric_cols


def build_operation_long_table(
    df: pd.DataFrame,
    ts_col: str,
    load_cols: List[str],
    interval_minutes: float,
    channel_threshold_map: Dict[str, float],
    apartment_name: str = "",
) -> pd.DataFrame:
    long_df = df[[ts_col] + load_cols].melt(
        id_vars=[ts_col],
        value_vars=load_cols,
        var_name="Channel",
        value_name="kW",
    )

    long_df["kW"] = pd.to_numeric(long_df["kW"], errors="coerce").fillna(0).abs()
    long_df["Channel_Clean"] = long_df["Channel"].apply(lambda x: clean_channel_name(x, apartment_name))
    long_df["Date"] = long_df[ts_col].dt.date
    long_df["Hour"] = long_df[ts_col].dt.hour
    long_df["DayType"] = np.where(long_df[ts_col].dt.weekday < 5, "Weekday", "Weekend")
    long_df["Runtime_Threshold_kW"] = long_df["Channel"].map(channel_threshold_map).fillna(0.02)

    # 1 if active, 0 if inactive
    long_df["Runtime_Flag"] = (long_df["kW"] > long_df["Runtime_Threshold_kW"]).astype(int)

    # Active minutes in each interval
    long_df["Active_Minutes"] = long_df["Runtime_Flag"] * interval_minutes

    return long_df


def build_schedule_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Returns Channel x Hour matrix."""
    out = (
        df.groupby(["Channel_Clean", "Hour"], as_index=False)[value_col]
        .mean()
        .pivot(index="Channel_Clean", columns="Hour", values=value_col)
        .fillna(0)
    )
    for h in range(24):
        if h not in out.columns:
            out[h] = 0
    out = out[sorted(out.columns)]
    return out


def build_daily_schedule_table(df: pd.DataFrame, channel_clean: str, value_col: str) -> pd.DataFrame:
    """Returns Date x Hour matrix for one channel."""
    out = (
        df[df["Channel_Clean"] == channel_clean]
        .groupby(["Date", "Hour"], as_index=False)[value_col]
        .mean()
        .pivot(index="Date", columns="Hour", values=value_col)
        .fillna(0)
    )
    for h in range(24):
        if h not in out.columns:
            out[h] = 0
    out = out[sorted(out.columns)]
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


# -----------------------------
# App
# -----------------------------
st.title("Operation Schedule Dashboard")
st.caption(
    "Operation schedules are based on channel-wise standby thresholds. "
    "All numeric values are converted to absolute values before analysis."
)

uploaded_file = st.file_uploader("Upload apartment CSV / Excel", type=["csv", "xlsx", "xls"])

if not uploaded_file:
    st.info("Upload the apartment data file to begin.")
    st.stop()

raw_df = load_input_file(uploaded_file)
df, ts_col, numeric_cols = preprocess_raw(raw_df)

if not numeric_cols:
    st.error("No numeric load columns found.")
    st.stop()

default_phase_cols = [c for c in numeric_cols if "phase" in c.lower()]

with st.sidebar:
    st.header("Configuration")

    apartment_name = st.text_input("Apartment name", value="Lakeside K 502")

    detected_interval = auto_detect_interval_minutes(df[ts_col])
    interval_minutes = st.number_input(
        "Interval (minutes)",
        min_value=1.0,
        value=float(detected_interval),
        step=1.0,
    )

    phase_cols = st.multiselect(
        "Phase total columns to exclude",
        options=numeric_cols,
        default=default_phase_cols,
    )

    load_cols = st.multiselect(
        "Channels for operation schedule",
        options=[c for c in numeric_cols if c not in phase_cols],
        default=[c for c in numeric_cols if c not in phase_cols],
    )

    if not load_cols:
        st.error("Select at least one channel.")
        st.stop()

    st.subheader("Channel Thresholds (kW)")
    channel_threshold_map: Dict[str, float] = {}

    with st.expander("Edit thresholds", expanded=False):
        for i, ch in enumerate(load_cols):
            channel_threshold_map[ch] = st.number_input(
                label=ch,
                min_value=0.0,
                value=float(default_channel_threshold(ch)),
                step=0.01,
                key=f"thr_{i}",
            )

    schedule_metric = st.radio(
        "Schedule metric",
        ["Operation Probability", "Average Active Minutes per Hour"],
        horizontal=False,
    )

# Build long table
long_df = build_operation_long_table(
    df=df,
    ts_col=ts_col,
    load_cols=load_cols,
    interval_minutes=interval_minutes,
    channel_threshold_map=channel_threshold_map,
    apartment_name=apartment_name,
)

# Metric selection
if schedule_metric == "Operation Probability":
    long_df["Schedule_Value"] = long_df["Runtime_Flag"]
    color_label = "Probability"
    zmax_week = 1.0
    zmax_day = 1.0
else:
    # Mean active minutes within the hour across observations
    # Since each interval contributes interval_minutes if active
    # averaging Active_Minutes across intervals gives average active minutes per interval,
    # so for schedule-style visualization we convert to equivalent share of hour:
    # Runtime_Flag mean * 60
    long_df["Schedule_Value"] = long_df["Runtime_Flag"] * 60.0
    color_label = "Active Minutes"
    zmax_week = 60.0
    zmax_day = 60.0

# Weekday / weekend tables
weekday_df = long_df[long_df["DayType"] == "Weekday"].copy()
weekend_df = long_df[long_df["DayType"] == "Weekend"].copy()

weekday_schedule = build_schedule_table(weekday_df, "Schedule_Value")
weekend_schedule = build_schedule_table(weekend_df, "Schedule_Value")

# Daily selected channel
all_channels_clean = sorted(long_df["Channel_Clean"].dropna().unique().tolist())
selected_channel_clean = st.selectbox("Select channel for daily operation schedule", options=all_channels_clean)
daily_schedule = build_daily_schedule_table(long_df, selected_channel_clean, "Schedule_Value")

# Summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Apartment", apartment_name)
c2.metric("Start", str(df[ts_col].min()))
c3.metric("End", str(df[ts_col].max()))
c4.metric("Channels", len(load_cols))

st.markdown("---")

# Applied thresholds
st.subheader("Applied Channel Thresholds")
threshold_df = pd.DataFrame({
    "Channel": list(channel_threshold_map.keys()),
    "Channel_Clean": [clean_channel_name(ch, apartment_name) for ch in channel_threshold_map.keys()],
    "Runtime Threshold (kW)": list(channel_threshold_map.values()),
})
st.dataframe(threshold_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Weekday schedule
st.subheader("1) Weekday Operation Schedule")
st.caption("Rows = channels, columns = hour of day, values based on channel-wise thresholded operation.")
fig_weekday = px.imshow(
    weekday_schedule,
    aspect="auto",
    color_continuous_scale="YlOrRd",
    labels={"x": "Hour", "y": "Channel", "color": color_label},
    zmin=0,
    zmax=zmax_week,
    title="Weekday Operation Schedule",
)
st.plotly_chart(fig_weekday, use_container_width=True)

# Weekend schedule
st.subheader("2) Weekend Operation Schedule")
fig_weekend = px.imshow(
    weekend_schedule,
    aspect="auto",
    color_continuous_scale="YlOrRd",
    labels={"x": "Hour", "y": "Channel", "color": color_label},
    zmin=0,
    zmax=zmax_week,
    title="Weekend Operation Schedule",
)
st.plotly_chart(fig_weekend, use_container_width=True)

# Daily schedule
st.subheader("3) Daily Operation Schedule")
st.caption("Daily schedule is shown for the selected channel.")
fig_daily = px.imshow(
    daily_schedule,
    aspect="auto",
    color_continuous_scale="YlOrRd",
    labels={"x": "Hour", "y": "Date", "color": color_label},
    zmin=0,
    zmax=zmax_day,
    title=f"Daily Operation Schedule - {selected_channel_clean}",
)
st.plotly_chart(fig_daily, use_container_width=True)

st.markdown("---")

# Optional detailed tables
with st.expander("Show processed operation table"):
    st.dataframe(
        long_df[
            [ts_col, "Channel", "Channel_Clean", "kW", "Runtime_Threshold_kW", "Runtime_Flag", "Active_Minutes", "Date", "Hour", "DayType"]
        ],
        use_container_width=True,
    )

# Downloads
st.subheader("Download Schedule Tables")
d1, d2, d3 = st.columns(3)

d1.download_button(
    "Download weekday schedule CSV",
    data=to_csv_bytes(weekday_schedule),
    file_name=f"{apartment_name.replace(' ', '_')}_weekday_operation_schedule.csv",
    mime="text/csv",
)

d2.download_button(
    "Download weekend schedule CSV",
    data=to_csv_bytes(weekend_schedule),
    file_name=f"{apartment_name.replace(' ', '_')}_weekend_operation_schedule.csv",
    mime="text/csv",
)

d3.download_button(
    "Download daily schedule CSV",
    data=to_csv_bytes(daily_schedule),
    file_name=f"{apartment_name.replace(' ', '_')}_{selected_channel_clean.replace(' ', '_')}_daily_operation_schedule.csv",
    mime="text/csv",
)