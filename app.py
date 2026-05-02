
import io
import re
import zipfile
import inspect
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Apartment Vampire Load & Operation Schedule", layout="wide")


# -----------------------------
# Streamlit compatibility helpers
# -----------------------------
def version_at_least(current: str, minimum: str) -> bool:
    """Simple version comparison without adding extra dependencies."""
    def parse(v: str) -> Tuple[int, int, int]:
        nums = re.findall(r"\d+", str(v))[:3]
        nums = [int(x) for x in nums]
        while len(nums) < 3:
            nums.append(0)
        return tuple(nums[:3])

    return parse(current) >= parse(minimum)


def streamlit_width_supported_for_plotly() -> bool:
    """Avoid deprecated Plotly kwargs warning in older Streamlit versions.

    In newer Streamlit versions, use width="stretch".
    In older Streamlit versions, width may be interpreted as a deprecated Plotly kwarg.
    """
    try:
        has_width = "width" in inspect.signature(st.plotly_chart).parameters
    except Exception:
        has_width = False

    # Streamlit 1.51+ avoids the known plotly_chart width kwargs warning.
    return has_width and version_at_least(st.__version__, "1.51.0")


def streamlit_width_supported_for_dataframe() -> bool:
    try:
        return "width" in inspect.signature(st.dataframe).parameters
    except Exception:
        return False


PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
}


def show_plotly(fig, key: str | None = None) -> None:
    """Render Plotly chart without deprecated keyword warnings."""
    fig.update_layout(autosize=True)

    kwargs = {
        "config": PLOTLY_CONFIG,
    }
    if key is not None:
        kwargs["key"] = key

    if streamlit_width_supported_for_plotly():
        kwargs["width"] = "stretch"
    else:
        # Compatibility path for older Streamlit versions.
        kwargs["use_container_width"] = True

    st.plotly_chart(fig, **kwargs)


def show_dataframe(df: pd.DataFrame, hide_index: bool = False) -> None:
    """Render dataframe with compatible width handling."""
    kwargs = {}

    try:
        dataframe_params = inspect.signature(st.dataframe).parameters
    except Exception:
        dataframe_params = {}

    if "hide_index" in dataframe_params:
        kwargs["hide_index"] = hide_index

    if streamlit_width_supported_for_dataframe():
        kwargs["width"] = "stretch"
    else:
        kwargs["use_container_width"] = True

    st.dataframe(df, **kwargs)


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


def default_active_threshold(channel: str) -> float:
    """Threshold above which the appliance is treated as active/operating."""
    c = str(channel).strip().lower()

    def has_pattern(patterns):
        return any(re.search(p, c) for p in patterns)

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
        r"\bwifi\b", r"\brouter\b", r"\bpoint\b", r"\bcharger\b",
    ]):
        return 0.02
    return 0.02


def default_standby_threshold(channel: str) -> float:
    """Threshold below which the channel is treated as OFF / noise.

    Values between standby threshold and active threshold are treated as vampire/standby.
    """
    c = str(channel).strip().lower()

    def has_pattern(patterns):
        return any(re.search(p, c) for p in patterns)

    if has_pattern([r"\blight\b", r"\blights\b", r"\blighting\b", r"\blamp\b"]):
        return 0.001
    if has_pattern([r"\bfridge\b", r"\brefrigerator\b", r"\bfreezer\b"]):
        return 0.005
    if has_pattern([r"\bgeyser\b", r"\bwater heater\b", r"\bheater\b"]):
        return 0.020
    if has_pattern([r"\bac\b", r"\bair conditioner\b", r"\bcooling\b"]):
        return 0.010
    if has_pattern([r"\bfan\b", r"\bfans\b"]):
        return 0.003
    if has_pattern([r"\bwashing machine\b", r"\bwasher\b", r"\blaundry\b"]):
        return 0.005
    if has_pattern([r"\boven\b", r"\bmicrowave\b", r"\binduction\b", r"\bcooking\b", r"\bkitchen\b"]):
        return 0.005
    if has_pattern([
        r"\bplug\b", r"\bpower\b", r"\bspare\b", r"\bsocket\b", r"\bmisc\b",
        r"\btv\b", r"\bcomputer\b", r"\bstudy plug\b", r"\bunknown\b",
        r"\bwifi\b", r"\brouter\b", r"\bpoint\b", r"\bcharger\b",
    ]):
        return 0.003
    return 0.003


def classify_end_use(channel: str) -> str:
    c = str(channel).strip().lower()

    patterns = [
        ("Cooling / AC", [r"\bac\b", r"\bair conditioner\b", r"\bcooling\b"]),
        ("Fan", [r"\bfan\b", r"\bfans\b"]),
        ("Lighting", [r"\blight\b", r"\blights\b", r"\blighting\b", r"\blamp\b"]),
        ("Refrigeration", [r"\bfridge\b", r"\brefrigerator\b", r"\bfreezer\b"]),
        ("Water Heating", [r"\bgeyser\b", r"\bwater heater\b", r"\bheater\b"]),
        ("Laundry", [r"\bwashing machine\b", r"\bwasher\b", r"\blaundry\b"]),
        ("Kitchen Appliance", [r"\boven\b", r"\bmicrowave\b", r"\binduction\b", r"\bcooking\b", r"\bkitchen\b"]),
        ("Plug / Misc", [
            r"\bplug\b", r"\bpower\b", r"\bspare\b", r"\bsocket\b", r"\bmisc\b",
            r"\btv\b", r"\bcomputer\b", r"\bstudy plug\b", r"\bunknown\b",
            r"\bwifi\b", r"\brouter\b", r"\bpoint\b", r"\bcharger\b",
        ]),
    ]

    for category, pats in patterns:
        if any(re.search(p, c) for p in pats):
            return category
    return "Other / Unclassified"


def preprocess_raw(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, List[str]]:
    df = df.copy()

    ts_col = infer_timestamp_column(df)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    non_ts_cols = [c for c in df.columns if c != ts_col]

    for c in non_ts_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    numeric_cols = [c for c in non_ts_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]

    # Corrected data-cleaning rule: readings are in kW and negative values are treated as positive load.
    df[numeric_cols] = df[numeric_cols].abs()

    return df, ts_col, numeric_cols


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator in [0, None] or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def build_operation_long_table(
    df: pd.DataFrame,
    ts_col: str,
    load_cols: List[str],
    interval_minutes: float,
    standby_threshold_map: Dict[str, float],
    active_threshold_map: Dict[str, float],
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
    long_df["End_Use_Category"] = long_df["Channel_Clean"].apply(classify_end_use)
    long_df["Date"] = long_df[ts_col].dt.date
    long_df["Hour"] = long_df[ts_col].dt.hour
    long_df["DayType"] = np.where(long_df[ts_col].dt.weekday < 5, "Weekday", "Weekend")

    long_df["Standby_Threshold_kW"] = long_df["Channel"].map(standby_threshold_map).fillna(0.003)
    long_df["Active_Threshold_kW"] = long_df["Channel"].map(active_threshold_map).fillna(0.02)

    # Ensure active threshold is always higher than standby threshold.
    long_df["Active_Threshold_kW"] = np.maximum(
        long_df["Active_Threshold_kW"],
        long_df["Standby_Threshold_kW"] + 0.001,
    )

    energy_factor = interval_minutes / 60.0
    long_df["Interval_Minutes"] = interval_minutes
    long_df["Energy_Factor_h"] = energy_factor

    # Since readings are in kW, energy per interval is kW × interval hours.
    # For 5-minute data, this is kW × (5/60) = kW × (1/12).
    long_df["Interval_kWh"] = long_df["kW"] * energy_factor

    long_df["Off_Flag"] = (long_df["kW"] <= long_df["Standby_Threshold_kW"]).astype(int)
    long_df["Vampire_Flag"] = (
        (long_df["kW"] > long_df["Standby_Threshold_kW"])
        & (long_df["kW"] <= long_df["Active_Threshold_kW"])
    ).astype(int)
    long_df["Active_Flag"] = (long_df["kW"] > long_df["Active_Threshold_kW"]).astype(int)

    long_df["State"] = np.select(
        [long_df["Active_Flag"] == 1, long_df["Vampire_Flag"] == 1],
        ["ACTIVE", "VAMPIRE / STANDBY"],
        default="OFF",
    )

    long_df["Off_Minutes"] = long_df["Off_Flag"] * interval_minutes
    long_df["Vampire_Minutes"] = long_df["Vampire_Flag"] * interval_minutes
    long_df["Active_Minutes"] = long_df["Active_Flag"] * interval_minutes

    long_df["Off_kWh"] = long_df["Interval_kWh"] * long_df["Off_Flag"]
    long_df["Vampire_kWh"] = long_df["Interval_kWh"] * long_df["Vampire_Flag"]
    long_df["Active_kWh"] = long_df["Interval_kWh"] * long_df["Active_Flag"]

    channel_peak = long_df.groupby("Channel_Clean")["kW"].transform("max").replace(0, np.nan)
    long_df["Load_Fraction"] = (long_df["kW"] / channel_peak).fillna(0).clip(0, 1)

    return long_df


def build_schedule_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
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
    out.columns = [f"{h:02d}:00" for h in out.columns]
    return out


def build_daily_schedule_table(df: pd.DataFrame, channel_clean: str, value_col: str) -> pd.DataFrame:
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
    out.columns = [f"{h:02d}:00" for h in out.columns]
    return out


def to_csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    return df.to_csv(index=index).encode("utf-8")


def calculate_equivalent_days(ts: pd.Series, interval_minutes: float) -> float:
    ts_unique = pd.to_datetime(ts, errors="coerce").dropna().drop_duplicates()
    if ts_unique.empty:
        return 0.0
    return len(ts_unique) * interval_minutes / 1440.0


def build_equipment_summary(long_df: pd.DataFrame, equivalent_days: float) -> pd.DataFrame:
    rows = []
    for channel_clean, g in long_df.groupby("Channel_Clean", dropna=False):
        total_kwh = g["Interval_kWh"].sum()
        vampire_kwh = g["Vampire_kWh"].sum()
        active_kwh = g["Active_kWh"].sum()
        active_hours = g["Active_Minutes"].sum() / 60.0
        vampire_hours = g["Vampire_Minutes"].sum() / 60.0

        active_values = g.loc[g["Active_Flag"] == 1, "kW"]
        vampire_values = g.loc[g["Vampire_Flag"] == 1, "kW"]

        row = {
            "Channel_Clean": channel_clean,
            "End_Use_Category": g["End_Use_Category"].iloc[0],
            "Peak_kW": g["kW"].max(),
            "Average_kW": g["kW"].mean(),
            "Average_Active_kW": active_values.mean() if not active_values.empty else 0.0,
            "Average_Vampire_kW": vampire_values.mean() if not vampire_values.empty else 0.0,
            "Standby_Threshold_kW": g["Standby_Threshold_kW"].iloc[0],
            "Active_Threshold_kW": g["Active_Threshold_kW"].iloc[0],
            "Total_kWh": total_kwh,
            "Active_kWh": active_kwh,
            "Vampire_kWh": vampire_kwh,
            "Vampire_Share_%": safe_divide(vampire_kwh, total_kwh) * 100,
            "Active_Hours_Total": active_hours,
            "Vampire_Hours_Total": vampire_hours,
            "Active_Hours_per_Day": safe_divide(active_hours, equivalent_days),
            "Vampire_Hours_per_Day": safe_divide(vampire_hours, equivalent_days),
            "Vampire_kWh_per_Day": safe_divide(vampire_kwh, equivalent_days),
            "Vampire_kWh_per_Month": safe_divide(vampire_kwh, equivalent_days) * 30,
            "Vampire_kWh_per_Year": safe_divide(vampire_kwh, equivalent_days) * 365,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("Vampire_kWh_per_Day", ascending=False).reset_index(drop=True)
    return summary


def build_category_summary(equipment_summary: pd.DataFrame) -> pd.DataFrame:
    if equipment_summary.empty:
        return equipment_summary

    cols = [
        "Total_kWh",
        "Active_kWh",
        "Vampire_kWh",
        "Active_Hours_Total",
        "Vampire_Hours_Total",
        "Vampire_kWh_per_Day",
        "Vampire_kWh_per_Month",
        "Vampire_kWh_per_Year",
    ]

    out = equipment_summary.groupby("End_Use_Category", as_index=False)[cols].sum()
    out["Vampire_Share_%"] = out.apply(lambda r: safe_divide(r["Vampire_kWh"], r["Total_kWh"]) * 100, axis=1)
    out = out.sort_values("Vampire_kWh_per_Day", ascending=False).reset_index(drop=True)
    return out


def build_total_load_timeseries(long_df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = (
        long_df.groupby(ts_col, as_index=False)
        .agg(
            Total_kW=("kW", "sum"),
            Total_kWh=("Interval_kWh", "sum"),
            Vampire_kWh=("Vampire_kWh", "sum"),
            Active_kWh=("Active_kWh", "sum"),
        )
    )
    out["Hour"] = out[ts_col].dt.hour
    out["Date"] = out[ts_col].dt.date
    out["DayType"] = np.where(out[ts_col].dt.weekday < 5, "Weekday", "Weekend")
    return out


def create_zip_download(files: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, data in files.items():
            zf.writestr(filename, data.to_csv(index=True))
    buffer.seek(0)
    return buffer.getvalue()


def format_df_numbers(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


# -----------------------------
# App
# -----------------------------
st.title("Apartment Vampire Load & Operation Schedule Dashboard")
st.caption(
    "Input readings are treated as kW. Energy is calculated as kWh = kW × interval_minutes / 60. "
    "For 5-minute data, this equals kW × 1/12. Negative readings are converted to absolute values before analysis."
)

uploaded_file = st.file_uploader("Upload apartment CSV / Excel", type=["csv", "xlsx", "xls"])

if not uploaded_file:
    st.info("Upload the apartment data file to begin.")
    st.stop()

try:
    raw_df = load_input_file(uploaded_file)
    df, ts_col, numeric_cols = preprocess_raw(raw_df)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

if not numeric_cols:
    st.error("No numeric load columns found.")
    st.stop()

# Default phase-total columns to exclude to avoid double counting.
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
        help="Energy is calculated as kW × interval_minutes / 60. For 5-minute data, this is kW × 1/12.",
    )

    energy_factor = interval_minutes / 60.0
    st.info(f"Energy conversion factor: {energy_factor:.6f} h per reading")

    tariff_rs_per_kwh = st.number_input(
        "Electricity tariff for savings estimate (₹/kWh)",
        min_value=0.0,
        value=8.0,
        step=0.5,
    )

    grid_ef_kgco2_per_kwh = st.number_input(
        "Grid emission factor (kgCO₂/kWh)",
        min_value=0.0,
        value=0.70,
        step=0.05,
    )

    night_start_hour = st.number_input("Night baseload start hour", min_value=0, max_value=23, value=0, step=1)
    night_end_hour = st.number_input("Night baseload end hour", min_value=1, max_value=24, value=5, step=1)

    phase_cols = st.multiselect(
        "Phase total columns to exclude",
        options=numeric_cols,
        default=default_phase_cols,
    )

    load_cols = st.multiselect(
        "Channels for analysis",
        options=[c for c in numeric_cols if c not in phase_cols],
        default=[c for c in numeric_cols if c not in phase_cols],
    )

    if not load_cols:
        st.error("Select at least one channel.")
        st.stop()

    st.subheader("Channel Thresholds (kW)")
    standby_threshold_map: Dict[str, float] = {}
    active_threshold_map: Dict[str, float] = {}

    with st.expander("Edit standby and active thresholds", expanded=False):
        st.caption(
            "OFF: kW ≤ standby threshold | VAMPIRE/STANDBY: standby threshold < kW ≤ active threshold | ACTIVE: kW > active threshold"
        )
        for i, ch in enumerate(load_cols):
            st.markdown(f"**{clean_channel_name(ch, apartment_name)}**")
            c1, c2 = st.columns(2)
            with c1:
                standby_threshold_map[ch] = st.number_input(
                    label="Standby threshold (kW)",
                    min_value=0.0,
                    value=float(default_standby_threshold(ch)),
                    step=0.001,
                    format="%.3f",
                    key=f"standby_thr_{i}",
                )
            with c2:
                active_threshold_map[ch] = st.number_input(
                    label="Active threshold (kW)",
                    min_value=0.0,
                    value=float(default_active_threshold(ch)),
                    step=0.01,
                    format="%.3f",
                    key=f"active_thr_{i}",
                )

    schedule_metric = st.radio(
        "Schedule metric",
        [
            "Active Operation Probability",
            "Equivalent Active Minutes per Hour",
            "Vampire / Standby Probability",
            "Equivalent Vampire Minutes per Hour",
            "Average kW",
            "Load Fraction",
        ],
        horizontal=False,
    )

# Build long table
long_df = build_operation_long_table(
    df=df,
    ts_col=ts_col,
    load_cols=load_cols,
    interval_minutes=interval_minutes,
    standby_threshold_map=standby_threshold_map,
    active_threshold_map=active_threshold_map,
    apartment_name=apartment_name,
)

equivalent_days = calculate_equivalent_days(df[ts_col], interval_minutes)
total_ts = build_total_load_timeseries(long_df, ts_col)
equipment_summary = build_equipment_summary(long_df, equivalent_days)
category_summary = build_category_summary(equipment_summary)

# Metric selection for schedules
if schedule_metric == "Active Operation Probability":
    long_df["Schedule_Value"] = long_df["Active_Flag"]
    color_label = "Active Probability"
    zmin, zmax = 0, 1
elif schedule_metric == "Equivalent Active Minutes per Hour":
    long_df["Schedule_Value"] = long_df["Active_Flag"] * 60.0
    color_label = "Active Minutes per Hour"
    zmin, zmax = 0, 60
elif schedule_metric == "Vampire / Standby Probability":
    long_df["Schedule_Value"] = long_df["Vampire_Flag"]
    color_label = "Vampire Probability"
    zmin, zmax = 0, 1
elif schedule_metric == "Equivalent Vampire Minutes per Hour":
    long_df["Schedule_Value"] = long_df["Vampire_Flag"] * 60.0
    color_label = "Vampire Minutes per Hour"
    zmin, zmax = 0, 60
elif schedule_metric == "Average kW":
    long_df["Schedule_Value"] = long_df["kW"]
    color_label = "Average kW"
    zmin, zmax = 0, None
else:
    long_df["Schedule_Value"] = long_df["Load_Fraction"]
    color_label = "Load Fraction"
    zmin, zmax = 0, 1

weekday_df = long_df[long_df["DayType"] == "Weekday"].copy()
weekend_df = long_df[long_df["DayType"] == "Weekend"].copy()

weekday_schedule = build_schedule_table(weekday_df, "Schedule_Value")
weekend_schedule = build_schedule_table(weekend_df, "Schedule_Value")

# Simulation-specific schedule tables
weekday_active_probability = build_schedule_table(weekday_df.assign(Sim_Value=weekday_df["Active_Flag"]), "Sim_Value")
weekend_active_probability = build_schedule_table(weekend_df.assign(Sim_Value=weekend_df["Active_Flag"]), "Sim_Value")
weekday_vampire_probability = build_schedule_table(weekday_df.assign(Sim_Value=weekday_df["Vampire_Flag"]), "Sim_Value")
weekend_vampire_probability = build_schedule_table(weekend_df.assign(Sim_Value=weekend_df["Vampire_Flag"]), "Sim_Value")
weekday_load_fraction = build_schedule_table(weekday_df.assign(Sim_Value=weekday_df["Load_Fraction"]), "Sim_Value")
weekend_load_fraction = build_schedule_table(weekend_df.assign(Sim_Value=weekend_df["Load_Fraction"]), "Sim_Value")
weekday_avg_kw = build_schedule_table(weekday_df.assign(Sim_Value=weekday_df["kW"]), "Sim_Value")
weekend_avg_kw = build_schedule_table(weekend_df.assign(Sim_Value=weekend_df["kW"]), "Sim_Value")

all_channels_clean = sorted(long_df["Channel_Clean"].dropna().unique().tolist())
selected_channel_clean = st.selectbox("Select channel for daily operation schedule", options=all_channels_clean)
daily_schedule = build_daily_schedule_table(long_df, selected_channel_clean, "Schedule_Value")

# -----------------------------
# KPI Summary
# -----------------------------
total_energy_kwh = long_df["Interval_kWh"].sum()
total_vampire_kwh = long_df["Vampire_kWh"].sum()
vampire_share = safe_divide(total_vampire_kwh, total_energy_kwh) * 100
vampire_kwh_per_day = safe_divide(total_vampire_kwh, equivalent_days)
vampire_kwh_per_month = vampire_kwh_per_day * 30
vampire_kwh_per_year = vampire_kwh_per_day * 365
avoidable_cost_month = vampire_kwh_per_month * tariff_rs_per_kwh
avoidable_emissions_year = vampire_kwh_per_year * grid_ef_kgco2_per_kwh

if not total_ts.empty:
    avg_apartment_kw = total_ts["Total_kW"].mean()
    peak_apartment_kw = total_ts["Total_kW"].max()
    night_ts = total_ts[(total_ts["Hour"] >= night_start_hour) & (total_ts["Hour"] < night_end_hour)]
    night_baseload_kw = night_ts["Total_kW"].mean() if not night_ts.empty else 0.0
else:
    avg_apartment_kw = 0.0
    peak_apartment_kw = 0.0
    night_baseload_kw = 0.0

top_vampire_channel = "NA"
if not equipment_summary.empty and equipment_summary["Vampire_kWh_per_Day"].max() > 0:
    top_vampire_channel = equipment_summary.iloc[0]["Channel_Clean"]

st.markdown("---")
st.subheader("1) Apartment-Level KPI Summary")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Apartment", apartment_name)
k2.metric("Monitoring Start", str(df[ts_col].min()))
k3.metric("Monitoring End", str(df[ts_col].max()))
k4.metric("Equivalent Days", f"{equivalent_days:.2f}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Total Energy", f"{total_energy_kwh:.2f} kWh")
k6.metric("Vampire Energy", f"{total_vampire_kwh:.2f} kWh")
k7.metric("Vampire Share", f"{vampire_share:.1f}%")
k8.metric("Top Vampire Channel", str(top_vampire_channel))

k9, k10, k11, k12 = st.columns(4)
k9.metric("Avg Apartment Load", f"{avg_apartment_kw:.3f} kW")
k10.metric("Peak Apartment Load", f"{peak_apartment_kw:.3f} kW")
k11.metric("Night Baseload", f"{night_baseload_kw:.3f} kW")
k12.metric("Channels Analysed", len(load_cols))

k13, k14, k15, k16 = st.columns(4)
k13.metric("Vampire kWh/day", f"{vampire_kwh_per_day:.2f}")
k14.metric("Vampire kWh/month", f"{vampire_kwh_per_month:.2f}")
k15.metric("Avoidable Cost/month", f"₹{avoidable_cost_month:,.0f}")
k16.metric("Avoidable CO₂/year", f"{avoidable_emissions_year:.0f} kgCO₂")

st.caption(
    "Energy calculation note: all input readings are kW. Each interval energy is calculated as "
    "kW × interval_minutes / 60. If interval = 5 minutes, the multiplier is 1/12."
)

# -----------------------------
# Thresholds and Classification
# -----------------------------
st.markdown("---")
st.subheader("2) Applied Thresholds and End-Use Classification")
threshold_df = pd.DataFrame({
    "Channel": list(active_threshold_map.keys()),
    "Channel_Clean": [clean_channel_name(ch, apartment_name) for ch in active_threshold_map.keys()],
    "End_Use_Category": [classify_end_use(clean_channel_name(ch, apartment_name)) for ch in active_threshold_map.keys()],
    "Standby Threshold (kW)": [standby_threshold_map[ch] for ch in active_threshold_map.keys()],
    "Active Threshold (kW)": [active_threshold_map[ch] for ch in active_threshold_map.keys()],
})
show_dataframe(format_df_numbers(threshold_df), hide_index=True)

# -----------------------------
# Vampire Load Summary
# -----------------------------
st.markdown("---")
st.subheader("3) Vampire Load KPI Tables")

t1, t2 = st.tabs(["Equipment Summary", "End-Use Category Summary"])

with t1:
    show_dataframe(format_df_numbers(equipment_summary), hide_index=True)

    if not equipment_summary.empty:
        fig_top_vampire = px.bar(
            equipment_summary.head(15),
            x="Channel_Clean",
            y="Vampire_kWh_per_Day",
            hover_data=["End_Use_Category", "Average_Vampire_kW", "Vampire_Hours_per_Day", "Vampire_Share_%"],
            title="Top Vampire-Load Channels: kWh/day",
            labels={"Channel_Clean": "Channel", "Vampire_kWh_per_Day": "Vampire kWh/day"},
        )
        show_plotly(fig_top_vampire, key="top_vampire_channels")

with t2:
    show_dataframe(format_df_numbers(category_summary), hide_index=True)

    if not category_summary.empty:
        fig_category = px.bar(
            category_summary,
            x="End_Use_Category",
            y="Vampire_kWh_per_Day",
            hover_data=["Total_kWh", "Vampire_Share_%"],
            title="Vampire Load by End-Use Category",
            labels={"End_Use_Category": "End-Use Category", "Vampire_kWh_per_Day": "Vampire kWh/day"},
        )
        show_plotly(fig_category, key="vampire_by_category")

# Hourly vampire profile
hourly_vampire = (
    long_df.groupby(["DayType", "Hour"], as_index=False)
    .agg(
        Vampire_kWh=("Vampire_kWh", "sum"),
        Total_kWh=("Interval_kWh", "sum"),
    )
)

hourly_vampire_kw = (
    long_df[long_df["Vampire_Flag"] == 1]
    .groupby(["DayType", "Hour"], as_index=False)["kW"]
    .mean()
    .rename(columns={"kW": "Average_Vampire_kW"})
)

hourly_vampire = hourly_vampire.merge(
    hourly_vampire_kw,
    on=["DayType", "Hour"],
    how="left",
)
hourly_vampire["Average_Vampire_kW"] = hourly_vampire["Average_Vampire_kW"].fillna(0)

fig_hourly_vampire = px.line(
    hourly_vampire,
    x="Hour",
    y="Average_Vampire_kW",
    color="DayType",
    markers=True,
    title="Hourly Vampire / Standby Load Profile",
    labels={"Hour": "Hour of Day", "Average_Vampire_kW": "Average Vampire kW"},
)
show_plotly(fig_hourly_vampire, key="hourly_vampire_profile")

# -----------------------------
# Operation Schedules
# -----------------------------
st.markdown("---")
st.subheader("4) Weekday / Weekend Operation Schedules")
st.caption("Rows = equipment channels, columns = hour of day. Values depend on the selected schedule metric.")

c1, c2 = st.columns(2)
with c1:
    fig_weekday = px.imshow(
        weekday_schedule,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={"x": "Hour", "y": "Channel", "color": color_label},
        zmin=zmin,
        zmax=zmax,
        title=f"Weekday Schedule - {schedule_metric}",
    )
    show_plotly(fig_weekday, key="weekday_schedule")

with c2:
    fig_weekend = px.imshow(
        weekend_schedule,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={"x": "Hour", "y": "Channel", "color": color_label},
        zmin=zmin,
        zmax=zmax,
        title=f"Weekend Schedule - {schedule_metric}",
    )
    show_plotly(fig_weekend, key="weekend_schedule")

st.subheader("5) Daily Operation Schedule")
st.caption("Daily schedule is shown for the selected channel.")
fig_daily = px.imshow(
    daily_schedule,
    aspect="auto",
    color_continuous_scale="YlOrRd",
    labels={"x": "Hour", "y": "Date", "color": color_label},
    zmin=zmin,
    zmax=zmax,
    title=f"Daily Schedule - {selected_channel_clean} - {schedule_metric}",
)
show_plotly(fig_daily, key="daily_schedule")

# -----------------------------
# Simulation Export Tables
# -----------------------------
st.markdown("---")
st.subheader("6) Simulation-Ready Schedule Exports")
st.caption(
    "Use active probability and load fraction schedules for equipment operation modelling. "
    "Use vampire probability and average vampire kW for standby/base plug-load modelling."
)

sim_tabs = st.tabs([
    "Weekday Active Probability",
    "Weekend Active Probability",
    "Weekday Load Fraction",
    "Weekend Load Fraction",
    "Weekday Average kW",
    "Weekend Average kW",
    "Weekday Vampire Probability",
    "Weekend Vampire Probability",
])

with sim_tabs[0]:
    show_dataframe(format_df_numbers(weekday_active_probability))
with sim_tabs[1]:
    show_dataframe(format_df_numbers(weekend_active_probability))
with sim_tabs[2]:
    show_dataframe(format_df_numbers(weekday_load_fraction))
with sim_tabs[3]:
    show_dataframe(format_df_numbers(weekend_load_fraction))
with sim_tabs[4]:
    show_dataframe(format_df_numbers(weekday_avg_kw))
with sim_tabs[5]:
    show_dataframe(format_df_numbers(weekend_avg_kw))
with sim_tabs[6]:
    show_dataframe(format_df_numbers(weekday_vampire_probability))
with sim_tabs[7]:
    show_dataframe(format_df_numbers(weekend_vampire_probability))

# -----------------------------
# Downloads
# -----------------------------
st.markdown("---")
st.subheader("7) Download Tables")

apartment_slug = apartment_name.replace(" ", "_").replace("/", "_")

export_files = {
    f"{apartment_slug}_equipment_summary.csv": equipment_summary,
    f"{apartment_slug}_category_summary.csv": category_summary,
    f"{apartment_slug}_weekday_active_probability.csv": weekday_active_probability,
    f"{apartment_slug}_weekend_active_probability.csv": weekend_active_probability,
    f"{apartment_slug}_weekday_load_fraction.csv": weekday_load_fraction,
    f"{apartment_slug}_weekend_load_fraction.csv": weekend_load_fraction,
    f"{apartment_slug}_weekday_average_kw.csv": weekday_avg_kw,
    f"{apartment_slug}_weekend_average_kw.csv": weekend_avg_kw,
    f"{apartment_slug}_weekday_vampire_probability.csv": weekday_vampire_probability,
    f"{apartment_slug}_weekend_vampire_probability.csv": weekend_vampire_probability,
    f"{apartment_slug}_{selected_channel_clean.replace(' ', '_')}_daily_schedule.csv": daily_schedule,
}

zipped_exports = create_zip_download(export_files)

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "Download simulation tables ZIP",
        data=zipped_exports,
        file_name=f"{apartment_slug}_simulation_exports.zip",
        mime="application/zip",
    )
with c2:
    st.download_button(
        "Download equipment summary CSV",
        data=to_csv_bytes(equipment_summary, index=False),
        file_name=f"{apartment_slug}_equipment_summary.csv",
        mime="text/csv",
    )
with c3:
    st.download_button(
        "Download category summary CSV",
        data=to_csv_bytes(category_summary, index=False),
        file_name=f"{apartment_slug}_category_summary.csv",
        mime="text/csv",
    )

st.info(
    "The full processed interval-level table has been intentionally removed from display and download. "
    "The available exports are simulation-ready schedules and KPI summaries only."
)

st.markdown("---")
st.subheader("Interpretation Notes")
st.markdown(
    """
- **Input unit:** All channel readings are treated as **kW**.
- **Energy conversion:** `kWh = kW × interval_minutes / 60`. For 5-minute readings, use `kW × 1/12`.
- **OFF state:** Reading is below or equal to the standby/noise threshold.
- **VAMPIRE / STANDBY state:** Reading is above the standby threshold but below or equal to the active threshold.
- **ACTIVE state:** Reading is above the active threshold.
- **Simulation use:** Use `active probability` or `load fraction` schedules for equipment operation. Use `vampire probability`, `average vampire kW`, and `vampire kWh/day` for standby/base-load modelling.
"""
)
