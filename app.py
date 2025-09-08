# app.py

# --- Compatibility & startup-hardening shim -----------------------------------
from __future__ import annotations
import warnings

# Silence *known* library deprecations that spam logs (not failures).
warnings.filterwarnings(
    "ignore",
    message="The default of observed=False is deprecated and will be changed to True",
    category=FutureWarning,
)

import streamlit as st  # noqa: E402

def _translate_width(kwargs: dict) -> dict:
    """Translate deprecated use_container_width -> width='stretch'/'content'."""
    if "use_container_width" in kwargs:
        val = kwargs.pop("use_container_width")
        kwargs.setdefault("width", "stretch" if val else "content")
    return kwargs

def _wrap_st_fn(fn):
    def _inner(*args, **kwargs):
        return fn(*args, **_translate_width(kwargs))
    _inner.__name__ = fn.__name__
    return _inner

# Patch common renderers so legacy code & libs won't warn
for _name in [
    "plotly_chart", "dataframe", "table", "altair_chart", "pydeck_chart",
    "map", "line_chart", "bar_chart", "area_chart", "pyplot",
]:
    if hasattr(st, _name):
        setattr(st, _name, _wrap_st_fn(getattr(st, _name)))

# Pandas groupby: keep legacy default & silence FutureWarning without changing behavior
import pandas as pd  # noqa: E402
_pd_df_groupby_orig = pd.DataFrame.groupby
def _pd_groupby_compat(self, *args, **kwargs):
    kwargs.setdefault("observed", False)
    return _pd_df_groupby_orig(self, *args, **kwargs)
pd.DataFrame.groupby = _pd_groupby_compat  # type: ignore[assignment]

# Lazy import helpers (avoid slow startup / health-check 503s)
def lazy_import(name: str):
    import importlib
    try:
        return importlib.import_module(name)
    except Exception as e:  # never block app start
        return e

def ensure_shap():
    mod = lazy_import("shap")
    if isinstance(mod, Exception):
        st.info("SHAP not available in this environment. Skipping SHAP visuals.")
        return None
    return mod

def ensure_prophet():
    mod = lazy_import("prophet")
    if isinstance(mod, Exception):
        st.info("Prophet not available. Falling back to ARIMA for forecasting.")
        return None
    return mod

# Page config early to avoid re-render churn
st.set_page_config(page_title="Uber NCR 2024 – Analytics & Decision Lab", page_icon="🚖", layout="wide")
# --- End shim -----------------------------------------------------------------


import os
import io
from typing import Tuple, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# Optional heavy libs (OK if missing)
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None

import statsmodels.api as sm

# ------------------------------#
# Constants
# ------------------------------#
RANDOM_STATE = 42
INSIGHT_COLOR = "#5e60ce"
DEMAND_COLOR = "#1f77b4"
RISK_COLOR = "#e76f51"
FIN_COLOR = "#2a9d8f"
CX_COLOR = "#9b5de5"

DATE_COL = "Date"
TIME_COL = "Time"

SCHEMA = [
    "Date", "Time", "Booking ID", "Booking Status", "Customer ID", "Vehicle Type",
    "Pickup Location", "Drop Location", "Avg VTAT", "Avg CTAT",
    "Cancelled Rides by Customer", "Reason for cancelling by Customer",
    "Cancelled Rides by Driver", "Driver Cancellation Reason",
    "Incomplete Rides", "Incomplete Rides Reason",
    "Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating", "Payment Method"
]

CANONICAL_STATUSES = ["Completed", "Customer Cancelled", "Driver Cancelled", "No Driver Found", "Incomplete"]

# ------------------------------#
# Helpers
# ------------------------------#

def _title_case_or_nan(x: any) -> Optional[str]:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s == "0" or s.lower() in {"na", "nan", "none", "null"}:
        return np.nan
    return s.title()

def time_bucket_from_hour(h: int) -> str:
    if 5 <= h < 12:
        return "Morning (05–11)"
    elif 12 <= h < 17:
        return "Afternoon (12–16)"
    elif 17 <= h < 21:
        return "Evening (17–20)"
    else:
        return "Night (21–04)"

def compress_categories(s: pd.Series, top_n: int = 30, other_label: str = "Other") -> pd.Series:
    top = s.value_counts().nlargest(top_n).index
    return s.where(s.isin(top), other_label)

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def canonical_status(row: pd.Series) -> str:
    raw = str(row.get("Booking Status", "")).strip()
    cust_cxl = row.get("Cancelled Rides by Customer", 0)
    drv_cxl = row.get("Cancelled Rides by Driver", 0)
    incomplete = row.get("Incomplete Rides", 0)
    if isinstance(raw, str) and raw.lower() == "completed":
        return "Completed"
    if safe_numeric(pd.Series([cust_cxl])).iloc[0] > 0 or "customer" in raw.lower():
        return "Customer Cancelled"
    if safe_numeric(pd.Series([drv_cxl])).iloc[0] > 0 or "driver" in raw.lower():
        return "Driver Cancelled"
    if "no driver found" in raw.lower():
        return "No Driver Found"
    if safe_numeric(pd.Series([incomplete])).iloc[0] > 0 or "incomplete" in raw.lower():
        return "Incomplete"
    return raw if raw in CANONICAL_STATUSES else raw

def revenue_mask_for_completed(status: pd.Series) -> pd.Series:
    return (status == "Completed")

@st.cache_data(show_spinner=False)
def load_csv(file: io.BytesIO | str) -> pd.DataFrame:
    """Load the CSV with strict schema."""
    df = pd.read_csv(file)
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Type casting, feature engineering, normalization."""
    msgs = []
    df = df.copy()

    # Parse Date and Time → timestamp
    df["parsed_date"] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df["parsed_time"] = pd.to_datetime(df[TIME_COL], format="%H:%M:%S", errors="coerce").dt.time

    def build_timestamp(r):
        if pd.isna(r["parsed_date"]) or pd.isna(r["parsed_time"]):
            return pd.NaT
        return pd.Timestamp.combine(r["parsed_date"].date(), r["parsed_time"])

    df["timestamp"] = df.apply(build_timestamp, axis=1)
    invalid = df["timestamp"].isna().sum()
    if invalid > 0:
        msgs.append(f"⚠️ Dropped {invalid} rows with invalid Date/Time.")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Feature extraction
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek  # 0=Mon
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["time_bucket"] = df["hour"].apply(time_bucket_from_hour)

    # Numeric casts
    rename_map = {
        "Avg VTAT": "avg_vtat",
        "Avg CTAT": "avg_ctat",
        "Cancelled Rides by Customer": "cancelled_by_customer",
        "Cancelled Rides by Driver": "cancelled_by_driver",
        "Incomplete Rides": "incomplete_rides",
        "Booking Value": "booking_value",
        "Ride Distance": "ride_distance",
        "Driver Ratings": "driver_ratings",
        "Customer Rating": "customer_rating",
    }
    for k, v in rename_map.items():
        df[v] = safe_numeric(df[k])

    # Reasons – clean / standardize
    df["reason_customer"] = df["Reason for cancelling by Customer"].map(_title_case_or_nan)
    df["reason_driver"] = df["Driver Cancellation Reason"].map(_title_case_or_nan)
    df["reason_incomplete"] = df.get("Incomplete Rides Reason", np.nan)
    df["reason_incomplete"] = df["reason_incomplete"].map(_title_case_or_nan)

    # Canonical status & target
    df["booking_status_canon"] = df.apply(canonical_status, axis=1)
    df["will_complete"] = (df["booking_status_canon"] == "Completed").astype(int)

    # Categoricals
    for c in [
        "Booking Status", "booking_status_canon", "Vehicle Type", "Pickup Location",
        "Drop Location", "Payment Method", "time_bucket"
    ]:
        df[c] = df[c].astype("category")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, msgs

def insight_box(text: str):
    st.markdown(
        f"""
        <div style="border-left:6px solid {INSIGHT_COLOR}; padding:0.6rem 0.8rem; background:#f7f7ff; border-radius:6px;">
        <strong>Insight</strong><br>{text}
        </div>
        """, unsafe_allow_html=True
    )

def kpi_cards(df: pd.DataFrame):
    total = len(df)
    comp = (df["booking_status_canon"] == "Completed").sum()
    cust_cxl = (df["booking_status_canon"] == "Customer Cancelled").sum()
    drv_cxl = (df["booking_status_canon"] == "Driver Cancelled").sum()
    avg_drv = df["driver_ratings"].replace(0, np.nan).mean()
    avg_cus = df["customer_rating"].replace(0, np.nan).mean()
    revenue = df.loc[revenue_mask_for_completed(df["booking_status_canon"]), "booking_value"].sum()

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Total Bookings", f"{total:,}")
    c2.metric("Completion %", f"{(comp/total*100) if total else 0:.1f}%")
    c3.metric("Customer Cancel %", f"{(cust_cxl/total*100) if total else 0:.1f}%")
    c4.metric("Driver Cancel %", f"{(drv_cxl/total*100) if total else 0:.1f}%")
    c5.metric("Avg Driver Rating", f"{avg_drv:.2f}" if not np.isnan(avg_drv) else "—")
    c6.metric("Avg Customer Rating", f"{avg_cus:.2f}" if not np.isnan(avg_cus) else "—")
    c7.metric("Total Revenue (Completed)", f"₹ {revenue:,.0f}")

def plot_series(df: pd.DataFrame):
    freq = st.selectbox("Aggregation frequency", ["Daily", "Weekly"], index=0, key="ts_freq")
    base = df.set_index("timestamp").assign(
        revenue=lambda x: x["booking_value"].where(x["booking_status_canon"] == "Completed", 0)
    )
    if freq == "Daily":
        s = base.resample("D").agg(bookings=("Booking ID", "count"), revenue=("revenue", "sum"))
    else:
        s = base.resample("W-SUN").agg(bookings=("Booking ID", "count"), revenue=("revenue", "sum"))
    s = s.reset_index()
    fig1 = px.line(s, x="timestamp", y="bookings", title="Bookings Over Time", markers=True, color_discrete_sequence=[DEMAND_COLOR])
    fig2 = px.line(s, x="timestamp", y="revenue", title="Revenue Over Time (Completed)", markers=True, color_discrete_sequence=[FIN_COLOR])
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

def descriptive_stats(df: pd.DataFrame):
    cols = ["ride_distance", "booking_value", "driver_ratings", "customer_rating"]
    rows = []
    for c in cols:
        s = df[c].replace(0, np.nan)
        rows.append({
            "Metric": c.replace("_", " ").title(),
            "Mean": np.nanmean(s),
            "Median": np.nanmedian(s),
            "Mode": s.mode().iloc[0] if s.dropna().size > 0 else np.nan
        })
    st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True)

def bar_from_series(series: pd.Series, title: str, x_label: str = None, y_label: str = "Count", color=DEMAND_COLOR):
    dfp = series.reset_index()
    dfp.columns = [x_label or series.index.name or "Category", y_label]
    fig = px.bar(dfp, x=dfp.columns[0], y=dfp.columns[1], title=title, color_discrete_sequence=[color])
    st.plotly_chart(fig, use_container_width=True)

def filter_block(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("📅 Global Filters")

    min_d, max_d = df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()
    drange = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(drange, tuple) and len(drange) == 2:
        start_date, end_date = drange
        mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
        df = df[mask]

    vtypes = st.sidebar.multiselect("Vehicle Type", sorted(df["Vehicle Type"].dropna().unique().tolist()))
    if vtypes:
        df = df[df["Vehicle Type"].isin(vtypes)]

    pm = st.sidebar.multiselect("Payment Method", sorted(df["Payment Method"].dropna().unique().tolist()))
    if pm:
        df = df[df["Payment Method"].isin(pm)]

    bs = st.sidebar.multiselect("Booking Status", sorted(df["booking_status_canon"].dropna().unique().tolist()))
    if bs:
        df = df[df["booking_status_canon"].isin(bs)]

    pls = st.sidebar.multiselect("Pickup Location", sorted(df["Pickup Location"].dropna().unique().tolist()))
    if pls:
        df = df[df["Pickup Location"].isin(pls)]

    dls = st.sidebar.multiselect("Drop Location", sorted(df["Drop Location"].dropna().unique().tolist()))
    if dls:
        df = df[df["Drop Location"].isin(dls)]

    return df

def empty_state(df: pd.DataFrame) -> bool:
    if df.empty:
        st.info("No rows match the current filters. Adjust filters in the sidebar.")
        return True
    return False

# ------------------------------#
# UI – Data Load
# ------------------------------#
st.sidebar.title("🚖 Uber NCR 2024 – Analytics & Decision Lab")

st.sidebar.markdown("**Data Source**")
data_source = st.sidebar.radio("Choose source", ["Auto-detect file", "Upload CSV"], index=0)
default_path = "ncr_ride_bookingsv1.csv"

df_raw = None
load_msgs: List[str] = []
try:
    if data_source == "Auto-detect file":
        if os.path.exists(default_path):
            df_raw = load_csv(default_path)
        else:
            st.sidebar.warning("Default file not found. Please upload the CSV.")
    if df_raw is None:
        uploaded = st.sidebar.file_uploader("Upload `ncr_ride_bookingsv1.csv`", type=["csv"])
        if uploaded:
            df_raw = load_csv(uploaded)
except Exception as e:
    st.sidebar.error(f"Failed to load CSV: {e}")

if df_raw is None:
    st.stop()

df, load_msgs = preprocess(df_raw)
for m in load_msgs:
    st.warning(m)

# Apply global filters
df_f = filter_block(df)
if empty_state(df_f):
    st.stop()

# Sidebar – Downloads (global filtered data)
st.sidebar.markdown("---")
st.sidebar.markdown("**Downloads**")
csv_filtered = df_f.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download Filtered Data (CSV)", csv_filtered, file_name="filtered_data.csv", mime="text/csv")

# Sidebar – Model selectors
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Preferences")
clf_choice = st.sidebar.selectbox("Classifier", ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"], index=1)
fcast_choice = st.sidebar.selectbox("Forecaster", ["ARIMA", "Prophet"], index=0)
clus_choice = st.sidebar.selectbox("Clustering", ["K-Means", "DBSCAN", "GMM"], index=0)
regr_choice = st.sidebar.selectbox("Regressor", ["Linear Regression", "Random Forest", "Gradient Boosting"], index=2)

# ------------------------------#
# Tabs
# ------------------------------#
tabs = st.tabs([
    "1) Executive Overview",
    "2) Ride Completion & Cancellation",
    "3) Geographical & Temporal",
    "4) Operational Efficiency",
    "5) Financial Analysis",
    "6) Ratings & Satisfaction",
    "7) Incomplete Rides",
    "8) ML Lab",
    "9) Risk & Fraud",
    "10) Operations Simulator",
    "11) Reports & Exports"
])

# ---------- Tab 1
with tabs[0]:
    st.markdown("## Executive Overview")
    kpi_cards(df_f)
    st.markdown("---")
    plot_series(df_f)
    st.markdown("---")

    # Funnel
    total = len(df_f)
    completed = (df_f["booking_status_canon"] == "Completed").sum()
    rated = df_f["customer_rating"].replace(0, np.nan).notna().sum()
    stages = ["Booked", "Completed", "Rated"]
    values = [total, completed, rated]
    fig = go.Figure(go.Funnel(y=stages, x=values, textinfo="value+percent previous"))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Descriptive Stats")
        descriptive_stats(df_f)
    with c2:
        st.markdown("### Top Frequencies")
        bar_from_series(df_f["Vehicle Type"].value_counts().head(10), "Vehicle Type (Top 10)", "Vehicle Type")
        bar_from_series(df_f["Pickup Location"].value_counts().head(10), "Pickup Location (Top 10)", "Pickup Location")
        bar_from_series(df_f["Payment Method"].value_counts().head(10), "Payment Method (Top 10)", "Payment Method")

    # Insight
    spike = (
        df_f.groupby("time_bucket")["Booking ID"].count()
        .sort_values(ascending=False)
        .head(1)
    )
    if len(spike) > 0:
        tb = spike.index[0]
        insight_box(f"**Demand peaks in {tb}**. Rebalance supply and incentives to reduce 'No Driver Found' and protect completion.")

# ---------- Tab 2
with tabs[1]:
    st.markdown("## Ride Completion & Cancellation")
    total = len(df_f)
    comp = (df_f["booking_status_canon"] == "Completed").sum()
    cust_cxl = (df_f["booking_status_canon"] == "Customer Cancelled").sum()
    drv_cxl = (df_f["booking_status_canon"] == "Driver Cancelled").sum()
    nd_found = (df_f["booking_status_canon"] == "No Driver Found").sum()
    inc = (df_f["booking_status_canon"] == "Incomplete").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Completion %", f"{(comp/total*100):.1f}%")
    c2.metric("Customer Cancel %", f"{(cust_cxl/total*100):.1f}%")
    c3.metric("Driver Cancel %", f"{(drv_cxl/total*100):.1f}%")
    c4.metric("No Driver Found %", f"{(nd_found/total*100):.1f}%")
    c5.metric("Incomplete %", f"{(inc/total*100):.1f}%")

    st.markdown("### Top Cancellation/Incomplete Reasons")
    rc = df_f["reason_customer"].value_counts().head(15)
    rd = df_f["reason_driver"].value_counts().head(15)
    ri = df_f["reason_incomplete"].value_counts().head(15)

    co1, co2, co3 = st.columns(3)
    with co1:
        bar_from_series(rc, "Customer Reasons", "Reason", color=RISK_COLOR)
    with co2:
        bar_from_series(rd, "Driver Reasons", "Reason", color=RISK_COLOR)
    with co3:
        bar_from_series(ri, "Incomplete Reasons", "Reason", color=RISK_COLOR)

    st.markdown("---")
    st.markdown("### Cancellation Rate by Vehicle / Time Bucket / Pickup")
    by_vehicle = (df_f.assign(is_cancel=(df_f["will_complete"] == 0))
                  .groupby("Vehicle Type")["is_cancel"].mean().sort_values(ascending=False))
    by_bucket = (df_f.assign(is_cancel=(df_f["will_complete"] == 0))
                 .groupby("time_bucket")["is_cancel"].mean().sort_values(ascending=False))
    by_pickup = (df_f.assign(is_cancel=(df_f["will_complete"] == 0))
                 .groupby("Pickup Location")["is_cancel"].mean().sort_values(ascending=False).head(20))

    bar_from_series(by_vehicle, "Cancellation Rate by Vehicle Type", "Vehicle Type", "Rate")
    bar_from_series(by_bucket, "Cancellation Rate by Time Bucket", "Time Bucket", "Rate")
    bar_from_series(by_pickup, "Cancellation Rate by Pickup (Top 20)", "Pickup Location", "Rate")

    worst_vehicle = by_vehicle.index[0] if len(by_vehicle) else None
    worst_bucket = by_bucket.index[0] if len(by_bucket) else None
    if worst_vehicle and worst_bucket:
        insight_box(
            f"Highest cancellation propensity: **{worst_vehicle}** × **{worst_bucket}**. "
            f"Deploy targeted driver incentives and VTAT caps in these micro-windows."
        )

# ---------- Tab 3
with tabs[2]:
    st.markdown("## Geographical & Temporal (No Maps)")

    st.markdown("### Busiest Locations")
    top_pick = df_f["Pickup Location"].value_counts().head(20).rename("count").reset_index().rename(columns={"index": "Pickup Location"})
    top_drop = df_f["Drop Location"].value_counts().head(20).rename("count").reset_index().rename(columns={"index": "Drop Location"})
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(top_pick, use_container_width=True)
        fig = px.bar(top_pick, x="Pickup Location", y="count", title="Top Pickups", color_discrete_sequence=[DEMAND_COLOR])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(top_drop, use_container_width=True)
        fig = px.bar(top_drop, x="Drop Location", y="count", title="Top Drops", color_discrete_sequence=[DEMAND_COLOR])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Peak Patterns")
    hh = df_f["hour"].value_counts().sort_index()
    dow = df_f["weekday"].value_counts().sort_index()
    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(hh, title="By Hour of Day", labels={"index": "Hour", "value": "Trips"}, color_discrete_sequence=[DEMAND_COLOR])
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        fig = px.bar(dow.rename(index=dow_map), title="By Day of Week", labels={"index": "Day", "value": "Trips"}, color_discrete_sequence=[DEMAND_COLOR])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Category Heat Tables")
    # Pickup × Hour heat
    heat_pick_hr = (df_f.assign(cnt=1)
                    .pivot_table(index="Pickup Location", columns="hour", values="cnt", aggfunc="sum", fill_value=0))
    heat_pick_hr = heat_pick_hr.loc[heat_pick_hr.sum(axis=1).sort_values(ascending=False).head(20).index]
    st.plotly_chart(px.imshow(heat_pick_hr, aspect="auto", color_continuous_scale="Blues",
                              title="Pickup × Hour Heat (Top 20 Pickups)"), use_container_width=True)
    # Pickup × Weekday heat
    heat_pick_dow = (df_f.assign(cnt=1)
                     .pivot_table(index="Pickup Location", columns="weekday", values="cnt", aggfunc="sum", fill_value=0))
    heat_pick_dow = heat_pick_dow.loc[heat_pick_dow.sum(axis=1).sort_values(ascending=False).head(20).index]
    heat_pick_dow.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    st.plotly_chart(px.imshow(heat_pick_dow, aspect="auto", color_continuous_scale="Blues",
                              title="Pickup × Day-of-Week Heat (Top 20 Pickups)"), use_container_width=True)

# ---------- Tab 4
with tabs[3]:
    st.markdown("## Operational Efficiency")

    st.markdown("### VTAT & CTAT Distributions")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df_f, x="avg_vtat", nbins=40, title="Avg VTAT", color_discrete_sequence=[RISK_COLOR])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df_f, x="avg_ctat", nbins=40, title="Avg CTAT", color_discrete_sequence=[RISK_COLOR])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### By Location & Vehicle")
    gv = df_f.groupby(["Pickup Location", "Vehicle Type"]).agg(
        vt=("avg_vtat", "mean"),
        ct=("avg_ctat", "mean"),
        n=("Booking ID", "count")
    ).reset_index().sort_values("n", ascending=False).head(30)
    st.dataframe(gv.round(2), use_container_width=True)

    st.markdown("---")
    st.markdown("### Correlations")
    corr_cols = ["avg_vtat", "avg_ctat", "driver_ratings", "customer_rating", "booking_value", "ride_distance", "will_complete"]
    cmat = df_f[corr_cols].replace(0, np.nan).corr()
    st.plotly_chart(px.imshow(cmat, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                              title="Correlation Matrix"), use_container_width=True)

    vt_min = float(np.nanmin(df_f["avg_vtat"])) if df_f["avg_vtat"].notna().any() else 0.0
    vt_max = float(np.nanmax(df_f["avg_vtat"])) if df_f["avg_vtat"].notna().any() else 1.0
    vt_default = float(np.nanpercentile(df_f["avg_vtat"].dropna(), 80)) if df_f["avg_vtat"].notna().any() else 0.5
    vt_thresh = st.slider("VTAT threshold highlighting (minutes)", min_value=vt_min, max_value=vt_max, value=vt_default)
    high_vt = df_f["avg_vtat"] >= vt_thresh
    cancel_rate_high = (df_f.loc[high_vt, "will_complete"] == 0).mean() if high_vt.any() else np.nan
    cancel_rate_low = (df_f.loc[~high_vt, "will_complete"] == 0).mean() if (~high_vt).any() else np.nan
    insight_box(f"When **VTAT ≥ {vt_thresh:.1f}**, cancellation rate is **{cancel_rate_high:.1%}** vs **{cancel_rate_low:.1%}** below threshold.")

# ---------- Tab 5
with tabs[4]:
    st.markdown("## Financial Analysis")

    rev_mask = revenue_mask_for_completed(df_f["booking_status_canon"])
    total_rev = df_f.loc[rev_mask, "booking_value"].sum()
    completed_count = int(rev_mask.sum())
    arpr = (total_rev / completed_count) if completed_count else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue (Completed)", f"₹ {total_rev:,.0f}")
    c2.metric("Completed Rides", f"{completed_count:,}")
    c3.metric("ARPR", f"₹ {arpr:,.2f}")

    st.markdown("---")
    st.markdown("### Revenue by Payment Method & Vehicle")
    grp = df_f[rev_mask].groupby(["Payment Method", "Vehicle Type"])["booking_value"].sum().reset_index()
    fig = px.bar(grp, x="Payment Method", y="booking_value", color="Vehicle Type", barmode="stack",
                 title="Revenue Mix", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Value vs Distance")
    fig2 = px.scatter(df_f[rev_mask], x="ride_distance", y="booking_value", color="Vehicle Type",
                      trendline="ols", title="Booking Value vs Ride Distance")
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Tab 6
with tabs[5]:
    st.markdown("## Ratings & Satisfaction")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df_f, x="driver_ratings", nbins=20, title="Driver Ratings",
                                     color_discrete_sequence=[CX_COLOR]), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(df_f, x="customer_rating", nbins=20, title="Customer Ratings",
                                     color_discrete_sequence=[CX_COLOR]), use_container_width=True)

    st.markdown("---")
    st.markdown("### Correlations & Risk Flags")
    r1 = df_f[["driver_ratings", "avg_vtat", "cancelled_by_driver"]].corr().iloc[0, 1:].to_frame("corr")
    st.write("**Driver Ratings vs VTAT & Driver Cancellations**")
    st.dataframe(r1.round(2), use_container_width=True)

    low_thr = st.slider("Low rating threshold", 1.0, 5.0, 3.5, 0.1)
    seg = (df_f.assign(low_rate=(df_f["customer_rating"] > 0) & (df_f["customer_rating"] < low_thr))
           .groupby(["Vehicle Type", "time_bucket"])["low_rate"].mean().reset_index()
           .sort_values("low_rate", ascending=False).head(20))
    st.plotly_chart(px.bar(seg, x="low_rate", y="Vehicle Type", color="time_bucket", orientation="h",
                           title=f"Probability of < {low_thr:.1f} Stars by Segment"), use_container_width=True)

# ---------- Tab 7
with tabs[6]:
    st.markdown("## Incomplete Rides")

    inc_df = df_f[df_f["booking_status_canon"] == "Incomplete"]
    share = len(inc_df) / len(df_f) if len(df_f) else 0
    st.metric("Incomplete Share", f"{share:.2%}")
    st.markdown("### Reasons")
    bar_from_series(inc_df["reason_incomplete"].value_counts().head(20), "Top Incomplete Reasons", "Reason", color=RISK_COLOR)
    st.markdown("### Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        bar_from_series(inc_df["Pickup Location"].value_counts().head(20), "Incomplete by Pickup (Top 20)", "Pickup Location", color=DEMAND_COLOR)
    with c2:
        bar_from_series(inc_df["Vehicle Type"].value_counts().head(20), "Incomplete by Vehicle", "Vehicle Type", color=DEMAND_COLOR)

# ------------------------------#
# ML Utilities
# ------------------------------#

def make_features_for_classification(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Avoid leakage: exclude post-outcome fields (CTAT, ratings, explicit cancels, incomplete counts & reasons)
    X = data[[
        "Vehicle Type", "Pickup Location", "Drop Location", "Payment Method",
        "hour", "weekday", "month", "is_weekend", "time_bucket",
        "avg_vtat", "ride_distance"
    ]].copy()
    y = data["will_complete"].copy()

    for c in ["Pickup Location", "Drop Location"]:
        X[c] = compress_categories(X[c].astype(str), top_n=30)

    for c in ["Vehicle Type", "Pickup Location", "Drop Location", "Payment Method", "time_bucket"]:
        X[c] = X[c].astype(str)

    return X, y

def time_aware_split(data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(data)
    cut = int((1 - test_size) * n)
    return data.iloc[:cut].copy(), data.iloc[cut:].copy()

def build_classifier(name: str):
    num_cols = ["hour", "weekday", "month", "is_weekend", "avg_vtat", "ride_distance"]
    cat_cols = ["Vehicle Type", "Pickup Location", "Drop Location", "Payment Method", "time_bucket"]

    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    if name == "Logistic Regression":
        clf = LogisticRegression(max_iter=200, class_weight="balanced")
    elif name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced_subsample")
    elif name == "XGBoost" and xgb is not None:
        clf = xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"
        )
    elif name == "LightGBM" and lgb is not None:
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=-1, subsample=0.9, colsample_bytree=0.8,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        st.warning(f"{name} not available; falling back to Random Forest.")
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced_subsample")

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    st.plotly_chart(fig, use_container_width=True)

def plot_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

def train_forecast(df_ts: pd.DataFrame, model_name: str, periods: int = 14):
    """Return (history_df, forecast_df). df_ts must have columns ['ds','y'] (daily)."""
    # Standardize on daily cadence for robustness
    y = df_ts.set_index("ds")["y"].asfreq("D").fillna(0)
    if model_name == "Prophet" and ensure_prophet() is not None:
        Prophet = ensure_prophet().Prophet  # type: ignore[attr-defined]
        m = Prophet(seasonality_mode="additive")
        m.fit(pd.DataFrame({"ds": y.index, "y": y.values}))
        future = m.make_future_dataframe(periods=periods, freq="D")
        fc = m.predict(future)
        return pd.DataFrame({"ds": y.index, "yhat": y.values}), fc
    else:
        try:
            model = sm.tsa.ARIMA(y, order=(2, 1, 2))
            res = model.fit()
            fc = res.get_forecast(steps=periods)
            fc_df = pd.DataFrame({
                "ds": pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D"),
                "yhat": fc.predicted_mean.values,
                "yhat_lower": fc.conf_int().iloc[:, 0].values,
                "yhat_upper": fc.conf_int().iloc[:, 1].values
            })
            hist = pd.DataFrame({"ds": y.index, "yhat": y.values})
            return hist, fc_df
        except Exception as e:
            st.error(f"ARIMA failed: {e}")
            return pd.DataFrame({"ds": y.index, "yhat": y.values}), None

def regression_models(name: str):
    if name == "Linear Regression":
        return LinearRegression()
    elif name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
    else:
        return GradientBoostingRegressor(random_state=RANDOM_STATE)

# ---------- Tab 8
with tabs[7]:
    st.markdown("## ML Lab")

    st.markdown("### A) Classification – Predict `will_complete`")
    X_all, y_all = make_features_for_classification(df_f)
    data_all = df_f.loc[X_all.index, ["timestamp"]].copy()
    data_all["y"] = y_all.values
    tr, te = time_aware_split(data_all, test_size=0.2)
    X_train, y_train = X_all.loc[tr.index], y_all.loc[tr.index]
    X_test, y_test = X_all.loc[te.index], y_all.loc[te.index]

    pipe = build_classifier(clf_choice)
    with st.spinner("Training classifier..."):
        pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
    else:
        try:
            y_score = pipe.decision_function(X_test)
            y_prob = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-6)
        except Exception:
            y_prob = y_pred.astype(float)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    c2.metric("F1", f"{f1_score(y_test, y_pred):.3f}")
    try:
        rocauc = roc_auc_score(y_test, y_prob)
        c3.metric("ROC AUC", f"{rocauc:.3f}")
    except Exception:
        c3.metric("ROC AUC", "—")
    c4.metric("Test Size", f"{len(y_test):,}")

    plot_confusion(y_test, y_pred)
    try:
        plot_roc(y_test, y_prob)
    except Exception:
        pass

    st.markdown("#### Feature Importance / Coefficients")
    try:
        model = pipe.named_steps["clf"]
        pre = pipe.named_steps["pre"]
        oh: OneHotEncoder = pre.named_transformers_["cat"]
        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][2]
        feature_names = list(num_cols) + list(oh.get_feature_names_out(cat_cols))

        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            imp_df = pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values("importance", ascending=False).head(25)
            st.plotly_chart(px.bar(imp_df, x="importance", y="feature", orientation="h", title="Top Features"), use_container_width=True)
        elif hasattr(model, "coef_"):
            coefs = model.coef_.ravel()
            coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs}).assign(abs_coef=lambda d: d["coef"].abs()).sort_values("abs_coef", ascending=False).head(25)
            st.plotly_chart(px.bar(coef_df, x="coef", y="feature", orientation="h", title="Top Coefficients"), use_container_width=True)
        else:
            st.info("Model does not expose importances/coefficients.")
    except Exception as e:
        st.info(f"Feature importance unavailable: {e}")

    # Optional SHAP (sampled)
    shap_mod = ensure_shap()
    if shap_mod is not None and hasattr(pipe.named_steps["clf"], "predict"):
        with st.expander("SHAP (sampled)"):
            sample_n = min(10000, len(X_test))
            if sample_n >= 100:
                Xs = X_test.sample(sample_n, random_state=RANDOM_STATE)
                try:
                    X_enc = pipe.named_steps["pre"].fit_transform(X_train)
                    model = pipe.named_steps["clf"]
                    explainer = shap_mod.Explainer(model, X_enc)  # type: ignore[attr-defined]
                    vals = explainer(pipe.named_steps["pre"].transform(Xs))
                    st.write("Mean |SHAP| (top 20)")
                    shap_sum = np.abs(vals.values).mean(axis=0)
                    top_idx = np.argsort(shap_sum)[::-1][:20]
                    # Recompute names in case fitting transformed columns changed
                    oh: OneHotEncoder = pipe.named_steps["pre"].named_transformers_["cat"]
                    num_cols = pipe.named_steps["pre"].transformers_[0][2]
                    cat_cols = pipe.named_steps["pre"].transformers_[1][2]
                    feature_names = list(num_cols) + list(oh.get_feature_names_out(cat_cols))
                    fidf = pd.DataFrame({"feature": [feature_names[i] for i in top_idx], "mean_abs_shap": shap_sum[top_idx]})
                    st.plotly_chart(px.bar(fidf, x="mean_abs_shap", y="feature", orientation="h"), use_container_width=True)
                except Exception as e:
                    st.info(f"SHAP failed: {e}")
            else:
                st.info("Not enough samples for SHAP.")

    pred_out = df_f.loc[te.index, ["Booking ID", "timestamp", "Vehicle Type", "Pickup Location", "Drop Location", "Payment Method"]].copy()
    pred_out["will_complete_true"] = y_test.values
    pred_out["will_complete_pred"] = y_pred
    pred_out["risk_score"] = 1 - y_prob
    st.download_button("Download Predictions (CSV)", pred_out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

    st.markdown("---")
    st.markdown("### B) Forecasting – Demand (Daily)")
    ts = df_f.set_index("timestamp").resample("D").size().reset_index(name="y").rename(columns={"timestamp": "ds"})
    periods = st.slider("Forecast Horizon (days)", 7, 60, 14)
    hist, fc = train_forecast(ts, fcast_choice, periods=periods)

    if fcast_choice == "Prophet" and ensure_prophet() is not None and fc is not None:
        fig = px.line(fc, x="ds", y="yhat", title="Forecast", color_discrete_sequence=[DEMAND_COLOR])
        if "yhat_lower" in fc.columns:
            fig.add_traces([
                go.Scatter(x=fc["ds"], y=fc["yhat_upper"], line=dict(width=0), showlegend=False),
                go.Scatter(x=fc["ds"], y=fc["yhat_lower"], line=dict(width=0), fill="tonexty",
                           fillcolor="rgba(31,119,180,0.2)", showlegend=False)
            ])
        st.plotly_chart(fig, use_container_width=True)
    elif fc is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["ds"], y=hist["yhat"], name="History"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast"))
        if "yhat_lower" in fc.columns:
            fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(31,119,180,0.2)", showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No forecast generated.")

    if fc is not None:
        fcsv = (fc if "yhat" in fc.columns else hist).to_csv(index=False).encode("utf-8")
        st.download_button("Download Forecast (CSV)", fcsv, "forecast.csv", "text/csv")

    st.markdown("---")
    st.markdown("### C) Clustering – Customer Segmentation")

    # ---- Build base customer metrics
    cust = df_f.groupby("Customer ID", observed=False).agg(
        freq=("Booking ID", "count"),
        # Mean booking value for completed rides only (per customer)
        avg_value=("booking_value", lambda s: s[df_f.loc[s.index, "booking_status_canon"].eq("Completed")].mean()),
        avg_distance=("ride_distance", "mean"),
        cancel_rate=("will_complete", lambda s: 1.0 - s.mean()),
        # Most frequent payment (fallback to "Unknown")
        u_payment=("Payment Method", lambda s: (s.astype(str).mode().iloc[0] if len(s) and not s.mode().empty else "Unknown")),
    ).reset_index()

    # ---- Payment-method share matrix (row-normalized)
    pm_counts = df_f.pivot_table(
        index="Customer ID",
        columns="Payment Method",
        values="Booking ID",
        aggfunc="count",
        fill_value=0,
    )
    denom = pm_counts.sum(axis=1).replace(0, np.nan)        # avoid 0/0
    pm_share = pm_counts.div(denom, axis=0).reset_index()   # shares in [0,1]; NaN where denom=0

    # ---- Merge; fill only numeric columns (avoid categorical/object fillna TypeError)
    cust = cust.merge(pm_share, on="Customer ID", how="left")
    num_cols_cust = cust.select_dtypes(include=[np.number]).columns
    cust[num_cols_cust] = cust[num_cols_cust].fillna(0)

    # Ensure u_payment is clean text (no NaN strings)
    if "u_payment" in cust.columns:
        cust["u_payment"] = cust["u_payment"].astype(str).replace({"nan": "Unknown", "NaN": "Unknown"})

    # ---- Feature matrix for clustering
    payment_cols = [c for c in pm_share.columns if c != "Customer ID"]
    feat_cols = ["freq", "avg_value", "avg_distance", "cancel_rate"] + payment_cols

    # Coerce to numeric just in case and fill any residual NaNs with 0
    Xc = cust[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Scale + cluster
    scaler = StandardScaler()
    Xc_scaled = scaler.fit_transform(Xc)

    if clus_choice == "K-Means":
        k = st.slider("K (clusters)", 2, 10, 4)
        clus = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = clus.fit_predict(Xc_scaled)
    elif clus_choice == "DBSCAN":
        eps = st.slider("DBSCAN eps", 0.3, 5.0, 1.5)
        min_samples = st.slider("min_samples", 3, 30, 10)
        clus = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clus.fit_predict(Xc_scaled)
    else:
        k = st.slider("GMM components", 2, 10, 4)
        clus = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
        labels = clus.fit_predict(Xc_scaled)

    cust["cluster"] = labels

    # Quality metric (when applicable)
    if len(set(labels)) > 1 and -1 not in set(labels):
        try:
            sil = silhouette_score(Xc_scaled, labels)
            st.metric("Silhouette Score", f"{sil:.3f}")
        except Exception:
            pass

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xc_scaled)
    viz = pd.DataFrame({"pc1": Xp[:, 0], "pc2": Xp[:, 1], "cluster": labels})
    st.plotly_chart(px.scatter(viz, x="pc1", y="pc2", color="cluster", title="Cluster Scatter (PCA)"), use_container_width=True)

    st.markdown("#### Cluster Personas")
    personas = cust.groupby("cluster", observed=False).agg(
        n=("Customer ID", "count"),
        freq=("freq", "mean"),
        avg_value=("avg_value", "mean"),
        avg_distance=("avg_distance", "mean"),
        cancel_rate=("cancel_rate", "mean"),
    ).round(2).reset_index()
    st.dataframe(personas, use_container_width=True)

    clus_csv = cust[["Customer ID", "cluster"] + feat_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download Clusters (CSV)", clus_csv, "clusters.csv", "text/csv")

    st.markdown("---")
    st.markdown("### D) Regression – Predict Booking Value")

    Xr = df_f[[
        "Vehicle Type", "Pickup Location", "Drop Location", "Payment Method",
        "hour", "weekday", "month", "is_weekend", "time_bucket",
        "avg_vtat", "ride_distance"
    ]].copy()
    yr = df_f["booking_value"].fillna(0)

    for c in ["Pickup Location", "Drop Location"]:
        Xr[c] = compress_categories(Xr[c].astype(str), top_n=30)
    for c in ["Vehicle Type", "Pickup Location", "Drop Location", "Payment Method", "time_bucket"]:
        Xr[c] = Xr[c].astype(str)

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, shuffle=False)

    pre_r = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), ["hour", "weekday", "month", "is_weekend", "avg_vtat", "ride_distance"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Vehicle Type", "Pickup Location", "Drop Location", "Payment Method", "time_bucket"])
    ])

    reg = regression_models(regr_choice)
    rpipe = Pipeline([("pre", pre_r), ("regr", reg)])
    with st.spinner("Training regressor..."):
        rpipe.fit(Xr_train, yr_train)
    yhat = rpipe.predict(Xr_test)

    rmse = mean_squared_error(yr_test, yhat, squared=False)
    mae = mean_absolute_error(yr_test, yhat)
    r2 = r2_score(yr_test, yhat)
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:,.2f}")
    c2.metric("MAE", f"{mae:,.2f}")
    c3.metric("R²", f"{r2:.3f}")

    fig = px.scatter(x=yr_test, y=yhat, labels={"x": "Actual", "y": "Predicted"}, title="Predicted vs Actual")
    st.plotly_chart(fig, use_container_width=True)

    fig_res, ax = plt.subplots()
    ax.hist(yr_test - yhat, bins=40)
    ax.set_title("Residuals")
    st.pyplot(fig_res, use_container_width=True)

    regr_out = df_f.loc[Xr_test.index, ["Booking ID", "timestamp", "Vehicle Type", "Pickup Location", "Drop Location", "Payment Method"]].copy()
    regr_out["actual_value"] = yr_test.values
    regr_out["pred_value"] = yhat
    st.download_button("Download Regression Predictions (CSV)", regr_out.to_csv(index=False).encode("utf-8"), "regression_predictions.csv", "text/csv")


with tabs[9]:
    st.markdown("## Reports & Exports")

    comp_rate = (df_f["will_complete"] == 1).mean()
    cust_cxl_rate = (df_f["booking_status_canon"] == "Customer Cancelled").mean()
    drv_cxl_rate = (df_f["booking_status_canon"] == "Driver Cancelled").mean()
    nd_rate = (df_f["booking_status_canon"] == "No Driver Found").mean()
    rev = df_f.loc[df_f["will_complete"] == 1, "booking_value"].sum()

    html = f"""
    <html>
    <head><meta charset="utf-8"><title>Uber NCR 2024 – Summary</title></head>
    <body style="font-family:Inter,Arial,sans-serif;">
      <h2>Uber NCR 2024 – Summary (Filtered)</h2>
      <p><strong>Total Bookings:</strong> {len(df_f):,}</p>
      <ul>
        <li><strong>Completion Rate:</strong> {comp_rate:.1%}</li>
        <li><strong>Customer Cancel %:</strong> {cust_cxl_rate:.1%}</li>
        <li><strong>Driver Cancel %:</strong> {drv_cxl_rate:.1%}</li>
        <li><strong>No Driver Found %:</strong> {nd_rate:.1%}</li>
        <li><strong>Total Revenue (Completed):</strong> ₹ {rev:,.0f}</li>
      </ul>
      <h3>Managerial Implications</h3>
      <ol>
        <li>Target high-cancellation micro-windows (vehicle × time bucket) with surge supply and queue SLAs.</li>
        <li>Use VTAT caps to curb abandonment; monitor CTAT hotspots tied to low ratings.</li>
        <li>Optimize incentives in peak locales; protect ARPR via smart pricing & mix shift.</li>
      </ol>
    </body>
    </html>
    """.strip()

    st.download_button("Download HTML Summary", html.encode("utf-8"), "summary.html", "text/html")

st.caption("© 2025 Uber NCR Analytics & Decision Lab – Streamlit single-file app (maps/screenshots skipped).")
