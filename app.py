# app.py
# Ultra-lean Streamlit app: no scikit-learn, no prophet/shap/xgboost/lightgbm.
from __future__ import annotations

import os
import io
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ------------------------------#
# Constants & Config
# ------------------------------#
st.set_page_config(page_title="Uber NCR 2024 — Analytics & Decision Lab (Ultra-Lean)", page_icon="🚖", layout="wide")

RANDOM_STATE = 42
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

INSIGHT_COLOR = "#5e60ce"
DEMAND_COLOR = "#1f77b4"
RISK_COLOR = "#e76f51"
FIN_COLOR = "#2a9d8f"
CX_COLOR = "#9b5de5"

np.random.seed(RANDOM_STATE)

# ------------------------------#
# Utilities
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
    inc = row.get("Incomplete Rides", 0)
    if raw.lower() == "completed":
        return "Completed"
    if pd.to_numeric(pd.Series([cust_cxl]), errors="coerce").iloc[0] > 0 or "customer" in raw.lower():
        return "Customer Cancelled"
    if pd.to_numeric(pd.Series([drv_cxl]), errors="coerce").iloc[0] > 0 or "driver" in raw.lower():
        return "Driver Cancelled"
    if "no driver found" in raw.lower():
        return "No Driver Found"
    if pd.to_numeric(pd.Series([inc]), errors="coerce").iloc[0] > 0 or "incomplete" in raw.lower():
        return "Incomplete"
    return raw if raw in CANONICAL_STATUSES else raw

def revenue_mask_for_completed(status: pd.Series) -> pd.Series:
    return status == "Completed"

def find_default_csv() -> Optional[str]:
    for p in [
        "ncr_ride_bookingsv1.csv",
        "ncr_ride_bookingsv1",
        "/mount/src/uber_v2/ncr_ride_bookingsv1.csv",
        "/mount/src/uber_v2/ncr_ride_bookingsv1",
        "/mnt/data/ncr_ride_bookingsv1.csv",
        "/mnt/data/ncr_ride_bookingsv1",
    ]:
        if os.path.exists(p):
            return p
    return None

# ------------------------------#
# Data I/O & Processing
# ------------------------------#
@st.cache_data(show_spinner=False)
def load_csv(file: io.BytesIO | str) -> pd.DataFrame:
    df = pd.read_csv(file)
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    msgs: List[str] = []
    df = df.copy()

    # Parse Date/Time -> timestamp
    df["parsed_date"] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df["parsed_time"] = pd.to_datetime(df[TIME_COL], format="%H:%M:%S", errors="coerce").dt.time

    def build_ts(r):
        if pd.isna(r["parsed_date"]) or pd.isna(r["parsed_time"]):
            return pd.NaT
        return pd.Timestamp.combine(r["parsed_date"].date(), r["parsed_time"])

    df["timestamp"] = df.apply(build_ts, axis=1)
    invalid = int(df["timestamp"].isna().sum())
    if invalid > 0:
        msgs.append(f"⚠️ Dropped {invalid} rows with invalid Date/Time.")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Features
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["time_bucket"] = df["hour"].apply(time_bucket_from_hour)

    # Numerics
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

    # Reasons
    df["reason_customer"] = df["Reason for cancelling by Customer"].map(_title_case_or_nan)
    df["reason_driver"] = df["Driver Cancellation Reason"].map(_title_case_or_nan)
    df["reason_incomplete"] = df["Incomplete Rides Reason"].map(_title_case_or_nan)

    # Canonical status & target
    df["booking_status_canon"] = df.apply(canonical_status, axis=1)
    df["will_complete"] = (df["booking_status_canon"] == "Completed").astype(int)

    # Categories
    for c in ["Booking Status", "booking_status_canon", "Vehicle Type", "Pickup Location",
              "Drop Location", "Payment Method", "time_bucket"]:
        df[c] = df[c].astype("category")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, msgs

# ------------------------------#
# UI helpers
# ------------------------------#
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
    comp = int((df["booking_status_canon"] == "Completed").sum())
    cust_cxl = int((df["booking_status_canon"] == "Customer Cancelled").sum())
    drv_cxl = int((df["booking_status_canon"] == "Driver Cancelled").sum())
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
        df = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)]

    vtypes = st.sidebar.multiselect("Vehicle Type", sorted(df["Vehicle Type"].dropna().unique().tolist()))
    if vtypes: df = df[df["Vehicle Type"].isin(vtypes)]

    pm = st.sidebar.multiselect("Payment Method", sorted(df["Payment Method"].dropna().unique().tolist()))
    if pm: df = df[df["Payment Method"].isin(pm)]

    bs = st.sidebar.multiselect("Booking Status", sorted(df["booking_status_canon"].dropna().unique().tolist()))
    if bs: df = df[df["booking_status_canon"].isin(bs)]

    pls = st.sidebar.multiselect("Pickup Location", sorted(df["Pickup Location"].dropna().unique().tolist()))
    if pls: df = df[df["Pickup Location"].isin(pls)]

    dls = st.sidebar.multiselect("Drop Location", sorted(df["Drop Location"].dropna().unique().tolist()))
    if dls: df = df[df["Drop Location"].isin(dls)]

    return df

def empty_state(df: pd.DataFrame) -> bool:
    if df.empty:
        st.info("No rows match the current filters. Adjust filters in the sidebar.")
        return True
    return False

# ------------------------------#
# Minimal ML/Math helpers (no sklearn)
# ------------------------------#
def time_aware_split_idx(n: int, test_size: float = 0.2):
    cut = int((1 - test_size) * n)
    train_idx = np.arange(0, cut)
    test_idx  = np.arange(cut, n)
    return train_idx, test_idx

def one_hot_fit_transform(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    # compress high-cardinality cats
    for c in ["Pickup Location", "Drop Location"]:
        if c in df.columns:
            df[c] = compress_categories(df[c].astype(str), top_n=30)
    for c in cat_cols:
        df[c] = df[c].astype(str)
    X = pd.get_dummies(df[cat_cols], drop_first=False)
    if num_cols:
        X = pd.concat([X, df[num_cols].astype(float)], axis=1)
    columns = X.columns.tolist()
    return X, columns

def one_hot_transform(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str], columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in ["Pickup Location", "Drop Location"]:
        if c in df.columns:
            df[c] = compress_categories(df[c].astype(str), top_n=30)
    for c in cat_cols:
        df[c] = df[c].astype(str)
    X = pd.get_dummies(df[cat_cols], drop_first=False)
    if num_cols:
        X = pd.concat([X, df[num_cols].astype(float)], axis=1)
    # align to training columns
    for col in columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X[columns].astype(float)
    return X

def zscore_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    return (X - mu) / sd, mu, sd

def kmeans_numpy(X: np.ndarray, k: int, max_iter: int = 100, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    # kmeans++ init
    centers = []
    centers.append(X[rng.integers(0, n)])
    for _ in range(1, k):
        dists = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        probs = dists / (dists.sum() + 1e-9)
        idx = rng.choice(n, p=probs)
        centers.append(X[idx])
    centers = np.array(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # assign
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # update
        for j in range(k):
            pts = X[labels == j]
            if len(pts) > 0:
                centers[j] = pts.mean(axis=0)
    return labels, centers

def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # rank-based AUC
    order = np.argsort(y_score)
    y = y_true[order]
    cum_pos = np.cumsum(y)
    total_pos = cum_pos[-1]
    total_neg = len(y) - total_pos
    if total_pos == 0 or total_neg == 0:
        return np.nan
    # sum of ranks for positives
    ranks = np.arange(1, len(y) + 1)
    sum_ranks_pos = (ranks * y).sum()
    auc = (sum_ranks_pos - total_pos * (total_pos + 1) / 2) / (total_pos * total_neg)
    return float(auc)

def confusion_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return np.array([[tn, fp], [fn, tp]])

def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_binary(y_true, y_pred)
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return 2 * p * r / (p + r + 1e-9)

def pca_numpy(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt.T[:, :n_components]

# ------------------------------#
# Sidebar — data input
# ------------------------------#
st.sidebar.title("🚖 Uber NCR 2024 — Analytics & Decision Lab (Ultra-Lean)")
st.sidebar.markdown("**Data Source**")
source = st.sidebar.radio("Choose source", ["Auto-detect file", "Upload CSV"], index=0)

df_raw = None
try:
    if source == "Auto-detect file":
        path = find_default_csv()
        if path:
            df_raw = load_csv(path)
        else:
            st.sidebar.warning("Default file not found. Please upload the CSV.")
    if df_raw is None:
        uploaded = st.sidebar.file_uploader("Upload `ncr_ride_bookingsv1.csv`", type=["csv", "txt"])
        if uploaded:
            df_raw = load_csv(uploaded)
except Exception as e:
    st.sidebar.error(f"Failed to load CSV: {e}")

if df_raw is None:
    st.stop()

df, load_msgs = preprocess(df_raw)
for m in load_msgs:
    st.warning(m)

# Sidebar filters & downloads
df_f = filter_block(df)
if empty_state(df_f):
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Downloads**")
st.sidebar.download_button("Download Filtered Data (CSV)", df_f.to_csv(index=False).encode("utf-8"), "filtered_data.csv", "text/csv")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Preferences (Lean)")
clf_choice = st.sidebar.selectbox("Classifier", ["Logistic (GLM)"], index=0)
fcast_choice = st.sidebar.selectbox("Forecaster", ["ARIMA"], index=0)
clus_choice = st.sidebar.selectbox("Clustering", ["K-Means (NumPy)"], index=0)
regr_choice = st.sidebar.selectbox("Regressor", ["OLS"], index=0)

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
    base = df_f.set_index("timestamp").assign(
        revenue=lambda x: x["booking_value"].where(x["booking_status_canon"] == "Completed", 0)
    )
    freq = st.selectbox("Aggregation", ["Daily", "Weekly"], index=0, key="ts_freq")
    if freq == "Daily":
        s = base.resample("D").agg(bookings=("Booking ID", "count"), revenue=("revenue", "sum"))
    else:
        s = base.resample("W-SUN").agg(bookings=("Booking ID", "count"), revenue=("revenue", "sum"))
    s = s.reset_index()
    st.plotly_chart(px.line(s, x="timestamp", y="bookings", title="Bookings Over Time", markers=True,
                            color_discrete_sequence=[DEMAND_COLOR]), use_container_width=True)
    st.plotly_chart(px.line(s, x="timestamp", y="revenue", title="Revenue Over Time (Completed)", markers=True,
                            color_discrete_sequence=[FIN_COLOR]), use_container_width=True)

    st.markdown("---")
    total = len(df_f)
    completed = int((df_f["booking_status_canon"] == "Completed").sum())
    rated = int(df_f["customer_rating"].replace(0, np.nan).notna().sum())
    fig = go.Figure(go.Funnel(y=["Booked", "Completed", "Rated"], x=[total, completed, rated], textinfo="value+percent previous"))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Descriptive Stats")
        rows = []
        for c in ["ride_distance", "booking_value", "driver_ratings", "customer_rating"]:
            s = df_f[c].replace(0, np.nan)
            rows.append({"Metric": c.replace("_", " ").title(), "Mean": np.nanmean(s), "Median": np.nanmedian(s),
                         "Mode": (s.mode().iloc[0] if s.dropna().size else np.nan)})
        st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True)
    with c2:
        st.markdown("### Top Frequencies")
        bar_from_series(df_f["Vehicle Type"].value_counts().head(10), "Vehicle Type (Top 10)", "Vehicle Type")
        bar_from_series(df_f["Pickup Location"].value_counts().head(10), "Pickup Location (Top 10)", "Pickup Location")
        bar_from_series(df_f["Payment Method"].value_counts().head(10), "Payment Method (Top 10)", "Payment Method")

    spike = df_f.groupby("time_bucket")["Booking ID"].count().sort_values(ascending=False).head(1)
    if len(spike) > 0:
        insight_box(f"**Demand peaks in {spike.index[0]}**. Rebalance supply/incentives to curb 'No Driver Found' & cancellations.")

# ---------- Tab 2
with tabs[1]:
    st.markdown("## Ride Completion & Cancellation")
    total = len(df_f)
    comp = int((df_f["booking_status_canon"] == "Completed").sum())
    cust_cxl = int((df_f["booking_status_canon"] == "Customer Cancelled").sum())
    drv_cxl = int((df_f["booking_status_canon"] == "Driver Cancelled").sum())
    nd_found = int((df_f["booking_status_canon"] == "No Driver Found").sum())
    inc = int((df_f["booking_status_canon"] == "Incomplete").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Completion %", f"{(comp/total*100):.1f}%")
    c2.metric("Customer Cancel %", f"{(cust_cxl/total*100):.1f}%")
    c3.metric("Driver Cancel %", f"{(drv_cxl/total*100):.1f}%")
    c4.metric("No Driver Found %", f"{(nd_found/total*100):.1f}%")
    c5.metric("Incomplete %", f"{(inc/total*100):.1f}%")

    st.markdown("### Top Reasons")
    rc = df_f["reason_customer"].value_counts().head(15)
    rd = df_f["reason_driver"].value_counts().head(15)
    ri = df_f["reason_incomplete"].value_counts().head(15)
    co1, co2, co3 = st.columns(3)
    with co1: bar_from_series(rc, "Customer Reasons", "Reason", color=RISK_COLOR)
    with co2: bar_from_series(rd, "Driver Reasons", "Reason", color=RISK_COLOR)
    with co3: bar_from_series(ri, "Incomplete Reasons", "Reason", color=RISK_COLOR)

    st.markdown("---")
    st.markdown("### Cancellation Rate by Vehicle / Time Bucket / Pickup")
    by_vehicle = (df_f.assign(is_cancel=(df_f["will_complete"] == 0))
                  .groupby("Vehicle Type", observed=False)["is_cancel"].mean().sort_values(ascending=False))
    by_bucket = (df_f.assign(is_cancel=(df_f["will_complete"] == 0))
                 .groupby("time_bucket", observed=False)["is_cancel"].mean().sort_values(ascending=False))
    by_pickup = (df_f.assign(is_cancel=(df_f["will_complete"] == 0))
                 .groupby("Pickup Location", observed=False)["is_cancel"].mean().sort_values(ascending=False).head(20))

    bar_from_series(by_vehicle, "Cancellation Rate by Vehicle Type", "Vehicle Type", "Rate")
    bar_from_series(by_bucket, "Cancellation Rate by Time Bucket", "Time Bucket", "Rate")
    bar_from_series(by_pickup, "Cancellation Rate by Pickup (Top 20)", "Pickup Location", "Rate")

    if len(by_vehicle) and len(by_bucket):
        insight_box(f"Highest cancellation propensity: **{by_vehicle.index[0]}** × **{by_bucket.index[0]}**. Target policies & VTAT caps here.")

# ---------- Tab 3
with tabs[2]:
    st.markdown("## Geographical & Temporal (No Maps)")
    st.markdown("### Busiest Locations")
    top_pick = df_f["Pickup Location"].value_counts().head(20).rename("count").reset_index().rename(columns={"index": "Pickup Location"})
    top_drop = df_f["Drop Location"].value_counts().head(20).rename("count").reset_index().rename(columns={"index": "Drop Location"})
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(top_pick, use_container_width=True)
        st.plotly_chart(px.bar(top_pick, x="Pickup Location", y="count", title="Top Pickups",
                               color_discrete_sequence=[DEMAND_COLOR]), use_container_width=True)
    with c2:
        st.dataframe(top_drop, use_container_width=True)
        st.plotly_chart(px.bar(top_drop, x="Drop Location", y="count", title="Top Drops",
                               color_discrete_sequence=[DEMAND_COLOR]), use_container_width=True)

    st.markdown("---")
    st.markdown("### Peak Patterns")
    hh = df_f["hour"].value_counts().sort_index()
    dow = df_f["weekday"].value_counts().sort_index()
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.bar(hh, title="By Hour of Day", labels={"index": "Hour", "value": "Trips"},
                               color_discrete_sequence=[DEMAND_COLOR]), use_container_width=True)
    with c4:
        dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        st.plotly_chart(px.bar(dow.rename(index=dow_map), title="By Day of Week", labels={"index": "Day", "value": "Trips"},
                               color_discrete_sequence=[DEMAND_COLOR]), use_container_width=True)

    st.markdown("---")
    st.markdown("### Category Heat Tables")
    heat_pick_hr = (df_f.assign(cnt=1).pivot_table(index="Pickup Location", columns="hour", values="cnt", aggfunc="sum", fill_value=0))
    heat_pick_hr = heat_pick_hr.loc[heat_pick_hr.sum(axis=1).sort_values(ascending=False).head(20).index]
    st.plotly_chart(px.imshow(heat_pick_hr, aspect="auto", color_continuous_scale="Blues",
                              title="Pickup × Hour Heat (Top 20 Pickups)"), use_container_width=True)
    heat_pick_dow = (df_f.assign(cnt=1).pivot_table(index="Pickup Location", columns="weekday", values="cnt", aggfunc="sum", fill_value=0))
    heat_pick_dow = heat_pick_dow.loc[heat_pick_dow.sum(axis=1).sort_values(ascending=False).head(20).index]
    heat_pick_dow.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    st.plotly_chart(px.imshow(heat_pick_dow, aspect="auto", color_continuous_scale="Blues",
                              title="Pickup × Day-of-Week Heat (Top 20 Pickups)"), use_container_width=True)

# ---------- Tab 4
with tabs[3]:
    st.markdown("## Operational Efficiency")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df_f, x="avg_vtat", nbins=40, title="Avg VTAT",
                                     color_discrete_sequence=[RISK_COLOR]), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(df_f, x="avg_ctat", nbins=40, title="Avg CTAT",
                                     color_discrete_sequence=[RISK_COLOR]), use_container_width=True)

    st.markdown("---")
    st.markdown("### By Location & Vehicle (Top 30 by volume)")
    gv = df_f.groupby(["Pickup Location", "Vehicle Type"], observed=False).agg(
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

    vt_series = df_f["avg_vtat"].dropna()
    vt_min, vt_max = (0.0, 1.0) if vt_series.empty else (float(vt_series.min()), float(vt_series.max()))
    vt_default = 0.5 if vt_series.empty else float(np.nanpercentile(vt_series, 80))
    vt_thresh = st.slider("VTAT threshold (minutes)", min_value=vt_min, max_value=vt_max, value=vt_default)
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
    grp = df_f[rev_mask].groupby(["Payment Method", "Vehicle Type"], observed=False)["booking_value"].sum().reset_index()
    st.plotly_chart(px.bar(grp, x="Payment Method", y="booking_value", color="Vehicle Type", barmode="stack",
                           title="Revenue by Payment Method & Vehicle",
                           color_discrete_sequence=px.colors.qualitative.Set2), use_container_width=True)

    st.markdown("---")
    st.plotly_chart(px.scatter(df_f[rev_mask], x="ride_distance", y="booking_value", color="Vehicle Type",
                               trendline="ols", title="Booking Value vs Ride Distance"),
                    use_container_width=True)

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
    st.markdown("### Correlations & Low-Rating Risk")
    rtab = df_f[["driver_ratings", "avg_vtat", "cancelled_by_driver"]].corr().iloc[0, 1:].to_frame("corr")
    st.dataframe(rtab.round(2), use_container_width=True)

    low_thr = st.slider("Low rating threshold", 1.0, 5.0, 3.5, 0.1)
    seg = (df_f.assign(low_rate=(df_f["customer_rating"] > 0) & (df_f["customer_rating"] < low_thr))
           .groupby(["Vehicle Type", "time_bucket"], observed=False)["low_rate"].mean().reset_index()
           .sort_values("low_rate", ascending=False).head(20))
    st.plotly_chart(px.bar(seg, x="low_rate", y="Vehicle Type", color="time_bucket", orientation="h",
                           title=f"Probability of < {low_thr:.1f} Stars by Segment"), use_container_width=True)

# ---------- Tab 7
with tabs[6]:
    st.markdown("## Incomplete Rides")
    inc_df = df_f[df_f["booking_status_canon"] == "Incomplete"]
    share = len(inc_df) / len(df_f) if len(df_f) else 0
    st.metric("Incomplete Share", f"{share:.2%}")
    bar_from_series(inc_df["reason_incomplete"].value_counts().head(20), "Top Incomplete Reasons", "Reason", color=RISK_COLOR)
    c1, c2 = st.columns(2)
    with c1:
        bar_from_series(inc_df["Pickup Location"].value_counts().head(20), "Incomplete by Pickup (Top 20)", "Pickup Location", color=DEMAND_COLOR)
    with c2:
        bar_from_series(inc_df["Vehicle Type"].value_counts().head(20), "Incomplete by Vehicle", "Vehicle Type", color=DEMAND_COLOR)

# ------------------------------#
# Tab 8 — ML Lab (lean, no sklearn)
# ------------------------------#
with tabs[7]:
    st.markdown("## ML Lab (Lean)")

    # ---------------- A) Classification — GLM (Logit)
    st.markdown("### A) Classification — Predict `will_complete` (GLM Logistic)")

    cat_cols = ["Vehicle Type", "Pickup Location", "Drop Location", "Payment Method", "time_bucket"]
    num_cols = ["hour", "weekday", "month", "is_weekend", "avg_vtat", "ride_distance"]

    df_model = df_f[cat_cols + num_cols + ["will_complete"]].copy()
    X_all, cols = one_hot_fit_transform(df_model, cat_cols, num_cols)
    y_all = df_model["will_complete"].astype(int).values

    n = len(X_all)
    train_idx, test_idx = time_aware_split_idx(n, test_size=0.2)
    X_train, X_test = X_all.iloc[train_idx].values, X_all.iloc[test_idx].values
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # Statsmodels GLM
    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_test_sm  = sm.add_constant(X_test,  has_constant="add")

    try:
        glm = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
        res = glm.fit(maxiter=200)
        y_prob = res.predict(X_test_sm)
    except Exception as e:
        st.error(f"GLM failed: {e}")
        y_prob = np.full_like(y_test, y_test.mean(), dtype=float)

    y_pred = (y_prob >= 0.5).astype(int)
    acc = float((y_pred == y_test).mean())
    f1  = f1_binary(y_test, y_pred)
    try:
        auc = auc_score(y_test, y_prob)
    except Exception:
        auc = np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("F1", f"{f1:.3f}")
    c3.metric("ROC AUC", f"{auc:.3f}" if not np.isnan(auc) else "—")
    c4.metric("Test Size", f"{len(y_test):,}")

    # Confusion matrix
    cm = confusion_binary(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
    fig_cm.update_xaxes(title="Predicted"); fig_cm.update_yaxes(title="Actual")
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC (simple)
    try:
        thresholds = np.linspace(0, 1, 101)
        tprs, fprs = [], []
        for t in thresholds:
            yp = (y_prob >= t).astype(int)
            cm_ = confusion_binary(y_test, yp)
            tn, fp, fn, tp = cm_[0,0], cm_[0,1], cm_[1,0], cm_[1,1]
            tprs.append(tp / (tp + fn + 1e-9))
            fprs.append(fp / (fp + tn + 1e-9))
        fig, ax = plt.subplots()
        ax.plot(fprs, tprs, label=f"AUC={auc:.3f}" if not np.isnan(auc) else "AUC=—")
        ax.plot([0, 1], [0, 1], "--", alpha=0.5)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve"); ax.legend()
        st.pyplot(fig, use_container_width=True)
    except Exception:
        pass

    pred_out = df_f.iloc[test_idx][["Booking ID", "timestamp", "Vehicle Type", "Pickup Location", "Drop Location", "Payment Method"]].copy()
    pred_out["will_complete_true"] = y_test
    pred_out["will_complete_pred"] = y_pred
    pred_out["risk_score"] = 1 - y_prob
    st.download_button("Download Predictions (CSV)", pred_out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

    # ---------------- B) Forecasting — ARIMA
    st.markdown("---")
    st.markdown("### B) Forecasting — Demand (Daily, ARIMA)")
    ts = df_f.set_index("timestamp").resample("D").size().reset_index(name="y").rename(columns={"timestamp": "ds"})
    periods = st.slider("Forecast Horizon (days)", 7, 60, 14)
    try:
        y = ts.set_index("ds")["y"].asfreq("D").fillna(0)
        model = sm.tsa.ARIMA(y, order=(2, 1, 2))
        res = model.fit()
        fc = res.get_forecast(steps=periods)
        fc_df = pd.DataFrame({
            "ds": pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D"),
            "yhat": fc.predicted_mean.values
        })
        conf = fc.conf_int()
        fc_df["yhat_lower"] = conf.iloc[:, 0].values
        fc_df["yhat_upper"] = conf.iloc[:, 1].values
        hist = pd.DataFrame({"ds": y.index, "yhat": y.values})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["ds"], y=hist["yhat"], name="History"))
        fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat"], name="Forecast"))
        fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat_upper"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat_lower"], line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(31,119,180,0.2)", showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download Forecast (CSV)", fc_df.to_csv(index=False).encode("utf-8"), "forecast.csv", "text/csv")
    except Exception as e:
        st.info(f"Forecast failed: {e}")

    # ---------------- C) Clustering — KMeans (NumPy)
    st.markdown("---")
    st.markdown("### C) Clustering — Customer Segmentation (KMeans, NumPy)")

    cust = df_f.groupby("Customer ID", observed=False).agg(
        freq=("Booking ID", "count"),
        avg_value=("booking_value", lambda s: s[df_f.loc[s.index, "booking_status_canon"].eq("Completed")].mean()),
        avg_distance=("ride_distance", "mean"),
        cancel_rate=("will_complete", lambda s: 1.0 - s.mean()),
        u_payment=("Payment Method", lambda s: (s.astype(str).mode().iloc[0] if len(s) and not s.mode().empty else "Unknown")),
    ).reset_index()

    pm_counts = df_f.pivot_table(index="Customer ID", columns="Payment Method", values="Booking ID", aggfunc="count", fill_value=0)
    denom = pm_counts.sum(axis=1).replace(0, np.nan)
    pm_share = pm_counts.div(denom, axis=0).reset_index()

    cust = cust.merge(pm_share, on="Customer ID", how="left")
    num_cols_cust = cust.select_dtypes(include=[np.number]).columns
    cust[num_cols_cust] = cust[num_cols_cust].fillna(0)
    if "u_payment" in cust.columns:
        cust["u_payment"] = cust["u_payment"].astype(str).replace({"nan": "Unknown", "NaN": "Unknown"})

    payment_cols = [c for c in pm_share.columns if c != "Customer ID"]
    feat_cols = ["freq", "avg_value", "avg_distance", "cancel_rate"] + payment_cols
    Xc = cust[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    Xc_scaled, mu_c, sd_c = zscore_scale(Xc)

    k = st.slider("K (clusters)", 2, 10, 4)
    labels, centers = kmeans_numpy(Xc_scaled, k=k, random_state=RANDOM_STATE)
    cust["cluster"] = labels

    # PCA scatter
    try:
        Xp = pca_numpy(Xc_scaled, n_components=2)
        viz = pd.DataFrame({"pc1": Xp[:, 0], "pc2": Xp[:, 1], "cluster": labels})
        st.plotly_chart(px.scatter(viz, x="pc1", y="pc2", color="cluster", title="Cluster Scatter (PCA)"),
                        use_container_width=True)
    except Exception:
        st.info("PCA plot unavailable.")

    st.markdown("#### Cluster Personas")
    personas = cust.groupby("cluster", observed=False).agg(
        n=("Customer ID", "count"),
        freq=("freq", "mean"),
        avg_value=("avg_value", "mean"),
        avg_distance=("avg_distance", "mean"),
        cancel_rate=("cancel_rate", "mean"),
    ).round(2).reset_index()
    st.dataframe(personas, use_container_width=True)
    st.download_button("Download Clusters (CSV)", cust[["Customer ID", "cluster"] + feat_cols].to_csv(index=False).encode("utf-8"),
                       "clusters.csv", "text/csv")

    # ---------------- D) Regression — OLS
    st.markdown("---")
    st.markdown("### D) Regression — Predict Booking Value (OLS)")

    df_reg = df_f[cat_cols + num_cols + ["booking_value"]].copy()
    Xr_all, r_cols = one_hot_fit_transform(df_reg, cat_cols, num_cols)
    yr_all = df_reg["booking_value"].fillna(0).astype(float).values

    n = len(Xr_all)
    tr_idx, te_idx = time_aware_split_idx(n, test_size=0.2)
    Xr_train, Xr_test = Xr_all.iloc[tr_idx].values, Xr_all.iloc[te_idx].values
    yr_train, yr_test = yr_all[tr_idx], yr_all[te_idx]

    Xr_train_sm = sm.add_constant(Xr_train, has_constant="add")
    Xr_test_sm  = sm.add_constant(Xr_test,  has_constant="add")

    try:
        ols = sm.OLS(yr_train, Xr_train_sm).fit()
        yhat = ols.predict(Xr_test_sm)
    except Exception as e:
        st.error(f"OLS failed: {e}")
        yhat = np.full_like(yr_test, yr_train.mean())

    rmse = float(np.sqrt(np.mean((yr_test - yhat) ** 2)))
    mae = float(np.mean(np.abs(yr_test - yhat)))
    ss_res = float(np.sum((yr_test - yhat) ** 2))
    ss_tot = float(np.sum((yr_test - yr_test.mean()) ** 2))
    r2 = 1 - ss_res / (ss_tot + 1e-9)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:,.2f}")
    c2.metric("MAE", f"{mae:,.2f}")
    c3.metric("R²", f"{r2:.3f}")

    st.plotly_chart(px.scatter(x=yr_test, y=yhat, labels={"x": "Actual", "y": "Predicted"}, title="Predicted vs Actual"),
                    use_container_width=True)

    fig_res, ax = plt.subplots()
    ax.hist(yr_test - yhat, bins=40)
    ax.set_title("Residuals")
    st.pyplot(fig_res, use_container_width=True)

    regr_out = df_f.iloc[te_idx][["Booking ID", "timestamp", "Vehicle Type", "Pickup Location", "Drop Location", "Payment Method"]].copy()
    regr_out["actual_value"] = yr_test
    regr_out["pred_value"] = yhat
    st.download_button("Download Regression Predictions (CSV)", regr_out.to_csv(index=False).encode("utf-8"),
                       "regression_predictions.csv", "text/csv")

# ---------- Tab 9
with tabs[8]:
    st.markdown("## Risk & Fraud (Z-score Anomalies)")
    fr_cols = [
        "avg_vtat", "avg_ctat", "ride_distance", "booking_value",
        "hour", "weekday", "is_weekend",
        "cancelled_by_customer", "cancelled_by_driver", "incomplete_rides"
    ]
    Xf = df_f[fr_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(float).values
    Xf_scaled, _, _ = zscore_scale(Xf)
    # aggregate absolute z-scores as a simple anomaly score
    scores = np.abs(Xf_scaled).mean(axis=1)
    thresh = st.slider("Anomaly threshold (aggregate Z)", 0.5, 5.0, 2.5, 0.1)
    is_anom = (scores >= thresh).astype(int)

    risk_df = df_f[["Booking ID", "timestamp", "Vehicle Type", "Pickup Location",
                    "Drop Location", "Payment Method", "booking_value", "ride_distance",
                    "booking_status_canon"]].copy()
    risk_df["risk_score"] = scores
    risk_df["is_anomaly"] = is_anom

    st.dataframe(risk_df.sort_values("risk_score", ascending=False).head(200), use_container_width=True)
    st.download_button("Download Risk Flags (CSV)", risk_df.to_csv(index=False).encode("utf-8"), "risk_flags.csv", "text/csv")

# ---------- Tab 10
with tabs[9]:
    st.markdown("## Operations Simulator")
    st.caption("""
    Simple elastic model:
    • Driver supply ↑ reduces 'No Driver Found' & driver cancels (elasticity −0.6).
    • Incentives ↑ reduce driver cancels (−0.4) and lift rating (+0.1 per +10%).
    • Pricing ↑ reduces demand (−0.8) but increases ARPR linearly.
    """)

    c1, c2, c3 = st.columns(3)
    supply_up = c1.slider("Driver Supply Δ (%)", -50, 50, 10)
    incent_up = c2.slider("Driver Incentive Δ (%)", 0, 100, 10)
    price_up  = c3.slider("Pricing Uplift Δ (%)", -20, 30, 5)

    base_total = len(df_f)
    base_complete = int((df_f["will_complete"] == 1).sum())
    base_comp_rate = base_complete / base_total if base_total else 0
    base_rev = df_f.loc[df_f["will_complete"] == 1, "booking_value"].sum()
    base_arpr = base_rev / base_complete if base_complete else 0
    base_rating = df_f["customer_rating"].replace(0, np.nan).mean()

    e_supply_cxl = -0.6
    e_incent_cxl = -0.4
    e_price_demand = -0.8

    demand_factor = max(0.0, 1 + (price_up / 100) * e_price_demand)
    cxl_factor = (1 + (supply_up / 100) * e_supply_cxl) * (1 + (incent_up / 100) * e_incent_cxl)
    cxl_factor = max(0.5, min(1.2, cxl_factor))

    scen_total = int(base_total * demand_factor)
    scen_comp_rate = min(0.995, base_comp_rate * (1 / cxl_factor))
    scen_completed = int(scen_total * scen_comp_rate)
    scen_arpr = base_arpr * (1 + price_up / 100)
    scen_rev = scen_completed * scen_arpr
    scen_rating = (base_rating if not np.isnan(base_rating) else 4.5) + 0.01 * incent_up
    scen_rating = min(5.0, scen_rating)

    st.markdown("### Baseline vs Scenario")
    compare = pd.DataFrame({
        "Metric": ["Total Bookings", "Completion Rate", "Completed Rides", "ARPR", "Total Revenue", "Avg Customer Rating"],
        "Baseline": [base_total, base_comp_rate, base_complete, base_arpr, base_rev, base_rating],
        "Scenario": [scen_total, scen_comp_rate, scen_completed, scen_arpr, scen_rev, scen_rating]
    })
    st.dataframe(compare.style.format({"Baseline": "{:,.2f}", "Scenario": "{:,.2f}"}).hide(axis="index"), use_container_width=True)
    st.plotly_chart(px.bar(compare, x="Metric", y=["Baseline", "Scenario"], barmode="group", title="Baseline vs Scenario",
                           color_discrete_sequence=px.colors.qualitative.Set2), use_container_width=True)

# ---------- Tab 11
with tabs[10]:
    st.markdown("## Reports & Exports")
    comp_rate = (df_f["will_complete"] == 1).mean()
    cust_cxl_rate = (df_f["booking_status_canon"] == "Customer Cancelled").mean()
    drv_cxl_rate = (df_f["booking_status_canon"] == "Driver Cancelled").mean()
    nd_rate = (df_f["booking_status_canon"] == "No Driver Found").mean()
    rev = df_f.loc[df_f["will_complete"] == 1, "booking_value"].sum()

    html = f"""
    <html>
    <head><meta charset="utf-8"><title>Uber NCR 2024 — Summary</title></head>
    <body style="font-family:Inter,Arial,sans-serif;">
      <h2>Uber NCR 2024 — Summary (Filtered)</h2>
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
        <li>Target high-cancellation windows (vehicle × time bucket) with supply & incentives.</li>
        <li>Cap VTAT in hotspots to reduce abandonment; monitor CTAT-linked rating dips.</li>
        <li>Optimize payment & vehicle mix to lift ARPR without eroding demand.</li>
      </ol>
    </body>
    </html>
    """.strip()

    st.download_button("Download HTML Summary", html.encode("utf-8"), "summary.html", "text/html")

st.caption("© 2025 — Ultra-lean single-file app: no sklearn/prophet/shap. Built for reliability on Streamlit Cloud.")
