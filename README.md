# README.md

# 🚖 Uber India 2024 – NCR Ride Analytics & Decision Lab (Streamlit)

![status](https://img.shields.io/badge/status-production--ready-green)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![streamlit](https://img.shields.io/badge/streamlit-1.49%2B-ff4b4b)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

A consulting-grade, **single-file Streamlit app** for a **150,001-row Uber-like NCR 2024** dataset. It analyzes **completion/cancellation**, **ops efficiency**, **finance**, **ratings**, and ships an **ML Lab** (classification, forecasting, clustering, regression), **risk/fraud** detection, and an **operations simulator**.  
**Maps & screenshots are skipped** by design—lean dependencies, faster setup.

---

## 📦 Data & Schema (STRICT)

**CSV name:** `ncr_ride_bookingsv1.csv` (day-first dates)

**Columns (exact headers):**


**Sample rows (for parsing & logic):**

1. `23-03-2024,12:29:38,"CNR5884300",No Driver Found,"CID1982111",eBike,Palam Vihar,Jhilmil,0,0,0,0,0,0,0,0,0,0,0,0,0`  
2. `29-11-2024,18:01:39,"CNR1326809",Incomplete,"CID4604802",Go Sedan,Shastri Nagar,Gurgaon Sector 56,4.9,14,0,0,1,Vehicle Breakdown,237,5.73,0,0,0,0,UPI`

---

## 🛠️ Data Processing & Feature Engineering

- **Date & Time** parsed (`dayfirst=True`) and combined → `timestamp` (timezone-naive).
- Features: `hour`, `weekday` (0=Mon), `month`, `is_weekend` (Sat/Sun), `time_bucket` (Morning/Afternoon/Evening/Night).
- Numeric casts: `avg_vtat`, `avg_ctat`, `cancelled_by_customer`, `cancelled_by_driver`, `incomplete_rides`, `booking_value`, `ride_distance`, `driver_ratings`, `customer_rating`.
- Reasons: `"0"` treated as missing; standardized to **Title Case** and trimmed.
- Categoricals: `booking_status`, `vehicle_type`, `pickup_location`, `drop_location`, `payment_method`.
- **Canonical Booking Status**:
  - `Completed` (if present), `Customer Cancelled`, `Driver Cancelled`, `No Driver Found`, `Incomplete`.
- **Target** `will_complete` ∈ {1,0}; 1 = Completed; 0 = all non-completed (Customer/Driver Cancelled, No Driver Found, Incomplete).
- Invalid dates → **coerced to NaT**; rows dropped with a **warning banner**.
- **No geocoding / maps.** (No `pydeck`, no mapping file required.)

---

## 🧩 App Structure (Single File: `app.py`)

**Sidebar (global filters):** Date range, Vehicle Type, Payment Method, Booking Status, Pickup/Drop Locations; Model selectors; Download buttons.

**Tabs:**
1. Executive Overview – KPI cards, time series (daily/weekly), funnel, descriptive stats, top frequencies.
2. Ride Completion & Cancellation – rates, top reasons, breakdowns (vehicle/time bucket/pickup), auto “Insight box”.
3. Geographical & Temporal – busiest pickup/drop, peak hour/day, heat tables (no maps).
4. Operational Efficiency – VTAT/CTAT distributions, correlation matrix, VTAT-threshold impact on cancels.
5. Financial Analysis – revenue (Completed only), ARPR, revenue mix, value–distance relationship.
6. Ratings & Satisfaction – distributions; correlations; low-rating risk flags by segment.
7. Incomplete Rides – share, reasons, breakdowns.
8. ML Lab – Classification (`will_complete`), Forecasting (ARIMA/Prophet), Clustering (K-Means/DBSCAN/GMM), Regression (Booking Value).
9. Risk & Fraud – Isolation Forest anomalies.
10. Operations Simulator – tweak supply, incentives, pricing; compare baseline vs scenario.
11. Reports & Exports – filtered data/predictions/clusters/forecasts; HTML summary.

---

## 🚀 Getting Started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
