"""
Turmerix — Indian Spice Price Forecast
Streamlit Community Cloud app
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Turmerix — Spice Price Forecast",
    page_icon="🌶️",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "ts_price_model.joblib"
META_PATH  = BASE_DIR / "models" / "ts_model_metadata.json"
DATA_PATH  = BASE_DIR / "data"   / "daily_spice_timeseries.csv"

PRESET_HORIZONS = {"7d": 7, "1m": 30, "3m": 90, "1y": 365}

SIGNAL_COLOR = {
    "BUY / HOLD STOCK": "🟢",
    "SELL / LIQUIDATE": "🔴",
    "HOLD":             "🟡",
}
TREND_ICON = {"upward": "📈", "downward": "📉", "sideways": "➡️"}


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    bundle   = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text())
    return bundle, metadata


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


# ── Forecast helpers ──────────────────────────────────────────────────────────
def resolve_horizon(h: str) -> int:
    return PRESET_HORIZONS[h] if h in PRESET_HORIZONS else int(h)


def buysell_signal(total_pct: float, spice_mape: float) -> dict:
    band = spice_mape * 0.5
    if abs(total_pct) < band:
        return {
            "signal": "HOLD",
            "reason": (
                f"Predicted move ({total_pct:+.1f}%) is within model uncertainty "
                f"(spice MAPE ≈ {spice_mape:.1f}%). No confident directional signal."
            ),
        }
    if total_pct > 0:
        return {
            "signal": "BUY / HOLD STOCK",
            "reason": (
                f"Price predicted to rise {total_pct:+.1f}% over the horizon. "
                f"Consider buying or holding inventory. (Spice MAPE ≈ {spice_mape:.1f}%)"
            ),
        }
    return {
        "signal": "SELL / LIQUIDATE",
        "reason": (
            f"Price predicted to fall {total_pct:+.1f}% over the horizon. "
            f"Consider selling or reducing stock. (Spice MAPE ≈ {spice_mape:.1f}%)"
        ),
    }


def run_forecast(bundle, metadata, row, spice_title, horizon_days, anchor_date_str):
    le         = bundle["label_encoder"]
    model      = bundle["model"]
    spice_enc  = int(le.transform([spice_title])[0])
    per_spice  = metadata.get("per_spice_test_mape", {})
    test_mape  = float(metadata.get("test_mape", 0.0))
    spice_mape = per_spice.get(spice_title, test_mape)

    def safe(col, default=0.0):
        val = row.get(col, default)
        return default if pd.isna(val) else float(val)

    lag1   = safe("lag1_price", safe("price_per_kg_inr_vwap"))
    lag7   = safe("lag7_price", lag1)
    r7avg  = safe("rolling7_avg_price", lag1)
    r7std  = safe("rolling7_price_std", 0.0)
    r7vol  = safe("rolling7_volume_kg", 0.0)
    r14vol = safe("rolling14_volume_kg", 0.0)
    dvol   = safe("daily_volume_kg", 0.0)
    r7avg_daily_vol = r7vol / 7.0 if r7vol > 0 else 1.0
    ship_cnt  = safe("daily_shipment_count", 1)
    buy_cnt   = safe("daily_buyer_count", 1)
    exp_cnt   = safe("daily_exporter_count", 1)

    anchor_price  = lag1
    anchor_ts     = pd.Timestamp(anchor_date_str)
    price_window  = [lag1] * 7
    forecast_days = []

    for step in range(1, horizon_days + 1):
        fdate = anchor_ts + pd.Timedelta(days=step)
        dow   = fdate.dayofweek
        mon   = fdate.month
        woy   = int(fdate.isocalendar()[1])

        mom1d  = (lag1 / price_window[-2] - 1.0) if len(price_window) >= 2 and price_window[-2] > 0 else 0.0
        mom7d  = (lag1 / lag7 - 1.0) if lag7 > 0 else 0.0
        vshock = dvol / r7avg_daily_vol if r7avg_daily_vol > 0 else 1.0

        feat = np.array([
            spice_enc, lag1, lag7, r7avg, r7std,
            mom1d, mom7d, r7vol, r14vol, vshock,
            dvol, ship_cnt, buy_cnt, exp_cnt,
            float(dow), float(mon), float(woy),
        ], dtype=np.float64).reshape(1, -1)

        log_pred  = model.predict(feat)[0]
        pred_vwap = float(max(np.expm1(log_pred), 0.0))
        pct       = (pred_vwap / anchor_price - 1.0) * 100.0

        forecast_days.append({
            "date": fdate.strftime("%Y-%m-%d"),
            "predicted_vwap_inr": round(pred_vwap, 2),
            "pct_change_vs_today": round(pct, 2),
        })

        if step >= 7:
            lag7 = price_window[0]
        price_window.append(pred_vwap)
        if len(price_window) > 7:
            price_window.pop(0)

        r7avg = float(np.mean(price_window))
        r7std = float(np.std(price_window)) if len(price_window) > 1 else 0.0
        r7avg_daily_vol = dvol
        r14vol = r7vol + dvol
        r7vol  = dvol * 7.0
        lag1   = pred_vwap

    prices    = [d["predicted_vwap_inr"] for d in forecast_days]
    final_pct = forecast_days[-1]["pct_change_vs_today"]
    trend     = "upward" if final_pct > 2 else "downward" if final_pct < -2 else "sideways"
    sig       = buysell_signal(final_pct, spice_mape)

    return {
        "anchor_price": round(anchor_price, 2),
        "spice_mape":   spice_mape,
        "horizon_days": horizon_days,
        "forecast":     forecast_days,
        "min_price":    round(min(prices), 2),
        "max_price":    round(max(prices), 2),
        "avg_price":    round(float(np.mean(prices)), 2),
        "final_pct":    round(final_pct, 2),
        "trend":        trend,
        "signal":       sig["signal"],
        "reason":       sig["reason"],
    }


def resolve_row(df, spice, anchor_date_str):
    """Return (row, used_date, context_note) for the given spice + date."""
    anchor_ts  = pd.Timestamp(anchor_date_str)
    spice_rows = df[df["Spice Name"] == spice].sort_values("date")

    if spice_rows.empty:
        return None, None, "No data for this spice."

    ds_start = spice_rows.iloc[0]["date"]
    ds_end   = spice_rows.iloc[-1]["date"]

    exact = spice_rows[spice_rows["date"] == anchor_ts]
    if not exact.empty:
        return exact.iloc[0], anchor_ts, f"Exact date {anchor_date_str} found in dataset."

    if anchor_ts < ds_start:
        return None, None, (
            f"Date {anchor_date_str} is before dataset start "
            f"({ds_start.strftime('%Y-%m-%d')}). Please choose a later date."
        )

    if anchor_ts > ds_end:
        row = spice_rows.iloc[-1]
        used = row["date"]
        return row, used, (
            f"⚠️ {anchor_date_str} is beyond the dataset (ends {ds_end.strftime('%Y-%m-%d')}). "
            f"Using latest available context: **{used.strftime('%Y-%m-%d')}**."
        )

    earlier = spice_rows[spice_rows["date"] <= anchor_ts]
    row     = earlier.iloc[-1]
    used    = row["date"]
    return row, used, (
        f"No shipments recorded for {spice} on {anchor_date_str}. "
        f"Using nearest earlier date with data: **{used.strftime('%Y-%m-%d')}**."
    )


# ── UI ─────────────────────────────────────────────────────────────────────────
def main():
    bundle, metadata = load_model()
    df               = load_data()

    spices    = sorted(metadata["spices"])
    ds_start  = df["date"].min().strftime("%Y-%m-%d")
    ds_end    = df["date"].max().strftime("%Y-%m-%d")

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🌶️ Turmerix — Indian Spice Price Forecast")
    st.markdown(
        "Predict future VWAP prices for Indian export spices and get a **Buy / Sell / Hold** signal. "
        f"Dataset: May–June 2025 · {len(spices)} spices · LightGBM time-series model"
    )
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Forecast Settings")

        spice = st.selectbox("Spice", spices, index=spices.index("Cumin"))

        anchor_date = st.date_input(
            "Anchor date (today's context)",
            value=pd.Timestamp(ds_end).date(),
            help=f"Dataset covers {ds_start} – {ds_end}. Future dates fall back to latest available.",
        ).strftime("%Y-%m-%d")

        horizon_label = st.radio(
            "Forecast horizon",
            options=["7d — 1 week", "1m — 30 days", "3m — 90 days", "1y — 365 days"],
            index=0,
        )
        horizon_key = horizon_label.split(" ")[0]

        st.divider()
        per_spice_mape = metadata.get("per_spice_test_mape", {})
        mape = per_spice_mape.get(spice, metadata.get("test_mape", 0.0))
        st.metric("Spice MAPE (test)", f"{mape:.1f}%", help="Lower = more reliable signal")
        st.caption(
            "MAPE < 25% → reliable signal\n\n"
            "MAPE 25–50% → directional only\n\n"
            "MAPE > 50% → treat as very approximate"
        )

        run = st.button("🔮 Run Forecast", use_container_width=True, type="primary")

    # ── Main panel ────────────────────────────────────────────────────────────
    if not run:
        st.info("Select a spice and horizon in the sidebar, then click **Run Forecast**.")

        # Show quick reference table
        st.subheader("📋 Spice reliability guide")
        mape_df = (
            pd.DataFrame(per_spice_mape.items(), columns=["Spice", "Test MAPE (%)"])
            .sort_values("Test MAPE (%)")
            .reset_index(drop=True)
        )
        mape_df["Reliability"] = mape_df["Test MAPE (%)"].apply(
            lambda x: "🟢 High" if x < 25 else ("🟡 Medium" if x < 50 else "🔴 Low")
        )
        st.dataframe(mape_df, use_container_width=True, hide_index=True)
        return

    # ── Resolve date → row ────────────────────────────────────────────────────
    row, used_date, context_note = resolve_row(df, spice, anchor_date)

    if row is None:
        st.error(context_note)
        return

    if anchor_date != (used_date.strftime("%Y-%m-%d") if hasattr(used_date, "strftime") else str(used_date)):
        st.warning(context_note)
    else:
        st.success(context_note)

    # ── Run forecast ──────────────────────────────────────────────────────────
    horizon_days = resolve_horizon(horizon_key)
    used_date_str = used_date.strftime("%Y-%m-%d") if hasattr(used_date, "strftime") else str(used_date)

    with st.spinner(f"Forecasting {horizon_days} days for {spice}…"):
        result = run_forecast(bundle, metadata, row, spice, horizon_days, used_date_str)

    # ── KPI row ───────────────────────────────────────────────────────────────
    sig_icon = SIGNAL_COLOR.get(result["signal"], "⚪")
    trend_icon = TREND_ICON.get(result["trend"], "")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Anchor price (INR/kg)", f"₹{result['anchor_price']:,.2f}")
    col2.metric(
        f"Final price ({horizon_days}d)",
        f"₹{result['forecast'][-1]['predicted_vwap_inr']:,.2f}",
        delta=f"{result['final_pct']:+.1f}%",
    )
    col3.metric("Avg forecast price", f"₹{result['avg_price']:,.2f}")
    col4.metric(
        "Signal",
        f"{sig_icon} {result['signal']}",
        delta=f"Trend: {trend_icon} {result['trend']}",
        delta_color="off",
    )

    # ── Signal reasoning ─────────────────────────────────────────────────────
    sig = result["signal"]
    if sig == "BUY / HOLD STOCK":
        st.success(f"**{sig_icon} {sig}** — {result['reason']}")
    elif sig == "SELL / LIQUIDATE":
        st.error(f"**{sig_icon} {sig}** — {result['reason']}")
    else:
        st.warning(f"**{sig_icon} {sig}** — {result['reason']}")

    # ── Price chart ───────────────────────────────────────────────────────────
    st.subheader(f"📊 {spice} — {horizon_days}-day VWAP forecast")

    forecast_df = pd.DataFrame(result["forecast"])
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    forecast_df = forecast_df.set_index("date")

    # Prepend anchor point so chart starts from today
    anchor_row = pd.DataFrame(
        {"predicted_vwap_inr": [result["anchor_price"]], "pct_change_vs_today": [0.0]},
        index=[pd.Timestamp(used_date_str)],
    )
    chart_df = pd.concat([anchor_row, forecast_df])

    st.line_chart(chart_df["predicted_vwap_inr"], use_container_width=True)

    # ── Min/max band ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Min forecast", f"₹{result['min_price']:,.2f}")
    c2.metric("Max forecast", f"₹{result['max_price']:,.2f}")
    c3.metric("Price range", f"₹{result['max_price'] - result['min_price']:,.2f}")

    # ── Detailed table ────────────────────────────────────────────────────────
    with st.expander("📋 Day-by-day forecast table"):
        display_df = forecast_df.reset_index().rename(columns={
            "date": "Date",
            "predicted_vwap_inr": "Predicted VWAP (INR/kg)",
            "pct_change_vs_today": "% Change vs Anchor",
        })
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        display_df["% Change vs Anchor"] = display_df["% Change vs Anchor"].apply(
            lambda x: f"{x:+.2f}%"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"Model: LightGBM auto-regressive time-series · "
        f"Dataset: May–June 2025 Indian spice exports · "
        f"Spice MAPE: {result['spice_mape']:.1f}% · "
        "For indicative use only."
    )


if __name__ == "__main__":
    main()
