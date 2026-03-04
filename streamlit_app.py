"""
Turmerix — Indian Spice Market Dashboard
MoneyControl-style market overview + forecast app
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Turmerix — Spice Market",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "ts_price_model.joblib"
META_PATH  = BASE_DIR / "models" / "ts_model_metadata.json"
DATA_PATH  = BASE_DIR / "data"   / "daily_spice_timeseries.csv"

PRESET_HORIZONS = {"7d": 7, "1m": 30, "3m": 90, "1y": 365}

SPICE_EMOJI = {
    "Cardamom": "💚", "Cumin": "🟤", "Turmeric": "🟡", "Black Pepper": "⚫",
    "Cloves": "🟫", "Ginger": "🫚", "Cinnamon": "🟠", "Saffron": "🔶",
    "Chilli": "🌶️", "Red Chilli": "🌶️", "Chilli Powder": "🌶️",
    "Coffee": "☕", "Tea": "🍵", "Coriander": "🌿", "Fennel": "🌿",
    "Fenugreek": "🌿", "Mustard": "🟡", "Pepper": "⚫", "Vanilla": "🤍",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Top header bar */
.market-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1rem 1.5rem 0.75rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
.market-header h1 { color: #e94560; margin: 0; font-size: 1.6rem; }
.market-header p  { color: #a8b2d8; margin: 0.2rem 0 0; font-size: 0.85rem; }

/* Ticker strip */
.ticker-wrap {
    background: #0d0d1a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.4rem 0;
    overflow: hidden;
    white-space: nowrap;
    margin-bottom: 1rem;
}
.ticker-track {
    display: inline-block;
    animation: ticker-scroll 40s linear infinite;
    white-space: nowrap;
}
.ticker-track:hover { animation-play-state: paused; }
@keyframes ticker-scroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.ticker-item { display: inline-block; margin-right: 2.5rem; font-size: 0.82rem; padding: 0 0.5rem; }
.ticker-name { color: #a8b2d8; font-weight: 600; }
.ticker-price { color: #e2e8f0; margin: 0 0.3rem; }
.ticker-up   { color: #22c55e; }
.ticker-down { color: #ef4444; }
.ticker-flat { color: #94a3b8; }
.ticker-sep  { color: #2d4a6e; margin-right: 2.5rem; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(145deg, #1e2a3a, #162032);
    border: 1px solid #2d4a6e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.kpi-label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { color: #e2e8f0; font-size: 1.4rem; font-weight: 700; margin: 0.2rem 0; }
.kpi-delta-up   { color: #22c55e; font-size: 0.82rem; }
.kpi-delta-down { color: #ef4444; font-size: 0.82rem; }
.kpi-delta-flat { color: #94a3b8; font-size: 0.82rem; }

/* Signal banners */
.signal-buy  { background: #052e16; border-left: 4px solid #22c55e; padding: 0.75rem 1rem; border-radius: 6px; color: #86efac; }
.signal-sell { background: #2d0606; border-left: 4px solid #ef4444; padding: 0.75rem 1rem; border-radius: 6px; color: #fca5a5; }
.signal-hold { background: #1c1a05; border-left: 4px solid #eab308; padding: 0.75rem 1rem; border-radius: 6px; color: #fde047; }

/* Market table rows */
.up-row   { color: #22c55e !important; }
.down-row { color: #ef4444 !important; }
</style>
""", unsafe_allow_html=True)


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    bundle   = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text())
    return bundle, metadata


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_horizon(h: str) -> int:
    return PRESET_HORIZONS[h] if h in PRESET_HORIZONS else int(h)


def fmt_price(v: float) -> str:
    return f"₹{v:,.2f}"


def fmt_pct(v: float) -> str:
    return "—" if pd.isna(v) else f"{v:+.2f}%"


def pct_color(v: float) -> str:
    if v > 0.5:  return "ticker-up"
    if v < -0.5: return "ticker-down"
    return "ticker-flat"


def pct_arrow(v: float) -> str:
    if v > 0.5:  return "▲"
    if v < -0.5: return "▼"
    return "─"


def buysell_signal(total_pct: float, spice_mape: float) -> dict:
    band = spice_mape * 0.5
    if abs(total_pct) < band:
        return {"signal": "HOLD", "icon": "🟡",
                "reason": f"Predicted move ({total_pct:+.1f}%) is within model uncertainty (MAPE ≈ {spice_mape:.1f}%). No confident directional signal."}
    if total_pct > 0:
        return {"signal": "BUY / HOLD STOCK", "icon": "🟢",
                "reason": f"Price predicted to rise {total_pct:+.1f}% over the horizon. Consider buying or holding inventory. (MAPE ≈ {spice_mape:.1f}%)"}
    return {"signal": "SELL / LIQUIDATE", "icon": "🔴",
            "reason": f"Price predicted to fall {total_pct:+.1f}% over the horizon. Consider selling or reducing stock. (MAPE ≈ {spice_mape:.1f}%)"}


def resolve_row(df: pd.DataFrame, spice: str, anchor_date_str: str):
    anchor_ts  = pd.Timestamp(anchor_date_str)
    spice_rows = df[df["Spice Name"] == spice].sort_values("date")
    if spice_rows.empty:
        return None, None, "No data for this spice."
    ds_start = spice_rows.iloc[0]["date"]
    ds_end   = spice_rows.iloc[-1]["date"]

    exact = spice_rows[spice_rows["date"] == anchor_ts]
    if not exact.empty:
        return exact.iloc[0], anchor_ts, None

    if anchor_ts < ds_start:
        return None, None, f"Date {anchor_date_str} is before dataset start ({ds_start.strftime('%Y-%m-%d')})."

    if anchor_ts > ds_end:
        row  = spice_rows.iloc[-1]
        used = row["date"]
        return row, used, f"⚠️ Beyond dataset end. Using latest available context: **{used.strftime('%Y-%m-%d')}**."

    earlier = spice_rows[spice_rows["date"] <= anchor_ts]
    row     = earlier.iloc[-1]
    used    = row["date"]
    return row, used, f"No data on {anchor_date_str}. Using nearest date: **{used.strftime('%Y-%m-%d')}**."


def run_forecast(bundle, metadata, row, spice_title: str, horizon_days: int, anchor_date_str: str) -> dict:
    le         = bundle["label_encoder"]
    model      = bundle["model"]
    spice_enc  = int(le.transform([spice_title])[0])
    per_spice  = metadata.get("per_spice_test_mape", {})
    spice_mape = per_spice.get(spice_title, float(metadata.get("test_mape", 0.0)))

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
    ship_cnt = safe("daily_shipment_count", 1)
    buy_cnt  = safe("daily_buyer_count", 1)
    exp_cnt  = safe("daily_exporter_count", 1)

    anchor_price = lag1
    anchor_ts    = pd.Timestamp(anchor_date_str)
    price_window = [lag1] * 7
    forecast_days = []

    for step in range(1, horizon_days + 1):
        fdate  = anchor_ts + pd.Timedelta(days=step)
        dow    = fdate.dayofweek
        mon    = fdate.month
        woy    = int(fdate.isocalendar()[1])
        mom1d  = (lag1 / price_window[-2] - 1.0) if price_window[-2] > 0 else 0.0
        mom7d  = (lag1 / lag7 - 1.0) if lag7 > 0 else 0.0
        vshock = dvol / r7avg_daily_vol if r7avg_daily_vol > 0 else 1.0

        feat = np.array([
            spice_enc, lag1, lag7, r7avg, r7std,
            mom1d, mom7d, r7vol, r14vol, vshock,
            dvol, ship_cnt, buy_cnt, exp_cnt,
            float(dow), float(mon), float(woy),
        ], dtype=np.float64).reshape(1, -1)

        pred_vwap = float(max(np.expm1(model.predict(feat)[0]), 0.0))
        pct       = (pred_vwap / anchor_price - 1.0) * 100.0
        forecast_days.append({"date": fdate.strftime("%Y-%m-%d"),
                               "predicted_vwap_inr": round(pred_vwap, 2),
                               "pct_change_vs_today": round(pct, 2)})

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
        "icon":         sig["icon"],
        "reason":       sig["reason"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def tab_market_overview(df: pd.DataFrame, metadata: dict):
    """MoneyControl-style market overview — ticker, gainers/losers, volume, heatmap."""
    ds_end   = df["date"].max()
    latest   = df[df["date"] == ds_end].copy()
    prev_day = df[df["date"] == df[df["date"] < ds_end]["date"].max()].copy() if len(df["date"].unique()) > 1 else latest.copy()

    # Merge with previous day for day-change
    mrg = latest.merge(
        prev_day[["Spice Name", "price_per_kg_inr_vwap"]].rename(
            columns={"price_per_kg_inr_vwap": "prev_vwap"}),
        on="Spice Name", how="left"
    )
    mrg["day_chg_pct"] = (mrg["price_per_kg_inr_vwap"] / mrg["prev_vwap"] - 1) * 100
    mrg["day_chg_pct"] = mrg["day_chg_pct"].fillna(mrg["price_momentum_1d"] * 100)

    # ── Ticker strip ──────────────────────────────────────────────────────────
    ticker_items = []
    for _, r in mrg.sort_values("daily_volume_kg", ascending=False).head(12).iterrows():
        sp    = r["Spice Name"]
        price = r["price_per_kg_inr_vwap"]
        chg   = r["day_chg_pct"]
        emoji = SPICE_EMOJI.get(sp, "🌿")
        if pd.isna(chg):
            chg_html = '<span class="ticker-flat">─</span>'
        else:
            cls = pct_color(chg)
            arr = pct_arrow(chg)
            chg_html = f'<span class="{cls}">{arr} {chg:+.1f}%</span>'
        ticker_items.append(
            f'<span class="ticker-item">'
            f'<span class="ticker-name">{emoji} {sp}</span>'
            f'<span class="ticker-price">₹{price:,.1f}</span>'
            f'{chg_html}'
            f'</span>'
        )
    items_html = "".join(ticker_items)
    st.markdown(
        f'<div class="ticker-wrap">'
        f'<span class="ticker-track">{items_html}{items_html}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Market summary KPIs ───────────────────────────────────────────────────
    total_vol   = mrg["daily_volume_kg"].sum()
    n_gainers   = (mrg["day_chg_pct"] > 0.5).sum()
    n_losers    = (mrg["day_chg_pct"] < -0.5).sum()
    avg_chg     = mrg["day_chg_pct"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📦 Total Export Volume", f"{total_vol/1e6:.2f}M kg",
              help="Sum of all spice volumes on latest date")
    c2.metric("🟢 Gainers", f"{n_gainers} spices")
    c3.metric("🔴 Losers", f"{n_losers} spices")
    c4.metric("📊 Avg Day Change", fmt_pct(avg_chg),
              delta_color="normal")
    c5.metric("📅 Data Date", ds_end.strftime("%Y-%m-%d"),
              delta_color="normal")

    st.divider()

    # ── Row 1: Top Gainers & Losers ───────────────────────────────────────────
    col_g, col_l = st.columns(2)

    with col_g:
        st.subheader("🚀 Top Gainers")
        gainers = mrg.nlargest(8, "day_chg_pct")[
            ["Spice Name", "price_per_kg_inr_vwap", "day_chg_pct", "daily_volume_kg"]
        ].copy()
        gainers.columns = ["Spice", "Price (₹/kg)", "Day Chg %", "Volume (kg)"]
        gainers["Day Chg %"]    = gainers["Day Chg %"].apply(fmt_pct)
        gainers["Price (₹/kg)"] = gainers["Price (₹/kg)"].apply(lambda x: f"₹{x:,.2f}")
        gainers["Volume (kg)"]  = gainers["Volume (kg)"].apply(lambda x: f"{x/1000:.1f}K")
        st.dataframe(gainers, use_container_width=True, hide_index=True)

    with col_l:
        st.subheader("📉 Top Losers")
        losers = mrg.nsmallest(8, "day_chg_pct")[
            ["Spice Name", "price_per_kg_inr_vwap", "day_chg_pct", "daily_volume_kg"]
        ].copy()
        losers.columns = ["Spice", "Price (₹/kg)", "Day Chg %", "Volume (kg)"]
        losers["Day Chg %"]    = losers["Day Chg %"].apply(fmt_pct)
        losers["Price (₹/kg)"] = losers["Price (₹/kg)"].apply(lambda x: f"₹{x:,.2f}")
        losers["Volume (kg)"]  = losers["Volume (kg)"].apply(lambda x: f"{x/1000:.1f}K")
        st.dataframe(losers, use_container_width=True, hide_index=True)

    st.divider()

    # ── Row 2: Most Active & Volume Shockers ─────────────────────────────────
    col_a, col_s = st.columns(2)

    with col_a:
        st.subheader("🔥 Most Active")
        st.caption("Highest export volume today — market depth leaders")
        active = mrg.nlargest(8, "daily_volume_kg")[
            ["Spice Name", "daily_volume_kg", "daily_shipment_count",
             "daily_buyer_count", "price_per_kg_inr_vwap", "day_chg_pct"]
        ].copy()
        active.columns = ["Spice", "Volume (kg)", "Shipments", "Countries", "VWAP ₹/kg", "Day %"]
        active["Volume (kg)"]  = active["Volume (kg)"].apply(lambda x: f"{x/1000:.1f}K")
        active["VWAP ₹/kg"]   = active["VWAP ₹/kg"].apply(lambda x: f"₹{x:,.2f}")
        active["Day %"]        = active["Day %"].apply(fmt_pct)
        active["Shipments"]    = active["Shipments"].apply(lambda x: int(x) if pd.notna(x) else 0)
        active["Countries"]    = active["Countries"].apply(lambda x: int(x) if pd.notna(x) else 0)
        st.dataframe(active, use_container_width=True, hide_index=True)

    with col_s:
        st.subheader("⚡ Volume Shockers")
        st.caption("Daily volume vs 7-day average — unusual demand spikes")
        shock_df = mrg[mrg["volume_shock"].notna()].nlargest(8, "volume_shock")[
            ["Spice Name", "volume_shock", "daily_volume_kg",
             "price_per_kg_inr_vwap", "day_chg_pct"]
        ].copy()
        shock_df.columns = ["Spice", "Shock Ratio", "Volume (kg)", "VWAP ₹/kg", "Day %"]
        shock_df["Shock Ratio"] = shock_df["Shock Ratio"].apply(lambda x: f"{x:.2f}×")
        shock_df["Volume (kg)"] = shock_df["Volume (kg)"].apply(lambda x: f"{x/1000:.1f}K")
        shock_df["VWAP ₹/kg"]  = shock_df["VWAP ₹/kg"].apply(lambda x: f"₹{x:,.2f}")
        shock_df["Day %"]       = shock_df["Day %"].apply(fmt_pct)
        st.dataframe(shock_df, use_container_width=True, hide_index=True)
        st.caption("Shock Ratio > 2× = significant demand surge vs recent 7-day average")

    st.divider()

    # ── Volume Leaders chart ──────────────────────────────────────────────────
    st.subheader("📦 Volume Leaders (Today)")
    vol_df = mrg.nlargest(10, "daily_volume_kg")[["Spice Name", "daily_volume_kg", "price_per_kg_inr_vwap", "day_chg_pct"]].copy()
    vol_df.columns = ["Spice", "Volume (kg)", "VWAP (₹/kg)", "Day Chg %"]
    st.bar_chart(vol_df.set_index("Spice")["Volume (kg)"], use_container_width=True)

    st.divider()

    # ── Full Market Snapshot table ────────────────────────────────────────────
    st.subheader("📋 Full Market Snapshot")
    snap = mrg[[
        "Spice Name", "price_per_kg_inr_vwap", "day_chg_pct",
        "price_momentum_7d", "rolling7_avg_price",
        "rolling7_price_std", "daily_volume_kg", "daily_shipment_count",
        "daily_buyer_count"
    ]].copy().sort_values("daily_volume_kg", ascending=False)

    snap.columns = [
        "Spice", "VWAP ₹/kg", "Day %", "7d Mom %",
        "7d Avg Price", "7d Volatility", "Volume (kg)", "Shipments", "Countries"
    ]
    snap["VWAP ₹/kg"]   = snap["VWAP ₹/kg"].apply(lambda x: f"₹{x:,.2f}")
    snap["Day %"]        = snap["Day %"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    snap["7d Mom %"]     = snap["7d Mom %"].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
    snap["7d Avg Price"] = snap["7d Avg Price"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
    snap["7d Volatility"]= snap["7d Volatility"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
    snap["Volume (kg)"]  = snap["Volume (kg)"].apply(lambda x: f"{x/1000:.1f}K" if pd.notna(x) else "—")
    snap["Shipments"]    = snap["Shipments"].apply(lambda x: int(x) if pd.notna(x) else 0)
    snap["Countries"]    = snap["Countries"].apply(lambda x: int(x) if pd.notna(x) else 0)

    st.dataframe(snap, use_container_width=True, hide_index=True)
    st.caption(f"Data as of {ds_end.strftime('%d %b %Y')} · May–June 2025 Indian spice export data")


def tab_spice_detail(df: pd.DataFrame, metadata: dict):
    """Deep-dive into one spice — price history, OHLC, volume, momentum."""
    spices = sorted(metadata["spices"])

    col_sel, col_period = st.columns([3, 2])
    with col_sel:
        spice = st.selectbox("Select spice", spices, index=spices.index("Cumin"), key="detail_spice")
    with col_period:
        period = st.radio("Period", ["All", "Last 30d", "Last 14d", "Last 7d"],
                          horizontal=True, key="detail_period")

    spice_df = df[df["Spice Name"] == spice].sort_values("date").copy()
    if period == "Last 7d":
        spice_df = spice_df.tail(7)
    elif period == "Last 14d":
        spice_df = spice_df.tail(14)
    elif period == "Last 30d":
        spice_df = spice_df.tail(30)

    if spice_df.empty:
        st.warning(f"No data for {spice}.")
        return

    latest = spice_df.iloc[-1]
    first  = spice_df.iloc[0]
    emoji  = SPICE_EMOJI.get(spice, "🌿")
    per_spice_mape = metadata.get("per_spice_test_mape", {})
    mape   = per_spice_mape.get(spice, metadata.get("test_mape", 0.0))
    reliability = "🟢 High" if mape < 25 else ("🟡 Medium" if mape < 50 else "🔴 Low")

    # ── Header KPIs ───────────────────────────────────────────────────────────
    st.markdown(f"### {emoji} {spice}")
    period_chg = (latest["price_per_kg_inr_vwap"] / first["price_per_kg_inr_vwap"] - 1) * 100 if first["price_per_kg_inr_vwap"] > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current VWAP", fmt_price(latest["price_per_kg_inr_vwap"]))
    c2.metric("Period Change", fmt_pct(period_chg), delta=fmt_pct(period_chg))
    c3.metric("7d Avg Price", fmt_price(latest["rolling7_avg_price"]) if pd.notna(latest.get("rolling7_avg_price")) else "—")
    c4.metric("7d Volatility (σ)", fmt_price(latest["rolling7_price_std"]) if pd.notna(latest.get("rolling7_price_std")) else "—")
    c5.metric(f"Forecast Reliability", reliability, delta=f"MAPE {mape:.1f}%", delta_color="off")

    st.divider()

    # ── Price history chart ───────────────────────────────────────────────────
    st.subheader("📈 VWAP Price History")
    price_chart = spice_df.set_index("date")[["price_per_kg_inr_vwap", "rolling7_avg_price"]].copy()
    price_chart.columns = ["VWAP (₹/kg)", "7d Rolling Avg"]
    st.line_chart(price_chart, use_container_width=True)

    # ── Price range (High/Low band) ───────────────────────────────────────────
    if "price_per_kg_inr_min" in spice_df.columns:
        st.subheader("📊 Daily Price Range (High / Low)")
        range_chart = spice_df.set_index("date")[["price_per_kg_inr_max", "price_per_kg_inr_vwap", "price_per_kg_inr_min"]].copy()
        range_chart.columns = ["High", "VWAP", "Low"]
        st.line_chart(range_chart, use_container_width=True)

    col_v, col_m = st.columns(2)

    # ── Volume chart ──────────────────────────────────────────────────────────
    with col_v:
        st.subheader("📦 Daily Export Volume (kg)")
        vol_chart = spice_df.set_index("date")[["daily_volume_kg"]].dropna()
        vol_chart.columns = ["Volume (kg)"]
        st.bar_chart(vol_chart, use_container_width=True)

    # ── Momentum chart ────────────────────────────────────────────────────────
    with col_m:
        st.subheader("⚡ Price Momentum")
        mom_chart = spice_df.set_index("date")[["price_momentum_1d", "price_momentum_7d"]].dropna() * 100
        mom_chart.columns = ["1-day Mom %", "7-day Mom %"]
        st.line_chart(mom_chart, use_container_width=True)

    # ── Volume shock ──────────────────────────────────────────────────────────
    st.subheader("🔥 Volume Shock (daily vs 7d avg)")
    shock = spice_df.set_index("date")[["volume_shock"]].dropna()
    shock.columns = ["Volume Shock Ratio"]
    st.line_chart(shock, use_container_width=True)
    st.caption("Ratio > 1 = above-average shipment day. Ratio > 2 = significant demand spike.")

    # ── Data table ────────────────────────────────────────────────────────────
    with st.expander("📋 Raw daily data"):
        show_cols = ["date", "price_per_kg_inr_vwap", "price_per_kg_inr_min",
                     "price_per_kg_inr_max", "daily_volume_kg", "daily_shipment_count",
                     "daily_buyer_count", "rolling7_avg_price", "volume_shock"]
        tbl = spice_df[[c for c in show_cols if c in spice_df.columns]].copy()
        tbl["date"] = tbl["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(tbl, use_container_width=True, hide_index=True)


def tab_forecast(df: pd.DataFrame, bundle, metadata: dict):
    """Price forecast with buy/sell/hold signal."""
    spices = sorted(metadata["spices"])
    ds_end = df["date"].max().strftime("%Y-%m-%d")
    per_spice_mape = metadata.get("per_spice_test_mape", {})

    st.subheader("🔮 Price Forecast & Buy/Sell/Hold Signal")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        spice = st.selectbox("Spice", spices, index=spices.index("Cumin"), key="fc_spice")
    with c2:
        anchor_date = st.date_input(
            "Anchor date",
            value=pd.Timestamp(ds_end).date(),
            help="Dataset: May–June 2025. Future dates auto-fallback to latest.",
            key="fc_date"
        ).strftime("%Y-%m-%d")
    with c3:
        horizon_label = st.selectbox(
            "Horizon",
            ["7d — 1 week", "1m — 30 days", "3m — 90 days", "1y — 365 days"],
            key="fc_horizon"
        )
        horizon_key = horizon_label.split(" ")[0]

    run = st.button("🔮 Run Forecast", type="primary", use_container_width=True)
    if not run:
        mape = per_spice_mape.get(spice, metadata.get("test_mape", 0.0))
        rel  = "🟢 High (reliable signal)" if mape < 25 else ("🟡 Medium (directional only)" if mape < 50 else "🔴 Low (treat as approximate)")
        st.info(f"**{spice}** — Model reliability: {rel} · MAPE = {mape:.1f}%")

        st.markdown("##### 📋 All-spice reliability guide")
        mape_df = (
            pd.DataFrame(per_spice_mape.items(), columns=["Spice", "MAPE (%)"])
            .sort_values("MAPE (%)")
            .reset_index(drop=True)
        )
        mape_df["Reliability"] = mape_df["MAPE (%)"].apply(
            lambda x: "🟢 High" if x < 25 else ("🟡 Medium" if x < 50 else "🔴 Low")
        )
        st.dataframe(mape_df, use_container_width=True, hide_index=True)
        return

    row, used_date, note = resolve_row(df, spice, anchor_date)
    if row is None:
        st.error(note)
        return
    if note:
        st.warning(note)

    horizon_days  = resolve_horizon(horizon_key)
    used_date_str = used_date.strftime("%Y-%m-%d") if hasattr(used_date, "strftime") else str(used_date)

    with st.spinner(f"Forecasting {horizon_days} days for {spice}…"):
        result = run_forecast(bundle, metadata, row, spice, horizon_days, used_date_str)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    emoji = SPICE_EMOJI.get(spice, "🌿")
    st.markdown(f"#### {emoji} {spice} — {horizon_days}-day Outlook")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Anchor Price", fmt_price(result["anchor_price"]))
    k2.metric(f"Forecast End ({horizon_days}d)",
              fmt_price(result["forecast"][-1]["predicted_vwap_inr"]),
              delta=fmt_pct(result["final_pct"]))
    k3.metric("Avg Forecast", fmt_price(result["avg_price"]))
    k4.metric("Range", f"₹{result['min_price']:,.0f} – ₹{result['max_price']:,.0f}")
    k5.metric("Model MAPE", f"{result['spice_mape']:.1f}%",
              delta="reliable" if result["spice_mape"] < 25 else "indicative",
              delta_color="off")

    # ── Signal banner ─────────────────────────────────────────────────────────
    sig  = result["signal"]
    icon = result["icon"]
    cls  = "signal-buy" if "BUY" in sig else ("signal-sell" if "SELL" in sig else "signal-hold")
    trend_arrow = "📈" if result["trend"] == "upward" else ("📉" if result["trend"] == "downward" else "➡️")
    st.markdown(
        f'<div class="{cls}"><strong>{icon} {sig}</strong> {trend_arrow} &nbsp;|&nbsp; {result["reason"]}</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # ── Forecast chart ────────────────────────────────────────────────────────
    fdf = pd.DataFrame(result["forecast"])
    fdf["date"] = pd.to_datetime(fdf["date"])

    # Historical context + forecast combined
    hist = df[df["Spice Name"] == spice].sort_values("date").tail(14)[["date", "price_per_kg_inr_vwap"]].copy()
    hist.columns = ["date", "Historical VWAP"]
    hist = hist.set_index("date")

    fc_chart = fdf.set_index("date")[["predicted_vwap_inr"]].copy()
    fc_chart.columns = ["Forecast VWAP"]

    combined = hist.join(fc_chart, how="outer")
    st.line_chart(combined, use_container_width=True)

    # ── % change chart ────────────────────────────────────────────────────────
    pct_chart = fdf.set_index("date")[["pct_change_vs_today"]].copy()
    pct_chart.columns = ["% Change vs Anchor"]
    st.area_chart(pct_chart, use_container_width=True)

    # ── Day-by-day table ──────────────────────────────────────────────────────
    with st.expander("📋 Day-by-day forecast table"):
        tbl = fdf.rename(columns={
            "date": "Date",
            "predicted_vwap_inr": "Forecast VWAP (₹/kg)",
            "pct_change_vs_today": "% vs Anchor"
        }).copy()
        tbl["Date"]                 = tbl["Date"].dt.strftime("%Y-%m-%d")
        tbl["Forecast VWAP (₹/kg)"] = tbl["Forecast VWAP (₹/kg)"].apply(lambda x: f"₹{x:,.2f}")
        tbl["% vs Anchor"]          = tbl["% vs Anchor"].apply(fmt_pct)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.caption(
        f"Auto-regressive LightGBM forecast · Dataset: May–June 2025 · "
        f"Spice MAPE: {result['spice_mape']:.1f}% · For indicative use only."
    )


def tab_compare(df: pd.DataFrame, metadata: dict):
    """Overlay price & volume trends for multiple spices."""
    spices = sorted(metadata["spices"])

    selected = st.multiselect(
        "Select spices to compare (2–6 recommended)",
        spices,
        default=["Cumin", "Turmeric", "Cardamom"],
        key="cmp_spices"
    )
    metric = st.radio(
        "Metric",
        ["VWAP Price (₹/kg)", "Daily Volume (kg)", "7d Rolling Avg Price", "Price Momentum 7d (%)"],
        horizontal=True,
        key="cmp_metric"
    )

    if not selected:
        st.info("Select at least one spice to compare.")
        return

    col_map = {
        "VWAP Price (₹/kg)":         "price_per_kg_inr_vwap",
        "Daily Volume (kg)":          "daily_volume_kg",
        "7d Rolling Avg Price":       "rolling7_avg_price",
        "Price Momentum 7d (%)":      "price_momentum_7d",
    }
    col = col_map[metric]

    pivot = (
        df[df["Spice Name"].isin(selected)]
        .pivot_table(index="date", columns="Spice Name", values=col)
        .sort_index()
    )

    if metric == "Price Momentum 7d (%)":
        pivot = pivot * 100

    st.subheader(f"📊 {metric} — Comparative View")
    st.line_chart(pivot, use_container_width=True)

    # ── Indexed to 100 (normalized) ───────────────────────────────────────────
    if metric in ("VWAP Price (₹/kg)", "7d Rolling Avg Price"):
        st.subheader("📐 Indexed Performance (Base = 100 on first day)")
        base  = pivot.iloc[0]
        normd = (pivot / base * 100).dropna(how="all")
        st.line_chart(normd, use_container_width=True)
        st.caption("All spices start at 100 — shows relative % gain/loss from start of period.")

    # ── Latest snapshot comparison ────────────────────────────────────────────
    st.subheader("📋 Latest Snapshot")
    ds_end  = df["date"].max()
    snap    = df[(df["Spice Name"].isin(selected)) & (df["date"] == ds_end)][
        ["Spice Name", "price_per_kg_inr_vwap", "price_momentum_1d",
         "price_momentum_7d", "daily_volume_kg", "rolling7_price_std"]
    ].copy()
    snap.columns = ["Spice", "VWAP ₹/kg", "1d Mom", "7d Mom", "Volume (kg)", "Volatility (σ)"]
    snap["VWAP ₹/kg"]    = snap["VWAP ₹/kg"].apply(lambda x: f"₹{x:,.2f}")
    snap["1d Mom"]        = snap["1d Mom"].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
    snap["7d Mom"]        = snap["7d Mom"].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
    snap["Volume (kg)"]   = snap["Volume (kg)"].apply(lambda x: f"{x/1000:.1f}K" if pd.notna(x) else "—")
    snap["Volatility (σ)"]= snap["Volatility (σ)"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
    st.dataframe(snap, use_container_width=True, hide_index=True)


def tab_insights(df: pd.DataFrame, metadata: dict):
    """Data insights — price kings, volume leaders, volatility, trend analysis."""

    st.subheader("💡 Market Insights & Data Analysis")
    st.caption("Full-period analysis across all spices · May–June 2025")

    # ── Compute per-spice summary stats ───────────────────────────────────────
    stats = (
        df.groupby("Spice Name")
        .agg(
            avg_price    = ("price_per_kg_inr_vwap",  "mean"),
            max_price    = ("price_per_kg_inr_vwap",  "max"),
            min_price    = ("price_per_kg_inr_vwap",  "min"),
            avg_volatility = ("rolling7_price_std",   "mean"),
            total_volume = ("daily_volume_kg",         "sum"),
            avg_volume   = ("daily_volume_kg",         "mean"),
            total_shipments = ("daily_shipment_count", "sum"),
            active_days  = ("date",                    "count"),
            avg_buyers   = ("daily_buyer_count",       "mean"),
            avg_shock    = ("volume_shock",            "mean"),
        )
        .reset_index()
    )
    stats["price_range"]     = stats["max_price"] - stats["min_price"]
    stats["price_range_pct"] = (stats["price_range"] / stats["min_price"] * 100).round(1)
    stats["cv_pct"]          = (stats["avg_volatility"] / stats["avg_price"] * 100).round(1)  # coeff of variation

    # Latest price & period return
    first_price = df.groupby("Spice Name").apply(
        lambda g: g.sort_values("date").dropna(subset=["price_per_kg_inr_vwap"]).iloc[0]["price_per_kg_inr_vwap"]
        if not g.dropna(subset=["price_per_kg_inr_vwap"]).empty else np.nan
    ).rename("first_price").reset_index()
    last_price = df.groupby("Spice Name").apply(
        lambda g: g.sort_values("date").dropna(subset=["price_per_kg_inr_vwap"]).iloc[-1]["price_per_kg_inr_vwap"]
        if not g.dropna(subset=["price_per_kg_inr_vwap"]).empty else np.nan
    ).rename("last_price").reset_index()
    stats = stats.merge(first_price, on="Spice Name").merge(last_price, on="Spice Name")
    stats["period_return_pct"] = ((stats["last_price"] / stats["first_price"] - 1) * 100).round(2)

    per_spice_mape = metadata.get("per_spice_test_mape", {})
    stats["forecast_mape"] = stats["Spice Name"].map(per_spice_mape)

    # ── Section 1: Price Analysis ─────────────────────────────────────────────
    st.markdown("### 💰 Price Analysis")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**👑 Most Expensive (Avg VWAP)**")
        top_price = stats.nlargest(8, "avg_price")[["Spice Name", "avg_price", "max_price", "min_price"]].copy()
        top_price.columns = ["Spice", "Avg ₹/kg", "High ₹/kg", "Low ₹/kg"]
        for col in ["Avg ₹/kg", "High ₹/kg", "Low ₹/kg"]:
            top_price[col] = top_price[col].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(top_price, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**📈 Best Period Return (May→Jun)**")
        top_return = stats.nlargest(8, "period_return_pct")[["Spice Name", "period_return_pct", "first_price", "last_price"]].copy()
        top_return.columns = ["Spice", "Return %", "Start ₹/kg", "End ₹/kg"]
        top_return["Return %"]   = top_return["Return %"].apply(fmt_pct)
        top_return["Start ₹/kg"] = top_return["Start ₹/kg"].apply(lambda x: f"₹{x:,.2f}")
        top_return["End ₹/kg"]   = top_return["End ₹/kg"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(top_return, use_container_width=True, hide_index=True)

    with c3:
        st.markdown("**📉 Worst Period Return (May→Jun)**")
        bot_return = stats.nsmallest(8, "period_return_pct")[["Spice Name", "period_return_pct", "first_price", "last_price"]].copy()
        bot_return.columns = ["Spice", "Return %", "Start ₹/kg", "End ₹/kg"]
        bot_return["Return %"]   = bot_return["Return %"].apply(fmt_pct)
        bot_return["Start ₹/kg"] = bot_return["Start ₹/kg"].apply(lambda x: f"₹{x:,.2f}")
        bot_return["End ₹/kg"]   = bot_return["End ₹/kg"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(bot_return, use_container_width=True, hide_index=True)

    # Price range bar chart
    st.markdown("**📊 Price Range per Spice (High − Low, entire period)**")
    range_chart = stats.sort_values("price_range", ascending=False)[["Spice Name", "price_range"]].set_index("Spice Name")
    range_chart.columns = ["Price Range ₹/kg"]
    st.bar_chart(range_chart, use_container_width=True)

    st.divider()

    # ── Section 2: Volume Analysis ────────────────────────────────────────────
    st.markdown("### 📦 Volume Analysis")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**🏆 Highest Total Export Volume**")
        top_vol = stats.nlargest(10, "total_volume")[["Spice Name", "total_volume", "avg_volume", "total_shipments", "active_days"]].copy()
        top_vol.columns = ["Spice", "Total (kg)", "Avg/Day (kg)", "Shipments", "Active Days"]
        top_vol["Total (kg)"]    = top_vol["Total (kg)"].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1000:.1f}K")
        top_vol["Avg/Day (kg)"]  = top_vol["Avg/Day (kg)"].apply(lambda x: f"{x/1000:.1f}K")
        top_vol["Shipments"]     = top_vol["Shipments"].astype(int)
        top_vol["Active Days"]   = top_vol["Active Days"].astype(int)
        st.dataframe(top_vol, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**🌍 Most International Reach (Avg Buyer Countries/Day)**")
        top_buyers = stats.nlargest(10, "avg_buyers")[["Spice Name", "avg_buyers", "total_volume", "total_shipments"]].copy()
        top_buyers.columns = ["Spice", "Avg Countries/Day", "Total Volume (kg)", "Shipments"]
        top_buyers["Avg Countries/Day"] = top_buyers["Avg Countries/Day"].apply(lambda x: f"{x:.1f}")
        top_buyers["Total Volume (kg)"] = top_buyers["Total Volume (kg)"].apply(lambda x: f"{x/1000:.1f}K")
        top_buyers["Shipments"]         = top_buyers["Shipments"].astype(int)
        st.dataframe(top_buyers, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 3: Volatility Analysis ───────────────────────────────────────
    st.markdown("### ⚡ Volatility & Risk Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**🌊 Most Volatile (7d Price Std Dev)**")
        top_vol2 = stats.nlargest(10, "avg_volatility")[["Spice Name", "avg_volatility", "cv_pct", "price_range_pct", "avg_price"]].copy()
        top_vol2.columns = ["Spice", "Avg σ (₹/kg)", "CV %", "Range %", "Avg Price"]
        top_vol2["Avg σ (₹/kg)"] = top_vol2["Avg σ (₹/kg)"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
        top_vol2["CV %"]         = top_vol2["CV %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        top_vol2["Range %"]      = top_vol2["Range %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        top_vol2["Avg Price"]    = top_vol2["Avg Price"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(top_vol2, use_container_width=True, hide_index=True)
        st.caption("CV % = Coefficient of Variation (σ / avg price) — normalised volatility across spices")

    with c2:
        st.markdown("**🏔️ Most Stable (Lowest CV %)**")
        stable = stats[stats["cv_pct"].notna()].nsmallest(10, "cv_pct")[["Spice Name", "cv_pct", "avg_volatility", "avg_price"]].copy()
        stable.columns = ["Spice", "CV %", "Avg σ (₹/kg)", "Avg Price"]
        stable["CV %"]        = stable["CV %"].apply(lambda x: f"{x:.1f}%")
        stable["Avg σ (₹/kg)"]= stable["Avg σ (₹/kg)"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
        stable["Avg Price"]   = stable["Avg Price"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(stable, use_container_width=True, hide_index=True)
        st.caption("Low CV % = price predictable and consistent — safer for long-term contracts")

    # Volatility bar chart
    st.markdown("**📊 Coefficient of Variation % — all spices ranked**")
    cv_chart = stats[stats["cv_pct"].notna()].sort_values("cv_pct", ascending=False)[["Spice Name", "cv_pct"]].set_index("Spice Name")
    cv_chart.columns = ["Volatility CV %"]
    st.bar_chart(cv_chart, use_container_width=True)

    st.divider()

    # ── Section 4: Volume Shock Ranking ──────────────────────────────────────
    st.markdown("### 🔥 Demand Shock Analysis")
    st.caption("Avg volume shock ratio: 1.0 = normal, >1.5 = frequent demand spikes")

    shock_rank = stats[stats["avg_shock"].notna()].sort_values("avg_shock", ascending=False)[
        ["Spice Name", "avg_shock", "total_volume", "avg_volume"]
    ].copy()
    shock_rank.columns = ["Spice", "Avg Shock Ratio", "Total Volume (kg)", "Avg Daily Vol (kg)"]
    shock_chart = shock_rank.set_index("Spice")[["Avg Shock Ratio"]]
    st.bar_chart(shock_chart, use_container_width=True)

    shock_rank["Avg Shock Ratio"]    = shock_rank["Avg Shock Ratio"].apply(lambda x: f"{x:.2f}×")
    shock_rank["Total Volume (kg)"]  = shock_rank["Total Volume (kg)"].apply(lambda x: f"{x/1000:.1f}K")
    shock_rank["Avg Daily Vol (kg)"] = shock_rank["Avg Daily Vol (kg)"].apply(lambda x: f"{x/1000:.1f}K")

    with st.expander("📋 Full shock ranking table"):
        st.dataframe(shock_rank, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 5: Full Summary Table ────────────────────────────────────────
    st.markdown("### 📋 Complete Spice Intelligence Summary")
    summary = stats[[
        "Spice Name", "avg_price", "period_return_pct", "avg_volatility",
        "cv_pct", "total_volume", "total_shipments", "avg_buyers", "forecast_mape"
    ]].copy().sort_values("total_volume", ascending=False)

    summary.columns = [
        "Spice", "Avg Price ₹/kg", "Period Return %", "Avg Volatility σ",
        "CV %", "Total Volume (kg)", "Shipments", "Avg Countries", "Forecast MAPE %"
    ]
    summary["Avg Price ₹/kg"]   = summary["Avg Price ₹/kg"].apply(lambda x: f"₹{x:,.2f}")
    summary["Period Return %"]  = summary["Period Return %"].apply(fmt_pct)
    summary["Avg Volatility σ"] = summary["Avg Volatility σ"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
    summary["CV %"]             = summary["CV %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    summary["Total Volume (kg)"]= summary["Total Volume (kg)"].apply(lambda x: f"{x/1000:.1f}K")
    summary["Shipments"]        = summary["Shipments"].astype(int)
    summary["Avg Countries"]    = summary["Avg Countries"].apply(lambda x: f"{x:.1f}")
    summary["Forecast MAPE %"]  = summary["Forecast MAPE %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")

    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption("Sorted by total export volume · May–June 2025 Indian spice export data")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    bundle, metadata = load_model()
    df               = load_data()
    ds_end           = df["date"].max().strftime("%d %b %Y")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="market-header">
        <h1>🌶️ Turmerix — Indian Spice Market</h1>
        <p>Live price intelligence · Export market analysis · AI-powered forecast &nbsp;·&nbsp; Data as of {ds_end}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation tabs ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Market Overview",
        "🔍 Spice Detail",
        "🔮 Forecast & Signal",
        "📊 Compare Spices",
        "💡 Insights",
    ])

    with tab1:
        tab_market_overview(df, metadata)

    with tab2:
        tab_spice_detail(df, metadata)

    with tab3:
        tab_forecast(df, bundle, metadata)

    with tab4:
        tab_compare(df, metadata)

    with tab5:
        tab_insights(df, metadata)


if __name__ == "__main__":
    main()
