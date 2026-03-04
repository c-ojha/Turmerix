# 📖 Turmerix Dashboard — Glossary of Terms

All metrics, indicators, and columns used across the Turmerix Spice Market Dashboard.

---

## 🏠 Market Overview

| Term | Definition |
|---|---|
| **VWAP (Volume-Weighted Average Price)** | Price per kg weighted by shipment quantity. Gives a true market price that accounts for size of each trade. Formula: `Σ(price × qty) / Σ(qty)`. |
| **Day Change %** | Percentage change in VWAP from the previous trading day. Positive = price rose, Negative = price fell. |
| **Gainers** | Spices whose Day Change % is greater than +0.5% on the latest date. |
| **Losers** | Spices whose Day Change % is less than −0.5% on the latest date. |
| **Total Export Volume** | Sum of all shipment quantities (in kg) across all spices for the latest available date. |
| **Shock Ratio** | Daily volume ÷ 7-day average volume. A ratio of 2× means today's shipment volume is double the recent norm — signals an unusual demand event. |
| **Most Active** | Spices ranked by highest export volume on the latest date — indicates market depth and trading activity. |
| **Volume Shockers** | Spices with the highest Shock Ratio — unusual demand spikes vs their own recent 7-day baseline. |

---

## 🔍 Spice Detail

| Term | Definition |
|---|---|
| **Price History (VWAP)** | Daily VWAP plotted over time for a selected spice. |
| **7-Day Rolling Avg Price** | Average of VWAP over the past 7 trading days. Smooths out short-term noise to show the underlying trend. |
| **Daily Price Range** | The spread between the highest and lowest price recorded across all shipments for a spice on a given day. |
| **Price High / Low** | Max and min `Unit Rate in INR` recorded among all shipments of the spice on that day. |
| **Momentum 1d** | Price change ratio from the previous day: `(today's VWAP / yesterday's VWAP) − 1`. Expressed as a percentage. |
| **Momentum 7d** | Price change ratio compared to 7 days ago: `(today's VWAP / VWAP[−7d]) − 1`. Shows medium-term trend direction. |
| **Volume (kg)** | Total weight of all shipments for the spice on a given day, in kilograms. |
| **Volume Shock** | Same as Shock Ratio above. Daily volume ÷ rolling 7-day average volume. Values consistently above 1.5× indicate a spice with erratic demand. |
| **Rolling 7-day Avg Volume** | Average daily shipment volume over the past 7 days — used as the baseline denominator for Volume Shock. |

---

## 🔮 Forecast & Signal

| Term | Definition |
|---|---|
| **Forecast Horizon** | Number of days ahead the model predicts prices: 7 days, 1 month (30d), 3 months (90d), or 1 year (365d). |
| **Predicted VWAP** | The model's estimated price per kg (VWAP) for each future day, output by the LightGBM time-series model. |
| **Signal** | A trading recommendation derived from the forecast trend: **BUY** (price rising), **SELL** (price falling), **HOLD** (flat/uncertain). |
| **Signal Threshold** | Signals trigger when cumulative forecast change exceeds ±3% over the horizon. Below this, signal is HOLD. |
| **MAPE (Mean Absolute Percentage Error)** | Average % error of model predictions vs actual prices on test data. Lower is better. E.g. 15% MAPE = predictions are off by 15% on average. |
| **SMAPE (Symmetric MAPE)** | A variant of MAPE that treats over- and under-predictions equally. Less sensitive to very low actual values. |
| **Lag 1 Price** | Yesterday's VWAP — the most important feature in the model. |
| **Lag 7 Price** | VWAP from 7 days ago — captures weekly seasonality patterns. |
| **Iterative Forecast** | For multi-day horizons, each predicted day's price is fed back as the next day's lag features, building a rolling forecast chain. |

---

## 📊 Compare Spices

| Term | Definition |
|---|---|
| **Indexed to 100** | All spices normalised to start at a value of 100 on the first day of the period. Shows relative % gain or loss from the same starting point — like a stock price index. Useful to compare spices that have very different absolute price levels (e.g. Saffron vs Cumin). |
| **Overlay Chart** | Multiple spices plotted on the same axis for direct visual comparison of price, volume, or momentum. |

---

## 💡 Insights Tab

| Term | Definition |
|---|---|
| **Avg VWAP (Period)** | Average of all daily VWAPs for a spice across the full data period (May–June 2025). |
| **Price High / Low** | Maximum and minimum VWAP recorded for a spice across the entire dataset period. |
| **Price Range** | `High − Low` in ₹/kg — absolute price swing over the full period. |
| **Price Range %** | `(High − Low) / Low × 100` — price swing as a percentage of the lowest price. |
| **Period Return %** | `(Last VWAP − First VWAP) / First VWAP × 100` — the overall price change from start to end of the dataset. Analogous to stock return over a holding period. |
| **Avg Volatility σ (Sigma)** | Average of the 7-day rolling standard deviation of VWAP. Measures how much the price fluctuates around its recent average. Higher σ = more volatile. |
| **CV % (Coefficient of Variation)** | `(Avg σ / Avg VWAP) × 100`. Normalises volatility relative to price level, enabling fair comparison across spices with very different prices (e.g. Saffron vs Cumin). Low CV% = stable, High CV% = risky. |
| **Total Export Volume** | Sum of all daily_volume_kg for the spice across the full period. |
| **Avg Daily Volume** | `Total Volume / Active Days` — average kg exported per active day. |
| **Active Days** | Number of distinct dates on which shipments were recorded for the spice. |
| **Shipments** | Total number of individual export transactions recorded. |
| **Avg Countries/Day** | Average number of distinct destination countries receiving shipments per active day — a measure of market reach and diversification. |
| **Avg Shock Ratio** | Average of daily Volume Shock values over the full period. Values near 1.0 = consistent volume. Values above 1.5 = spice frequently experiences demand spikes. |
| **Forecast MAPE %** | The LightGBM model's per-spice test MAPE — how accurately the AI can forecast that spice's price. Lower = more predictable. |

---

## 📐 Data & Model Reference

| Term | Definition |
|---|---|
| **Source Data** | Indian spice export customs data, May–June 2025 (~63,000 shipment records). |
| **Unit Rate in INR** | Customs-reported price per original unit in Indian Rupees, converted to ₹/kg using the shipment's unit code (KGS, MTS, LBS, etc.). |
| **FOB (Free on Board)** | Total invoice value of a shipment at the point of export. Not used in price prediction (leaky feature). |
| **daily_spice_timeseries.csv** | Daily aggregated dataset: one row per spice per trading day, 24 columns of price and volume features. |
| **LightGBM** | A gradient-boosting machine learning model used for price forecasting. Trained on daily time-series data, cross-spice (all spices in one model). |
| **Train/Test Split** | Time-based split: training data ≤ 2025-06-14, test data ≥ 2025-06-15. Prevents data leakage from future into past. |
| **Label Encoding** | Spice names are converted to numeric codes so the model can process them as a feature. |
