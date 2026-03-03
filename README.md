# Turmerix — Indian Spice Price Prediction

ML pipeline + REST API for two complementary prediction tasks on Indian spice export trade data (May–June 2025):

1. **Transaction-level price prediction** — predicts `Unit Rate in INR` for a single shipment given trade/logistics features.
2. **Next-day VWAP forecast** — predicts tomorrow's volume-weighted average price (INR/kg) for a spice, given today's market context.

---

## Project Structure

```
Turmerix/
├── data/
│   ├── may_and_june_2025.csv           # Raw export trade data (63k rows)
│   ├── prepared_spice_data.csv         # Cleaned transaction-level dataset (31,738 × 26)
│   ├── daily_spice_timeseries.csv      # Daily aggregated per-spice dataset (1,566 × 24)
│   ├── ts_model_eval.png               # TS model: feature importance + scatter
│   └── ts_price_traces.png             # TS model: price traces for 6 spices
├── notebooks/
│   ├── 01_eda_and_training.ipynb       # Original EDA (v1, LEAKY — do not use for training)
│   ├── 02_improved_pipeline.ipynb      # v2 transaction-level model (leakage-free)
│   ├── 03_data_preparation.ipynb       # Data prep: unit standardisation + feature engineering
│   └── 04_timeseries_model.ipynb       # Time-series LightGBM forecast model
├── models/
│   ├── spice_price_model.joblib        # Transaction-level RandomForest pipeline
│   ├── model_metadata.json             # Metrics for transaction model
│   ├── ts_price_model.joblib           # Time-series LightGBM + label encoder
│   └── ts_model_metadata.json          # Metrics for TS model (incl. per-spice MAPE)
├── api/
│   ├── main.py                         # FastAPI inference server
│   └── test_api.py                     # Smoke tests (16/16)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data & train models

Run notebooks in order (all cells top-to-bottom):

```bash
# Step 1 — clean raw data, standardise units, engineer features
jupyter notebook notebooks/03_data_preparation.ipynb

# Step 2 — train time-series LightGBM forecast model
jupyter notebook notebooks/04_timeseries_model.ipynb

# (Optional) — re-train transaction-level model
jupyter notebook notebooks/02_improved_pipeline.ipynb
```

Each notebook auto-saves its outputs to `data/` and `models/`.

### 3. Start the API

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

### 4. Interactive docs (Swagger UI)

```
http://localhost:8000/docs
```

### 5. Run smoke tests

```bash
python api/test_api.py
# Expected: 16/16 tests passed
```

---

## API Endpoints

### Health & Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check (`model_loaded: true/false`) |
| `GET` | `/model/info` | Transaction model — metrics & feature list |
| `GET` | `/model/spices` | Spice names seen during training |
| `GET` | `/model/countries` | Destination countries seen during training |
| `GET` | `/model/ts-info` | TS model — metrics, per-spice MAPE, feature list |

### Transaction-Level Prediction

Predicts `Unit Rate in INR` for a single spice export shipment.  
**No FOB / item rate required** — leakage-free model.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single shipment prediction |
| `POST` | `/predict/batch` | Batch (up to 100 items) |
| `GET` | `/predict/spice` | Quick GET endpoint (browser/curl friendly) |

### Time-Series Forecast

Predicts the **next active shipment day VWAP** (INR/kg) given today's rolling price and volume context.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/forecast` | Next-day VWAP forecast for one spice |
| `POST` | `/forecast/range` | Multi-step forecast: 7d / 1m / 3m / 1y + buy/sell/hold signal |
| `GET` | `/forecast/range/lookup` | **Simplified** — pass only spice + date + horizon; rolling params auto-filled from dataset |

---

## Request & Response Examples

### `POST /predict` — Transaction price

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "spice_name": "Cumin",
    "quantity": 3.45,
    "uqc": "MTS",
    "currency": "USD",
    "mode_of_transport": "SEA",
    "destination_country": "SOUTH AFRICA",
    "exporter_state": "Madhya Pradesh",
    "port": "MUNDRA PORT",
    "day_of_month": 2,
    "month_num": 5,
    "day_of_week": 3,
    "week_of_year": 18
  }'
```

```json
{
  "predicted_unit_rate_inr": 167752.79,
  "spice_name": "Cumin",
  "destination_country": "SOUTH AFRICA",
  "quantity": 3.45,
  "uqc": "MTS",
  "confidence_note": "Model MAPE on test set: 42.1%. Predictions are for indicative/export price guidance only."
}
```

> `predicted_unit_rate_inr` is the price **per original UQC unit** (here: per metric ton). Divide by 1000 for INR/kg.

**`/predict` field reference:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `spice_name` | string | ✅ | e.g. `"Cumin"`, `"Cardamom"`, `"Turmeric"` |
| `quantity` | float > 0 | ✅ | Shipment quantity in UQC units |
| `uqc` | string | ✅ | Unit: `MTS`, `KGS`, `NOS`, `PCS`, `GMS`, etc. |
| `currency` | string | — | Invoice currency (`USD`, `EUR`, `INR`, …). Default: `"USD"` |
| `mode_of_transport` | string | — | `SEA`, `LAND`, or `AIR`. Default: `"SEA"` |
| `destination_country` | string | ✅ | e.g. `"UNITED ARAB EMIRATES"` |
| `exporter_state` | string | — | Indian state. Default: `"Madhya Pradesh"` |
| `port` | string | — | Port of export. Default: `"JNPT - NHAVA SHEVA SEA PORT"` |
| `day_of_month` | int 1–31 | — | Default: `15` |
| `month_num` | int 1–12 | — | Default: `5` |
| `day_of_week` | int 0–6 | — | 0 = Monday. Default: `3` |
| `week_of_year` | int 1–53 | — | ISO week. Default: `20` |
| `log_exporter_volume` | float | — | `log1p(exporter transaction count)`. Null → dataset median |

---

### `POST /forecast` — Next-day VWAP

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "spice_name": "Cumin",
    "lag1_price": 226.0,
    "rolling7_avg_price": 218.5,
    "rolling7_price_std": 15.2,
    "lag7_price": 210.0,
    "price_momentum_1d": 0.035,
    "price_momentum_7d": 0.076,
    "rolling7_volume_kg": 250000.0,
    "rolling14_volume_kg": 490000.0,
    "daily_volume_kg": 38000.0,
    "volume_shock": 1.06,
    "daily_shipment_count": 12,
    "daily_buyer_count": 5,
    "daily_exporter_count": 8,
    "day_of_week": 2,
    "month": 6,
    "week_of_year": 24
  }'
```

```json
{
  "spice_name": "Cumin",
  "predicted_next_day_vwap_inr": 207.07,
  "lag1_price_inr": 226.0,
  "price_change_pct": -8.38,
  "model_test_mape_pct": 12.1,
  "confidence_note": "Spice-level test MAPE: 12.1%. Predicts next active shipment day VWAP. For indicative use only."
}
```

**`/forecast` field reference:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `spice_name` | string | ✅ | — | Must be a known spice (see `/model/ts-info`) |
| `lag1_price` | float > 0 | ✅ | — | Yesterday's VWAP (INR/kg) |
| `rolling7_avg_price` | float > 0 | ✅ | — | 7-day rolling mean VWAP (INR/kg) |
| `rolling7_volume_kg` | float ≥ 0 | ✅ | — | 7-day rolling total volume (kg) |
| `rolling14_volume_kg` | float ≥ 0 | ✅ | — | 14-day rolling total volume (kg) |
| `daily_volume_kg` | float ≥ 0 | ✅ | — | Today's total shipment volume (kg) |
| `daily_shipment_count` | int ≥ 0 | — | `1` | Number of shipments today |
| `daily_buyer_count` | int ≥ 0 | — | `1` | Unique destination countries today |
| `daily_exporter_count` | int ≥ 0 | — | `1` | Unique exporters today |
| `day_of_week` | int 0–6 | — | `0` | 0 = Monday |
| `month` | int 1–12 | — | `5` | Calendar month |
| `week_of_year` | int 1–53 | — | `20` | ISO week number |
| `rolling7_price_std` | float ≥ 0 | — | `0.0` | 7-day rolling std of VWAP |
| `lag7_price` | float > 0 | — | `lag1_price` | VWAP 7 days ago (INR/kg) |
| `price_momentum_1d` | float | — | `0.0` | 1-day return: `vwap/lag1 - 1` |
| `price_momentum_7d` | float | — | `0.0` | 7-day return: `vwap/lag7 - 1` |
| `volume_shock` | float | — | `1.0` | `daily_volume / 7-day avg daily volume` |

### `POST /forecast/range` — Multi-step price outlook + buy/sell/hold

Runs the model **auto-regressively** for the chosen horizon — each predicted VWAP feeds back as `lag1_price` for the next step.

**Supported horizons:** `7d`, `1m` (30 days), `3m` (90 days), `1y` (365 days), or any custom integer up to `730`.

```bash
curl -X POST http://localhost:8000/forecast/range \
  -H "Content-Type: application/json" \
  -d '{
    "spice_name": "Cumin",
    "start_date": "2025-06-20",
    "lag1_price": 226.0,
    "rolling7_avg_price": 218.5,
    "rolling7_price_std": 15.2,
    "lag7_price": 210.0,
    "rolling7_volume_kg": 250000.0,
    "rolling14_volume_kg": 490000.0,
    "daily_volume_kg": 38000.0,
    "daily_shipment_count": 12,
    "daily_buyer_count": 5,
    "daily_exporter_count": 8,
    "horizon": "7d"
  }'
```

```json
{
  "spice_name": "Cumin",
  "anchor_date": "2025-06-20",
  "anchor_price_inr": 226.0,
  "horizon_days": 7,
  "forecast": [
    { "date": "2025-06-21", "predicted_vwap_inr": 199.22, "pct_change_vs_today": -11.85 },
    { "date": "2025-06-22", "predicted_vwap_inr": 194.94, "pct_change_vs_today": -13.74 },
    { "date": "2025-06-23", "predicted_vwap_inr": 175.42, "pct_change_vs_today": -22.38 },
    "..."
  ],
  "summary": {
    "min_price_inr": 168.31,
    "max_price_inr": 199.22,
    "avg_price_inr": 180.55,
    "final_price_inr": 171.06,
    "total_pct_change": -24.31,
    "trend": "downward",
    "signal": "SELL / LIQUIDATE",
    "reasoning": "Price predicted to fall -24.3% over the horizon, exceeding the model's uncertainty band. Consider selling or reducing stock exposure. (Spice MAPE ≈ 12.1% — treat as indicative.)"
  },
  "confidence_note": "Auto-regressive 7-day forecast. Spice-level test MAPE: 12.1%. Uncertainty compounds with horizon — use short-range forecasts for trading decisions."
}
```

**Signal logic:**

| Signal | Condition |
|--------|-----------|
| `BUY / HOLD STOCK` | Predicted rise > 50% of spice MAPE |
| `SELL / LIQUIDATE` | Predicted fall > 50% of spice MAPE |
| `HOLD` | Move within model uncertainty band |

> Uncertainty compounds with horizon length. `7d` forecasts are most reliable; `1y` forecasts are directional only.

**`/forecast/range` field reference:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `spice_name` | string | ✅ | — | Must be a known spice |
| `start_date` | string | ✅ | — | `YYYY-MM-DD` — today's anchor date |
| `lag1_price` | float > 0 | ✅ | — | Today's VWAP (INR/kg) |
| `rolling7_avg_price` | float > 0 | ✅ | — | 7-day rolling mean VWAP |
| `rolling7_volume_kg` | float ≥ 0 | ✅ | — | 7-day rolling total volume (kg) |
| `rolling14_volume_kg` | float ≥ 0 | ✅ | — | 14-day rolling total volume (kg) |
| `daily_volume_kg` | float ≥ 0 | ✅ | — | Today's total shipment volume (kg) |
| `horizon` | string | — | `"7d"` | `7d`, `1m`, `3m`, `1y`, or integer 1–730 |
| `rolling7_price_std` | float ≥ 0 | — | `0.0` | 7-day rolling std of VWAP |
| `lag7_price` | float > 0 | — | `lag1_price` | VWAP 7 days ago |
| `daily_shipment_count` | int ≥ 0 | — | `1` | Shipments today |
| `daily_buyer_count` | int ≥ 0 | — | `1` | Unique destination countries today |
| `daily_exporter_count` | int ≥ 0 | — | `1` | Unique exporters today |

---

### `GET /forecast/range/lookup` — Simplified forecast (recommended)

The easiest way to call the forecast. Pass only **three parameters** — rolling context is looked up automatically from `daily_spice_timeseries.csv`.

```bash
# Browser / curl — no JSON body needed
curl "http://localhost:8000/forecast/range/lookup?spice_name=Cumin&anchor_date=2025-06-30&horizon=7d"
```

Or open directly in a browser:
```
http://localhost:8000/forecast/range/lookup?spice_name=Turmeric&anchor_date=2025-06-30&horizon=1m
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `spice_name` | ✅ | — | e.g. `Cumin`, `Turmeric`, `Cardamom` |
| `anchor_date` | ✅ | — | `YYYY-MM-DD`. Falls back to nearest earlier date if exact date has no data for that spice |
| `horizon` | — | `7d` | `7d`, `1m`, `3m`, `1y`, or integer 1–730 |

**Response** is identical to `POST /forecast/range` — day-by-day prices + buy/sell/hold signal.

> **Dataset coverage:** May 1 – Jun 30 2025. Dates outside this range will use the nearest available date.

---

### `GET /predict/spice` — Quick browser endpoint

```
http://localhost:8000/predict/spice?spice_name=Turmeric&quantity=0.5&uqc=MTS&destination_country=UNITED+ARAB+EMIRATES&currency=USD&month_num=6&day_of_month=15
```

### `GET /model/ts-info`

Returns full TS model metadata including per-spice test MAPE for all 25 spices.

```bash
curl http://localhost:8000/model/ts-info | python -m json.tool
```

---

## Model Performance Summary

### Transaction-Level Model (`/predict`)

| Metric | Value |
|--------|-------|
| Algorithm | RandomForest |
| Target | `log1p(Unit Rate in INR)` |
| Train R² | 0.974 |
| Test R² | 0.881 |
| CV R² | 0.956 ± 0.002 |
| Test MAPE | 42.1% |
| Test SMAPE | 28.7% |

Features: `Spice Name`, `currency_bucket`, `UQC`, `Mode of Transport`, `Country`, `Exporter State`, `location`, `qty_bin`, `log_quantity`, `day_of_month`, `month_num`, `day_of_week`, `week_of_year`, `is_month_end`, `log_exporter_volume`

**Excluded (data leakage):** `FOB`, `ITEM_RATE`, `Total Value in FC`, `EX EX RATE`, `FOB USD`, `VALUE USD`, `UNIT RATE US$`, `log_fob_per_unit`

### Time-Series Forecast Model (`/forecast`)

| Metric | Value |
|--------|-------|
| Algorithm | LightGBM (GBDT) |
| Target | `log1p(next-day VWAP INR/kg)` |
| Train period | May 1 – Jun 14, 2025 (854 rows) |
| Test period | Jun 15 – Jun 29, 2025 (277 rows) |
| Train R² | 0.903 |
| Test R² | 0.797 |
| Test MAPE | 35.6% |
| Test SMAPE | 29.9% |
| Naive lag-1 baseline MAPE | 46.8% |
| Best iteration | 79 |

**Per-spice test MAPE (selected):**

| Spice | MAPE | Spice | MAPE |
|-------|------|-------|------|
| Red Chilli | 4.4% | Cloves | 34.1% |
| Coffee | 11.0% | Chilli | 34.1% |
| Cumin | 12.1% | Pepper | 44.4% |
| Turmeric | 14.9% | Tea | 50.6% |
| Fenugreek | 17.9% | Anise | 59.5% |
| Ajwain | 21.5% | Bay Leaf | 75.1% |
| Coriander | 24.5% | Ginger | 77.1% |
| Cardamom | 27.4% | Black Pepper | 91.4% |

> High MAPE for Ginger/Black Pepper is driven by extreme VWAP volatility from mixed NOS/PCS retail shipments. Model still outperforms naive lag-1 baseline for both.

---

## Data Pipeline

```
may_and_june_2025.csv (63k rows)
        │
        ▼  notebooks/03_data_preparation.ipynb
        │  • Standardise all units → KG
        │  • Parse NOS/PCS weights from ITEM descriptions
        │  • Drop non-standardisable rows (MTR/SET/DOZ/PRS)
        │  • Compute price_per_kg_inr
        │  • IQR outlier removal (3× fence per spice)
        │  • Engineer demand features (rolling volume, VWAP, lags)
        │  • Add volatility, momentum, volume-shock features
        ├──▶ prepared_spice_data.csv    (31,738 × 26) — transaction-level
        └──▶ daily_spice_timeseries.csv (1,566 × 24)  — daily aggregated

daily_spice_timeseries.csv
        │
        ▼  notebooks/04_timeseries_model.ipynb
        │  • Next-day VWAP target construction
        │  • log1p transform
        │  • Time-based 75/25 split
        │  • LightGBM with early stopping (MAPE metric)
        └──▶ models/ts_price_model.joblib + ts_model_metadata.json
```

---

## Supported Spices

Ajwain, Anise, Bay Leaf, Black Pepper, Cardamom, Chilli, Chilli Powder, Cinnamon, Cloves, Coffee, Coriander, Cumin, Dill, Fennel, Fenugreek, Ginger, Mace, Mustard, Nutmeg, Pepper, Red Chilli, Saffron, Tea, Thyme, Turmeric, Vanilla

Use `GET /model/ts-info` for the exact list accepted by `/forecast`.
