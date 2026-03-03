"""
Turmerix — Indian Spice Price Prediction API
FastAPI inference server v2 (leakage-free) + time-series forecast
"""

import os
import json
import logging
from typing import Optional, List
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("turmerix")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH    = BASE_DIR / "models" / "spice_price_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
TS_MODEL_PATH = BASE_DIR / "models" / "ts_price_model.joblib"
TS_META_PATH  = BASE_DIR / "models" / "ts_model_metadata.json"
TS_DATA_PATH  = BASE_DIR / "data"   / "daily_spice_timeseries.csv"

# ─── Load model on startup ────────────────────────────────────────────────────
_model = None
_metadata = None
_ts_bundle = None   # dict: {model, label_encoder, features}
_ts_metadata = None
_ts_data: Optional["pd.DataFrame"] = None  # daily_spice_timeseries.csv loaded at startup


def load_artifacts():
    global _model, _metadata, _ts_bundle, _ts_metadata, _ts_data
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run the training notebook first: notebooks/01_eda_and_training.ipynb"
        )
    logger.info(f"Loading model from {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)

    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
        logger.info(f"Model metadata loaded. R²={_metadata.get('test_r2', 'N/A')}, "
                    f"MAPE={_metadata.get('test_mape_pct', 'N/A')}%")
    else:
        _metadata = {}
        logger.warning("No metadata file found, running without it")

    if TS_MODEL_PATH.exists():
        logger.info(f"Loading time-series model from {TS_MODEL_PATH}")
        _ts_bundle = joblib.load(TS_MODEL_PATH)
        if TS_META_PATH.exists():
            with open(TS_META_PATH) as f:
                _ts_metadata = json.load(f)
            logger.info(f"TS model metadata loaded. Test MAPE={_ts_metadata.get('test_mape', 'N/A')}%")
        else:
            _ts_metadata = {}
    else:
        logger.warning(f"Time-series model not found at {TS_MODEL_PATH}. /forecast endpoint will be unavailable.")

    if TS_DATA_PATH.exists():
        logger.info(f"Loading timeseries data from {TS_DATA_PATH}")
        _ts_data = pd.read_csv(TS_DATA_PATH, parse_dates=["date"])
        logger.info(f"Timeseries data loaded: {len(_ts_data)} rows, {_ts_data['Spice Name'].nunique()} spices")
    else:
        logger.warning(f"Timeseries data not found at {TS_DATA_PATH}. /forecast/range/lookup will be unavailable.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Turmerix — Indian Spice Price Prediction API",
    description=(
        "Predict the unit price (INR) of Indian spices based on export/import transaction features. "
        "Trained on May–June 2025 trade data."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    load_artifacts()


# ─── Feature Engineering v2 (leakage-free, mirrors 02_improved_pipeline.ipynb) ─
# NOTE: FOB, ITEM_RATE, log_fob_per_unit are EXCLUDED — they directly encode the target.

# Currencies treated as their own bucket; null → UNKNOWN; literal N/A → NA_CURR
MAJOR_CURRENCIES = {
    "USD", "EUR", "GBP", "CAD", "AUD", "SGD", "AED",
    "INR", "NA_CURR", "BHD", "SAR", "QAR"
}

# Quantity bin boundaries matching training
QTY_BINS   = [0, 0.1, 1, 10, 100, 1000, float("inf")]
QTY_LABELS = ["micro", "tiny", "small", "medium", "large", "bulk"]

# Default exporter volume: log1p of median train volume (populated after model load)
_DEFAULT_LOG_EXPORTER_VOL = 5.0  # ≈ exp(5)-1 ≈ 147 transactions; safe fallback

CAT_FEATURES = [
    "Spice Name", "currency_bucket", "UQC", "Mode of Transport",
    "Country", "Exporter State", "location", "qty_bin",
]
NUM_FEATURES = [
    "log_quantity", "day_of_month", "month_num", "day_of_week",
    "week_of_year", "is_month_end", "log_exporter_volume",
]


def _currency_bucket(raw: str) -> str:
    """Normalise currency string to bucket label."""
    s = str(raw).strip().upper()
    if s in ("NAN", "NONE", "", "NULL"):
        return "UNKNOWN"
    if s == "N/A":
        return "NA_CURR"
    return s if s in MAJOR_CURRENCIES else "OTHER"


def _qty_bin(quantity: float) -> str:
    """Assign quantity bin label matching training bins."""
    for i, (lo, hi) in enumerate(zip(QTY_BINS[:-1], QTY_BINS[1:])):
        if lo < quantity <= hi:
            return QTY_LABELS[i]
    return "bulk"


def build_feature_row(req: "PricePredictRequest") -> pd.DataFrame:
    quantity = max(req.quantity, 0.0)

    row = {
        # ── Categorical ────────────────────────────────────────────────────────
        "Spice Name":         req.spice_name.strip().title(),
        "currency_bucket":    _currency_bucket(req.currency),
        "UQC":                req.uqc.upper().strip(),
        "Mode of Transport":  req.mode_of_transport.upper().strip(),
        "Country":            req.destination_country.upper().strip(),
        "Exporter State":     req.exporter_state.strip().title(),
        "location":           req.port.strip(),
        "qty_bin":            _qty_bin(quantity),
        # ── Numerical (NO leaky features) ──────────────────────────────────────
        "log_quantity":          float(np.log1p(quantity)),
        "day_of_month":          int(req.day_of_month),
        "month_num":             int(req.month_num),
        "day_of_week":           int(req.day_of_week),
        "week_of_year":          int(req.week_of_year),
        "is_month_end":          int(req.day_of_month >= 25),
        "log_exporter_volume":   float(
            req.log_exporter_volume
            if req.log_exporter_volume is not None
            else _DEFAULT_LOG_EXPORTER_VOL
        ),
    }
    return pd.DataFrame([row])


# ─── Request / Response Schemas ───────────────────────────────────────────────
class PricePredictRequest(BaseModel):
    spice_name: str = Field(
        ..., description="Spice category name (e.g. Cumin, Coriander, Turmeric, Ginger)")
    quantity: float = Field(
        ..., gt=0, description="Export quantity in UQC units")
    uqc: str = Field(
        "MTS", description="Unit of quantity: MTS, KGS, NOS, PCS, GMS, etc.")
    currency: str = Field(
        "USD", description="Invoice currency (USD, EUR, CAD, INR, AED, etc.)")
    mode_of_transport: str = Field(
        "SEA", description="SEA, LAND, or AIR")
    destination_country: str = Field(
        ..., description="Destination country name (e.g. UNITED ARAB EMIRATES)")
    exporter_state: str = Field(
        "Madhya Pradesh", description="Indian state of the exporter")
    port: str = Field(
        "JNPT - NHAVA SHEVA SEA PORT", description="Port of export")
    day_of_month: int = Field(
        15, ge=1, le=31, description="Day of the shipment date (1–31)")
    month_num: int = Field(
        5, ge=1, le=12, description="Month number (5=May, 6=June)")
    day_of_week: int = Field(
        3, ge=0, le=6, description="Day of week (0=Monday … 6=Sunday)")
    week_of_year: int = Field(
        20, ge=1, le=53, description="ISO week number (1–53)")
    log_exporter_volume: Optional[float] = Field(
        None,
        description=(
            "log1p(number of transactions by this exporter in training data). "
            "Leave null to use dataset median (~5.0)."
        )
    )

    @validator("spice_name")
    def spice_name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("spice_name must not be empty")
        return v


class PricePredictResponse(BaseModel):
    predicted_unit_rate_inr: float = Field(..., description="Predicted unit price in INR")
    spice_name: str
    destination_country: str
    quantity: float
    uqc: str
    confidence_note: str


class BatchPricePredictRequest(BaseModel):
    requests: List[PricePredictRequest] = Field(..., max_items=100)


class BatchPricePredictResponse(BaseModel):
    predictions: List[PricePredictResponse]
    count: int


class ModelInfoResponse(BaseModel):
    version: str
    model_name: str
    model_class: str
    train_r2: float
    test_r2: float
    r2_overfit_gap: float
    cv_r2_mean: float
    cv_r2_std: float
    test_mae_inr: float
    test_mape_pct: float
    test_smape_pct: float
    test_rmse_inr: float
    train_rows: int
    test_rows: int
    n_features: int
    target: str
    cat_features: List[str]
    num_features: List[str]
    leaky_features_excluded: List[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Turmerix Spice Price Prediction API",
        "status": "running",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    model_loaded = _model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    if _metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    return ModelInfoResponse(
        version=_metadata.get("version", "v2"),
        model_name=_metadata.get("model_name", "unknown"),
        model_class=_metadata.get("model_class", "unknown"),
        train_r2=_metadata.get("train_r2", 0.0),
        test_r2=_metadata.get("test_r2", 0.0),
        r2_overfit_gap=_metadata.get("r2_overfit_gap", 0.0),
        cv_r2_mean=_metadata.get("cv_r2_mean", 0.0),
        cv_r2_std=_metadata.get("cv_r2_std", 0.0),
        test_mae_inr=_metadata.get("test_mae_inr", 0.0),
        test_mape_pct=_metadata.get("test_mape_pct", 0.0),
        test_smape_pct=_metadata.get("test_smape_pct", 0.0),
        test_rmse_inr=_metadata.get("test_rmse_inr", 0.0),
        train_rows=_metadata.get("train_rows", 0),
        test_rows=_metadata.get("test_rows", 0),
        n_features=_metadata.get("n_features", len(CAT_FEATURES) + len(NUM_FEATURES)),
        target=_metadata.get("target", "Unit Rate in INR"),
        cat_features=_metadata.get("cat_features", CAT_FEATURES),
        num_features=_metadata.get("num_features", NUM_FEATURES),
        leaky_features_excluded=_metadata.get("leaky_features_excluded", []),
    )


@app.get("/model/spices", tags=["Model"])
def list_spices():
    """Return all spice names seen during training."""
    if _metadata:
        return {"spices": _metadata.get("spice_names", [])}
    return {"spices": []}


@app.get("/model/countries", tags=["Model"])
def list_countries():
    """Return all destination countries seen during training."""
    if _metadata:
        return {"countries": _metadata.get("countries", [])}
    return {"countries": []}


@app.post("/predict", response_model=PricePredictResponse, tags=["Prediction"])
def predict_price(request: PricePredictRequest):
    """
    Predict the unit price in INR for a single spice export transaction.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    try:
        X = build_feature_row(request)
        log_pred = _model.predict(X)[0]
        predicted_inr = float(np.expm1(log_pred))
        predicted_inr = max(predicted_inr, 0.0)

        mape = _metadata.get("test_mape_pct", None) if _metadata else None
        note = (
            f"Model MAPE on test set: {mape:.1f}%. "
            "Predictions are for indicative/export price guidance only."
            if mape else "Use for indicative purposes only."
        )

        return PricePredictResponse(
            predicted_unit_rate_inr=round(predicted_inr, 4),
            spice_name=request.spice_name,
            destination_country=request.destination_country,
            quantity=request.quantity,
            uqc=request.uqc,
            confidence_note=note,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPricePredictResponse, tags=["Prediction"])
def predict_batch(batch: BatchPricePredictRequest):
    """
    Predict unit prices in INR for a batch of up to 100 transactions.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    if len(batch.requests) == 0:
        raise HTTPException(status_code=400, detail="Empty request list")

    try:
        rows = [build_feature_row(r) for r in batch.requests]
        X_batch = pd.concat(rows, ignore_index=True)
        log_preds = _model.predict(X_batch)
        preds_inr = np.expm1(log_preds)

        mape = _metadata.get("test_mape_pct", None) if _metadata else None
        note = (
            f"Model MAPE on test set: {mape:.1f}%." if mape else "Indicative only."
        )

        predictions = []
        for req, pred in zip(batch.requests, preds_inr):
            predictions.append(PricePredictResponse(
                predicted_unit_rate_inr=round(float(max(pred, 0.0)), 4),
                spice_name=req.spice_name,
                destination_country=req.destination_country,
                quantity=req.quantity,
                uqc=req.uqc,
                confidence_note=note,
            ))

        return BatchPricePredictResponse(predictions=predictions, count=len(predictions))

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/predict/spice", response_model=PricePredictResponse, tags=["Prediction"])
def predict_by_spice(
    spice_name: str = Query(...),
    quantity: float = Query(1.0, gt=0),
    uqc: str = Query("MTS"),
    destination_country: str = Query("UNITED ARAB EMIRATES"),
    currency: str = Query("USD"),
    mode_of_transport: str = Query("SEA"),
    exporter_state: str = Query("Madhya Pradesh"),
    port: str = Query("JNPT - NHAVA SHEVA SEA PORT"),
    month_num: int = Query(5, ge=1, le=12),
    day_of_month: int = Query(15, ge=1, le=31),
):
    """
    Quick GET endpoint for price prediction — useful for browser/curl testing.
    No FOB or item rate required (leakage-free model).
    """
    import datetime
    date = datetime.date(2025, month_num, day_of_month)
    req = PricePredictRequest(
        spice_name=spice_name,
        quantity=quantity,
        uqc=uqc,
        currency=currency,
        mode_of_transport=mode_of_transport,
        destination_country=destination_country,
        exporter_state=exporter_state,
        port=port,
        day_of_month=day_of_month,
        month_num=month_num,
        day_of_week=date.weekday(),
        week_of_year=date.isocalendar()[1],
    )
    return predict_price(req)


# ─── Time-Series Forecast Schemas ─────────────────────────────────────────────

class ForecastRequest(BaseModel):
    spice_name: str = Field(
        ..., description="Spice name (must match training set, e.g. 'Cumin', 'Cardamom')")
    lag1_price: float = Field(
        ..., gt=0, description="Yesterday's VWAP (INR/kg)")
    rolling7_avg_price: float = Field(
        ..., gt=0, description="7-day rolling average VWAP (INR/kg)")
    rolling7_price_std: float = Field(
        0.0, ge=0, description="7-day rolling std of VWAP (0 if unknown)")
    lag7_price: Optional[float] = Field(
        None, description="VWAP 7 days ago (INR/kg). Null → filled with lag1_price")
    price_momentum_1d: Optional[float] = Field(
        None, description="1-day price return (vwap/lag1 - 1). Null → computed from lag1 if possible")
    price_momentum_7d: Optional[float] = Field(
        None, description="7-day price return (vwap/lag7 - 1). Null → 0.0")
    rolling7_volume_kg: float = Field(
        ..., ge=0, description="7-day rolling total volume in kg")
    rolling14_volume_kg: float = Field(
        ..., ge=0, description="14-day rolling total volume in kg")
    daily_volume_kg: float = Field(
        ..., ge=0, description="Today's total shipment volume in kg")
    volume_shock: Optional[float] = Field(
        None, description="daily_volume / 7-day avg daily volume. Null → 1.0")
    daily_shipment_count: int = Field(
        1, ge=0, description="Number of shipments today")
    daily_buyer_count: int = Field(
        1, ge=0, description="Number of unique destination countries today")
    daily_exporter_count: int = Field(
        1, ge=0, description="Number of unique exporters today")
    day_of_week: int = Field(
        0, ge=0, le=6, description="Day of week (0=Monday … 6=Sunday)")
    month: int = Field(
        5, ge=1, le=12, description="Calendar month (1–12)")
    week_of_year: int = Field(
        20, ge=1, le=53, description="ISO week number (1–53)")

    @validator("spice_name")
    def spice_not_empty(cls, v):
        if not v.strip():
            raise ValueError("spice_name must not be empty")
        return v.strip().title()


class ForecastResponse(BaseModel):
    spice_name: str
    predicted_next_day_vwap_inr: float = Field(..., description="Predicted next-day VWAP in INR/kg")
    lag1_price_inr: float = Field(..., description="Input lag1 price used as context")
    price_change_pct: float = Field(..., description="Predicted change vs lag1 price (%)")
    model_test_mape_pct: float = Field(..., description="Model test MAPE for reference")
    confidence_note: str


class TSModelInfoResponse(BaseModel):
    model_type: str
    target: str
    features: List[str]
    spices: List[str]
    train_date_range: List[str]
    test_date_range: List[str]
    train_rows: int
    test_rows: int
    best_iteration: int
    train_r2: float
    test_r2: float
    train_mape: float
    test_mape: float
    test_smape: float
    naive_baseline_mape: float
    r2_overfit_gap: float
    per_spice_test_mape: dict


# ─── Forecast Feature Builder ─────────────────────────────────────────────────

def build_forecast_row(req: ForecastRequest, spice_encoded: int) -> np.ndarray:
    """Map a ForecastRequest to the feature vector expected by ts_price_model."""
    lag7  = req.lag7_price if req.lag7_price is not None else req.lag1_price
    mom1d = req.price_momentum_1d if req.price_momentum_1d is not None else 0.0
    mom7d = req.price_momentum_7d if req.price_momentum_7d is not None else 0.0
    vshock = req.volume_shock if req.volume_shock is not None else 1.0

    # Feature order must exactly match FEATURES list in notebook
    row = [
        spice_encoded,
        req.lag1_price,
        lag7,
        req.rolling7_avg_price,
        req.rolling7_price_std,
        mom1d,
        mom7d,
        req.rolling7_volume_kg,
        req.rolling14_volume_kg,
        vshock,
        req.daily_volume_kg,
        float(req.daily_shipment_count),
        float(req.daily_buyer_count),
        float(req.daily_exporter_count),
        float(req.day_of_week),
        float(req.month),
        float(req.week_of_year),
    ]
    return np.array(row, dtype=np.float64).reshape(1, -1)


# ─── Forecast Endpoints ───────────────────────────────────────────────────────

@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
def forecast_next_day(request: ForecastRequest):
    """
    Predict the **next active shipment day VWAP** (INR/kg) for a spice,
    given today's market context (lags, rolling stats, volume signals).

    Uses the time-series LightGBM model trained on `daily_spice_timeseries.csv`.
    """
    if _ts_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Time-series model not loaded. Run notebooks/04_timeseries_model.ipynb first."
        )

    le: "LabelEncoder" = _ts_bundle["label_encoder"]
    ts_model = _ts_bundle["model"]

    known_spices = list(le.classes_)
    spice_title = request.spice_name  # already title-cased by validator
    if spice_title not in known_spices:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown spice '{spice_title}'. Known spices: {known_spices}"
        )

    try:
        spice_encoded = int(le.transform([spice_title])[0])
        X = build_forecast_row(request, spice_encoded)
        log_pred = ts_model.predict(X)[0]
        predicted_vwap = float(np.expm1(log_pred))
        predicted_vwap = max(predicted_vwap, 0.0)

        price_change_pct = (predicted_vwap / request.lag1_price - 1.0) * 100.0

        test_mape = float(_ts_metadata.get("test_mape", 0.0)) if _ts_metadata else 0.0
        per_spice = (_ts_metadata or {}).get("per_spice_test_mape", {})
        spice_mape = per_spice.get(spice_title, test_mape)

        note = (
            f"Spice-level test MAPE: {spice_mape:.1f}%. "
            "Predicts next active shipment day VWAP. For indicative use only."
        )

        return ForecastResponse(
            spice_name=spice_title,
            predicted_next_day_vwap_inr=round(predicted_vwap, 4),
            lag1_price_inr=round(request.lag1_price, 4),
            price_change_pct=round(price_change_pct, 2),
            model_test_mape_pct=round(spice_mape, 1),
            confidence_note=note,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


PRESET_HORIZONS = {"7d": 7, "1m": 30, "3m": 90, "1y": 365}


class ForecastRangeRequest(BaseModel):
    spice_name: str = Field(..., description="Spice name (must match training set)")
    start_date: str = Field(
        ..., description="Start date in YYYY-MM-DD format (today's date, i.e. the day whose lag1 you provide)")
    lag1_price: float = Field(..., gt=0, description="Today's VWAP (INR/kg) — the known anchor price")
    rolling7_avg_price: float = Field(..., gt=0, description="7-day rolling average VWAP (INR/kg) as of today")
    rolling7_price_std: float = Field(0.0, ge=0, description="7-day rolling std of VWAP as of today")
    lag7_price: Optional[float] = Field(None, description="VWAP 7 days ago. Null → lag1_price")
    rolling7_volume_kg: float = Field(..., ge=0, description="7-day rolling total volume (kg) as of today")
    rolling14_volume_kg: float = Field(..., ge=0, description="14-day rolling total volume (kg) as of today")
    daily_volume_kg: float = Field(..., ge=0, description="Today's total shipment volume (kg)")
    daily_shipment_count: int = Field(1, ge=0, description="Number of shipments today")
    daily_buyer_count: int = Field(1, ge=0, description="Number of unique destination countries today")
    daily_exporter_count: int = Field(1, ge=0, description="Number of unique exporters today")
    horizon: str = Field(
        "7d",
        description=(
            "Forecast horizon. Presets: '7d' (7 days), '1m' (30 days), "
            "'3m' (90 days), '1y' (365 days). Or pass a custom integer like '45'."
        ),
    )

    @validator("spice_name")
    def spice_not_empty(cls, v):
        if not v.strip():
            raise ValueError("spice_name must not be empty")
        return v.strip().title()

    @validator("start_date")
    def valid_date(cls, v):
        try:
            pd.Timestamp(v)
        except Exception:
            raise ValueError(f"start_date must be YYYY-MM-DD, got '{v}'")
        return v

    @validator("horizon")
    def valid_horizon(cls, v):
        if v in PRESET_HORIZONS:
            return v
        try:
            n = int(v)
            if n < 1 or n > 730:
                raise ValueError()
            return v
        except (ValueError, TypeError):
            raise ValueError(
                f"horizon must be one of {list(PRESET_HORIZONS.keys())} or a number 1–730, got '{v}'"
            )


class DayForecast(BaseModel):
    date: str = Field(..., description="Calendar date (YYYY-MM-DD)")
    predicted_vwap_inr: float = Field(..., description="Predicted VWAP for that date (INR/kg)")
    pct_change_vs_today: float = Field(..., description="% change vs today's anchor price")


class ForecastRangeResponse(BaseModel):
    spice_name: str
    anchor_date: str = Field(..., description="The start date requested")
    context_date_used: str = Field(..., description="Actual dataset date whose rolling context was used (may differ from anchor_date)")
    context_note: str = Field(..., description="Explains any date fallback applied")
    anchor_price_inr: float = Field(..., description="VWAP on context_date_used (INR/kg)")
    horizon_days: int
    forecast: List[DayForecast]
    summary: dict = Field(
        ...,
        description=(
            "Aggregated outlook: min/max/avg predicted price, total % change, "
            "trend direction, and a buy/sell/hold signal."
        ),
    )
    confidence_note: str


def _resolve_horizon(horizon: str) -> int:
    if horizon in PRESET_HORIZONS:
        return PRESET_HORIZONS[horizon]
    return int(horizon)


def _buysell_signal(total_pct_change: float, spice_mape: float) -> dict:
    """
    Return a buy / sell / hold recommendation with reasoning.
    Signal is suppressed (hold) when the predicted move is within model uncertainty.
    """
    uncertainty_band = spice_mape / 100.0 * 100.0  # convert to same units as pct_change

    if abs(total_pct_change) < uncertainty_band * 0.5:
        signal = "HOLD"
        reason = (
            f"Predicted price change ({total_pct_change:+.1f}%) is within the model's "
            f"uncertainty band (spice MAPE ≈ {spice_mape:.1f}%). No confident directional signal."
        )
    elif total_pct_change >= uncertainty_band * 0.5:
        signal = "SELL / EXPORT NOW"
        reason = (
            f"Price predicted to rise {total_pct_change:+.1f}% over the horizon, "
            "but this exceeds the uncertainty threshold — "
            "if you have stock, locking in current prices before the projected peak may be beneficial. "
            f"(Spice MAPE ≈ {spice_mape:.1f}% — treat as indicative.)"
        )
        if total_pct_change > 0:
            signal = "BUY / HOLD STOCK"
            reason = (
                f"Price predicted to rise {total_pct_change:+.1f}% over the horizon. "
                "Consider buying or holding inventory — expected appreciation exceeds model uncertainty. "
                f"(Spice MAPE ≈ {spice_mape:.1f}% — treat as indicative.)"
            )
    else:
        signal = "SELL / LIQUIDATE"
        reason = (
            f"Price predicted to fall {total_pct_change:+.1f}% over the horizon, "
            "exceeding the model's uncertainty band. "
            "Consider selling or reducing stock exposure. "
            f"(Spice MAPE ≈ {spice_mape:.1f}% — treat as indicative.)"
        )

    return {"signal": signal, "reasoning": reason}


@app.post("/forecast/range", response_model=ForecastRangeResponse, tags=["Forecast"])
def forecast_range(request: ForecastRangeRequest):
    """
    Iterative multi-step price forecast for a spice over a chosen horizon.

    Supported horizons: **7d** (1 week), **1m** (30 days), **3m** (90 days),
    **1y** (365 days), or any custom integer up to 730.

    The model is run **auto-regressively** — each predicted VWAP becomes the
    `lag1_price` for the next step, with rolling stats updated accordingly.
    Uncertainty compounds with horizon length; treat longer-range forecasts as
    directional guidance only.

    Returns:
    - Day-by-day predicted VWAP series
    - Summary stats (min / max / avg / trend)
    - **Buy / Sell / Hold signal** calibrated against the spice's own test MAPE
    """
    if _ts_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Time-series model not loaded. Run notebooks/04_timeseries_model.ipynb first.",
        )

    le = _ts_bundle["label_encoder"]
    ts_model = _ts_bundle["model"]
    known_spices = list(le.classes_)
    spice_title = request.spice_name

    if spice_title not in known_spices:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown spice '{spice_title}'. Known spices: {known_spices}",
        )

    try:
        spice_encoded = int(le.transform([spice_title])[0])
        per_spice = (_ts_metadata or {}).get("per_spice_test_mape", {})
        test_mape  = float((_ts_metadata or {}).get("test_mape", 0.0))
        spice_mape = per_spice.get(spice_title, test_mape)

        horizon_days = _resolve_horizon(request.horizon)
        anchor_date  = pd.Timestamp(request.start_date)
        anchor_price = request.lag1_price

        # ── Rolling state initialised from request ────────────────────────────
        lag1   = request.lag1_price
        lag7   = request.lag7_price if request.lag7_price is not None else request.lag1_price
        r7avg  = request.rolling7_avg_price
        r7std  = request.rolling7_price_std
        r7vol  = request.rolling7_volume_kg
        r14vol = request.rolling14_volume_kg
        dvol   = request.daily_volume_kg
        r7avg_daily_vol = r7vol / 7.0 if r7vol > 0 else 1.0

        # Circular buffer of last 7 predictions to update rolling avg/std
        price_window: list[float] = [lag1] * 7

        forecast_days: list[DayForecast] = []

        for step in range(1, horizon_days + 1):
            forecast_date = anchor_date + pd.Timedelta(days=step)
            dow  = forecast_date.dayofweek
            mon  = forecast_date.month
            woy  = int(forecast_date.isocalendar()[1])

            mom1d = (lag1 / price_window[-2] - 1.0) if len(price_window) >= 2 and price_window[-2] > 0 else 0.0
            mom7d = (lag1 / lag7 - 1.0) if lag7 > 0 else 0.0
            vshock = dvol / r7avg_daily_vol if r7avg_daily_vol > 0 else 1.0

            row = np.array([
                spice_encoded,
                lag1,
                lag7,
                r7avg,
                r7std,
                mom1d,
                mom7d,
                r7vol,
                r14vol,
                vshock,
                dvol,
                float(request.daily_shipment_count),
                float(request.daily_buyer_count),
                float(request.daily_exporter_count),
                float(dow),
                float(mon),
                float(woy),
            ], dtype=np.float64).reshape(1, -1)

            log_pred = ts_model.predict(row)[0]
            pred_vwap = float(max(np.expm1(log_pred), 0.0))

            pct_vs_today = (pred_vwap / anchor_price - 1.0) * 100.0
            forecast_days.append(DayForecast(
                date=forecast_date.strftime("%Y-%m-%d"),
                predicted_vwap_inr=round(pred_vwap, 2),
                pct_change_vs_today=round(pct_vs_today, 2),
            ))

            # ── Update rolling state for next step ───────────────────────────
            # Slide the 7-price window
            if step >= 7:
                lag7 = price_window[0]       # the value that's now 7 steps old
            price_window.append(pred_vwap)
            if len(price_window) > 7:
                price_window.pop(0)

            r7avg = float(np.mean(price_window))
            r7std = float(np.std(price_window)) if len(price_window) > 1 else 0.0

            # Volume: keep constant (no future volume data available)
            r7avg_daily_vol = dvol  # treat every predicted day as similar volume
            r14vol = r7vol + dvol   # rough approximation
            r7vol  = dvol * 7.0

            lag1 = pred_vwap

        # ── Summary ──────────────────────────────────────────────────────────
        prices = [d.predicted_vwap_inr for d in forecast_days]
        final_pct = forecast_days[-1].pct_change_vs_today
        trend = "upward" if final_pct > 2 else "downward" if final_pct < -2 else "sideways"

        signal_info = _buysell_signal(final_pct, spice_mape)

        summary = {
            "min_price_inr": round(min(prices), 2),
            "max_price_inr": round(max(prices), 2),
            "avg_price_inr": round(float(np.mean(prices)), 2),
            "final_price_inr": round(prices[-1], 2),
            "total_pct_change": round(final_pct, 2),
            "trend": trend,
            "signal": signal_info["signal"],
            "reasoning": signal_info["reasoning"],
        }

        note = (
            f"Auto-regressive {horizon_days}-day forecast. "
            f"Spice-level test MAPE: {spice_mape:.1f}%. "
            "Uncertainty compounds with horizon — use short-range forecasts for trading decisions. "
            "For indicative use only."
        )

        return ForecastRangeResponse(
            spice_name=spice_title,
            anchor_date=request.start_date,
            context_date_used=request.start_date,
            context_note="Context values provided directly in request.",
            anchor_price_inr=round(anchor_price, 2),
            horizon_days=horizon_days,
            forecast=forecast_days,
            summary=summary,
            confidence_note=note,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"forecast/range failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast range error: {str(e)}")


@app.get("/forecast/range/lookup", response_model=ForecastRangeResponse, tags=["Forecast"])
def forecast_range_lookup(
    spice_name: str = Query(..., description="Spice name, e.g. 'Cumin', 'Turmeric'"),
    anchor_date: str = Query(
        ...,
        description=(
            "Anchor date in YYYY-MM-DD format. "
            "Dataset covers May 1 – Jun 30 2025. "
            "Dates outside this range fall back to the nearest available date automatically."
        ),
    ),
    horizon: str = Query("7d", description="Forecast horizon: 7d, 1m, 3m, 1y, or a custom integer up to 730."),
):
    """
    Simplified forecast endpoint — pass only **spice name, date, and horizon**.

    All rolling context values are looked up automatically from `daily_spice_timeseries.csv`.

    **Date fallback rules:**
    - Exact date found → use it
    - Date is within dataset range but has no shipments that day → use nearest earlier date with data
    - Date is **after** the dataset ends → use the latest available date (with a warning in `context_note`)
    - Date is **before** the dataset starts → return 404
    """
    if _ts_bundle is None:
        raise HTTPException(status_code=503, detail="Time-series model not loaded.")
    if _ts_data is None:
        raise HTTPException(status_code=503, detail="Timeseries dataset not loaded. Ensure daily_spice_timeseries.csv exists.")

    # ── Validate spice ────────────────────────────────────────────────────────
    le = _ts_bundle["label_encoder"]
    spice_title = spice_name.strip().title()
    known_spices = list(le.classes_)
    if spice_title not in known_spices:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown spice '{spice_title}'. Known spices: {known_spices}",
        )

    # ── Validate horizon ──────────────────────────────────────────────────────
    if horizon not in PRESET_HORIZONS:
        try:
            n = int(horizon)
            if n < 1 or n > 730:
                raise ValueError()
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail=f"horizon must be one of {list(PRESET_HORIZONS.keys())} or a number 1–730, got '{horizon}'",
            )

    # ── Validate anchor_date format ───────────────────────────────────────────
    try:
        anchor_ts = pd.Timestamp(anchor_date)
    except Exception:
        raise HTTPException(status_code=422, detail=f"anchor_date must be YYYY-MM-DD, got '{anchor_date}'")

    spice_rows = _ts_data[_ts_data["Spice Name"] == spice_title].sort_values("date")
    if spice_rows.empty:
        raise HTTPException(status_code=404, detail=f"No data found for spice '{spice_title}'.")

    dataset_start = spice_rows.iloc[0]["date"]
    dataset_end   = spice_rows.iloc[-1]["date"]

    # ── Date resolution with clear fallback logic ─────────────────────────────
    context_note: str

    # Case 1: exact date exists
    exact = spice_rows[spice_rows["date"] == anchor_ts]
    if not exact.empty:
        row = exact.iloc[0]
        used_date = anchor_ts
        context_note = f"Exact date {anchor_date} found in dataset."

    # Case 2: date is before the dataset starts — hard error
    elif anchor_ts < dataset_start:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data for '{spice_title}' on or before {anchor_date}. "
                f"Dataset starts at {dataset_start.strftime('%Y-%m-%d')}. "
                "Use a date within the available range."
            ),
        )

    # Case 3: date is after the dataset ends — use latest available date + warn
    elif anchor_ts > dataset_end:
        row = spice_rows.iloc[-1]
        used_date = row["date"]
        context_note = (
            f"Requested date {anchor_date} is beyond the dataset (ends {dataset_end.strftime('%Y-%m-%d')}). "
            f"Using latest available data from {used_date.strftime('%Y-%m-%d')} as context. "
            "Forecast reflects market conditions as of that date — accuracy degrades for dates far from the training window."
        )
        logger.warning(f"lookup: {spice_title} {anchor_date} beyond dataset end, using {used_date.date()}")

    # Case 4: within range but no shipments that exact day — nearest earlier date
    else:
        earlier = spice_rows[spice_rows["date"] <= anchor_ts]
        row = earlier.iloc[-1]
        used_date = row["date"]
        context_note = (
            f"No shipment data for '{spice_title}' on {anchor_date}. "
            f"Using nearest earlier date with data: {used_date.strftime('%Y-%m-%d')}."
        )
        logger.info(f"lookup: {spice_title} {anchor_date} has no data, using {used_date.date()}")

    # ── Build ForecastRangeRequest from the resolved row ─────────────────────
    def _safe(col, default=0.0):
        val = row.get(col, default)
        return default if pd.isna(val) else val

    range_req = ForecastRangeRequest(
        spice_name=spice_title,
        start_date=used_date.strftime("%Y-%m-%d"),
        lag1_price=float(_safe("lag1_price", row["price_per_kg_inr_vwap"])),
        lag7_price=float(_safe("lag7_price")),
        rolling7_avg_price=float(_safe("rolling7_avg_price", row["price_per_kg_inr_vwap"])),
        rolling7_price_std=float(_safe("rolling7_price_std", 0.0)),
        rolling7_volume_kg=float(_safe("rolling7_volume_kg", 0.0)),
        rolling14_volume_kg=float(_safe("rolling14_volume_kg", 0.0)),
        daily_volume_kg=float(_safe("daily_volume_kg", 0.0)),
        daily_shipment_count=int(_safe("daily_shipment_count", 1)),
        daily_buyer_count=int(_safe("daily_buyer_count", 1)),
        daily_exporter_count=int(_safe("daily_exporter_count", 1)),
        horizon=horizon,
    )

    # ── Run forecast and patch context fields ─────────────────────────────────
    result = forecast_range(range_req)
    result.anchor_date = anchor_date          # keep the originally requested date
    result.context_date_used = used_date.strftime("%Y-%m-%d")
    result.context_note = context_note
    return result


@app.get("/model/ts-info", response_model=TSModelInfoResponse, tags=["Model"])
def ts_model_info():
    """Return metadata for the time-series forecast model."""
    if _ts_metadata is None:
        raise HTTPException(status_code=503, detail="Time-series model metadata not available")
    return TSModelInfoResponse(
        model_type=_ts_metadata.get("model_type", "LightGBM"),
        target=_ts_metadata.get("target", ""),
        features=_ts_metadata.get("features", []),
        spices=_ts_metadata.get("spices", []),
        train_date_range=_ts_metadata.get("train_date_range", []),
        test_date_range=_ts_metadata.get("test_date_range", []),
        train_rows=_ts_metadata.get("train_rows", 0),
        test_rows=_ts_metadata.get("test_rows", 0),
        best_iteration=_ts_metadata.get("best_iteration", 0),
        train_r2=_ts_metadata.get("train_r2", 0.0),
        test_r2=_ts_metadata.get("test_r2", 0.0),
        train_mape=_ts_metadata.get("train_mape", 0.0),
        test_mape=_ts_metadata.get("test_mape", 0.0),
        test_smape=_ts_metadata.get("test_smape", 0.0),
        naive_baseline_mape=_ts_metadata.get("naive_baseline_mape", 0.0),
        r2_overfit_gap=_ts_metadata.get("r2_overfit_gap", 0.0),
        per_spice_test_mape=_ts_metadata.get("per_spice_test_mape", {}),
    )
