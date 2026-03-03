"""
Smoke tests for Turmerix API v2 (leakage-free model) + time-series forecast.
Run after starting: uvicorn api.main:app --reload --port 8000
"""

import sys
import json
import httpx

BASE = "http://localhost:8000"


def check(label, r):
    status = "PASS" if r.status_code < 400 else "FAIL"
    print(f"[{status}] {label} — HTTP {r.status_code}")
    if r.status_code >= 400:
        print(f"       ERROR: {r.text}")
    else:
        try:
            data = r.json()
            print(f"       {json.dumps(data, indent=2)[:400]}")
        except Exception:
            pass
    return r.status_code < 400


# ── v2 payloads: NO fob_inr / item_rate (leakage-free) ───────────────────────
CUMIN_ZA = {
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
    "week_of_year": 18,
}

TURMERIC_UAE = {
    "spice_name": "Turmeric",
    "quantity": 100.0,
    "uqc": "KGS",
    "currency": "USD",
    "mode_of_transport": "SEA",
    "destination_country": "UNITED ARAB EMIRATES",
    "exporter_state": "Uttar Pradesh",
    "port": "JNPT - NHAVA SHEVA SEA PORT",
    "day_of_month": 27,
    "month_num": 6,
    "day_of_week": 4,
    "week_of_year": 26,
}

GINGER_BD = {
    "spice_name": "Ginger",
    "quantity": 16000.0,
    "uqc": "KGS",
    "currency": "USD",
    "mode_of_transport": "LAND",
    "destination_country": "BANGLADESH",
    "exporter_state": "West Bengal",
    "port": "HILLI LCS",
    "day_of_month": 3,
    "month_num": 5,
    "day_of_week": 5,
    "week_of_year": 18,
}

CARDAMOM_UK = {
    "spice_name": "Cardamom",
    "quantity": 0.5,
    "uqc": "MTS",
    "currency": "GBP",
    "mode_of_transport": "AIR",
    "destination_country": "UNITED KINGDOM",
    "exporter_state": "Kerala",
    "port": "COCHIN AIR CARGO",
    "day_of_month": 15,
    "month_num": 6,
    "day_of_week": 6,
    "week_of_year": 24,
    "log_exporter_volume": 6.2,  # known large exporter
}

# ── Time-series forecast payloads ─────────────────────────────────────────────
CUMIN_FORECAST = {
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
    "week_of_year": 24,
}

CARDAMOM_FORECAST = {
    "spice_name": "Cardamom",
    "lag1_price": 2400.0,
    "rolling7_avg_price": 2380.0,
    "rolling7_price_std": 85.0,
    "lag7_price": 2350.0,
    "rolling7_volume_kg": 35000.0,
    "rolling14_volume_kg": 68000.0,
    "daily_volume_kg": 5000.0,
    "daily_shipment_count": 8,
    "daily_buyer_count": 4,
    "daily_exporter_count": 5,
    "day_of_week": 3,
    "month": 6,
    "week_of_year": 25,
}

UNKNOWN_SPICE_FORECAST = {**CUMIN_FORECAST, "spice_name": "Unicorn Spice"}

# ── /forecast/range payloads ───────────────────────────────────────────────────
CUMIN_RANGE_BASE = {
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
}

TURMERIC_RANGE_BASE = {
    "spice_name": "Turmeric",
    "start_date": "2025-06-20",
    "lag1_price": 155.0,
    "rolling7_avg_price": 152.0,
    "rolling7_price_std": 8.0,
    "rolling7_volume_kg": 180000.0,
    "rolling14_volume_kg": 350000.0,
    "daily_volume_kg": 26000.0,
}


def main():
    client = httpx.Client(timeout=30)
    passed = 0
    total = 0

    # 1. Health
    total += 1
    r = client.get(f"{BASE}/health")
    if check("GET /health", r):
        passed += 1

    # 2. Model info — check v2 fields present
    total += 1
    r = client.get(f"{BASE}/model/info")
    if check("GET /model/info", r):
        data = r.json()
        assert "version" in data, "Missing 'version' field"
        assert "r2_overfit_gap" in data, "Missing 'r2_overfit_gap' field"
        assert "leaky_features_excluded" in data, "Missing 'leaky_features_excluded' field"
        assert "cv_r2_mean" in data, "Missing 'cv_r2_mean' field"
        assert len(data["leaky_features_excluded"]) > 0, "Leaky features list should not be empty"
        print(f"       ✅ train_r2={data['train_r2']:.4f} test_r2={data['test_r2']:.4f}"
              f" gap={data['r2_overfit_gap']:+.4f} cv_r2={data['cv_r2_mean']:.4f}")
        passed += 1

    # 3. Spice list
    total += 1
    r = client.get(f"{BASE}/model/spices")
    if check("GET /model/spices", r):
        passed += 1

    # 4. Countries list
    total += 1
    r = client.get(f"{BASE}/model/countries")
    if check("GET /model/countries", r):
        passed += 1

    # 5. Cumin → South Africa (bulk shipment)
    total += 1
    r = client.post(f"{BASE}/predict", json=CUMIN_ZA)
    if check("POST /predict (Cumin → South Africa)", r):
        pred = r.json()["predicted_unit_rate_inr"]
        print(f"       → Predicted: ₹{pred:,.2f}/unit")
        passed += 1

    # 6. Turmeric → UAE
    total += 1
    r = client.post(f"{BASE}/predict", json=TURMERIC_UAE)
    if check("POST /predict (Turmeric → UAE)", r):
        pred = r.json()["predicted_unit_rate_inr"]
        print(f"       → Predicted: ₹{pred:,.2f}/unit")
        passed += 1

    # 7. Ginger → Bangladesh (land route, large qty)
    total += 1
    r = client.post(f"{BASE}/predict", json=GINGER_BD)
    if check("POST /predict (Ginger → Bangladesh, LAND)", r):
        pred = r.json()["predicted_unit_rate_inr"]
        print(f"       → Predicted: ₹{pred:,.2f}/unit")
        passed += 1

    # 8. Cardamom → UK (AIR, with explicit exporter volume)
    total += 1
    r = client.post(f"{BASE}/predict", json=CARDAMOM_UK)
    if check("POST /predict (Cardamom → UK, AIR)", r):
        pred = r.json()["predicted_unit_rate_inr"]
        print(f"       → Predicted: ₹{pred:,.2f}/unit")
        passed += 1

    # 9. Batch prediction (4 items)
    total += 1
    batch_payload = {"requests": [CUMIN_ZA, TURMERIC_UAE, GINGER_BD, CARDAMOM_UK]}
    r = client.post(f"{BASE}/predict/batch", json=batch_payload)
    if check("POST /predict/batch (4 items)", r):
        data = r.json()
        assert data["count"] == 4, f"Expected 4 predictions, got {data['count']}"
        passed += 1

    # 10. GET quick endpoint (no FOB needed)
    total += 1
    r = client.get(
        f"{BASE}/predict/spice",
        params={
            "spice_name": "Coriander",
            "quantity": 0.75,
            "uqc": "MTS",
            "destination_country": "SOUTH AFRICA",
            "currency": "USD",
            "month_num": 5,
            "day_of_month": 2,
        },
    )
    if check("GET /predict/spice (Coriander)", r):
        pred = r.json()["predicted_unit_rate_inr"]
        print(f"       → Predicted: ₹{pred:,.2f}/unit")
        passed += 1

    # 11. Validation: empty spice_name should return 422
    total += 1
    bad_payload = {**CUMIN_ZA, "spice_name": ""}
    r = client.post(f"{BASE}/predict", json=bad_payload)
    if r.status_code == 422:
        print(f"[PASS] Validation (empty spice_name → 422)")
        passed += 1
    else:
        print(f"[FAIL] Expected 422 for empty spice_name, got {r.status_code}")

    # ── Time-series forecast tests ────────────────────────────────────────────

    # 12. TS model info
    total += 1
    r = client.get(f"{BASE}/model/ts-info")
    if check("GET /model/ts-info", r):
        data = r.json()
        assert "test_mape" in data, "Missing 'test_mape' field"
        assert "per_spice_test_mape" in data, "Missing 'per_spice_test_mape' field"
        assert len(data["spices"]) > 0, "Spices list should not be empty"
        print(f"       ✅ test_r2={data['test_r2']:.3f} test_mape={data['test_mape']:.1f}%"
              f" naive_mape={data['naive_baseline_mape']:.1f}%")
        passed += 1

    # 13. Cumin forecast (fully specified)
    total += 1
    r = client.post(f"{BASE}/forecast", json=CUMIN_FORECAST)
    if check("POST /forecast (Cumin, full context)", r):
        data = r.json()
        pred = data["predicted_next_day_vwap_inr"]
        chg  = data["price_change_pct"]
        mape = data["model_test_mape_pct"]
        print(f"       → Predicted VWAP: ₹{pred:,.2f}/kg  ({chg:+.1f}% vs lag1)  spice MAPE={mape:.1f}%")
        assert pred > 0, "Predicted VWAP must be positive"
        passed += 1

    # 14. Cardamom forecast (optional fields omitted)
    total += 1
    r = client.post(f"{BASE}/forecast", json=CARDAMOM_FORECAST)
    if check("POST /forecast (Cardamom, optional fields omitted)", r):
        pred = r.json()["predicted_next_day_vwap_inr"]
        print(f"       → Predicted VWAP: ₹{pred:,.2f}/kg")
        assert pred > 0, "Predicted VWAP must be positive"
        passed += 1

    # 15. Forecast: unknown spice → 422
    total += 1
    r = client.post(f"{BASE}/forecast", json=UNKNOWN_SPICE_FORECAST)
    if r.status_code == 422:
        print(f"[PASS] Forecast validation (unknown spice → 422)")
        passed += 1
    else:
        print(f"[FAIL] Expected 422 for unknown spice, got {r.status_code}: {r.text[:200]}")

    # 16. Forecast: lag1_price ≤ 0 → 422
    total += 1
    bad_forecast = {**CUMIN_FORECAST, "lag1_price": -1.0}
    r = client.post(f"{BASE}/forecast", json=bad_forecast)
    if r.status_code == 422:
        print(f"[PASS] Forecast validation (lag1_price <= 0 → 422)")
        passed += 1
    else:
        print(f"[FAIL] Expected 422 for lag1_price=-1, got {r.status_code}")

    # ── /forecast/range tests ─────────────────────────────────────────────────

    # 17. Cumin — 7-day range (preset)
    total += 1
    r = client.post(f"{BASE}/forecast/range", json={**CUMIN_RANGE_BASE, "horizon": "7d"})
    if check("POST /forecast/range (Cumin, 7d)", r):
        data = r.json()
        assert data["horizon_days"] == 7, "Expected 7 horizon days"
        assert len(data["forecast"]) == 7, "Expected 7 daily entries"
        s = data["summary"]
        print(f"       → avg ₹{s['avg_price_inr']:,.2f}  trend={s['trend']}  signal={s['signal']}")
        assert s["signal"] in {"BUY / HOLD STOCK", "SELL / LIQUIDATE", "HOLD"}
        passed += 1

    # 18. Turmeric — 30-day range (1m preset)
    total += 1
    r = client.post(f"{BASE}/forecast/range", json={**TURMERIC_RANGE_BASE, "horizon": "1m"})
    if check("POST /forecast/range (Turmeric, 1m)", r):
        data = r.json()
        assert data["horizon_days"] == 30
        assert len(data["forecast"]) == 30
        s = data["summary"]
        print(f"       → avg ₹{s['avg_price_inr']:,.2f}  ({s['total_pct_change']:+.1f}%)  signal={s['signal']}")
        passed += 1

    # 19. Cumin — 3m preset (90 days)
    total += 1
    r = client.post(f"{BASE}/forecast/range", json={**CUMIN_RANGE_BASE, "horizon": "3m"})
    if check("POST /forecast/range (Cumin, 3m)", r):
        data = r.json()
        assert data["horizon_days"] == 90
        assert len(data["forecast"]) == 90
        assert "reasoning" in data["summary"]
        passed += 1

    # 20. Cumin — custom 45-day horizon
    total += 1
    r = client.post(f"{BASE}/forecast/range", json={**CUMIN_RANGE_BASE, "horizon": "45"})
    if check("POST /forecast/range (Cumin, custom 45d)", r):
        data = r.json()
        assert data["horizon_days"] == 45
        assert len(data["forecast"]) == 45
        passed += 1

    # 21. Range validation: invalid horizon → 422
    total += 1
    r = client.post(f"{BASE}/forecast/range", json={**CUMIN_RANGE_BASE, "horizon": "999d"})
    if r.status_code == 422:
        print(f"[PASS] Range validation (bad horizon → 422)")
        passed += 1
    else:
        print(f"[FAIL] Expected 422 for bad horizon, got {r.status_code}: {r.text[:200]}")

    # ── /forecast/range/lookup tests ──────────────────────────────────────────

    # 22. Cumin on exact known date — 7d
    total += 1
    r = client.get(f"{BASE}/forecast/range/lookup",
                   params={"spice_name": "Cumin", "anchor_date": "2025-06-30", "horizon": "7d"})
    if check("GET /forecast/range/lookup (Cumin, exact date, 7d)", r):
        data = r.json()
        assert data["anchor_date"] == "2025-06-30"
        assert data["context_date_used"] == "2025-06-30", "context_date_used should match exact date"
        assert "Exact date" in data["context_note"], "context_note should confirm exact match"
        assert data["horizon_days"] == 7
        assert len(data["forecast"]) == 7
        s = data["summary"]
        print(f"       → anchor ₹{data['anchor_price_inr']:,.2f}  signal={s['signal']}  trend={s['trend']}")
        print(f"       → context_note: {data['context_note']}")
        passed += 1

    # 23. Turmeric — within-range fallback (no shipments on that specific day)
    total += 1
    r = client.get(f"{BASE}/forecast/range/lookup",
                   params={"spice_name": "Turmeric", "anchor_date": "2025-06-25", "horizon": "1m"})
    if check("GET /forecast/range/lookup (Turmeric, within-range fallback, 1m)", r):
        data = r.json()
        assert data["anchor_date"] == "2025-06-25", "anchor_date should be the requested date"
        assert data["horizon_days"] == 30
        assert len(data["forecast"]) == 30
        assert "context_date_used" in data
        print(f"       → requested=2025-06-25  used={data['context_date_used']}  signal={data['summary']['signal']}")
        print(f"       → context_note: {data['context_note']}")
        passed += 1

    # 24. Cardamom — 3m horizon
    total += 1
    r = client.get(f"{BASE}/forecast/range/lookup",
                   params={"spice_name": "Cardamom", "anchor_date": "2025-06-30", "horizon": "3m"})
    if check("GET /forecast/range/lookup (Cardamom, 3m)", r):
        data = r.json()
        assert data["horizon_days"] == 90
        assert "reasoning" in data["summary"]
        print(f"       → avg ₹{data['summary']['avg_price_inr']:,.2f}  ({data['summary']['total_pct_change']:+.1f}%)")
        passed += 1

    # 25. Future date (beyond dataset) — should use latest available date + warn
    total += 1
    r = client.get(f"{BASE}/forecast/range/lookup",
                   params={"spice_name": "Cumin", "anchor_date": "2026-03-01", "horizon": "7d"})
    if check("GET /forecast/range/lookup (Cumin, future date → latest fallback)", r):
        data = r.json()
        assert data["anchor_date"] == "2026-03-01", "anchor_date should reflect what was requested"
        assert data["context_date_used"] != "2026-03-01", "context_date_used should be last known date"
        assert "beyond" in data["context_note"].lower() or "latest" in data["context_note"].lower()
        print(f"       → requested=2026-03-01  used={data['context_date_used']}")
        print(f"       → context_note: {data['context_note'][:100]}...")
        passed += 1

    # 26. Date before dataset start → 404
    total += 1
    r = client.get(f"{BASE}/forecast/range/lookup",
                   params={"spice_name": "Cumin", "anchor_date": "2020-01-01", "horizon": "7d"})
    if r.status_code == 404:
        print(f"[PASS] Lookup (date before dataset start → 404): {r.json()['detail'][:80]}")
        passed += 1
    else:
        print(f"[FAIL] Expected 404 for pre-dataset date, got {r.status_code}: {r.text[:200]}")

    # 27. Lookup validation: unknown spice → 422
    total += 1
    r = client.get(f"{BASE}/forecast/range/lookup",
                   params={"spice_name": "Unicorn Spice", "anchor_date": "2025-06-30", "horizon": "7d"})
    if r.status_code == 422:
        print(f"[PASS] Lookup validation (unknown spice → 422)")
        passed += 1
    else:
        print(f"[FAIL] Expected 422 for unknown spice in lookup, got {r.status_code}")

    print(f"\n{'='*55}")
    print(f"Results: {passed}/{total} tests passed")
    if passed < total:
        print("⚠ Some tests failed — check API logs")
    else:
        print("✅ All tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
