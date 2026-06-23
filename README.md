# RRT BiLSTM Surveillance System

Real-Time Rapid Response Team (RRT) Patient Deterioration Forecasting Dashboard.

## Features

- **18-point RRT Scoring** — Real-time early warning score across 6 vital parameters
- **Bidirectional LSTM Forecasting** — 4-hour and 8-hour deterioration predictions
- **Live Monitoring** — Synthetic vital-sign perturbation with 60-second auto-refresh
- **Role-Based Auth** — Nurse / Physician / RRT Team / Admin with SHA-256 password hashing
- **Priority Queue** — Triage board sorted by predicted 8-hour RRT risk
- **Patient Detail View** — Gauge, forecast cards, Plotly trend charts with normal-range bands
- **Alert System** — Automatic flagging of critical and rapidly deteriorating patients
- **Explainable AI** — Per-parameter sub-score breakdown and clinical narrative

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Train the BiLSTM model
python ai/train_bilstm.py

# 3. Launch the dashboard
streamlit run app.py
```

The system runs in **fallback mode** without a trained model — predictions use
random-walk perturbation of current vitals. Train the model for full accuracy.

## Project Structure

```
rrt_system/
├── app.py                  # Streamlit entry point
├── config.py               # All constants and paths
├── auth.py                 # Auth gate, login/signup, admin panel
├── patient_detail.py       # Single-patient deep-dive page
├── requirements.txt
├── README.md
├── ai/
│   ├── rrt_calculator.py   # 18-point RRT scoring (single source of truth)
│   ├── predict_bilstm.py   # Bi-LSTM inference
│   ├── train_bilstm.py     # Model training
│   ├── explainability.py   # Clinical narrative & feature importance
│   └── model_utils.py      # Model artifact helpers
├── realtime/
│   └── realtime.py         # Synthetic vital engine, connection status, alerts
├── data/
│   ├── live_future_records.csv
│   ├── vital_history.csv
│   ├── realtime_state.json
│   └── trained/            # Saved model & scalers
├── utils/
│   ├── csv_utils.py
│   ├── logger.py
│   ├── validators.py
│   └── helpers.py
└── tests/
    ├── test_rrt.py
    ├── test_bilstm.py
    └── test_realtime.py
```

## RRT Scoring Guide

| Score | Category | Action |
|-------|----------|--------|
| 0–5   | 🟢 Stable   | Routine monitoring |
| 6–11  | 🟠 Warning  | Increased surveillance |
| 12–18 | 🔴 Critical | RRT activation |

## Roles

| Role       | Register Patients | View Details | Admin Panel |
|------------|:-----------------:|:------------:|:-----------:|
| Nurse      | ✅ | ✅ | ❌ |
| Physician  | ❌ | ✅ | ❌ |
| RRT Team   | ✅ | ✅ | ❌ |
| Admin      | ✅ | ✅ | ✅ |

## Running Tests

```bash
pytest tests/ -v
```
