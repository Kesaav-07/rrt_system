"""
patient_detail.py
-----------------
Single-patient deep-dive page for the RRT BiLSTM Dashboard.

Sections (2×2 grid layout):
    1. Current Vital Parameters   — color-coded vital tiles, overdue flag
    2. RRT Risk Score Panel       — gauge + 4h/8h BiLSTM forecast cards
    3. Patient Information        — role-based field visibility
    4. Vital Trends               — Plotly time-series with normal-range bands

All RRT scoring: ai/rrt_calculator.py only.
All forecasting: ai/predict_bilstm.py only.
"""

from __future__ import annotations

import os
import sys
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OVERDUE_MINUTES_BY_WARD,
    AVPU_ENCODING,
    AVPU_DECODING,
    SEQUENCE_LENGTH,
    RRT_MAX_SCORE,
)
from ai.rrt_calculator import calculate_rrt_score, rrt_category_from_score
from realtime import load_patient_history

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def _inject_detail_css() -> None:
    st.markdown("""
    <style>
    :root, .stApp { color-scheme: light !important; }

    /* Main background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    body { background-color: #F4F6F8 !important; }

    /* Text colors */
    div[data-testid="stMarkdownContainer"] p { color: #0B2545 !important; }
    div[data-testid="stCaptionContainer"] p  { color: #5A7184 !important; }
    h1, h2, h3, h4, h5, h6 { color: #0B2545 !important; }
    label[data-testid="stWidgetLabel"] p { color: #0B2545 !important; }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0B2545 !important;
        border: 1px solid #DCE3E8 !important;
    }
    div[data-baseweb="select"] * { color: #0B2545 !important; }
    ul[data-testid="stSelectboxVirtualDropdown"] li,
    div[data-baseweb="popover"] li {
        background-color: #FFFFFF !important;
        color: #0B2545 !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #EAF6EF !important;
        color: #0B2545 !important;
    }
    .stRadio label p { color: #0B2545 !important; }

    /* Panel containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border: 1.5px solid #C3CFD9;
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
        box-shadow: 0 1px 4px rgba(11,37,69,0.07);
        margin-bottom: 0.8rem;
    }

    /* Vital tiles */
    .vt { background:#FFFFFF; border:1px solid #DCE3E8; border-left:4px solid #0F6E68;
           border-radius:8px; padding:0.5rem 0.65rem; margin-bottom:0.5rem; }
    .vt .vl { font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
               letter-spacing:0.06em; text-transform:uppercase; color:#5A7184 !important; }
    .vt .vv { font-family:'IBM Plex Mono',monospace; font-size:1.35rem;
               font-weight:700; line-height:1.25; }
    .vt .vs { font-size:0.65rem; color:#5A7184; }

    /* Info tiles */
    .it { background:#F4F6F8; border:1px solid #DCE3E8; border-radius:8px;
           padding:0.45rem 0.65rem; margin-bottom:0.5rem; }
    .it .il { font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
               letter-spacing:0.06em; text-transform:uppercase; color:#5A7184 !important; }
    .it .iv { font-size:0.92rem; font-weight:600; color:#0B2545 !important; }

    .panel-heading { font-size:1.05rem; font-weight:700; color:#0B2545 !important;
                      margin:0 0 0.5rem 0; }
    hr { border-color:#DCE3E8 !important; }
    div[data-testid="stButton"] button,
    div[data-testid="stButton"] button p {
    color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)


def _panel():
    """Bordered container for one grid cell."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


# ---------------------------------------------------------------------------
# Clinical reference ranges
# ---------------------------------------------------------------------------
NORMAL_RANGES: dict = {
    "heart_rate":       {"low": 60,  "high": 100,  "unit": "bpm"},
    "respiratory_rate": {"low": 12,  "high": 20,   "unit": "br/min"},
    "spo2":             {"low": 95,  "high": 100,  "unit": "%"},
    "systolic_bp":      {"low": 90,  "high": 140,  "unit": "mmHg"},
    "temperature":      {"low": 36.0, "high": 37.5, "unit": "°C"},
}

AVPU_STATUS: dict = {
    "Alert": "stable", "Voice": "warning",
    "Pain": "critical", "Unresponsive": "critical",
}

TIER_COLOR: dict = {
    "Critical": "#D7263D",
    "Warning":  "#E8A33D",
    "Stable":   "#2E8B57",
    "critical": "#D7263D",
    "warning":  "#E8A33D",
    "stable":   "#2E8B57",
}


def _vital_status(value: float | None, low: float, high: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    span = max(high - low, 1)
    margin = span * 0.15
    if low <= value <= high:
        return "stable"
    elif (low - margin) <= value <= (high + margin):
        return "warning"
    return "critical"


def _status_color(status: str) -> str:
    return {
        "stable": "#2E8B57", "warning": "#E8A33D",
        "critical": "#D7263D", "unknown": "#9AA7B0",
    }.get(status, "#9AA7B0")


# ---------------------------------------------------------------------------
# Section 1 — Current Vital Parameters
# ---------------------------------------------------------------------------

def render_vitals_panel(patient_row: pd.Series, last_observed_at: datetime, ward: str) -> None:
    """Render the current vitals panel with color coding and overdue flags."""
    st.markdown('<p class="panel-heading">Current Vital Parameters</p>', unsafe_allow_html=True)

    overdue_limit = OVERDUE_MINUTES_BY_WARD.get(ward, 180)
    now_ref = datetime.now(timezone.utc) if getattr(last_observed_at, "tzinfo", None) else datetime.now()
    minutes_ago = (now_ref - last_observed_at).total_seconds() / 60

    if minutes_ago > overdue_limit:
        st.error(
            f"⏱ Overdue — last recorded {int(minutes_ago)} min ago "
            f"(expected within {overdue_limit} min on {ward})."
        )

    vitals = [
        ("Heart Rate",       patient_row.get("heart_rate",       np.nan), NORMAL_RANGES["heart_rate"]),
        ("Respiratory Rate", patient_row.get("respiratory_rate", np.nan), NORMAL_RANGES["respiratory_rate"]),
        ("SpO₂",             patient_row.get("spo2",             np.nan), NORMAL_RANGES["spo2"]),
        ("Systolic BP",      patient_row.get("systolic_bp",      np.nan), NORMAL_RANGES["systolic_bp"]),
        ("Temperature",      patient_row.get("temperature",      np.nan), NORMAL_RANGES["temperature"]),
    ]

    cols = st.columns(2)
    for i, (label, value, rng) in enumerate(vitals):
        is_nan = value is None or (isinstance(value, float) and np.isnan(value))
        status = "unknown" if is_nan else _vital_status(float(value), rng["low"], rng["high"])
        color  = _status_color(status)
        disp   = "—" if is_nan else f"{value:.1f}" if isinstance(value, float) else value
        ts_str = last_observed_at.strftime("%H:%M")
        with cols[i % 2]:
            st.markdown(f"""
            <div class="vt" style="border-left-color:{color};">
                <div class="vl">{label}</div>
                <div class="vv" style="color:{color};">{disp} <span style="font-size:0.78rem;">{rng['unit']}</span></div>
                <div class="vs">Normal {rng['low']}–{rng['high']} · {ts_str}</div>
            </div>
            """, unsafe_allow_html=True)

    # AVPU
    avpu_val = str(patient_row.get("avpu", "Alert"))
    avpu_status = AVPU_STATUS.get(avpu_val, "unknown")
    avpu_color  = _status_color(avpu_status)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="vt" style="border-left-color:{avpu_color};">
            <div class="vl">AVPU</div>
            <div class="vv" style="color:{avpu_color}; font-size:1.1rem;">{avpu_val}</div>
        </div>
        """, unsafe_allow_html=True)

    # RRT components breakdown
    with c2:
        score = int(patient_row.get("current_rrt_score", 0))
        cat, label = rrt_category_from_score(score)
        color = TIER_COLOR[cat]
        st.markdown(f"""
        <div class="vt" style="border-left-color:{color};">
            <div class="vl">Current RRT Score</div>
            <div class="vv" style="color:{color};">{score} <span style="font-size:0.78rem;">/ {RRT_MAX_SCORE}</span></div>
            <div class="vs">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 2 — RRT Risk Score Panel with BiLSTM forecast
# ---------------------------------------------------------------------------

def render_risk_score_panel(patient_row: pd.Series) -> None:
    """Render the RRT gauge and BiLSTM forecast cards."""
    st.markdown('<p class="panel-heading">Risk Score Forecast (Bi-LSTM)</p>', unsafe_allow_html=True)

    current  = int(patient_row.get("current_rrt_score", 0))
    pred_4hr = int(patient_row.get("predicted_rrt_4hr", 0))
    pred_8hr = int(patient_row.get("predicted_rrt_8hr", 0))

    cat_curr, label_curr = rrt_category_from_score(current)
    gauge_color = TIER_COLOR[cat_curr]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current,
        number={"suffix": f" / {RRT_MAX_SCORE}", "font": {"size": 28, "color": gauge_color}},
        gauge={
            "axis":  {"range": [0, RRT_MAX_SCORE], "tickcolor": "#0B2545"},
            "bar":   {"color": gauge_color},
            "steps": [
                {"range": [0, 5],  "color": "#EAF6EF"},
                {"range": [6, 11], "color": "#FCEAC9"},
                {"range": [12, 18], "color": "#FBDADD"},
            ],
            "threshold": {
                "line":  {"color": "#D7263D", "width": 2},
                "thickness": 0.75,
                "value": 12,
            },
        },
        title={"text": f"Current · {label_curr}", "font": {"size": 13, "color": "#0B2545"}},
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=38, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    def _arrow(curr: int, fut: int) -> tuple[str, str]:
        if fut > curr:   return "▲ Rising",  "#D7263D"
        elif fut < curr: return "▼ Falling", "#2E8B57"
        return "→ Stable",  "#5A7184"

    cat_4, label_4 = rrt_category_from_score(pred_4hr)
    cat_8, label_8 = rrt_category_from_score(pred_8hr)

    col_4, col_8 = st.columns(2)
    with col_4:
        arrow, acolor = _arrow(current, pred_4hr)
        st.markdown(f"""
        <div class="vt" style="border-left-color:{TIER_COLOR[cat_4]};">
            <div class="vl">+4 Hours (BiLSTM)</div>
            <div class="vv" style="color:{TIER_COLOR[cat_4]};">{pred_4hr} <span style="font-size:0.78rem;">/ {RRT_MAX_SCORE}</span></div>
            <div style="font-size:0.72rem; color:{acolor}; font-weight:600;">{arrow}</div>
            <div class="vs">{label_4}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_8:
        arrow, acolor = _arrow(current, pred_8hr)
        st.markdown(f"""
        <div class="vt" style="border-left-color:{TIER_COLOR[cat_8]};">
            <div class="vl">+8 Hours (BiLSTM)</div>
            <div class="vv" style="color:{TIER_COLOR[cat_8]};">{pred_8hr} <span style="font-size:0.78rem;">/ {RRT_MAX_SCORE}</span></div>
            <div style="font-size:0.72rem; color:{acolor}; font-weight:600;">{arrow}</div>
            <div class="vs">{label_8}</div>
        </div>
        """, unsafe_allow_html=True)

    # RRT component breakdown
    avpu_val = str(patient_row.get("avpu", "Alert"))
    _, components, _, _ = calculate_rrt_score(
        rr=float(patient_row.get("respiratory_rate", 16)),
        spo2=float(patient_row.get("spo2", 97)),
        hr=float(patient_row.get("heart_rate", 80)),
        sbp=float(patient_row.get("systolic_bp", 115)),
        temperature=float(patient_row.get("temperature", 36.8)),
        avpu=avpu_val,
    )

    with st.expander("▸ Component Scores", expanded=False):
        comp_cols = st.columns(3)
        for i, (param, sub_score) in enumerate(components.items()):
            with comp_cols[i % 3]:
                sub_color = ["#2E8B57", "#E8A33D", "#D7263D"][min(sub_score, 2)]
                st.markdown(f"""
                <div class="it">
                    <div class="il">{param}</div>
                    <div class="iv" style="color:{sub_color};">{sub_score} / 3</div>
                </div>
                """, unsafe_allow_html=True)

    st.caption(f"BiLSTM model · Updated {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ---------------------------------------------------------------------------
# Section 3 — Patient Information
# ---------------------------------------------------------------------------

def render_patient_info_panel(patient_row: pd.Series, user_role: str) -> None:
    """Render patient information with role-based field visibility."""
    st.markdown('<p class="panel-heading">Patient Information</p>', unsafe_allow_html=True)

    can_view_id = user_role in ("Physician", "Admin", "RRT Team")
    restricted  = "🔒 Restricted"

    def _tile(label: str, value: object) -> str:
        return f"""
        <div class="it">
            <div class="il">{label}</div>
            <div class="iv">{value}</div>
        </div>
        """

    name_disp      = patient_row.get("name", "Not recorded") if can_view_id else restricted
    diag_disp      = patient_row.get("diagnosis", "—")        if can_view_id else restricted
    physician_disp = patient_row.get("attending_physician", "—") if user_role in ("Physician", "Admin") else restricted

    fields = [
        ("Patient ID",          patient_row.get("patient_id",  "—")),
        ("Name",                name_disp),
        ("Age",                 patient_row.get("age",         "—")),
        ("Ward",                patient_row.get("ward",        "—")),
        ("Block",               patient_row.get("block",       "—")),
        ("Diagnosis",           diag_disp),
        ("Attending Physician", physician_disp),
        ("Admission Date",      patient_row.get("admission_date", "—")),
        ("RRT Category",        patient_row.get("rrt_category", "—")),
    ]

    cols = st.columns(2)
    for i, (label, value) in enumerate(fields):
        with cols[i % 2]:
            st.markdown(_tile(label, value), unsafe_allow_html=True)

    if not can_view_id:
        st.caption("🔒 Some fields restricted for your role. Contact Admin for access.")


# ---------------------------------------------------------------------------
# Section 4 — Vital Trends (Plotly)
# ---------------------------------------------------------------------------

def _build_demo_history(patient_row: pd.Series, hours_back: int) -> tuple[pd.DataFrame, bool]:
    """Generate synthetic history for newly registered patients with no real history."""
    rng = np.random.default_rng(abs(hash(str(patient_row.get("patient_id", 0)))) % (2**32))
    n_pts = max(8, hours_back * 2)
    now = datetime.now()
    timestamps = [
        now - timedelta(minutes=(hours_back * 60 / n_pts) * i)
        for i in range(n_pts)
    ][::-1]

    def _walk(center: float, spread: float, lo: float, hi: float) -> np.ndarray:
        steps = rng.normal(0, spread, n_pts).cumsum()
        vals  = center + steps - steps[-1]
        return np.clip(vals, lo, hi)

    df = pd.DataFrame({
        "timestamp":      timestamps,
        "Heart Rate":     _walk(float(patient_row.get("heart_rate", 80)),       3.0, 35,  220),
        "Respiratory Rate": _walk(float(patient_row.get("respiratory_rate", 16)), 1.2, 6,   45),
        "SpO2":           _walk(float(patient_row.get("spo2", 97)),              0.8, 70,  100),
        "Systolic BP":    _walk(float(patient_row.get("systolic_bp", 115)),      4.0, 55,  230),
        "Temperature":    _walk(float(patient_row.get("temperature", 36.8)),     0.1, 34.0, 42.0),
        "RRT Score":      [0.0] * n_pts,
    })
    return df, True


def _load_history_for_chart(patient_row: pd.Series, hours_back: int) -> tuple[pd.DataFrame, bool]:
    """Load real history or return demo data."""
    pid  = str(patient_row.get("patient_id", ""))
    hist = load_patient_history(pid, hours_back)

    if hist.empty:
        return _build_demo_history(patient_row, hours_back)

    # Repair timestamps
    ts_col = "recorded_at" if "recorded_at" in hist.columns else hist.columns[0]
    try:
        hist[ts_col] = pd.to_datetime(hist[ts_col], utc=True, errors="coerce")
        hist = hist.dropna(subset=[ts_col])
        hist[ts_col] = hist[ts_col].dt.tz_localize(None)
    except Exception:
        return _build_demo_history(patient_row, hours_back)

    if len(hist) < 2:
        return _build_demo_history(patient_row, hours_back)

    df = pd.DataFrame({
        "timestamp":        hist[ts_col].values,
        "Heart Rate":       hist.get("heart_rate", pd.Series(dtype=float)).values,
        "Respiratory Rate": hist.get("respiratory_rate", pd.Series(dtype=float)).values,
        "SpO2":             hist.get("spo2", pd.Series(dtype=float)).values,
        "Systolic BP":      hist.get("systolic_bp", pd.Series(dtype=float)).values,
        "Temperature":      hist.get("temperature", pd.Series(dtype=float)).values,
        "RRT Score":        hist.get("current_rrt_score", pd.Series(dtype=float)).values,
    })
    return df.dropna(subset=["timestamp"]), False


def render_vital_trends_panel(patient_row: pd.Series) -> None:
    """Render interactive Plotly vital trend charts."""
    st.markdown('<p class="panel-heading">Vital Trends</p>', unsafe_allow_html=True)

    range_opts = ["Last 4 hours", "Last 8 hours", "Last 24 hours", "Last 72 hours"]
    range_label = st.radio("Time range", range_opts, horizontal=True,
                           label_visibility="collapsed", key="trend_range")
    hours_map   = {"Last 4 hours": 4, "Last 8 hours": 8,
                   "Last 24 hours": 24, "Last 72 hours": 72}
    hours_back  = hours_map[range_label]

    history, is_demo = _load_history_for_chart(patient_row, hours_back)

    if is_demo:
        st.caption("⚠ No live history yet — showing synthetic placeholder data.")
    else:
        st.caption(f"✅ Live feed · {len(history)} readings in this window.")

    vital_options = ["Heart Rate", "Respiratory Rate", "SpO2", "Systolic BP", "Temperature"]
    selected_vital = st.selectbox(
        "Vital sign", vital_options, key="trend_vital_select"
    )

    normal_bands: dict = {
        "Heart Rate":       (60, 100),
        "Respiratory Rate": (12, 20),
        "SpO2":             (95, 100),
        "Systolic BP":      (90, 140),
        "Temperature":      (36.0, 37.5),
    }
    low, high = normal_bands[selected_vital]

    if selected_vital not in history.columns or history.empty:
        st.warning("Insufficient data for this vital sign.")
        return

    y_vals = pd.to_numeric(history[selected_vital], errors="coerce")
    ts     = pd.to_datetime(history["timestamp"], errors="coerce")

    valid  = ts.notna() & y_vals.notna()
    if valid.sum() < 2:
        st.warning("Insufficient valid readings for this time window.")
        return

    ts    = ts[valid]
    y_vals = y_vals[valid]

    # Separate normal / abnormal points
    is_abnormal = (y_vals < low) | (y_vals > high)

    fig = go.Figure()

    # Normal band
    fig.add_hrect(y0=low, y1=high, fillcolor="#2E8B57", opacity=0.08,
                  layer="below", line_width=0,
                  annotation_text="Normal", annotation_position="top right",
                  annotation_font_color="#2E8B57", annotation_font_size=11)

    # Trend line
    fig.add_trace(go.Scatter(
        x=ts, y=y_vals,
        mode="lines",
        line=dict(color="#0F6E68", width=2.5),
        name=selected_vital,
        showlegend=False,
    ))

    # Normal points
    if (~is_abnormal).any():
        fig.add_trace(go.Scatter(
            x=ts[~is_abnormal], y=y_vals[~is_abnormal],
            mode="markers",
            marker=dict(color="#0F6E68", size=6),
            name="Normal",
            hovertemplate=f"{selected_vital}: %{{y:.1f}}<br>%{{x}}<extra></extra>",
        ))

    # Abnormal points
    if is_abnormal.any():
        fig.add_trace(go.Scatter(
            x=ts[is_abnormal], y=y_vals[is_abnormal],
            mode="markers",
            marker=dict(color="#D7263D", size=8, symbol="circle-open", line=dict(width=2)),
            name="Abnormal",
            hovertemplate=f"{selected_vital}: %{{y:.1f}} ⚠<br>%{{x}}<extra></extra>",
        ))

    # Threshold lines
    fig.add_hline(y=low,  line_dash="dot", line_color="#2E8B57", line_width=1, opacity=0.5)
    fig.add_hline(y=high, line_dash="dot", line_color="#2E8B57", line_width=1, opacity=0.5)

    # RRT score overlay (secondary y-axis)
    if "RRT Score" in history.columns:
        rrt_vals = pd.to_numeric(history["RRT Score"][valid], errors="coerce")
        if rrt_vals.notna().any():
            fig.add_trace(go.Scatter(
                x=ts, y=rrt_vals,
                mode="lines",
                line=dict(color="#E8A33D", width=1.5, dash="dash"),
                name="RRT Score",
                yaxis="y2",
                hovertemplate="RRT: %{y}<br>%{x}<extra></extra>",
            ))

    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis=dict(
            title="Time",
            showgrid=True, gridcolor="#DCE3E8",
            tickfont=dict(family="IBM Plex Mono", size=10, color="#0B2545"),
        ),
        yaxis=dict(
            title=selected_vital,
            showgrid=True, gridcolor="#DCE3E8",
            tickfont=dict(family="IBM Plex Mono", size=10, color="#0B2545"),
        ),
        yaxis2=dict(
            title="RRT Score",
            overlaying="y",
            side="right",
            range=[0, 18],
            showgrid=False,
            tickfont=dict(family="IBM Plex Mono", size=10, color="#E8A33D"),
        ),
        legend=dict(
            orientation="h", y=1.1, x=0,
            font=dict(family="IBM Plex Sans", size=11, color="#0B2545"),
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🟢 Green band = normal range · 🔴 Red points = abnormal · "
        "🟡 Dashed = RRT score (right axis)"
    )


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def render_patient_detail_page(
    df_patients: pd.DataFrame,
    patient_id: str,
    on_back,
) -> None:
    """
    Render the full Patient Detail page.

    Args:
        df_patients: Live patient records DataFrame.
        patient_id:  Patient ID string selected from triage board.
        on_back:     Callback invoked when user clicks "Back to Triage Board".
    """
    _inject_detail_css()

    match = df_patients[df_patients["patient_id"].astype(str) == str(patient_id)]
    if match.empty:
        st.error(f"Patient {patient_id} not found in current records.")
        if st.button("← Back to Triage Board"):
            on_back()
        return

    patient_row = match.iloc[0]

    # ---- Header ----
    top_l, top_r = st.columns([3, 1])
    with top_l:
        if st.button("← Back to Triage Board"):
            on_back()

        curr_score = int(patient_row.get("current_rrt_score", 0))
        cat, label = rrt_category_from_score(curr_score)
        hero_color = TIER_COLOR[cat]

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0B2545 0%,#123A63 100%);
                    border-radius:12px; border-left:5px solid {hero_color};
                    padding:1.2rem 1.5rem; margin:0.6rem 0 1rem 0;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
                        letter-spacing:0.14em; text-transform:uppercase; color:#8FD9CE;">
                PATIENT DETAIL
            </div>
            <div style="color:#FFFFFF; font-size:1.5rem; font-weight:700; margin:0.35rem 0 0.1rem 0;">
                {patient_row.get('patient_id','—')}
                · {patient_row.get('name', 'No Name')}
            </div>
            <div style="color:#B9C7DA; font-size:0.88rem;">
                Ward: {patient_row.get('ward','—')} &nbsp;·&nbsp;
                RRT Score: <span style="color:{hero_color}; font-weight:700;">{curr_score} ({label})</span>
                &nbsp;·&nbsp; 4h Forecast: {int(patient_row.get('predicted_rrt_4hr', 0))}
                &nbsp;·&nbsp; 8h Forecast: {int(patient_row.get('predicted_rrt_8hr', 0))}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with top_r:
        logged_role = st.session_state.get("user_role", "Nurse")
        if logged_role == "Admin":
            display_role = st.selectbox(
                "Viewing as (Admin preview)",
                ["Nurse", "Physician", "RRT Team", "Admin"],
                key="detail_role_selector",
            )
        else:
            display_role = logged_role
            st.markdown(f"""
            <div class="it" style="margin-top:1.6rem;">
                <div class="il">Signed in as</div>
                <div class="iv">{st.session_state.get('username','')} · {display_role}</div>
            </div>
            """, unsafe_allow_html=True)

    # Resolve last_observed_at
    raw_ts = patient_row.get("last_recorded_at", None)
    if pd.notna(raw_ts) and raw_ts:
        try:
            last_observed_at = pd.to_datetime(raw_ts, utc=True).to_pydatetime()
        except Exception:
            last_observed_at = datetime.now(timezone.utc)
    else:
        last_observed_at = datetime.now(timezone.utc)

    # ---- 2×2 Grid ----
    row1_l, row1_r = st.columns(2)
    with row1_l:
        with _panel():
            render_vitals_panel(patient_row, last_observed_at, str(patient_row.get("ward", "General")))
    with row1_r:
        with _panel():
            render_risk_score_panel(patient_row)

    row2_l, row2_r = st.columns(2)
    with row2_l:
        with _panel():
            render_patient_info_panel(patient_row, display_role)
    with row2_r:
        with _panel():
            render_vital_trends_panel(patient_row)
