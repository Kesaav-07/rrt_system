"""
app.py
------
Main entry point for the RRT BiLSTM Surveillance Dashboard.

Features:
    - Role-based authentication (Nurse, Physician, RRT Team, Admin)
    - Real-time patient monitoring with 60-second auto-refresh
    - 18-point RRT scoring (current, 4h forecast, 8h forecast)
    - Bi-LSTM powered deterioration forecasting
    - Patient registration (Nurses)
    - Priority queue sorted by predicted 8h RRT
    - Alert system for critical patients
    - Patient detail deep-dive page
    - Admin user management panel

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import logging
from datetime import datetime


import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR,
    LIVE_RECORDS_FILE,
    REFRESH_INTERVAL_SECONDS,
    AVPU_OPTIONS,
    AVPU_ENCODING,
    PATIENT_ID_PREFIX,
    PATIENT_ID_START,
)
from auth import render_auth_gate, render_logout_control, render_admin_panel
from realtime import run_synthetic_tick_if_due, connection_status, STATUS_DISPLAY, get_alerts
from patient_detail import render_patient_detail_page
from ai.rrt_calculator import calculate_rrt_score, rrt_category_from_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="RRT BiLSTM Surveillance Dashboard",
    page_icon="🩺",
)

# ---------------------------------------------------------------------------
# Authentication gate — halts until logged in
# ---------------------------------------------------------------------------
render_auth_gate()

# ---------------------------------------------------------------------------
# Auto-refresh (60 seconds)
# ---------------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    AUTOREFRESH_AVAILABLE = True
    refresh_count = st_autorefresh(
        interval=REFRESH_INTERVAL_SECONDS * 1000, key="data_autorefresh"
    )
except ImportError:
    AUTOREFRESH_AVAILABLE = False
    refresh_count = None

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "view" not in st.session_state:
    st.session_state["view"] = "dashboard"
if "selected_patient" not in st.session_state:
    st.session_state["selected_patient"] = None

# ---------------------------------------------------------------------------
# Global CSS — clinical "bedside monitor" theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

:root {
    --ink: #0B2545;
    --teal: #0F6E68;
    --teal-dark: #0A4F4A;
    --bg: #F4F6F8;
    --card: #FFFFFF;
    --border: #DCE3E8;
    --critical: #D7263D;
    --warning: #E8A33D;
    --stable: #2E8B57;
}

.stApp { font-family: 'IBM Plex Sans', sans-serif; background-color: var(--bg); }
h1, h2, h3, h4 { font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); letter-spacing:-0.01em; }
label[data-testid="stWidgetLabel"] p { font-weight:600; font-size:0.82rem; color:var(--ink); text-transform:uppercase; letter-spacing:0.03em; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color:#0B2545; border-right:1px solid #0A1E3A; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color:#FFFFFF; }
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p { color:#B9C7DA; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color:#DCE6F0; }
div[data-testid="stForm"] { background-color:#112F58; border:1px solid #1C4373; border-radius:10px; padding:1.1rem 1.2rem 0.4rem; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color:#FFFFFF !important; color:var(--ink) !important;
    border-radius:6px !important; border:none !important;
}

/* Buttons */
.stButton button, button[kind="formSubmit"], button[kind="secondaryFormSubmit"] {
    background-color:var(--teal) !important; color:#FFFFFF !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; letter-spacing:0.02em; width:100%;
}
.stButton button:hover { background-color:var(--teal-dark) !important; }

/* Hero banner */
.hero-banner {
    background:linear-gradient(135deg,#0B2545 0%,#123A63 100%);
    border-radius:12px; border-left:5px solid var(--teal);
    padding:1.5rem 1.8rem; margin-bottom:1.3rem;
}
.hero-kicker { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase; color:#8FD9CE; }
.live-dot { display:inline-block; width:8px; height:8px; background-color:#3FE0A6; border-radius:50%; margin-right:8px; animation:pulse 1.8s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(63,224,166,.55)} 70%{box-shadow:0 0 0 8px rgba(63,224,166,0)} 100%{box-shadow:0 0 0 0 rgba(63,224,166,0)} }
.hero-title { color:#FFFFFF; font-size:1.65rem; font-weight:700; margin:0.45rem 0 0.15rem; }
.hero-subtitle { color:#B9C7DA; font-size:0.92rem; margin:0; }

/* KPI tiles */
.vitals-strip { display:grid; grid-template-columns:repeat(5,1fr); gap:0.9rem; margin-bottom:1.5rem; }
.vital-tile { background-color:var(--card); border:1px solid var(--border); border-left:4px solid var(--teal); border-radius:10px; padding:0.85rem 1rem; }
.vital-tile.critical { border-left-color:var(--critical); }
.vital-tile.warning  { border-left-color:var(--warning); }
.vital-tile.stable   { border-left-color:var(--stable); }
.vital-tile.teal     { border-left-color:var(--teal); }
.vital-label { font-family:'IBM Plex Mono',monospace; font-size:0.66rem; letter-spacing:0.08em; text-transform:uppercase; color:#5A7184; }
.vital-value { font-family:'IBM Plex Mono',monospace; font-size:2.05rem; font-weight:700; color:var(--ink); line-height:1.25; }
.vital-tile.critical .vital-value { color:var(--critical); }
.vital-tile.warning  .vital-value { color:#B97818; }
.vital-tile.stable   .vital-value { color:var(--stable); }

/* Dataframe */
div[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:10px; overflow:hidden; box-shadow:0 1px 3px rgba(11,37,69,.06); }

.section-label { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--teal-dark); margin-bottom:-0.4rem; }
hr { border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def _load_live_records() -> pd.DataFrame:
    """Load live patient records from CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LIVE_RECORDS_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(LIVE_RECORDS_FILE)
        df["patient_id"] = df["patient_id"].astype(str)
        return df
    except Exception as exc:
        logger.warning(f"Failed to load live records: {exc}")
        return pd.DataFrame()


def _save_live_records(df: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(LIVE_RECORDS_FILE, index=False)
    st.cache_data.clear()


def _next_patient_id(df: pd.DataFrame) -> str:
    if df.empty:
        return f"{PATIENT_ID_PREFIX}{PATIENT_ID_START}"
    nums = pd.to_numeric(
        df["patient_id"].astype(str).str.replace(PATIENT_ID_PREFIX, "", case=False),
        errors="coerce",
    )
    valid = nums[nums >= PATIENT_ID_START]
    next_num = int(valid.max() + 1) if not valid.empty else PATIENT_ID_START
    return f"{PATIENT_ID_PREFIX}{next_num}"


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _rrt_highlight(val: int) -> str:
    """Cell style for current RRT score."""
    try:
        v = int(val)
    except (TypeError, ValueError):
        return ""
    if v >= 12:  return "background-color:#D7263D; color:#FFFFFF; font-weight:700;"
    elif v >= 6: return "background-color:#F3B23B; color:#2A1B00; font-weight:600;"
    elif v >= 0: return "background-color:#E3F3E9; color:#1F6E45;"
    return ""


def _rrt_highlight_pred(val: int) -> str:
    """Cell style for predicted RRT scores (lighter)."""
    try:
        v = int(val)
    except (TypeError, ValueError):
        return ""
    if v >= 12:  return "background-color:#FBDADD; color:#9A1228; font-weight:600;"
    elif v >= 6: return "background-color:#FCEAC9; color:#7A4E0E; font-weight:600;"
    elif v >= 0: return "background-color:#EAF6EF; color:#2A7350;"
    return ""


# ---------------------------------------------------------------------------
# Patient registration form (Nurses)
# ---------------------------------------------------------------------------

def _render_patient_registration(df: pd.DataFrame) -> pd.DataFrame:
    """Render the sidebar patient registration form. Returns updated DataFrame."""
    next_id = _next_patient_id(df)
    current_role = st.session_state.get("user_role", "Nurse")

    st.sidebar.markdown('<p class="section-label" style="color:#8FD9CE;">PATIENT REGISTRATION</p>', unsafe_allow_html=True)
    st.sidebar.subheader("Register New Patient")

    if current_role == "Physician":
        st.sidebar.info(
            "Physicians have view-only access. "
            "Log in as Nurse, RRT Team, or Admin to register patients."
        )
        return df

    with st.sidebar.form("patient_intake_form", clear_on_submit=True):
        # Demographics
        p_id      = st.text_input("Patient ID", value=next_id, disabled=True)
        p_name    = st.text_input("Patient Name", placeholder="Full name")
        p_age     = st.number_input("Age", min_value=0, max_value=120, value=50)
        p_ward    = st.selectbox("Ward", ["General", "Special", "ICU"])
        p_block   = st.selectbox("Block", ["A", "B", "C"])
        p_diag    = st.text_input("Diagnosis", placeholder="Primary diagnosis")

        st.markdown("---")
        st.markdown("**Vital Signs**")

        # Vitals
        p_rr   = st.number_input("Respiratory Rate (br/min)", min_value=4, max_value=60, value=18)
        p_spo2 = st.number_input("SpO₂ (%)",                  min_value=50, max_value=100, value=97)
        p_hr   = st.number_input("Heart Rate (bpm)",           min_value=20, max_value=250, value=80)
        p_sbp  = st.number_input("Systolic BP (mmHg)",         min_value=40, max_value=250, value=120)
        p_temp = st.number_input("Temperature (°C)",            min_value=30.0, max_value=43.0, value=36.8, step=0.1)
        p_avpu = st.selectbox("AVPU", AVPU_OPTIONS)

        submitted = st.form_submit_button("✅ Register Patient", use_container_width=True)

    if submitted:
        # Compute current RRT
        curr_score, _, curr_cat, curr_label = calculate_rrt_score(
            rr=p_rr, spo2=p_spo2, hr=p_hr, sbp=p_sbp,
            temperature=p_temp, avpu=p_avpu,
        )

        # Get BiLSTM predictions for new patient
        try:
            from ai.predict_bilstm import predict_from_current_vitals as _predict
            pred_result = _predict({
                "RR": p_rr, "SpO2": p_spo2, "HR": p_hr,
                "SBP": p_sbp, "Temperature": p_temp,
                "AVPU": AVPU_ENCODING.get(p_avpu, 0),
            })
            pred_4hr = int(pred_result["t4_rrt"]["score"])
            pred_8hr = int(pred_result["t8_rrt"]["score"])
        except Exception:
            pred_4hr = curr_score
            pred_8hr = curr_score

        new_row = pd.DataFrame([{
            "patient_id":         p_id,
            "name":               p_name,
            "age":                p_age,
            "ward":               p_ward,
            "block":              p_block,
            "diagnosis":          p_diag,
            "respiratory_rate":   p_rr,
            "spo2":               p_spo2,
            "heart_rate":         p_hr,
            "systolic_bp":        p_sbp,
            "temperature":        p_temp,
            "avpu":               p_avpu,
            "avpu_encoded":       AVPU_ENCODING.get(p_avpu, 0),
            "current_rrt_score":  curr_score,
            "rrt_category":       curr_cat,
            "predicted_rrt_4hr":  pred_4hr,
            "predicted_rrt_8hr":  pred_8hr,
            "last_recorded_at":   datetime.now().isoformat(),
            "sequence":           0,
        }])

        df_updated = pd.concat([df, new_row], ignore_index=True)
        _save_live_records(df_updated)
        st.sidebar.success(f"✅ Patient {p_id} registered! RRT={curr_score} ({curr_label})")
        return df_updated

    return df


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def _render_dashboard(df: pd.DataFrame) -> None:
    """Render the main triage dashboard."""
    total_n    = len(df)
    critical_n = int((df["current_rrt_score"].fillna(0) >= 12).sum()) if total_n else 0
    warning_n  = int(((df["current_rrt_score"].fillna(0) >= 6) & (df["current_rrt_score"].fillna(0) < 12)).sum()) if total_n else 0
    stable_n   = int((df["current_rrt_score"].fillna(0) <= 5).sum()) if total_n else 0
    pred_crit  = int((df.get("predicted_rrt_8hr", pd.Series(dtype=int)).fillna(0) >= 12).sum()) if total_n else 0

    # KPI strip
    st.markdown(f"""
    <div class="vitals-strip">
        <div class="vital-tile teal">
            <div class="vital-label">Total Patients</div>
            <div class="vital-value">{total_n}</div>
        </div>
        <div class="vital-tile stable">
            <div class="vital-label">Stable · RRT 0–5</div>
            <div class="vital-value">{stable_n}</div>
        </div>
        <div class="vital-tile warning">
            <div class="vital-label">Warning · RRT 6–11</div>
            <div class="vital-value">{warning_n}</div>
        </div>
        <div class="vital-tile critical">
            <div class="vital-label">Critical · RRT 12–18</div>
            <div class="vital-value">{critical_n}</div>
        </div>
        <div class="vital-tile critical">
            <div class="vital-label">Pred. Critical in 8h</div>
            <div class="vital-value">{pred_crit}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active alerts banner
    alerts_df = get_alerts(df)
    if not alerts_df.empty:
        alert_ids = ", ".join(alerts_df["patient_id"].astype(str).tolist()[:10])
        if len(alerts_df) > 10:
            alert_ids += f" … (+{len(alerts_df) - 10} more)"
        st.error(
            f"🚨 **RRT ALERT** — {len(alerts_df)} patient(s) require immediate attention: {alert_ids}"
        )

    # Filters
    st.markdown('<p class="section-label">WARD CENSUS</p>', unsafe_allow_html=True)
    st.subheader("Priority Queue — Live Triage Board")

    if df.empty:
        st.info("No patients registered yet. Use the sidebar form to register the first patient.")
        return

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        all_blocks  = sorted(df["block"].dropna().unique().tolist()) if "block" in df else ["A", "B", "C"]
        s_block = st.multiselect("Filter Block", options=all_blocks, default=all_blocks)
    with fc2:
        all_wards = sorted(df["ward"].dropna().unique().tolist()) if "ward" in df else ["General", "Special", "ICU"]
        s_ward = st.multiselect("Filter Ward", options=all_wards, default=all_wards)
    with fc3:
        search = st.text_input("🔍 Search Patient ID / Name").strip().lower()

    disp = df.copy()
    if "block" in disp.columns and s_block:
        disp = disp[disp["block"].isin(s_block)]
    if "ward" in disp.columns and s_ward:
        disp = disp[disp["ward"].isin(s_ward)]
    if search:
        mask = disp["patient_id"].astype(str).str.lower().str.contains(search, na=False)
        if "name" in disp.columns:
            mask |= disp["name"].astype(str).str.lower().str.contains(search, na=False)
        disp = disp[mask]

    # Sort by predicted 8h RRT (priority queue — highest risk first)
    sort_col = "predicted_rrt_8hr" if "predicted_rrt_8hr" in disp.columns else "current_rrt_score"
    disp = disp.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    # Build display table
    display_cols = ["patient_id"]
    if "name" in disp.columns:
        display_cols.append("name")
    display_cols += ["age", "ward", "block", "current_rrt_score", "predicted_rrt_4hr", "predicted_rrt_8hr"]
    display_cols  = [c for c in display_cols if c in disp.columns]

    ui_df = disp[display_cols].copy()
    rename_map = {
        "patient_id":         "Patient ID",
        "name":               "Name",
        "age":                "Age",
        "ward":               "Ward",
        "block":              "Block",
        "current_rrt_score":  "⚠️ Current RRT",
        "predicted_rrt_4hr":  "🔮 4h Forecast",
        "predicted_rrt_8hr":  "🔮 8h Forecast",
    }
    ui_df.rename(columns=rename_map, inplace=True)

    # Style
    style = ui_df.style
    apply_fn = style.map if hasattr(style, "map") else style.applymap
    style = apply_fn(_rrt_highlight, subset=["⚠️ Current RRT"])
    apply_fn2 = style.map if hasattr(style, "map") else style.applymap
    pred_cols = [c for c in ["🔮 4h Forecast", "🔮 8h Forecast"] if c in ui_df.columns]
    if pred_cols:
        style = apply_fn2(_rrt_highlight_pred, subset=pred_cols)

    # Interactive table with click-to-detail
    try:
        event = st.dataframe(
            style,
            use_container_width=True,
            height=420,
            on_select="rerun",
            selection_mode="single-row",
            key="triage_table",
        )
        st.caption("👆 Click any row to open the full Patient Detail page.")

        if event is not None:
            sel = event.selection if hasattr(event, "selection") else event.get("selection", {})
            rows = sel["rows"] if isinstance(sel, dict) else getattr(sel, "rows", [])
            if rows:
                clicked_id = disp.iloc[rows[0]]["patient_id"]
                st.session_state["view"]             = "detail"
                st.session_state["selected_patient"] = clicked_id
                st.rerun()

    except TypeError:
        st.dataframe(style, use_container_width=True, height=420)
        dc1, dc2 = st.columns([3, 1])
        with dc1:
            pick = st.selectbox("Open Patient Detail for:", disp["patient_id"].tolist(), key="detail_picker")
        with dc2:
            st.write("")
            if st.button("View Detail →"):
                st.session_state["view"]             = "detail"
                st.session_state["selected_patient"] = pick
                st.rerun()

    # Risk trajectory chart
    _render_risk_trajectory(disp)


def _render_risk_trajectory(disp: pd.DataFrame) -> None:
    """Render Now / +4h / +8h risk trajectory chart for a selected patient."""
    if disp.empty:
        return

    import altair as alt

    st.markdown("---")
    st.markdown('<p class="section-label">FORECAST DETAIL</p>', unsafe_allow_html=True)
    st.subheader("Risk Trajectory — Now / +4h / +8h")

    selected_id = st.selectbox(
        "Select patient:", disp["patient_id"].unique().tolist(), key="trajectory_picker"
    )
    if not selected_id:
        return

    row = disp[disp["patient_id"] == selected_id].iloc[0]

    curr   = int(row.get("current_rrt_score", 0))
    p4     = int(row.get("predicted_rrt_4hr", 0))
    p8     = int(row.get("predicted_rrt_8hr", 0))

    def _tier(s: int) -> str:
        cat, _ = rrt_category_from_score(s)
        return cat.capitalize()

    trend_df = pd.DataFrame({
        "Horizon": ["Now", "+4h", "+8h"],
        "Score":   [curr, p4, p8],
        "Tier":    [_tier(curr), _tier(p4), _tier(p8)],
    })

    base = alt.Chart(trend_df).encode(
        x=alt.X("Horizon:N", sort=["Now", "+4h", "+8h"], title=None),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 18]), title="RRT Score"),
    )

    warn_line = alt.Chart(pd.DataFrame({"y": [6]})).mark_rule(
        strokeDash=[4, 4], color="#E8A33D", opacity=0.7
    ).encode(y="y:Q")

    crit_line = alt.Chart(pd.DataFrame({"y": [12]})).mark_rule(
        strokeDash=[4, 4], color="#D7263D", opacity=0.7
    ).encode(y="y:Q")

    line = base.mark_line(color="#0F6E68", strokeWidth=3)

    points = base.mark_point(filled=True, size=180).encode(
        color=alt.Color(
            "Tier:N",
            scale=alt.Scale(
                domain=["Critical", "Warning", "Stable"],
                range=["#D7263D", "#E8A33D", "#2E8B57"],
            ),
            legend=alt.Legend(title="Risk tier", orient="top"),
        ),
        tooltip=["Horizon:N", "Score:Q", "Tier:N"],
    )

    chart = (
        (warn_line + crit_line + line + points)
        .properties(height=300)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelFont="IBM Plex Mono", titleFont="IBM Plex Sans",
            labelColor="#0B2545", titleColor="#0B2545", gridColor="#DCE3E8",
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"BiLSTM forecast · Patient {selected_id} · Dashed lines = Warning (6) and Critical (12) thresholds")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render the full sidebar and return possibly updated df."""
    render_logout_control()
    st.sidebar.markdown("---")

    user_role = st.session_state.get("user_role", "Nurse")

    # Admin-only panel
    if user_role == "Admin":
        with st.sidebar.expander("⚙️ Admin Panel", expanded=False):
            render_admin_panel()
        st.sidebar.markdown("---")

    # Model status
    try:
        from ai.predict_bilstm import model_is_ready
        model_ok = model_is_ready()
    except Exception:
        model_ok = False

    model_color = "#3FE0A6" if model_ok else "#E8A33D"
    model_text  = "BiLSTM Model Ready" if model_ok else "BiLSTM Model Not Trained"
    model_icon  = "🟢" if model_ok else "🟠"

    st.sidebar.markdown(
        f'<p style="color:{model_color}; font-size:0.82rem; font-weight:600;">'
        f'{model_icon} {model_text}</p>',
        unsafe_allow_html=True,
    )

    if not model_ok:
        if st.sidebar.button("🚀 Train BiLSTM Model", use_container_width=True):
            with st.spinner("Training BiLSTM model — this may take a few minutes …"):
                try:
                    from ai.train_bilstm import train
                    metrics = train()
                    st.sidebar.success(
                        f"✅ Model trained! val_loss={metrics['val_loss']:.4f}, "
                        f"val_mae={metrics['val_mae']:.4f}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"Training failed: {exc}")

    st.sidebar.markdown("---")

    # Patient registration
    df = _render_patient_registration(df)
    return df


# ---------------------------------------------------------------------------
# App router
# ---------------------------------------------------------------------------

# Load and tick data
df_patients = _load_live_records()
df_patients = run_synthetic_tick_if_due(df_patients, interval_seconds=REFRESH_INTERVAL_SECONDS)

# Sidebar
df_patients = _render_sidebar(df_patients)

# Hero banner
status, last_tick_dt = connection_status(REFRESH_INTERVAL_SECONDS)
status_label, status_color = STATUS_DISPLAY[status]
last_tick_str = last_tick_dt.astimezone().strftime("%H:%M:%S") if last_tick_dt else "—"

st.markdown(f"""
<div class="hero-banner">
    <span class="hero-kicker"><span class="live-dot"></span>LIVE TRIAGE FEED · RRT BI-LSTM</span>
    <div class="hero-title">AI Patient Deterioration Forecasting Dashboard</div>
    <p class="hero-subtitle">
        18-Point RRT Scoring · Bidirectional LSTM 4h &amp; 8h Deterioration Forecasts · Real-Time Monitoring
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f'<p style="color:{status_color}; font-weight:600; font-size:0.88rem; margin-bottom:0.1rem;">'
    f'{status_label} · Last data tick: {last_tick_str}</p>',
    unsafe_allow_html=True,
)


# ---- Route ----
if st.session_state["view"] == "detail" and st.session_state["selected_patient"]:
    def _go_back() -> None:
        st.session_state["view"]             = "dashboard"
        st.session_state["selected_patient"] = None
        st.rerun()

    render_patient_detail_page(
        df_patients,
        st.session_state["selected_patient"],
        _go_back,
    )
else:
    _render_dashboard(df_patients)
