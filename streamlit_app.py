"""
streamlit_app.py

Production dashboard for the Retail Demand Forecasting platform.
Connects to the FastAPI backend when available and falls back to
direct Python model loading when the API is offline.

Features
--------
- Historical sales exploration with store/department breakdowns
- Champion model forecasting with interactive visualizations
- MLflow experiment leaderboard from walk-forward validation
- Inventory optimization (Safety Stock, ROP, EOQ)
- Stockout and overstock risk classification
- Data drift detection via Kolmogorov-Smirnov tests
- AI Copilot panel powered by Hugging Face serverless inference
- System health and endpoint monitoring

Author: Nani
"""

import math
import time
import html
import os
from pathlib import Path

# -------------------------------------------------------------------
# Python 3.13 / 3.14 compatibility hotfix for Altair
# Altair v5 schemas use closed=True on Python 3.13+, but the PEP 728
# 'closed' TypedDict feature was deferred, causing TypeErrors in 3.13/3.14.
# -------------------------------------------------------------------
import sys
import typing
if sys.version_info >= (3, 13):
    try:
        orig_new = typing._TypedDictMeta.__new__
        def _patched_typeddict_new(cls, *args, **kwargs):
            kwargs.pop('closed', None)
            kwargs.pop('extra_items', None)
            return orig_new(cls, *args, **kwargs)
        typing._TypedDictMeta.__new__ = _patched_typeddict_new
    except Exception:
        pass

import altair as alt

import numpy as np
import pandas as pd
import requests
import streamlit as st

from pipeline.preprocessing.data_loader import WalmartDataLoader
from pipeline.preprocessing.aggregations import WalmartAggregator
from pipeline.api.forecast_service import ForecastService
from pipeline.inventory.optimization import optimize_inventory
from pipeline.inventory.risk import classify_risk
from pipeline.monitoring.drift_detector import DataDriftDetector
from pipeline.utils.conversational_assistant import ConversationalRetailAssistant
from pipeline.utils.llm_client import HFLLMClient

# -------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="RetailCast | Demand Forecasting Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Global constants
# -------------------------------------------------------------------
# Try to resolve backend API URL from environment variables or Streamlit secrets
# Fallback to local 127.0.0.1 if not defined (ideal for local development/docker compose)
API_BASE = (
    os.getenv("RETAIL_API_BASE")
    or (st.secrets.get("RETAIL_API_BASE") if "secrets" in dir(st) else None)
    or "http://127.0.0.1:8000"
)
COLOR_PALETTE = {
    "primary":     "#6366f1",
    "secondary":   "#8b5cf6",
    "accent":      "#f59e0b",
    "positive":    "#22c55e",
    "negative":    "#ef4444",
    "neutral":     "#94a3b8",
    "bg_dark":     "#0f172a",
    "bg_card":     "#1e293b",
    "text_bright": "#f8fafc",
    "text_muted":  "#94a3b8",
}

# -------------------------------------------------------------------
# Custom CSS — injected once per session
# -------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* --- product story header --- */
.hero-header {
    background: linear-gradient(135deg, #101827 0%, #172033 55%, #203a63 100%);
    border: 1px solid rgba(148,163,184,.22);
    padding: 2.4rem 2.6rem 2rem 2.6rem;
    border-radius: 10px;
    margin-bottom: 1.25rem;
    box-shadow: 0 18px 45px rgba(15,23,42,.28);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 34%;
    height: 100%;
    background: linear-gradient(135deg, rgba(38,87,214,.22), rgba(8,127,91,.16));
    clip-path: polygon(24% 0, 100% 0, 100% 100%, 0 100%);
}
.hero-header h1 {
    color: #f8fafc !important;
    font-weight: 780;
    font-size: 2.5rem;
    margin: 0;
    letter-spacing: 0;
    max-width: 760px;
}
.hero-header p {
    color: #cbd5e1 !important;
    margin-top: .75rem;
    font-size: 1.02rem;
    font-weight: 400;
    max-width: 760px;
    line-height: 1.55;
}
.hero-kicker {
    color: #93c5fd;
    font-size: .78rem;
    font-weight: 760;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: .75rem;
}
.hero-metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .8rem;
    margin-top: 1.55rem;
    max-width: 930px;
}
.hero-metric {
    border: 1px solid rgba(148,163,184,.22);
    background: rgba(15,23,42,.48);
    border-radius: 8px;
    padding: .95rem 1rem;
}
.hero-metric strong {
    display: block;
    color: #f8fafc;
    font-size: 1.45rem;
    line-height: 1.1;
    font-weight: 760;
}
.hero-metric span {
    display: block;
    color: #94a3b8;
    margin-top: .35rem;
    font-size: .75rem;
    line-height: 1.32;
    font-weight: 560;
}
.product-chips {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
    margin-top: 1.2rem;
}
.product-chip {
    border: 1px solid rgba(148,163,184,.28);
    background: rgba(248,250,252,.06);
    color: #dbeafe;
    border-radius: 999px;
    padding: .42rem .7rem;
    font-size: .78rem;
    font-weight: 640;
}
.architecture-strip {
    background: #111827;
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 10px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 1.45rem;
}
.architecture-strip h3 {
    margin: 0 0 .9rem 0;
    color: #e5e7eb;
    font-size: 1.02rem;
    font-weight: 740;
}
.architecture-flow {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: .65rem;
}
.architecture-step {
    border: 1px solid rgba(148,163,184,.18);
    background: #172033;
    border-radius: 8px;
    padding: .8rem .75rem;
    min-height: 84px;
}
.architecture-step .step-label {
    color: #60a5fa;
    font-size: .66rem;
    font-weight: 760;
    letter-spacing: .05em;
    text-transform: uppercase;
}
.architecture-step strong {
    display: block;
    color: #f8fafc;
    margin-top: .25rem;
    font-size: .88rem;
    line-height: 1.25;
}
.architecture-step span {
    display: block;
    color: #94a3b8;
    margin-top: .28rem;
    font-size: .72rem;
    line-height: 1.28;
}
.summary-panel {
    background: linear-gradient(135deg, rgba(15,23,42,.92), rgba(30,41,59,.92));
    border: 1px solid rgba(148,163,184,.2);
    border-radius: 10px;
    padding: 1.55rem;
    margin-top: 1.2rem;
}
.summary-panel h3 {
    margin: 0 0 .35rem 0;
    color: #f8fafc;
    font-size: 1.3rem;
}
.summary-panel p {
    color: #cbd5e1;
    margin: 0 0 1rem 0;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .8rem;
}
.summary-item {
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 8px;
    padding: .85rem;
    color: #e5e7eb;
    background: rgba(15,23,42,.45);
    font-weight: 650;
}
.summary-item span {
    display: block;
    color: #94a3b8;
    font-size: .75rem;
    margin-top: .25rem;
    font-weight: 540;
}
@media (max-width: 1100px) {
    .hero-metrics,
    .architecture-flow,
    .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 700px) {
    .hero-metrics,
    .architecture-flow,
    .summary-grid {
        grid-template-columns: 1fr;
    }
    .hero-header h1 {
        font-size: 1.8rem;
    }
}

/* --- metric tiles --- */
.kpi-tile {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99,102,241,.2);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    min-height: 110px;
    transition: transform .18s ease, box-shadow .18s ease;
}
.kpi-tile:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(99,102,241,.18);
}
.kpi-tile .kpi-label {
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: .35rem;
}
.kpi-tile .kpi-value {
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0;
}
.kpi-tile .kpi-sub {
    font-size: .75rem;
    color: #64748b;
    margin-top: .2rem;
}

/* --- section headers --- */
.section-hdr {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #6366f1;
    padding-left: .8rem;
    margin: 1.6rem 0 .8rem 0;
}

/* --- ai-panel --- */
.ai-panel {
    background: linear-gradient(135deg, rgba(99,102,241,.08) 0%, rgba(139,92,246,.06) 100%);
    border: 1px solid rgba(139,92,246,.22);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: .6rem;
}
.ai-panel h4 {
    color: #a78bfa !important;
    font-size: 1rem;
    font-weight: 700;
    margin: 0 0 .6rem 0;
}
.chat-panel {
    background: #111827;
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
}
.chat-panel h4 {
    color: #e5e7eb !important;
    font-size: .98rem;
    margin: 0 0 .3rem 0;
}
.chat-note {
    color: #94a3b8;
    font-size: .78rem;
    line-height: 1.45;
    margin-bottom: .75rem;
}
.chat-row {
    border: 1px solid rgba(148,163,184,.14);
    border-radius: 8px;
    padding: .7rem .8rem;
    margin-bottom: .55rem;
    background: rgba(15,23,42,.62);
}
.chat-role {
    color: #93c5fd;
    font-size: .7rem;
    font-weight: 760;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-bottom: .25rem;
}
.chat-text {
    color: #dbeafe;
    font-size: .84rem;
    line-height: 1.5;
}

/* --- status badges --- */
.badge { display: inline-block; padding: .15rem .55rem; border-radius: 4px; font-size: .78rem; font-weight: 700; }
.badge-online  { background: #166534; color: #bbf7d0; }
.badge-offline { background: #7f1d1d; color: #fecaca; }
.badge-warn    { background: #78350f; color: #fde68a; }
.badge-risk-low      { background: #14532d; color: #bbf7d0; }
.badge-risk-medium   { background: #78350f; color: #fde68a; }
.badge-risk-high     { background: #7f1d1d; color: #fecaca; }
.badge-risk-critical { background: #450a0a; color: #fca5a5; }

/* --- tweak tabs --- */
div[data-baseweb="tab-list"] { gap: 6px; }
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: .82rem !important;
}

/* --- divider --- */
.soft-divider { border: none; border-top: 1px solid rgba(148,163,184,.15); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ===================================================================
#  DATA LAYER — cached loaders
# ===================================================================
@st.cache_data(show_spinner=False)
def load_train_data():
    loader = WalmartDataLoader()
    df = loader.load_train_data()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_features_data():
    loader = WalmartDataLoader()
    df = loader.load_features_data()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_stores_data():
    loader = WalmartDataLoader()
    return loader.load_stores_data()


@st.cache_data(show_spinner=False)
def load_leaderboard():
    path = Path("artifacts/leaderboards/leaderboard.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_champion_metadata():
    path = Path("artifacts/models/champion_metadata.json")
    if not path.exists():
        return None
    import json
    with open(path, "r") as f:
        return json.load(f)


# ===================================================================
#  UTILITY HELPERS
# ===================================================================
def check_api_health():
    """Ping the FastAPI backend and return status dict."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "online": True,
                "model_loaded": data.get("model_loaded", False),
                "model_name": data.get("model_name", "Unknown"),
            }
    except Exception:
        pass
    return {"online": False, "model_loaded": False, "model_name": None}


def aggregate_series(train_df, level, store_id, dept_id):
    """Return (series_df, label) for the selected aggregation level."""
    agg = WalmartAggregator()
    if level == "Enterprise (Company)":
        series = agg.get_company_sales(train_df)
        label = "All Stores — Company Level"
    elif level == "Single Store":
        series = agg.get_store_sales(train_df, store_id=store_id)
        label = f"Store #{store_id}"
    else:
        series = agg.get_store_department_sales(train_df, store_id=store_id, dept_id=dept_id)
        label = f"Store #{store_id} · Dept #{dept_id}"
    series["Date"] = pd.to_datetime(series["Date"])
    series = series.sort_values("Date").reset_index(drop=True)
    return series, label


def build_kpi_html(label, value, sub="", color="#e2e8f0"):
    return f"""
    <div class='kpi-tile'>
        <div class='kpi-label'>{label}</div>
        <p class='kpi-value' style='color:{color};'>{value}</p>
        <div class='kpi-sub'>{sub}</div>
    </div>
    """


def risk_badge(level_str):
    css_map = {
        "Low":      "badge-risk-low",
        "Medium":   "badge-risk-medium",
        "High":     "badge-risk-high",
        "Critical": "badge-risk-critical",
    }
    cls = css_map.get(level_str, "badge-risk-medium")
    return f"<span class='badge {cls}'>{level_str}</span>"


def render_retail_chat(forecast_context, llm_model_choice):
    """
    Render the full-width conversational analytics assistant.
    """

    st.markdown(
        """
        <div class='chat-panel'>
            <h4>Conversational Retail Analytics Assistant</h4>
            <div class='chat-note'>
                Ask business questions about forecasted sales, inventory gaps,
                what-if scenarios, stockout risk, and recommended actions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2, q3, q4 = st.columns(4)
    example_questions = [
        "How will sales perform next month?",
        "Which inventory action should we take?",
        "What if demand increases by 20%?",
        "Which category contributes the most revenue?",
    ]
    with q1:
        if st.button(example_questions[0], use_container_width=True):
            st.session_state.retail_chat_question = example_questions[0]
    with q2:
        if st.button(example_questions[1], use_container_width=True):
            st.session_state.retail_chat_question = example_questions[1]
    with q3:
        if st.button(example_questions[2], use_container_width=True):
            st.session_state.retail_chat_question = example_questions[2]
    with q4:
        if st.button(example_questions[3], use_container_width=True):
            st.session_state.retail_chat_question = example_questions[3]

    with st.form("retail_chat_form", clear_on_submit=True):
        user_question = st.text_area(
            "Ask a business question",
            value=st.session_state.retail_chat_question,
            placeholder=(
                "Example: If Product A demand increases by 20%, current inventory is "
                "1000 units, forecasted demand is 1400 units, and lead time is 7 days, "
                "what action should we take?"
            ),
            height=90
        )
        ask_clicked = st.form_submit_button(
            "Ask Assistant",
            type="primary",
            use_container_width=True
        )

    if ask_clicked and user_question.strip():
        with st.spinner("Assistant is calculating business impact and generating an answer..."):
            chat_client = HFLLMClient(model_name=llm_model_choice)
            assistant = ConversationalRetailAssistant(
                llm_client=chat_client
            )
            chat_result = assistant.answer_question(
                question=user_question,
                business_context=forecast_context,
                conversation_history=st.session_state.retail_chat_history
            )
            st.session_state.retail_chat_history = chat_result["conversation_history"]
            st.session_state.retail_chat_question = ""
            st.session_state.last_chat_status = {
                "verified": chat_result["verified"],
                "model_used": chat_result["model_used"],
                "intent": chat_result["detected_intent"],
                "entity": chat_result["referenced_entity"],
            }

    if st.session_state.retail_chat_history:
        chat_status = st.session_state.get("last_chat_status", {})
        status_label = "Generated" if chat_status.get("verified") else "Validated fallback"
        st.caption(
            f"{status_label} response | "
            f"Intent: {chat_status.get('intent', 'not available')} | "
            f"Context: {chat_status.get('entity', 'active selection')} | "
            f"Model: {chat_status.get('model_used', 'not available')}"
        )
        for message in st.session_state.retail_chat_history[-6:]:
            role = "Business User" if message["role"] == "user" else "RetailCast Assistant"
            safe_content = html.escape(message["content"]).replace("\n", "<br>")
            st.markdown(
                f"""
                <div class='chat-row'>
                    <div class='chat-role'>{role}</div>
                    <div class='chat-text'>{safe_content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Clear Assistant Conversation", use_container_width=True):
            st.session_state.retail_chat_history = []
            st.session_state.retail_chat_question = ""
            st.session_state.last_chat_status = {}


# ===================================================================
#  LOAD CORE DATA
# ===================================================================
with st.spinner("Loading Walmart datasets…"):
    train_data = load_train_data()
    features_data = load_features_data()
    stores_data = load_stores_data()

api_status = check_api_health()

# ===================================================================
#  SIDEBAR
# ===================================================================
st.sidebar.markdown("## RetailCast")
st.sidebar.caption("Demand Forecasting | MLOps Platform v2.0")
st.sidebar.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

# --- backend health ---
if api_status["online"]:
    if api_status["model_loaded"]:
        st.sidebar.markdown(
            "**API Status** <span class='badge badge-online'>CONNECTED</span>",
            unsafe_allow_html=True,
        )
        st.sidebar.caption(f"Serving **{api_status['model_name']}** model")
    else:
        st.sidebar.markdown(
            "**API Status** <span class='badge badge-warn'>NO MODEL</span>",
            unsafe_allow_html=True,
        )
        st.sidebar.caption("Backend is running but no champion artifact loaded.")
else:
    st.sidebar.markdown(
        "**API Status** <span class='badge badge-offline'>OFFLINE</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Falling back to direct Python model loading.")

st.sidebar.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

# --- forecast controls ---
st.sidebar.subheader("Forecast Controls")
level = st.sidebar.selectbox(
    "Aggregation Level",
    ["Enterprise (Company)", "Single Store", "Store & Department"],
)

store_id = 1
dept_id = 1
if level in ("Single Store", "Store & Department"):
    store_id = st.sidebar.number_input("Store ID", min_value=1, max_value=45, value=1)
if level == "Store & Department":
    dept_id = st.sidebar.number_input("Department ID", min_value=1, max_value=99, value=1)

horizon = st.sidebar.slider("Forecast Horizon (weeks)", 1, 52, 12)

st.sidebar.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

# --- llm model picker ---
st.sidebar.subheader("AI Copilot")
llm_model_choice = st.sidebar.selectbox(
    "LLM Backend",
    [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ],
)

if "retail_chat_history" not in st.session_state:
    st.session_state.retail_chat_history = []
if "retail_chat_question" not in st.session_state:
    st.session_state.retail_chat_question = ""
if "latest_inventory_result" not in st.session_state:
    st.session_state.latest_inventory_result = {}
if "latest_risk_result" not in st.session_state:
    st.session_state.latest_risk_result = {}

# ===================================================================
#  HERO HEADER
# ===================================================================
st.markdown("""
<div class='hero-header'>
    <div class='hero-kicker'>Enterprise Retail Demand Forecasting Platform</div>
    <h1>RetailCast: Demand Forecasting & Decision Intelligence</h1>
    <p>
        A production-style ML system that turns weekly retail sales history into forecasts,
        inventory decisions, risk signals, and business recommendations.
    </p>
    <div class='hero-metrics'>
        <div class='hero-metric'><strong>421K+</strong><span>Walmart weekly sales records processed</span></div>
        <div class='hero-metric'><strong>3.95%</strong><span>MAPE from the SARIMA champion run</span></div>
        <div class='hero-metric'><strong>45</strong><span>Stores represented in the dataset</span></div>
        <div class='hero-metric'><strong>81</strong><span>Departments included in the demand history</span></div>
    </div>
    <div class='product-chips'>
        <span class='product-chip'>SARIMA</span>
        <span class='product-chip'>Prophet</span>
        <span class='product-chip'>XGBoost</span>
        <span class='product-chip'>Walk-Forward Validation</span>
        <span class='product-chip'>MLflow</span>
        <span class='product-chip'>FastAPI</span>
        <span class='product-chip'>Inventory Optimization</span>
        <span class='product-chip'>Llama 3.1</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='architecture-strip'>
    <h3>System Architecture</h3>
    <div class='architecture-flow'>
        <div class='architecture-step'>
            <div class='step-label'>Data</div>
            <strong>Raw Retail Sales</strong>
            <span>Stores, departments, weekly sales, external features</span>
        </div>
        <div class='architecture-step'>
            <div class='step-label'>Modeling</div>
            <strong>SARIMA / Prophet / XGBoost</strong>
            <span>Feature engineering and walk-forward validation</span>
        </div>
        <div class='architecture-step'>
            <div class='step-label'>Lifecycle</div>
            <strong>Champion Selection</strong>
            <span>Leaderboard, metadata, full-history retraining</span>
        </div>
        <div class='architecture-step'>
            <div class='step-label'>Serving</div>
            <strong>FastAPI Inference</strong>
            <span>Forecast, inventory, risk, and monitoring endpoints</span>
        </div>
        <div class='architecture-step'>
            <div class='step-label'>Decision</div>
            <strong>Inventory + LLM Insights</strong>
            <span>Operational recommendations for business users</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================================================================
#  MAIN TABS
# ===================================================================
tab_overview, tab_forecast, tab_inventory, tab_drift, tab_system = st.tabs([
    "Data Explorer",
    "Forecast & Insights",
    "Inventory Optimizer",
    "Drift Monitor",
    "System Health",
])

# ===================================================================
#  TAB 1 — DATA EXPLORER
# ===================================================================
with tab_overview:
    st.markdown("<div class='section-hdr'>Dataset Overview</div>", unsafe_allow_html=True)

    ov1, ov2, ov3, ov4 = st.columns(4)
    n_stores = train_data["Store"].nunique()
    n_depts = train_data["Dept"].nunique()
    date_range = f"{train_data['Date'].min():%b %Y} — {train_data['Date'].max():%b %Y}"
    total_records = f"{len(train_data):,}"

    with ov1:
        st.markdown(build_kpi_html("Total Records", total_records, "train.csv rows", COLOR_PALETTE["primary"]),
                    unsafe_allow_html=True)
    with ov2:
        st.markdown(build_kpi_html("Stores", str(n_stores), "distinct locations", COLOR_PALETTE["accent"]),
                    unsafe_allow_html=True)
    with ov3:
        st.markdown(build_kpi_html("Departments", str(n_depts), "product categories", COLOR_PALETTE["secondary"]),
                    unsafe_allow_html=True)
    with ov4:
        st.markdown(build_kpi_html("Date Range", date_range, "weekly cadence", COLOR_PALETTE["positive"]),
                    unsafe_allow_html=True)

    st.write("")

    # --- top stores & departments side by side ---
    agg = WalmartAggregator()
    col_ts, col_td = st.columns(2)

    with col_ts:
        st.markdown("<div class='section-hdr'>Top 10 Stores by Revenue</div>", unsafe_allow_html=True)
        top_stores = agg.get_top_stores(train_data, top_n=10)
        bar_stores = (
            alt.Chart(top_stores)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#6366f1")
            .encode(
                x=alt.X("Store:N", sort="-y", title="Store ID"),
                y=alt.Y("Weekly_Sales:Q", title="Total Sales ($)", axis=alt.Axis(format="$,.0f")),
                tooltip=[
                    alt.Tooltip("Store:N", title="Store"),
                    alt.Tooltip("Weekly_Sales:Q", title="Sales", format="$,.0f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(bar_stores, use_container_width=True)

    with col_td:
        st.markdown("<div class='section-hdr'>Top 10 Departments by Revenue</div>", unsafe_allow_html=True)
        top_depts = agg.get_top_departments(train_data, top_n=10)
        bar_depts = (
            alt.Chart(top_depts)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#8b5cf6")
            .encode(
                x=alt.X("Dept:N", sort="-y", title="Department"),
                y=alt.Y("Weekly_Sales:Q", title="Total Sales ($)", axis=alt.Axis(format="$,.0f")),
                tooltip=[
                    alt.Tooltip("Dept:N", title="Department"),
                    alt.Tooltip("Weekly_Sales:Q", title="Sales", format="$,.0f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(bar_depts, use_container_width=True)

    # --- company-level time series ---
    st.markdown("<div class='section-hdr'>Weekly Enterprise Sales Timeline</div>", unsafe_allow_html=True)
    company_ts = agg.get_company_sales(train_data)
    company_ts["Date"] = pd.to_datetime(company_ts["Date"])
    area_chart = (
        alt.Chart(company_ts)
        .mark_area(
            line={"color": "#6366f1", "strokeWidth": 2},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="rgba(99,102,241,.35)", offset=0),
                    alt.GradientStop(color="rgba(99,102,241,.02)", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
            interpolate="monotone",
        )
        .encode(
            x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("Weekly_Sales:Q", title="Weekly Sales ($)", axis=alt.Axis(format="$,.0f")),
            tooltip=[
                alt.Tooltip("Date:T", title="Week"),
                alt.Tooltip("Weekly_Sales:Q", title="Sales", format="$,.0f"),
            ],
        )
        .properties(height=340)
        .interactive()
    )
    st.altair_chart(area_chart, use_container_width=True)

    # --- store info table ---
    with st.expander("📋 Store Metadata Table", expanded=False):
        st.dataframe(stores_data.style.format({"Size": "{:,}"}), use_container_width=True)


# ===================================================================
#  TAB 2 — FORECAST & INSIGHTS
# ===================================================================
with tab_forecast:
    st.markdown("<div class='section-hdr'>Demand Forecast</div>", unsafe_allow_html=True)

    # Aggregate the selected historical series
    try:
        hist_df, series_label = aggregate_series(train_data, level, store_id, dept_id)
    except ValueError as err:
        st.error(f"No data found for the selected filters: {err}")
        st.stop()

    # --- produce forecast ---
    predictions = []
    model_name = "Unknown"
    via_api = False

    if api_status["online"] and api_status["model_loaded"]:
        try:
            payload = {
                "store_id": store_id,
                "department_id": dept_id,
                "forecast_horizon": horizon,
            }
            resp = requests.post(f"{API_BASE}/forecast", json=payload, timeout=8)
            if resp.status_code == 200:
                body = resp.json()
                predictions = body.get("forecast", [])
                model_name = body.get("model_name", "Champion")
                via_api = True
        except Exception:
            pass

    if not via_api:
        service = ForecastService()
        try:
            service.load_model()
            predictions = service.forecast(horizon)
            model_name = service.get_model_name()
        except FileNotFoundError:
            st.warning(
                "Champion model artifact not found.  "
                "Please run `python run_experiments.py` to train and serialize the champion model first."
            )
        except (ModuleNotFoundError, AttributeError) as exc:
            st.error(
                f"Champion model artifact is incompatible with this environment: {exc}. "
                "Run `python run_experiments.py` to regenerate a clean artifact."
            )
        except Exception as exc:
            st.error(f"Failed to load champion model: {exc}")

    if len(predictions) > 0:
        last_date = hist_df["Date"].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=horizon,
            freq="W",
        )
        forecast_df = pd.DataFrame({"Date": forecast_dates, "Weekly_Sales": predictions})

        avg_hist = hist_df["Weekly_Sales"].mean()
        avg_fc = float(np.mean(predictions))
        total_fc = float(np.sum(predictions))
        pct_change = ((avg_fc - avg_hist) / avg_hist) * 100 if avg_hist != 0 else 0.0
        trend_dir = "Upward" if pct_change >= 0 else "Downward"
        trend_arrow = "▲" if pct_change >= 0 else "▼"
        trend_color = COLOR_PALETTE["positive"] if pct_change >= 0 else COLOR_PALETTE["negative"]

        # --- KPI Row ---
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                build_kpi_html("Champion Model", model_name, "via API" if via_api else "local fallback",
                               COLOR_PALETTE["primary"]),
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                build_kpi_html("Historical Avg", f"${avg_hist:,.0f}", "weekly mean", "#e2e8f0"),
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                build_kpi_html("Forecast Avg", f"${avg_fc:,.0f}", f"next {horizon} weeks", "#e2e8f0"),
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                build_kpi_html(
                    "Trend Direction",
                    f"{trend_arrow} {pct_change:+.1f}%",
                    f"{trend_dir} vs history",
                    trend_color,
                ),
                unsafe_allow_html=True,
            )

        st.write("")

        # --- combined line chart ---
        st.markdown("<div class='section-hdr'>Historical vs Forecast — " + series_label + "</div>",
                    unsafe_allow_html=True)

        plot_hist = hist_df.tail(52).copy()
        plot_hist["Type"] = "Historical"

        plot_fc = forecast_df.copy()
        plot_fc["Type"] = "Forecast"

        # bridge: connect lines
        bridge = plot_hist.iloc[[-1]].copy()
        bridge["Type"] = "Forecast"
        plot_fc = pd.concat([bridge, plot_fc], ignore_index=True)

        combined = pd.concat([plot_hist, plot_fc], ignore_index=True)

        line_chart = (
            alt.Chart(combined)
            .mark_line(strokeWidth=3, interpolate="monotone")
            .encode(
                x=alt.X("Date:T", title="Timeline", axis=alt.Axis(format="%b %Y", grid=True)),
                y=alt.Y("Weekly_Sales:Q", title="Weekly Sales ($)", axis=alt.Axis(format="$,.0f", grid=True)),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(domain=["Historical", "Forecast"], range=["#6366f1", "#f59e0b"]),
                    legend=alt.Legend(title="Source"),
                ),
                strokeDash=alt.condition(
                    alt.datum.Type == "Forecast",
                    alt.value([6, 4]),
                    alt.value([0]),
                ),
                tooltip=[
                    alt.Tooltip("Date:T", title="Week"),
                    alt.Tooltip("Weekly_Sales:Q", title="Sales", format="$,.2f"),
                    alt.Tooltip("Type:N", title="Source"),
                ],
            )
            .properties(height=380)
            .interactive()
        )
        st.altair_chart(line_chart, use_container_width=True)

        forecast_context = {
            "store_id": store_id,
            "department_id": dept_id if level == "Store & Department" else None,
            "horizon": horizon,
            "average_historical": float(avg_hist),
            "average_forecast": float(avg_fc),
            "total_forecast": float(total_fc),
            "trend_direction": f"{trend_dir} of {pct_change:+.1f}%",
            "change_pct": float(round(pct_change, 2)),
            "forecast_values": [float(value) for value in predictions],
            "inventory": st.session_state.latest_inventory_result,
            "risk": st.session_state.latest_risk_result,
            "current_inventory": st.session_state.latest_risk_result.get("metrics", {}).get("current_inventory"),
            "items": [
                {
                    "name": f"Department {int(row['Dept'])}",
                    "forecast_revenue": float(row["Weekly_Sales"]),
                    "growth_pct": 0.0,
                }
                for _, row in agg.get_top_departments(train_data, top_n=10).head(10).iterrows()
            ],
        }

        render_retail_chat(
            forecast_context=forecast_context,
            llm_model_choice=llm_model_choice
        )

        # --- leaderboard + AI insights ---
        col_lb, col_ai = st.columns([2, 3])

        with col_lb:
            st.markdown("<div class='section-hdr'>Model Leaderboard</div>", unsafe_allow_html=True)
            leaderboard = load_leaderboard()
            if leaderboard is not None and not leaderboard.empty:
                display_lb = leaderboard.copy()
                display_lb = display_lb.set_index("Rank")
                st.dataframe(
                    display_lb.style.format({
                        "RMSE": "${:,.2f}",
                        "MAE": "${:,.2f}",
                        "MAPE": "{:.2f}%",
                    }).highlight_min(subset=["MAPE"], color="#1e3a2f"),
                    use_container_width=True,
                )
            else:
                st.info("No leaderboard found. Run `python run_experiments.py` to generate one.")

            # champion metadata
            meta = load_champion_metadata()
            if meta:
                st.markdown("##### Champion Hyperparameters")
                params = meta.get("parameters", {})
                params_df = pd.DataFrame(
                    [{"Parameter": k, "Value": str(v)} for k, v in params.items()]
                )
                st.dataframe(params_df, use_container_width=True, hide_index=True)

        with col_ai:
            st.markdown("<div class='section-hdr'>AI Copilot — Business Insights</div>", unsafe_allow_html=True)
            st.caption(f"Powered by `{llm_model_choice}` via HF serverless inference")

            with st.spinner("AI Copilot is analyzing demand patterns…"):
                llm_client = HFLLMClient(model_name=llm_model_choice)
                llm_payload = {
                    "store_id": store_id,
                    "department_id": dept_id if level == "Store & Department" else None,
                    "horizon": horizon,
                    "average_historical": float(avg_hist),
                    "average_forecast": float(avg_fc),
                    "total_forecast": float(total_fc),
                    "trend_direction": f"{trend_dir} of {pct_change:+.1f}%",
                    "change_pct": float(round(pct_change, 2)),
                }
                llm_result = llm_client.generate_retail_insights(llm_payload)

            insights_text = llm_result["raw_insights"]
            verified = llm_result["verified"]
            model_used = llm_result["model_used"]

            verification_tag = "✔ Verified" if verified else "⚠ Fallback"
            st.markdown(
                f"""
                <div class='ai-panel'>
                    <span style='float:right;font-size:.75rem;color:#94a3b8;'>{verification_tag} · {model_used}</span>
                    <h4>💡 Copilot Advisory</h4>
                    <div style='font-size:.88rem;line-height:1.65;color:#cbd5e1;'>
                        {insights_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if verified:
                st.success(
                    "LLM output passed structural validation: section headers present, "
                    "references sales data, minimum length met."
                )
            else:
                st.warning(
                    "LLM API was unavailable or output failed validation. "
                    "Displaying rule-based analytical fallback report."
                )

    # --- forecast table (expandable) ---
    if len(predictions) > 0:
        with st.expander("📋 Raw Forecast Table", expanded=False):
            display_fc = forecast_df.copy()
            display_fc["Date"] = display_fc["Date"].dt.strftime("%Y-%m-%d")
            display_fc.index = range(1, len(display_fc) + 1)
            display_fc.index.name = "Week"
            st.dataframe(
                display_fc.style.format({"Weekly_Sales": "${:,.2f}"}),
                use_container_width=True,
            )


# ===================================================================
#  TAB 3 — INVENTORY OPTIMIZER
# ===================================================================
with tab_inventory:
    st.markdown("<div class='section-hdr'>Inventory Optimization Engine</div>", unsafe_allow_html=True)
    st.caption(
        "Calculate Safety Stock, Reorder Point, and Economic Order Quantity from the generated forecast. "
        "Then classify stockout and overstock risk levels."
    )

    # check if predictions exist from previous tab
    if len(predictions) == 0:
        st.info("Switch to the **Forecast & Insights** tab first to generate a forecast before optimizing inventory.")
    else:
        inv_col1, inv_col2 = st.columns(2)

        with inv_col1:
            st.markdown("##### Supply Chain Parameters")
            lead_time = st.number_input("Lead Time (weeks)", min_value=0.5, max_value=12.0, value=2.0, step=0.5)
            service_level = st.select_slider(
                "Service Level",
                options=[0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.999],
                value=0.95,
                format_func=lambda x: f"{x:.1%}",
            )
            holding_cost = st.number_input("Holding Cost ($/unit/year)", min_value=0.1, max_value=50.0, value=1.5, step=0.1)
            setup_cost = st.number_input("Setup Cost ($/order)", min_value=1.0, max_value=500.0, value=50.0, step=5.0)
            hist_std = float(hist_df["Weekly_Sales"].std())

        with inv_col2:
            st.markdown("##### Current Inventory Position")
            current_inventory = st.number_input(
                "Current Inventory ($)",
                min_value=0.0,
                max_value=float(total_fc * 5),
                value=float(total_fc * 0.4),
                step=1000.0,
                format="%.0f",
            )

        if st.button("🔧 Run Optimization", type="primary", use_container_width=True):
            with st.spinner("Calculating optimal inventory parameters…"):
                opt_result = optimize_inventory(
                    forecast_demands=predictions,
                    historical_sales_std=hist_std,
                    lead_time_weeks=lead_time,
                    service_level=service_level,
                    holding_cost_unit_year=holding_cost,
                    setup_cost_order=setup_cost,
                )
                st.session_state.latest_inventory_result = opt_result
                time.sleep(0.3)  # slight delay for polish

            # display optimization results
            st.markdown("<div class='section-hdr'>Optimization Results</div>", unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(
                    build_kpi_html("Avg Forecasted Demand", f"${opt_result['average_forecasted_demand']:,.0f}",
                                   "per week", COLOR_PALETTE["primary"]),
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    build_kpi_html("Safety Stock", f"${opt_result['safety_stock']:,.0f}",
                                   f"SL={service_level:.0%}", COLOR_PALETTE["accent"]),
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    build_kpi_html("Reorder Point", f"${opt_result['reorder_point']:,.0f}",
                                   f"LT={lead_time}wk", COLOR_PALETTE["secondary"]),
                    unsafe_allow_html=True,
                )
            with r4:
                st.markdown(
                    build_kpi_html("EOQ", f"${opt_result['economic_order_quantity']:,.0f}",
                                   "optimal order qty", COLOR_PALETTE["positive"]),
                    unsafe_allow_html=True,
                )

            st.write("")

            # --- risk classification ---
            st.markdown("<div class='section-hdr'>Risk Assessment</div>", unsafe_allow_html=True)
            risk_result = classify_risk(
                current_inventory=current_inventory,
                reorder_point=opt_result["reorder_point"],
                safety_stock=opt_result["safety_stock"],
                total_forecasted_demand=total_fc,
            )
            st.session_state.latest_risk_result = risk_result

            rc1, rc2 = st.columns(2)
            with rc1:
                so = risk_result["stockout_risk"]
                st.markdown(
                    f"""
                    <div class='kpi-tile'>
                        <div class='kpi-label'>STOCKOUT RISK</div>
                        <p class='kpi-value'>{risk_badge(so['level'])}</p>
                        <div class='kpi-sub'>{so['description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with rc2:
                ov = risk_result["overstock_risk"]
                st.markdown(
                    f"""
                    <div class='kpi-tile'>
                        <div class='kpi-label'>OVERSTOCK RISK</div>
                        <p class='kpi-value'>{risk_badge(ov['level'])}</p>
                        <div class='kpi-sub'>{ov['description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")

            # --- inventory metrics table ---
            with st.expander("📋 Detailed Inventory Metrics", expanded=False):
                inv_metrics = risk_result["metrics"]
                metrics_df = pd.DataFrame(
                    [{"Metric": k.replace("_", " ").title(), "Value": f"${v:,.2f}"} for k, v in inv_metrics.items()]
                )
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ===================================================================
#  TAB 4 — DRIFT MONITOR
# ===================================================================
with tab_drift:
    st.markdown("<div class='section-hdr'>Data Drift Detection</div>", unsafe_allow_html=True)
    st.caption(
        "Kolmogorov-Smirnov (KS) test comparing a baseline window against a recent window "
        "to detect distribution shift in sales data. Useful for triggering model retraining."
    )

    try:
        hist_df_drift, _ = aggregate_series(train_data, level, store_id, dept_id)
    except ValueError:
        st.error("No data available for the selected filters.")
        st.stop()

    n_total = len(hist_df_drift)
    split_point = int(n_total * 0.7)

    if n_total < 20:
        st.warning("Not enough data points for reliable drift detection. Need at least 20 weekly observations.")
    else:
        baseline = hist_df_drift.iloc[:split_point].copy()
        recent = hist_df_drift.iloc[split_point:].copy()

        alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

        if st.button("🔬 Run Drift Analysis", type="primary", use_container_width=True):
            detector = DataDriftDetector(alpha=alpha)
            result = detector.detect_drift(
                baseline_df=baseline,
                current_df=recent,
                columns=["Weekly_Sales"],
            )

            drift_found = result["drift_detected"]

            d1, d2, d3 = st.columns(3)
            with d1:
                drift_status = "YES" if drift_found else "NO"
                drift_color = COLOR_PALETTE["negative"] if drift_found else COLOR_PALETTE["positive"]
                st.markdown(
                    build_kpi_html("Drift Detected", drift_status, f"α = {alpha}", drift_color),
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown(
                    build_kpi_html("Baseline Window", f"{len(baseline)} weeks", f"rows 1–{split_point}",
                                   COLOR_PALETTE["primary"]),
                    unsafe_allow_html=True,
                )
            with d3:
                st.markdown(
                    build_kpi_html("Recent Window", f"{len(recent)} weeks", f"rows {split_point + 1}–{n_total}",
                                   COLOR_PALETTE["secondary"]),
                    unsafe_allow_html=True,
                )

            st.write("")

            # KS result details
            for col_name, metrics in result["metrics"].items():
                st.markdown(f"**Column: `{col_name}`**")
                dm1, dm2, dm3 = st.columns(3)
                with dm1:
                    st.metric("KS Statistic", f"{metrics['ks_statistic']:.4f}")
                with dm2:
                    st.metric("P-Value", f"{metrics['p_value']:.6f}")
                with dm3:
                    if metrics["drift_detected"]:
                        st.error("⚠ Drift detected — consider retraining the champion model.")
                    else:
                        st.success("✔ No significant drift found.")

            # Distribution comparison chart
            st.markdown("<div class='section-hdr'>Distribution Comparison</div>", unsafe_allow_html=True)

            baseline_hist = baseline[["Weekly_Sales"]].copy()
            baseline_hist["Window"] = "Baseline (70%)"
            recent_hist = recent[["Weekly_Sales"]].copy()
            recent_hist["Window"] = "Recent (30%)"
            dist_df = pd.concat([baseline_hist, recent_hist], ignore_index=True)

            hist_chart = (
                alt.Chart(dist_df)
                .mark_bar(opacity=0.55, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Weekly_Sales:Q", bin=alt.Bin(maxbins=30), title="Weekly Sales ($)"),
                    y=alt.Y("count():Q", title="Frequency", stack=None),
                    color=alt.Color(
                        "Window:N",
                        scale=alt.Scale(
                            domain=["Baseline (70%)", "Recent (30%)"],
                            range=["#6366f1", "#f59e0b"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Window:N", title="Window"),
                        alt.Tooltip("count():Q", title="Count"),
                    ],
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(hist_chart, use_container_width=True)


# ===================================================================
#  TAB 5 — SYSTEM HEALTH
# ===================================================================
with tab_system:
    st.markdown("<div class='section-hdr'>System Health & Monitoring</div>", unsafe_allow_html=True)

    # API status
    s1, s2, s3 = st.columns(3)
    with s1:
        status_text = "ONLINE" if api_status["online"] else "OFFLINE"
        status_color = COLOR_PALETTE["positive"] if api_status["online"] else COLOR_PALETTE["negative"]
        st.markdown(
            build_kpi_html("FastAPI Backend", status_text, API_BASE, status_color),
            unsafe_allow_html=True,
        )
    with s2:
        model_text = api_status["model_name"] or "Not Loaded"
        st.markdown(
            build_kpi_html("Served Model", model_text, "champion artifact", COLOR_PALETTE["primary"]),
            unsafe_allow_html=True,
        )
    with s3:
        artifact_exists = Path("artifacts/models/champion_model.pkl").exists()
        st.markdown(
            build_kpi_html(
                "Champion Artifact",
                "Present" if artifact_exists else "Missing",
                "artifacts/models/champion_model.pkl",
                COLOR_PALETTE["positive"] if artifact_exists else COLOR_PALETTE["negative"],
            ),
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown("""
    <div class='summary-panel'>
        <h3>RetailCast Final Summary</h3>
        <p>
            Enterprise retail forecasting platform built to connect model accuracy,
            serving infrastructure, inventory decisions, and business recommendations.
        </p>
        <div class='summary-grid'>
            <div class='summary-item'>421K+ Retail Records<span>45 stores and 81 departments</span></div>
            <div class='summary-item'>Champion Forecasting<span>SARIMA, Prophet, and XGBoost comparison</span></div>
            <div class='summary-item'>3.95% MAPE<span>Best SARIMA validation result</span></div>
            <div class='summary-item'>FastAPI Serving<span>Forecast, inventory, risk, and metrics endpoints</span></div>
            <div class='summary-item'>Inventory Decisions<span>Safety stock, reorder point, EOQ, risk analysis</span></div>
            <div class='summary-item'>LLM Recommendations<span>Llama 3.1 business insight layer</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Endpoint metrics if API is online
    if api_status["online"]:
        st.markdown("<div class='section-hdr'>Endpoint Usage Metrics</div>", unsafe_allow_html=True)
        try:
            metrics_resp = requests.get(f"{API_BASE}/monitoring/metrics", timeout=3)
            if metrics_resp.status_code == 200:
                metrics_data = metrics_resp.json()
                em1, em2, em3 = st.columns([1.2, 2, 2.2])
                with em1:
                    st.metric("Total API Requests", f"{metrics_data.get('total_requests', 0):,}")
                    st.metric("Model Active", "Yes" if metrics_data.get('model_loaded') else "No")
                with em2:
                    st.markdown("<p style='font-size:0.85rem;font-weight:600;color:#94a3b8;'>ENDPOINT REQUESTS</p>", unsafe_allow_html=True)
                    ep_data = metrics_data.get("requests_by_endpoint", {})
                    if ep_data:
                        ep_df = pd.DataFrame(
                            [{"Endpoint": k, "Requests": v} for k, v in ep_data.items()]
                        ).sort_values("Requests", ascending=False)
                        ep_chart = (
                            alt.Chart(ep_df)
                            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#6366f1")
                            .encode(
                                x=alt.X("Requests:Q", title="Request Count"),
                                y=alt.Y("Endpoint:N", sort="-x", title=""),
                                tooltip=[
                                    alt.Tooltip("Endpoint:N"),
                                    alt.Tooltip("Requests:Q"),
                                ],
                            )
                            .properties(height=180)
                        )
                        st.altair_chart(ep_chart, use_container_width=True)
                with em3:
                    st.markdown("<p style='font-size:0.85rem;font-weight:600;color:#94a3b8;'>LATENCY PROFILE (ms)</p>", unsafe_allow_html=True)
                    lat_data = metrics_data.get("latencies", {})
                    if lat_data:
                        lat_df = pd.DataFrame(
                            [{"Endpoint": k, "Latency (ms)": v} for k, v in lat_data.items()]
                        ).sort_values("Latency (ms)", ascending=False)
                        lat_chart = (
                            alt.Chart(lat_df)
                            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#f59e0b")
                            .encode(
                                x=alt.X("Latency (ms):Q", title="Latency (ms)"),
                                y=alt.Y("Endpoint:N", sort="-x", title=""),
                                tooltip=[
                                    alt.Tooltip("Endpoint:N"),
                                    alt.Tooltip("Latency (ms):Q", format=".2f"),
                                ],
                            )
                            .properties(height=180)
                        )
                        st.altair_chart(lat_chart, use_container_width=True)
            else:
                st.warning(f"Metrics endpoint returned status {metrics_resp.status_code}")
        except Exception as exc:
            st.error(f"Failed to fetch metrics: {exc}")
    else:
        st.info("Start the FastAPI backend to view endpoint metrics: `python run_api.py`")

    st.write("")

    # --- pipeline architecture overview ---
    st.markdown("<div class='section-hdr'>Pipeline Architecture</div>", unsafe_allow_html=True)

    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("""
        **Data Layer**
        - `WalmartDataLoader` — CSV ingestion with validation
        - `WalmartAggregator` — Company / Store / Department rollups
        - `FeatureEngineer` — Lag, rolling, and calendar features

        **Model Layer**
        - `SARIMAForecaster` — Seasonal ARIMA with auto-tuning
        - `ProphetForecaster` — Facebook Prophet with holiday effects
        - `XGBoostForecaster` — Gradient boosted trees with lag features
        """)
    with arch_col2:
        st.markdown("""
        **Training & Evaluation**
        - `WalkForwardValidator` — Time series cross-validation
        - `ExperimentRunner` — Multi-model benchmarking
        - `ChampionPipeline` — Retrain, package, register

        **Serving & Monitoring**
        - `FastAPI` + `ForecastService` — REST inference
        - `DataDriftDetector` — KS test for distribution shift
        - `HFLLMClient` — Generative business insights via HF API
        """)

    # --- artifact file listing ---
    st.markdown("<div class='section-hdr'>Artifact Registry</div>", unsafe_allow_html=True)
    artifact_root = Path("artifacts")
    if artifact_root.exists():
        artifact_files = []
        for file_path in sorted(artifact_root.rglob("*")):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                artifact_files.append({
                    "File": str(file_path.relative_to(artifact_root)),
                    "Size": f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_path.stat().st_size / 1024:.1f} KB",
                })
        if artifact_files:
            st.dataframe(pd.DataFrame(artifact_files), use_container_width=True, hide_index=True)
        else:
            st.info("No artifacts found.")
    else:
        st.info("Artifacts directory does not exist.")
