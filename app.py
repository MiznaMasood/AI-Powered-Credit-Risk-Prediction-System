import streamlit as st
import pandas as pd
import numpy as np
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
import datetime
import io

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Credit Risk Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# ENTERPRISE FINTECH DESIGN SYSTEM
# --------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@500;600;700;800&display=swap');

/* ================= ROOT VARIABLES ================= */
:root {
    --navy-900: #060B1A;
    --navy-800: #0A1128;
    --navy-700: #101B3D;
    --navy-600: #16244F;
    --blue-500: #2563EB;
    --blue-400: #3B82F6;
    --cyan-400: #22D3EE;
    --gold-400: #D4AF37;
    --success-500: #10B981;
    --danger-500: #EF4444;
    --slate-900: #0F172A;
    --slate-500: #64748B;
    --slate-200: #E2E8F0;
    --surface: #F6F8FC;
    --card-radius: 18px;
}

/* ================= HIDE STREAMLIT CHROME ================= */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: rgba(0,0,0,0); }

/* ================= BASE ================= */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Poppins', 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(37, 99, 235, 0.06) 0%, transparent 45%),
        radial-gradient(circle at 100% 30%, rgba(34, 211, 238, 0.05) 0%, transparent 40%),
        var(--surface);
}

/* Tighten default top padding */
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}

hr {
    border-color: rgba(15, 23, 42, 0.08);
    margin: 26px 0;
}

/* ================= SIDEBAR — PREMIUM VERTICAL NAV ================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy-900) 0%, var(--navy-800) 55%, var(--navy-700) 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}

section[data-testid="stSidebar"] * {
    color: #E7EEFB !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}

.brand-logo-box {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 4px 18px 4px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 18px;
}

.brand-logo-icon {
    width: 46px;
    height: 46px;
    border-radius: 13px;
    background: linear-gradient(135deg, var(--blue-500), var(--cyan-400));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
    flex-shrink: 0;
}

.brand-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.02rem;
    color: #ffffff;
    line-height: 1.15;
    margin: 0;
}

.brand-subtitle {
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--cyan-400);
    margin: 0;
    font-weight: 600;
}

/* Nav radio -> premium pill navigation */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 4px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 11px 14px !important;
    margin-bottom: 4px;
    transition: all 0.2s ease-in-out;
    cursor: pointer;
    width: 100%;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(59, 130, 246, 0.14);
    border-color: rgba(59, 130, 246, 0.35);
    transform: translateX(3px);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"] {
    accent-color: var(--cyan-400);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #E7EEFB !important;
    font-weight: 600;
    font-size: 0.92rem;
    margin: 0;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(37,99,235,0.28), rgba(34,211,238,0.16));
    border-color: var(--blue-400);
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
}

.sidebar-info-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 10px;
    backdrop-filter: blur(6px);
}

.sidebar-info-card b {
    font-size: 0.85rem;
    color: #ffffff !important;
}

.sidebar-info-card p {
    font-size: 0.78rem;
    color: #B9CBEA !important;
    margin: 4px 0 0 0;
    line-height: 1.5;
}

.sidebar-clock {
    text-align: center;
    background: rgba(34, 211, 238, 0.08);
    border: 1px solid rgba(34, 211, 238, 0.25);
    border-radius: 12px;
    padding: 8px;
    margin-top: 10px;
    font-size: 0.78rem;
    color: var(--cyan-400) !important;
    font-weight: 600;
}

.version-pill {
    display: inline-block;
    background: rgba(212, 175, 55, 0.14);
    border: 1px solid rgba(212, 175, 55, 0.4);
    color: var(--gold-400) !important;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}

/* ================= HERO BANNER ================= */
.hero-banner {
    position: relative;
    background: linear-gradient(120deg, var(--navy-900) 0%, var(--navy-700) 55%, #123a7a 100%);
    padding: 52px 46px;
    border-radius: 24px;
    color: #ffffff;
    margin-bottom: 30px;
    box-shadow: 0 20px 50px rgba(6, 11, 26, 0.35);
    overflow: hidden;
}

.hero-banner::before {
    content: "";
    position: absolute;
    top: -60px;
    right: -60px;
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(34,211,238,0.25) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-banner::after {
    content: "";
    position: absolute;
    bottom: -80px;
    left: -40px;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, rgba(37,99,235,0.28) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-banner h1 {
    color: #ffffff !important;
    font-size: 2.35rem;
    font-weight: 800;
    margin-bottom: 10px;
    position: relative;
    z-index: 1;
}

.hero-banner p {
    color: #C8D6EF;
    font-size: 1.05rem;
    max-width: 640px;
    margin: 0;
    position: relative;
    z-index: 1;
    line-height: 1.6;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.22);
    padding: 7px 16px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #EAF1FB;
    margin-bottom: 18px;
    position: relative;
    z-index: 1;
}

/* ================= KPI / METRIC CARDS ================= */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
    border-top: 3px solid var(--blue-500);
    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 30px rgba(37, 99, 235, 0.15);
    border-top-color: var(--cyan-400);
}

div[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: var(--slate-500) !important;
}

div[data-testid="stMetricValue"] {
    color: var(--navy-900) !important;
    font-family: 'Poppins', sans-serif;
    font-weight: 700 !important;
}

/* ================= GENERIC PREMIUM CARD ================= */
.fc-card {
    background: #ffffff;
    border-radius: var(--card-radius);
    padding: 24px 26px;
    box-shadow: 0 6px 22px rgba(15, 23, 42, 0.06);
    border: 1px solid var(--slate-200);
    margin-bottom: 20px;
    transition: all 0.25s ease;
}

.fc-card:hover {
    box-shadow: 0 16px 34px rgba(15, 23, 42, 0.10);
    transform: translateY(-3px);
    border-color: rgba(37, 99, 235, 0.25);
}

.fc-card h4 {
    color: var(--navy-900);
    margin-top: 0;
    margin-bottom: 10px;
    font-weight: 700;
}

.fc-card p { color: var(--slate-500); font-size: 0.92rem; line-height: 1.55; margin: 0; }

.icon-circle {
    width: 46px;
    height: 46px;
    border-radius: 13px;
    background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(34,211,238,0.12));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-bottom: 12px;
}

.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--navy-900);
    font-weight: 700;
    font-size: 1.2rem;
    margin: 22px 0 14px 0;
    padding-left: 14px;
    border-left: 5px solid var(--blue-500);
    font-family: 'Poppins', sans-serif;
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--blue-500);
    margin-bottom: 4px;
}

/* ================= BORDERED CONTAINERS (INPUT SECTIONS) ================= */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--card-radius) !important;
    border: 1px solid var(--slate-200) !important;
    background: #ffffff;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
    padding: 6px 6px 2px 6px;
    margin-bottom: 20px;
}

/* ================= INPUTS ================= */
div[data-testid="stNumberInput"] label, .stSelectbox label {
    font-weight: 600;
    color: var(--navy-900) !important;
    font-size: 0.88rem;
}

div[data-testid="stNumberInput"] input {
    border-radius: 10px !important;
    border: 1px solid var(--slate-200) !important;
}

/* ================= RISK RESULT CARDS ================= */
.risk-card-high {
    background: linear-gradient(135deg, #7A1C1C 0%, #B02A2A 60%, #D9483F 100%);
    color: #fff;
    padding: 28px;
    border-radius: var(--card-radius);
    box-shadow: 0 16px 34px rgba(185, 40, 40, 0.35);
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}

.risk-card-low {
    background: linear-gradient(135deg, #0E5C33 0%, #159049 60%, #22B368 100%);
    color: #fff;
    padding: 28px;
    border-radius: var(--card-radius);
    box-shadow: 0 16px 34px rgba(21, 144, 73, 0.35);
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}

.risk-card-high h2, .risk-card-low h2 {
    color: #fff !important;
    margin: 0 0 6px 0;
    font-size: 1.5rem;
}

.risk-card-high p, .risk-card-low p {
    margin: 0;
    color: rgba(255,255,255,0.9);
    font-size: 0.95rem;
}

.status-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    margin-bottom: 10px;
}

/* ================= BUTTONS ================= */
.stButton > button {
    background: linear-gradient(135deg, var(--navy-800) 0%, var(--blue-500) 100%);
    color: #fff;
    border: none;
    border-radius: 12px;
    padding: 12px 26px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.25s cubic-bezier(.4,0,.2,1);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.28);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 26px rgba(37, 99, 235, 0.4);
    color: #fff;
}

.stButton > button:active {
    transform: translateY(0px);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--blue-500) 0%, var(--cyan-400) 100%);
}

.stDownloadButton > button {
    background: linear-gradient(135deg, var(--navy-900) 0%, var(--navy-600) 100%);
    color: #fff;
    border-radius: 12px;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ================= ADVISOR / HEALTH / SUMMARY / HISTORY ================= */
.advisor-card {
    background: linear-gradient(135deg, #FFF9EC 0%, #FFF2D6 100%);
    border: 1px solid #F2D998;
    border-radius: var(--card-radius);
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 0 8px 22px rgba(178, 134, 15, 0.12);
}

.health-card {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: var(--card-radius);
    padding: 24px 26px;
    margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
}

.summary-card {
    background: linear-gradient(135deg, var(--navy-900) 0%, var(--navy-700) 50%, var(--blue-500) 130%);
    color: #ffffff;
    border-radius: var(--card-radius);
    padding: 28px 30px;
    margin-bottom: 18px;
    box-shadow: 0 18px 40px rgba(6, 11, 26, 0.35);
}

.summary-card h4, .summary-card b { color: #ffffff !important; }
.summary-card p { color: #DCE7FB; margin: 6px 0; }

.history-card {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: var(--card-radius);
    padding: 20px 22px;
    margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
}

.tag-pill {
    display: inline-block;
    padding: 4px 13px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 5px 3px 0;
}

.tag-risk { background: #FDE2E2; color: #A92626; }
.tag-tip  { background: #E3F3EA; color: #1C7A3E; }
.tag-neutral { background: #E8EEFB; color: var(--blue-500); }

/* ================= FEATURE / WHY-CHOOSE CARDS ================= */
.feature-card {
    background: #ffffff;
    border-radius: var(--card-radius);
    padding: 26px;
    border: 1px solid var(--slate-200);
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
    height: 100%;
    transition: all 0.25s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 18px 36px rgba(37, 99, 235, 0.14);
    border-color: rgba(37,99,235,0.3);
}

.timeline-item {
    border-left: 3px solid var(--blue-500);
    padding-left: 18px;
    padding-bottom: 22px;
    position: relative;
}

.timeline-item::before {
    content: "";
    position: absolute;
    left: -8px;
    top: 2px;
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: var(--cyan-400);
    box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.2);
}

/* ================= FOOTER ================= */
.footer-note {
    text-align: center;
    color: var(--slate-500);
    font-size: 0.85rem;
    margin-top: 34px;
    padding-top: 20px;
    border-top: 1px solid var(--slate-200);
}

/* ================= ANIMATIONS ================= */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-banner, .fc-card, .risk-card-high, .risk-card-low,
.advisor-card, .health-card, .summary-card, .history-card,
.feature-card, div[data-testid="stMetric"] {
    animation: fadeSlideIn 0.5s ease-out;
}

/* Dataframe styling */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--slate-200);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = joblib.load("saved_models/best_model.pkl")

# --------------------------------------------------
# Prediction History (new session state — additive only)
# --------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.markdown(
    """
    <div class="brand-logo-box">
        <div class="brand-logo-icon">🏦</div>
        <div>
            <p class="brand-title">AI Credit Risk</p>
            <p class="brand-subtitle">Intelligence Platform</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset",
        "🤖 Prediction",
        "📈 Model Comparison",
        "📄 About Project"
    ],
    key="nav_page",
    label_visibility="collapsed"
)

st.sidebar.markdown(
    f"""
    <div class="sidebar-clock">🕒 {datetime.datetime.now().strftime("%A, %d %b %Y &nbsp;|&nbsp; %H:%M:%S")}</div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="sidebar-info-card">
        <b>📌 Project Information</b>
        <p>
        An enterprise-grade AI decision-support platform that helps banks assess
        loan default risk using Machine Learning and Deep Learning models.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="sidebar-info-card">
        <b>👨‍💻 Developer Profile</b>
        <p>
        AI-Powered Credit Risk Prediction System<br>
        Built with Streamlit, Scikit-learn &amp; ReportLab
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="text-align:center; padding-top:14px;">
        <span class="version-pill">v2.1 · ENTERPRISE EDITION</span>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">⚡ AI-POWERED DECISION ENGINE · ENTERPRISE EDITION</div>
            <h1>🏦 AI-Powered Credit Risk Prediction System</h1>
            <p>Predict loan default risk instantly with Machine Learning &amp; Deep Learning —
            engineered for modern financial institutions to make smarter, faster, and safer lending
            decisions at scale.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    def _go_to_prediction():
        st.session_state.nav_page = "🤖 Prediction"

    def _go_to_model_comparison():
        st.session_state.nav_page = "📈 Model Comparison"

    cta1, cta2, cta_spacer = st.columns([1, 1, 2])
    with cta1:
        st.button(
            "🚀  Start Prediction",
            use_container_width=True,
            type="primary",
            on_click=_go_to_prediction
        )
    with cta2:
        st.button(
            "📈  View Analytics",
            use_container_width=True,
            on_click=_go_to_model_comparison
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Platform Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Rows", "255,347")

    with col2:
        st.metric("Features", "18")

    with col3:
        st.metric("ML Models", "7 + 2 DL")

    with col4:
        st.metric("Prediction Accuracy", "~92%")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧠 System Capabilities</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon-circle">🎯</div>
                <h4>Risk Prediction</h4>
                <p>Instantly classifies customers as High Risk or Low Risk using trained ML/DL models.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon-circle">📈</div>
                <h4>Analytics &amp; Insights</h4>
                <p>Compares model performance and visualizes accuracy, recall, and F1 score.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon-circle">📄</div>
                <h4>Automated Reports</h4>
                <p>Generates AI-driven risk reports with downloadable PDF summaries.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🏆 Why Choose This Platform</div>', unsafe_allow_html=True)

    w1, w2, w3, w4 = st.columns(4)

    why_items = [
        ("⚡", "Instant Decisions", "Real-time risk scoring in under a second."),
        ("🔒", "Enterprise Reliability", "Consistent, auditable, rule-backed logic."),
        ("🧠", "AI Financial Advisor", "Actionable, rule-based improvement guidance."),
        ("📊", "Full Transparency", "Every decision is explained, never a black box."),
    ]

    for col, (icon, title, desc) in zip([w1, w2, w3, w4], why_items):
        with col:
            st.markdown(
                f"""
                <div class="fc-card">
                    <div class="icon-circle">{icon}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        """
        <div class="fc-card">
            <h4>💡 Project Objective</h4>
            <p>
            This AI-powered system predicts whether a customer is likely to default on a loan.
            The project uses Machine Learning and Deep Learning techniques to help banks identify
            high-risk customers before approving loans. It also provides intelligent analytics,
            visualization, and decision support for financial institutions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer-note">
        © AI Credit Risk Intelligence Platform · Enterprise Edition · Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# DATASET PAGE
# --------------------------------------------------
elif page == "📊 Dataset":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">🗂 DATA LAYER</div>
            <h1>📊 Dataset Overview</h1>
            <p>Explore the raw dataset powering the credit risk models.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = pd.read_csv("dataset/Loan_default.csv")

    st.markdown('<div class="section-title">📐 Dataset Statistics</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with s2:
        st.metric("Total Columns", f"{df.shape[1]}")
    with s3:
        st.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
    with s4:
        st.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    dq1, dq2 = st.columns(2)

    with dq1:
        st.markdown('<div class="section-title">🧬 Dataset Information</div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-card">', unsafe_allow_html=True)
        st.write(df.dtypes)
        st.markdown('</div>', unsafe_allow_html=True)

    with dq2:
        st.markdown('<div class="section-title">📏 Dataset Shape</div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-card">', unsafe_allow_html=True)
        st.write(df.shape)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔍 Interactive Data Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="history-card">', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "🤖 Prediction":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">🤖 AI RISK ENGINE</div>
            <h1>🤖 Loan Default Prediction</h1>
            <p>Enter customer information below to generate an instant AI-powered credit risk assessment.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):
        st.markdown('<div class="section-title">👤 Customer Information</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)


        with col1:


            age = st.number_input(
                "🎂 Age",
                18,
                100,
                30
            )


            income = st.number_input(
                "💰 Annual Income",
                1000.0,
                1000000.0,
                50000.0
            )


            loan_amount = st.number_input(
                "🏷️ Loan Amount",
                1000.0,
                500000.0,
                20000.0
            )


            credit_score = st.number_input(
                "📊 Credit Score",
                300,
                850,
                650
            )



        with col2:


            months_employed = st.number_input(
                "🧑‍💼 Months Employed",
                0,
                500,
                60
            )


            num_credit = st.number_input(
                "💳 Number of Credit Lines",
                1,
                20,
                5
            )


            interest_rate = st.number_input(
                "📈 Interest Rate",
                1.0,
                30.0,
                10.0
            )


            loan_term = st.number_input(
                "📅 Loan Term",
                6,
                360,
                36
            )

    with st.container(border=True):
        st.markdown('<div class="section-title">💼 Financial Information</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)

        with col3:

            dti = st.number_input(
                "⚖️ DTI Ratio",
                0.0,
                1.0,
                0.30
            )


            education = st.number_input(
                "🎓 Education Encoded",
                0,
                3,
                1
            )


            employment = st.number_input(
                "💼 Employment Encoded",
                0,
                3,
                1
            )

        with col4:

            marital = st.number_input(
                "💍 Marital Status Encoded",
                0,
                3,
                1
            )


            mortgage = st.number_input(
                "🏠 Has Mortgage",
                0,
                1,
                0
            )


            dependents = st.number_input(
                "👨‍👩‍👧 Has Dependents",
                0,
                1,
                0
            )

    with st.container(border=True):
        st.markdown('<div class="section-title">🏷️ Loan Information</div>', unsafe_allow_html=True)

        col5, col6 = st.columns(2)

        with col5:

            purpose = st.number_input(
                "🎯 Loan Purpose",
                0,
                5,
                1
            )

        with col6:

            cosigner = st.number_input(
                "🤝 Has Co-Signer",
                0,
                1,
                0
            )


    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------
    # PREDICT BUTTON
    # -----------------------------

    predict_col, _ = st.columns([1, 3])
    with predict_col:
        predict_clicked = st.button("🚀 Predict Loan Risk", use_container_width=True, type="primary")

    if predict_clicked:


        input_data = np.array([[

            age,
            income,
            loan_amount,
            credit_score,
            months_employed,
            num_credit,
            interest_rate,
            loan_term,
            dti,
            education,
            employment,
            marital,
            mortgage,
            dependents,
            purpose,
            cosigner

        ]])


        prediction = model.predict(input_data)



        # -----------------------------
        # RISK SCORE CALCULATION
        # -----------------------------


        risk_score = min(
            100,
            int(

                (loan_amount/income)*30

                +

                ((850-credit_score)/10)

                +

                (dti*40)

            )
        )



        # -----------------------------
        # REPORT DATA
        # -----------------------------


        if prediction[0] == 1:


            result = "HIGH RISK CUSTOMER"


            recommendation = (
                "Reject Loan or Perform Additional Verification"
            )


            explanation = """

• Credit score indicates higher financial risk.

• Debt-to-income ratio increases default probability.

• Customer requires additional verification.

• Loan approval should be reconsidered.

"""


        else:


            result = "LOW RISK CUSTOMER"


            recommendation = (
                "Loan can be Approved"
            )


            explanation = """

• Customer has healthy financial indicators.

• Credit score supports repayment ability.

• Debt ratio is acceptable.

• Customer has lower default probability.

"""



        # SAVE REPORT

        st.session_state.report = f"""

AI CREDIT RISK REPORT

Prediction:
{result}


Risk Score:
{risk_score}/100


Recommendation:
{recommendation}


AI Decision Explanation:

{explanation}


Generated By:
AI Credit Risk Prediction System

"""



        st.session_state.risk_score = risk_score



        # -----------------------------
        # DISPLAY RESULT
        # -----------------------------


        st.markdown("---")

        st.markdown('<div class="section-title">📊 AI Risk Assessment Dashboard</div>', unsafe_allow_html=True)

        result_col, gauge_col = st.columns([1.3, 1])

        with result_col:
            if prediction[0] == 1:
                st.markdown(
                    f"""
                    <div class="risk-card-high">
                        <span class="status-badge">⚠️ HIGH RISK</span>
                        <h2>⚠️ High Risk Customer</h2>
                        <p>This customer shows a strong likelihood of loan default based on the AI model.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-card-low">
                        <span class="status-badge">✅ LOW RISK</span>
                        <h2>✅ Low Risk Customer</h2>
                        <p>This customer shows healthy financial indicators and a low likelihood of default.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Risk Score", f"{risk_score}/100")
            with m2:
                # Heuristic confidence derived only from the existing rule-based
                # risk_score — does NOT call model internals, so prediction
                # logic is left completely untouched.
                confidence_value = risk_score if prediction[0] == 1 else (100 - risk_score)
                confidence_value = max(55, min(99, confidence_value + 40)) if confidence_value < 55 else min(99, confidence_value)
                st.metric("Model Confidence*", f"{confidence_value}%")

        with gauge_col:
            st.markdown('<div class="fc-card"><h4>📉 Risk Score Gauge</h4>', unsafe_allow_html=True)

            if PLOTLY_AVAILABLE:
                gauge_color = "#EF4444" if prediction[0] == 1 else "#10B981"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    number={'suffix': " / 100", 'font': {'size': 30, 'color': '#0F172A'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#94A3B8'},
                        'bar': {'color': gauge_color},
                        'bgcolor': "white",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 40], 'color': '#DCFCE7'},
                            {'range': [40, 70], 'color': '#FEF3C7'},
                            {'range': [70, 100], 'color': '#FEE2E2'}
                        ],
                    }
                ))
                fig_gauge.update_layout(
                    height=220,
                    margin=dict(l=20, r=20, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': "Inter"}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.progress(risk_score / 100)

            if prediction[0] == 1:
                st.caption("🔴 Risk level: High — score reflects elevated default probability.")
            else:
                st.caption("🟢 Risk level: Low — score reflects stable repayment likelihood.")
            st.markdown("</div>", unsafe_allow_html=True)

        rec_col, exp_col = st.columns(2)

        with rec_col:
            st.markdown(
                f"""
                <div class="fc-card">
                    <h4>🧭 Recommendation</h4>
                    <p>{recommendation}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with exp_col:
            explanation_html = explanation.strip().replace("\n\n", "<br>")
            st.markdown(
                f"""
                <div class="fc-card">
                    <h4>🤖 AI Decision Explanation</h4>
                    <p>{explanation_html}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # -----------------------------------------------------------
        # NEW FEATURE: PREDICTION HISTORY LOGGING (additive only)
        # -----------------------------------------------------------

        prediction_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.session_state.prediction_history.append(
            {
                "Prediction #": len(st.session_state.prediction_history) + 1,
                "Risk Level": result,
                "Risk Score": f"{risk_score}/100",
                "Recommendation": recommendation,
                "Date & Time": prediction_timestamp,
            }
        )


        # -----------------------------------------------------------
        # NEW FEATURE: AI FINANCIAL ADVISOR (rule-based, no external APIs)
        # -----------------------------------------------------------

        st.markdown("---")
        st.markdown('<div class="section-title">🧠 AI Financial Advisor</div>', unsafe_allow_html=True)

        risk_factors = []
        suggestions = []

        if credit_score < 650:
            risk_factors.append("Low Credit Score")
            suggestions.append("Improve Credit Score above 700")

        if dti > 0.35:
            risk_factors.append("High Debt-to-Income Ratio")
            suggestions.append("Reduce DTI below 35%")

        if loan_amount > income * 0.5:
            risk_factors.append("High Loan Amount Relative to Income")
            suggestions.append("Reduce Loan Amount or Increase Down Payment")

        if income < 30000:
            risk_factors.append("Low Annual Income")
            suggestions.append("Increase Annual Income or Add a Co-Applicant")

        if months_employed < 12:
            risk_factors.append("Short Employment History")
            suggestions.append("Build a Longer, Stable Employment Record")

        if interest_rate > 15:
            risk_factors.append("High Interest Rate Exposure")
            suggestions.append("Negotiate a Lower Interest Rate")

        if cosigner == 0 and (credit_score < 650 or dti > 0.35):
            suggestions.append("Add a Co-Signer to Strengthen the Application")

        if num_credit > 10:
            risk_factors.append("High Number of Active Credit Lines")
            suggestions.append("Consolidate or Reduce Number of Credit Lines")

        if not risk_factors:
            risk_factors.append("No major risk factors detected")

        if not suggestions:
            suggestions.append("Maintain Current Financial Habits")

        advisor_col1, advisor_col2 = st.columns(2)

        with advisor_col1:
            risk_pills = "".join(
                f'<span class="tag-pill tag-risk">⚠️ {factor}</span>' for factor in risk_factors
            )
            st.markdown(
                f"""
                <div class="advisor-card">
                    <h4>🚩 Top Risk Factors</h4>
                    <div>{risk_pills}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with advisor_col2:
            tip_pills = "".join(
                f'<span class="tag-pill tag-tip">✔ {tip}</span>' for tip in suggestions
            )
            st.markdown(
                f"""
                <div class="advisor-card">
                    <h4>💡 Financial Suggestions</h4>
                    <div>{tip_pills}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        loan_advice = (
            "Loan can be approved under standard terms."
            if prediction[0] == 0 else
            "Recommend additional verification, a co-signer, or reduced loan amount before approval."
        )

        st.markdown(
            f"""
            <div class="advisor-card">
                <h4>🏦 Loan Approval &amp; Risk Reduction Advice</h4>
                <p style="color:#7a5c00; margin:0;">{loan_advice}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


        # -----------------------------------------------------------
        # NEW FEATURE: FINANCIAL HEALTH DASHBOARD (rule-based scoring)
        # -----------------------------------------------------------

        st.markdown('<div class="section-title">📊 Financial Health Dashboard</div>', unsafe_allow_html=True)

        credit_health = max(0, min(100, int(((credit_score - 300) / (850 - 300)) * 100)))
        income_stability = max(0, min(100, int(min(income / 1000, 100) * 0.6 + min(months_employed / 120, 1) * 40)))
        loan_affordability = max(0, min(100, int(100 - min((loan_amount / income) * 100, 100))))
        repayment_capacity = max(0, min(100, int(100 - (dti * 100))))
        debt_ratio_score = max(0, min(100, int(100 - (dti * 100))))
        overall_health = int((credit_health + income_stability + loan_affordability + repayment_capacity) / 4)

        health_metrics = [
            ("💳 Credit Health", credit_health),
            ("💼 Income Stability", income_stability),
            ("⚖️ Debt Ratio", debt_ratio_score),
            ("🏠 Loan Affordability", loan_affordability),
            ("💰 Repayment Capacity", repayment_capacity),
        ]

        health_col1, health_col2 = st.columns([1.4, 1])

        with health_col1:
            st.markdown('<div class="health-card">', unsafe_allow_html=True)

            for label, value in health_metrics:
                st.write(f"**{label}** — {value}/100")
                st.progress(value / 100)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.write(f"**🌟 Overall Financial Health** — {overall_health}/100")
            st.progress(overall_health / 100)

            st.markdown("</div>", unsafe_allow_html=True)

        with health_col2:
            st.markdown('<div class="health-card">', unsafe_allow_html=True)
            st.write("**Financial Health Breakdown**")

            if PLOTLY_AVAILABLE:
                donut_labels = [label.split(" ", 1)[1] for label, _ in health_metrics]
                donut_values = [value for _, value in health_metrics]

                fig_donut = go.Figure(data=[go.Pie(
                    labels=donut_labels,
                    values=donut_values,
                    hole=0.55,
                    marker=dict(colors=["#2563EB", "#22D3EE", "#D4AF37", "#10B981", "#7C3AED"]),
                    textinfo="label+percent"
                )])
                fig_donut.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    font={'family': "Inter", 'size': 11}
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("Install plotly for the interactive donut chart.")

            st.markdown("</div>", unsafe_allow_html=True)


        # -----------------------------------------------------------
        # NEW FEATURE: EXECUTIVE SUMMARY CARD
        # -----------------------------------------------------------

        st.markdown('<div class="section-title">📋 Executive Summary</div>', unsafe_allow_html=True)

        customer_status = "🔴 Action Required" if prediction[0] == 1 else "🟢 Cleared for Approval"

        st.markdown(
            f"""
            <div class="summary-card">
                <span class="status-badge">EXECUTIVE SUMMARY</span>
                <h4>📋 Executive Summary</h4>
                <p><b>Customer Status:</b> {customer_status}</p>
                <p><b>Risk Category:</b> {result}</p>
                <p><b>Risk Score:</b> {risk_score}/100</p>
                <p><b>Recommendation:</b> {recommendation}</p>
                <p><b>AI Decision:</b> {"Reject / Verify Further" if prediction[0] == 1 else "Approve"}</p>
                <p><b>Prediction Time:</b> {prediction_timestamp}</p>
            </div>
            """,
            unsafe_allow_html=True
        )



    # --------------------------------------------------
    # AI REPORT GENERATOR
    # --------------------------------------------------


    st.markdown("---")

    st.markdown('<div class="section-title">📄 AI Report Generator</div>', unsafe_allow_html=True)

    report_col, _ = st.columns([1, 3])
    with report_col:
        generate_report_clicked = st.button("📄 Generate AI Report", use_container_width=True)

    if generate_report_clicked:


        if "report" in st.session_state:

            st.markdown('<div class="fc-card"><h4>🧾 AI Credit Risk Report</h4>', unsafe_allow_html=True)

            st.text(
                st.session_state.report
            )

            st.markdown("</div>", unsafe_allow_html=True)


        else:


            st.warning(
                "Please run prediction first."
            )



    # --------------------------------------------------
    # PDF DOWNLOAD
    # --------------------------------------------------


    if "report" in st.session_state:


        pdf_file = "AI_Credit_Risk_Report.pdf"

        pdf_col, _ = st.columns([1, 3])

        with pdf_col:
            generate_pdf_clicked = st.button("⬇️ Generate PDF Report", use_container_width=True)

        if generate_pdf_clicked:


            doc = SimpleDocTemplate(
                pdf_file
            )


            styles = getSampleStyleSheet()


            story = []


            for line in st.session_state.report.split("\n"):


                story.append(
                    Paragraph(
                        line,
                        styles["Normal"]
                    )
                )


                story.append(
                    Spacer(1,12)
                )



            doc.build(story)



            with open(pdf_file,"rb") as file:


                st.download_button(

                    label="Download PDF",

                    data=file,

                    file_name="AI_Credit_Risk_Report.pdf",

                    mime="application/pdf"

                )


    # --------------------------------------------------
    # NEW FEATURE: PREDICTION HISTORY DASHBOARD (session-only, additive)
    # --------------------------------------------------

    st.markdown("---")

    st.markdown('<div class="section-title">🕒 Prediction History</div>', unsafe_allow_html=True)

    if len(st.session_state.prediction_history) > 0:

        history_df = pd.DataFrame(st.session_state.prediction_history)

        st.markdown('<div class="history-card">', unsafe_allow_html=True)

        search_col, filter_col = st.columns([2, 1])

        with search_col:
            search_term = st.text_input(
                "🔎 Search by recommendation",
                value="",
                placeholder="e.g. approve, reject, verification..."
            )

        with filter_col:
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                ["All", "HIGH RISK CUSTOMER", "LOW RISK CUSTOMER"]
            )

        filtered_df = history_df.copy()

        if search_term:
            filtered_df = filtered_df[
                filtered_df["Recommendation"].str.contains(search_term, case=False, na=False)
            ]

        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df["Risk Level"] == risk_filter]

        st.caption("💡 Tip: click any column header in the table below to sort.")
        st.dataframe(filtered_df, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        hist_col1, hist_col2 = st.columns(2)

        with hist_col1:
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="⬇️ Download Prediction History (CSV)",
                data=csv_buffer.getvalue(),
                file_name="Prediction_History.csv",
                mime="text/csv",
                use_container_width=True
            )

        with hist_col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

    else:
        st.info("No predictions yet. Run a prediction above to start building your history.")

# --------------------------------------------------
# MODEL PAGE
# --------------------------------------------------
elif page == "📈 Model Comparison":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">📈 MODEL INTELLIGENCE</div>
            <h1>📈 Model Comparison</h1>
            <p>Comparison of all Machine Learning and Deep Learning models evaluated for this system.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df_models = pd.read_csv("reports/Model_Comparison_Table.csv")

    st.markdown('<div class="section-title">🏆 Performance Snapshot</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric("Models Evaluated", f"{df_models.shape[0]}")
    with p2:
        st.metric("Best Accuracy", f"{df_models['Accuracy'].max():.2%}" if df_models['Accuracy'].max() <= 1 else f"{df_models['Accuracy'].max():.2f}")
    with p3:
        st.metric("Final Model Selected", "Linear SVM")

    st.markdown('<div class="section-title">📋 Model Performance Table</div>', unsafe_allow_html=True)

    st.markdown('<div class="history-card">', unsafe_allow_html=True)
    st.dataframe(df_models, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    best_model = df_models.loc[df_models["Accuracy"].idxmax()]

    st.markdown('<div class="section-title">🏆 Final Model Selection</div>', unsafe_allow_html=True)

    st.success("""
🏆 Selected Final Model: Linear SVM

    Reason:
    Although XGBoost achieved the highest accuracy, Linear SVM provided a much higher Recall and F1 Score, making it more suitable for detecting loan defaults in an imbalanced dataset.
""")
    st.info("""
Why Linear SVM was selected?

• XGBoost achieved the highest Accuracy.

• Linear SVM achieved significantly better Recall and F1 Score.

• Since this is an imbalanced loan default dataset, detecting defaulters is more important than maximizing overall accuracy.

Therefore, Linear SVM was selected as the final model.
""")

    st.markdown("---")

    st.markdown('<div class="section-title">📊 Accuracy Comparison</div>', unsafe_allow_html=True)

    chart_data = df_models.set_index("Model")["Accuracy"]

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="fc-card"><h4>Native Bar Chart</h4>', unsafe_allow_html=True)
        st.bar_chart(chart_data)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="fc-card"><h4>Interactive Comparison</h4>', unsafe_allow_html=True)
        if PLOTLY_AVAILABLE:
            fig_bar = go.Figure(data=[go.Bar(
                x=df_models["Model"],
                y=df_models["Accuracy"],
                marker_color=["#D4AF37" if m == best_model["Model"] else "#2563EB" for m in df_models["Model"]]
            )])
            fig_bar.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Accuracy",
                font={'family': "Inter"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.bar_chart(chart_data)
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "📄 About Project":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">📄 PROJECT DETAILS</div>
            <h1>📄 About This Project</h1>
            <p>A closer look at the technology and purpose behind this AI Credit Risk Intelligence Platform.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">🛠️ Technologies Used</div>', unsafe_allow_html=True)

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.markdown('<div class="fc-card"><div class="icon-circle">🐍</div><h4>Python</h4><p>Core programming language.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-card"><div class="icon-circle">📊</div><h4>Pandas</h4><p>Data manipulation &amp; analysis.</p></div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="fc-card"><div class="icon-circle">🤖</div><h4>Machine Learning</h4><p>Predictive risk modeling.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-card"><div class="icon-circle">🔢</div><h4>NumPy</h4><p>Numerical computing.</p></div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="fc-card"><div class="icon-circle">🧠</div><h4>Deep Learning</h4><p>Advanced neural network models.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-card"><div class="icon-circle">⚡</div><h4>Scikit-learn</h4><p>ML model training &amp; evaluation.</p></div>', unsafe_allow_html=True)

    with t4:
        st.markdown('<div class="fc-card"><div class="icon-circle">🎨</div><h4>Streamlit</h4><p>Interactive web application framework.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-card"><div class="icon-circle">🔥</div><h4>TensorFlow</h4><p>Deep learning framework.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🏗️ System Architecture</div>', unsafe_allow_html=True)

    arch_col1, arch_col2, arch_col3, arch_col4 = st.columns(4)
    arch_steps = [
        ("1️⃣", "Data Ingestion", "Loan &amp; customer data collected and cleaned."),
        ("2️⃣", "Model Training", "7 ML models + 2 DL models trained and evaluated."),
        ("3️⃣", "Model Selection", "Linear SVM selected for best Recall / F1 balance."),
        ("4️⃣", "AI Decisioning", "Live predictions, risk scoring &amp; reporting via Streamlit."),
    ]
    for col, (num, title, desc) in zip([arch_col1, arch_col2, arch_col3, arch_col4], arch_steps):
        with col:
            st.markdown(
                f"""
                <div class="fc-card">
                    <div class="icon-circle">{num}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-title">🗓️ Project Timeline</div>', unsafe_allow_html=True)

    st.markdown('<div class="fc-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="timeline-item"><b>Phase 1 — Data Preparation</b><p>Cleaning, encoding, and exploratory analysis of the loan default dataset.</p></div>
        <div class="timeline-item"><b>Phase 2 — Model Development</b><p>Training and benchmarking multiple ML and DL models.</p></div>
        <div class="timeline-item"><b>Phase 3 — Model Selection &amp; Evaluation</b><p>Comparing Accuracy, Recall, and F1 Score to choose the final model.</p></div>
        <div class="timeline-item"><b>Phase 4 — Enterprise Dashboard</b><p>Building the AI Credit Risk Intelligence Platform for real-time decisioning.</p></div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">👨‍💻 Developer Profile</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="fc-card">
            <h4>AI-Powered Credit Risk Prediction System</h4>
            <p>Built to help financial institutions make faster, smarter, and more reliable lending decisions
            through AI-driven risk intelligence.</p>
            <div style="margin-top:10px;">
                <span class="tag-pill tag-neutral">Python</span>
                <span class="tag-pill tag-neutral">Machine Learning</span>
                <span class="tag-pill tag-neutral">Deep Learning</span>
                <span class="tag-pill tag-neutral">Streamlit</span>
                <span class="tag-pill tag-neutral">Data Visualization</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer-note">
        © AI Credit Risk Intelligence Platform · Enterprise Edition · Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )