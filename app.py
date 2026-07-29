import streamlit as st
import pandas as pd
import numpy as np
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Credit Risk Prediction System",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# Global Premium Fintech CSS Theme
# --------------------------------------------------
st.markdown("""
<style>

    /* ---------- Base ---------- */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }

    .main {
        background-color: #f4f7fb;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f3a 0%, #0d2a52 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #eaf1fb !important;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        color: #eaf1fb !important;
    }

    /* ---------- Headings ---------- */
    h1, h2, h3 {
        color: #0d2a52;
        font-weight: 700;
    }

    /* ---------- Hero Banner ---------- */
    .hero-banner {
        background: linear-gradient(135deg, #0d2a52 0%, #14396e 60%, #1b4b8f 100%);
        padding: 42px 40px;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(13, 42, 82, 0.25);
    }

    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        margin-bottom: 6px;
    }

    .hero-banner p {
        color: #cfe0f7;
        font-size: 1.02rem;
        margin: 0;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        color: #eaf1fb;
        margin-bottom: 14px;
    }

    /* ---------- Generic Card ---------- */
    .fc-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 18px rgba(13, 42, 82, 0.08);
        border: 1px solid #e9eef7;
        margin-bottom: 18px;
    }

    .fc-card h4 {
        color: #0d2a52;
        margin-top: 0;
        margin-bottom: 10px;
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #0d2a52;
        font-weight: 700;
        font-size: 1.15rem;
        margin: 18px 0 10px 0;
        border-left: 5px solid #1b4b8f;
        padding-left: 10px;
    }

    /* ---------- Metric Cards ---------- */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e9eef7;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 14px rgba(13, 42, 82, 0.06);
    }

    /* ---------- Result Cards ---------- */
    .risk-card-high {
        background: linear-gradient(135deg, #7a1c1c 0%, #a92626 100%);
        color: #fff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(169, 38, 38, 0.35);
        margin-bottom: 16px;
    }

    .risk-card-low {
        background: linear-gradient(135deg, #1c7a3e 0%, #219150 100%);
        color: #fff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(33, 145, 80, 0.35);
        margin-bottom: 16px;
    }

    .risk-card-high h2, .risk-card-low h2 {
        color: #fff !important;
        margin: 0 0 4px 0;
    }

    .risk-card-high p, .risk-card-low p {
        margin: 0;
        color: #f2f2f2;
        font-size: 0.95rem;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #14396e 0%, #1b4b8f 100%);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(27, 75, 143, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(27, 75, 143, 0.35);
        color: #fff;
    }

    /* ---------- Download Button ---------- */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0d2a52 0%, #1b4b8f 100%);
        color: #fff;
        border-radius: 10px;
        font-weight: 600;
        border: none;
    }

    /* ---------- Divider spacing ---------- */
    hr {
        margin: 22px 0;
    }

    /* ---------- Footer note ---------- */
    .footer-note {
        text-align: center;
        color: #7c8aa0;
        font-size: 0.85rem;
        margin-top: 30px;
    }

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = joblib.load("saved_models/best_model.pkl")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 10px 0 4px 0;">
        <h2 style="color:#ffffff; margin-bottom:0;">🏦 AI Credit Risk</h2>
        <p style="color:#9fb6d9; margin-top:2px; font-size:0.85rem;">Intelligence Platform</p>
    </div>
    <hr style="border-color: rgba(255,255,255,0.15);">
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
    ]
)

st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.15);'>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div style="background: rgba(255,255,255,0.06); padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.12);">
        <b>📌 Project Info</b>
        <p style="font-size:0.82rem; color:#cfe0f7; margin-bottom:0;">
        An AI-powered decision support system that helps banks assess loan default risk
        using Machine Learning and Deep Learning models.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div style="background: rgba(255,255,255,0.06); padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.12);">
        <b>👨‍💻 Developer</b>
        <p style="font-size:0.82rem; color:#cfe0f7; margin-bottom:0;">
        AI-Powered Credit Risk Prediction System<br>
        Built with Streamlit &amp; Scikit-learn
        </p>
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
            <div class="hero-badge">⚡ AI-Powered Decision Engine</div>
            <h1>🏦 AI-Powered Credit Risk Prediction System</h1>
            <p>Predict loan default risk instantly with Machine Learning &amp; Deep Learning —
            built for modern financial institutions to make smarter, faster lending decisions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

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

    st.markdown('<div class="section-title">🧠 What This System Does</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="fc-card">
                <h4>🎯 Risk Prediction</h4>
                <p>Instantly classifies customers as High Risk or Low Risk using trained ML/DL models.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="fc-card">
                <h4>📈 Analytics &amp; Insights</h4>
                <p>Compares model performance and visualizes accuracy, recall, and F1 score.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="fc-card">
                <h4>📄 Automated Reports</h4>
                <p>Generates AI-driven risk reports with downloadable PDF summaries.</p>
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

# --------------------------------------------------
# DATASET PAGE
# --------------------------------------------------
elif page == "📊 Dataset":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">🗂 Data Layer</div>
            <h1>📊 Dataset Overview</h1>
            <p>Explore the raw dataset powering the credit risk models.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = pd.read_csv("dataset/Loan_default.csv")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📐 Dataset Shape</div>', unsafe_allow_html=True)
        st.write(df.shape)

    with col2:
        st.markdown('<div class="section-title">🧬 Dataset Information</div>', unsafe_allow_html=True)
        st.write(df.dtypes)

    st.markdown('<div class="section-title">🔍 First Five Rows</div>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "🤖 Prediction":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">🤖 AI Risk Engine</div>
            <h1>🤖 Loan Default Prediction</h1>
            <p>Enter customer information below to generate an instant AI-powered credit risk assessment.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

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



    with col2:


        loan_term = st.number_input(
            "📅 Loan Term",
            6,
            360,
            36
        )


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


        purpose = st.number_input(
            "🎯 Loan Purpose",
            0,
            5,
            1
        )


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
        predict_clicked = st.button("🚀 Predict Loan Risk", use_container_width=True)

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
                        <h2>✅ Low Risk Customer</h2>
                        <p>This customer shows healthy financial indicators and a low likelihood of default.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.metric("Risk Score", f"{risk_score}/100")

        with gauge_col:
            st.markdown('<div class="fc-card"><h4>📉 Risk Score Gauge</h4>', unsafe_allow_html=True)
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
# MODEL PAGE
# --------------------------------------------------
elif page == "📈 Model Comparison":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">📈 Model Intelligence</div>
            <h1>📈 Model Comparison</h1>
            <p>Comparison of all Machine Learning and Deep Learning models evaluated for this system.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df_models = pd.read_csv("reports/Model_Comparison_Table.csv")

    st.markdown('<div class="section-title">📋 Model Performance Table</div>', unsafe_allow_html=True)

    st.dataframe(df_models, use_container_width=True)

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

    st.bar_chart(chart_data)
# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "📄 About Project":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">📄 Project Details</div>
            <h1>📄 About This Project</h1>
            <p>A closer look at the technology and purpose behind this AI Credit Risk Intelligence Platform.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">🛠️ Technologies Used</div>', unsafe_allow_html=True)

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.markdown('<div class="fc-card"><h4>🐍 Python</h4><p>Core programming language.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-card"><h4>📊 Pandas</h4><p>Data manipulation &amp; analysis.</p></div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="fc-card"><h4>🤖 Machine Learning</h4><p>Predictive risk modeling.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-card"><h4>🔢 NumPy</h4><p>Numerical computing.</p></div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="fc-card"><h4>🧠 Deep Learning</h4><p>Advanced neural network models.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-card"><h4>⚡ Scikit-learn</h4><p>ML model training &amp; evaluation.</p></div>', unsafe_allow_html=True)

    with t4:
        st.markdown('<div class="fc-card"><h4>🎨 Streamlit</h4><p>Interactive web application framework.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-card"><h4>🔥 TensorFlow</h4><p>Deep learning framework.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">👨‍💻 Developed By</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="fc-card">
            <h4>AI-Powered Credit Risk Prediction System</h4>
            <p>Built to help financial institutions make faster, smarter, and more reliable lending decisions
            through AI-driven risk intelligence.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer-note">
        © AI Credit Risk Intelligence Platform · Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )