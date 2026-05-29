import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="🩸",
    layout="centered"
)

# =========================
# CUSTOM CSS — DEXTER RED THEME
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0404;
    color: #e8d5d5;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(160,10,10,0.22) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(100,0,0,0.15) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"] > div { position: relative; z-index: 1; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Title ── */
h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.4rem !important;
    letter-spacing: 0.12em;
    background: linear-gradient(135deg, #ff1a1a 0%, #c0392b 40%, #8b0000 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0 !important;
    line-height: 1.1 !important;
}

/* ── Subtitle / caption ── */
.subtitle {
    font-size: 0.78rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #7a3030;
    margin-top: 0.1rem;
    margin-bottom: 2.4rem;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #3a0a0a;
    margin: 1.6rem 0;
}

/* ── Section headers ── */
.section-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #c0392b;
    margin: 1.6rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #3a0a0a, transparent);
}

/* ── Labels ── */
label, [data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9a5a5a !important;
    font-weight: 500 !important;
}

/* ── Inputs, selects ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #130606 !important;
    border: 1px solid #3a0a0a !important;
    border-radius: 4px !important;
    color: #e8d5d5 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s;
}

[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #8b0000 !important;
    box-shadow: 0 0 0 2px rgba(139,0,0,0.18) !important;
    outline: none !important;
}

/* Dropdown arrow & options */
[data-testid="stSelectbox"] svg { color: #8b0000 !important; }
[data-baseweb="select"] [data-baseweb="popover"] {
    background: #130606 !important;
    border: 1px solid #3a0a0a !important;
}
[data-baseweb="menu"] li {
    background: #130606 !important;
    color: #e8d5d5 !important;
}
[data-baseweb="menu"] li:hover {
    background: #1f0a0a !important;
    color: #ff4444 !important;
}

/* Number input buttons */
[data-testid="stNumberInput"] button {
    background: #1f0a0a !important;
    border-color: #3a0a0a !important;
    color: #c0392b !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #8b0000 0%, #c0392b 50%, #8b0000 100%) !important;
    background-size: 200% 200% !important;
    border: none !important;
    color: #fff !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.25em !important;
    padding: 0.65rem 2.2rem !important;
    border-radius: 3px !important;
    cursor: pointer !important;
    transition: background-position 0.4s, box-shadow 0.3s, transform 0.15s !important;
    width: 100%;
    margin-top: 1rem;
}

[data-testid="stButton"] > button:hover {
    background-position: right center !important;
    box-shadow: 0 0 28px rgba(192,57,43,0.45) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    border-left-width: 3px !important;
}

/* Success */
[data-testid="stAlert"][data-baseweb="notification"][kind="positive"],
div[data-testid="stAlert"].st-emotion-cache-ztfqz8 {
    background: #0d1a0d !important;
    border-left-color: #2d7a2d !important;
}

/* Error */
[data-testid="stAlert"][data-baseweb="notification"][kind="negative"] {
    background: #1a0505 !important;
    border-left-color: #c0392b !important;
}

/* ── Columns gap ── */
[data-testid="stHorizontalBlock"] { gap: 1rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0404; }
::-webkit-scrollbar-thumb { background: #3a0a0a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #8b0000; }

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL + SCALER
# =========================
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("churn_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# =========================
# HEADER
# =========================
st.markdown("<h1>CHURN INTELLIGENCE</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Customer Attrition Risk · Neural Analysis</p>', unsafe_allow_html=True)

# =========================
# SECTION — IDENTITY
# =========================
st.markdown('<div class="section-label">Identity</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
with col2:
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
with col3:
    Partner = st.selectbox("Partner", ["Yes", "No"])

col4, col5 = st.columns([1, 2])
with col4:
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
with col5:
    tenure = st.number_input("Tenure (months)", 0, 100, step=1)

# =========================
# SECTION — SERVICES
# =========================
st.markdown('<div class="section-label">Services</div>', unsafe_allow_html=True)

col6, col7 = st.columns(2)
with col6:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
with col7:
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

col8, col9 = st.columns(2)
with col8:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
with col9:
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

col10, col11 = st.columns(2)
with col10:
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
with col11:
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

col12, col13 = st.columns(2)
with col12:
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
with col13:
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# =========================
# SECTION — BILLING
# =========================
st.markdown('<div class="section-label">Billing</div>', unsafe_allow_html=True)

col14, col15 = st.columns(2)
with col14:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
with col15:
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

col16, col17 = st.columns(2)
with col16:
    MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 500.0, step=0.01)
with col17:
    TotalCharges = st.text_input("Total Charges ($)", "0")

# =========================
# ENCODING
# =========================
def encode_value(val, mapping):
    return mapping.get(val, 0)

gender_map       = {"Female": 0, "Male": 1}
yes_no_map       = {"No": 0, "Yes": 1}
multiple_map     = {"No": 0, "Yes": 1, "No phone service": 2}
internet_map     = {"DSL": 0, "Fiber optic": 1, "No": 2}
service_map      = {"No": 0, "Yes": 1, "No internet service": 2}
contract_map     = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map      = {
    "Electronic check": 0, "Mailed check": 1,
    "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
}

try:
    total_charges = float(TotalCharges)
except:
    total_charges = 0.0

input_data = np.array([[
    encode_value(gender, gender_map),
    SeniorCitizen,
    encode_value(Partner, yes_no_map),
    encode_value(Dependents, yes_no_map),
    tenure,
    encode_value(PhoneService, yes_no_map),
    encode_value(MultipleLines, multiple_map),
    encode_value(InternetService, internet_map),
    encode_value(OnlineSecurity, service_map),
    encode_value(OnlineBackup, service_map),
    encode_value(DeviceProtection, service_map),
    encode_value(TechSupport, service_map),
    encode_value(StreamingTV, service_map),
    encode_value(StreamingMovies, service_map),
    encode_value(Contract, contract_map),
    encode_value(PaperlessBilling, yes_no_map),
    encode_value(PaymentMethod, payment_map),
    MonthlyCharges,
    total_charges
]])

# =========================
# PREDICTION
# =========================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("⬤  RUN ANALYSIS"):
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0][0]
    pct          = prediction * 100

    st.markdown("<hr>", unsafe_allow_html=True)

    if prediction > 0.5:
        st.error(
            f"**⚠ HIGH CHURN RISK — {pct:.1f}%**\n\n"
            f"This customer shows significant attrition signals. "
            f"Immediate retention action is recommended."
        )
    else:
        st.success(
            f"**✓ LOW CHURN RISK — {pct:.1f}%**\n\n"
            f"This customer appears stable. "
            f"Continue standard engagement protocols."
        )

    # Confidence bar
    bar_color = "#c0392b" if prediction > 0.5 else "#2d7a2d"
    st.markdown(f"""
    <div style="margin-top:1rem;">
        <div style="font-size:0.7rem;letter-spacing:0.2em;text-transform:uppercase;color:#7a3030;margin-bottom:0.4rem;">
            Risk Score
        </div>
        <div style="background:#130606;border:1px solid #3a0a0a;border-radius:3px;height:6px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:100%;background:{bar_color};
                        border-radius:3px;transition:width 0.8s ease;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.68rem;color:#5a2020;margin-top:0.3rem;letter-spacing:0.1em;">
            <span>0%</span><span>50%</span><span>100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)