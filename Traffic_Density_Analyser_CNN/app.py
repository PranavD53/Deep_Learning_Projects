import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import time
import base64
from io import BytesIO

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Traffic Sign Detector",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS — Road-Sign Theme
# ---------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@400;500&display=swap');

/* ── ROOT PALETTE ── */
:root {
    --asphalt:       #1a1c1e;
    --asphalt-mid:   #22252a;
    --asphalt-light: #2e3138;
    --lane-yellow:   #f5c518;
    --stop-red:      #c0392b;
    --go-green:      #27ae60;
    --sign-white:    #eef0f2;
    --muted:         #8a909a;
    --border:        #383d47;
}

/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: var(--asphalt) !important;
    color: var(--sign-white) !important;
}

/* Road-stripe top banner */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 6px;
    background: repeating-linear-gradient(
        90deg,
        var(--lane-yellow) 0px,
        var(--lane-yellow) 40px,
        transparent 40px,
        transparent 70px
    );
    z-index: 9999;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background-color: var(--asphalt-mid) !important;
    border-right: 2px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--sign-white) !important; }

/* ── MAIN AREA ── */
[data-testid="stAppViewContainer"] > .main {
    background-color: var(--asphalt) !important;
}
[data-testid="block-container"] {
    padding-top: 2rem !important;
}

/* ── HERO TITLE ── */
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(2.5rem, 6vw, 5rem);
    letter-spacing: 0.12em;
    line-height: 1;
    color: var(--lane-yellow);
    text-shadow: 3px 3px 0px rgba(0,0,0,0.5);
    margin: 0;
}
.hero-sub {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.25rem;
}

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploader"] {
    background: var(--asphalt-light) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 8px !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--lane-yellow) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── CARDS ── */
.sign-card {
    background: var(--asphalt-mid);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    height: 100%;
}
.sign-card h3 {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── RESULT BADGE ── */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: var(--asphalt-light);
    border-left: 5px solid var(--lane-yellow);
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.4rem;
    margin: 0.75rem 0;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 0.08em;
    color: var(--sign-white);
    width: 100%;
}
.result-badge.stop   { border-left-color: var(--stop-red); }
.result-badge.speed  { border-left-color: var(--lane-yellow); }
.result-badge.noentry{ border-left-color: var(--muted); }

/* ── CONFIDENCE BAR ── */
.conf-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.conf-track {
    background: var(--asphalt-light);
    border-radius: 4px;
    height: 12px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--go-green), var(--lane-yellow));
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}
.conf-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    letter-spacing: 0.05em;
    color: var(--lane-yellow);
    margin-top: 0.5rem;
}

/* ── STAT TILES ── */
.stat-tile {
    background: var(--asphalt-mid);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1rem;
    text-align: center;
}
.stat-tile .stat-icon { font-size: 2rem; }
.stat-tile .stat-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    letter-spacing: 0.05em;
    color: var(--lane-yellow);
    line-height: 1;
    margin: 0.25rem 0;
}
.stat-tile .stat-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── DIVIDER ── */
.road-divider {
    height: 2px;
    background: repeating-linear-gradient(
        90deg,
        var(--border) 0px,
        var(--border) 30px,
        transparent 30px,
        transparent 50px
    );
    margin: 2.5rem 0;
}

/* ── SECTION HEADER ── */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 0.12em;
    color: var(--sign-white);
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── TABLE ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] thead th {
    background: var(--asphalt-light) !important;
    color: var(--muted) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
}
[data-testid="stDataFrame"] tbody td {
    color: var(--sign-white) !important;
}

/* ── CHART AREA ── */
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
    background: var(--asphalt-mid) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    border: 1px solid var(--border) !important;
}

/* ── SIDEBAR NAV ── */
.sidebar-nav-item {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.6rem 0;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
}
.sidebar-badge {
    display: inline-block;
    background: var(--lane-yellow);
    color: #000 !important;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.05em;
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    float: right;
}

/* ── STREAMLIT OVERRIDES ── */
.stButton > button {
    background: var(--lane-yellow) !important;
    color: #000 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }

div[data-testid="stMetric"] {
    background: var(--asphalt-mid) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    border: 1px solid var(--border) !important;
}
div[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--lane-yellow) !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.5rem !important;
}

/* Streamlit alerts */
.stSuccess { background: rgba(39,174,96,0.15) !important; border-left: 4px solid var(--go-green) !important; }
.stWarning { background: rgba(245,197,24,0.12) !important; border-left: 4px solid var(--lane-yellow) !important; }
.stInfo    { background: rgba(138,144,154,0.12) !important; border-left: 4px solid var(--muted) !important; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* Caption */
.footer-caption {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    text-align: center;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    padding-top: 1rem;
}

/* History log */
.history-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--asphalt-light);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
}
.history-label { color: var(--sign-white); letter-spacing: 0.05em; }
.history-conf  { color: var(--lane-yellow); font-weight: 700; }
.history-time  { color: var(--muted); font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_cnn_model():
    return load_model("traffic_sign_model.h5")

try:
    model = load_cnn_model()
    model_loaded = True
except Exception:
    model = None
    model_loaded = False


# ---------------------------------------------------
# CLASS CONFIG — signs with metadata
# ---------------------------------------------------

SIGNS = {
    0: {
        "label": "Speed Limit",
        "icon": "🚗",
        "color": "#f5c518",
        "css_class": "speed",
        "description": "Indicates the maximum speed allowed on this road section.",
        "action": "Reduce speed to comply with the posted limit.",
        "risk": "Medium",
    },
    1: {
        "label": "Stop Sign",
        "icon": "🛑",
        "color": "#c0392b",
        "css_class": "stop",
        "description": "Requires all vehicles to come to a complete stop.",
        "action": "Stop completely before the line and check all directions.",
        "risk": "High",
    },
    2: {
        "label": "No Entry",
        "icon": "⛔",
        "color": "#8a909a",
        "css_class": "noentry",
        "description": "Prohibits entry of any vehicle into this road.",
        "action": "Do not enter. Find an alternative route.",
        "risk": "High",
    },
}


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

for key, default in [
    ("stop_count", 0),
    ("speed_count", 0),
    ("noentry_count", 0),
    ("history", []),
    ("total_scans", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

def predict_image(image: Image.Image):
    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=0)
    class_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    all_scores = {SIGNS[i]["label"]: float(prediction[0][i]) * 100
                  for i in range(len(SIGNS))}
    return class_idx, confidence, all_scores


def confidence_color(conf: float) -> str:
    if conf >= 80:
        return "#27ae60"
    elif conf >= 50:
        return "#f5c518"
    else:
        return "#c0392b"


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-size:3rem;">🚦</div>
        <div style="font-family:'Bebas Neue',sans-serif; font-size:1.5rem;
                    letter-spacing:0.12em; color:#f5c518;">
            TRAFFIC SIGN<br>DETECTOR
        </div>
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
                    letter-spacing:0.25em; color:#8a909a; text-transform:uppercase;">
            CNN-Powered System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model status
    if model_loaded:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Not Found")
        st.caption("Place `traffic_sign_model.h5` in the app directory.")

    st.markdown("---")

    # Session stats
    total = st.session_state.total_scans
    st.markdown(f"""
    <div style="font-family:'Barlow Condensed',sans-serif;
                font-size:0.8rem; letter-spacing:0.2em;
                text-transform:uppercase; color:#8a909a; margin-bottom:0.5rem;">
        Session Stats
    </div>
    <div class="sidebar-nav-item">
        Total Scans <span class="sidebar-badge">{total}</span>
    </div>
    <div class="sidebar-nav-item">
        🛑 Stop Signs <span class="sidebar-badge">{st.session_state.stop_count}</span>
    </div>
    <div class="sidebar-nav-item">
        🚗 Speed Limits <span class="sidebar-badge">{st.session_state.speed_count}</span>
    </div>
    <div class="sidebar-nav-item">
        ⛔ No Entry <span class="sidebar-badge">{st.session_state.noentry_count}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Detection sensitivity threshold
    threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=10, max_value=95, value=50, step=5,
        help="Results below this threshold are flagged as uncertain."
    )

    st.markdown("---")

    if st.button("🗑 Reset Session", use_container_width=True):
        for k in ["stop_count", "speed_count", "noentry_count", "history", "total_scans"]:
            st.session_state[k] = 0 if k != "history" else []
        st.rerun()


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("""
<div style="margin-bottom:2rem;">
    <p class="hero-title">TRAFFIC SIGN<br>DETECTION SYSTEM</p>
    <p class="hero-sub">CNN · Real-Time Classification · Confidence Scoring</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# UPLOAD + PREDICTION
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Drop a traffic sign image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="sign-card">
            <h3>📸 Uploaded Image</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<div class="sign-card"><h3>🔍 Analysis Result</h3>', unsafe_allow_html=True)

        if not model_loaded:
            st.error("Model not loaded. Cannot run prediction.")
        else:
            with st.spinner("Scanning sign..."):
                time.sleep(0.4)  # slight drama
                class_idx, confidence, all_scores = predict_image(image)

            sign = SIGNS.get(class_idx, {"label": "Unknown", "icon": "❓",
                                          "css_class": "", "description": "",
                                          "action": "", "risk": "Unknown", "color": "#fff"})

            # Main result badge
            st.markdown(f"""
            <div class="result-badge {sign['css_class']}">
                {sign['icon']} &nbsp; {sign['label']}
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            bar_color = confidence_color(confidence)
            low_conf = confidence < threshold
            st.markdown(f"""
            <div class="conf-label">Confidence Score</div>
            <div class="conf-track">
                <div class="conf-fill"
                     style="width:{confidence:.1f}%;
                            background:linear-gradient(90deg,{bar_color},{bar_color}cc);">
                </div>
            </div>
            <div class="conf-value">{confidence:.1f}%</div>
            """, unsafe_allow_html=True)

            if low_conf:
                st.warning(f"⚠️ Confidence below threshold ({threshold}%). Result may be uncertain.")

            st.markdown("---")

            # Sign info card
            st.markdown(f"""
            <div style="background:var(--asphalt-light,#2e3138); border-radius:8px;
                        padding:1rem; margin-top:0.5rem;">
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.8rem;
                            letter-spacing:0.2em; text-transform:uppercase; color:#8a909a;">
                    Sign Information
                </div>
                <div style="margin-top:0.5rem; font-size:0.95rem; line-height:1.5;">
                    {sign['description']}
                </div>
                <div style="margin-top:0.75rem; font-family:'Barlow Condensed',sans-serif;
                            font-size:0.8rem; letter-spacing:0.15em; text-transform:uppercase;
                            color:#8a909a;">
                    Recommended Action
                </div>
                <div style="margin-top:0.25rem; font-size:0.95rem; color:#f5c518;">
                    {sign['action']}
                </div>
                <div style="margin-top:0.75rem; font-family:'Barlow Condensed',sans-serif;
                            font-size:0.8rem; letter-spacing:0.15em; text-transform:uppercase;
                            color:#8a909a;">
                    Risk Level
                </div>
                <div style="margin-top:0.25rem; font-size:0.95rem;
                            color:{sign.get('color','#fff')}; font-weight:600;">
                    {sign['risk']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Probability breakdown (full width)
    if model_loaded:
        st.markdown("<div class='road-divider'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Probability Breakdown</div>',
                    unsafe_allow_html=True)

        prob_cols = st.columns(len(SIGNS))
        for i, (idx, s) in enumerate(SIGNS.items()):
            score = all_scores.get(s["label"], 0)
            with prob_cols[i]:
                st.markdown(f"""
                <div class="stat-tile">
                    <div class="stat-icon">{s['icon']}</div>
                    <div class="stat-num">{score:.1f}%</div>
                    <div class="stat-name">{s['label']}</div>
                    <div style="margin-top:0.5rem; height:6px; background:#383d47;
                                border-radius:3px; overflow:hidden;">
                        <div style="height:100%; width:{score:.1f}%;
                                    background:{s['color']}; border-radius:3px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Update counters & history
        label = sign["label"]
        if label == "Stop Sign":
            st.session_state.stop_count += 1
        elif label == "Speed Limit":
            st.session_state.speed_count += 1
        elif label == "No Entry":
            st.session_state.noentry_count += 1

        st.session_state.total_scans += 1
        st.session_state.history.insert(0, {
            "Sign": f"{sign['icon']} {label}",
            "Confidence": f"{confidence:.1f}%",
            "Risk": sign["risk"],
            "Status": "⚠️ Low" if confidence < threshold else "✅ OK"
        })
        # Keep last 20 entries
        st.session_state.history = st.session_state.history[:20]


# ---------------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------------

st.markdown("<div class='road-divider'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-header">📈 Analytics Dashboard</div>',
            unsafe_allow_html=True)

# KPI tiles
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="stat-tile">
        <div class="stat-icon">📸</div>
        <div class="stat-num">{st.session_state.total_scans}</div>
        <div class="stat-name">Total Scans</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="stat-tile">
        <div class="stat-icon">🛑</div>
        <div class="stat-num">{st.session_state.stop_count}</div>
        <div class="stat-name">Stop Signs</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="stat-tile">
        <div class="stat-icon">🚗</div>
        <div class="stat-num">{st.session_state.speed_count}</div>
        <div class="stat-name">Speed Limits</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="stat-tile">
        <div class="stat-icon">⛔</div>
        <div class="stat-num">{st.session_state.noentry_count}</div>
        <div class="stat-name">No Entry</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Chart + History side-by-side
chart_col, hist_col = st.columns([1, 1], gap="large")

with chart_col:
    st.markdown('<div class="section-header" style="font-size:1.1rem;">Detection Distribution</div>',
                unsafe_allow_html=True)

    analytics_df = pd.DataFrame({
        "Traffic Sign": ["Stop Sign", "Speed Limit", "No Entry"],
        "Detections": [
            st.session_state.stop_count,
            st.session_state.speed_count,
            st.session_state.noentry_count,
        ]
    })

    st.bar_chart(
        analytics_df.set_index("Traffic Sign"),
        color="#f5c518",
        use_container_width=True,
        height=260
    )

with hist_col:
    st.markdown('<div class="section-header" style="font-size:1.1rem;">Detection History</div>',
                unsafe_allow_html=True)

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
            height=260
        )
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem;
                    color:#8a909a; font-family:'Barlow Condensed',sans-serif;
                    letter-spacing:0.15em; text-transform:uppercase; font-size:0.9rem;">
            No detections yet.<br>Upload an image to begin.
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("""
<div class="footer-caption">
    Smart Traffic Detection System &nbsp;|&nbsp; CNN + Streamlit
    &nbsp;|&nbsp; For research &amp; educational use only
</div>
""", unsafe_allow_html=True)