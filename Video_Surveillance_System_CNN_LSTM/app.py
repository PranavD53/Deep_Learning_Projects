import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent

IMG_SIZE = 128
SEQ_LEN = 20

CLASSES = [
    'Abuse', 'Arrest', 'Arson', 'Assault',
    'Burglary', 'Explosion', 'Fighting', 'NormalVideos',
    'Robbery', 'RoadAccidents', 'Shooting',
    'Shoplifting', 'Stealing', 'Vandalism'
]

THREAT_LEVEL = {
    'NormalVideos': ('LOW', '#00ff88', '🟢'),
    'Shoplifting':  ('MEDIUM', '#ffd700', '🟡'),
    'Stealing':     ('MEDIUM', '#ffd700', '🟡'),
    'Burglary':     ('HIGH', '#ff6b35', '🟠'),
    'Arrest':       ('HIGH', '#ff6b35', '🟠'),
    'Vandalism':    ('HIGH', '#ff6b35', '🟠'),
    'RoadAccidents':('HIGH', '#ff6b35', '🟠'),
    'Abuse':        ('CRITICAL', '#ff2d55', '🔴'),
    'Arson':        ('CRITICAL', '#ff2d55', '🔴'),
    'Assault':      ('CRITICAL', '#ff2d55', '🔴'),
    'Explosion':    ('CRITICAL', '#ff2d55', '🔴'),
    'Fighting':     ('CRITICAL', '#ff2d55', '🔴'),
    'Robbery':      ('CRITICAL', '#ff2d55', '🔴'),
    'Shooting':     ('CRITICAL', '#ff2d55', '🔴'),
}

# =========================
# STYLING
# =========================

st.set_page_config(
    page_title="SentinelAI — Surveillance System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

:root {
    --bg-primary:    #070b0f;
    --bg-panel:      #0d1117;
    --bg-card:       #111820;
    --border:        #1c2d3a;
    --border-bright: #1e4060;
    --accent:        #00c8ff;
    --accent-dim:    #005f80;
    --green:         #00ff88;
    --red:           #ff2d55;
    --amber:         #ffd700;
    --orange:        #ff6b35;
    --text-primary:  #c8dde8;
    --text-dim:      #4a6272;
    --text-label:    #6f9ab0;
    --mono:          'Share Tech Mono', monospace;
    --sans:          'Rajdhani', sans-serif;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary);
    font-family: var(--sans);
}

/* Grid scanline overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,200,255,0.012) 2px,
        rgba(0,200,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 0;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* Header bar */
.sentinel-header {
    background: linear-gradient(90deg, #050a10 0%, #091520 60%, #050a10 100%);
    border-bottom: 1px solid var(--border-bright);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
}
.sentinel-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.sentinel-logo {
    font-family: var(--mono);
    font-size: 1.5rem;
    color: var(--accent);
    letter-spacing: 0.15em;
    text-shadow: 0 0 20px rgba(0,200,255,0.6);
}
.sentinel-logo span { color: var(--text-dim); font-size: 0.85rem; display: block; letter-spacing: 0.3em; margin-top: 2px; }
.header-status {
    display: flex; gap: 24px; align-items: center;
}
.status-dot {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
}
.status-dot::before {
    content: '●';
    color: var(--green);
    margin-right: 6px;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* Panels */
.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px 24px;
    margin-bottom: 16px;
    position: relative;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-dim), var(--accent), var(--accent-dim));
    border-radius: 4px 4px 0 0;
}
.panel-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 0.3em;
    margin-bottom: 12px;
    text-transform: uppercase;
}

/* Upload zone */
[data-testid="stFileUploader"] > div {
    background: var(--bg-card) !important;
    border: 1px dashed var(--accent-dim) !important;
    border-radius: 4px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-label) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}

/* Video player */
video {
    border: 1px solid var(--border-bright) !important;
    border-radius: 4px !important;
    width: 100% !important;
}

/* Analyze button */
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.2em !important;
    padding: 12px 32px !important;
    border-radius: 2px !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
    width: 100% !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stButton"] > button:hover {
    background: rgba(0,200,255,0.08) !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.2) !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }
[data-testid="stSpinner"] p { font-family: var(--mono) !important; font-size: 0.8rem !important; color: var(--accent) !important; letter-spacing: 0.1em; }

/* Threat card */
.threat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px;
    text-align: center;
    position: relative;
}
.threat-level-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    font-weight: bold;
    margin-bottom: 8px;
}
.detected-class {
    font-family: var(--sans);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    line-height: 1.1;
}
.confidence-val {
    font-family: var(--mono);
    font-size: 1.1rem;
    color: var(--text-dim);
    margin-top: 4px;
}

/* Progress bars (probability) */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
    font-family: var(--mono);
    font-size: 0.72rem;
}
.prob-label {
    width: 130px;
    color: var(--text-label);
    flex-shrink: 0;
    text-align: right;
}
.prob-bar-bg {
    flex: 1;
    height: 6px;
    background: var(--bg-card);
    border-radius: 2px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.prob-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}
.prob-val {
    width: 48px;
    color: var(--text-dim);
    text-align: right;
}
.prob-row.top-result .prob-label { color: var(--accent); }
.prob-row.top-result .prob-val   { color: var(--accent); }

/* Grid divider */
.grid-div {
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0;
}

/* Info box */
.info-box {
    background: rgba(0,200,255,0.04);
    border: 1px solid var(--border-bright);
    border-left: 3px solid var(--accent);
    padding: 12px 16px;
    border-radius: 0 4px 4px 0;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-label);
    letter-spacing: 0.05em;
    line-height: 1.8;
}

/* Supported classes grid */
.classes-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 8px;
    margin-top: 8px;
}
.class-chip {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 6px 10px;
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text-label);
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Alert banner */
.alert-banner {
    padding: 12px 20px;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
    border-left: 3px solid;
}
.alert-critical { background: rgba(255,45,85,0.08);  border-color: var(--red);    color: var(--red); }
.alert-high     { background: rgba(255,107,53,0.08); border-color: var(--orange); color: var(--orange); }
.alert-medium   { background: rgba(255,215,0,0.08);  border-color: var(--amber);  color: var(--amber); }
.alert-low      { background: rgba(0,255,136,0.08);  border-color: var(--green);  color: var(--green); }

/* Streamlit column gaps */
[data-testid="stColumns"] { gap: 16px; }

/* Success message override */
[data-testid="stAlert"] {
    background: rgba(0,255,136,0.05) !important;
    border: 1px solid rgba(0,255,136,0.3) !important;
    color: var(--green) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():
    lstm_model = load_model(BASE_DIR / "crime_classifier.h5")
    cnn_model  = load_model(BASE_DIR / "cnn_feature_extractor.h5")
    return lstm_model, cnn_model

lstm_model, cnn_model = load_models()

# =========================
# FRAME EXTRACTION
# =========================

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // SEQ_LEN, 1)
    frames = []

    for i in range(SEQ_LEN):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) / 255.0
        frames.append(frame)

    cap.release()

    while len(frames) < SEQ_LEN:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(frames)

# =========================
# PREDICTION
# =========================

def predict_video(video_path):
    frames   = extract_frames(video_path)
    features = cnn_model.predict(frames, verbose=0)
    features = np.expand_dims(features, axis=0)
    prediction     = lstm_model.predict(features, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence      = float(np.max(prediction))
    return CLASSES[predicted_class], confidence, prediction[0]

# =========================
# HEADER
# =========================

st.markdown("""
<div class="sentinel-header">
    <div>
        <div class="sentinel-logo">⬡ SENTINEL<em style="color:#005f80">AI</em>
            <span>INTELLIGENT SURVEILLANCE DETECTION SYSTEM</span>
        </div>
    </div>
    <div class="header-status">
        <span class="status-dot">SYSTEM ONLINE</span>
        <span class="status-dot">CNN+LSTM ENGINE</span>
        <span class="status-dot">14 CLASSES</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# MAIN LAYOUT
# =========================

left_col, right_col = st.columns([1.1, 1], gap="medium")

with left_col:
    st.markdown('<div class="panel-label">// FOOTAGE INGESTION</div>', unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "UPLOAD CCTV / SECURITY FOOTAGE",
        type=["mp4", "avi", "mov"],
        label_visibility="visible"
    )

    if uploaded_video:
        st.video(uploaded_video)

        st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)

        analyze_clicked = st.button("⬡  INITIATE THREAT ANALYSIS")

        if analyze_clicked:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                temp_path = tmp.name

            with st.spinner("PROCESSING TEMPORAL SEQUENCES..."):
                pred_class, confidence, probs = predict_video(temp_path)

            os.remove(temp_path)

            threat, color, icon = THREAT_LEVEL.get(pred_class, ('UNKNOWN', '#00c8ff', '⬡'))

            # Store results in session state to persist them
            st.session_state["result"] = {
                "pred_class": pred_class,
                "confidence": confidence,
                "probs": probs,
                "threat": threat,
                "color": color,
                "icon": icon,
            }

    else:
        st.markdown("""
        <div class="info-box">
            ACCEPTED FORMATS  : MP4 · AVI · MOV<br>
            FRAME SAMPLING    : 20 FRAMES / SEQUENCE<br>
            INPUT RESOLUTION  : 128 × 128 px<br>
            DETECTION CLASSES : 14 ACTIVITY TYPES
        </div>
        """, unsafe_allow_html=True)

    # Supported classes
    st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">// DETECTABLE ACTIVITIES</div>', unsafe_allow_html=True)
    chips_html = '<div class="classes-grid">'
    for c in CLASSES:
        _, col, ico = THREAT_LEVEL.get(c, ('', '#4a6272', '⬡'))
        chips_html += f'<div class="class-chip"><span style="color:{col}">{ico}</span>{c}</div>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="panel-label">// ANALYSIS OUTPUT</div>', unsafe_allow_html=True)

    result = st.session_state.get("result")

    if result:
        pred_class = result["pred_class"]
        confidence = result["confidence"]
        probs      = result["probs"]
        threat     = result["threat"]
        color      = result["color"]
        icon       = result["icon"]

        # Alert banner
        alert_cls = {
            'CRITICAL': 'alert-critical',
            'HIGH':     'alert-high',
            'MEDIUM':   'alert-medium',
            'LOW':      'alert-low'
        }.get(threat, 'alert-low')

        st.markdown(f"""
        <div class="alert-banner {alert_cls}">
            {icon} THREAT LEVEL: {threat} — {pred_class.upper()} DETECTED
        </div>
        """, unsafe_allow_html=True)

        # Primary result card
        st.markdown(f"""
        <div class="threat-card" style="border-top: 2px solid {color};">
            <div class="threat-level-badge" style="background: {color}22; color: {color}; border: 1px solid {color}55;">
                {threat} THREAT
            </div>
            <div class="detected-class" style="color: {color};">{pred_class}</div>
            <div class="confidence-val">{confidence*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:20px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">// CLASS PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)

        # Sort by probability descending
        sorted_pairs = sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True)

        bars_html = ""
        for i, (cls, prob) in enumerate(sorted_pairs):
            _, bar_color, _ = THREAT_LEVEL.get(cls, ('', '#2a4a5a', ''))
            is_top = (cls == pred_class)
            row_cls = "prob-row top-result" if is_top else "prob-row"
            fill_pct = prob * 100
            fill_color = bar_color if is_top else "#1c3a4a"
            bars_html += f"""
            <div class="{row_cls}">
                <div class="prob-label">{cls}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{fill_pct:.1f}%; background:{fill_color};"></div>
                </div>
                <div class="prob-val">{prob*100:.1f}%</div>
            </div>"""

        st.markdown(bars_html, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            height: 380px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 1px dashed #1c2d3a;
            border-radius: 4px;
            background: #0d1117;
        ">
            <div style="font-family:'Share Tech Mono',monospace; font-size:2.5rem; color:#1c2d3a; margin-bottom:12px;">⬡</div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#2a4a5a; letter-spacing:0.25em;">
                AWAITING FOOTAGE INPUT
            </div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:#1c3040; letter-spacing:0.15em; margin-top:6px;">
                UPLOAD VIDEO TO BEGIN ANALYSIS
            </div>
        </div>
        """, unsafe_allow_html=True)