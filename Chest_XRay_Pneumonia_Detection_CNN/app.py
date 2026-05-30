import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from pathlib import Path
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="centered",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

/* ---------- root palette ---------- */
:root {
    --bg:         #030a10;
    --panel:      #07141e;
    --border:     #0e3a52;
    --glow:       #00d4ff;
    --glow-dim:   #0099bb;
    --danger:     #ff3b3b;
    --safe:       #00ff99;
    --text:       #c8e8f0;
    --muted:      #4a7a8a;
    --mono:       'Share Tech Mono', monospace;
    --sans:       'Rajdhani', sans-serif;
}

/* ---------- page background ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,180,255,.12), transparent),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,180,255,.03) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,180,255,.03) 40px);
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
footer { display: none !important; }

/* ---------- hide default streamlit chrome ---------- */
#MainMenu { visibility: hidden; }

/* ---------- main content width ---------- */
[data-testid="stMainBlockContainer"] {
    max-width: 780px;
    padding-top: 2rem;
}

/* ---------- scanline overlay ---------- */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,.18) 2px,
        rgba(0,0,0,.18) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ---------- header banner ---------- */
.header-block {
    text-align: center;
    padding: 2.4rem 1rem 1.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    position: relative;
}
.header-block::after {
    content: "";
    display: block;
    width: 60%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--glow), transparent);
    margin: 1rem auto 0;
}
.site-label {
    font-family: var(--mono);
    font-size: .72rem;
    letter-spacing: .3em;
    color: var(--glow-dim);
    text-transform: uppercase;
    margin-bottom: .5rem;
}
.site-title {
    font-family: var(--sans);
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: .06em;
    color: #fff;
    text-shadow: 0 0 30px rgba(0,212,255,.5);
    line-height: 1.1;
}
.site-title span { color: var(--glow); }
.site-sub {
    font-family: var(--mono);
    font-size: .78rem;
    color: var(--muted);
    margin-top: .6rem;
    letter-spacing: .12em;
}

/* ---------- upload zone ---------- */
[data-testid="stFileUploader"] {
    background: var(--panel) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
    transition: border-color .25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--glow) !important;
}
[data-testid="stFileUploaderDropzone"] p {
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: .82rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--text) !important;
}

/* ---------- xray display frame ---------- */
.xray-frame {
    background: #000;
    border: 1px solid var(--border);
    border-radius: 4px;
    box-shadow: 0 0 40px rgba(0,180,255,.08), inset 0 0 60px rgba(0,0,0,.8);
    padding: .5rem;
    position: relative;
    margin: 1.2rem 0;
}
.xray-frame::before {
    content: "CHEST · PA VIEW";
    font-family: var(--mono);
    font-size: .62rem;
    letter-spacing: .2em;
    color: var(--muted);
    position: absolute;
    top: 10px; left: 14px;
}
.xray-frame::after {
    content: "▶ ANALYZING";
    font-family: var(--mono);
    font-size: .62rem;
    letter-spacing: .15em;
    color: var(--glow-dim);
    position: absolute;
    top: 10px; right: 14px;
}

/* ---------- metric cards ---------- */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.4rem 0;
}
.metric-card {
    flex: 1;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--glow);
}
.metric-label {
    font-family: var(--mono);
    font-size: .62rem;
    letter-spacing: .2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: .35rem;
}
.metric-value {
    font-family: var(--sans);
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}

/* ---------- result banner ---------- */
.result-banner {
    border-radius: 5px;
    padding: 1.2rem 1.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-family: var(--sans);
    font-weight: 600;
    font-size: 1.15rem;
    letter-spacing: .06em;
    margin: 1.4rem 0;
    position: relative;
    overflow: hidden;
}
.result-banner::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(255,255,255,.04), transparent);
}
.result-danger {
    background: rgba(255,59,59,.1);
    border: 1px solid rgba(255,59,59,.5);
    color: #ff6b6b;
    box-shadow: 0 0 30px rgba(255,59,59,.1);
}
.result-safe {
    background: rgba(0,255,153,.08);
    border: 1px solid rgba(0,255,153,.4);
    color: #00ff99;
    box-shadow: 0 0 30px rgba(0,255,153,.08);
}
.result-icon { font-size: 1.6rem; }
.result-text { flex: 1; }
.result-title { font-size: 1.2rem; font-weight: 700; }
.result-sub {
    font-family: var(--mono);
    font-size: .72rem;
    opacity: .7;
    margin-top: .15rem;
    letter-spacing: .1em;
}

/* ---------- progress bar for confidence ---------- */
.conf-wrap { margin: 1.2rem 0; }
.conf-label-row {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: .7rem;
    color: var(--muted);
    margin-bottom: .4rem;
    letter-spacing: .12em;
}
.conf-track {
    background: #0a1a25;
    border: 1px solid var(--border);
    border-radius: 2px;
    height: 10px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 1s ease;
}
.conf-fill-danger { background: linear-gradient(90deg, #c00, var(--danger)); }
.conf-fill-safe   { background: linear-gradient(90deg, #009955, var(--safe)); }

/* ---------- info grid ---------- */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .6rem;
    margin: 1.4rem 0;
}
.info-cell {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: .7rem 1rem;
}
.info-cell-label {
    font-family: var(--mono);
    font-size: .6rem;
    letter-spacing: .18em;
    color: var(--muted);
    margin-bottom: .2rem;
}
.info-cell-val {
    font-family: var(--mono);
    font-size: .88rem;
    color: var(--glow);
}

/* ---------- section divider ---------- */
.sec-divider {
    display: flex;
    align-items: center;
    gap: .8rem;
    margin: 1.8rem 0 1rem;
}
.sec-divider-label {
    font-family: var(--mono);
    font-size: .65rem;
    letter-spacing: .25em;
    color: var(--muted);
    white-space: nowrap;
}
.sec-divider-line { flex: 1; height: 1px; background: var(--border); }

/* ---------- status ticker ---------- */
.ticker {
    font-family: var(--mono);
    font-size: .68rem;
    color: var(--glow-dim);
    letter-spacing: .12em;
    text-align: center;
    padding: .5rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
    opacity: .7;
}

/* ---------- streamlit image caption ---------- */
[data-testid="stImageCaption"] {
    font-family: var(--mono) !important;
    font-size: .65rem !important;
    color: var(--muted) !important;
    letter-spacing: .15em !important;
    text-align: center !important;
}

/* ---------- hide streamlit alerts, use custom ---------- */
[data-testid="stAlert"] { display: none !important; }

/* ---------- button ---------- */
[data-testid="stFileUploaderDeleteBtn"] button {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <div class="site-label">Radiology · AI Diagnostic Suite v2.4</div>
    <div class="site-title">PNEUMO<span>SCAN</span></div>
    <div class="site-sub">CHEST X-RAY ANALYSIS &nbsp;·&nbsp; DEEP NEURAL NETWORK &nbsp;·&nbsp; BINARY CLASSIFICATION</div>
</div>
""", unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        BASE_DIR = Path(__file__).resolve().parent
        model_path = BASE_DIR / "pneumonia_model.h5"
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.markdown("""
    <div class="result-banner result-danger">
        <div class="result-icon">⚠</div>
        <div class="result-text">
            <div class="result-title">MODEL NOT FOUND</div>
            <div class="result-sub">Place pneumonia_model.h5 in the app directory · Demo mode active</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-divider">
    <div class="sec-divider-label">01 · INPUT</div>
    <div class="sec-divider-line"></div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"],
    help="Supports JPEG and PNG · Recommended: posteroanterior (PA) view",
    label_visibility="collapsed",
)

st.markdown("""
<div style="font-family:var(--mono,monospace);font-size:.65rem;color:#4a7a8a;
            letter-spacing:.18em;text-align:center;margin-top:.4rem;">
    ↑ &nbsp; DRAG & DROP OR CLICK TO UPLOAD &nbsp; · &nbsp; JPG / PNG &nbsp; · &nbsp; PA VIEW PREFERRED
</div>
""", unsafe_allow_html=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file)
    w, h = image.size

    st.markdown("""
    <div class="sec-divider">
        <div class="sec-divider-label">02 · RADIOGRAPH</div>
        <div class="sec-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="xray-frame">', unsafe_allow_html=True)
    st.image(image, use_container_width=True, caption=f"FILE · {uploaded_file.name.upper()}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Image meta
    mode_map = {"L": "GRAYSCALE", "RGB": "RGB", "RGBA": "RGBA"}
    img_mode = mode_map.get(image.mode, image.mode)
    st.markdown(f"""
    <div class="info-grid">
        <div class="info-cell">
            <div class="info-cell-label">RESOLUTION</div>
            <div class="info-cell-val">{w} × {h} px</div>
        </div>
        <div class="info-cell">
            <div class="info-cell-label">COLOR MODE</div>
            <div class="info-cell-val">{img_mode}</div>
        </div>
        <div class="info-cell">
            <div class="info-cell-label">INPUT TARGET</div>
            <div class="info-cell-val">224 × 224 px</div>
        </div>
        <div class="info-cell">
            <div class="info-cell-label">FILE NAME</div>
            <div class="info-cell-val">{uploaded_file.name[:22].upper()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pre-process ──────────────────────────────────────────────────────────
    img_resized = image.resize((224, 224))
    img_arr = np.array(img_resized) / 255.0
    if len(img_arr.shape) == 2:
        img_arr = np.stack((img_arr,) * 3, axis=-1)
    elif img_arr.shape[2] == 4:
        img_arr = img_arr[:, :, :3]
    img_input = np.expand_dims(img_arr, axis=0)

    # ── Inference ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-divider">
        <div class="sec-divider-label">03 · DIAGNOSTIC RESULT</div>
        <div class="sec-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(""):
        placeholder = st.empty()
        placeholder.markdown("""
        <div style="font-family:var(--mono,monospace);font-size:.75rem;color:#0099bb;
                    letter-spacing:.2em;text-align:center;padding:1rem;">
            ◈ &nbsp; RUNNING INFERENCE &nbsp; · &nbsp; PLEASE WAIT…
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.6)

        if model is not None:
            pred = model.predict(img_input, verbose=0)[0][0]
        else:
            pred = float(np.random.uniform(0.05, 0.95))

        placeholder.empty()

    pneumonia = pred > 0.5
    confidence = float(pred if pneumonia else 1 - pred)
    conf_pct = confidence * 100

    # Result banner
    if pneumonia:
        st.markdown(f"""
        <div class="result-banner result-danger">
            <div class="result-icon">⚡</div>
            <div class="result-text">
                <div class="result-title">PNEUMONIA DETECTED</div>
                <div class="result-sub">Abnormal opacities identified · Clinical correlation recommended</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-banner result-safe">
            <div class="result-icon">✓</div>
            <div class="result-text">
                <div class="result-title">NORMAL — NO PNEUMONIA</div>
                <div class="result-sub">No significant pulmonary opacities detected</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Confidence bar
    fill_class = "conf-fill-danger" if pneumonia else "conf-fill-safe"
    st.markdown(f"""
    <div class="conf-wrap">
        <div class="conf-label-row">
            <span>CONFIDENCE SCORE</span>
            <span>{conf_pct:.2f}%</span>
        </div>
        <div class="conf-track">
            <div class="conf-fill {fill_class}" style="width:{conf_pct:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    risk = "HIGH" if pneumonia else "LOW"
    risk_color = "#ff6b6b" if pneumonia else "#00ff99"
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">RAW SCORE</div>
            <div class="metric-value">{pred:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">CONFIDENCE</div>
            <div class="metric-value">{conf_pct:.1f}<span style="font-size:1rem;opacity:.6">%</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">RISK LEVEL</div>
            <div class="metric-value" style="color:{risk_color}">{risk}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer ticker
    st.markdown("""
    <div class="ticker">
        ⚠ &nbsp; FOR INVESTIGATIONAL USE ONLY &nbsp;·&nbsp; NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL DIAGNOSIS
        &nbsp;·&nbsp; CONSULT A QUALIFIED RADIOLOGIST FOR CLINICAL DECISIONS
    </div>
    """, unsafe_allow_html=True)

else:
    # Idle state illustration
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 2rem;opacity:.5;">
        <div style="font-size:5rem;line-height:1;margin-bottom:1rem;
                    filter:drop-shadow(0 0 20px rgba(0,212,255,.3))">🫁</div>
        <div style="font-family:var(--mono,monospace);font-size:.72rem;
                    letter-spacing:.22em;color:#4a7a8a;">
            AWAITING X-RAY UPLOAD
        </div>
    </div>
    """, unsafe_allow_html=True)