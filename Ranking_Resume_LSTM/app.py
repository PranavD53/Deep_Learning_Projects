import os
import re
import pickle
import pathlib

import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pdfplumber
import plotly.graph_objects as go
import plotly.express as px
from docx import Document
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────
#  PATHS  (all relative to this script's folder)
# ─────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "resume_lstm.h5"
TOK_PATH   = BASE_DIR / "tokenizer.pkl"
ENC_PATH   = BASE_DIR / "label_encoder.pkl"

MAX_LEN = 300

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ · AI Ranking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  (dark editorial theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,300&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #0a0a0f;
    --surface:  #13131c;
    --card:     #1a1a28;
    --border:   #2a2a40;
    --accent:   #c8f23a;       /* electric lime */
    --accent2:  #4f7cff;       /* cobalt blue   */
    --accent3:  #ff6b6b;       /* coral         */
    --text:     #e8e8f0;
    --muted:    #6b6b8a;
    --success:  #3affa0;
}

/* ── Base reset ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

/* ── Remove streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #151530 50%, #0d0d1a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(200,242,58,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(79,124,255,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 6vw, 5.5rem);
    letter-spacing: 0.03em;
    line-height: 1;
    color: var(--text);
    margin: 0 0 0.5rem;
}
.hero h1 span { color: var(--accent); }
.hero p {
    font-size: 1.05rem;
    color: var(--muted);
    max-width: 520px;
    margin: 0;
    line-height: 1.6;
}

/* ── Section heading ── */
.section-head {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 0.08em;
    color: var(--text);
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 0.5rem;
}

/* ── Upload zone ── */
.upload-zone {
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--accent); }

[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.25s ease;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Primary button ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 2.5rem !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(200,242,58,0.3) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Metric cards ── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-card .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: var(--accent);
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* ── Winner card ── */
.winner-card {
    background: linear-gradient(135deg, #1a2810 0%, #1a2815 100%);
    border: 1px solid rgba(200,242,58,0.4);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.winner-card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(200,242,58,0.15) 0%, transparent 70%);
}
.winner-card .role-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    color: var(--accent);
    letter-spacing: 0.05em;
    line-height: 1;
}
.winner-card .role-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.winner-card .score-badge {
    background: var(--accent);
    color: #0a0a0f;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    padding: 0.6rem 1.4rem;
    border-radius: 10px;
    letter-spacing: 0.04em;
    flex-shrink: 0;
}

/* ── ATS score bar ── */
.ats-bar-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 2rem;
}
.ats-bar-label {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.8rem;
}
.ats-bar-label .title {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: var(--text);
}
.ats-bar-label .pct {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--accent);
}
.ats-track {
    background: var(--border);
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
}
.ats-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    transition: width 1s ease;
}
.ats-tiers {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
}
.ats-tiers span {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.12em;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Tooltip chip ── */
.chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent2);
    background: rgba(79,124,255,0.12);
    border: 1px solid rgba(79,124,255,0.3);
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    letter-spacing: 0.1em;
    margin-right: 0.3rem;
}

/* ── Rank badge ── */
.rank-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem 1.2rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s, background 0.2s;
}
.rank-row:hover {
    border-color: var(--accent);
    background: rgba(200,242,58,0.04);
}
.rank-badge {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    color: var(--muted);
    width: 2rem;
    text-align: center;
    flex-shrink: 0;
}
.rank-badge.gold { color: var(--accent); }
.rank-badge.silver { color: #a8a8cc; }
.rank-badge.bronze { color: #c87533; }
.rank-role {
    flex: 1;
    font-weight: 500;
    font-size: 0.95rem;
}
.rank-bar-bg {
    flex: 2;
    background: var(--border);
    border-radius: 99px;
    height: 6px;
    overflow: hidden;
}
.rank-bar-fill {
    height: 100%;
    border-radius: 99px;
}
.rank-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent);
    width: 4rem;
    text-align: right;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_files():
    model   = tf.keras.models.load_model(str(MODEL_PATH))
    with open(TOK_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(ENC_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, tokenizer, encoder


try:
    model, tokenizer, encoder = load_model_files()
    model_ready = True
except Exception as e:
    model_ready = False
    model_error = str(e)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text


def read_docx(file) -> str:
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


def predict_resume(text: str) -> np.ndarray:
    cleaned = clean_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    return model.predict(padded, verbose=0)[0]


def ats_grade(score: float) -> tuple[str, str]:
    if score >= 85:
        return "Excellent", "#3affa0"
    elif score >= 70:
        return "Good", "#c8f23a"
    elif score >= 50:
        return "Average", "#ffa03a"
    else:
        return "Low", "#ff6b6b"


BAR_COLORS = [
    "#c8f23a", "#4f7cff", "#ff6b6b", "#3affa0", "#ffa03a",
    "#c87bff", "#ff9de2", "#5fd4f4", "#ffdc3a", "#3ad4ff",
]

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">// AI-Powered · NLP · Deep Learning</div>
  <h1>RESUME<span>IQ</span></h1>
  <p>Upload your résumé and our LSTM model instantly ranks it across every job category — with ATS scoring, probability breakdown, and full analytics.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL STATUS BANNER
# ─────────────────────────────────────────────
if not model_ready:
    st.error(
        f"⚠️ **Model files not found.** "
        f"Ensure `resume_lstm.h5`, `tokenizer.pkl`, and `label_encoder.pkl` "
        f"live in the same folder as `app.py`.\n\n`{model_error}`"
    )
    st.stop()

# ─────────────────────────────────────────────
#  LAYOUT: two columns
# ─────────────────────────────────────────────
left, right = st.columns([1, 1.65], gap="large")

# ── LEFT: Upload panel ──────────────────────
with left:
    st.markdown('<div class="section-head">📁 Upload Résumé</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["pdf", "docx", "txt"],
        label_visibility="visible",
    )

    resume_text = ""

    if uploaded_file is not None:
        name = uploaded_file.name
        ext  = pathlib.Path(name).suffix.lower()

        with st.spinner("Reading file…"):
            try:
                if ext == ".pdf":
                    resume_text = read_pdf(uploaded_file)
                elif ext == ".docx":
                    resume_text = read_docx(uploaded_file)
                else:
                    resume_text = uploaded_file.read().decode("utf-8", errors="replace")
            except Exception as e:
                st.error(f"Could not parse file: {e}")

        if resume_text.strip():
            st.success(f"✅ **{name}** loaded successfully")

            words = len(resume_text.split())
            chars = len(resume_text)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">Words</div>
                  <div class="value">{words:,}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">Chars</div>
                  <div class="value">{chars:,}</div>
                </div>""", unsafe_allow_html=True)

            with st.expander("👁 Preview résumé text"):
                st.text(resume_text[:4000] + ("…" if len(resume_text) > 4000 else ""))
        else:
            st.warning("File appears to be empty or unreadable.")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🎯 Analyze Résumé", disabled=not bool(resume_text.strip()))

    # Tips
    st.markdown("""
    <br>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1.4rem;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--muted);letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.7rem;">Tips for best results</div>
      <ul style="color:var(--muted);font-size:0.85rem;margin:0;padding-left:1.2rem;line-height:1.9;">
        <li>Use text-selectable PDFs (not scanned images)</li>
        <li>Include skills, experience, and education sections</li>
        <li>Longer résumés give the model richer signals</li>
        <li>Avoid excessive formatting symbols or tables</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT: Results panel ────────────────────
with right:
    st.markdown('<div class="section-head">📊 Analysis Results</div>', unsafe_allow_html=True)

    if not analyze_clicked:
        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:16px;
                    padding:3rem 2rem;text-align:center;color:var(--muted);">
          <div style="font-size:3rem;margin-bottom:1rem;">🎯</div>
          <div style="font-family:'DM Sans',sans-serif;font-size:1rem;line-height:1.6;">
            Upload a résumé on the left<br>then click <strong style="color:var(--accent)">Analyze Résumé</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Running LSTM inference…"):
            probs       = predict_resume(resume_text)
            top_indices = np.argsort(probs)[::-1]
            classes     = encoder.inverse_transform(top_indices)
            scores      = probs[top_indices] * 100

        best_role  = classes[0]
        best_score = scores[0]
        ats_score  = min(100, float(best_score))
        grade, grade_color = ats_grade(ats_score)

        # ── Winner card ──────────────────────
        st.markdown(f"""
        <div class="winner-card">
          <div>
            <div class="role-label">🎯 Top Matched Role</div>
            <div class="role-name">{best_role}</div>
          </div>
          <div class="score-badge">{best_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ── ATS bar ──────────────────────────
        st.markdown(f"""
        <div class="ats-bar-wrap" style="margin-bottom:1.5rem;">
          <div class="ats-bar-label">
            <div class="title">⭐ ATS Match Score
              <span style="margin-left:0.5rem;font-size:0.78rem;
                           color:{grade_color};font-weight:700;">{grade}</span>
            </div>
            <div class="pct">{ats_score:.0f}<span style="font-size:1rem;">%</span></div>
          </div>
          <div class="ats-track">
            <div class="ats-fill" style="width:{ats_score}%;"></div>
          </div>
          <div class="ats-tiers">
            <span>Low</span><span>Average</span><span>Good</span><span>Excellent</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top-5 ranked list ─────────────────
        st.markdown('<div class="section-head" style="margin-top:0;">🏆 Top 5 Role Rankings</div>', unsafe_allow_html=True)

        badge_classes = ["gold", "silver", "bronze", "", ""]
        rank_symbols  = ["#1", "#2", "#3", "#4", "#5"]
        top5_roles    = classes[:5]
        top5_scores   = scores[:5]
        max_score     = top5_scores[0] if top5_scores[0] > 0 else 1

        for i, (role, sc) in enumerate(zip(top5_roles, top5_scores)):
            bar_w  = sc / max_score * 100
            color  = BAR_COLORS[i % len(BAR_COLORS)]
            bc     = badge_classes[i]
            rank_n = rank_symbols[i]
            st.markdown(f"""
            <div class="rank-row">
              <div class="rank-badge {bc}">{rank_n}</div>
              <div class="rank-role">{role}</div>
              <div class="rank-bar-bg">
                <div class="rank-bar-fill" style="width:{bar_w}%;background:{color};"></div>
              </div>
              <div class="rank-score">{sc:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Plotly polar chart ────────────────
        st.markdown('<div class="section-head">🕸 Confidence Radar</div>', unsafe_allow_html=True)

        top_n  = min(10, len(classes))
        r_vals = list(scores[:top_n]) + [scores[0]]
        theta  = list(classes[:top_n]) + [classes[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=r_vals,
            theta=theta,
            fill="toself",
            fillcolor="rgba(200,242,58,0.10)",
            line=dict(color="#c8f23a", width=2),
            marker=dict(color="#c8f23a", size=5),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#13131c",
                radialaxis=dict(
                    visible=True,
                    range=[0, max(scores[:top_n]) * 1.15],
                    gridcolor="#2a2a40",
                    tickfont=dict(color="#6b6b8a", size=9, family="JetBrains Mono"),
                    linecolor="#2a2a40",
                ),
                angularaxis=dict(
                    gridcolor="#2a2a40",
                    tickfont=dict(color="#a8a8cc", size=9, family="DM Sans"),
                    linecolor="#2a2a40",
                ),
            ),
            paper_bgcolor="#0a0a0f",
            plot_bgcolor="#0a0a0f",
            font=dict(color="#e8e8f0"),
            margin=dict(t=30, b=20, l=60, r=60),
            height=350,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Full probability bar chart ────────
        st.markdown('<div class="section-head">📈 All Category Probabilities</div>', unsafe_allow_html=True)

        full_df = pd.DataFrame({
            "Category":    encoder.classes_,
            "Probability": probs * 100,
        }).sort_values("Probability", ascending=False)

        fig_bar = go.Figure(go.Bar(
            x=full_df["Probability"],
            y=full_df["Category"],
            orientation="h",
            marker=dict(
                color=full_df["Probability"],
                colorscale=[[0, "#4f7cff"], [0.5, "#c8f23a"], [1, "#3affa0"]],
                showscale=False,
                line=dict(width=0),
            ),
            text=[f"{v:.1f}%" for v in full_df["Probability"]],
            textposition="outside",
            textfont=dict(color="#6b6b8a", size=9, family="JetBrains Mono"),
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0a0a0f",
            plot_bgcolor="#13131c",
            font=dict(color="#e8e8f0", family="DM Sans"),
            xaxis=dict(
                gridcolor="#2a2a40",
                tickfont=dict(color="#6b6b8a", size=9),
                showgrid=True,
                zeroline=False,
                title=dict(text="Confidence (%)", font=dict(color="#6b6b8a")),
            ),
            yaxis=dict(
                tickfont=dict(color="#a8a8cc", size=10),
                categoryorder="total ascending",
            ),
            margin=dict(t=10, b=40, l=140, r=70),
            height=max(350, len(encoder.classes_) * 28),
            bargap=0.35,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Full table ────────────────────────
        with st.expander("📋 Full probability table"):
            styled = full_df.reset_index(drop=True)
            styled.index = styled.index + 1
            st.dataframe(
                styled.style.format({"Probability": "{:.3f}%"})
                    .background_gradient(subset=["Probability"], cmap="YlGn"),
                use_container_width=True,
            )

# ── Footer ───────────────────────────────────
st.markdown("""
<hr style="margin-top:3rem;">
<div style="text-align:center;color:var(--muted);font-size:0.78rem;
            font-family:'JetBrains Mono',monospace;padding-bottom:1rem;">
  ResumeIQ · LSTM + Bidirectional NLP · Built with Streamlit &amp; TensorFlow
</div>
""", unsafe_allow_html=True)