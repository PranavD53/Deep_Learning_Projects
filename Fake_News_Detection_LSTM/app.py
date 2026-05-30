import os
import re
import streamlit as st
import tensorflow as tf
import pickle
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "fake_news_lstm.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="TruthLens · News Verifier",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==========================================================
# CUSTOM CSS  –  warm brown & cream palette
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --cream-0:   #fdf6ec;
    --cream-1:   #f5e9d3;
    --cream-2:   #ecdcbc;
    --brown-1:   #c8a97e;
    --brown-2:   #a07850;
    --brown-3:   #7a5530;
    --brown-4:   #4e3318;
    --ink:       #2c1a0e;
    --rust:      #9b3a1a;
    --rust-dim:  #c45e38;
    --forest:    #2a5c3f;
    --forest-lt: #d5ead9;
    --alert-bg:  #f9ede6;
}

html, body, [class*="css"] {
    background-color: var(--cream-0) !important;
    color: var(--ink) !important;
    font-family: 'Lato', sans-serif !important;
}

#MainMenu, header, footer { visibility: hidden; }
.block-container {
    padding-top: 1.8rem !important;
    max-width: 760px !important;
}

/* Masthead */
.masthead-wrap {
    text-align: center;
    padding: 0 0 18px;
    border-bottom: 2px solid var(--brown-2);
    margin-bottom: 28px;
}
.masthead-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--brown-2);
    margin-bottom: 8px;
}
.masthead-logo {
    font-family: 'Libre Baskerville', serif;
    font-size: clamp(2.6rem, 9vw, 4.8rem);
    font-weight: 700;
    color: var(--brown-4);
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0;
}
.masthead-logo span { color: var(--brown-2); }
.masthead-tagline {
    font-family: 'Libre Baskerville', serif;
    font-style: italic;
    font-size: 0.82rem;
    color: var(--brown-2);
    margin-top: 6px;
}

/* Section labels */
.field-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--brown-2);
    margin-bottom: 6px;
    display: block;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background-color: var(--cream-1) !important;
    border: 1.5px solid var(--brown-1) !important;
    border-radius: 4px !important;
    color: var(--ink) !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.94rem !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: var(--brown-3) !important;
}

/* Textarea */
textarea {
    background-color: var(--cream-1) !important;
    border: 1.5px solid var(--brown-1) !important;
    border-radius: 4px !important;
    color: var(--ink) !important;
    font-family: 'Libre Baskerville', serif !important;
    font-size: 0.93rem !important;
    line-height: 1.8 !important;
}
textarea:focus {
    border-color: var(--brown-3) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(122,85,48,0.12) !important;
}
textarea::placeholder { color: var(--brown-1) !important; font-style: italic; }

/* Word count */
.word-count {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--brown-2);
    text-align: right;
    margin-top: -4px;
}

/* Button */
div.stButton > button {
    background-color: var(--brown-3) !important;
    color: var(--cream-0) !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: .16em !important;
    text-transform: uppercase !important;
    padding: 0.72rem 1.8rem !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
    width: 100% !important;
}
div.stButton > button:hover {
    background-color: var(--brown-4) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

/* Verdict card */
.verdict-wrap {
    background: var(--cream-1);
    border: 1.5px solid var(--brown-1);
    border-radius: 6px;
    padding: 1.5rem 1.8rem 1.3rem;
    margin-top: 1.8rem;
    position: relative;
}
.verdict-top-bar {
    height: 4px;
    border-radius: 6px 6px 0 0;
    position: absolute;
    top: -1.5px; left: -1.5px; right: -1.5px;
}
.verdict-top-bar.real { background: var(--forest); }
.verdict-top-bar.fake { background: var(--rust); }

.verdict-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--brown-2);
    margin-bottom: 4px;
}
.verdict-headline {
    font-family: 'Libre Baskerville', serif;
    font-weight: 700;
    font-size: 2rem;
    line-height: 1.1;
    margin: 0 0 4px;
}
.verdict-headline.real { color: var(--forest); }
.verdict-headline.fake { color: var(--rust); }

.verdict-icon {
    display: inline-block;
    width: 28px; height: 28px;
    border-radius: 50%;
    text-align: center; line-height: 28px;
    font-size: 0.9rem;
    margin-right: 8px;
    vertical-align: middle;
    font-weight: 700;
}
.verdict-icon.real { background: var(--forest-lt); color: var(--forest); }
.verdict-icon.fake { background: var(--alert-bg);  color: var(--rust); }

.divider-rule { border: none; border-top: 1px solid var(--brown-1); margin: 1rem 0; }

.conf-row { display: flex; align-items: center; gap: 12px; }
.conf-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--brown-2);
    white-space: nowrap;
    min-width: 80px;
}
.conf-track {
    flex: 1; height: 7px;
    background: var(--cream-2);
    border-radius: 4px; overflow: hidden;
}
.conf-fill { height: 100%; border-radius: 4px; }
.conf-fill.real { background: var(--forest); }
.conf-fill.fake { background: var(--rust-dim); }
.conf-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem; font-weight: 500;
    color: var(--ink);
    min-width: 44px; text-align: right;
}

.verdict-note {
    font-family: 'Libre Baskerville', serif;
    font-style: italic;
    font-size: 0.8rem;
    color: var(--brown-3);
    line-height: 1.65;
    margin-top: 0.8rem;
}

.footer-text {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--brown-1);
    margin-top: 0.4rem;
}

hr { border-color: var(--brown-1) !important; opacity: 0.5 !important; }
.stAlert { border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL & TOKENIZER
# ==========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

model     = load_model()
tokenizer = load_tokenizer()

# ==========================================================
# HELPERS
# ==========================================================

MAX_LEN = 300

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def predict_news(text: str):
    seq    = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    score  = float(model.predict(padded, verbose=0)[0][0])
    if score > 0.5:
        return "real", score
    return "fake", 1.0 - score

# ==========================================================
# MASTHEAD
# ==========================================================

today = datetime.now().strftime("%A, %B %d, %Y").upper()

st.markdown(f"""
<div class="masthead-wrap">
    <div class="masthead-eyebrow">{today} &nbsp;·&nbsp; AI-Powered Credibility Check</div>
    <div class="masthead-logo">Truth<span>Lens</span></div>
    <div class="masthead-tagline">Separating fact from fiction, one article at a time</div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SAMPLES
# ==========================================================

SAMPLES = {
    "Federal Reserve Rate Decision":
        "The Federal Reserve announced that interest rates will remain unchanged following "
        "its latest policy meeting. Officials cited stable economic growth and moderating "
        "inflation as key reasons for maintaining the current rate. Financial markets "
        "responded positively to the decision.",
    "Coffee Enables Animal Communication":
        "Scientists have confirmed that drinking ten cups of coffee per day enables humans "
        "to communicate with animals. The breakthrough was reportedly discovered in a secret "
        "underground laboratory and is expected to change the world within weeks.",
    "AI Developer Conference Keynote":
        "A leading technology company unveiled its latest artificial intelligence platform "
        "during an annual developer conference. The company stated that the new system will "
        "improve productivity, automate repetitive tasks, and assist developers in writing "
        "software more efficiently.",
}

st.markdown('<span class="field-label">Try a sample</span>', unsafe_allow_html=True)
choice = st.selectbox(
    "Sample selector",
    ["— Paste your own article —"] + list(SAMPLES.keys()),
    label_visibility="collapsed",
)
default_text = SAMPLES.get(choice, "")

# ==========================================================
# TEXT AREA
# ==========================================================

st.markdown('<span class="field-label" style="margin-top:1.2rem;">Article text</span>',
            unsafe_allow_html=True)

news_input = st.text_area(
    "Article text",
    value=default_text,
    height=220,
    placeholder="Paste or type a news article here…",
    label_visibility="collapsed",
)

word_count = len(news_input.split()) if news_input.strip() else 0
st.markdown(
    f'<div class="word-count">{word_count} word{"s" if word_count != 1 else ""}</div>',
    unsafe_allow_html=True,
)

# ==========================================================
# BUTTON
# ==========================================================

st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
analyse = st.button("Analyse Article", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# VERDICT
# ==========================================================

if analyse:
    if not news_input.strip():
        st.warning("Please enter or select a news article before analysing.")
    else:
        with st.spinner("Scanning article…"):
            label, confidence = predict_news(news_input)

        pct   = round(confidence * 100, 1)
        bar_w = int(pct)

        if label == "real":
            headline  = "Credible"
            icon_char = "✓"
            note      = (
                "The model found patterns consistent with factual reporting — "
                "structured claims, measured tone, and verifiable detail. "
                "Always cross-check with primary sources before sharing."
            )
        else:
            headline  = "Suspect"
            icon_char = "✕"
            note      = (
                "The model detected patterns commonly linked to misleading content — "
                "sensationalist framing, unverified claims, or manipulative language. "
                "Treat with caution and verify through trusted sources."
            )

        st.markdown(f"""
        <div class="verdict-wrap">
            <div class="verdict-top-bar {label}"></div>
            <div class="verdict-label">Verdict</div>
            <div class="verdict-headline {label}">
                <span class="verdict-icon {label}">{icon_char}</span>{headline}
            </div>
            <hr class="divider-rule"/>
            <div class="conf-row">
                <span class="conf-label">Confidence</span>
                <div class="conf-track">
                    <div class="conf-fill {label}" style="width:{bar_w}%"></div>
                </div>
                <span class="conf-pct">{pct}%</span>
            </div>
            <p class="verdict-note">{note}</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown(
    '<p class="footer-text">TruthLens &nbsp;·&nbsp; LSTM Classifier &nbsp;·&nbsp; NLP &amp; Deep Learning</p>',
    unsafe_allow_html=True,
)