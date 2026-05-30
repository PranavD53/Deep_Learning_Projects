import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="NeuralTrade · Stock Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS — Dark Trading Terminal Theme
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=Orbitron:wght@400;700;900&display=swap');

/* ── Root Variables ─────────────────────────────── */
:root {
    --bg-base:       #070b14;
    --bg-card:       #0d1421;
    --bg-elevated:   #111c2d;
    --border:        #1a2d4a;
    --border-bright: #1e3a5f;
    --accent-green:  #00d4aa;
    --accent-red:    #ff4d6d;
    --accent-blue:   #3b82f6;
    --accent-amber:  #f59e0b;
    --text-primary:  #e8f4fd;
    --text-secondary:#7a9bbf;
    --text-muted:    #3d5a7a;
    --glow-green:    0 0 20px rgba(0, 212, 170, 0.3);
    --glow-red:      0 0 20px rgba(255, 77, 109, 0.3);
    --glow-blue:     0 0 20px rgba(59, 130, 246, 0.2);
}

/* ── Global ─────────────────────────────────────── */
html, body, .stApp {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(26,45,74,0.15) 1px, transparent 1px),
        linear-gradient(90deg, rgba(26,45,74,0.15) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1220 0%, #070b14 100%) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

/* ── Header Brand ────────────────────────────────── */
.brand-header {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, #00d4aa 0%, #3b82f6 50%, #00d4aa 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s ease-in-out infinite;
    margin-bottom: 0.25rem;
}

@keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50%       { background-position: 100% 50%; }
}

.brand-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Section Headings ────────────────────────────── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* ── Metric Cards ────────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
}

.metric-card:hover {
    border-color: var(--border-bright);
    transform: translateY(-2px);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent-blue);
    border-radius: 2px 0 0 2px;
}

.metric-card.green::before { background: var(--accent-green); }
.metric-card.red::before   { background: var(--accent-red); }
.metric-card.amber::before { background: var(--accent-amber); }

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}

.metric-value.green { color: var(--accent-green); }
.metric-value.red   { color: var(--accent-red); }

.metric-delta {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    margin-top: 0.4rem;
}

/* ── Signal Banner ───────────────────────────────── */
.signal-buy {
    background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(0,212,170,0.03));
    border: 1px solid rgba(0,212,170,0.35);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: var(--glow-green);
    animation: pulse-green 2.5s ease-in-out infinite;
}

.signal-sell {
    background: linear-gradient(135deg, rgba(255,77,109,0.08), rgba(255,77,109,0.03));
    border: 1px solid rgba(255,77,109,0.35);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: var(--glow-red);
    animation: pulse-red 2.5s ease-in-out infinite;
}

@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 15px rgba(0,212,170,0.2); }
    50%      { box-shadow: 0 0 30px rgba(0,212,170,0.4); }
}

@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 15px rgba(255,77,109,0.2); }
    50%      { box-shadow: 0 0 30px rgba(255,77,109,0.4); }
}

.signal-label {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.1em;
}

.signal-buy  .signal-label { color: var(--accent-green); }
.signal-sell .signal-label { color: var(--accent-red); }

.signal-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 0.4rem;
}

/* ── Confidence Bar ──────────────────────────────── */
.confidence-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
}

.conf-track {
    background: var(--bg-base);
    border-radius: 999px;
    height: 8px;
    margin-top: 0.5rem;
    overflow: hidden;
}

.conf-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
}

.conf-fill.green { background: linear-gradient(90deg, #00d4aa, #00f5c8); }
.conf-fill.red   { background: linear-gradient(90deg, #ff4d6d, #ff8fa3); }

/* ── Stat Grid ───────────────────────────────────── */
.stat-table {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1.25rem;
    border-bottom: 1px solid var(--border);
}

.stat-row:last-child { border-bottom: none; }

.stat-key {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.stat-val {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    color: var(--text-primary);
}

/* ── Divider ─────────────────────────────────────── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ── Streamlit overrides ─────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

div[data-baseweb="select"] > div {
    background: var(--bg-elevated) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

.stDataFrame {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-elevated) !important;
    color: var(--accent-green) !important;
    border: 1px solid var(--border-bright) !important;
}

/* Sidebar stock badge */
.stock-badge {
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(0,212,170,0.08));
    border: 1px solid var(--border-bright);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    text-align: center;
}

.stock-badge-symbol {
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent-blue);
    letter-spacing: 0.1em;
}

.stock-badge-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* Live ticker dot */
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--accent-green);
    border-radius: 50%;
    margin-right: 6px;
    animation: blink 1.2s ease-in-out infinite;
    vertical-align: middle;
}

@keyframes blink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.2; }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_rnn_model():
    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / "stock_rnn_model.h5"
    return load_model(model_path, compile=False)

@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR / "Stock_Price_Forecasting_RNN" / "NIFTY50_all.csv"
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"])
    return df

model = load_rnn_model()
df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown('<div class="brand-header">NEURAL<br>TRADE</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">⚡ AI Forecasting Engine</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Select Instrument</div>', unsafe_allow_html=True)
    stocks = sorted(df["Symbol"].unique())
    selected_stock = st.selectbox("Stock Symbol", stocks, label_visibility="collapsed")

    st.markdown(f"""
    <div class="stock-badge">
        <div class="stock-badge-symbol">{selected_stock}</div>
        <div class="stock-badge-label">NIFTY 50 · NSE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Chart Options</div>', unsafe_allow_html=True)
    chart_type = st.radio(
        "Chart Type",
        ["Line", "Candlestick", "Area"],
        label_visibility="collapsed"
    )

    lookback = st.slider("Lookback Window (days)", min_value=30, max_value=180, value=90, step=10)

    st.markdown('<div class="section-label">Prediction Settings</div>', unsafe_allow_html=True)
    show_confidence = st.checkbox("Show Confidence Interval", value=True)
    show_volume = st.checkbox("Show Volume Overlay", value=False)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:#3d5a7a; line-height:1.8;">
        MODEL &nbsp;&nbsp;&nbsp;&nbsp; SimpleRNN<br>
        INPUT &nbsp;&nbsp;&nbsp;&nbsp; 30-day window<br>
        OUTPUT &nbsp;&nbsp; T+1 close price<br>
        ENGINE &nbsp;&nbsp; TensorFlow 2.x
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# FILTER & VALIDATE DATA
# --------------------------------------------------

stock_df = df[df["Symbol"] == selected_stock].copy()

# Detect available columns
has_ohlcv = all(c in stock_df.columns for c in ["Open", "High", "Low", "Volume"])

keep_cols = ["Date", "Close"]
if has_ohlcv:
    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]

stock_df = stock_df[keep_cols].dropna().reset_index(drop=True)

if len(stock_df) < 31:
    st.error("⚠️ Insufficient data for this instrument. Minimum 31 trading days required.")
    st.stop()

# --------------------------------------------------
# PREDICTION ENGINE
# --------------------------------------------------

prices = stock_df["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

last_30_scaled = scaled_prices[-30:]
X = last_30_scaled.reshape(1, 30, 1)

prediction = model.predict(X, verbose=0)
predicted_price = scaler.inverse_transform(prediction)[0][0]

latest_price = stock_df["Close"].iloc[-1]
prev_price   = stock_df["Close"].iloc[-2]

change        = predicted_price - latest_price
change_pct    = (change / latest_price) * 100
day_change    = latest_price - prev_price
day_change_pct = (day_change / prev_price) * 100

is_bullish = predicted_price > latest_price

# Confidence (heuristic: based on recent volatility)
recent_std  = stock_df["Close"].tail(30).std()
recent_mean = stock_df["Close"].tail(30).mean()
cv          = (recent_std / recent_mean)
confidence  = max(30, min(95, int(100 - cv * 300)))

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------

# Brand header row
col_title, col_live = st.columns([4, 1])
with col_title:
    st.markdown(f"""
    <div style="font-family:'Orbitron',monospace; font-size:1.5rem; font-weight:700;
                color:#e8f4fd; letter-spacing:0.05em;">
        {selected_stock}
        <span style="font-size:0.8rem; color:#7a9bbf; font-weight:400;
                     font-family:'Space Mono',monospace; margin-left:1rem;">
            NIFTY50 · NSE · INR
        </span>
    </div>
    """, unsafe_allow_html=True)

with col_live:
    st.markdown(f"""
    <div style="text-align:right; padding-top:0.5rem;">
        <span class="live-dot"></span>
        <span style="font-family:'Space Mono',monospace; font-size:0.6rem;
                     color:#7a9bbf; letter-spacing:0.15em; text-transform:uppercase;">
            LIVE
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── KPI Metrics Row ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    color = "green" if day_change >= 0 else "red"
    arrow = "▲" if day_change >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card {color}">
        <div class="metric-label">Current Price</div>
        <div class="metric-value">₹{latest_price:,.2f}</div>
        <div class="metric-delta" style="color:var(--accent-{'green' if day_change>=0 else 'red'})">
            {arrow} ₹{abs(day_change):.2f} ({abs(day_change_pct):.2f}%) today
        </div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    color = "green" if is_bullish else "red"
    arrow = "▲" if is_bullish else "▼"
    st.markdown(f"""
    <div class="metric-card {color}">
        <div class="metric-label">Predicted T+1</div>
        <div class="metric-value {'green' if is_bullish else 'red'}">₹{predicted_price:,.2f}</div>
        <div class="metric-delta" style="color:var(--accent-{'green' if is_bullish else 'red'})">
            {arrow} ₹{abs(change):.2f} ({abs(change_pct):.2f}%) forecast
        </div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    high_30 = stock_df["Close"].tail(30).max()
    low_30  = stock_df["Close"].tail(30).min()
    pos_in_range = ((latest_price - low_30) / (high_30 - low_30) * 100) if high_30 != low_30 else 50
    st.markdown(f"""
    <div class="metric-card amber">
        <div class="metric-label">30D Range</div>
        <div class="metric-value" style="font-size:1rem; padding-top:0.2rem;">
            ₹{low_30:,.0f} – ₹{high_30:,.0f}
        </div>
        <div class="metric-delta" style="color:var(--accent-amber)">
            Position: {pos_in_range:.0f}th percentile
        </div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    vol = stock_df["Close"].tail(30).pct_change().std() * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">30D Volatility</div>
        <div class="metric-value" style="font-size:1.4rem; padding-top:0.15rem;">{vol:.2f}%</div>
        <div class="metric-delta" style="color:var(--text-secondary)">
            daily std of returns
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Price Chart", "🔮  Prediction", "📋  Statistics"])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    chart_df = stock_df.tail(lookback).copy()

    # Build plot
    fig = go.Figure()

    bg_color    = "#070b14"
    paper_color = "#070b14"
    grid_color  = "#1a2d4a"
    text_color  = "#7a9bbf"

    if chart_type == "Candlestick" and has_ohlcv:
        fig.add_trace(go.Candlestick(
            x=chart_df["Date"],
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="OHLC",
            increasing_line_color="#00d4aa",
            decreasing_line_color="#ff4d6d",
            increasing_fillcolor="rgba(0,212,170,0.2)",
            decreasing_fillcolor="rgba(255,77,109,0.2)",
        ))
    elif chart_type == "Area":
        fig.add_trace(go.Scatter(
            x=chart_df["Date"], y=chart_df["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#3b82f6", width=2),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=chart_df["Date"], y=chart_df["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#3b82f6", width=2),
        ))

    # Moving averages
    if len(chart_df) >= 20:
        chart_df["MA20"] = chart_df["Close"].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=chart_df["Date"], y=chart_df["MA20"],
            mode="lines", name="MA 20",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
        ))

    if len(chart_df) >= 50:
        chart_df["MA50"] = chart_df["Close"].rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=chart_df["Date"], y=chart_df["MA50"],
            mode="lines", name="MA 50",
            line=dict(color="#a78bfa", width=1.5, dash="dash"),
        ))

    # Volume subplot if available
    if show_volume and has_ohlcv:
        vol_colors = ["rgba(0,212,170,0.4)" if c >= o else "rgba(255,77,109,0.4)"
                      for c, o in zip(chart_df["Close"], chart_df["Open"])]
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.75, 0.25], vertical_spacing=0.03)
        for trace in fig.data:
            fig2.add_trace(trace, row=1, col=1)
        fig2.add_trace(go.Bar(
            x=chart_df["Date"], y=chart_df["Volume"],
            name="Volume", marker_color=vol_colors, showlegend=False,
        ), row=2, col=1)
        fig = fig2

    fig.update_layout(
        paper_bgcolor=paper_color,
        plot_bgcolor=bg_color,
        font=dict(family="Space Mono", color=text_color, size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            bgcolor="rgba(13,20,33,0.8)",
            bordercolor="#1a2d4a",
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(
            gridcolor=grid_color, gridwidth=1,
            zeroline=False,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            gridcolor=grid_color, gridwidth=1,
            zeroline=False,
            tickprefix="₹",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#111c2d",
            bordercolor="#1a2d4a",
            font=dict(family="Space Mono", size=11, color="#e8f4fd"),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    pred_col, sig_col = st.columns([3, 2])

    with pred_col:
        # Historical + predicted point chart
        hist = stock_df["Close"].tail(30).values
        hist_dates = stock_df["Date"].tail(30).values

        next_date = pd.Timestamp(hist_dates[-1]) + timedelta(days=1)
        # skip weekends
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        fig_pred = go.Figure()

        # Confidence band
        if show_confidence:
            std_est = hist[-10:].std()
            upper = list(hist) + [predicted_price + std_est]
            lower = list(hist) + [predicted_price - std_est]
            dates_band = list(hist_dates) + [next_date]
            fig_pred.add_trace(go.Scatter(
                x=list(dates_band) + list(reversed(dates_band)),
                y=upper + list(reversed(lower)),
                fill="toself",
                fillcolor="rgba(59,130,246,0.06)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Confidence Band",
                hoverinfo="skip",
            ))

        # Historical line
        fig_pred.add_trace(go.Scatter(
            x=hist_dates, y=hist,
            mode="lines+markers",
            name="Historical",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=3, color="#3b82f6"),
        ))

        # Bridge dashed line
        bridge_color = "#00d4aa" if is_bullish else "#ff4d6d"
        fig_pred.add_trace(go.Scatter(
            x=[hist_dates[-1], next_date],
            y=[hist[-1], predicted_price],
            mode="lines",
            name="Forecast",
            line=dict(color=bridge_color, width=2, dash="dash"),
        ))

        # Predicted point
        fig_pred.add_trace(go.Scatter(
            x=[next_date],
            y=[predicted_price],
            mode="markers",
            name=f"T+1 Prediction",
            marker=dict(
                size=14, color=bridge_color,
                symbol="diamond",
                line=dict(color="#ffffff", width=2),
            ),
            hovertemplate=f"<b>Predicted</b><br>₹{predicted_price:,.2f}<extra></extra>",
        ))

        fig_pred.update_layout(
            paper_bgcolor="#070b14",
            plot_bgcolor="#070b14",
            font=dict(family="Space Mono", color="#7a9bbf", size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(bgcolor="rgba(13,20,33,0.8)", bordercolor="#1a2d4a", borderwidth=1),
            xaxis=dict(gridcolor="#1a2d4a", zeroline=False),
            yaxis=dict(gridcolor="#1a2d4a", zeroline=False, tickprefix="₹"),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#111c2d", bordercolor="#1a2d4a",
                            font=dict(family="Space Mono", size=11, color="#e8f4fd")),
        )

        st.plotly_chart(fig_pred, use_container_width=True)

    with sig_col:
        # Trading signal
        if is_bullish:
            st.markdown(f"""
            <div class="signal-buy">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">📈</div>
                <div class="signal-label">BUY SIGNAL</div>
                <div class="signal-desc">Model forecasts upward movement<br>of ₹{abs(change):.2f} ({abs(change_pct):.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="signal-sell">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">📉</div>
                <div class="signal-label">SELL SIGNAL</div>
                <div class="signal-desc">Model forecasts downward movement<br>of ₹{abs(change):.2f} ({abs(change_pct):.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence meter
        conf_color = "green" if confidence >= 60 else "red"
        st.markdown(f"""
        <div class="confidence-wrap">
            <div class="metric-label">Model Confidence</div>
            <div style="font-family:'Orbitron',monospace; font-size:1.8rem;
                        color:var(--accent-{'green' if confidence>=60 else 'red'});">
                {confidence}%
            </div>
            <div class="conf-track">
                <div class="conf-fill {conf_color}" style="width:{confidence}%;"></div>
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:0.6rem;
                        color:#3d5a7a; margin-top:0.5rem; letter-spacing:0.1em;">
                Based on 30-day volatility profile
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Model inputs summary
        st.markdown("""
        <div class="stat-table">
            <div class="stat-row">
                <span class="stat-key">Model</span>
                <span class="stat-val">SimpleRNN</span>
            </div>
            <div class="stat-row">
                <span class="stat-key">Window</span>
                <span class="stat-val">30 days</span>
            </div>
            <div class="stat-row">
                <span class="stat-key">Horizon</span>
                <span class="stat-val">T+1</span>
            </div>
            <div class="stat-row">
                <span class="stat-key">Scaling</span>
                <span class="stat-val">Min-Max</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    s1, s2 = st.columns(2)

    with s1:
        st.markdown('<div class="section-label">Descriptive Statistics</div>', unsafe_allow_html=True)
        desc = stock_df["Close"].describe()
        rows_html = "".join([
            f'<div class="stat-row"><span class="stat-key">{k.upper()}</span>'
            f'<span class="stat-val">₹{v:,.2f}</span></div>'
            for k, v in desc.items()
        ])
        st.markdown(f'<div class="stat-table">{rows_html}</div>', unsafe_allow_html=True)

    with s2:
        st.markdown('<div class="section-label">Recent 10 Sessions</div>', unsafe_allow_html=True)

        recent = stock_df[["Date", "Close"]].tail(10).copy()
        recent["Return"] = recent["Close"].pct_change() * 100
        recent["Date"] = recent["Date"].dt.strftime("%d %b %Y")
        recent = recent.iloc[::-1].reset_index(drop=True)
        recent.columns = ["Date", "Close (₹)", "Return (%)"]
        recent["Close (₹)"] = recent["Close (₹)"].map("₹{:,.2f}".format)
        recent["Return (%)"] = recent["Return (%)"].map(
            lambda x: f"+{x:.2f}%" if x > 0 else (f"{x:.2f}%" if pd.notna(x) else "–")
        )

        st.dataframe(recent, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Returns distribution
    st.markdown('<div class="section-label">Returns Distribution (All History)</div>', unsafe_allow_html=True)
    returns = stock_df["Close"].pct_change().dropna() * 100

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=returns,
        nbinsx=60,
        name="Daily Returns",
        marker=dict(
            color=returns.apply(lambda x: "rgba(0,212,170,0.5)" if x >= 0 else "rgba(255,77,109,0.5)"),
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
    ))
    fig_hist.add_vline(x=0, line_color="#7a9bbf", line_width=1, line_dash="dash")
    fig_hist.add_vline(x=returns.mean(), line_color="#f59e0b", line_width=1.5,
                       annotation_text=f"μ={returns.mean():.2f}%",
                       annotation_font=dict(color="#f59e0b", size=10))

    fig_hist.update_layout(
        paper_bgcolor="#070b14",
        plot_bgcolor="#070b14",
        font=dict(family="Space Mono", color="#7a9bbf", size=10),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(gridcolor="#1a2d4a", zeroline=False, title="Daily Return (%)"),
        yaxis=dict(gridcolor="#1a2d4a", zeroline=False, title="Frequency"),
        bargap=0.05,
        height=260,
        hoverlabel=dict(bgcolor="#111c2d", bordercolor="#1a2d4a",
                        font=dict(family="Space Mono", size=11, color="#e8f4fd")),
    )

    st.plotly_chart(fig_hist, use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
            font-family:'Space Mono',monospace; font-size:0.6rem; color:#3d5a7a;
            letter-spacing:0.1em; text-transform:uppercase;">
    <span>⚡ NeuralTrade · AI Stock Forecasting</span>
    <span>SimpleRNN · NIFTY50 · NSE India</span>
    <span>⚠ For educational use only. Not financial advice.</span>
</div>
""", unsafe_allow_html=True)