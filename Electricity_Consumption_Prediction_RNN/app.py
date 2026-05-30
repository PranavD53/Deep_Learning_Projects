import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from pathlib import Path


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PowerCast – Electricity Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ---- Root palette ---- */
:root {
    --bg:          #0a0e17;
    --surface:     #111827;
    --surface2:    #1a2236;
    --border:      #1f2e47;
    --accent:      #f5a623;
    --accent2:     #00d4ff;
    --danger:      #ff4d6d;
    --text:        #e8edf5;
    --muted:       #6b7a99;
    --glow-amber:  0 0 24px rgba(245,166,35,.35);
    --glow-cyan:   0 0 24px rgba(0,212,255,.25);
}

/* ---- Global reset ---- */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Scanline overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,.08) 2px,
        rgba(0,0,0,.08) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="collapsedControl"] { display: none !important; }

/* ---- Hero header ---- */
.hero {
    padding: 2.5rem 0 1.5rem;
    text-align: center;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: .6rem;
}
.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    font-weight: 800;
    line-height: 1.08;
    background: linear-gradient(135deg, #f5a623 0%, #ffd166 45%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .5rem;
}
.hero-sub {
    color: var(--muted);
    font-size: .95rem;
    font-weight: 400;
    letter-spacing: .03em;
}

/* ---- Control card ---- */
.ctrl-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 32px rgba(0,0,0,.4);
}
.ctrl-label {
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: .5rem;
}

/* ---- Streamlit widget overrides ---- */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] > div > div:focus {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    box-shadow: none !important;
}
[data-testid="stSelectbox"] label {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .7rem !important;
    letter-spacing: .14em !important;
    text-transform: uppercase !important;
}

/* ---- Predict button ---- */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #f5a623, #e8920e) !important;
    color: #0a0e17 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: .06em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .85rem 2rem !important;
    cursor: pointer !important;
    transition: all .25s ease !important;
    box-shadow: var(--glow-amber) !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 40px rgba(245,166,35,.55) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ---- KPI row ---- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(0,0,0,.35);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.amber::before { background: linear-gradient(90deg, #f5a623, #ffd166); }
.kpi-card.cyan::before  { background: linear-gradient(90deg, #00d4ff, #7b61ff); }
.kpi-card.green::before { background: linear-gradient(90deg, #06d6a0, #1de9b6); }
.kpi-card.red::before   { background: linear-gradient(90deg, #ff4d6d, #ff8fab); }

.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: .62rem;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .4rem;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1;
    color: var(--text);
}
.kpi-unit {
    font-size: .75rem;
    font-weight: 400;
    color: var(--muted);
    margin-left: .2rem;
}

/* ---- Chart container ---- */
.chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 2rem 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 32px rgba(0,0,0,.4);
}
.chart-title {
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 1rem;
}

/* ---- Altair / Vega chart background ---- */
.vega-embed, canvas { background: transparent !important; }

/* ---- Table overrides ---- */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] table { background: var(--surface2) !important; }
[data-testid="stDataFrame"] th {
    background: var(--surface) !important;
    color: var(--accent) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .68rem !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .85rem !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ---- Status bar ---- */
.status-bar {
    display: flex;
    align-items: center;
    gap: .6rem;
    font-family: 'DM Mono', monospace;
    font-size: .7rem;
    letter-spacing: .1em;
    color: var(--muted);
    margin-bottom: 1.5rem;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #06d6a0;
    box-shadow: 0 0 8px #06d6a0;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: .5; transform: scale(1.4); }
}

/* ---- Hour badge ---- */
.peak-badge {
    display: inline-block;
    background: rgba(245,166,35,.15);
    border: 1px solid rgba(245,166,35,.4);
    border-radius: 6px;
    padding: .15rem .55rem;
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    color: var(--accent);
}
.trough-badge {
    display: inline-block;
    background: rgba(0,212,255,.1);
    border: 1px solid rgba(0,212,255,.3);
    border-radius: 6px;
    padding: .15rem .55rem;
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    color: var(--accent2);
}

/* ---- Section divider ---- */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.2rem 0;
}

/* ---- Spinner override ---- */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ---- Altair dark config via streamlit ---- */
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_regions() -> list[str]:
    if not os.path.isdir("models"):
        return ["Region_A", "Region_B", "Region_C"]   # demo fallback
    return sorted([
        f.replace(".h5", "")
        for f in os.listdir("models")
        if f.endswith(".h5")
    ])

def run_forecast(region: str) -> pd.DataFrame:
    """Load model + scaler, build 24-hour rolling forecast."""
    from tensorflow.keras.models import load_model  # lazy import
    
    BASE_DIR = Path(__file__).resolve().parent 
    MODEL_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    model = load_model(MODEL_DIR / f"{region}.h5", compile=False)
    scaler = joblib.load(MODEL_DIR / f"{region}_scaler.pkl")

    df = pd.read_csv(DATA_DIR / f"{region}.csv")
    df.columns = ["Datetime", "Load"]
    scaled = scaler.transform(df[["Load"]])
    seq = scaled[-24:].flatten()

    forecasts = []
    for _ in range(24):
        x = seq.reshape(1, 24, 1)
        pred = float(model.predict(x, verbose=0)[0][0])
        forecasts.append(pred)
        seq = np.append(seq[1:], pred)

    forecasts_inv = scaler.inverse_transform(
        np.array(forecasts).reshape(-1, 1)
    ).flatten()

    # Build hour labels starting from next full hour
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    hours = [(now + timedelta(hours=i + 1)) for i in range(24)]

    result = pd.DataFrame({
        "Hour": range(1, 25),
        "Time": [h.strftime("%H:%M") for h in hours],
        "Forecast_MW": forecasts_inv,
    })
    return result

def demo_forecast() -> pd.DataFrame:
    """Synthetic sine-wave forecast for demo/dev mode."""
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    hours = [(now + timedelta(hours=i + 1)) for i in range(24)]
    np.random.seed(42)
    base = 2400 + 800 * np.sin(np.linspace(0, 2 * np.pi, 24) - 1.5)
    noise = np.random.normal(0, 50, 24)
    return pd.DataFrame({
        "Hour": range(1, 25),
        "Time": [h.strftime("%H:%M") for h in hours],
        "Forecast_MW": (base + noise).round(1),
    })

# ── App state ─────────────────────────────────────────────────────────────────
regions = load_regions()
DEMO_MODE = not os.path.isdir("models")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">⚡ Neural Load Intelligence</div>
    <div class="hero-title">PowerCast</div>
    <div class="hero-sub">24-Hour Electricity Consumption Forecast · LSTM Deep Learning</div>
</div>
""", unsafe_allow_html=True)

if DEMO_MODE:
    st.info("🔌 **Demo mode** — no `models/` directory found. Showing synthetic forecast data.", icon="ℹ️")

# ── Control row ───────────────────────────────────────────────────────────────
col_sel, col_btn, col_info = st.columns([3, 2, 2])

with col_sel:
    selected_region = st.selectbox(
        "Select Grid Region",
        regions,
        help="Regions are loaded from the models/ directory"
    )

with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡ Forecast Next 24 Hours", use_container_width=True)

with col_info:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    now_str = datetime.now().strftime("%d %b %Y  %H:%M")
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;
                padding:.75rem 1rem;font-family:'DM Mono',monospace;font-size:.72rem;
                color:var(--muted);line-height:1.6;">
        <span style="color:var(--accent2);">REGION</span><br>
        <span style="color:var(--text);font-weight:600;">{selected_region}</span><br>
        <span style="color:var(--accent2);">TIMESTAMP</span><br>
        <span style="color:var(--text);">{now_str}</span>
    </div>
    """, unsafe_allow_html=True)

# ── Forecast execution ────────────────────────────────────────────────────────
if predict_clicked:
    with st.spinner("Running LSTM inference…"):
        try:
            if DEMO_MODE:
                result = demo_forecast()
            else:
                result = run_forecast(selected_region)
        except Exception as e:
            st.error(f"❌ Forecast failed: {e}")
            st.stop()

    mw = result["Forecast_MW"].values
    peak_h  = int(result.loc[mw.argmax(), "Hour"])
    trough_h = int(result.loc[mw.argmin(), "Hour"])
    peak_t  = result.loc[mw.argmax(), "Time"]
    trough_t = result.loc[mw.argmin(), "Time"]

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Status bar
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot"></div>
        FORECAST LIVE · {selected_region} · {datetime.now().strftime('%d %b %Y %H:%M UTC')}
    </div>
    """, unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card amber">
            <div class="kpi-label">Peak Load</div>
            <div class="kpi-value">{mw.max():,.0f}<span class="kpi-unit">MW</span></div>
            <div style="margin-top:.4rem">
                <span class="peak-badge">Hour {peak_h} · {peak_t}</span>
            </div>
        </div>
        <div class="kpi-card cyan">
            <div class="kpi-label">Trough Load</div>
            <div class="kpi-value">{mw.min():,.0f}<span class="kpi-unit">MW</span></div>
            <div style="margin-top:.4rem">
                <span class="trough-badge">Hour {trough_h} · {trough_t}</span>
            </div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Avg Load</div>
            <div class="kpi-value">{mw.mean():,.0f}<span class="kpi-unit">MW</span></div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">Peak–Trough Δ</div>
            <div class="kpi-value">{mw.max()-mw.min():,.0f}<span class="kpi-unit">MW</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────────────────
    try:
        import altair as alt

        chart_df = result.copy()
        chart_df["is_peak"] = chart_df["Forecast_MW"] == mw.max()

        base = alt.Chart(chart_df).encode(
            x=alt.X("Hour:O", axis=alt.Axis(
                labelColor="#6b7a99", gridColor="#1f2e47",
                domainColor="#1f2e47", tickColor="#1f2e47",
                labelFont="DM Mono", title="Hour Ahead",
                titleColor="#6b7a99"
            )),
            y=alt.Y("Forecast_MW:Q", axis=alt.Axis(
                labelColor="#6b7a99", gridColor="#1f2e47",
                domainColor="#1f2e47", tickColor="#1f2e47",
                labelFont="DM Mono", title="MW",
                titleColor="#6b7a99"
            )),
            tooltip=[
                alt.Tooltip("Hour:O", title="Hour"),
                alt.Tooltip("Time:N", title="Clock"),
                alt.Tooltip("Forecast_MW:Q", title="MW", format=",.1f"),
            ]
        )

        area = base.mark_area(
            line={"color": "#f5a623", "strokeWidth": 2},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="rgba(245,166,35,0.35)", offset=0),
                    alt.GradientStop(color="rgba(245,166,35,0.0)",  offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            )
        )

        points = base.mark_point(
            filled=True, size=60,
            color="#f5a623", opacity=0.9
        )

        peak_rule = alt.Chart(
            pd.DataFrame({"Hour": [peak_h]})
        ).mark_rule(color="#ffd166", strokeDash=[4, 4], strokeWidth=1.5).encode(
            x="Hour:O"
        )

        chart = (area + points + peak_rule).properties(
            height=340,
            background="transparent",
            padding={"left": 10, "right": 10, "top": 10, "bottom": 10},
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            labelFontSize=11, titleFontSize=11
        )

        st.markdown('<div class="chart-card"><div class="chart-title">⚡ Hourly Load Forecast (MW)</div>', unsafe_allow_html=True)
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except ImportError:
        # Fallback to native line chart
        st.markdown('<div class="chart-card"><div class="chart-title">⚡ Hourly Load Forecast (MW)</div>', unsafe_allow_html=True)
        st.line_chart(result.set_index("Hour")[["Forecast_MW"]])
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Table + Download ──────────────────────────────────────────────────────
    col_tbl, col_dl = st.columns([3, 1])

    with col_tbl:
        display_df = result.copy()
        display_df["Forecast_MW"] = display_df["Forecast_MW"].round(2)
        display_df.columns = ["Hour", "Clock Time", "Forecast (MW)"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col_dl:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        csv_bytes = result.to_csv(index=False).encode()
        st.download_button(
            label="⬇ Download CSV",
            data=csv_bytes,
            file_name=f"forecast_{selected_region}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(f"""
        <div style="margin-top:.8rem;background:var(--surface);border:1px solid var(--border);
                    border-radius:10px;padding:.9rem 1rem;font-family:'DM Mono',monospace;
                    font-size:.68rem;color:var(--muted);line-height:1.8;">
            <span style="color:var(--accent);">MODEL</span><br>LSTM · 24-step<br>
            <span style="color:var(--accent);">HORIZON</span><br>+24 Hours<br>
            <span style="color:var(--accent);">REGION</span><br>{selected_region}
        </div>
        """, unsafe_allow_html=True)