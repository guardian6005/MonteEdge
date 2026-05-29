"""
═══════════════════════════════════════════════════════════════
NIFTY 50 FALSE BREAKOUT STRATEGY — MONTE CARLO DASHBOARD
═══════════════════════════════════════════════════════════════
Interactive Streamlit dashboard built on top of Modules 1-7.

Run locally:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, json, datetime
from pathlib import Path

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty False Breakout MC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS (UI POLISH)
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar background gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1419 0%, #1a2332 50%, #0f1419 100%);
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stRadio > label { color: #e6edf3 !important; }

/* Sidebar title */
.sidebar-title {
    font-size: 24px; font-weight: 800; color: #58a6ff !important;
    padding: 12px 0; border-bottom: 2px solid #30363d; margin-bottom: 20px;
}

/* Custom navigation buttons */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 6px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 2px 0;
    transition: all 0.2s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(88, 166, 255, 0.15);
    border-color: #58a6ff;
    transform: translateX(2px);
}

/* Main header */
.main-header {
    background: linear-gradient(90deg, #1f6feb 0%, #58a6ff 100%);
    color: white !important;
    padding: 18px 24px; border-radius: 12px;
    margin-bottom: 22px;
    box-shadow: 0 4px 14px rgba(31, 111, 235, 0.25);
}
.main-header h1 { color: white !important; margin: 0; font-size: 28px; }
.main-header p  { color: rgba(255,255,255,0.85) !important;
                  margin: 4px 0 0 0; font-size: 14px; }

/* Metric cards polish */
[data-testid="stMetric"] {
    background: rgba(130, 130, 130, 0.08);
    border: 1px solid rgba(130, 130, 130, 0.2);
    border-radius: 12px; padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Tag badges (Strategy Explorer) */
.tag-badge {
    display: inline-block;
    padding: 5px 12px; margin: 3px;
    border-radius: 14px;
    font-size: 12px; font-weight: 600;
    border: 1.5px solid;
}
.tag-direction-long   { background:#dcfce7; color:#15803d; border-color:#15803d; }
.tag-direction-short  { background:#fee2e2; color:#b91c1c; border-color:#b91c1c; }
.tag-conv-very-high   { background:#fef3c7; color:#92400e; border-color:#d97706; }
.tag-conv-high        { background:#dbeafe; color:#1e40af; border-color:#3b82f6; }
.tag-conv-medium      { background:#e5e7eb; color:#374151; border-color:#6b7280; }
.tag-trigger          { background:#f3e8ff; color:#6b21a8; border-color:#a855f7; }
.tag-adx              { background:#ccfbf1; color:#0f766e; border-color:#14b8a6; }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #fef9c3 0%, #fef3c7 100%);
    border-left: 5px solid #eab308;
    padding: 18px 22px; border-radius: 10px;
    margin: 18px 0;
    box-shadow: 0 2px 8px rgba(234, 179, 8, 0.12);
}
.insight-box h3 { margin-top: 0; color: #713f12 !important; }
.insight-box p, .insight-box b, .insight-box i { color: #451a03 !important; }
            
/* Section divider */
hr { margin: 28px 0; border: none; border-top: 1px solid rgba(130, 130, 130, 0.2); }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
ROOT = Path(__file__).parent
OUT_DIR   = ROOT / "outputs"
CHART_DIR = OUT_DIR / "charts"
DATA_DIR  = ROOT / "data"


# ──────────────────────────────────────────────
# CACHED DATA LOADERS
# ──────────────────────────────────────────────
@st.cache_data
def load_signals():
    return pd.read_csv(OUT_DIR / "signals_log.csv",
                       parse_dates=['signal_time', 'breakout_time'])

@st.cache_data
def load_backtests():
    v1 = pd.read_csv(OUT_DIR / "signals_backtested_V1.csv",
                     parse_dates=['signal_time'])
    v2 = pd.read_csv(OUT_DIR / "signals_backtested_V2.csv",
                     parse_dates=['signal_time'])
    v1 = v1[v1['outcome'].isin(['target_hit', 'sl_hit', 'time_stop'])]
    v2 = v2[v2['outcome'].isin(['target_hit', 'sl_hit', 'time_stop'])]
    return v1, v2

@st.cache_data
def load_mc_summary():
    """Load MC summaries with SAFE FALLBACKS to prevent KeyErrors."""
    def safe_load(path):
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    v1 = safe_load(OUT_DIR / "mc_summary_V1.json")
    v2 = safe_load(OUT_DIR / "mc_summary_V2.json")
    return v1, v2

@st.cache_data
def load_intraday():
    df = pd.read_csv(DATA_DIR / "nifty_5m.csv",
                     index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return df.between_time("09:15", "15:30")


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def safe_get(d, key, default=0):
    """Defensive dict accessor."""
    v = d.get(key, default)
    return v if v is not None else default

def fmt_pct(v, decimals=1):
    """Format as percentage; handles missing/None values."""
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v:+.{decimals}f}%"

def fmt_R(v, decimals=2):
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v:+.{decimals}f}R"

def ts_to_ms(ts):
    """Convert pandas Timestamp/datetime → milliseconds since epoch.
    Workaround for known Plotly add_vline+annotation_text bug with datetime axes."""
    return pd.Timestamp(ts).timestamp() * 1000

def render_tag(text, css_class):
    return f'<span class="tag-badge {css_class}">{text}</span>'


# ══════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════
st.sidebar.markdown('<div class="sidebar-title">📊 Nifty MC Dashboard</div>',
                     unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["🏠  Home",
     "📈  Strategy Explorer",
     "🎲  MC Playground",
     "📉  V1 vs V2",
     "🧠  Methodology"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Project**\n\nNifty 50 Options False-Breakout Strategy\n\n"
    "**Stack**\n\nPython · NumPy · SciPy · Pandas · "
    "Black-Scholes · Monte Carlo · Streamlit\n\n"
    "**Data Source**\n\nYahoo Finance"
)


# ════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ════════════════════════════════════════════════════════════════
if page.startswith("🏠"):
    st.markdown("""
    <div class="main-header">
      <h1>📊 Monte Carlo Simulation on NIFTY 50 Reversal Strategies</h1>
      <p>Monte Carlo validation of a multi-timeframe options setup</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    > **Hypothesis**: Multi-timeframe confluence levels + swing-based reversals
    > at HTF support/resistance produce a tradeable edge in Nifty weekly options.

    > **Question**: After 200 trades, what range of outcomes can I expect?
    > Which signal types actually carry the edge?
    """)

    v1_sum, v2_sum = load_mc_summary()

    # ── Headline metrics row (with safe accessors) ──
    st.markdown("### 🏆 Headline Comparison: V1 vs V2")

    v1_med_R   = safe_get(v1_sum, 'median_final_R', 0)
    v2_med_R   = safe_get(v2_sum, 'median_final_R', 0)
    v1_med_pct = safe_get(v1_sum, 'median_return_pct', None)
    v2_med_pct = safe_get(v2_sum, 'median_return_pct', None)
    v1_prof    = safe_get(v1_sum, 'prob_profit', 0)
    v2_prof    = safe_get(v2_sum, 'prob_profit', 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("V1 Median P&L (200 trades)", fmt_R(v1_med_R),
                fmt_pct(v1_med_pct) if v1_med_pct is not None else None)
    col2.metric("V2 Median P&L (200 trades)", fmt_R(v2_med_R),
                fmt_pct(v2_med_pct) if v2_med_pct is not None else None)
    col3.metric("V1 Probability of Profit",  f"{v1_prof*100:.1f}%")
    col4.metric("V2 Probability of Profit",  f"{v2_prof*100:.1f}%",
                f"+{(v2_prof-v1_prof)*100:.1f}%")

    st.markdown("---")

    # ── Killer insight box ──
    st.markdown("""
    <div class="insight-box">
      <h3>🎯 The Killer Insight</h3>
      <p style="margin-bottom: 0; font-size: 15px; line-height: 1.6;">
      By <b>dropping 50%</b> of low-quality signals (failed_breakout trigger), the
      strategy's expectancy increased <b>4×</b> and max-drawdown risk dropped <b>95%</b>.
      <br><br>
      <i>Same edge, half the work, dramatically smoother equity curve.</i>
      This is the <b>"Kill 50% of your signals to make MORE money"</b> paradox —
      made visible only through Monte Carlo simulation.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Executive dashboard image ──
    st.markdown("### 📊 Full Comparison Dashboard")
    dashboard_img = CHART_DIR / "07_executive_dashboard.png"
    if dashboard_img.exists():
        st.image(str(dashboard_img), use_container_width=True)
    else:
        st.warning("Executive dashboard image not found. Run Module 7 first.")

    # ── Pipeline ──
    with st.expander("🛠️ Project Architecture", expanded=False):
        st.code("""
Module 1: Data Fetcher        → yfinance (60d intraday + 10y daily)
    ↓
Module 2: Multi-TF Levels     → TDH/TDL, PDH/PDL, PWH/PWL, PMH/PML, ATH
    ↓
Module 3: Signal Engine       → 4-gate filter + 3 trigger types
    ↓
Module 4: Black-Scholes       → Synthetic option pricing + Greeks
    ↓
Module 5: Realized Backtest   → 49 signals → calibrated win-rate + RRR
    ↓
Module 6: Bootstrap MC        → 10,000 simulations × 200 trades
    ↓
Module 7: Advanced Analytics  → 7 publication-grade charts
    ↓
Module 8: Streamlit Dashboard ← YOU ARE HERE
        """, language="text")


# ════════════════════════════════════════════════════════════════
# PAGE 2: STRATEGY EXPLORER
# ════════════════════════════════════════════════════════════════
elif page.startswith("📈"):
    st.markdown("""
    <div class="main-header">
      <h1>📈 Strategy Explorer</h1>
      <p>Browse, filter, and drill into every signal generated by the engine</p>
    </div>
    """, unsafe_allow_html=True)

    sigs = load_signals()

    # ── Filters ──
    col1, col2, col3, col4 = st.columns(4)
    convictions = col1.multiselect(
        "Conviction", options=sorted(sigs['conviction'].unique()),
        default=sorted(sigs['conviction'].unique()))
    directions = col2.multiselect(
        "Direction", options=sigs['direction'].unique(),
        default=list(sigs['direction'].unique()))
    triggers = col3.multiselect(
        "Trigger Type", options=sigs['trigger_type'].unique(),
        default=list(sigs['trigger_type'].unique()))
    date_range = col4.date_input(
        "Date Range",
        value=(sigs['signal_time'].min().date(),
               sigs['signal_time'].max().date()))

    filtered = sigs[
        sigs['conviction'].isin(convictions) &
        sigs['direction'].isin(directions) &
        sigs['trigger_type'].isin(triggers)
    ]
    if len(date_range) == 2:
        filtered = filtered[
            (filtered['signal_time'].dt.date >= date_range[0]) &
            (filtered['signal_time'].dt.date <= date_range[1])
        ]

    st.markdown(f"### Showing **{len(filtered)}** signals")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Signals", len(filtered))
    col2.metric("Long / Short",
                f"{(filtered['direction']=='long').sum()} / "
                f"{(filtered['direction']=='short').sum()}")
    col3.metric("Avg Confluence Score",
                f"{filtered['confluence_score'].mean():.1f}"
                if len(filtered) > 0 else "N/A")

    show_cols = ['date', 'signal_time', 'direction', 'option_type',
                 'broken_level', 'speed_candles', 'confluence_score',
                 'conviction', 'trigger_type', 'adx']
    st.dataframe(filtered[show_cols].sort_values('signal_time', ascending=False),
                 use_container_width=True, height=380)

    # ── Drill into trade ──
    st.markdown("### 🔍 Drill into a Specific Trade")
    if len(filtered) > 0:
        trade_labels = filtered.apply(
            lambda r: f"{r['signal_time']}  |  {r['direction'].upper()}  |  "
                       f"{r['conviction']}  |  {r['trigger_type']}", axis=1)
        choice = st.selectbox("Select a trade", options=range(len(filtered)),
                               format_func=lambda i: trade_labels.iloc[i])
        sel = filtered.iloc[choice]

        col1, col2 = st.columns([2, 1])

        with col1:
            intra = load_intraday()
            sig_time = pd.Timestamp(sel['signal_time'])
            day_data = intra[intra.index.normalize() == sig_time.normalize()]

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=day_data.index,
                open=day_data['open'], high=day_data['high'],
                low=day_data['low'], close=day_data['close'],
                name='Nifty 5m',
                increasing_line_color='#22c55e',
                decreasing_line_color='#ef4444',
            ))

            # ── FIX: convert Timestamp to ms-since-epoch for add_vline ──
            sig_time_ms = ts_to_ms(sig_time)
            fig.add_vline(
                x=sig_time_ms,
                line_dash="dash", line_color="#3b82f6", line_width=2,
                annotation_text=f"  Signal: {sel['direction'].upper()}  ",
                annotation_position="top",
                annotation_font_color="#3b82f6",
                annotation_font_size=12,
            )
            fig.add_hline(
                y=float(sel['broken_level']),
                line_color="#f59e0b", line_width=1.5, line_dash="dot",
                annotation_text=f"  Broken Level: {sel['broken_level']:.2f}  ",
                annotation_position="top right",
                annotation_font_color="#b45309",
            )

            fig.update_layout(
                title=dict(
                    text=f"<b>{sel['date']}</b> — "
                         f"{sel['direction'].upper()} Signal at {sig_time.time()}",
                    font=dict(size=16)
                ),
                xaxis_rangeslider_visible=False,
                height=520,
                plot_bgcolor='#fafbfc',
                paper_bgcolor='white',
                xaxis=dict(gridcolor='#e5e7eb'),
                yaxis=dict(gridcolor='#e5e7eb'),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # ── Beautiful tag badges ──
            dir_class = f"tag-direction-{sel['direction']}"
            conv_class = f"tag-conv-{sel['conviction'].lower().replace(' ', '-')}"

            st.markdown("#### 🏷️ Signal Tags")
            tags_html = (
                render_tag(f"📍 {sel['direction'].upper()}", dir_class) +
                render_tag(f"⭐ {sel['conviction']}", conv_class) +
                render_tag(f"🎯 {sel['trigger_type']}", "tag-trigger") +
                render_tag(f"📊 ADX {sel['adx']:.1f}", "tag-adx") +
                render_tag(f"📈 {sel['option_type']}",
                           "tag-direction-long" if sel['option_type'] == 'CE'
                           else "tag-direction-short")
            )
            st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown("#### 📋 Trade Details")
            details = {
                'Date':            str(sel['date']),
                'Time':            str(sig_time.time()),
                'Direction':       sel['direction'],
                'Option Type':     sel['option_type'],
                'Broken Level':    f"₹{float(sel['broken_level']):,.2f}",
                'Nifty Spot':      f"₹{float(sel['nifty_spot']):,.2f}",
                'Speed (candles)': int(sel['speed_candles']),
                'Confluence':      int(sel['confluence_score']),
                'Conviction':      sel['conviction'],
                'Trigger':         sel['trigger_type'],
                'ADX':             f"{float(sel['adx']):.2f}",
                'Trend':           sel['trend_dir'],
            }
            for k, v in details.items():
                col_a, col_b = st.columns([1, 1.4])
                col_a.markdown(f"**{k}**")
                col_b.markdown(f"`{v}`")


# ════════════════════════════════════════════════════════════════
# PAGE 3: LIVE MC PLAYGROUND
# ════════════════════════════════════════════════════════════════
elif page.startswith("🎲"):
    st.markdown("""
    <div class="main-header">
      <h1>🎲 Live Monte Carlo Playground</h1>
      <p>Tweak the parameters and watch the distribution shift in real time</p>
    </div>
    """, unsafe_allow_html=True)

    preset = st.radio("Preset", ["Custom", "V1 Calibrated", "V2 Calibrated"],
                       horizontal=True)
    if preset == "V1 Calibrated":
        default_wr, default_aw, default_al = 0.375, 0.911, -0.473
    elif preset == "V2 Calibrated":
        default_wr, default_aw, default_al = 0.417, 0.993, -0.385
    else:
        default_wr, default_aw, default_al = 0.40, 1.0, -0.5

    col1, col2, col3 = st.columns(3)
    win_rate = col1.slider("Win Rate", 0.20, 0.70, default_wr, 0.01)
    avg_win  = col2.slider("Avg Win (R)", 0.3, 3.0, default_aw, 0.05)
    avg_loss = col3.slider("Avg Loss (R)", -2.0, -0.2, default_al, 0.05)

    col1, col2, col3 = st.columns(3)
    n_trades = col1.slider("Trades per Career", 50, 500, 200, 10)
    n_sims   = col2.slider("Number of Simulations", 500, 20_000, 5_000, 500)
    risk_pct = col3.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.25)

    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    rrr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    breakeven_wr = 1 / (1 + rrr) if rrr > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expectancy", f"{expectancy:+.3f}R",
                "Positive ✅" if expectancy > 0 else "Negative ❌")
    col2.metric("RRR", f"1:{rrr:.2f}")
    col3.metric("Breakeven WR", f"{breakeven_wr*100:.1f}%")
    col4.metric("Edge", f"{(win_rate - breakeven_wr)*100:+.1f}%",
                "Above ✅" if win_rate > breakeven_wr else "Below ❌")

    if st.button("🎲 Run Monte Carlo", type="primary", use_container_width=True):
        with st.spinner(f"Running {n_sims:,} simulations × {n_trades} trades..."):
            rng = np.random.default_rng(42)
            equity = np.zeros((n_sims, n_trades + 1))
            for s in range(n_sims):
                outcomes = np.where(
                    rng.random(n_trades) < win_rate, avg_win, avg_loss)
                equity[s, 1:] = np.cumsum(outcomes)

        finals = equity[:, -1]
        med = np.median(finals)
        prob_profit = (finals > 0).mean() * 100

        peaks = np.maximum.accumulate(equity, axis=1)
        dds = equity - peaks
        max_dds = dds.min(axis=1)
        dd_at_risk_pct = -max_dds * risk_pct
        prob_20dd = (dd_at_risk_pct >= 20).mean() * 100

        st.markdown("### 📊 Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median Final R", f"{med:+.2f}R")
        col2.metric("Median Return %", f"{med * risk_pct:+.1f}%")
        col3.metric("P(profit)", f"{prob_profit:.1f}%")
        col4.metric("P(>20% DD)", f"{prob_20dd:.1f}%")

        sample_idx = np.random.choice(n_sims, min(100, n_sims), replace=False)
        fig = go.Figure()
        for idx in sample_idx:
            fig.add_trace(go.Scatter(
                y=equity[idx], mode='lines',
                line=dict(color='rgba(59,130,246,0.12)', width=0.7),
                showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(y=np.median(equity, axis=0), mode='lines',
                                  line=dict(color='#1d4ed8', width=3),
                                  name='Median'))
        fig.add_trace(go.Scatter(y=np.percentile(equity, 10, axis=0), mode='lines',
                                  line=dict(color='#dc2626', width=2, dash='dash'),
                                  name='10th percentile'))
        fig.add_trace(go.Scatter(y=np.percentile(equity, 90, axis=0), mode='lines',
                                  line=dict(color='#16a34a', width=2, dash='dash'),
                                  name='90th percentile'))
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(
            title=f"Equity Curves — {n_sims:,} sims × {n_trades} trades",
            xaxis_title="Trade #", yaxis_title="Cumulative R",
            height=500, hovermode='x unified',
            plot_bgcolor='#fafbfc',
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(finals, nbins=60,
                title="Final P&L Distribution",
                labels={'value': 'Final R', 'count': 'Simulations'},
                color_discrete_sequence=['#3b82f6'])
            fig_hist.add_vline(x=0, line_color="black")
            fig_hist.add_vline(x=med, line_color="#dc2626", line_dash="dash",
                                annotation_text=f"Median {med:+.1f}R")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            fig_dd = px.histogram(max_dds, nbins=60,
                title="Max Drawdown Distribution",
                labels={'value': 'Max DD (R)', 'count': 'Simulations'},
                color_discrete_sequence=['#ef4444'])
            fig_dd.add_vline(x=np.median(max_dds),
                              line_color="#1e293b", line_dash="dash",
                              annotation_text=f"Median {np.median(max_dds):.1f}R")
            st.plotly_chart(fig_dd, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 4: V1 vs V2
# ════════════════════════════════════════════════════════════════
elif page.startswith("📉"):
    st.markdown("""
    <div class="main-header">
      <h1>📉 V1 vs V2 Comparison</h1>
      <p>Side-by-side analysis of both strategy versions</p>
    </div>
    """, unsafe_allow_html=True)

    v1_sum, v2_sum = load_mc_summary()

    def gv(d, k, default="-"):
        v = d.get(k)
        return v if v is not None else default

    metrics = [
        ('Calibrated win rate',
            f"{safe_get(v1_sum,'win_rate',0)*100:.1f}%",
            f"{safe_get(v2_sum,'win_rate',0)*100:.1f}%"),
        ('Calibrated expectancy',
            fmt_R(safe_get(v1_sum,'expectancy_R'), 3),
            fmt_R(safe_get(v2_sum,'expectancy_R'), 3)),
        ('Median final R',
            fmt_R(safe_get(v1_sum,'median_final_R')),
            fmt_R(safe_get(v2_sum,'median_final_R'))),
        ('Median return %',
            fmt_pct(safe_get(v1_sum,'median_return_pct')),
            fmt_pct(safe_get(v2_sum,'median_return_pct'))),
        ('Probability of profit',
            f"{safe_get(v1_sum,'prob_profit',0)*100:.1f}%",
            f"{safe_get(v2_sum,'prob_profit',0)*100:.1f}%"),
        ('Probability of 2x',
            f"{safe_get(v1_sum,'prob_2x',0)*100:.1f}%",
            f"{safe_get(v2_sum,'prob_2x',0)*100:.1f}%"),
        ('Median max DD',
            fmt_pct(-safe_get(v1_sum,'median_max_dd_pct')),
            fmt_pct(-safe_get(v2_sum,'median_max_dd_pct'))),
        ('Risk of >20% DD',
            f"{safe_get(v1_sum,'risk_20pct_dd',0)*100:.1f}%",
            f"{safe_get(v2_sum,'risk_20pct_dd',0)*100:.1f}%"),
        ('Risk of >30% DD',
            f"{safe_get(v1_sum,'risk_30pct_dd',0)*100:.1f}%",
            f"{safe_get(v2_sum,'risk_30pct_dd',0)*100:.1f}%"),
        ('Median Sharpe',
            f"{safe_get(v1_sum,'median_sharpe',0):.2f}",
            f"{safe_get(v2_sum,'median_sharpe',0):.2f}"),
    ]
    df_comp = pd.DataFrame(metrics,
                            columns=['Metric', 'V1 (All Signals)', 'V2 (Type B Only)'])
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("---")
    mc_chart = CHART_DIR / "monte_carlo_V1_vs_V2.png"
    if mc_chart.exists():
        st.image(str(mc_chart), use_container_width=True,
                 caption="Monte Carlo comparison — 10,000 simulations × 200 trades")

    st.markdown("### 📊 All Analytical Charts")
    chart_files = sorted(CHART_DIR.glob("0*.png"))
    for chart in chart_files:
        with st.expander(f"📈 {chart.stem.replace('_', ' ').title()}",
                         expanded=False):
            st.image(str(chart), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 5: METHODOLOGY
# ════════════════════════════════════════════════════════════════
elif page.startswith("🧠"):
    st.markdown("""
    <div class="main-header">
      <h1>🧠 Methodology Deep-Dive</h1>
      <p>How the strategy works under the hood</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Trigger Types",
        "📐 Black-Scholes",
        "🎲 Monte Carlo",
        "⚖️ Filter Gates"
    ])

    with tab1:
        st.markdown("""
### Trigger 1: `swing_low_reclaim` — BUY CE 📈
**Pattern**: Price was making lower lows → forms a higher low → reclaims prior swing low.

```
Price: 24000 → 23900 → 23800 (swing low) → 23850 → 23920 ✓ RECLAIM
```
**Why it works**: Buyers absorb selling pressure → exhaustion → reversal.
**Performance**: 40.0% WR, +0.21R/trade ✅

---

### Trigger 2: `swing_high_rejection` — BUY PE 📉 ⭐ STAR PERFORMER
**Pattern**: Price was making higher highs → forms a lower high → rejects below prior swing high.

```
Price: 23800 → 23900 → 24000 (swing high) → 23950 → 23880 ✓ REJECT
```
**Why it works**: Sellers absorb buying pressure → exhaustion → reversal.
**Performance**: 44.4% WR, +0.43R/trade ✅✅✅

---

### Trigger 3: `failed_breakout` — KILLED IN V2 ❌
**Pattern**: Price breaks above/below a level then immediately reverses.

**Why it failed**: Too generic, gets faked out by noise.
Backtest: **-0.05R/trade** — negative expectancy.

**Action**: Removed in V2 → expectancy 4× improvement.
        """)

    with tab2:
        st.markdown("""
### Black-Scholes Synthetic Option Pricing

Since real historical NSE option chains are messy, we use
**Black-Scholes with Historical Volatility (HV)** as an IV proxy.

**Formula** (European Put):
```
P = K·e^(-rT)·N(-d₂) - S·N(-d₁)

where:
  d₁ = [ln(S/K) + (r + σ²/2)T] / (σ·√T)
  d₂ = d₁ - σ·√T
```

| Variable | Source | Typical Value |
|---|---|---|
| S (Spot) | Live Nifty | ~₹24,000 |
| K (Strike) | Nearest ₹50 | ATM or ITM-1 |
| T (Time) | Days to Tuesday weekly expiry / 365 | 0.5 – 7 days |
| r (Risk-free) | India 10Y G-Sec | 6.65% |
| σ (Volatility) | 30-day HV × 1.10 | 14% – 30% |

**Why HV × 1.10?** Implied Volatility runs 5–15% above realized HV
(volatility risk premium). 1.10× gives a defensible proxy.
        """)

    with tab3:
        st.markdown("""
### Bootstrap Monte Carlo Methodology

Instead of theoretical distributions, we **bootstrap from actual trade outcomes**.

```
FOR each simulation s in 1..10,000:
    outcomes = []
    FOR each trade t in 1..200:
        r = random uniform [0, 1]
        IF r < calibrated_win_rate:
            outcome = RANDOM_PICK from actual_winning_trades
        ELSE:
            outcome = RANDOM_PICK from actual_losing_trades
        outcomes.append(outcome)
    
    equity_curve_s = cumulative_sum(outcomes)
```

**Why Bootstrap > Bernoulli**:
- Preserves the **shape** of real trade outcomes
- Gives **realistic drawdown distributions**
- Defensible: *"I sampled from my actual trades, not assumed math"*
        """)

    with tab4:
        st.markdown("""
### The 4-Gate Filter System

Every signal must pass **all 4 gates** before firing.

#### 🚦 Gate 1: Counter-Trend Block (ADX-based)
- Blocks LONG in strong BEARISH trends (ADX ≥ 25)
- Blocks SHORT in strong BULLISH trends
- **Lesson**: don't fade momentum

#### 🚦 Gate 2: Freefall Block
- Blocks LONGs during continuous downward momentum
- **Lesson**: don't catch falling knives

#### 🚦 Gate 3: Reversal Quality
- Requires meaningful drop/rise from peak/trough
- Blocks "sluggish" reversals
- **Lesson**: insist on conviction

#### 🚦 Gate 4: Day Structure
- Requires structural evidence (HL formed after LLs)
  OR strong bounce magnitude (≥35% retrace)
- **Lesson**: avoid dead-cat bounces
        """)
