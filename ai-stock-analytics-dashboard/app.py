"""
╔══════════════════════════════════════════════════════════════╗
║          QUANTUM STOCK ANALYTICS DASHBOARD                   ║
║          AI-Powered Fintech Analytics Platform               ║
╚══════════════════════════════════════════════════════════════╝
Author  : AI-Powered Dashboard
Stack   : Python · Streamlit · Plotly · scikit-learn · yfinance
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLING
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Stock Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Root palette ── */
  :root {
    --bg:        #07090f;
    --surface:   #0d1117;
    --border:    #1c2333;
    --accent:    #00e5ff;
    --accent2:   #ff4b6e;
    --accent3:   #39d353;
    --text:      #c9d1d9;
    --muted:     #6e7681;
    --card:      #0d1117;
  }

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
  }
  .stApp { background: var(--bg); }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] * { color: var(--text) !important; }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
  }
  [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem; letter-spacing: .08em; text-transform: uppercase; }
  [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace; font-size: 1.5rem; }
  [data-testid="stMetricDelta"] svg { display: none; }

  /* ── Section headers ── */
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--accent);
    border-left: 3px solid var(--accent);
    padding-left: 10px;
    margin: 32px 0 16px;
  }

  /* ── Signal badge ── */
  .signal-buy  { background:#0d2a1a; color:#39d353; border:1px solid #39d353; border-radius:6px; padding:8px 18px; font-family:'Space Mono',monospace; font-size:0.85rem; display:inline-block; }
  .signal-sell { background:#2a0d14; color:#ff4b6e; border:1px solid #ff4b6e; border-radius:6px; padding:8px 18px; font-family:'Space Mono',monospace; font-size:0.85rem; display:inline-block; }
  .signal-hold { background:#1a1a0d; color:#e3b341; border:1px solid #e3b341; border-radius:6px; padding:8px 18px; font-family:'Space Mono',monospace; font-size:0.85rem; display:inline-block; }

  /* ── Prediction box ── */
  .pred-box {
    background: linear-gradient(135deg, #0d1117 0%, #111827 100%);
    border: 1px solid var(--accent);
    border-radius: 12px;
    padding: 24px 28px;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,229,255,.07);
  }
  .pred-label { font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:.15em; color:var(--muted); text-transform:uppercase; }
  .pred-value { font-family:'Space Mono',monospace; font-size:2.2rem; color:var(--accent); margin:6px 0 2px; }
  .pred-meta  { font-size:0.78rem; color:var(--muted); }

  /* ── Inputs ── */
  input, .stTextInput>div>div>input, .stSelectbox select {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
  }
  .stSelectbox [data-baseweb="select"] { background: var(--surface); border-color: var(--border); }

  /* ── Dividers ── */
  hr { border-color: var(--border); margin: 24px 0; }

  /* ── Plotly chart border ── */
  .js-plotly-plot { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

  /* hide streamlit branding ── */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PLOTLY DARK TEMPLATE
# ──────────────────────────────────────────────────────────────
CHART_THEME = dict(
    plot_bgcolor  = "#07090f",
    paper_bgcolor = "#07090f",
    font          = dict(family="DM Sans", color="#c9d1d9", size=12),
    xaxis         = dict(gridcolor="#1c2333", linecolor="#1c2333", showgrid=True, zeroline=False),
    yaxis         = dict(gridcolor="#1c2333", linecolor="#1c2333", showgrid=True, zeroline=False),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1c2333"),
    margin        = dict(l=50, r=30, t=50, b=50),
)

# ──────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance. Returns empty df on failure."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_info(ticker: str) -> dict:
    """Fetch company metadata (name, sector, market cap)."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":       info.get("longName", ticker),
            "sector":     info.get("sector", "—"),
            "market_cap": info.get("marketCap", None),
            "pe":         info.get("trailingPE", None),
            "52w_high":   info.get("fiftyTwoWeekHigh", None),
            "52w_low":    info.get("fiftyTwoWeekLow", None),
        }
    except Exception:
        return {"name": ticker, "sector": "—", "market_cap": None, "pe": None, "52w_high": None, "52w_low": None}

# ──────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────
def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA50 and MA200 columns (where enough data exists)."""
    df = df.copy()
    df["MA50"]  = df["Close"].rolling(window=50,  min_periods=1).mean()
    df["MA200"] = df["Close"].rolling(window=200, min_periods=1).mean()
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Classic Wilder RSI."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_signal(df: pd.DataFrame) -> str:
    """MA crossover signal: BUY when MA50 crosses above MA200, SELL below, else HOLD."""
    if len(df) < 2:
        return "HOLD"
    last, prev = df.iloc[-1], df.iloc[-2]
    if prev["MA50"] <= prev["MA200"] and last["MA50"] > last["MA200"]:
        return "BUY"
    if prev["MA50"] >= prev["MA200"] and last["MA50"] < last["MA200"]:
        return "SELL"
    return "HOLD" if last["MA50"] >= last["MA200"] else "SELL"

# ──────────────────────────────────────────────────────────────
# MACHINE LEARNING
# ──────────────────────────────────────────────────────────────
def train_rf_model(df: pd.DataFrame):
    """
    Train a Random Forest to predict next-day Close.
    Features: Open, High, Low, Volume + lag features.
    Returns (predicted_price, mae, model).
    """
    data = df[["Open", "High", "Low", "Volume", "Close"]].copy()

    # Lag features (t-1, t-2, t-3)
    for lag in range(1, 4):
        data[f"Close_lag{lag}"] = data["Close"].shift(lag)
    data["MA5"]  = data["Close"].rolling(5,  min_periods=1).mean()
    data["MA10"] = data["Close"].rolling(10, min_periods=1).mean()
    data["RSI"]  = compute_rsi(data["Close"])
    data.dropna(inplace=True)

    if len(data) < 30:
        return None, None, None

    feature_cols = ["Open", "High", "Low", "Volume", "Close_lag1",
                    "Close_lag2", "Close_lag3", "MA5", "MA10", "RSI"]
    X = data[feature_cols].values
    y = data["Close"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))

    # Predict next day using latest row
    last_features = scaler.transform([X[-1]])
    predicted = model.predict(last_features)[0]

    return predicted, mae, model

# ──────────────────────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────────────────────
def plot_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Full OHLCV candlestick with MA overlays and volume sub-plot."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("", "Volume")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        name="OHLC",
        increasing_line_color="#39d353", increasing_fillcolor="#39d353",
        decreasing_line_color="#ff4b6e", decreasing_fillcolor="#ff4b6e",
        line_width=1,
    ), row=1, col=1)

    # MA lines
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA50"], name="MA 50",
        line=dict(color="#00e5ff", width=1.5, dash="solid"), opacity=0.85
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA200"], name="MA 200",
        line=dict(color="#f7c948", width=1.5, dash="dot"), opacity=0.85
    ), row=1, col=1)

    # Volume bars
    colors = ["#39d353" if c >= o else "#ff4b6e"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.6, showlegend=False
    ), row=2, col=1)

    fig.update_layout(
        **CHART_THEME,
        title=dict(text=f"<b>{ticker}</b> · Price & Volume", x=0.02,
                   font=dict(family="Space Mono", size=14, color="#c9d1d9")),
        xaxis_rangeslider_visible=False,
        height=560,
        hovermode="x unified",
    )
    fig.update_xaxes(showspikes=True, spikecolor="#1c2333", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor="#1c2333", spikethickness=1)
    return fig


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    """RSI chart with overbought/oversold bands."""
    rsi = compute_rsi(df["Close"])
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,75,110,0.08)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(57,211,83,0.08)",  line_width=0)
    fig.add_hline(y=70, line=dict(color="#ff4b6e", width=1, dash="dash"))
    fig.add_hline(y=30, line=dict(color="#39d353", width=1, dash="dash"))
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, name="RSI",
        line=dict(color="#00e5ff", width=2), fill="tozeroy",
        fillcolor="rgba(0,229,255,0.04)"
    ))
    fig.update_layout(
        **CHART_THEME,
        title=dict(text="<b>RSI</b> · Relative Strength Index (14)", x=0.02,
                   font=dict(family="Space Mono", size=13, color="#c9d1d9")),
        height=280, yaxis_range=[0, 100],
        hovermode="x unified",
    )
    return fig


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def fmt_large(n):
    if n is None: return "—"
    if n >= 1e12: return f"${n/1e12:.2f}T"
    if n >= 1e9:  return f"${n/1e9:.2f}B"
    if n >= 1e6:  return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def delta_color(val):
    return "#39d353" if val >= 0 else "#ff4b6e"

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Space Mono",monospace; font-size:1.1rem; color:#00e5ff;
                border-bottom:1px solid #1c2333; padding-bottom:14px; margin-bottom:20px;'>
      ⚡ QUANTUM ANALYTICS
    </div>
    """, unsafe_allow_html=True)

    ticker = st.text_input("Primary Ticker", value="AAPL").upper().strip()

    period = st.selectbox(
        "Time Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=3,
    )

    st.markdown("---")
    run_ml = st.toggle("🤖 Enable AI Price Prediction", value=True)

    st.markdown("""
    <div style='margin-top:40px; font-size:0.7rem; color:#3d4451; line-height:1.6;'>
      Data via Yahoo Finance · Not financial advice.<br>
      Refresh every 5 min (cached).
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────
# Hero header
st.markdown(f"""
<div style='padding:12px 0 28px;'>
  <div style='font-family:"Space Mono",monospace; font-size:0.68rem;
              letter-spacing:.22em; color:#6e7681; text-transform:uppercase;
              margin-bottom:6px;'>Real-Time Stock Intelligence</div>
  <div style='font-size:2.4rem; font-weight:600; color:#c9d1d9; line-height:1.1;'>
    {ticker}
    <span style='font-size:1rem; color:#6e7681; font-weight:300; margin-left:10px;'>Dashboard</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Fetch data ──
with st.spinner(f"Fetching {ticker} …"):
    df_raw = fetch_stock_data(ticker, period)
    info   = fetch_stock_info(ticker)

if df_raw.empty:
    st.error(f"❌  Could not fetch data for **{ticker}**. Check the ticker symbol and try again.")
    st.stop()

df = add_moving_averages(df_raw)

# ── Key metrics row ──
close_today  = float(df["Close"].iloc[-1])
close_prev   = float(df["Close"].iloc[-2]) if len(df) > 1 else close_today
day_chg      = close_today - close_prev
day_chg_pct  = day_chg / close_prev * 100
period_ret   = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
avg_vol      = df["Volume"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price",    f"${close_today:,.2f}", f"{day_chg:+.2f} ({day_chg_pct:+.2f}%)")
col2.metric("Period Return",    f"{period_ret:+.2f}%")
col3.metric("Market Cap",       fmt_large(info["market_cap"]))
col4.metric("P/E Ratio",        f"{info['pe']:.1f}" if info["pe"] else "—")
col5.metric("52W High / Low",   f"${info['52w_high']:,.2f}" if info["52w_high"] else "—",
                                f"↓ ${info['52w_low']:,.2f}" if info["52w_low"] else None)

st.markdown(f"<div style='font-size:0.78rem; color:#6e7681; margin-top:6px;'>Sector: <b style='color:#c9d1d9'>{info['sector']}</b> &nbsp;·&nbsp; Company: <b style='color:#c9d1d9'>{info['name']}</b></div>", unsafe_allow_html=True)
st.markdown("---")

# ── Signal ──
signal = generate_signal(df)
signal_html = {
    "BUY":  "<span class='signal-buy'>▲ BUY SIGNAL — MA50 crossed above MA200</span>",
    "SELL": "<span class='signal-sell'>▼ SELL SIGNAL — MA50 crossed below MA200</span>",
    "HOLD": "<span class='signal-hold'>◆ HOLD — No fresh crossover detected</span>",
}[signal]
st.markdown(f"<div class='section-title'>Trading Signal</div>{signal_html}", unsafe_allow_html=True)

# ── Candlestick chart ──
st.markdown("<div class='section-title'>Price Chart · OHLCV + Moving Averages</div>", unsafe_allow_html=True)
st.plotly_chart(plot_candlestick(df, ticker), use_container_width=True)

# ── RSI only ──
st.markdown("<div class='section-title'>Technical Indicator · RSI</div>", unsafe_allow_html=True)
st.plotly_chart(plot_rsi(df), use_container_width=True)

# ── AI Prediction ──
if run_ml:
    st.markdown("<div class='section-title'>AI Price Prediction · Random Forest</div>", unsafe_allow_html=True)
    with st.spinner("Training model on historical data …"):
        pred_price, mae, model = train_rf_model(df)

    if pred_price is None:
        st.warning("Not enough data to train the model (need ≥ 30 rows).")
    else:
        direction = "▲" if pred_price > close_today else "▼"
        diff      = pred_price - close_today
        diff_pct  = diff / close_today * 100
        col_a, col_b, col_c = st.columns([1.2, 1, 1])
        with col_a:
            st.markdown(f"""
            <div class='pred-box'>
              <div class='pred-label'>Predicted Next-Day Close</div>
              <div class='pred-value'>{direction} ${pred_price:,.2f}</div>
              <div class='pred-meta'>Δ {diff:+.2f} &nbsp;({diff_pct:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.metric("Current Close",  f"${close_today:,.2f}")
            st.metric("Model MAE",      f"${mae:.2f}")
        with col_c:
            st.metric("Training Rows",  f"{len(df)}")
            st.metric("Features Used",  "10")


# ── Raw data expander ──
with st.expander("📋  Raw OHLCV Data"):
    st.dataframe(df[["Open","High","Low","Close","Volume","MA50","MA200"]].tail(60).sort_index(ascending=False), use_container_width=True)

# ── Footer ──
st.markdown("""
<hr>
<div style='text-align:center; font-size:0.72rem; color:#3d4451; padding:8px 0 4px;
            font-family:"Space Mono",monospace; letter-spacing:.08em;'>
  QUANTUM STOCK ANALYTICS · Built with Streamlit · Data by Yahoo Finance · Not financial advice
</div>
""", unsafe_allow_html=True)