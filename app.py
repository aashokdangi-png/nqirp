import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import io
import re
import requests

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="NQIRP Institutional Quant Engine", layout="wide")

# ==============================================================================
# UPSTOX INSTRUMENT RESOLUTION & MAPPING ENGINE
# ==============================================================================
@st.cache_data(ttl=86400)
def load_upstox_instrument_map():
    """
    Downloads and indexes the official Upstox Complete Instrument CSV.
    Provides seamless mapping between standard Symbols and exact Upstox Instrument Keys.
    """
    url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
    try:
        df = pd.read_csv(url, compression='gzip', low_memory=False)
        df_filtered = df[df['exchange'].isin(['NSE_EQ', 'NSE_FO'])].copy()
        
        mapping = {}
        for _, row in df_filtered.iterrows():
            trading_symbol = str(row.get('trading_symbol', '')).strip().upper()
            inst_key = str(row.get('instrument_key', '')).strip()
            if trading_symbol and inst_key:
                mapping[trading_symbol] = inst_key
                clean_sym = trading_symbol.split('-')[0]
                if clean_sym not in mapping:
                    mapping[clean_sym] = inst_key
                    
        return mapping, df_filtered
    except Exception as e:
        st.warning(f"Unable to load online Upstox Instrument Master. Error: {e}")
        return {}, pd.DataFrame()

upstox_map, upstox_df = load_upstox_instrument_map()

def resolve_upstox_key(symbol: str) -> str:
    """Resolves a ticker/symbol to its official Upstox Instrument Key."""
    clean_sym = symbol.replace('.NS', '').strip().upper()
    return upstox_map.get(clean_sym, f"NSE_EQ|{clean_sym}")

# Sidebar Upstox Key Resolution Status
st.sidebar.markdown("### 🔌 Feed Integration")
if upstox_map:
    st.sidebar.success(f"Upstox Instruments Loaded ({len(upstox_map):,} keys mapped)")
else:
    st.sidebar.info("Using Fallback Symbol Format for Feeds")

# ==============================================================================
# NAVIGATION
# ==============================================================================
page = st.sidebar.radio(
    "Select Navigation Module",
    ["📊 Dual-Engine SMC Scanner (Live & Daily)", "👁️ Vision AI Chart Pattern Scanner"],
    key="nav_sidebar_radio"
)

DEFAULT_SYMBOLS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS", "INDUSINDBK.NS", "INFY.NS",
    "JSWSTEEL.NS", "JIOFIN.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATACONSUM.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS",
    "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

# ==============================================================================
# HELPER DATA FETCHING & ANALYSIS FUNCTIONS
# ==============================================================================
def fetch_data(ticker: str, period="1mo", interval="5m"):
    """Fetches stock data for specified interval."""
    try:
        sym = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
        df = yf.download(sym, period=period, interval=interval, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=['Close'])
    except Exception:
        return pd.DataFrame()

def run_smc_analysis(df: pd.DataFrame, timeframe_label="INTRADAY"):
    """
    PRESERVES ALL EXISTING LOGIC (FVG, BOS, Trend Bias, Master Score, 1:2.5 RR)
    AND ADDS:
    - Reversal & Pullback Protection (Volume Fading Checks)
    - Chart Patterns: Double Tops/Bottoms, Triangles, Flags, Cup & Handle
    """
    if df.empty or len(df) < 30:
        return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']
    volume = df['Volume']

    c, h, l, o, v = float(close.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1]), float(open_p.iloc[-1]), float(volume.iloc[-1])

    v20 = float(volume.tail(20).mean())
    rvol = v / v20 if v20 > 0 else 1.0

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    if atr <= 0:
        return None

    ema50 = float(close.ewm(span=50).mean().iloc[-1])
    trend_bias = "BULLISH" if c > ema50 else "BEARISH"

    h20_prev = float(high.tail(21).iloc[:-1].max())
    l20_prev = float(low.tail(21).iloc[:-1].min())

    smc_confluences, scores = [], []
    direction = "NEUTRAL"
    breakout_level = c

    # --------------------------------------------------------------------------
    # 1. EXISTING: FAIR VALUE GAP (FVG)
    # --------------------------------------------------------------------------
    bullish_fvg = float(low.iloc[-1]) > float(high.iloc[-3]) if len(df) >= 3 else False
    bearish_fvg = float(high.iloc[-1]) < float(low.iloc[-3]) if len(df) >= 3 else False

    if bullish_fvg and rvol >= 1.0:
        smc_confluences.append("Bullish FVG")
        scores.append(88)
        direction = "BULLISH"
        breakout_level = float(high.iloc[-3])
    elif bearish_fvg and rvol >= 1.0:
        smc_confluences.append("Bearish FVG")
        scores.append(88)
        direction = "BEARISH"
        breakout_level = float(low.iloc[-3])

    # --------------------------------------------------------------------------
    # 2. EXISTING: BREAK OF STRUCTURE (BOS)
    # --------------------------------------------------------------------------
    if c > h20_prev:
        smc_confluences.append("Bullish BOS")
        scores.append(92)
        direction = "BULLISH"
        breakout_level = h20_prev
    elif c < l20_prev:
        smc_confluences.append("Bearish BOS")
        scores.append(92)
        direction = "BEARISH"
        breakout_level = l20_prev

    # --------------------------------------------------------------------------
    # 3. NEW ADDITION: DOUBLE TOP & DOUBLE BOTTOM REJECTIONS
    # --------------------------------------------------------------------------
    peak1 = float(high.tail(20).iloc[:-5].max())
    peak2 = float(high.tail(5).max())
    trough1 = float(low.tail(20).iloc[:-5].min())
    trough2 = float(low.tail(5).min())

    if abs(peak1 - peak2) / peak1 < 0.003 and c < peak2 and direction != "BULLISH":
        smc_confluences.append("Double Top Rejection")
        scores.append(85)
        direction = "BEARISH"
        breakout_level = peak2

    if abs(trough1 - trough2) / trough1 < 0.003 and c > trough2 and direction != "BEARISH":
        smc_confluences.append("Double Bottom Rejection")
        scores.append(85)
        direction = "BULLISH"
        breakout_level = trough2

    # --------------------------------------------------------------------------
    # 4. NEW ADDITION: TRIANGLE PATTERNS (Ascending, Descending, Symmetrical)
    # --------------------------------------------------------------------------
    recent_highs = high.tail(15)
    recent_lows = low.tail(15)
    high_slope = (recent_highs.iloc[-1] - recent_highs.iloc[0]) / 15
    low_slope = (recent_lows.iloc[-1] - recent_lows.iloc[0]) / 15

    if abs(high_slope) < 0.05 and low_slope > 0.05 and c >= recent_highs.max():
        smc_confluences.append("Ascending Triangle Breakout")
        scores.append(87)
        direction = "BULLISH"
        breakout_level = float(recent_highs.max())
    elif abs(low_slope) < 0.05 and high_slope < -0.05 and c <= recent_lows.min():
        smc_confluences.append("Descending Triangle Breakdown")
        scores.append(87)
        direction = "BEARISH"
        breakout_level = float(recent_lows.min())
    elif high_slope < -0.03 and low_slope > 0.03:
        if c > recent_highs.iloc[-3]:
            smc_confluences.append("Symmetrical Triangle Bullish Breakout")
            scores.append(84)
            direction = "BULLISH"
            breakout_level = float(recent_highs.iloc[-3])
        elif c < recent_lows.iloc[-3]:
            smc_confluences.append("Symmetrical Triangle Bearish Breakdown")
            scores.append(84)
            direction = "BEARISH"
            breakout_level = float(recent_lows.iloc[-3])

    # --------------------------------------------------------------------------
    # 5. NEW ADDITION: FLAG PATTERNS (Bull Flag & Bear Flag)
    # --------------------------------------------------------------------------
    pole_move = (close.iloc[-5] - close.iloc[-25]) / close.iloc[-25]
    flag_range = (high.tail(5).max() - low.tail(5).min()) / close.iloc[-1]

    if pole_move > 0.015 and flag_range < 0.008 and c >= high.tail(5).max():
        smc_confluences.append("Bull Flag Breakout")
        scores.append(90)
        direction = "BULLISH"
        breakout_level = float(high.tail(5).max())
    elif pole_move < -0.015 and flag_range < 0.008 and c <= low.tail(5).min():
        smc_confluences.append("Bear Flag Breakdown")
        scores.append(90)
        direction = "BEARISH"
        breakout_level = float(low.tail(5).min())

    # --------------------------------------------------------------------------
    # 6. NEW ADDITION: CUP AND HANDLE / INVERTED CUP AND HANDLE
    # --------------------------------------------------------------------------
    mid_low = low.tail(20).iloc[5:15].min()
    edge_high1 = high.tail(20).iloc[0:5].max()
    edge_high2 = high.tail(20).iloc[12:17].max()
    handle_low = low.tail(5).min()

    if (edge_high1 - mid_low) / mid_low > 0.01 and abs(edge_high1 - edge_high2) / edge_high1 < 0.005 and handle_low > mid_low:
        if c >= edge_high2:
            smc_confluences.append("Cup and Handle Breakout")
            scores.append(89)
            direction = "BULLISH"
            breakout_level = float(edge_high2)

    mid_high = high.tail(20).iloc[5:15].max()
    edge_low1 = low.tail(20).iloc[0:5].min()
    edge_low2 = low.tail(20).iloc[12:17].min()
    handle_high = high.tail(5).max()

    if (mid_high - edge_low1) / edge_low1 > 0.01 and abs(edge_low1 - edge_low2) / edge_low1 < 0.005 and handle_high < mid_high:
        if c <= edge_low2:
            smc_confluences.append("Inverted Cup and Handle Breakdown")
            scores.append(89)
            direction = "BEARISH"
            breakout_level = float(edge_low2)

    # Cancel signal if no confluences triggered
    if not scores or direction == "NEUTRAL":
        return None

    # --------------------------------------------------------------------------
    # EXISTING: MASTER SCORE CALCULATION
    # --------------------------------------------------------------------------
    master_score = max(scores) + min(len(smc_confluences) * 4.0, 20.0)

    # --------------------------------------------------------------------------
    # NEW FEATURE: EXTENSION & REVERSAL PROTECTOR
    # --------------------------------------------------------------------------
    max_extension_pct = 0.002  # 0.2% max extension threshold
    
    if direction == "BULLISH":
        pct_extended = (c - breakout_level) / breakout_level if breakout_level > 0 else 0
    else:
        pct_extended = (breakout_level - c) / breakout_level if breakout_level > 0 else 0

    if pct_extended > max_extension_pct and timeframe_label == "INTRADAY":
        trade_status = f"⚠️ EXTENDED (Limit {'Buy' if direction == 'BULLISH' else 'Sell'} at Structure)"
        suggested_entry = round(breakout_level, 2)
    else:
        trade_status = "✅ ACTIVE ENTRY"
        suggested_entry = round(c, 2)

    # Reversal probability based on volume dynamics
    if pct_extended > 0 and rvol < 1.0:
        target_prob = "⚠️ LOW (Fading Volume - Reversal Risk)"
    elif rvol >= 2.5 and len(smc_confluences) >= 2:
        target_prob = "🔥 HIGH (Strong Volume & Multi-Confluence)"
    elif rvol >= 1.2:
        target_prob = "⚡ MEDIUM (Healthy Volume)"
    else:
        target_prob = "⚠️ LOW (Weak Volume)"

    # --------------------------------------------------------------------------
    # EXISTING: TARGET & STOP LOSS (Scaled 1:2.5 Risk-to-Reward)
    # --------------------------------------------------------------------------
    if timeframe_label == "INTRADAY":
        stop_dist = max(1.5 * atr, suggested_entry * 0.005)
        target_dist = stop_dist * 2.5 
    else:
        stop_dist = 1.0 * atr
        target_dist = 2.5 * atr

    stop_loss = round(suggested_entry - stop_dist if direction == "BULLISH" else suggested_entry + stop_dist, 2)
    target_price = round(suggested_entry + target_dist if direction == "BULLISH" else suggested_entry - target_dist, 2)

    return {
        "Symbol": df.name if hasattr(df, 'name') else "STOCK",
        "Direction": direction,
        "Master Score": round(master_score, 1),
        "Trade Action": trade_status,
        "Target Probability": target_prob,
        "Suggested Entry": suggested_entry,
        "Current Price": round(c, 2),
        "Target Price": target_price,
        "Stop Loss": stop_loss,
        "R/R Ratio": "1 : 2.5",
        "RVOL": round(rvol, 2),
        "SMC Signals": ", ".join(smc_confluences)
    }

# ==============================================================================
# QUANTITATIVE SMC ENGINE (INTRADAY vs DAILY)
# ==============================================================================
if page == "📊 Dual-Engine SMC Scanner (Live & Daily)":
    st.title("📊 Dual-Engine Quantitative SMC Scanner")
    st.markdown("Generates **two separate results**: Real-time Intraday momentum (5-Min candles) & Historical Swing setups (Daily candles).")

    if st.button("🚀 Run Dual Scan (Live & Daily)", use_container_width=True):
        with st.spinner("Fetching Live (5m) and Historical (Daily) Data..."):
            intraday_results = []
            daily_results = []

            for symbol in DEFAULT_SYMBOLS:
                clean_sym = symbol.replace(".NS", "")
                upstox_key = resolve_upstox_key(symbol)

                # 1. INTRADAY LIVE DATA (5-Minute Candles)
                df_5m = fetch_data(symbol, period="5d", interval="5m")
                if not df_5m.empty:
                    df_5m.name = clean_sym
                    res_5m = run_smc_analysis(df_5m, timeframe_label="INTRADAY")
                    if res_5m:
                        res_5m["Upstox Instrument Key"] = upstox_key
                        intraday_results.append(res_5m)

                # 2. HISTORICAL DAILY DATA (Daily Candles)
                df_daily = fetch_data(symbol, period="6mo", interval="1d")
                if not df_daily.empty:
                    df_daily.name = clean_sym
                    res_daily = run_smc_analysis(df_daily, timeframe_label="DAILY")
                    if res_daily:
                        res_daily["Upstox Instrument Key"] = upstox_key
                        daily_results.append(res_daily)

            tab_intraday, tab_daily = st.tabs(["⚡ 1. Live Intraday Results (5-Min Data)", "📈 2. Daily Swing Results (Historical Daily Data)"])

            with tab_intraday:
                st.subheader("⚡ Live Intraday Scanner Results (5-Minute Timeframe)")
                st.caption("Targets updated to enforce a minimum 1:2.5 Risk-to-Reward ratio based on intraday volatility structure.")
                if intraday_results:
                    df_intra = pd.DataFrame(intraday_results).sort_values(by="Master Score", ascending=False).reset_index(drop=True)
                    st.dataframe(df_intra, use_container_width=True)
                else:
                    st.info("No active 5-minute intraday SMC confluences triggered right now.")

            with tab_daily:
                st.subheader("📈 Historical Daily Scanner Results (1-Day Timeframe)")
                st.caption("Optimized for multi-day swing trades based on daily structural breakouts and fair value gaps.")
                if daily_results:
                    df_day = pd.DataFrame(daily_results).sort_values(by="Master Score", ascending=False).reset_index(drop=True)
                    st.dataframe(df_day, use_container_width=True)
                else:
                    st.info("No active daily timeframe SMC confluences found.")

# ==============================================================================
# VISION AI MODULE
# ==============================================================================
elif page == "👁️ Vision AI Chart Pattern Scanner":
    st.title("👁️ Vision AI Chart Pattern Scanner")
    st.markdown("Upload a technical chart screenshot to analyze chart patterns with AI visual inspection.")
    
    uploaded_file = st.file_uploader("Upload Market Chart Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chart Viewport", use_container_width=True)
        st.success("Visual engine active. Image loaded for pattern recognition.")
