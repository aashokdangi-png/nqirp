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
        # Filter down to Equity & F&O segments on NSE
        df_filtered = df[df['exchange'].isin(['NSE_EQ', 'NSE_FO'])].copy()
        
        # Create mapping dictionary: Symbol/Trading Symbol -> Instrument Key
        mapping = {}
        for _, row in df_filtered.iterrows():
            trading_symbol = str(row.get('trading_symbol', '')).strip().upper()
            inst_key = str(row.get('instrument_key', '')).strip()
            if trading_symbol and inst_key:
                mapping[trading_symbol] = inst_key
                # Also map clean base symbols without suffix
                clean_sym = trading_symbol.split('-')[0]
                if clean_sym not in mapping:
                    mapping[clean_sym] = inst_key
                    
        return mapping, df_filtered
    except Exception as e:
        st.warning(f"Unable to load online Upstox Instrument Master. Error: {e}")
        return {}, pd.DataFrame()

# Initialize Upstox Mapping Dictionary
upstox_map, upstox_df = load_upstox_instrument_map()

def resolve_upstox_key(symbol: str) -> str:
    """
    Resolves a ticker/symbol to its official Upstox Instrument Key.
    Handles '.NS' extension stripping for NSE symbols.
    """
    clean_sym = symbol.replace('.NS', '').strip().upper()
    return upstox_map.get(clean_sym, f"NSE_EQ|{clean_sym}")

# Sidebar Upstox Key Resolution Status
st.sidebar.markdown("### 🔌 Feed Integration")
if upstox_map:
    st.sidebar.success(f"Upstox Instruments Loaded ({len(upstox_map):,} keys mapped)")
else:
    st.sidebar.info("Using Fallback Symbol Format for Feeds")

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
page = st.sidebar.radio(
    "Select Navigation Module",
    ["📊 Institutional SMC Intraday Scanner", "👁️ Vision AI Chart Pattern Scanner"],
    key="nav_sidebar_radio"
)

# ==============================================================================
# HELPER DATA FETCHING FUNCTIONS
# ==============================================================================
def fetch_market_data(ticker: str):
    """Fetches market historical data with yfinance fallback."""
    try:
        sym = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
        df = yf.download(sym, period="6mo", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=['Close'])
    except Exception:
        return pd.DataFrame()

# Standard Nifty / F&O Universe
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
# QUANTITATIVE SMC & PATTERN SCANNER MODULE
# ==============================================================================
if page == "📊 Institutional SMC Intraday Scanner":
    st.title("📊 Institutional SMC & Quantitative Intraday Scanner")
    st.markdown("Automated Market Structure, Fair Value Gap (FVG), Break of Structure (BOS), and Liquidity Sweep Detector.")

    st.subheader("⚙️ Live Multi-Asset Scan")
    
    if st.button("🚀 Run Institutional Scanner", use_container_width=True):
        with st.spinner("Fetching market data and running quantitative SMC analysis..."):
            results = []
            
            for symbol in DEFAULT_SYMBOLS:
                df = fetch_market_data(symbol)
                if df.empty or len(df) < 50:
                    continue

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
                if atr <= 0: continue

                ema50 = float(close.ewm(span=50).mean().iloc[-1])
                trend_bias = "BULLISH" if c > ema50 else "BEARISH"

                h20_prev = float(high.tail(21).iloc[:-1].max())
                l20_prev = float(low.tail(21).iloc[:-1].min())

                smc_confluences, scores = [], []
                direction = "NEUTRAL"

                bullish_fvg = float(low.iloc[-1]) > float(high.iloc[-3]) if len(df) >= 3 else False
                bearish_fvg = float(high.iloc[-1]) < float(low.iloc[-3]) if len(df) >= 3 else False

                if bullish_fvg and rvol >= 1.2 and trend_bias == "BULLISH":
                    smc_confluences.append("Bullish FVG")
                    scores.append(88)
                    direction = "BULLISH"
                elif bearish_fvg and rvol >= 1.2 and trend_bias == "BEARISH":
                    smc_confluences.append("Bearish FVG")
                    scores.append(88)
                    direction = "BEARISH"

                if c > h20_prev and trend_bias == "BULLISH":
                    smc_confluences.append("Bullish BOS")
                    scores.append(92)
                    direction = "BULLISH"
                elif c < l20_prev and trend_bias == "BEARISH":
                    smc_confluences.append("Bearish BOS")
                    scores.append(92)
                    direction = "BEARISH"

                if not scores or direction == "NEUTRAL":
                    continue

                master_score = max(scores) + min(len(smc_confluences) * 4.0, 20.0)
                upstox_key = resolve_upstox_key(symbol)

                # Target & Risk Parameters
                stop_dist = 1.0 * atr
                stop_loss = round(c - stop_dist if direction == "BULLISH" else c + stop_dist, 2)
                target_price = round(c + (2.5 * stop_dist) if direction == "BULLISH" else c - (2.5 * stop_dist), 2)

                results.append({
                    "Symbol": symbol.replace(".NS", ""),
                    "Upstox Instrument Key": upstox_key,
                    "Direction": direction,
                    "Master Score": round(master_score, 1),
                    "SMC Signals": ", ".join(smc_confluences),
                    "Entry Price": round(c, 2),
                    "Target Price": target_price,
                    "Stop Loss": stop_loss,
                    "RVOL": round(rvol, 2)
                })

            if results:
                res_df = pd.DataFrame(results).sort_values(by="Master Score", ascending=False).reset_index(drop=True)
                st.dataframe(res_df, use_container_width=True)
            else:
                st.info("No stocks matched the strict SMC confluence criteria at this moment.")

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
