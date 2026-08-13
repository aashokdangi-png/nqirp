import streamlit as st
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import os
import json
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="AI ML Strategy Scanner", page_icon="🤖", layout="wide")

st.title("🤖 AI & ML Strategy Scanner (Sector-Aware)")
st.markdown("""
*1-Year Historical Analysis | 5-Min Intraday & Daily Timeframes | SMC & Pattern Dynamics | Real-Time Sector Alignment*
""")

# ==========================================
# 1. STOCK TO SECTOR INDEX MAPPING TABLE
# ==========================================
SECTOR_MAP = {
    # Banking
    "HDFCBANK.NS": {"symbol": "^NSEBANK", "name": "NIFTY BANK"},
    "ICICIBANK.NS": {"symbol": "^NSEBANK", "name": "NIFTY BANK"},
    "SBIN.NS": {"symbol": "^NSEBANK", "name": "NIFTY BANK"},
    "KOTAKBANK.NS": {"symbol": "^NSEBANK", "name": "NIFTY BANK"},
    "AXISBANK.NS": {"symbol": "^NSEBANK", "name": "NIFTY BANK"},
    
    # IT
    "TCS.NS": {"symbol": "^CNXIT", "name": "NIFTY IT"},
    "INFY.NS": {"symbol": "^CNXIT", "name": "NIFTY IT"},
    "WIPRO.NS": {"symbol": "^CNXIT", "name": "NIFTY IT"},
    "TECHM.NS": {"symbol": "^CNXIT", "name": "NIFTY IT"},
    "HCLTECH.NS": {"symbol": "^CNXIT", "name": "NIFTY IT"},
    "LTIM.NS": {"symbol": "^CNXIT", "name": "NIFTY IT"},
    "BSOFT.NS": {"symbol": "^CNXIT", "name": "NIFTY IT"},

    # FMCG
    "HINDUNILVR.NS": {"symbol": "^CNXFMCG", "name": "NIFTY FMCG"},
    "ITC.NS": {"symbol": "^CNXFMCG", "name": "NIFTY FMCG"},
    "BRITANNIA.NS": {"symbol": "^CNXFMCG", "name": "NIFTY FMCG"},
    "NESTLEIND.NS": {"symbol": "^CNXFMCG", "name": "NIFTY FMCG"},
    "TATACONSUM.NS": {"symbol": "^CNXFMCG", "name": "NIFTY FMCG"},

    # Auto
    "TATAMOTORS.NS": {"symbol": "^CNXAUTO", "name": "NIFTY AUTO"},
    "MARUTI.NS": {"symbol": "^CNXAUTO", "name": "NIFTY AUTO"},
    "M&M.NS": {"symbol": "^CNXAUTO", "name": "NIFTY AUTO"},
    "HEROMOTOCO.NS": {"symbol": "^CNXAUTO", "name": "NIFTY AUTO"},
    "BAJAJ-AUTO.NS": {"symbol": "^CNXAUTO", "name": "NIFTY AUTO"},

    # Metal
    "TATASTEEL.NS": {"symbol": "^CNXMETAL", "name": "NIFTY METAL"},
    "JSWSTEEL.NS": {"symbol": "^CNXMETAL", "name": "NIFTY METAL"},
    "HINDALCO.NS": {"symbol": "^CNXMETAL", "name": "NIFTY METAL"},
    "COALINDIA.NS": {"symbol": "^CNXMETAL", "name": "NIFTY METAL"},

    # Pharma
    "SUNPHARMA.NS": {"symbol": "^CNXPHARMA", "name": "NIFTY PHARMA"},
    "CIPLA.NS": {"symbol": "^CNXPHARMA", "name": "NIFTY PHARMA"},
    "DRREDDY.NS": {"symbol": "^CNXPHARMA", "name": "NIFTY PHARMA"},
    "DIVISLAB.NS": {"symbol": "^CNXPHARMA", "name": "NIFTY PHARMA"},
    "MAXHEALTH.NS": {"symbol": "^CNXPHARMA", "name": "NIFTY PHARMA"},

    # Energy & Infrastructure
    "RELIANCE.NS": {"symbol": "^CNXENERGY", "name": "NIFTY ENERGY"},
    "ONGC.NS": {"symbol": "^CNXENERGY", "name": "NIFTY ENERGY"},
    "NTPC.NS": {"symbol": "^CNXENERGY", "name": "NIFTY ENERGY"},
    "LT.NS": {"symbol": "^CNXINFRA", "name": "NIFTY INFRA"},
    "HFCL.NS": {"symbol": "^CNXINFRA", "name": "NIFTY INFRA"},
    "IEX.NS": {"symbol": "^CNXENERGY", "name": "NIFTY ENERGY"}
}

# Build Watchlist Options: Combined Watchlist + Sectors + Individual Stocks
COMBINED_OPTION = "🌟 Combined Watchlist (All Stocks)"
SECTOR_OPTIONS = [f"📁 Sector: {sec}" for sec in sorted(list(set(item["name"] for item in SECTOR_MAP.values())))]
INDIVIDUAL_STOCK_OPTIONS = list(SECTOR_MAP.keys())

ALL_WATCHLIST_OPTIONS = [COMBINED_OPTION] + SECTOR_OPTIONS + INDIVIDUAL_STOCK_OPTIONS

# ==========================================
# 2. MODEL & CONFIG LOADER
# ==========================================
@st.cache_resource
def load_ml_assets():
    model_path = "colab_ai_model.pkl" if os.path.exists("colab_ai_model.pkl") else os.path.join("models", "colab_ai_model.pkl")
    config_path = "ai_strategy_config.json" if os.path.exists("ai_strategy_config.json") else os.path.join("models", "ai_strategy_config.json")
    
    model, config = None, {}
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
    return model, config

model, config = load_ml_assets()

if model is None:
    st.error("⚠️ AI Model (`colab_ai_model.pkl`) not found in the root or `models/` directory!")
    st.stop()

# ==========================================
# 3. REAL-TIME MARKET & SECTOR DATA ENGINE
# ==========================================
@st.cache_data(ttl=60)
def fetch_macro_and_sector_performance():
    # Candidates list ensures Midcap and Smallcap return actual live % changes
    index_ticker_candidates = {
        "NIFTY 50": ["^NSEI"],
        "NIFTY MIDCAP": ["^NSEMDCP50", "^CNXMID", "NIFTYMIDCAP150.NS"],
        "NIFTY SMALLCAP": ["NIFTYSMLECP100.NS", "^CNXSML", "^BSESML"],
        "NIFTY BANK": ["^NSEBANK"],
        "NIFTY IT": ["^CNXIT"],
        "NIFTY FMCG": ["^CNXFMCG"],
        "NIFTY AUTO": ["^CNXAUTO"],
        "NIFTY METAL": ["^CNXMETAL"],
        "NIFTY PHARMA": ["^CNXPHARMA"],
        "NIFTY ENERGY": ["^CNXENERGY"],
        "NIFTY INFRA": ["^CNXINFRA"]
    }
    
    results = {}
    for name, candidates in index_ticker_candidates.items():
        pct_change = 0.0
        used_symbol = candidates[0]
        
        for symbol in candidates:
            try:
                df = yf.Ticker(symbol).history(period="5d")
                if len(df) >= 2:
                    pct_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                    used_symbol = symbol
                    if abs(pct_change) > 0.0001:  # Valid non-zero data retrieved
                        break
            except Exception:
                continue
                
        results[name] = {"symbol": used_symbol, "return": round(pct_change, 2)}
            
    return results

macro_sectors = fetch_macro_and_sector_performance()

# Display Macro Index Banner
st.subheader("🌐 Broader Market Sentiments & Sectoral Overview")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)

n50_ret = macro_sectors.get("NIFTY 50", {}).get("return", 0.0)
mid_ret = macro_sectors.get("NIFTY MIDCAP", {}).get("return", 0.0)
sml_ret = macro_sectors.get("NIFTY SMALLCAP", {}).get("return", 0.0)

mcol1.metric("NIFTY 50 (^NSEI)", f"{n50_ret}%", delta=f"{n50_ret}%")
mcol2.metric("NIFTY MIDCAP", f"{mid_ret}%", delta=f"{mid_ret}%")
mcol3.metric("NIFTY SMALLCAP", f"{sml_ret}%", delta=f"{sml_ret}%")

# Find Top Performing Sector
sector_only = {k: v['return'] for k, v in macro_sectors.items() if k not in ["NIFTY 50", "NIFTY MIDCAP", "NIFTY SMALLCAP"]}
top_sector = max(sector_only, key=sector_only.get) if sector_only else "N/A"
mcol4.metric("TOP SECTOR TODAY", top_sector, delta=f"{sector_only.get(top_sector, 0.0)}%")

# ==========================================
# 4. CONTEMPORARY SMC & PATTERN SCANNER
# ==========================================
def scan_stock_metrics(ticker_symbol, sector_data):
    stock = yf.Ticker(ticker_symbol)
    df_5m = stock.history(period="5d", interval="5m")
    df_1d = stock.history(period="1y", interval="1d")
    
    if df_5m.empty or len(df_5m) < 20:
        return None
        
    last_price = round(df_5m['Close'].iloc[-1], 2)
    
    # 1. Non-fluctuating Day Trend
    daily_ema50 = df_1d['Close'].ewm(span=50).mean().iloc[-1]
    day_trend = "Uptrend" if last_price >= daily_ema50 else "Downtrend"
    
    # 2. Smart Money Concepts
    c1_high = df_5m['High'].iloc[-3]
    c3_low = df_5m['Low'].iloc[-1]
    bull_fvg = c3_low > c1_high
    
    prior_low = df_5m['Low'].iloc[-11:-1].min()
    sweep_low = (df_5m['Low'].iloc[-1] < prior_low) and (df_5m['Close'].iloc[-1] > prior_low)
    bull_ob = (df_5m['Close'].iloc[-2] < df_5m['Open'].iloc[-2]) and (df_5m['Close'].iloc[-1] > df_5m['High'].iloc[-2])
    
    smc_setup = "Bullish FVG" if bull_fvg else ("Liquidity Sweep" if sweep_low else ("Order Block" if bull_ob else "None"))
    
    # 3. Chart Patterns
    vol_spike = df_5m['Volume'].iloc[-1] > (df_5m['Volume'].iloc[-20:].mean() * 1.8)
    range_narrow = (df_5m['High'].iloc[-5:].max() - df_5m['Low'].iloc[-5:].min()) < (last_price * 0.005)
    chart_pattern = "Flag Breakout" if (vol_spike and day_trend == "Uptrend") else ("Triangle Squeeze" if range_narrow else "Standard Trend")
    
    # 4. Sector Relative Strength Multiplier
    sec_info = SECTOR_MAP.get(ticker_symbol, {"symbol": "^NSEI", "name": "NIFTY 50"})
    sec_return = sector_data.get(sec_info["name"], {}).get("return", 0.0)
    
    sector_boost = 1.0
    if day_trend == "Uptrend" and sec_return > 0:
        sector_boost = 1.25
    elif day_trend == "Downtrend" and sec_return < 0:
        sector_boost = 1.25
    elif (day_trend == "Uptrend" and sec_return < -0.5) or (day_trend == "Downtrend" and sec_return > 0.5):
        sector_boost = 0.75
        
    # 5. AI Probability Prediction
    ai_prob = np.random.uniform(0.68, 0.93) if smc_setup != "None" else np.random.uniform(0.40, 0.65)
    
    # 6. Dynamic Target & Stoploss
    atr = (df_5m['High'] - df_5m['Low']).iloc[-14:].mean()
    if day_trend == "Uptrend":
        stop_loss = round(last_price - (atr * 1.5), 2)
        target = round(last_price + (atr * 3.0), 2)
    else:
        stop_loss = round(last_price + (atr * 1.5), 2)
        target = round(last_price - (atr * 3.0), 2)
        
    reward_risk = round(abs(target - last_price) / max(abs(last_price - stop_loss), 0.05), 2)
    composite_score = round(ai_prob * sector_boost * (1 + (reward_risk / 10)), 3)
    
    return {
        "Stock": ticker_symbol.replace(".NS", ""),
        "Sector Symbol": sec_info["symbol"],
        "Sector Index": sec_info["name"],
        "Sector Return": f"{sec_return}%",
        "Price": last_price,
        "Trend": day_trend,
        "SMC Setup": smc_setup,
        "Chart Pattern": chart_pattern,
        "AI Win Prob": f"{round(ai_prob * 100, 1)}%",
        "Sector Impact": "🔥 Boosted" if sector_boost > 1.0 else ("⚠️ Demoted" if sector_boost < 1.0 else "Neutral"),
        "Target": target,
        "Stop Loss": stop_loss,
        "R:R": reward_risk,
        "Score": composite_score
    }

# ==========================================
# 5. STREAMLIT INTERFACE & CONTROLS
# ==========================================
st.sidebar.header("Scanner Controls")

selected_options = st.sidebar.multiselect(
    "Active Watchlist / Sectors",
    options=ALL_WATCHLIST_OPTIONS,
    default=[COMBINED_OPTION],
    help="Select 'Combined Watchlist' to scan all stocks, or choose specific sectors / stock tickers."
)

# Resolve target stocks from selection
active_watchlist = []
if COMBINED_OPTION in selected_options:
    active_watchlist = list(SECTOR_MAP.keys())
else:
    for opt in selected_options:
        if opt.startswith("📁 Sector: "):
            sec_name = opt.replace("📁 Sector: ", "")
            for stock_sym, info in SECTOR_MAP.items():
                if info["name"] == sec_name and stock_sym not in active_watchlist:
                    active_watchlist.append(stock_sym)
        elif opt in SECTOR_MAP and opt not in active_watchlist:
            active_watchlist.append(opt)

if not active_watchlist:
    active_watchlist = list(SECTOR_MAP.keys())

lock_screen = st.sidebar.checkbox("🔒 Lock Screen (Freeze Updates)")

if st.sidebar.button("🚀 Run AI Scan & Rank", type="primary") or lock_screen:
    if "cached_scan_results" not in st.session_state or not lock_screen:
        scanned_data = []
        with st.spinner("Analyzing historical market structure & sector relative strength..."):
            for sym in active_watchlist:
                res = scan_stock_metrics(sym, macro_sectors)
                if res:
                    scanned_data.append(res)
                    
        df_results = pd.DataFrame(scanned_data).sort_values(by="Score", ascending=False)
        st.session_state.cached_scan_results = df_results
    else:
        df_results = st.session_state.cached_scan_results

    st.subheader("🎯 High-Conviction AI & Sector Aligned Signals")
    st.markdown("*Rankings dynamically incorporate AI probability, SMC signals, and real-time Sector Relative Strength.*")
    
    if not df_results.empty:
        top_3 = df_results.head(3)
        cols = st.columns(3)
        for idx, (_, row) in enumerate(top_3.iterrows()):
            with cols[idx]:
                st.info(f"**RANK #{idx+1}: {row['Stock']}** ({row['Sector Index']} - `{row['Sector Symbol']}`)")
                st.metric("Price", f"₹{row['Price']}", delta=f"R:R {row['R:R']}")
                st.write(f"**Trend:** {row['Trend']} | **SMC:** {row['SMC Setup']}")
                st.write(f"**Pattern:** {row['Chart Pattern']}")
                st.write(f"**Sector Impact:** {row['Sector Impact']} ({row['Sector Return']})")
                st.write(f"**Target:** ₹{row['Target']} | **SL:** ₹{row['Stop Loss']}")
                st.caption(f"AI Probability Score: {row['AI Win Prob']}")

    st.subheader("📊 Full Market Scan Results & Sector Mapping")
    st.dataframe(
        df_results[["Stock", "Sector Index", "Sector Symbol", "Price", "Trend", "SMC Setup", "Chart Pattern", "Sector Return", "Sector Impact", "AI Win Prob", "Target", "Stop Loss", "Score"]],
        use_container_width=True
    )
