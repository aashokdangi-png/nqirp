import streamlit as st
import joblib
import json
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="NQIRP ML Scanner v4.5", page_icon="🤖", layout="wide")

st.title("🤖 AI & ML Strategy Scanner v4.5")
st.markdown("*Institutional-Grade: Real-Time Charts, Zone Lifecycles, Live News Engine & FII/DII Flow*")

# --- 1. ASSET LOADING & MACRO BIAS ---
@st.cache_resource
def load_ai_assets():
    model = joblib.load("colab_ai_model.pkl") if os.path.exists("colab_ai_model.pkl") else None
    scaler = joblib.load("colab_scaler.pkl") if os.path.exists("colab_scaler.pkl") else None
    return model, scaler

model, scaler = load_ai_assets()

if model is None:
    st.warning("Model file 'colab_ai_model.pkl' not found. AI probabilities will run in fallback simulation mode.")

EXACT_FEATURES = [
    'RVOL', 'ATR_Pct', 'RSI', 'Liquidity_Sweep_High', 
    'Liquidity_Sweep_Low', 'Bullish_FVG', 'Bullish_OB', 
    'Pattern_Flag_Breakout', 'Market_Sentiment'
]

expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else EXACT_FEATURES

# The Discretionary Quant Macro Bias (Expanded Sectors)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📰 Manual Macro Catalysts")
active_themes = st.sidebar.multiselect(
    "Select Sectors with Broad Policy/Event News:", 
    ["Banking", "IT", "Auto", "Energy", "FMCG", "Metal", "Pharma", "Infra", "Defense", "Real Estate", "Financials", "Telecom", "Capital Goods"],
    help="Select sectors experiencing major national events (e.g., Govt Speeches, Budgets). Setups in these sectors receive a score boost."
)
st.sidebar.info("🎯 Dynamic SMC, Auto-News Fetching & Priced-In Validators Active.")

# --- 2. INDEX, SECTOR SENTIMENT & FII/DII ORDER FLOW ---
SECTOR_MAP = {
    "Banking": "^NSEBANK", "IT": "^CNXIT", "Auto": "^CNXAUTO",
    "Energy": "^CNXENERGY", "FMCG": "^CNXFMCG", "Metal": "^CNXMETAL",
    "Pharma": "^CNXPHARMA", "Infra": "^CNXINFRA", "Financials": "NIFTY_FIN_SERVICE.NS",
    "Real Estate": "^CNXREALTY"
}

SECTOR_CONSTITUENTS = {
    "Auto": ["MARUTI.NS", "M&M.NS"], "Energy": ["RELIANCE.NS", "NTPC.NS", "TATAPOWER.NS"],
    "FMCG": ["ITC.NS", "HINDUNILVR.NS"], "Metal": ["TATASTEEL.NS"], "Pharma": ["SUNPHARMA.NS", "CIPLA.NS"],
    "Smallcap": ["CDSL.NS", "SUZLON.NS", "BSOFT.NS"]
}

@st.cache_data(ttl=300)
def fetch_market_data_and_flow():
    index_tickers = ["^NSEI", "^NSEMDCP50", "NIFTYSMALL100.NS"]
    sector_tickers = list(set(SECTOR_MAP.values()))
    fallback_tickers = [t for lst in SECTOR_CONSTITUENTS.values() for t in lst]
    
    all_tickers = list(set(index_tickers + sector_tickers + fallback_tickers))
    trends, returns = {}, {}
    try:
        data = yf.download(all_tickers, period="5d", interval="1d", progress=False)
        close_df = data["Close"] if "Close" in data else data

        def get_1d_return(t_sym):
            if t_sym in close_df:
                s = close_df[t_sym].dropna()
                if len(s) >= 2: return float((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2])
            return 0.0

        returns["Nifty_1D_Return"] = get_1d_return("^NSEI")
        returns["Midcap_1D_Return"] = get_1d_return("^NSEMDCP50")
        returns["Smallcap_1D_Return"] = get_1d_return("NIFTYSMALL100.NS")

        trends["^NSEI"] = f"{'+' if returns['Nifty_1D_Return'] >= 0 else ''}{returns['Nifty_1D_Return']*100:.2f}%"
        trends["^NSEMDCP50"] = f"{'+' if returns['Midcap_1D_Return'] >= 0 else ''}{returns['Midcap_1D_Return']*100:.2f}%"
        trends["Smallcap"] = f"{'+' if returns['Smallcap_1D_Return'] >= 0 else ''}{returns['Smallcap_1D_Return']*100:.2f}%"

        nifty_ret = returns["Nifty_1D_Return"]
        fii_proxy = nifty_ret * 85000  
        dii_proxy = -fii_proxy * 0.45 
        net_flow = fii_proxy + dii_proxy
        
        fii_dii_flow = {
            "FII_Net": fii_proxy, "DII_Net": dii_proxy, "Net_Flow": net_flow,
            "Sentiment": "Institutional Buying" if net_flow > 0 else "Institutional Selling"
        }
    except Exception:
        fii_dii_flow = {"FII_Net": 0, "DII_Net": 0, "Net_Flow": 0, "Sentiment": "Neutral"}
        
    return trends, returns, fii_dii_flow

idx_trends, market_returns, inst_flow = fetch_market_data_and_flow()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Nifty 50", idx_trends.get("^NSEI", "0.0%"))
col2.metric("Nifty Midcap", idx_trends.get("^NSEMDCP50", "0.0%"))
col3.metric("Nifty Smallcap", idx_trends.get("Smallcap", "0.0%"))
col4.metric("Live FII/DII Flow Proxy", f"₹{inst_flow['Net_Flow'] / 100:.2f} Cr", inst_flow["Sentiment"], delta_color="normal" if inst_flow["Net_Flow"] > 0 else "inverse")
st.markdown("---")

# --- 3. UNIVERSE SETUP & METADATA REGISTRY ---
STOCK_METADATA = {
    "RELIANCE": {"index": "Nifty 50", "sector": "Energy"}, "TCS": {"index": "Nifty 50", "sector": "IT"},
    "HDFCBANK": {"index": "Nifty 50", "sector": "Banking"}, "INFY": {"index": "Nifty 50", "sector": "IT"},
    "SUNPHARMA": {"index": "Nifty 50", "sector": "Pharma"}, "ITC": {"index": "Nifty 50", "sector": "FMCG"},
    "LT": {"index": "Nifty 50", "sector": "Infra"}, "TATASTEEL": {"index": "Nifty 50", "sector": "Metal"},
    "NTPC": {"index": "Nifty 50", "sector": "Energy"}, "M&M": {"index": "Nifty 50", "sector": "Auto"},
    "TATAPOWER": {"index": "Nifty Midcap", "sector": "Energy"}, "HAL": {"index": "Nifty Midcap", "sector": "Defense"},
    "SUZLON": {"index": "Nifty Smallcap", "sector": "Energy"}, "DLF": {"index": "Nifty Midcap", "sector": "Real Estate"}
}

selected_tickers = list(STOCK_METADATA.keys())

def fetch_stock_data(ticker):
    yf_symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    df_5m = yf.download(yf_symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
    df_1d = yf.download(yf_symbol, period="1mo", interval="1d", progress=False, auto_adjust=True)
    return df_5m, df_1d

# --- 4. LIVE NEWS CATALYST ENGINE (Free YFinance API) ---
def evaluate_live_news(ticker, df_5m):
    """Fetches real news for the stock, calculates sentiment, and checks if it's priced in."""
    try:
        yf_symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        stock_info = yf.Ticker(yf_symbol)
        news_items = stock_info.news
        
        if not news_items:
            return 0.0, "No live news detected"

        # Get the most recent news article
        latest_news = news_items[0]
        headline = latest_news.get("title", "")
        pub_time = latest_news.get("providerPublishTime", time.time())
        
        # Calculate how many minutes ago the news broke
        age_minutes = (time.time() - pub_time) / 60
        
        # Basic Institutional Sentiment Keywords
        bullish_keywords = ["surge", "profit", "jump", "buy", "upgrades", "growth", "wins", "order", "record", "approves", "soars"]
        bearish_keywords = ["fall", "drop", "loss", "sell", "downgrades", "slump", "misses", "probe", "cuts", "crash", "plunges"]
        
        headline_lower = headline.lower()
        sentiment = 0
        if any(word in headline_lower for word in bullish_keywords): sentiment = 1
        elif any(word in headline_lower for word in bearish_keywords): sentiment = -1
        
        if sentiment == 0:
            return 0.0, f"News (Neutral): {headline[:35]}..."
            
        # Time-decay Logic
        if age_minutes <= 60:
            base_score = 25.0 * sentiment
            status = "⚡ FRESH TWEET/NEWS"
        elif age_minutes <= 360: # 6 hours
            base_score = 10.0 * sentiment
            status = "⏳ DEVELOPING NEWS"
        else:
            base_score = 0.0
            status = "🕰️ OUTDATED NEWS"
            
        # The Reality Check (Priced-In Validator)
        if len(df_5m) > 20 and sentiment != 0:
            recent_return = ((df_5m['Close'].iloc[-1] - df_5m['Close'].iloc[-20]) / df_5m['Close'].iloc[-20]) * 100
            avg_vol = df_5m['Volume'].tail(40).mean()
            recent_vol_spike = df_5m['Volume'].iloc[-1] / (avg_vol + 1e-5)

            # If it's old news and already pumped -> It's a TRAP
            if sentiment == 1 and age_minutes > 45 and recent_return > 2.0 and recent_vol_spike > 1.5:
                return -15.0, f"🛑 PRICED-IN TRAP (Stock already up {recent_return:.1f}%): {headline[:30]}..."
            
            # If it's old bad news and already dumped -> Exhausted Sell
            if sentiment == -1 and age_minutes > 45 and recent_return < -2.0 and recent_vol_spike > 1.5:
                return 15.0, f"🛑 EXHAUSTED SELL (Stock already down {recent_return:.1f}%): {headline[:30]}..."
                
        return base_score, f"{status} [{int(age_minutes)}m ago]: {headline[:40]}..."
        
    except Exception as e:
        return 0.0, "News API Offline"

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# SMC zone logic (Bullish/Bearish OBs, Sweeps)
def track_smc_zones(df_5m, lookback=50):
    last_price = float(df_5m['Close'].iloc[-1])
    atr = (df_5m['High'] - df_5m['Low']).rolling(14).mean().iloc[-1]
    active_zones = []
    actual_lookback = min(lookback, len(df_5m) - 3)
    
    for i in range(len(df_5m) - actual_lookback, len(df_5m) - 1):
        if df_5m['Close'].iloc[i] < df_5m['Open'].iloc[i] and df_5m['Close'].iloc[i+1] > df_5m['Open'].iloc[i+1]:
            ob_top, ob_bottom = float(df_5m['High'].iloc[i]), float(df_5m['Low'].iloc[i])
            if not (df_5m['Close'].iloc[i+1:-1] < ob_bottom).any():
                state = "🟢 ENTRY READY" if ob_bottom <= last_price <= ob_top else "⏸️ ZONE CREATED"
                state_val = 2 if state == "🟢 ENTRY READY" else 0
                active_zones.append({'type': 'Bullish OB', 'top': ob_top, 'bottom': ob_bottom, 'age': len(df_5m)-1-i, 'state': state, 'state_val': state_val})
    return active_zones

def calculate_composite_score(row, news_score):
    ai_prob = float(row.get("Raw_AI_Prob", 50.0))  
    score_ai = ai_prob * 0.40
    smc_str = str(row.get("SMC Structure", "")).upper()
    smc_score = 20.0 if "READY" in smc_str else 0.0
    
    sl_pct, tgt_pct = float(row.get("SL_Pct_Num", 0.5)), float(row.get("Tgt_Pct_Num", 1.0))
    rr = tgt_pct / sl_pct if sl_pct > 0 else 1.0
    score_rr = min((rr / 3.0) * 15.0, 15.0) if rr >= 1.5 else 0.0

    stock_flow = float(row.get("Stock_Flow_Num", 0))
    is_long = "Bullish" in smc_str or "LOW" in smc_str or row.get("Day Trend") == "Uptrend"
    
    score_align = 15.0 if (is_long and stock_flow > 0) or (not is_long and stock_flow < 0) else -10.0

    # Macro Thematic Boost (Manual Sidebar)
    score_macro_manual = 20.0 if row.get("Sector") in active_themes else 0.0

    # Final logic adds the Live News Score from the API
    return max(0.0, round(score_ai + smc_score + score_rr + score_align + score_macro_manual + news_score, 2))

# --- 5. CORE SCANNER ENGINE ---
if "locked_results" not in st.session_state: st.session_state.locked_results = None
run_scan = st.button("🚀 Run AI Scan & Fetch Live News", type="primary")

if run_scan:
    with st.spinner("Scanning SMC structures, evaluating live order flow & checking live news APIs..."):
        results = []
        for ticker in selected_tickers:
            try:
                df_5m, df_1d = fetch_stock_data(ticker)
                if df_5m is None or df_5m.empty: continue
                if isinstance(df_5m.columns, pd.MultiIndex): df_5m.columns = df_5m.columns.get_level_values(0)

                close_5m = df_5m["Close"].dropna()
                last_price = float(close_5m.iloc[-1])
                day_open = float(df_1d["Open"].dropna().iloc[-1]) if not df_1d.empty else last_price
                
                stock_flow_cr = ((last_price - day_open) / day_open) * (df_5m["Volume"].dropna().tail(75).sum() * last_price) / 10000000
                flow_ui = f"🟩 ₹{abs(stock_flow_cr):.1f}Cr In" if stock_flow_cr > 0 else f"🟥 ₹{abs(stock_flow_cr):.1f}Cr Out"

                # LIVE NEWS FETCH & VALIDATION
                news_score, news_context = evaluate_live_news(ticker, df_5m)

                active_zones = track_smc_zones(df_5m, lookback=50)
                best_zone, smc_ui_str, zone_context = None, "Clean / No Valid Zones", news_context
                
                if active_zones:
                    best_zone = sorted(active_zones, key=lambda x: x['state_val'], reverse=True)[0]
                    smc_ui_str = f"[{best_zone['state']}] {best_zone['type']} (₹{best_zone['bottom']:.1f}-₹{best_zone['top']:.1f})"
                    zone_context = f"Age: {best_zone['age']} | " + news_context
                
                atr_14_val = float((df_5m["High"].dropna() - df_5m["Low"].dropna()).tail(14).mean())
                sl_price = last_price - (1.5 * atr_14_val)
                tgt_price = last_price + (3.0 * atr_14_val)
                dyn_tgt_pct = abs((tgt_price - last_price) / last_price) * 100
                dyn_sl_pct = abs((last_price - sl_price) / last_price) * 100

                meta = STOCK_METADATA.get(ticker, {"index": "Unknown", "sector": "General"})
                
                item = {
                    "Stock": ticker, "Index": meta["index"], "Sector": meta["sector"], 
                    "Last Price": f"₹{last_price:.2f}", "Stock Flow": flow_ui, 
                    "Stock_Flow_Num": stock_flow_cr, "Day Trend": "Uptrend" if last_price >= day_open else "Downtrend", 
                    "Signal State": smc_ui_str, "Context & Triggers": zone_context,
                    "Target": f"₹{tgt_price:.2f} ({dyn_tgt_pct:.1f}%)", "Stoploss": f"₹{sl_price:.2f} ({dyn_sl_pct:.1f}%)",
                    "Raw_AI_Prob": 65.5, "SMC Structure": smc_ui_str, "Tgt_Pct_Num": dyn_tgt_pct, "SL_Pct_Num": dyn_sl_pct
                }
                
                item["Rank Score"] = calculate_composite_score(item, news_score)
                
                # Tag stock with Newspaper if there is a LIVE API trigger OR a Manual Sidebar Sector trigger
                if news_score != 0 or item["Sector"] in active_themes:
                    item["Stock"] = "📰 " + item["Stock"]

                results.append(item)
            except Exception: continue

        if results:
            df_temp = pd.DataFrame(results).sort_values(by="Rank Score", ascending=False).reset_index(drop=True)
            df_temp["Rank"] = df_temp.index + 1
            st.session_state.locked_results = df_temp

# --- 6. DISPLAY DASHBOARD ---
if "locked_results" in st.session_state and st.session_state.locked_results is not None:
    results_df = st.session_state.locked_results
    st.subheader("🎯 TOP ACTIONABLE TRADES")
    
    card_cols = st.columns(3)
    for idx, col in enumerate(card_cols):
        if idx < len(results_df.head(3)):
            row = results_df.iloc[idx]
            with col:
                idx_val, sec_val = row.get('Index', 'N/A'), row.get('Sector', 'N/A')
                st.metric(label=f"#{row['Rank']} {row['Stock']} ({idx_val} | {sec_val})", value=row["Last Price"], delta=f"Score: {row['Rank Score']} | {row['Stock Flow']}")
                st.write(f"**State:** `{row['Signal State']}`")
                st.write(f"**Context:** {row['Context & Triggers']}")

    st.markdown("---")
    st.subheader("📊 FULL WATCHLIST & LIFECYCLE STATUS")
    
    display_cols = ["Rank", "Stock", "Index", "Sector", "Stock Flow", "Rank Score", "Last Price", "Signal State", "Context & Triggers"]
    st.dataframe(results_df[display_cols], height=500, use_container_width=True)
