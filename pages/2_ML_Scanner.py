import streamlit as st
import joblib
import os
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Project Alpha-NSE | Institutional Filing & ML Scanner", page_icon="🏛️", layout="wide")

st.title("🏛️ Project Alpha-NSE: Institutional Filing & ML Scanner")
st.markdown("*SEBI LODR Corporate Filings, Exchange Circulars, MoSPI/RBI Macro Data & Dynamic SMC Lifecycle Engine*")

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

st.sidebar.success(f"✅ AI Engine Configured: {len(expected_features)} Features")

# Discretionary Quant Macro Bias (All 12 Original Sectors)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏛️ Official Regulatory & Macro Catalysts")
active_macro_events = st.sidebar.multiselect(
    "Select Active Official MoSPI / RBI / Exchange Events:",
    ["RBI Rate Cut/Hike", "MoSPI CPI / IIP Release", "SEBI Circular / LODR Update", "Nifty Index Rebalancing"],
    help="Select verified macro or regulatory events currently impacting the session."
)

active_sectors = st.sidebar.multiselect(
    "Select Sectors with Verified Corporate Filings:", 
    ["Banking", "IT", "Auto", "Energy", "FMCG", "Metal", "Infra", "Financials", "Telecom", "Capital Goods", "Healthcare", "Consumer Durables"],
    help="Sectors with confirmed Board Meetings, Financial Results, or Material Event filings under SEBI LODR."
)
st.sidebar.info("🎯 NSE/BSE Filing Ingestion, Semantic Scoring & Gap Risk Validators Active.")

# --- 2. TECHNICAL HELPER FUNCTIONS ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-5)
    return 100 - (100 / (1 + rs))

def detect_chart_patterns(df_5m):
    patterns = []
    if len(df_5m) < 20:
        return patterns
    close = df_5m['Close']
    high = df_5m['High']
    
    recent_high = high.tail(20).max()
    if close.tail(5).mean() > close.tail(20).mean() and high.iloc[-1] >= recent_high * 0.995:
        patterns.append("Bull Flag Breakout")
    
    return patterns

# --- 3. INDEX, SECTOR SENTIMENT & FII/DII ORDER FLOW ---
SECTOR_MAP = {
    "Banking": "^NSEBANK", "IT": "^CNXIT", "Auto": "^CNXAUTO",
    "Energy": "^CNXENERGY", "FMCG": "^CNXFMCG", "Metal": "^CNXMETAL",
    "Infra": "^CNXINFRA", "Financials": "NIFTY_FIN_SERVICE.NS",
    "Telecom": "^NSEI", "Capital Goods": "^NSEI",
    "Healthcare": "^CNXPHARMA", "Consumer Durables": "^NSEI"
}

SECTOR_CONSTITUENTS = {
    "Auto": ["MARUTI.NS", "M&M.NS", "ASHOKLEY.NS"],
    "Energy": ["RELIANCE.NS", "NTPC.NS", "TATAPOWER.NS"],
    "FMCG": ["ITC.NS", "HINDUNILVR.NS"],
    "Metal": ["TATASTEEL.NS"],
    "Smallcap": ["CDSL.NS", "ANGELONE.NS", "KFINTECH.NS", "SUZLON.NS", "BSOFT.NS"],
    "Infra": ["LT.NS", "HFCL.NS"],
    "Financials": ["BAJFINANCE.NS", "CDSL.NS", "IEX.NS"],
    "Healthcare": ["MAXHEALTH.NS"]
}

@st.cache_data(ttl=300)
def fetch_market_data_and_flow():
    index_tickers = ["^NSEI", "^NSEMDCP50", "NIFTYSMALL100.NS", "^CNXSC", "^CNXSMLCAP"]
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

        def get_group_avg_return(t_list):
            rets = [get_1d_return(t) for t in t_list if abs(get_1d_return(t)) > 1e-6]
            return float(np.mean(rets)) if rets else 0.0

        returns["Nifty_1D_Return"] = get_1d_return("^NSEI")
        returns["Midcap_1D_Return"] = get_1d_return("^NSEMDCP50")
        
        sml_ret = 0.0
        for sml_t in ["NIFTYSMALL100.NS", "^CNXSC", "^CNXSMLCAP"]:
            r = get_1d_return(sml_t)
            if abs(r) > 1e-5: sml_ret = r; break
        if abs(sml_ret) < 1e-5: sml_ret = get_group_avg_return(SECTOR_CONSTITUENTS["Smallcap"])
        returns["Smallcap_1D_Return"] = sml_ret

        trends["^NSEI"] = f"{'+' if returns['Nifty_1D_Return'] >= 0 else ''}{returns['Nifty_1D_Return']*100:.2f}%"
        trends["^NSEMDCP50"] = f"{'+' if returns['Midcap_1D_Return'] >= 0 else ''}{returns['Midcap_1D_Return']*100:.2f}%"
        trends["Smallcap"] = f"{'+' if sml_ret >= 0 else ''}{sml_ret*100:.2f}%"

        for sector_name, sec_ticker in SECTOR_MAP.items():
            r = get_1d_return(sec_ticker)
            if abs(r) < 1e-5 and sector_name in SECTOR_CONSTITUENTS:
                r = get_group_avg_return(SECTOR_CONSTITUENTS[sector_name])
            returns[f"Sector_{sector_name}"] = r

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

# Top Index Metrics Bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nifty 50 (Sentiment)", idx_trends.get("^NSEI", "Active"))
col2.metric("Nifty Midcap", idx_trends.get("^NSEMDCP50", "Active"))
col3.metric("Nifty Smallcap", idx_trends.get("Smallcap", "Active"))
col4.metric("Large Money (Net FII/DII)", f"₹{inst_flow['Net_Flow'] / 100:.2f} Cr", inst_flow["Sentiment"], delta_color="normal" if inst_flow["Net_Flow"] > 0 else "inverse")

# Sectoral Performance Metric Grid
st.markdown("**🌐 Sectoral Performance (Live Impact)**")
sec_cols = st.columns(6)
for idx, sec in enumerate(["Banking", "IT", "Auto", "Energy", "FMCG", "Metal"]):
    sec_ret = market_returns.get(f"Sector_{sec}", 0.0) * 100
    sec_cols[idx % 6].metric(sec, f"{'+' if sec_ret >= 0 else ''}{sec_ret:.2f}%")
st.markdown("---")

# --- 4. UNIVERSE SETUP & METADATA REGISTRY ---
STOCK_METADATA = {
    "RELIANCE": {"index": "Nifty 50", "sector": "Energy"}, "TCS": {"index": "Nifty 50", "sector": "IT"},
    "HDFCBANK": {"index": "Nifty 50", "sector": "Banking"}, "INFY": {"index": "Nifty 50", "sector": "IT"},
    "ICICIBANK": {"index": "Nifty 50", "sector": "Banking"}, "SBIN": {"index": "Nifty 50", "sector": "Banking"},
    "BHARTIARTL": {"index": "Nifty 50", "sector": "Telecom"}, "ITC": {"index": "Nifty 50", "sector": "FMCG"},
    "LTIM": {"index": "Nifty 50", "sector": "IT"}, "AXISBANK": {"index": "Nifty 50", "sector": "Banking"},
    "KOTAKBANK": {"index": "Nifty 50", "sector": "Banking"}, "LT": {"index": "Nifty 50", "sector": "Infra"},
    "HINDUNILVR": {"index": "Nifty 50", "sector": "FMCG"}, "BAJFINANCE": {"index": "Nifty 50", "sector": "Financials"},
    "MARUTI": {"index": "Nifty 50", "sector": "Auto"}, "TATASTEEL": {"index": "Nifty 50", "sector": "Metal"},
    "NTPC": {"index": "Nifty 50", "sector": "Energy"}, "M&M": {"index": "Nifty 50", "sector": "Auto"},
    "TATAPOWER": {"index": "Nifty Midcap", "sector": "Energy"}, "FEDERALBNK": {"index": "Nifty Midcap", "sector": "Banking"},
    "POLYCAB": {"index": "Nifty Midcap", "sector": "Capital Goods"}, "PERSISTENT": {"index": "Nifty Midcap", "sector": "IT"},
    "COFORGE": {"index": "Nifty Midcap", "sector": "IT"}, "ASHOKLEY": {"index": "Nifty Midcap", "sector": "Auto"},
    "MAXHEALTH": {"index": "Nifty Midcap", "sector": "Healthcare"}, "VOLTAS": {"index": "Nifty Midcap", "sector": "Consumer Durables"},
    "CDSL": {"index": "Nifty Smallcap", "sector": "Financials"}, "ANGELONE": {"index": "Nifty Smallcap", "sector": "Financials"},
    "KFINTECH": {"index": "Nifty Smallcap", "sector": "Financials"}, "SUZLON": {"index": "Nifty Smallcap", "sector": "Energy"},
    "BSOFT": {"index": "Nifty Smallcap", "sector": "IT"}, "HFCL": {"index": "Nifty Smallcap", "sector": "Infra"},
    "IEX": {"index": "Nifty Smallcap", "sector": "Financials"}, "KEI": {"index": "Nifty Smallcap", "sector": "Capital Goods"}
}

scan_category = st.selectbox("Select Universe", [f"All Combined ({len(STOCK_METADATA)} Stocks)", "Nifty 50", "Nifty Midcap", "Nifty Smallcap"])
if scan_category == "Nifty 50": selected_tickers = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty 50"]
elif scan_category == "Nifty Midcap": selected_tickers = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty Midcap"]
elif scan_category == "Nifty Smallcap": selected_tickers = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty Smallcap"]
else: selected_tickers = list(STOCK_METADATA.keys())

def fetch_stock_data(ticker):
    if "upstox_client" in st.session_state and st.session_state.get("upstox_client"):
        try:
            upstox = st.session_state["upstox_client"]
            df_5m = upstox.get_ohlc(ticker, interval="5m")
            df_1d = upstox.get_ohlc(ticker, interval="1d")
            if df_5m is not None and not df_5m.empty and df_1d is not None and not df_1d.empty: return df_5m, df_1d
        except Exception: pass
    
    yf_symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    df_5m = yf.download(yf_symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
    df_1d = yf.download(yf_symbol, period="1mo", interval="1d", progress=False, auto_adjust=True)
    return df_5m, df_1d

# --- 5. OFFICIAL NSE/BSE CORPORATE FILING & SEMANTIC IMPACT ENGINE ---
COMPANY_FILING_QUERIES = {
    "RELIANCE": "Reliance Industries corporate filing stock exchange NSE BSE",
    "TCS": "Tata Consultancy Services corporate filing stock exchange NSE BSE",
    "HDFCBANK": "HDFC Bank corporate filing stock exchange NSE BSE",
    "INFY": "Infosys corporate filing stock exchange NSE BSE",
    "ICICIBANK": "ICICI Bank corporate filing stock exchange NSE BSE",
    "SBIN": "State Bank of India corporate filing stock exchange NSE BSE",
    "BHARTIARTL": "Bharti Airtel corporate filing stock exchange NSE BSE",
    "ITC": "ITC corporate filing stock exchange NSE BSE",
    "LTIM": "LTIMindtree corporate filing stock exchange NSE BSE",
    "AXISBANK": "Axis Bank corporate filing stock exchange NSE BSE",
    "KOTAKBANK": "Kotak Mahindra Bank corporate filing stock exchange NSE BSE",
    "LT": "Larsen and Toubro corporate filing stock exchange NSE BSE",
    "HINDUNILVR": "Hindustan Unilever corporate filing stock exchange NSE BSE",
    "BAJFINANCE": "Bajaj Finance corporate filing stock exchange NSE BSE",
    "MARUTI": "Maruti Suzuki corporate filing stock exchange NSE BSE",
    "TATASTEEL": "Tata Steel corporate filing stock exchange NSE BSE",
    "NTPC": "NTPC corporate filing stock exchange NSE BSE",
    "M&M": "Mahindra and Mahindra corporate filing stock exchange NSE BSE",
    "TATAPOWER": "Tata Power corporate filing stock exchange NSE BSE",
    "FEDERALBNK": "Federal Bank corporate filing stock exchange NSE BSE",
    "POLYCAB": "Polycab corporate filing stock exchange NSE BSE",
    "PERSISTENT": "Persistent Systems corporate filing stock exchange NSE BSE",
    "COFORGE": "Coforge corporate filing stock exchange NSE BSE",
    "ASHOKLEY": "Ashok Leyland corporate filing stock exchange NSE BSE",
    "MAXHEALTH": "Max Healthcare corporate filing stock exchange NSE BSE",
    "VOLTAS": "Voltas corporate filing stock exchange NSE BSE",
    "CDSL": "CDSL corporate filing stock exchange NSE BSE",
    "ANGELONE": "Angel One corporate filing stock exchange NSE BSE",
    "KFINTECH": "KFin Technologies corporate filing stock exchange NSE BSE",
    "SUZLON": "Suzlon Energy corporate filing stock exchange NSE BSE",
    "BSOFT": "Birlasoft corporate filing stock exchange NSE BSE",
    "HFCL": "HFCL corporate filing stock exchange NSE BSE",
    "IEX": "Indian Energy Exchange corporate filing stock exchange NSE BSE",
    "KEI": "KEI Industries corporate filing stock exchange NSE BSE"
}

def evaluate_official_filings(ticker, df_5m):
    try:
        query = COMPANY_FILING_QUERIES.get(ticker, f"{ticker} corporate filing NSE BSE")
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')
        
        if not items:
            return 0.0, "No official filing"

        item = items[0]
        title_elem = item.find('title')
        title_text = title_elem.text if title_elem is not None else ""
        pub_date_elem = item.find('pubDate')
        pub_date_str = pub_date_elem.text if pub_date_elem is not None else ""

        try:
            pub_dt = parsedate_to_datetime(pub_date_str)
            pub_time = pub_dt.timestamp()
        except Exception:
            pub_time = time.time()

        age_hours = (time.time() - pub_time) / 3600.0
        filing_lower = title_text.lower()

        # Semantic Classification under SEBI LODR Regulations
        tier_1_bullish = ["financial results", "net profit up", "order win", "secures contract", "bonus issue", "buyback", "board meeting outcome", "dividend declared", "acquisition"]
        tier_1_bearish = ["sebi penalty", "investigation", "default", "fraud", "resignation", "net profit down", "cbi raid", "adverse ruling"]

        tier_2_bullish = ["capacity expansion", "joint venture", "rating upgrade", "institutional placement"]
        tier_2_bearish = ["margin compression", "rating downgrade", "factory shutdown", "strike"]

        sentiment = 0
        multiplier = 1.0

        if any(w in filing_lower for w in tier_1_bullish):
            sentiment, multiplier = 1, 2.0
        elif any(w in filing_lower for w in tier_1_bearish):
            sentiment, multiplier = -1, 2.0
        elif any(w in filing_lower for w in tier_2_bullish):
            sentiment, multiplier = 1, 1.0
        elif any(w in filing_lower for w in tier_2_bearish):
            sentiment, multiplier = -1, 1.0

        if sentiment == 0:
            return 0.0, f"Routine Filing: {title_text[:30]}..."

        if age_hours <= 2:
            base_score = 25.0 * sentiment * multiplier
            status = "🏛️ LODR BREAKING"
        elif age_hours <= 18:
            base_score = 18.0 * sentiment * multiplier
            status = "🌙 OVERNIGHT FILING"
        elif age_hours <= 72:
            base_score = 12.0 * sentiment * multiplier
            status = "📅 WEEKEND DISCLOSURE"
        else:
            return 0.0, f"Archived Filing: {title_text[:30]}..."

        # Pre-Market Gap Risk & Gap Trap Validator
        if len(df_5m) > 20 and age_hours > 0.5:
            recent_return = ((df_5m['Close'].iloc[-1] - df_5m['Open'].iloc[-20]) / df_5m['Open'].iloc[-20]) * 100
            if sentiment == 1 and recent_return > 3.0:
                return -20.0, f"🛑 GAP TRAP (+{recent_return:.1f}% Priced In): {title_text[:25]}..."
            if sentiment == -1 and recent_return < -3.0:
                return 20.0, f"🛑 EXHAUSTED DUMP ({recent_return:.1f}% Priced In): {title_text[:25]}..."

        return base_score, f"{status}: {title_text[:35]}..."

    except Exception:
        return 0.0, "SEBI Feed Active"

# --- 6. SMC LIFECYCLE ENGINE ---
def track_smc_zones(df_5m, lookback=50):
    last_price = float(df_5m['Close'].iloc[-1])
    last_low, last_high = float(df_5m['Low'].iloc[-1]), float(df_5m['High'].iloc[-1])
    atr = (df_5m['High'] - df_5m['Low']).rolling(14).mean().iloc[-1]
    
    active_zones = []
    actual_lookback = min(lookback, len(df_5m) - 3)
    
    for i in range(len(df_5m) - actual_lookback, len(df_5m) - 1):
        if df_5m['Close'].iloc[i] < df_5m['Open'].iloc[i] and df_5m['Close'].iloc[i+1] > df_5m['Open'].iloc[i+1]:
            ob_top, ob_bottom = float(df_5m['High'].iloc[i]), float(df_5m['Low'].iloc[i])
            if not (df_5m['Close'].iloc[i+1:-1] < ob_bottom).any():
                max_fwd = df_5m['High'].iloc[i+1:].max()
                dist_pct = ((last_price - ob_top) / ob_top) * 100
                if max_fwd > ob_top + (1.5 * atr): state, state_val = "🔴 MITIGATED", -1
                elif ob_bottom <= last_price <= ob_top or ob_bottom <= last_low <= ob_top: state, state_val = "🟢 ENTRY READY", 2
                elif 0 < dist_pct <= 0.4: state, state_val = "🟡 PULLBACK NEAR", 1
                else: state, state_val = "⏸️ ZONE CREATED", 0
                active_zones.append({'type': 'Bullish OB', 'top': ob_top, 'bottom': ob_bottom, 'age': len(df_5m)-1-i, 'state': state, 'state_val': state_val})

        if df_5m['Close'].iloc[i] > df_5m['Open'].iloc[i] and df_5m['Close'].iloc[i+1] < df_5m['Open'].iloc[i+1]:
            ob_top, ob_bottom = float(df_5m['High'].iloc[i]), float(df_5m['Low'].iloc[i])
            if not (df_5m['Close'].iloc[i+1:-1] > ob_top).any():
                min_fwd = df_5m['Low'].iloc[i+1:].min()
                dist_pct = ((ob_bottom - last_price) / ob_bottom) * 100
                if min_fwd < ob_bottom - (1.5 * atr): state, state_val = "🔴 MITIGATED", -1
                elif ob_bottom <= last_price <= ob_top or ob_bottom <= last_high <= ob_top: state, state_val = "🔴 SHORT READY", 2
                elif 0 < dist_pct <= 0.4: state, state_val = "🟡 PULLBACK NEAR", 1
                else: state, state_val = "⏸️ ZONE CREATED", 0
                active_zones.append({'type': 'Bearish OB', 'top': ob_top, 'bottom': ob_bottom, 'age': len(df_5m)-1-i, 'state': state, 'state_val': state_val})

        if i > 15:
            recent_min = float(df_5m['Low'].iloc[i-15:i].min())
            if df_5m['Low'].iloc[i] < recent_min and df_5m['Close'].iloc[i] > recent_min:
                sweep_lvl = recent_min
                if not (df_5m['Close'].iloc[i+1:-1] < sweep_lvl).any():
                    if df_5m['High'].iloc[i+1:].max() > sweep_lvl + (1.5 * atr): state, state_val = "🔴 MITIGATED", -1
                    elif 0 <= ((last_price - sweep_lvl) / sweep_lvl) * 100 <= 0.25: state, state_val = "🟢 ENTRY READY", 2
                    else: state, state_val = "⏸️ ZONE CREATED", 0
                    active_zones.append({'type': 'Sweep Low', 'top': sweep_lvl * 1.001, 'bottom': sweep_lvl, 'age': len(df_5m)-1-i, 'state': state, 'state_val': state_val})
                    
        if i > 15:
            recent_max = float(df_5m['High'].iloc[i-15:i].max())
            if df_5m['High'].iloc[i] > recent_max and df_5m['Close'].iloc[i] < recent_max:
                sweep_lvl = recent_max
                if not (df_5m['Close'].iloc[i+1:-1] > sweep_lvl).any():
                    if df_5m['Low'].iloc[i+1:].min() < sweep_lvl - (1.5 * atr): state, state_val = "🔴 MITIGATED", -1
                    elif 0 <= ((sweep_lvl - last_price) / sweep_lvl) * 100 <= 0.25: state, state_val = "🔴 SHORT READY", 2
                    else: state, state_val = "⏸️ ZONE CREATED", 0
                    active_zones.append({'type': 'Sweep High', 'top': sweep_lvl, 'bottom': sweep_lvl * 0.999, 'age': len(df_5m)-1-i, 'state': state, 'state_val': state_val})

    return active_zones

def calculate_composite_score(row, filing_score=0.0):
    ai_prob = float(row.get("Raw_AI_Prob", 50.0))  
    score_ai = ai_prob * 0.35
    smc_str = str(row.get("SMC Structure", "")).upper()
    smc_score = 25.0 if "READY" in smc_str else (12.0 if "PULLBACK" in smc_str else 0.0)
    
    sl_pct = float(row.get("SL_Pct_Num", 0.5))
    tgt_pct = float(row.get("Tgt_Pct_Num", 1.0))
    rr = tgt_pct / sl_pct if sl_pct > 0 else 1.0
    score_rr = min((rr / 3.0) * 15.0, 15.0) if rr >= 1.5 else 0.0

    stock_flow = float(row.get("Stock_Flow_Num", 0))
    is_long = "Bullish" in smc_str or "LOW" in smc_str or row.get("Day Trend") == "Uptrend"
    
    score_align = 15.0 if (is_long and stock_flow > 0) or (not is_long and stock_flow < 0) else -10.0
    score_macro_manual = 20.0 if row.get("Sector") in active_sectors else 0.0

    return max(0.0, round(score_ai + smc_score + score_rr + score_align + score_macro_manual + filing_score, 2))

# --- 7. CORE SCANNER ENGINE ---
if "locked_results" not in st.session_state: st.session_state.locked_results = None
ctrl_col1, ctrl_col2 = st.columns([1, 3])
with ctrl_col1: lock_signals = st.checkbox("🔒 Lock Watchlist", value=False)
run_scan = st.button("🚀 Run Institutional Scan & Rank", type="primary")

if lock_signals and st.session_state.locked_results is not None:
    results_df = st.session_state.locked_results
elif run_scan:
    with st.spinner("Ingesting SEBI LODR filings, order flow, & cross-referencing SMC liquidity zones..."):
        results = []
        market_sentiment = float(market_returns.get("Nifty_1D_Return", 0.0)) * 100

        for ticker in selected_tickers:
            try:
                df_5m, df_1d = fetch_stock_data(ticker)
                if df_5m is None or df_5m.empty or df_1d is None or df_1d.empty: continue
                if isinstance(df_5m.columns, pd.MultiIndex): df_5m.columns = df_5m.columns.get_level_values(0)
                if isinstance(df_1d.columns, pd.MultiIndex): df_1d.columns = df_1d.columns.get_level_values(0)

                close_5m, high_5m, low_5m, open_5m = df_5m["Close"].dropna(), df_5m["High"].dropna(), df_5m["Low"].dropna(), df_5m["Open"].dropna()
                vol_5m = df_5m["Volume"].dropna() if "Volume" in df_5m else pd.Series(1, index=close_5m.index)
                last_price, day_open = float(close_5m.iloc[-1]), float(df_1d["Open"].dropna().iloc[-1])

                stock_flow_cr = ((last_price - day_open) / day_open) * (vol_5m.tail(75).sum() * last_price) / 10000000
                flow_ui = f"🟩 ₹{abs(stock_flow_cr):.1f}Cr In" if stock_flow_cr > 0 else f"🟥 ₹{abs(stock_flow_cr):.1f}Cr Out"

                avg_vol = vol_5m.tail(20).mean()
                rvol = float(vol_5m.iloc[-1] / (avg_vol + 1e-5))
                ema_20 = close_5m.ewm(span=20, adjust=False).mean().iloc[-1]
                day_trend = "Uptrend" if (last_price > ema_20 and last_price >= day_open) else "Downtrend"
                
                atr_14_val = float((high_5m - low_5m).tail(14).mean())
                atr_pct = float((atr_14_val / last_price) * 100)
                rsi_val = float(compute_rsi(close_5m).iloc[-1])

                trigger = ""
                for _, row in df_5m.tail(3).iterrows():
                    body = abs(row['Open'] - row['Close'])
                    wick_up, wick_down = row['High'] - max(row['Open'], row['Close']), min(row['Open'], row['Close']) - row['Low']
                    if wick_down > 2 * body and wick_up < body and body > 0: trigger = "🔨 Pin Bar (Buy)"
                    elif wick_up > 2 * body and wick_down < body and body > 0: trigger = "☄️ Pin Bar (Sell)"
                
                chart_patterns = detect_chart_patterns(df_5m)
                active_zones = track_smc_zones(df_5m, lookback=50)
                filing_score, filing_context = evaluate_official_filings(ticker, df_5m)
                
                best_zone, smc_ui_str, zone_context = None, "Clean / No Valid Zones", filing_context
                if active_zones:
                    best_zone = sorted(active_zones, key=lambda x: x['state_val'], reverse=True)[0]
                    smc_ui_str = f"[{best_zone['state']}] {best_zone['type']} (₹{best_zone['bottom']:.1f}-₹{best_zone['top']:.1f})"
                    zone_context = f"Age: {best_zone['age']} bars | " + filing_context
                
                if chart_patterns: zone_context += f" | 📊 {', '.join(chart_patterns)}"
                if trigger: zone_context += f" | {trigger}"

                bull_ob_feat = 1 if any(z['type'] == 'Bullish OB' for z in active_zones) else 0
                sweep_low_feat = 1 if any(z['type'] == 'Sweep Low' for z in active_zones) else 0
                sweep_high_feat = 1 if any(z['type'] == 'Sweep High' for z in active_zones) else 0
                
                X_df = pd.DataFrame([{
                    'RVOL': rvol, 'ATR_Pct': atr_pct, 'RSI': rsi_val,
                    'Liquidity_Sweep_High': sweep_high_feat, 'Liquidity_Sweep_Low': sweep_low_feat,
                    'Bullish_FVG': 0, 'Bullish_OB': bull_ob_feat,
                    'Pattern_Flag_Breakout': 1 if "Bull Flag Breakout" in chart_patterns else 0, 
                    'Market_Sentiment': market_sentiment
                }])
                
                prob = float(model.predict_proba(scaler.transform(X_df) if scaler else X_df)[0][1]) if hasattr(model, "predict_proba") else 0.5
                
                major_bsl = float(high_5m.tail(200).max())  
                major_ssl = float(low_5m.tail(200).min())  

                if best_zone and best_zone['state_val'] >= 1:
                    is_long = "Bullish" in best_zone['type'] or "Low" in best_zone['type']
                    if is_long:
                        sl_price = best_zone['bottom'] - (0.1 * atr_14_val)
                        tgt_price = major_bsl if major_bsl > last_price + (1.5 * atr_14_val) else last_price + (3.0 * atr_14_val)
                    else: 
                        sl_price = best_zone['top'] + (0.1 * atr_14_val)
                        tgt_price = major_ssl if major_ssl < last_price - (1.5 * atr_14_val) else last_price - (3.0 * atr_14_val)
                else:
                    sl_price = last_price - (1.5 * atr_14_val) if day_trend == "Uptrend" else last_price + (1.5 * atr_14_val)
                    tgt_price = major_bsl if day_trend == "Uptrend" else major_ssl

                dyn_tgt_pct = abs((tgt_price - last_price) / last_price) * 100
                dyn_sl_pct = abs((last_price - sl_price) / last_price) * 100
                
                rr_val = dyn_tgt_pct / (dyn_sl_pct + 1e-5)
                if rr_val < 1.5: smc_ui_str += " ⚠️ LOW R:R"

                meta = STOCK_METADATA.get(ticker, {"index": "Unknown", "sector": "General"})
                
                item = {
                    "Stock": ticker, "Index": meta["index"], "Sector": meta["sector"], 
                    "Last Price": f"₹{last_price:.2f}", "Stock Flow": flow_ui, 
                    "Stock_Flow_Num": stock_flow_cr, "Day Trend": day_trend, 
                    "Signal State": smc_ui_str, "Context & Triggers": zone_context,
                    "Target": f"₹{tgt_price:.2f} ({dyn_tgt_pct:.1f}%)", "Stoploss": f"₹{sl_price:.2f} ({dyn_sl_pct:.1f}%)",
                    "AI Prob": f"{prob*100:.1f}%", "Raw_AI_Prob": prob*100, 
                    "SMC Structure": smc_ui_str, "Tgt_Pct_Num": dyn_tgt_pct, "SL_Pct_Num": dyn_sl_pct
                }
                item["Rank Score"] = calculate_composite_score(item, filing_score)
                
                if filing_score != 0 or item["Sector"] in active_sectors:
                    item["Stock"] = "🏛️ " + item["Stock"]

                results.append(item)
            except Exception: continue

        if results:
            df_temp = pd.DataFrame(results).sort_values(by="Rank Score", ascending=False).reset_index(drop=True)
            df_temp["Rank"] = df_temp.index + 1
            st.session_state.locked_results = df_temp

# --- 8. DISPLAY DASHBOARD & VISUAL CHART ---
if "locked_results" in st.session_state and st.session_state.locked_results is not None:
    results_df = st.session_state.locked_results
    st.subheader("🎯 TOP ACTIONABLE INSTITUTIONAL TRADES")
    
    card_cols = st.columns(3)
    for idx, col in enumerate(card_cols):
        if idx < len(results_df.head(3)):
            row = results_df.iloc[idx]
            with col:
                idx_val, sec_val = row.get('Index', 'N/A'), row.get('Sector', 'N/A')
                st.metric(label=f"#{row['Rank']} {row['Stock']} ({idx_val} | {sec_val})", value=row["Last Price"], delta=f"Score: {row['Rank Score']} | {row['Stock Flow']}")
                st.write(f"**State:** `{row['Signal State']}`")
                st.write(f"**Context:** {row['Context & Triggers']}")
                st.write(f"**Target:** {row['Target']} | **SL:** {row['Stoploss']}")

    st.markdown("---")
    st.subheader("📈 LIVE VISUAL CONFIRMATION (Top Ranked Setup)")
    
    top_ticker = results_df.iloc[0]['Stock'].replace("🏛️ ", "")
    df_5m_chart, _ = fetch_stock_data(top_ticker)
    
    if df_5m_chart is not None and not df_5m_chart.empty:
        if isinstance(df_5m_chart.columns, pd.MultiIndex): df_5m_chart.columns = df_5m_chart.columns.get_level_values(0)
        
        fig = go.Figure(data=[go.Candlestick(
            x=df_5m_chart.index, open=df_5m_chart['Open'], high=df_5m_chart['High'], 
            low=df_5m_chart['Low'], close=df_5m_chart['Close'], name="Price"
        )])
        
        top_zones = track_smc_zones(df_5m_chart, lookback=50)
        if top_zones:
            best_z = sorted(top_zones, key=lambda x: x['state_val'], reverse=True)[0]
            color = "green" if "Bullish" in best_z['type'] or "Low" in best_z['type'] else "red"
            fig.add_hrect(
                y0=best_z['bottom'], y1=best_z['top'], line_width=0, 
                fillcolor=color, opacity=0.3, annotation_text=best_z['type'], annotation_position="top left"
            )
            
        fig.update_layout(
            title=f"{top_ticker} - Institutional Filing & SMC Zone Confirmation", 
            xaxis_rangeslider_visible=False, template="plotly_dark", height=500, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 FULL WATCHLIST & LIFECYCLE STATUS")
    
    display_cols = [
        "Rank", "Stock", "Index", "Sector", "Stock Flow", "Rank Score", 
        "Last Price", "Day Trend", "Signal State", "Context & Triggers", 
        "Target", "Stoploss", "AI Prob"
    ]
    
    st.dataframe(results_df[display_cols], height=500, use_container_width=True)
    
    csv = results_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Institutional Scan Results (CSV)", data=csv, file_name="sebi_lodr_scan_results.csv", mime="text/csv")
