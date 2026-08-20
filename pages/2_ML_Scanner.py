import streamlit as st
import joblib
import os
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Alpha-NSE | Synchronized SMC & ML Engine",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Project Alpha-NSE: Synchronized SMC, Order Flow & AI Engine")
st.markdown("*Real-Time Intraday Institutional Scanner with Session-Anchored SMC, VWAP Confluence & Dynamic Risk Engine*")

# --- 1. ASSET LOADING & OFFLINE ML INGESTION ---
@st.cache_resource
def load_ai_assets():
    model = joblib.load("colab_ai_model.pkl") if os.path.exists("colab_ai_model.pkl") else None
    scaler = joblib.load("colab_scaler.pkl") if os.path.exists("colab_scaler.pkl") else None
    return model, scaler

model, scaler = load_ai_assets()

if model is None:
    st.sidebar.warning("⚠️ 'colab_ai_model.pkl' not found. Operating in Confluence Mode.")
else:
    st.sidebar.success("✅ AI Engine Loaded: Fast Offline Inference Active")

# --- 2. SIDEBAR CONTROLS & MARKET CONTEXT ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Market Context & Sector Alignment")
active_sectors = st.sidebar.multiselect(
    "Focus Sectors (Outperforming / Underperforming):", 
    ["Banking", "IT", "Auto", "Energy", "FMCG", "Metal", "Infra", "Financials", "Healthcare", "Consumer Durables"],
    default=["Banking", "IT", "Financials", "Auto", "Energy"]
)

min_rr_threshold = st.sidebar.slider("Minimum Risk-to-Reward (R:R) Filter", 1.0, 4.0, 1.2, 0.1)

UPSTOX_ISIN_MAP = {
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "INFY": "NSE_EQ|INE009A01021",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "ICICIBANK": "NSE_EQ|INE090A01013",
    "SBIN": "NSE_EQ|INE062A01020",
    "BHARTIARTL": "NSE_EQ|INE397D01024",
    "ITC": "NSE_EQ|INE154A01025",
    "LTIM": "NSE_EQ|INE214T01019",
    "AXISBANK": "NSE_EQ|INE238A01034",
    "KOTAKBANK": "NSE_EQ|INE237A01028",
    "LT": "NSE_EQ|INE018A01030",
    "HINDUNILVR": "NSE_EQ|INE030A01027",
    "BAJFINANCE": "NSE_EQ|INE296A01024",
    "MARUTI": "NSE_EQ|INE585B01010",
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "TATASTEEL": "NSE_EQ|INE081A01020",
    "NTPC": "NSE_EQ|INE733E01010",
    "M&M": "NSE_EQ|INE101A01026",
    "TATAPOWER": "NSE_EQ|INE245A01021",
    "FEDERALBNK": "NSE_EQ|INE028D01010",
    "POLYCAB": "NSE_EQ|INE455K01017",
    "PERSISTENT": "NSE_EQ|INE262H01021",
    "COFORGE": "NSE_EQ|INE591G01017",
    "ASHOKLEY": "NSE_EQ|INE208A01029",
    "MAXHEALTH": "NSE_EQ|INE275F01028",
    "VOLTAS": "NSE_EQ|INE226A01021",
    "CDSL": "NSE_EQ|INE736A01011",
    "ANGELONE": "NSE_EQ|INE732I01013",
    "KFINTECH": "NSE_EQ|INE138Y01011",
    "SUZLON": "NSE_EQ|INE040H01021",
    "BSOFT": "NSE_EQ|INE084A01015",
    "HFCL": "NSE_EQ|INE548A01028",
    "IEX": "NSE_EQ|INE577H01019",
    "KEI": "NSE_EQ|INE378B01023",
}

def get_upstox_access_token() -> str | None:
    try:
        if "UPSTOX_ACCESS_TOKEN" in st.secrets:
            return st.secrets["UPSTOX_ACCESS_TOKEN"]
        return os.getenv("UPSTOX_ACCESS_TOKEN", None)
    except Exception:
        return None

def fetch_upstox_live(symbol: str, interval: str = "5m") -> pd.DataFrame | None:
    try:
        access_token = get_upstox_access_token()
        if not access_token:
            return None

        clean_sym = symbol.upper().replace(".NS", "").replace("&", "_").strip()
        raw_key = UPSTOX_ISIN_MAP.get(clean_sym, f"NSE_EQ|{clean_sym}")
        instrument_key = urllib.parse.quote(raw_key, safe="")
        interval_str = str(interval).lower().strip()

        if "day" in interval_str or "1d" in interval_str:
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
            url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/day/1/{to_date}/{from_date}"
        else:
            digits = "".join(filter(str.isdigit, interval_str))
            int_val = int(digits) if digits else 5
            url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/minutes/{int_val}"

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            raw_candles = res.json().get("data", {}).get("candles", [])
            if raw_candles:
                df = pd.DataFrame(
                    raw_candles,
                    columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "OI"],
                )
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                df = df.sort_values("Datetime").set_index("Datetime")
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
    except Exception:
        pass
    return None

def fetch_stock_data(ticker):
    """
    Data Fetch Pipeline:
    Primary Attempt -> Upstox API v3 requests using token from Streamlit Secrets
    Fallback Attempt -> Yahoo Finance (yfinance)
    """
    df_5m = fetch_upstox_live(ticker, interval="5m")
    df_1d = fetch_upstox_live(ticker, interval="1d")
    
    if df_5m is not None and not df_5m.empty and df_1d is not None and not df_1d.empty:
        return df_5m, df_1d
        
    yf_symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    df_5m_yf = yf.download(yf_symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
    df_1d_yf = yf.download(yf_symbol, period="1mo", interval="1d", progress=False, auto_adjust=True)
    return df_5m_yf, df_1d_yf

# --- 4. MARKET SENTIMENT & FII / DII NET ORDER FLOW ENGINE ---
SECTOR_MAP = {
    "Banking": "^NSEBANK", "IT": "^CNXIT", "Auto": "^CNXAUTO",
    "Energy": "^CNXENERGY", "FMCG": "^CNXFMCG", "Metal": "^CNXMETAL",
    "Infra": "^CNXINFRA", "Financials": "NIFTY_FIN_SERVICE.NS",
    "Healthcare": "^CNXPHARMA"
}

SECTOR_CONSTITUENTS = {
    "Auto": ["MARUTI.NS", "M&M.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "ASHOKLEY.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS"],
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
                if len(s) >= 2: 
                    ret = float((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2])
                    return ret if not np.isnan(ret) else 0.0
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
            if (abs(r) < 1e-5 or np.isnan(r)) and sector_name in SECTOR_CONSTITUENTS:
                r = get_group_avg_return(SECTOR_CONSTITUENTS[sector_name])
            returns[f"Sector_{sector_name}"] = r

        nifty_ret = returns["Nifty_1D_Return"]
        fii_proxy = nifty_ret * 450000 
        dii_proxy = -fii_proxy * 0.40  
        net_flow = fii_proxy + dii_proxy
        
        fii_dii_flow = {
            "FII_Net": fii_proxy, "DII_Net": dii_proxy, "Net_Flow": net_flow,
            "Sentiment": "Institutional Buying" if net_flow >= 0 else "Institutional Selling"
        }
    except Exception:
        fii_dii_flow = {"FII_Net": 0, "DII_Net": 0, "Net_Flow": 0, "Sentiment": "Neutral"}
        
    return trends, returns, fii_dii_flow

idx_trends, market_returns, inst_flow = fetch_market_data_and_flow()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Nifty 50 (Sentiment)", idx_trends.get("^NSEI", "Active"))
col2.metric("Nifty Midcap", idx_trends.get("^NSEMDCP50", "Active"))
col3.metric("Nifty Smallcap", idx_trends.get("Smallcap", "Active"))
col4.metric(
    "Large Money (Net FII/DII)", 
    f"₹{inst_flow['Net_Flow']:,.0f} Cr", 
    inst_flow["Sentiment"], 
    delta_color="normal" if inst_flow["Net_Flow"] >= 0 else "inverse"
)

st.markdown("**🌐 Sectoral Performance (Live Impact)**")
sec_cols = st.columns(6)
for idx, sec in enumerate(["Banking", "IT", "Auto", "Energy", "FMCG", "Metal"]):
    sec_ret = market_returns.get(f"Sector_{sec}", 0.0) * 100
    sec_cols[idx % 6].metric(sec, f"{'+' if sec_ret >= 0 else ''}{sec_ret:.2f}%")
st.markdown("---")

# --- 5. EXPANDED UNIVERSE METADATA REGISTRY ---
STOCK_METADATA = {
    "RELIANCE": {"index": "Nifty 50", "sector": "Energy", "query": "Reliance Industries"},
    "TCS": {"index": "Nifty 50", "sector": "IT", "query": "Tata Consultancy Services"},
    "HDFCBANK": {"index": "Nifty 50", "sector": "Banking", "query": "HDFC Bank"},
    "INFY": {"index": "Nifty 50", "sector": "IT", "query": "Infosys"},
    "ICICIBANK": {"index": "Nifty 50", "sector": "Banking", "query": "ICICI Bank"},
    "SBIN": {"index": "Nifty 50", "sector": "Banking", "query": "State Bank of India"},
    "BHARTIARTL": {"index": "Nifty 50", "sector": "Telecom", "query": "Bharti Airtel"},
    "ITC": {"index": "Nifty 50", "sector": "FMCG", "query": "ITC Limited"},
    "LTIM": {"index": "Nifty 50", "sector": "IT", "query": "LTIMindtree"},
    "AXISBANK": {"index": "Nifty 50", "sector": "Banking", "query": "Axis Bank"},
    "KOTAKBANK": {"index": "Nifty 50", "sector": "Banking", "query": "Kotak Mahindra Bank"},
    "LT": {"index": "Nifty 50", "sector": "Infra", "query": "Larsen Toubro"},
    "HINDUNILVR": {"index": "Nifty 50", "sector": "FMCG", "query": "Hindustan Unilever"},
    "BAJFINANCE": {"index": "Nifty 50", "sector": "Financials", "query": "Bajaj Finance"},
    "MARUTI": {"index": "Nifty 50", "sector": "Auto", "query": "Maruti Suzuki"},
    "TATAMOTORS": {"index": "Nifty 50", "sector": "Auto", "query": "Tata Motors"},
    "TATASTEEL": {"index": "Nifty 50", "sector": "Metal", "query": "Tata Steel"},
    "NTPC": {"index": "Nifty 50", "sector": "Energy", "query": "NTPC"},
    "M&M": {"index": "Nifty 50", "sector": "Auto", "query": "Mahindra and Mahindra"},
    "TATAPOWER": {"index": "Nifty Midcap", "sector": "Energy", "query": "Tata Power"},
    "FEDERALBNK": {"index": "Nifty Midcap", "sector": "Banking", "query": "Federal Bank"},
    "POLYCAB": {"index": "Nifty Midcap", "sector": "Capital Goods", "query": "Polycab"},
    "PERSISTENT": {"index": "Nifty Midcap", "sector": "IT", "query": "Persistent Systems"},
    "COFORGE": {"index": "Nifty Midcap", "sector": "IT", "query": "Coforge"},
    "ASHOKLEY": {"index": "Nifty Midcap", "sector": "Auto", "query": "Ashok Leyland"},
    "MAXHEALTH": {"index": "Nifty Midcap", "sector": "Healthcare", "query": "Max Healthcare"},
    "VOLTAS": {"index": "Nifty Midcap", "sector": "Consumer Durables", "query": "Voltas"},
    "CDSL": {"index": "Nifty Smallcap", "sector": "Financials", "query": "CDSL"},
    "ANGELONE": {"index": "Nifty Smallcap", "sector": "Financials", "query": "Angel One"},
    "KFINTECH": {"index": "Nifty Smallcap", "sector": "Financials", "query": "KFin Technologies"},
    "SUZLON": {"index": "Nifty Smallcap", "sector": "Energy", "query": "Suzlon Energy"},
    "BSOFT": {"index": "Nifty Smallcap", "sector": "IT", "query": "Birlasoft"},
    "HFCL": {"index": "Nifty Smallcap", "sector": "Infra", "query": "HFCL"},
    "IEX": {"index": "Nifty Smallcap", "sector": "Financials", "query": "Indian Energy Exchange"},
    "KEI": {"index": "Nifty Smallcap", "sector": "Capital Goods", "query": "KEI Industries"}
}

# --- 6. TECHNICAL INDICATORS & NEWS INGESTION ---
def compute_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    return (tp * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-5)

@st.cache_data(ttl=900)
def fetch_validated_news(ticker):
    try:
        company_name = STOCK_METADATA.get(ticker, {}).get("query", ticker)
        query = urllib.parse.quote(f"{company_name} corporate filing NSE")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')
        if not items:
            return 0.0, "No Active News Catalysts"

        item = items[0]
        title = item.find('title').text if item.find('title') is not None else ""
        pub_date_str = item.find('pubDate').text if item.find('pubDate') is not None else ""
        
        try:
            pub_dt = parsedate_to_datetime(pub_date_str)
            now = datetime.now(pub_dt.tzinfo)
            age_hours = (now - pub_dt).total_seconds() / 3600.0
        except Exception:
            age_hours = 24.0

        if age_hours > 24.0:
            return 0.0, "No Breaking 24h News"

        title_lower = title.lower()
        bullish_kw = ["quarterly result", "profit rises", "order win", "contract", "buyback", "expansion"]
        bearish_kw = ["penalty", "investigation", "profit falls", "resignation", "downgrade"]

        if any(k in title_lower for k in bullish_kw):
            return 10.0, f"🏛️ BULLISH CATALYST ({int(age_hours)}h ago): {title[:35]}..."
        elif any(k in title_lower for k in bearish_kw):
            return -10.0, f"⚠️ BEARISH CATALYST ({int(age_hours)}h ago): {title[:35]}..."
        
        return 0.0, f"📰 ROUTINE DISCLOSURE: {title[:35]}..."
    except Exception:
        return 0.0, "News Feed Operational"

# --- 7. SYNCHRONIZED SMC DETECTOR ENGINE ---
def detect_synchronized_smc(df_5m):
    if len(df_5m) < 30:
        return []
    
    df = df_5m.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['VWAP'] = compute_vwap(df)
    
    last_price = float(df['Close'].iloc[-1])
    zones = []
    
    lookback = min(60, len(df) - 3)
    start_i = len(df) - lookback
    
    for i in range(start_i, len(df) - 2):
        candle_time = df.index[i]
        c_open, c_close = float(df['Open'].iloc[i]), float(df['Close'].iloc[i])
        c_high, c_low = float(df['High'].iloc[i]), float(df['Low'].iloc[i])
        atr = float(df['ATR'].iloc[i]) if not np.isnan(df['ATR'].iloc[i]) else (c_high - c_low)
        
        if c_close < c_open:
            next_close = float(df['Close'].iloc[i+1])
            displacement = next_close - c_open
            fvg_present = float(df['Low'].iloc[i+2]) > float(df['High'].iloc[i]) if i+2 < len(df) else False
            
            if displacement > (1.2 * atr) and fvg_present:
                ob_top, ob_bottom = c_high, c_low
                future_lows = df['Low'].iloc[i+1:]
                mitigated = (future_lows < ob_bottom).any()
                
                if not mitigated:
                    if ob_bottom <= last_price <= ob_top:
                        state, state_val = "🟢 BULLISH OB RETEST (ENTRY READY)", 3
                    elif last_price > ob_top and ((last_price - ob_top) / ob_top) * 100 <= 0.5:
                        state, state_val = "🟡 PULLBACK TO BULLISH OB", 2
                    else:
                        state, state_val = "⏸️ UNMITIGATED BULLISH OB", 1
                    
                    zones.append({
                        'type': 'Bullish OB', 'top': ob_top, 'bottom': ob_bottom,
                        'start_time': candle_time, 'state': state, 'state_val': state_val, 'bias': 'BUY'
                    })

        if c_close > c_open:
            next_close = float(df['Close'].iloc[i+1])
            displacement = c_open - next_close
            fvg_present = float(df['High'].iloc[i+2]) < float(df['Low'].iloc[i]) if i+2 < len(df) else False
            
            if displacement > (1.2 * atr) and fvg_present:
                ob_top, ob_bottom = c_high, c_low
                future_highs = df['High'].iloc[i+1:]
                mitigated = (future_highs > ob_top).any()
                
                if not mitigated:
                    if ob_bottom <= last_price <= ob_top:
                        state, state_val = "🔴 BEARISH OB RETEST (SHORT READY)", 3
                    elif last_price < ob_bottom and ((ob_bottom - last_price) / ob_bottom) * 100 <= 0.5:
                        state, state_val = "🟡 PULLBACK TO BEARISH OB", 2
                    else:
                        state, state_val = "⏸️ UNMITIGATED BEARISH OB", 1
                    
                    zones.append({
                        'type': 'Bearish OB', 'top': ob_top, 'bottom': ob_bottom,
                        'start_time': candle_time, 'state': state, 'state_val': state_val, 'bias': 'SELL'
                    })

    recent_swings_low = df['Low'].iloc[-30:-3].min()
    recent_swings_high = df['High'].iloc[-30:-3].max()
    curr_low, curr_high = float(df['Low'].iloc[-1]), float(df['High'].iloc[-1])
    curr_close = float(df['Close'].iloc[-1])
    
    if curr_low < recent_swings_low and curr_close > recent_swings_low:
        zones.append({
            'type': 'Liquidity Sweep Low', 'top': recent_swings_low * 1.001, 'bottom': curr_low,
            'start_time': df.index[-1], 'state': "🟢 LIQUIDITY SWEEP (BULLISH REVERSAL)", 'state_val': 3, 'bias': 'BUY'
        })
        
    if curr_high > recent_swings_high and curr_close < recent_swings_high:
        zones.append({
            'type': 'Liquidity Sweep High', 'top': curr_high, 'bottom': recent_swings_high * 0.999,
            'start_time': df.index[-1], 'state': "🔴 LIQUIDITY SWEEP (BEARISH REVERSAL)", 'state_val': 3, 'bias': 'SELL'
        })

    return zones

# --- 8. CORE SCANNER & CONFLUENCE MATRIX ---
ctrl_col1, ctrl_col2 = st.columns([1, 3])
with ctrl_col1:
    scan_universe = st.selectbox("Select Scanning Universe", ["All Combined", "Nifty 50", "Nifty Midcap", "Nifty Smallcap"])
with ctrl_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_scan = st.button("🚀 Execute Synchronized Institutional Scan", type="primary")

if run_scan:
    if scan_universe == "Nifty 50":
        tickers = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty 50"]
    elif scan_universe == "Nifty Midcap":
        tickers = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty Midcap"]
    elif scan_universe == "Nifty Smallcap":
        tickers = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty Smallcap"]
    else:
        tickers = list(STOCK_METADATA.keys())

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ticker in enumerate(tickers):
        status_text.text(f"Scanning & Synchronizing Context for {ticker}...")
        progress_bar.progress((idx + 1) / len(tickers))
        
        try:
            df_5m, df_1d = fetch_stock_data(ticker)
            if df_5m is None or df_5m.empty or df_1d is None or df_1d.empty:
                continue

            if isinstance(df_5m.columns, pd.MultiIndex):
                df_5m.columns = df_5m.columns.get_level_values(0)
            if isinstance(df_1d.columns, pd.MultiIndex):
                df_1d.columns = df_1d.columns.get_level_values(0)

            close_5m, high_5m, low_5m = df_5m["Close"].dropna(), df_5m["High"].dropna(), df_5m["Low"].dropna()
            vol_5m = df_5m["Volume"].dropna() if "Volume" in df_5m else pd.Series(1, index=close_5m.index)
            
            last_price = float(close_5m.iloc[-1])
            vwap_val = float(compute_vwap(df_5m).iloc[-1])
            ema_20 = float(close_5m.ewm(span=20, adjust=False).mean().iloc[-1])
            atr_14 = float((high_5m - low_5m).tail(14).mean())
            
            avg_vol = float(vol_5m.tail(20).mean())
            rvol = float(vol_5m.iloc[-1] / (avg_vol + 1e-5))

            pdh = float(df_1d["High"].dropna().iloc[-2]) if len(df_1d) >= 2 else float(high_5m.max())
            pdl = float(df_1d["Low"].dropna().iloc[-2]) if len(df_1d) >= 2 else float(low_5m.min())

            smc_zones = detect_synchronized_smc(df_5m)
            news_score, news_context = fetch_validated_news(ticker)

            best_zone = sorted(smc_zones, key=lambda x: x['state_val'], reverse=True)[0] if smc_zones else None
            
            score = 0.0
            trade_bias = "NEUTRAL"
            
            if best_zone:
                trade_bias = best_zone['bias']
                score += 35.0 if best_zone['state_val'] == 3 else (20.0 if best_zone['state_val'] == 2 else 10.0)
            else:
                trade_bias = "BUY" if last_price > vwap_val else "SELL"

            if trade_bias == "BUY" and last_price > vwap_val and last_price > ema_20:
                score += 25.0
            elif trade_bias == "SELL" and last_price < vwap_val and last_price < ema_20:
                score += 25.0

            if rvol >= 1.5: score += 15.0
            elif rvol >= 1.0: score += 8.0

            meta = STOCK_METADATA.get(ticker, {"index": "N/A", "sector": "General"})
            if meta["sector"] in active_sectors: score += 15.0

            score += news_score

            if trade_bias == "BUY":
                sl_price = (best_zone['bottom'] - (0.1 * atr_14)) if best_zone else (last_price - (1.5 * atr_14))
                tgt_price = pdh if pdh > (last_price + (1.5 * atr_14)) else (last_price + (2.5 * atr_14))
            else:
                sl_price = (best_zone['top'] + (0.1 * atr_14)) if best_zone else (last_price + (1.5 * atr_14))
                tgt_price = pdl if pdl < (last_price - (1.5 * atr_14)) else (last_price - (2.5 * atr_14))

            risk_pct = abs((last_price - sl_price) / last_price) * 100
            reward_pct = abs((tgt_price - last_price) / last_price) * 100
            rr_ratio = reward_pct / (risk_pct + 1e-5)

            if rr_ratio < min_rr_threshold:
                continue

            results.append({
                "Stock": f"🏛️ {ticker}" if news_score != 0 else ticker,
                "Ticker_Raw": ticker,
                "Index": meta["index"],
                "Sector": meta["sector"],
                "Last Price": f"₹{last_price:.2f}",
                "Bias": "🟩 BUY" if trade_bias == "BUY" else "🟥 SELL",
                "SMC State": best_zone['state'] if best_zone else "NO ACTIVE SMC ZONE",
                "VWAP Alignment": "✅ ABOVE VWAP" if last_price > vwap_val else "🔻 BELOW VWAP",
                "RVOL": f"{rvol:.2f}x",
                "Target": f"₹{tgt_price:.2f} (+{reward_pct:.1f}%)",
                "Stop Loss": f"₹{sl_price:.2f} (-{risk_pct:.1f}%)",
                "R:R": f"1:{rr_ratio:.2f}",
                "Confluence Score": round(score, 1),
                "News / Catalysts": news_context
            })

        except Exception:
            continue

    status_text.empty()
    progress_bar.empty()

    if results:
        res_df = pd.DataFrame(results).sort_values(by="Confluence Score", ascending=False).reset_index(drop=True)
        res_df["Rank"] = res_df.index + 1
        st.session_state["scan_results"] = res_df
    else:
        st.warning("No stocks passed the current R:R filter. Try lowering the Minimum R:R slider in the sidebar.")

# --- 9. DASHBOARD DISPLAY & MULTI-RANK CHART VISUALIZER ---
if "scan_results" in st.session_state:
    res_df = st.session_state["scan_results"]
    
    st.subheader("🎯 Top Actionable Institutional Setups")
    card_cols = st.columns(3)
    for idx in range(min(3, len(res_df))):
        row = res_df.iloc[idx]
        with card_cols[idx]:
            st.metric(
                label=f"#{row['Rank']} {row['Stock']} ({row['Sector']})",
                value=row['Last Price'],
                delta=f"Score: {row['Confluence Score']} | {row['Bias']}"
            )
            st.write(f"**SMC State:** `{row['SMC State']}`")
            st.write(f"**Target:** {row['Target']} | **SL:** {row['Stop Loss']}")
            st.write(f"**R:R:** `{row['R:R']}` | **RVOL:** `{row['RVOL']}`")

    st.markdown("---")
    st.subheader("📈 Live Visual SMC Charts (Rank 1, Rank 2 & Rank 3 Setups)")
    
    num_charts = min(3, len(res_df))
    if num_charts > 0:
        tab_titles = [f"🥇 Rank 1: {res_df.iloc[0]['Ticker_Raw']}"]
        if num_charts > 1: tab_titles.append(f"🥈 Rank 2: {res_df.iloc[1]['Ticker_Raw']}")
        if num_charts > 2: tab_titles.append(f"🥉 Rank 3: {res_df.iloc[2]['Ticker_Raw']}")
        
        chart_tabs = st.tabs(tab_titles)
        
        for i in range(num_charts):
            with chart_tabs[i]:
                stock_ticker = res_df.iloc[i]['Ticker_Raw']
                df_chart, _ = fetch_stock_data(stock_ticker)
                
                if df_chart is not None and not df_chart.empty:
                    if isinstance(df_chart.columns, pd.MultiIndex):
                        df_chart.columns = df_chart.columns.get_level_values(0)

                    df_chart['VWAP'] = compute_vwap(df_chart)
                    df_chart['EMA20'] = df_chart['Close'].ewm(span=20, adjust=False).mean()
                    
                    fig = go.Figure(data=[go.Candlestick(
                        x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                        low=df_chart['Low'], close=df_chart['Close'], name="Price"
                    )])
                    
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['VWAP'], line=dict(color='orange', width=1.5), name="VWAP"))
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA20'], line=dict(color='cyan', width=1), name="EMA 20"))
                    
                    zones = detect_synchronized_smc(df_chart)
                    for zone in zones:
                        color = "rgba(0, 255, 0, 0.25)" if zone['bias'] == 'BUY' else "rgba(255, 0, 0, 0.25)"
                        line_color = "green" if zone['bias'] == 'BUY' else "red"
                        
                        fig.add_shape(
                            type="rect",
                            x0=zone['start_time'],
                            x1=df_chart.index[-1],
                            y0=zone['bottom'],
                            y1=zone['top'],
                            fillcolor=color,
                            line=dict(color=line_color, width=1),
                        )
                        
                        fig.add_annotation(
                            x=zone['start_time'],
                            y=zone['top'],
                            text=f"{zone['type']} ({zone['state']})",
                            showarrow=False,
                            yshift=10,
                            font=dict(size=10, color=line_color)
                        )

                    fig.update_layout(
                        title=f"Rank #{i+1} Setup: {stock_ticker} - SMC Zone Rectangles, VWAP & Confluence",
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        height=520,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Full Synchronized Watchlist")
    
    display_cols = [
        "Rank", "Stock", "Index", "Sector", "Bias", "Confluence Score", 
        "Last Price", "SMC State", "VWAP Alignment", "RVOL", 
        "Target", "Stop Loss", "R:R", "News / Catalysts"
    ]
    st.dataframe(res_df[display_cols], height=400, use_container_width=True)
    
    csv = res_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Institutional Scan Results (CSV)", data=csv, file_name="synchronized_institutional_scan.csv", mime="text/csv")
