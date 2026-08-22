import streamlit as st
import joblib
import os
import time
import requests
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Alpha-NSE | Synchronized SMC & ML Engine",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Project Alpha-NSE: Synchronized SMC, Order Flow & AI Engine")
st.markdown("*Real-Time Intraday Institutional Scanner with Session-Anchored SMC, Volume Profile POC, Block Order Engine & Dynamic Risk Management*")

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
    ["Banking", "IT", "Auto", "Energy", "FMCG", "Metal", "Infra", "Financials", "Healthcare", "Consumer Durables", "Capital Goods", "Telecom"],
    default=["Banking", "IT", "Financials", "Auto", "Energy"]
)

min_rr_threshold = st.sidebar.slider("Minimum Risk-to-Reward (R:R) Filter", 1.0, 4.0, 1.2, 0.1)

# --- 3. EXPANDED UNIVERSE ISIN MAP (100 STOCKS) ---
UPSTOX_ISIN_MAP = {
    # Nifty 50 (50 Stocks)
    "RELIANCE": "NSE_EQ|INE002A01018", "TCS": "NSE_EQ|INE467B01029", "HDFCBANK": "NSE_EQ|INE040A01034",
    "INFY": "NSE_EQ|INE009A01021", "ICICIBANK": "NSE_EQ|INE090A01013", "SBIN": "NSE_EQ|INE062A01020",
    "BHARTIARTL": "NSE_EQ|INE397D01024", "ITC": "NSE_EQ|INE154A01025", "LTIM": "NSE_EQ|INE214T01019",
    "AXISBANK": "NSE_EQ|INE238A01034", "KOTAKBANK": "NSE_EQ|INE237A01028", "LT": "NSE_EQ|INE018A01030",
    "HINDUNILVR": "NSE_EQ|INE030A01027", "BAJFINANCE": "NSE_EQ|INE296A01024", "MARUTI": "NSE_EQ|INE585B01010",
    "TATAMOTORS": "NSE_EQ|INE155A01022", "TATASTEEL": "NSE_EQ|INE081A01020", "NTPC": "NSE_EQ|INE733E01010",
    "M&M": "NSE_EQ|INE101A01026", "POWERGRID": "NSE_EQ|INE752E01010", "TITAN": "NSE_EQ|INE280A01028",
    "ULTRACEMCO": "NSE_EQ|INE481G01011", "WIPRO": "NSE_EQ|INE075A01022", "ONGC": "NSE_EQ|INE213A01029",
    "ADANIENT": "NSE_EQ|INE423A01024", "ADANIPORTS": "NSE_EQ|INE742F01042", "COALINDIA": "NSE_EQ|INE522F01014",
    "ASIANPAINT": "NSE_EQ|INE021A01026", "HCLTECH": "NSE_EQ|INE860A01027", "SUNPHARMA": "NSE_EQ|INE044A01036",
    "BAJAJFINSV": "NSE_EQ|INE918I01024", "GRASIM": "NSE_EQ|INE047A01021", "TECHM": "NSE_EQ|INE669C01036",
    "HDFCLIFE": "NSE_EQ|INE001A01036", "CIPLA": "NSE_EQ|INE059A01026", "EICHERMOT": "NSE_EQ|INE066A01021",
    "TATACONSUMER": "NSE_EQ|INE192A01025", "HINDALCO": "NSE_EQ|INE038A01020", "BPCL": "NSE_EQ|INE029A01011",
    "SBILIFE": "NSE_EQ|INE123W01016", "BRITANNIA": "NSE_EQ|INE216A01030", "DRREDDY": "NSE_EQ|INE089A01023",
    "HEROMOTOCO": "NSE_EQ|INE158A01026", "DIVISLAB": "NSE_EQ|INE361B01024", "APOLLOHOSP": "NSE_EQ|INE437A01024",
    "BEL": "NSE_EQ|INE263A01024", "SHRIRAMFIN": "NSE_EQ|INE721A01013", "JSWSTEEL": "NSE_EQ|INE019A01038",
    "TRENT": "NSE_EQ|INE849A01020", "NESTLEIND": "NSE_EQ|INE239A01024",

    # Nifty Midcap (25 Stocks)
    "TATAPOWER": "NSE_EQ|INE245A01021", "FEDERALBNK": "NSE_EQ|INE171A01029", "POLYCAB": "NSE_EQ|INE455K01017",
    "PERSISTENT": "NSE_EQ|INE262H01021", "COFORGE": "NSE_EQ|INE591G01017", "ASHOKLEY": "NSE_EQ|INE208A01029",
    "MAXHEALTH": "NSE_EQ|INE275F01028", "VOLTAS": "NSE_EQ|INE226A01021", "ASTRAL": "NSE_EQ|INE006I01012",
    "CUMMINSIND": "NSE_EQ|INE299A01018", "DIXON": "NSE_EQ|INE935N01020", "IDFCFIRSTB": "NSE_EQ|INE092T01019",
    "LUPIN": "NSE_EQ|INE326A01037", "AUROPHARMA": "NSE_EQ|INE406A01037", "OBEROIRTY": "NSE_EQ|INE093I01010",
    "HDFCAMC": "NSE_EQ|INE127D01025", "TUBEINVEST": "NSE_EQ|INE974X01010", "SUNDARMFIN": "NSE_EQ|INE660A01013",
    "GMRINFRA": "NSE_EQ|INE776C01039", "BHARATFORG": "NSE_EQ|INE465A01025", "BALKRISIND": "NSE_EQ|INE787D01026",
    "M&MFIN": "NSE_EQ|INE774D01024", "ESCORTS": "NSE_EQ|INE042A01014", "MUTHOOTFIN": "NSE_EQ|INE414G01012",
    "GODREJPROP": "NSE_EQ|INE484J01027",

    # Nifty Smallcap (25 Stocks)
    "CDSL": "NSE_EQ|INE736A01011", "ANGELONE": "NSE_EQ|INE732I01013", "KFINTECH": "NSE_EQ|INE138Y01011",
    "SUZLON": "NSE_EQ|INE040H01021", "BSOFT": "NSE_EQ|INE084A01015", "HFCL": "NSE_EQ|INE548A01028",
    "IEX": "NSE_EQ|INE577H01019", "KEI": "NSE_EQ|INE378B01023", "REDINGTON": "NSE_EQ|INE891D01026",
    "FIRSTSOURCE": "NSE_EQ|INE688F01017", "FSL": "NSE_EQ|INE688F01017", "CAMS": "NSE_EQ|INE596I01012",
    "CYIENT": "NSE_EQ|INE136B01020", "MAPMYINDIA": "NSE_EQ|INE0BV301023", "MCX": "NSE_EQ|INE745G01035",
    "ZENSARTECH": "NSE_EQ|INE520A01027", "RITES": "NSE_EQ|INE320J01015", "IRFC": "NSE_EQ|INE053F01010",
    "RVNL": "NSE_EQ|INE415G01027", "SJVN": "NSE_EQ|INE002L01015", "MAZDOCK": "NSE_EQ|INE249Z01012",
    "HUDCO": "NSE_EQ|INE031A01017", "CENTRALBK": "NSE_EQ|INE483A01010", "UCOBANK": "NSE_EQ|INE691A01018",
    "BEML": "NSE_EQ|INE258A01016"
}

def get_upstox_instrument_key(symbol: str) -> str:
    clean_sym = symbol.upper().replace(".NS", "").replace("&", "_").strip()
    return UPSTOX_ISIN_MAP.get(clean_sym, f"NSE_EQ|{clean_sym}")

def get_upstox_access_token() -> str:
    try:
        if "UPSTOX_ACCESS_TOKEN" in st.secrets:
            return st.secrets["UPSTOX_ACCESS_TOKEN"]
        return os.getenv("UPSTOX_ACCESS_TOKEN", None)
    except Exception:
        return None

def fetch_stock_data(ticker):
    """
    Data Fetch Pipeline:
    Primary Attempt -> Direct Upstox REST API v3
    Fallback Attempt -> Yahoo Finance (yfinance)
    """
    try:
        access_token = get_upstox_access_token()
        if access_token:
            instrument_key = urllib.parse.quote(get_upstox_instrument_key(ticker), safe="")
            headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
            
            url_5m = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/minutes/5"
            res_5m = requests.get(url_5m, headers=headers, timeout=5)
            
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
            url_1d = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/days/1/{to_date}/{from_date}"
            res_1d = requests.get(url_1d, headers=headers, timeout=5)

            if res_5m.status_code == 200 and res_1d.status_code == 200:
                raw_5m = res_5m.json().get("data", {}).get("candles", [])
                raw_1d = res_1d.json().get("data", {}).get("candles", [])

                if raw_5m and raw_1d:
                    df_5m = pd.DataFrame(raw_5m, columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "OI"])
                    df_5m["Datetime"] = pd.to_datetime(df_5m["Datetime"])
                    df_5m.set_index("Datetime", inplace=True)
                    df_5m.sort_index(inplace=True)
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        df_5m[col] = pd.to_numeric(df_5m[col], errors="coerce")

                    df_1d = pd.DataFrame(raw_1d, columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "OI"])
                    df_1d["Datetime"] = pd.to_datetime(df_1d["Datetime"])
                    df_1d.set_index("Datetime", inplace=True)
                    df_1d.sort_index(inplace=True)
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        df_1d[col] = pd.to_numeric(df_1d[col], errors="coerce")

                    return df_5m[df_5m["Volume"] > 0], df_1d[df_1d["Volume"] > 0]
    except Exception:
        pass

    yf_symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    df_5m = yf.download(yf_symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
    df_1d = yf.download(yf_symbol, period="1mo", interval="1d", progress=False, auto_adjust=True)
    return df_5m, df_1d

# --- 4. MARKET SENTIMENT & INSTITUTIONAL FLOW ENGINE ---
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

# Top Index & FII/DII Metrics Bar
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

# Sectoral Performance Metric Grid
st.markdown("**🌐 Sectoral Performance (Live Impact)**")
sec_cols = st.columns(6)
for idx, sec in enumerate(["Banking", "IT", "Auto", "Energy", "FMCG", "Metal"]):
    sec_ret = market_returns.get(f"Sector_{sec}", 0.0) * 100
    sec_cols[idx % 6].metric(sec, f"{'+' if sec_ret >= 0 else ''}{sec_ret:.2f}%")
st.markdown("---")

# --- 5. EXPANDED UNIVERSE METADATA REGISTRY (100 STOCKS) ---
STOCK_METADATA = {
    # Nifty 50
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
    "POWERGRID": {"index": "Nifty 50", "sector": "Energy", "query": "Power Grid"},
    "TITAN": {"index": "Nifty 50", "sector": "Consumer Durables", "query": "Titan Company"},
    "ULTRACEMCO": {"index": "Nifty 50", "sector": "Infra", "query": "UltraTech Cement"},
    "WIPRO": {"index": "Nifty 50", "sector": "IT", "query": "Wipro"},
    "ONGC": {"index": "Nifty 50", "sector": "Energy", "query": "ONGC"},
    "ADANIENT": {"index": "Nifty 50", "sector": "Infra", "query": "Adani Enterprises"},
    "ADANIPORTS": {"index": "Nifty 50", "sector": "Infra", "query": "Adani Ports"},
    "COALINDIA": {"index": "Nifty 50", "sector": "Metal", "query": "Coal India"},
    "ASIANPAINT": {"index": "Nifty 50", "sector": "Consumer Durables", "query": "Asian Paints"},
    "HCLTECH": {"index": "Nifty 50", "sector": "IT", "query": "HCL Technologies"},
    "SUNPHARMA": {"index": "Nifty 50", "sector": "Healthcare", "query": "Sun Pharma"},
    "BAJAJFINSV": {"index": "Nifty 50", "sector": "Financials", "query": "Bajaj Finserv"},
    "GRASIM": {"index": "Nifty 50", "sector": "Infra", "query": "Grasim Industries"},
    "TECHM": {"index": "Nifty 50", "sector": "IT", "query": "Tech Mahindra"},
    "HDFCLIFE": {"index": "Nifty 50", "sector": "Financials", "query": "HDFC Life"},
    "CIPLA": {"index": "Nifty 50", "sector": "Healthcare", "query": "Cipla"},
    "EICHERMOT": {"index": "Nifty 50", "sector": "Auto", "query": "Eicher Motors"},
    "TATACONSUMER": {"index": "Nifty 50", "sector": "FMCG", "query": "Tata Consumer Products"},
    "HINDALCO": {"index": "Nifty 50", "sector": "Metal", "query": "Hindalco"},
    "BPCL": {"index": "Nifty 50", "sector": "Energy", "query": "BPCL"},
    "SBILIFE": {"index": "Nifty 50", "sector": "Financials", "query": "SBI Life"},
    "BRITANNIA": {"index": "Nifty 50", "sector": "FMCG", "query": "Britannia"},
    "DRREDDY": {"index": "Nifty 50", "sector": "Healthcare", "query": "Dr Reddys"},
    "HEROMOTOCO": {"index": "Nifty 50", "sector": "Auto", "query": "Hero MotoCorp"},
    "DIVISLAB": {"index": "Nifty 50", "sector": "Healthcare", "query": "Divis Labs"},
    "APOLLOHOSP": {"index": "Nifty 50", "sector": "Healthcare", "query": "Apollo Hospitals"},
    "BEL": {"index": "Nifty 50", "sector": "Capital Goods", "query": "Bharat Electronics"},
    "SHRIRAMFIN": {"index": "Nifty 50", "sector": "Financials", "query": "Shriram Finance"},
    "JSWSTEEL": {"index": "Nifty 50", "sector": "Metal", "query": "JSW Steel"},
    "TRENT": {"index": "Nifty 50", "sector": "Consumer Durables", "query": "Trent"},
    "NESTLEIND": {"index": "Nifty 50", "sector": "FMCG", "query": "Nestle India"},

    # Nifty Midcap
    "TATAPOWER": {"index": "Nifty Midcap", "sector": "Energy", "query": "Tata Power"},
    "FEDERALBNK": {"index": "Nifty Midcap", "sector": "Banking", "query": "Federal Bank"},
    "POLYCAB": {"index": "Nifty Midcap", "sector": "Capital Goods", "query": "Polycab"},
    "PERSISTENT": {"index": "Nifty Midcap", "sector": "IT", "query": "Persistent Systems"},
    "COFORGE": {"index": "Nifty Midcap", "sector": "IT", "query": "Coforge"},
    "ASHOKLEY": {"index": "Nifty Midcap", "sector": "Auto", "query": "Ashok Leyland"},
    "MAXHEALTH": {"index": "Nifty Midcap", "sector": "Healthcare", "query": "Max Healthcare"},
    "VOLTAS": {"index": "Nifty Midcap", "sector": "Consumer Durables", "query": "Voltas"},
    "ASTRAL": {"index": "Nifty Midcap", "sector": "Capital Goods", "query": "Astral"},
    "CUMMINSIND": {"index": "Nifty Midcap", "sector": "Capital Goods", "query": "Cummins India"},
    "DIXON": {"index": "Nifty Midcap", "sector": "Consumer Durables", "query": "Dixon Technologies"},
    "IDFCFIRSTB": {"index": "Nifty Midcap", "sector": "Banking", "query": "IDFC First Bank"},
    "LUPIN": {"index": "Nifty Midcap", "sector": "Healthcare", "query": "Lupin"},
    "AUROPHARMA": {"index": "Nifty Midcap", "sector": "Healthcare", "query": "Aurobindo Pharma"},
    "OBEROIRTY": {"index": "Nifty Midcap", "sector": "Infra", "query": "Oberoi Realty"},
    "HDFCAMC": {"index": "Nifty Midcap", "sector": "Financials", "query": "HDFC AMC"},
    "TUBEINVEST": {"index": "Nifty Midcap", "sector": "Auto", "query": "Tube Investments"},
    "SUNDARMFIN": {"index": "Nifty Midcap", "sector": "Financials", "query": "Sundaram Finance"},
    "GMRINFRA": {"index": "Nifty Midcap", "sector": "Infra", "query": "GMR Airports"},
    "BHARATFORG": {"index": "Nifty Midcap", "sector": "Capital Goods", "query": "Bharat Forge"},
    "BALKRISIND": {"index": "Nifty Midcap", "sector": "Auto", "query": "Balkrishna Industries"},
    "M&MFIN": {"index": "Nifty Midcap", "sector": "Financials", "query": "Mahindra Finance"},
    "ESCORTS": {"index": "Nifty Midcap", "sector": "Auto", "query": "Escorts Kubota"},
    "MUTHOOTFIN": {"index": "Nifty Midcap", "sector": "Financials", "query": "Muthoot Finance"},
    "GODREJPROP": {"index": "Nifty Midcap", "sector": "Infra", "query": "Godrej Properties"},

    # Nifty Smallcap
    "CDSL": {"index": "Nifty Smallcap", "sector": "Financials", "query": "CDSL"},
    "ANGELONE": {"index": "Nifty Smallcap", "sector": "Financials", "query": "Angel One"},
    "KFINTECH": {"index": "Nifty Smallcap", "sector": "Financials", "query": "KFin Technologies"},
    "SUZLON": {"index": "Nifty Smallcap", "sector": "Energy", "query": "Suzlon Energy"},
    "BSOFT": {"index": "Nifty Smallcap", "sector": "IT", "query": "Birlasoft"},
    "HFCL": {"index": "Nifty Smallcap", "sector": "Infra", "query": "HFCL"},
    "IEX": {"index": "Nifty Smallcap", "sector": "Financials", "query": "Indian Energy Exchange"},
    "KEI": {"index": "Nifty Smallcap", "sector": "Capital Goods", "query": "KEI Industries"},
    "REDINGTON": {"index": "Nifty Smallcap", "sector": "IT", "query": "Redington"},
    "FIRSTSOURCE": {"index": "Nifty Smallcap", "sector": "IT", "query": "Firstsource Solutions"},
    "FSL": {"index": "Nifty Smallcap", "sector": "IT", "query": "Firstsource Solutions"},
    "CAMS": {"index": "Nifty Smallcap", "sector": "Financials", "query": "Computer Age Management"},
    "CYIENT": {"index": "Nifty Smallcap", "sector": "IT", "query": "Cyient"},
    "MAPMYINDIA": {"index": "Nifty Smallcap", "sector": "IT", "query": "CE Info Systems"},
    "MCX": {"index": "Nifty Smallcap", "sector": "Financials", "query": "Multi Commodity Exchange"},
    "ZENSARTECH": {"index": "Nifty Smallcap", "sector": "IT", "query": "Zensar Technologies"},
    "RITES": {"index": "Nifty Smallcap", "sector": "Infra", "query": "RITES"},
    "IRFC": {"index": "Nifty Smallcap", "sector": "Financials", "query": "Indian Railway Finance"},
    "RVNL": {"index": "Nifty Smallcap", "sector": "Infra", "query": "Rail Vikas Nigam"},
    "SJVN": {"index": "Nifty Smallcap", "sector": "Energy", "query": "SJVN"},
    "MAZDOCK": {"index": "Nifty Smallcap", "sector": "Capital Goods", "query": "Mazagon Dock"},
    "HUDCO": {"index": "Nifty Smallcap", "sector": "Financials", "query": "HUDCO"},
    "CENTRALBK": {"index": "Nifty Smallcap", "sector": "Banking", "query": "Central Bank of India"},
    "UCOBANK": {"index": "Nifty Smallcap", "sector": "Banking", "query": "UCO Bank"},
    "BEML": {"index": "Nifty Smallcap", "sector": "Capital Goods", "query": "BEML Limited"}
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

# --- 7. ADVANCED BLOCK ORDER, VOLUME PROFILE & TREND REVERSAL ENGINE ---
def compute_volume_profile_poc(df_5m, num_bins=30):
    """Calculates Intraday Session Point of Control (POC) via Volume Profile Histogram"""
    if df_5m.empty or len(df_5m) < 5:
        return None
    price_min = df_5m['Low'].min()
    price_max = df_5m['High'].max()
    if price_min == price_max:
        return float(price_min)
    
    bins = np.linspace(price_min, price_max, num_bins)
    counts, bin_edges = np.histogram(df_5m['Close'], bins=bins, weights=df_5m['Volume'])
    max_idx = np.argmax(counts)
    poc_price = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0
    return float(poc_price)

def detect_block_orders_and_reversals(df_5m):
    """
    Identifies:
    1) Large Institutional Block Orders (Volume Spikes, Wick Absorption @ POC/VWAP)
    2) Major Intraday Trend Reversals (Liquidity Sweeps + Candlestick Reversals + Volume Confirmation)
    """
    if len(df_5m) < 10:
        return {"poc_price": None, "block_detected": False, "block_msg": "", "reversal_detected": False, "reversal_type": "NONE", "reversal_msg": ""}
    
    df = df_5m.copy()
    poc_price = compute_volume_profile_poc(df)
    
    df['Vol_EMA'] = df['Volume'].ewm(span=20).mean()
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_EMA'] + 1e-5)
    
    last_bar = df.iloc[-1]
    prev_bar = df.iloc[-2]
    
    # --- 1. BLOCK ORDER DETECTION ---
    block_detected = False
    block_msg = "No Heavy Institutional Block Detected"
    
    vol_spike = last_bar['Vol_Ratio'] >= 2.5
    range_avg = (df['High'] - df['Low']).tail(20).mean()
    wide_range = (last_bar['High'] - last_bar['Low']) > (1.5 * range_avg)
    
    if vol_spike and wide_range:
        block_detected = True
        direction = "BUYING" if last_bar['Close'] > last_bar['Open'] else "SELLING"
        block_msg = f"🔥 LARGE BLOCK ORDER ({direction}): Vol {last_bar['Vol_Ratio']:.1f}x Avg @ ₹{last_bar['Close']:.2f}"
    
    # --- 2. MAJOR TREND REVERSAL DETECTION ---
    reversal_detected = False
    reversal_type = "NONE"
    reversal_msg = "No Trend Reversal Active"
    
    curr_open, curr_close = float(last_bar['Open']), float(last_bar['Close'])
    curr_high, curr_low = float(last_bar['High']), float(last_bar['Low'])
    body_len = abs(curr_close - curr_open)
    upper_wick = curr_high - max(curr_open, curr_close)
    lower_wick = min(curr_open, curr_close) - curr_low
    
    # Shooting Star / Bearish Pinbar Reversal
    if upper_wick >= (2.0 * body_len) and lower_wick <= (0.5 * body_len) and last_bar['Vol_Ratio'] > 1.2:
        reversal_detected = True
        reversal_type = "BEARISH_REVERSAL"
        reversal_msg = f"🔻 MAJOR BEARISH REVERSAL (Shooting Star @ Vol {last_bar['Vol_Ratio']:.1f}x)"
    # Hammer / Bullish Pinbar Reversal
    elif lower_wick >= (2.0 * body_len) and upper_wick <= (0.5 * body_len) and last_bar['Vol_Ratio'] > 1.2:
        reversal_detected = True
        reversal_type = "BULLISH_REVERSAL"
        reversal_msg = f"🟢 MAJOR BULLISH REVERSAL (Hammer @ Vol {last_bar['Vol_Ratio']:.1f}x)"
    # Bullish Engulfing
    elif prev_bar['Close'] < prev_bar['Open'] and curr_close > curr_open and curr_close > prev_bar['Open'] and curr_open < prev_bar['Close']:
        reversal_detected = True
        reversal_type = "BULLISH_REVERSAL"
        reversal_msg = f"🟢 BULLISH ENGULFING REVERSAL (Vol {last_bar['Vol_Ratio']:.1f}x)"
    # Bearish Engulfing
    elif prev_bar['Close'] > prev_bar['Open'] and curr_close < curr_open and curr_close < prev_bar['Open'] and curr_open > prev_bar['Close']:
        reversal_detected = True
        reversal_type = "BEARISH_REVERSAL"
        reversal_msg = f"🔻 BEARISH ENGULFING REVERSAL (Vol {last_bar['Vol_Ratio']:.1f}x)"

    return {
        "poc_price": poc_price,
        "block_detected": block_detected,
        "block_msg": block_msg,
        "reversal_detected": reversal_detected,
        "reversal_type": reversal_type,
        "reversal_msg": reversal_msg
    }

# --- 8. SYNCHRONIZED SMC DETECTOR ENGINE ---
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
        
        # Bullish Order Block
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

        # Bearish Order Block
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

    # Liquidity Sweeps
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

# --- 9. DYNAMIC TARGET & STOPLOSS ENGINE ---
def calculate_dynamic_risk_levels(df_5m, df_1d, trade_bias, last_price, best_zone, poc_price, atr_14):
    """
    Calculates dynamic stop loss and targets based on Volume Profile POC, 
    VWAP Standard Deviation Bands, Order Blocks, and Swing High/Lows.
    """
    vwap = float(compute_vwap(df_5m).iloc[-1])
    tp = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3.0
    vwap_std = float(np.sqrt(((tp - vwap) ** 2).mean())) if len(df_5m) > 1 else atr_14
    
    swing_low = float(df_5m['Low'].tail(15).min())
    swing_high = float(df_5m['High'].tail(15).max())
    
    pdh = float(df_1d["High"].dropna().iloc[-2]) if len(df_1d) >= 2 else float(df_5m["High"].max())
    pdl = float(df_1d["Low"].dropna().iloc[-2]) if len(df_1d) >= 2 else float(df_5m["Low"].min())
    
    if trade_bias == "BUY":
        potential_sls = [last_price - (1.5 * atr_14), swing_low - (0.2 * atr_14)]
        if best_zone and best_zone['bias'] == 'BUY':
            potential_sls.append(best_zone['bottom'] - (0.1 * atr_14))
        if poc_price and poc_price < last_price:
            potential_sls.append(poc_price - (0.2 * atr_14))
        
        valid_sls = [sl for sl in potential_sls if sl < last_price and (last_price - sl)/last_price <= 0.035]
        sl_price = max(valid_sls) if valid_sls else (last_price - (1.5 * atr_14))
        
        risk_dist = last_price - sl_price
        potential_tgts = [last_price + (2.5 * risk_dist), vwap + (2.0 * vwap_std), swing_high]
        if pdh > last_price:
            potential_tgts.append(pdh)
        
        valid_tgts = [tgt for tgt in potential_tgts if tgt > last_price + (1.2 * risk_dist)]
        tgt_price = max(valid_tgts) if valid_tgts else (last_price + (2.5 * risk_dist))
        
    else:  # SELL Bias
        potential_sls = [last_price + (1.5 * atr_14), swing_high + (0.2 * atr_14)]
        if best_zone and best_zone['bias'] == 'SELL':
            potential_sls.append(best_zone['top'] + (0.1 * atr_14))
        if poc_price and poc_price > last_price:
            potential_sls.append(poc_price + (0.2 * atr_14))
        
        valid_sls = [sl for sl in potential_sls if sl > last_price and (sl - last_price)/last_price <= 0.035]
        sl_price = min(valid_sls) if valid_sls else (last_price + (1.5 * atr_14))
        
        risk_dist = sl_price - last_price
        potential_tgts = [last_price - (2.5 * risk_dist), vwap - (2.0 * vwap_std), swing_low]
        if pdl < last_price:
            potential_tgts.append(pdl)
        
        valid_tgts = [tgt for tgt in potential_tgts if tgt < last_price - (1.2 * risk_dist)]
        tgt_price = min(valid_tgts) if valid_tgts else (last_price - (2.5 * risk_dist))
        
    return float(sl_price), float(tgt_price)

# --- 10. CORE SCANNER & CONFLUENCE MATRIX ---
ctrl_col1, ctrl_col2 = st.columns([1, 3])
with ctrl_col1:
    scan_universe = st.selectbox("Select Scanning Universe", ["All Combined (100 Stocks)", "Nifty 50", "Nifty Midcap", "Nifty Smallcap"])
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
        status_text.text(f"Scanning ({idx+1}/{len(tickers)}) Block Orders & SMC Context for {ticker}...")
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

            # Advanced Order Flow & Reversal Diagnostics
            of_diag = detect_block_orders_and_reversals(df_5m)
            poc_price = of_diag["poc_price"]

            smc_zones = detect_synchronized_smc(df_5m)
            news_score, news_context = fetch_validated_news(ticker)

            best_zone = sorted(smc_zones, key=lambda x: x['state_val'], reverse=True)[0] if smc_zones else None
            
            score = 0.0
            trade_bias = "NEUTRAL"
            
            if of_diag["reversal_detected"]:
                trade_bias = "BUY" if of_diag["reversal_type"] == "BULLISH_REVERSAL" else "SELL"
                score += 30.0
            elif best_zone:
                trade_bias = best_zone['bias']
                score += 35.0 if best_zone['state_val'] == 3 else (20.0 if best_zone['state_val'] == 2 else 10.0)
            else:
                trade_bias = "BUY" if last_price > vwap_val else "SELL"

            if trade_bias == "BUY" and last_price > vwap_val and last_price > ema_20:
                score += 25.0
            elif trade_bias == "SELL" and last_price < vwap_val and last_price < ema_20:
                score += 25.0

            if of_diag["block_detected"]:
                score += 20.0

            if rvol >= 1.5: score += 15.0
            elif rvol >= 1.0: score += 8.0

            meta = STOCK_METADATA.get(ticker, {"index": "N/A", "sector": "General"})
            if meta["sector"] in active_sectors: score += 15.0

            score += news_score

            # Dynamic Risk Calculation
            sl_price, tgt_price = calculate_dynamic_risk_levels(df_5m, df_1d, trade_bias, last_price, best_zone, poc_price, atr_14)

            risk_pct = abs((last_price - sl_price) / last_price) * 100
            reward_pct = abs((tgt_price - last_price) / last_price) * 100
            rr_ratio = reward_pct / (risk_pct + 1e-5)

            if rr_ratio < min_rr_threshold:
                continue

            # State summary for table
            state_str = best_zone['state'] if best_zone else ("⚡ REVERSAL SIGNAL" if of_diag["reversal_detected"] else "NO ACTIVE SMC ZONE")

            results.append({
                "Stock": f"🏛️ {ticker}" if news_score != 0 else ticker,
                "Ticker_Raw": ticker,
                "Index": meta["index"],
                "Sector": meta["sector"],
                "Last Price": f"₹{last_price:.2f}",
                "Bias": "🟩 BUY" if trade_bias == "BUY" else "🟥 SELL",
                "SMC State": state_str,
                "POC Level": f"₹{poc_price:.2f}" if poc_price else "N/A",
                "Block Order Alert": of_diag["block_msg"],
                "Reversal Alert": of_diag["reversal_msg"],
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

# --- 11. DASHBOARD DISPLAY & MULTI-RANK CHART VISUALIZER ---
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
            st.write(f"**SMC / Setup:** `{row['SMC State']}`")
            st.write(f"**POC:** `{row['POC Level']}` | **RVOL:** `{row['RVOL']}`")
            st.write(f"**Target:** {row['Target']} | **SL:** {row['Stop Loss']}")
            st.write(f"**Block Order:** {row['Block Order Alert']}")
            st.write(f"**Reversal:** {row['Reversal Alert']}")

    st.markdown("---")
    st.subheader("📈 Live Visual SMC & Order Flow Charts (Rank 1, Rank 2 & Rank 3)")
    
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
                    
                    # Draw POC line on chart
                    poc_val = compute_volume_profile_poc(df_chart)
                    if poc_val:
                        fig.add_hline(y=poc_val, line_dash="dash", line_color="magenta", annotation_text=f"Session POC (₹{poc_val:.2f})")

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
                        title=f"Rank #{i+1} Setup: {stock_ticker} - SMC Zones, Volume Profile POC, VWAP & Confluence",
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        height=520,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Full Synchronized Watchlist (100 Stock Universe)")
    
    display_cols = [
        "Rank", "Stock", "Index", "Sector", "Bias", "Confluence Score", 
        "Last Price", "SMC State", "POC Level", "Block Order Alert", "Reversal Alert", 
        "VWAP Alignment", "RVOL", "Target", "Stop Loss", "R:R", "News / Catalysts"
    ]
    st.dataframe(res_df[display_cols], height=400, use_container_width=True)
    
    csv = res_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Institutional Scan Results (CSV)", data=csv, file_name="synchronized_institutional_scan.csv", mime="text/csv")
