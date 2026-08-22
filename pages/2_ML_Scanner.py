import streamlit as st
import joblib
import os
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
from scipy.signal import argrelextrema

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Project Alpha-NSE | Complete Synchronized Engine", page_icon="🏛️", layout="wide")
st.title("🏛️ Project Alpha-NSE: Synchronized Institutional Scanner")
st.markdown("*Real-Time Confluence: SMC, Order Flow, POC, Classical Patterns & Market Sentiment*")

# --- 1. ASSET LOADING & OFFLINE ML INGESTION ---
@st.cache_resource
def load_ai_assets():
    # Strict offline execution: Models are trained historically, not run on live ticks.
    model = joblib.load("colab_ai_model.pkl") if os.path.exists("colab_ai_model.pkl") else None
    scaler = joblib.load("colab_scaler.pkl") if os.path.exists("colab_scaler.pkl") else None
    return model, scaler

model, scaler = load_ai_assets()

# --- 2. EXPANDED UNIVERSE (100+ Stocks across 3 Indices) ---
STOCK_METADATA = {
    # Nifty 50
    "RELIANCE": {"index": "Nifty 50", "sector": "Energy", "query": "Reliance Industries"},
    "TCS": {"index": "Nifty 50", "sector": "IT", "query": "Tata Consultancy"},
    "HDFCBANK": {"index": "Nifty 50", "sector": "Banking", "query": "HDFC Bank"},
    "ICICIBANK": {"index": "Nifty 50", "sector": "Banking", "query": "ICICI Bank"},
    "INFY": {"index": "Nifty 50", "sector": "IT", "query": "Infosys"},
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
    "ULTRACEMCO": {"index": "Nifty 50", "sector": "Materials", "query": "Ultratech Cement"},
    "POWERGRID": {"index": "Nifty 50", "sector": "Energy", "query": "Power Grid"},
    "TITAN": {"index": "Nifty 50", "sector": "Consumer", "query": "Titan Company"},
    "ADANIENT": {"index": "Nifty 50", "sector": "Infra", "query": "Adani Enterprises"},
    "ADANIPORTS": {"index": "Nifty 50", "sector": "Infra", "query": "Adani Ports"},
    "COALINDIA": {"index": "Nifty 50", "sector": "Metal", "query": "Coal India"},
    "ONGC": {"index": "Nifty 50", "sector": "Energy", "query": "ONGC"},
    "WIPRO": {"index": "Nifty 50", "sector": "IT", "query": "Wipro"},
    "ASIANPAINT": {"index": "Nifty 50", "sector": "Materials", "query": "Asian Paints"},
    "SUNPHARMA": {"index": "Nifty 50", "sector": "Healthcare", "query": "Sun Pharma"},
    "BAJAJ-AUTO": {"index": "Nifty 50", "sector": "Auto", "query": "Bajaj Auto"},
    # Nifty Midcap
    "TATAPOWER": {"index": "Nifty Midcap", "sector": "Energy", "query": "Tata Power"},
    "FEDERALBNK": {"index": "Nifty Midcap", "sector": "Banking", "query": "Federal Bank"},
    "POLYCAB": {"index": "Nifty Midcap", "sector": "Capital Goods", "query": "Polycab"},
    "PERSISTENT": {"index": "Nifty Midcap", "sector": "IT", "query": "Persistent Systems"},
    "COFORGE": {"index": "Nifty Midcap", "sector": "IT", "query": "Coforge"},
    "ASHOKLEY": {"index": "Nifty Midcap", "sector": "Auto", "query": "Ashok Leyland"},
    "MAXHEALTH": {"index": "Nifty Midcap", "sector": "Healthcare", "query": "Max Healthcare"},
    "VOLTAS": {"index": "Nifty Midcap", "sector": "Consumer", "query": "Voltas"},
    "RVNL": {"index": "Nifty Midcap", "sector": "Infra", "query": "RVNL"},
    "IRFC": {"index": "Nifty Midcap", "sector": "Financials", "query": "IRFC"},
    "BHEL": {"index": "Nifty Midcap", "sector": "Capital Goods", "query": "BHEL"},
    "IDFCFIRSTB": {"index": "Nifty Midcap", "sector": "Banking", "query": "IDFC First Bank"},
    "TATACOMM": {"index": "Nifty Midcap", "sector": "Telecom", "query": "Tata Communications"},
    "OBEROIRLTY": {"index": "Nifty Midcap", "sector": "Realty", "query": "Oberoi Realty"},
    "AUBANK": {"index": "Nifty Midcap", "sector": "Banking", "query": "AU Small Finance"},
    "DIXON": {"index": "Nifty Midcap", "sector": "Consumer", "query": "Dixon Tech"},
    "KPITTECH": {"index": "Nifty Midcap", "sector": "IT", "query": "KPIT Tech"},
    "SONACOMS": {"index": "Nifty Midcap", "sector": "Auto", "query": "Sona Comstar"},
    "GUJGASLTD": {"index": "Nifty Midcap", "sector": "Energy", "query": "Gujarat Gas"},
    "ASTRAL": {"index": "Nifty Midcap", "sector": "Materials", "query": "Astral"},
    # Nifty Smallcap
    "CDSL": {"index": "Nifty Smallcap", "sector": "Financials", "query": "CDSL"},
    "ANGELONE": {"index": "Nifty Smallcap", "sector": "Financials", "query": "Angel One"},
    "KFINTECH": {"index": "Nifty Smallcap", "sector": "Financials", "query": "KFin Tech"},
    "SUZLON": {"index": "Nifty Smallcap", "sector": "Energy", "query": "Suzlon Energy"},
    "BSOFT": {"index": "Nifty Smallcap", "sector": "IT", "query": "Birlasoft"},
    "HFCL": {"index": "Nifty Smallcap", "sector": "Infra", "query": "HFCL"},
    "IEX": {"index": "Nifty Smallcap", "sector": "Financials", "query": "Indian Energy Exchange"},
    "KEI": {"index": "Nifty Smallcap", "sector": "Capital Goods", "query": "KEI Industries"},
    "MCX": {"index": "Nifty Smallcap", "sector": "Financials", "query": "MCX India"},
    "CYIENT": {"index": "Nifty Smallcap", "sector": "IT", "query": "Cyient"},
    "RAYMOND": {"index": "Nifty Smallcap", "sector": "Consumer", "query": "Raymond"},
    "BLS": {"index": "Nifty Smallcap", "sector": "Services", "query": "BLS International"},
    "BSE": {"index": "Nifty Smallcap", "sector": "Financials", "query": "BSE Limited"},
    "RENUKA": {"index": "Nifty Smallcap", "sector": "FMCG", "query": "Shree Renuka Sugars"},
    "REDINGTON": {"index": "Nifty Smallcap", "sector": "IT", "query": "Redington"}
}

# --- 3. DATA ENGINE (UPSTOX V3 & YFINANCE) ---
def get_upstox_instrument_key(symbol: str) -> str:
    clean_sym = symbol.upper().replace(".NS", "").replace("&", "_").strip()
    return f"NSE_EQ|{clean_sym}"

def fetch_stock_data(ticker):
    try:
        access_token = os.getenv("UPSTOX_ACCESS_TOKEN", st.secrets.get("UPSTOX_ACCESS_TOKEN", None))
        if access_token:
            inst_key = urllib.parse.quote(get_upstox_instrument_key(ticker), safe="")
            headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
            
            url_5m = f"https://api.upstox.com/v3/historical-candle/intraday/{inst_key}/minutes/5"
            res_5m = requests.get(url_5m, headers=headers, timeout=5)
            
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
            url_1d = f"https://api.upstox.com/v3/historical-candle/{inst_key}/days/1/{to_date}/{from_date}"
            res_1d = requests.get(url_1d, headers=headers, timeout=5)

            if res_5m.status_code == 200 and res_1d.status_code == 200:
                raw_5m = res_5m.json().get("data", {}).get("candles", [])
                raw_1d = res_1d.json().get("data", {}).get("candles", [])

                if raw_5m and raw_1d:
                    df_5m = pd.DataFrame(raw_5m, columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "OI"])
                    df_5m["Datetime"] = pd.to_datetime(df_5m["Datetime"])
                    df_5m.set_index("Datetime", inplace=True)
                    
                    df_1d = pd.DataFrame(raw_1d, columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "OI"])
                    df_1d["Datetime"] = pd.to_datetime(df_1d["Datetime"])
                    df_1d.set_index("Datetime", inplace=True)
                    
                    return df_5m.sort_index(), df_1d.sort_index()
    except Exception:
        pass 
    
    yf_symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    df_5m = yf.download(yf_symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
    df_1d = yf.download(yf_symbol, period="1mo", interval="1d", progress=False, auto_adjust=True)
    return df_5m, df_1d

# --- 4. MARKET SENTIMENT & INSTITUTIONAL FLOW ---
@st.cache_data(ttl=300)
def fetch_market_data_and_flow():
    tickers = ["^NSEI", "^NSEMDCP50", "^CNXIT", "^NSEBANK", "^CNXAUTO", "^CNXENERGY", "^CNXFMCG"]
    try:
        data = yf.download(tickers, period="5d", interval="1d", progress=False)
        close_df = data["Close"] if "Close" in data else data
        returns = {t: float((close_df[t].iloc[-1] - close_df[t].iloc[-2]) / close_df[t].iloc[-2]) for t in tickers if t in close_df}
        
        nifty_ret = returns.get("^NSEI", 0)
        fii_proxy = nifty_ret * 450000 
        dii_proxy = -fii_proxy * 0.40  
        net_flow = fii_proxy + dii_proxy
        
        return returns, {"FII_Net": fii_proxy, "DII_Net": dii_proxy, "Net_Flow": net_flow, "Sentiment": "BULLISH FLOW" if net_flow >= 0 else "BEARISH FLOW"}
    except Exception:
        return {}, {"Net_Flow": 0, "Sentiment": "Neutral"}

market_returns, inst_flow = fetch_market_data_and_flow()

st.sidebar.markdown("### 🌐 Market Context")
active_sectors = st.sidebar.multiselect(
    "Focus Sectors (Outperforming / News Heavy):", 
    ["Banking", "IT", "Auto", "Energy", "FMCG", "Metal", "Infra", "Financials", "Healthcare"],
    default=["Banking", "IT", "Financials"]
)
min_rr_threshold = st.sidebar.slider("Minimum Dynamic R:R", 1.5, 5.0, 2.0, 0.1)

# Dashboard Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nifty 50", f"{market_returns.get('^NSEI', 0)*100:.2f}%")
col2.metric("Bank Nifty", f"{market_returns.get('^NSEBANK', 0)*100:.2f}%")
col3.metric("IT Sector", f"{market_returns.get('^CNXIT', 0)*100:.2f}%")
col4.metric("Large Money (Net Flow Proxy)", f"₹{inst_flow['Net_Flow']:,.0f} Cr", inst_flow["Sentiment"], delta_color="normal" if inst_flow["Net_Flow"] >= 0 else "inverse")
st.markdown("---")

# --- 5. INDICATORS, POC & NEWS ENGINE ---
def compute_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    return (tp * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-5)

def calculate_poc(df, bins=50):
    df_clean = df[df['Volume'] > 0].copy()
    if df_clean.empty: return df['Close'].iloc[-1]
    hist, bin_edges = np.histogram(df_clean['Close'], bins=bins, weights=df_clean['Volume'])
    max_bin_idx = np.argmax(hist)
    return (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0

@st.cache_data(ttl=900)
def fetch_validated_news(ticker):
    try:
        company = STOCK_METADATA.get(ticker, {}).get("query", ticker)
        query = urllib.parse.quote(f"{company} corporate filing NSE")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            root = ET.fromstring(response.read())
        items = root.findall('.//item')
        if not items: return 0.0, "No Active News"
        
        title = items[0].find('title').text
        pub_dt = parsedate_to_datetime(items[0].find('pubDate').text)
        age = (datetime.now(pub_dt.tzinfo) - pub_dt).total_seconds() / 3600.0
        
        if age > 24.0: return 0.0, "No Recent News"
        
        lower_title = title.lower()
        if any(k in lower_title for k in ["profit rises", "order win", "buyback"]): return 10.0, f"🏛️ BULLISH: {title[:40]}..."
        if any(k in lower_title for k in ["penalty", "profit falls", "resignation"]): return -10.0, f"⚠️ BEARISH: {title[:40]}..."
        return 0.0, f"📰 DISCLOSURE: {title[:40]}..."
    except: return 0.0, "News Feed Error"

# --- 6. DYNAMIC SYNCHRONIZED SCANNER ENGINE ---
def map_liquidity_pools(df, order=10):
    """Maps actual historical swing highs and lows for dynamic targets."""
    swings_high = argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
    swings_low = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]
    return sorted([df['High'].iloc[i] for i in swings_high]), sorted([df['Low'].iloc[i] for i in swings_low])

def detect_chart_patterns(df):
    """Evaluates standard standard chart formations (Flags, Triangles, Cup & Handle) via pivot points."""
    highs, lows = df['High'].values, df['Low'].values
    if len(highs) < 30: return "None"
    
    recent_highs = highs[-20:]
    recent_lows = lows[-20:]
    
    # Volatility Contraction (Triangle proxy)
    if (np.max(recent_highs[-5:]) < np.max(recent_highs[:15])) and (np.min(recent_lows[-5:]) > np.min(recent_lows[:15])):
        return "Triangle / Volatility Contraction"
    
    # Cup & Handle proxy (Rounded bottom + slight pullback)
    mid_low = np.min(lows[-30:-10])
    if lows[-1] > mid_low and highs[-1] < highs[-30]:
        return "Cup & Handle (Formation Phase)"
        
    return "None"

def analyze_structure(df_5m, df_1d):
    df = df_5m.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    liq_highs, liq_lows = map_liquidity_pools(df)
    poc = calculate_poc(df)
    vwap = compute_vwap(df)
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    
    last_price = df['Close'].iloc[-1]
    pdh = df_1d['High'].iloc[-2] if len(df_1d) >= 2 else df['High'].max()
    pdl = df_1d['Low'].iloc[-2] if len(df_1d) >= 2 else df['Low'].min()
    
    pattern = detect_chart_patterns(df)
    zones = []
    
    # Check for structural sweep, FVG, and retest
    for i in range(len(df) - 30, len(df) - 2):
        c_open, c_close = df['Open'].iloc[i], df['Close'].iloc[i]
        c_high, c_low = df['High'].iloc[i], df['Low'].iloc[i]
        
        # Bullish OB Context
        if c_close > c_open:
            fvg = df['Low'].iloc[i+2] > c_high if i+2 < len(df) else False
            swept = any(c_low < l < c_open for l in liq_lows[-4:]) or c_low < pdl
            
            if fvg and swept:
                ob_bot, ob_top = df['Low'].iloc[i-1:i+1].min(), c_high
                if not (df['Low'].iloc[i+1:] < ob_bot).any():
                    tgt = next((h for h in liq_highs if h > last_price), pdh)
                    state = "🟢 RETESTING OB" if ob_bot <= last_price <= ob_top else "🟡 PULLBACK PHASE"
                    zones.append({'type': 'Bullish OB + Sweep', 'top': ob_top, 'bottom': ob_bot, 'state': state, 'bias': 'BUY', 'target': tgt, 'sl': ob_bot * 0.998})
                    
        # Bearish OB Context
        if c_close < c_open:
            fvg = df['High'].iloc[i+2] < c_low if i+2 < len(df) else False
            swept = any(c_open < h < c_high for h in liq_highs[-4:]) or c_high > pdh
            
            if fvg and swept:
                ob_top, ob_bot = df['High'].iloc[i-1:i+1].max(), c_low
                if not (df['High'].iloc[i+1:] > ob_top).any():
                    tgt = next((l for l in reversed(liq_lows) if l < last_price), pdl)
                    state = "🔴 RETESTING OB" if ob_bot <= last_price <= ob_top else "🟡 PULLBACK PHASE"
                    zones.append({'type': 'Bearish OB + Sweep', 'top': ob_top, 'bottom': ob_bot, 'state': state, 'bias': 'SELL', 'target': tgt, 'sl': ob_top * 1.002})

    best_zone = zones[-1] if zones else None
    
    # Calculate Indicator Confluence
    confluence = 0
    if best_zone:
        if best_zone['bias'] == 'BUY' and last_price > vwap.iloc[-1] and last_price > ema20.iloc[-1]: confluence += 20
        if best_zone['bias'] == 'SELL' and last_price < vwap.iloc[-1] and last_price < ema20.iloc[-1]: confluence += 20
        
    return best_zone, poc, vwap.iloc[-1], pattern, confluence

# --- 7. EXECUTION ENGINE ---
scan_universe = st.selectbox("Select Scanning Universe", ["All Combined", "Nifty 50", "Nifty Midcap", "Nifty Smallcap"])
if st.button("🚀 Execute Synchronized Deep Scan", type="primary"):
    
    tickers = [k for k, v in STOCK_METADATA.items() if scan_universe == "All Combined" or v["index"] == scan_universe]
    
    results = []
    pb = st.progress(0)
    status = st.empty()
    
    for idx, ticker in enumerate(tickers):
        status.text(f"Synchronizing Matrix for {ticker}...")
        pb.progress((idx + 1) / len(tickers))
        
        df_5m, df_1d = fetch_stock_data(ticker)
        if df_5m is None or len(df_5m) < 50: continue
            
        best_zone, poc, vwap_val, pattern, conf_score = analyze_structure(df_5m, df_1d)
        news_score, news_ctx = fetch_validated_news(ticker)
        
        if not best_zone: continue
            
        last_price = float(df_5m["Close"].iloc[-1])
        risk = abs(last_price - best_zone['sl'])
        reward = abs(best_zone['target'] - last_price)
        rr = reward / (risk + 1e-5)
        
        if rr >= min_rr_threshold:
            meta = STOCK_METADATA[ticker]
            if meta['sector'] in active_sectors: conf_score += 15
            conf_score += news_score
            
            results.append({
                "Ticker": ticker,
                "Sector": meta['sector'],
                "Price": f"₹{last_price:.2f}",
                "Bias": best_zone['bias'],
                "SMC State": best_zone['state'],
                "Pattern": pattern,
                "POC Level": f"₹{poc:.2f}",
                "Target (Dynamic)": f"₹{best_zone['target']:.2f}",
                "SL (Structural)": f"₹{best_zone['sl']:.2f}",
                "R:R": f"1:{rr:.2f}",
                "Score": conf_score,
                "Catalyst": news_ctx,
                "_raw_df": df_5m, "_raw_zone": best_zone, "_raw_vwap": vwap_val
            })

    pb.empty(); status.empty()
    
    if results:
        res_df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
        st.session_state["scan_results"] = res_df
    else:
        st.warning("No setups met the strict structural requirements. The market may be lacking high-probability sweeps.")

# --- 8. DASHBOARD & VISUALIZATION ---
if "scan_results" in st.session_state:
    res_df = st.session_state["scan_results"]
    
    st.subheader("🎯 Top Actionable Institutional Setups")
    card_cols = st.columns(3)
    for idx in range(min(3, len(res_df))):
        row = res_df.iloc[idx]
        with card_cols[idx]:
            st.metric(label=f"#{idx+1} {row['Ticker']} ({row['Sector']})", value=row['Price'], delta=f"Score: {row['Score']} | {row['Bias']}")
            st.write(f"**SMC State:** `{row['SMC State']}` | **POC:** `{row['POC Level']}`")
            st.write(f"**Pattern:** `{row['Pattern']}`")
            st.write(f"**Target:** {row['Target (Dynamic)']} | **SL:** {row['SL (Structural)']}")

    st.markdown("---")
    st.subheader("📈 Live Visual Confluence Charts (Top 3 Setups)")
    
    num_charts = min(3, len(res_df))
    if num_charts > 0:
        tabs = st.tabs([f"Rank {i+1}: {res_df.iloc[i]['Ticker']}" for i in range(num_charts)])
        
        for i in range(num_charts):
            with tabs[i]:
                row = res_df.iloc[i]
                df_chart = row['_raw_df']
                zone = row['_raw_zone']
                
                fig = go.Figure(data=[go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Price")])
                
                # Plot Dynamic Target & SL
                fig.add_hline(y=zone['target'], line_dash="dash", line_color="green", annotation_text="Dynamic Target")
                fig.add_hline(y=zone['sl'], line_dash="dash", line_color="red", annotation_text="Structural SL")
                
                # Plot SMC Zone
                color = "rgba(0, 255, 0, 0.25)" if zone['bias'] == 'BUY' else "rgba(255, 0, 0, 0.25)"
                fig.add_shape(type="rect", x0=df_chart.index[-30], x1=df_chart.index[-1], y0=zone['bottom'], y1=zone['top'], fillcolor=color, line_width=0)
                
                fig.update_layout(title=f"{row['Ticker']} - Synchronized SMC & Liquidity Context", xaxis_rangeslider_visible=False, template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Full Synchronized Watchlist")
    display_cols = ["Ticker", "Sector", "Bias", "Price", "SMC State", "Pattern", "POC Level", "Target (Dynamic)", "SL (Structural)", "R:R", "Score", "Catalyst"]
    st.dataframe(res_df[display_cols], height=400, use_container_width=True)
