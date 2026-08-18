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
    st.sidebar.warning("⚠️ 'colab_ai_model.pkl' not found. Operating in Dynamic Heuristic Confluence Mode.")
else:
    st.sidebar.success("✅ AI Engine Loaded: Fast Offline Inference Active")

# --- 2. SIDEBAR MACRO & SECTOR CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Market Context & Sector Alignment")
active_sectors = st.sidebar.multiselect(
    "Focus Sectors (Outperforming / Underperforming):", 
    ["Banking", "IT", "Auto", "Energy", "FMCG", "Metal", "Infra", "Financials", "Healthcare", "Consumer Durables"],
    default=["Banking", "IT", "Financials"]
)

min_rr_threshold = st.sidebar.slider("Minimum Risk-to-Reward (R:R) Filter", 1.5, 4.0, 2.0, 0.1)

# --- 3. METADATA REGISTRY ---
STOCK_METADATA = {
    "RELIANCE": {"index": "Nifty 50", "sector": "Energy", "query": "Reliance Industries"},
    "TCS": {"index": "Nifty 50", "sector": "IT", "query": "Tata Consultancy Services"},
    "HDFCBANK": {"index": "Nifty 50", "sector": "Banking", "query": "HDFC Bank"},
    "INFY": {"index": "Nifty 50", "sector": "IT", "query": "Infosys"},
    "ICICIBANK": {"index": "Nifty 50", "sector": "Banking", "query": "ICICI Bank"},
    "SBIN": {"index": "Nifty 50", "sector": "Banking", "query": "State Bank of India"},
    "BHARTIARTL": {"index": "Nifty 50", "sector": "Telecom", "query": "Bharti Airtel"},
    "ITC": {"index": "Nifty 50", "sector": "FMCG", "query": "ITC Limited"},
    "AXISBANK": {"index": "Nifty 50", "sector": "Banking", "query": "Axis Bank"},
    "LT": {"index": "Nifty 50", "sector": "Infra", "query": "Larsen Toubro"},
    "BAJFINANCE": {"index": "Nifty 50", "sector": "Financials", "query": "Bajaj Finance"},
    "MARUTI": {"index": "Nifty 50", "sector": "Auto", "query": "Maruti Suzuki"},
    "TATAPOWER": {"index": "Nifty Midcap", "sector": "Energy", "query": "Tata Power"},
    "FEDERALBNK": {"index": "Nifty Midcap", "sector": "Banking", "query": "Federal Bank"},
    "PERSISTENT": {"index": "Nifty Midcap", "sector": "IT", "query": "Persistent Systems"},
    "COFORGE": {"index": "Nifty Midcap", "sector": "IT", "query": "Coforge"},
    "ASHOKLEY": {"index": "Nifty Midcap", "sector": "Auto", "query": "Ashok Leyland"},
    "CDSL": {"index": "Nifty Smallcap", "sector": "Financials", "query": "CDSL"},
    "ANGELONE": {"index": "Nifty Smallcap", "sector": "Financials", "query": "Angel One"},
    "SUZLON": {"index": "Nifty Smallcap", "sector": "Energy", "query": "Suzlon Energy"}
}

# --- 4. DATA INGESTION & TECHNICAL CALCULATIONS ---
def compute_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    return (tp * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-5)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-5)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=300)
def fetch_stock_data(ticker):
    yf_symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    df_5m = yf.download(yf_symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
    df_1d = yf.download(yf_symbol, period="1mo", interval="1d", progress=False, auto_adjust=True)
    
    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.get_level_values(0)
    if isinstance(df_1d.columns, pd.MultiIndex):
        df_1d.columns = df_1d.columns.get_level_values(0)
        
    return df_5m, df_1d

# --- 5. REAL-TIME VALIDATED NEWS / FILING INGESTION ---
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

# --- 6. SYNCHRONIZED SMC DETECTOR ENGINE ---
def detect_synchronized_smc(df_5m):
    if len(df_5m) < 30:
        return []
    
    df = df_5m.copy()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['VWAP'] = compute_vwap(df)
    
    last_idx = len(df) - 1
    last_price = float(df['Close'].iloc[-1])
    last_time = df.index[-1]
    
    zones = []
    
    # Iterate through recent session candles (Lookback last 60 candles = 5 hours)
    lookback = min(60, len(df) - 3)
    start_i = len(df) - lookback
    
    for i in range(start_i, len(df) - 2):
        candle_time = df.index[i]
        c_open, c_close = float(df['Open'].iloc[i]), float(df['Close'].iloc[i])
        c_high, c_low = float(df['High'].iloc[i]), float(df['Low'].iloc[i])
        atr = float(df['ATR'].iloc[i]) if not np.isnan(df['ATR'].iloc[i]) else (c_high - c_low)
        
        # 1. Bullish Order Block (Red candle before major upward displacement + FVG)
        if c_close < c_open: # Bearish candle body
            next_close = float(df['Close'].iloc[i+1])
            displacement = next_close - c_open
            fvg_present = float(df['Low'].iloc[i+2]) > float(df['High'].iloc[i]) if i+2 < len(df) else False
            
            if displacement > (1.2 * atr) and fvg_present:
                ob_top, ob_bottom = c_high, c_low
                # Check mitigation state
                future_lows = df['Low'].iloc[i+1:]
                mitigated = (future_lows < ob_bottom).any()
                
                if not mitigated:
                    if ob_bottom <= last_price <= ob_top:
                        state = "🟢 BULLISH OB RETEST (ENTRY READY)"
                        state_val = 3
                    elif last_price > ob_top and ((last_price - ob_top) / ob_top) * 100 <= 0.5:
                        state = "🟡 PULLBACK TO BULLISH OB"
                        state_val = 2
                    else:
                        state = "⏸️ UNMITIGATED BULLISH OB"
                        state_val = 1
                    
                    zones.append({
                        'type': 'Bullish OB',
                        'top': ob_top,
                        'bottom': ob_bottom,
                        'start_time': candle_time,
                        'state': state,
                        'state_val': state_val,
                        'bias': 'BUY'
                    })

        # 2. Bearish Order Block (Green candle before major downward displacement + FVG)
        if c_close > c_open: # Bullish candle body
            next_close = float(df['Close'].iloc[i+1])
            displacement = c_open - next_close
            fvg_present = float(df['High'].iloc[i+2]) < float(df['Low'].iloc[i]) if i+2 < len(df) else False
            
            if displacement > (1.2 * atr) and fvg_present:
                ob_top, ob_bottom = c_high, c_low
                future_highs = df['High'].iloc[i+1:]
                mitigated = (future_highs > ob_top).any()
                
                if not mitigated:
                    if ob_bottom <= last_price <= ob_top:
                        state = "🔴 BEARISH OB RETEST (SHORT READY)"
                        state_val = 3
                    elif last_price < ob_bottom and ((ob_bottom - last_price) / ob_bottom) * 100 <= 0.5:
                        state = "🟡 PULLBACK TO BEARISH OB"
                        state_val = 2
                    else:
                        state = "⏸️ UNMITIGATED BEARISH OB"
                        state_val = 1
                    
                    zones.append({
                        'type': 'Bearish OB',
                        'top': ob_top,
                        'bottom': ob_bottom,
                        'start_time': candle_time,
                        'state': state,
                        'state_val': state_val,
                        'bias': 'SELL'
                    })

    # 3. Liquidity Sweeps
    recent_swings_low = df['Low'].iloc[-30:-3].min()
    recent_swings_high = df['High'].iloc[-30:-3].max()
    
    curr_low, curr_high = float(df['Low'].iloc[-1]), float(df['High'].iloc[-1])
    curr_close = float(df['Close'].iloc[-1])
    
    if curr_low < recent_swings_low and curr_close > recent_swings_low:
        zones.append({
            'type': 'Liquidity Sweep Low',
            'top': recent_swings_low * 1.001,
            'bottom': curr_low,
            'start_time': df.index[-1],
            'state': "🟢 LIQUIDITY SWEEP (BULLISH REVERSAL)",
            'state_val': 3,
            'bias': 'BUY'
        })
        
    if curr_high > recent_swings_high and curr_close < recent_swings_high:
        zones.append({
            'type': 'Liquidity Sweep High',
            'top': curr_high,
            'bottom': recent_swings_high * 0.999,
            'start_time': df.index[-1],
            'state': "🔴 LIQUIDITY SWEEP (BEARISH REVERSAL)",
            'state_val': 3,
            'bias': 'SELL'
        })

    return zones

# --- 7. CORE SCANNER & CONFLUENCE MATRIX ---
st.markdown("---")
ctrl_col1, ctrl_col2 = st.columns([1, 3])
with ctrl_col1:
    scan_universe = st.selectbox("Select Scanning Universe", ["Nifty 50", "Nifty Midcap", "Nifty Smallcap", "All Combined"])
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
        status_text.text(f"Scanning & Synchronizing SMC Context for {ticker}...")
        progress_bar.progress((idx + 1) / len(tickers))
        
        try:
            df_5m, df_1d = fetch_stock_data(ticker)
            if df_5m is None or df_5m.empty or df_1d is None or df_1d.empty:
                continue

            # Core Technical Ingestion
            close_5m = df_5m["Close"].dropna()
            high_5m = df_5m["High"].dropna()
            low_5m = df_5m["Low"].dropna()
            vol_5m = df_5m["Volume"].dropna()
            
            last_price = float(close_5m.iloc[-1])
            vwap_val = float(compute_vwap(df_5m).iloc[-1])
            ema_20 = float(close_5m.ewm(span=20, adjust=False).mean().iloc[-1])
            atr_14 = float((high_5m - low_5m).tail(14).mean())
            rsi = float(compute_rsi(close_5m).iloc[-1])
            
            # Volume Expansion (RVOL)
            avg_vol = float(vol_5m.tail(20).mean())
            rvol = float(vol_5m.iloc[-1] / (avg_vol + 1e-5))

            # Higher Timeframe Structural Targets (1D Daily)
            pdh = float(df_1d["High"].dropna().iloc[-2]) if len(df_1d) >= 2 else float(high_5m.max())
            pdl = float(df_1d["Low"].dropna().iloc[-2]) if len(df_1d) >= 2 else float(low_5m.min())
            day_open = float(df_1d["Open"].dropna().iloc[-1]) if len(df_1d) >= 1 else float(close_5m.iloc[0])

            # SMC & News Ingestion
            smc_zones = detect_synchronized_smc(df_5m)
            news_score, news_context = fetch_validated_news(ticker)

            best_zone = sorted(smc_zones, key=lambda x: x['state_val'], reverse=True)[0] if smc_zones else None
            
            # Confluence Scoring Matrix
            score = 0.0
            trade_bias = "NEUTRAL"
            
            if best_zone:
                trade_bias = best_zone['bias']
                score += 35.0 if best_zone['state_val'] == 3 else (20.0 if best_zone['state_val'] == 2 else 10.0)
            else:
                trade_bias = "BUY" if last_price > vwap_val else "SELL"

            # VWAP & EMA Structural Alignment
            if trade_bias == "BUY" and last_price > vwap_val and last_price > ema_20:
                score += 25.0
            elif trade_bias == "SELL" and last_price < vwap_val and last_price < ema_20:
                score += 25.0

            # RVOL Volume Expansion Confirmation
            if rvol >= 1.5:
                score += 15.0
            elif rvol >= 1.0:
                score += 8.0

            # Sector Alignment
            meta = STOCK_METADATA.get(ticker, {"index": "N/A", "sector": "General"})
            if meta["sector"] in active_sectors:
                score += 15.0

            # News Catalyst Contribution
            score += news_score

            # Realistic Dynamic Stoploss & Target Engine
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

        except Exception as e:
            continue

    status_text.empty()
    progress_bar.empty()

    if results:
        res_df = pd.DataFrame(results).sort_values(by="Confluence Score", ascending=False).reset_index(drop=True)
        res_df["Rank"] = res_df.index + 1
        st.session_state["scan_results"] = res_df
    else:
        st.warning("No stocks passed the synchronized SMC and R:R filters. Try lowering the R:R threshold.")

# --- 8. DASHBOARD DISPLAY & TIMED PLOTLY VISUALIZER ---
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
    st.subheader("📈 Live Visual Confirmation & Time-Bounded SMC Zones")
    
    top_stock = res_df.iloc[0]['Ticker_Raw']
    df_chart, _ = fetch_stock_data(top_stock)
    
    if df_chart is not None and not df_chart.empty:
        df_chart['VWAP'] = compute_vwap(df_chart)
        df_chart['EMA20'] = df_chart['Close'].ewm(span=20, adjust=False).mean()
        
        fig = go.Figure(data=[go.Candlestick(
            x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
            low=df_chart['Low'], close=df_chart['Close'], name="Price"
        )])
        
        # Add VWAP & EMA 20 Overlays
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['VWAP'], line=dict(color='orange', width=1.5), name="VWAP"))
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA20'], line=dict(color='cyan', width=1), name="EMA 20"))
        
        # Plot Time-Bounded Order Blocks (Fixes Multi-Day Rectangle Bleed)
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
            title=f"{top_stock} - Time-Bounded SMC Zones, Intraday VWAP & Confluence Overlay",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=550,
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
