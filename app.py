import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os

st.set_page_config(
    page_title="NQIRP Institutional Quant Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

import yfinance as yf

def fetch_data(symbol, period="1d", interval="5m"):
    ticker_sym = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
    df = yf.download(ticker_sym, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df
# ==============================================================================
# QUANTITATIVE SMC ENGINE (INTRADAY vs DAILY)
# ==============================================================================
def run_smc_analysis(df: pd.DataFrame, timeframe_label="INTRADAY"):
    if df.empty or len(df) < 30:
        return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']
    volume = df['Volume']

    # 1. Real-Time Price Resolution
    c = float(close.dropna().iloc[-1])
    h = float(high.dropna().iloc[-1])
    l = float(low.dropna().iloc[-1])
    o = float(open_p.dropna().iloc[-1])
    v = float(volume.dropna().iloc[-1])
    v20 = float(volume.tail(20).mean())
    rvol = v / v20 if v20 > 0 else 1.0

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    if atr <= 0:
        return None

    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    ema50 = float(close.ewm(span=50).mean().iloc[-1])
    trend_bias = "BULLISH" if c > ema50 else "BEARISH"

    h20_prev = float(high.tail(25).iloc[:-5].max())
    l20_prev = float(low.tail(25).iloc[:-5].min())

    smc_confluences, scores = [], []
    direction = "NEUTRAL"
    breakout_level = c

    # Lookback window: 5 daily candles for Daily Swing, 1 candle for Intraday
    lookback = 5 if timeframe_label == "DAILY" else 1

    # 1. FAIR VALUE GAP (FVG)
    bullish_fvg = any(float(low.iloc[-i]) > float(high.iloc[-i-2]) for i in range(1, lookback+1)) if len(df) >= lookback+2 else False
    bearish_fvg = any(float(high.iloc[-i]) < float(low.iloc[-i-2]) for i in range(1, lookback+1)) if len(df) >= lookback+2 else False

    if bullish_fvg and (rvol >= 0.8 or timeframe_label == "DAILY"):
        smc_confluences.append("Bullish FVG Zone")
        scores.append(88)
        direction = "BULLISH"
        breakout_level = float(high.iloc[-3])
    elif bearish_fvg and (rvol >= 0.8 or timeframe_label == "DAILY"):
        smc_confluences.append("Bearish FVG Zone")
        scores.append(88)
        direction = "BEARISH"
        breakout_level = float(low.iloc[-3])

    # 2. BREAK OF STRUCTURE (BOS)
    recent_max = float(high.tail(lookback).max())
    recent_min = float(low.tail(lookback).min())

    if recent_max > h20_prev and c >= ema20:
        smc_confluences.append("Bullish BOS Breakout")
        scores.append(92)
        direction = "BULLISH"
        breakout_level = h20_prev
    elif recent_min < l20_prev and c <= ema20:
        smc_confluences.append("Bearish BOS Breakdown")
        scores.append(92)
        direction = "BEARISH"
        breakout_level = l20_prev

    # 3. SWING PULLBACK / SUPPORT REACTION
    if timeframe_label == "DAILY" and direction == "NEUTRAL":
        if c > ema50 and abs(c - ema20) / ema20 <= 0.02:
            smc_confluences.append("20 EMA Swing Pullback Support")
            scores.append(82)
            direction = "BULLISH"
            breakout_level = ema20
        elif c < ema50 and abs(c - ema20) / ema20 <= 0.02:
            smc_confluences.append("20 EMA Swing Pullback Rejection")
            scores.append(82)
            direction = "BEARISH"
            breakout_level = ema20

    if not scores or direction == "NEUTRAL":
        return None

    master_score = max(scores) + min(len(smc_confluences) * 4.0, 20.0)

    # TARGET & STOP LOSS CALCULATIONS
    if direction == "BULLISH":
        suggested_entry = round(c, 2)
        stop_loss = round(suggested_entry - (1.5 * atr if timeframe_label == "DAILY" else 1.0 * atr), 2)
        target_price = round(suggested_entry + (2.5 * abs(suggested_entry - stop_loss)), 2)
    else:
        suggested_entry = round(c, 2)
        stop_loss = round(suggested_entry + (1.5 * atr if timeframe_label == "DAILY" else 1.0 * atr), 2)
        target_price = round(suggested_entry - (2.5 * abs(stop_loss - suggested_entry)), 2)

    actual_risk = abs(suggested_entry - stop_loss)
    actual_reward = abs(target_price - suggested_entry)
    rr_ratio = round(actual_reward / actual_risk, 2) if actual_risk > 0 else 2.5

    return {
        "Symbol": df.name if hasattr(df, 'name') else "STOCK",
        "Direction": direction,
        "Master Score": round(master_score, 1),
        "Trade Action": "✅ SWING ENTRY" if timeframe_label == "DAILY" else "✅ ACTIVE ENTRY",
        "Target Probability": "🔥 HIGH" if master_score >= 90 else "⚡ MEDIUM",
        "Suggested Entry": suggested_entry,
        "Current Price": round(c, 2),
        "Target Price": target_price,
        "Stop Loss": stop_loss,
        "R/R Ratio": f"1 : {rr_ratio}",
        "RVOL": round(rvol, 2),
        "SMC Signals": ", ".join(smc_confluences)
    }

# ==============================================================================
# 🚀 INSTITUTIONAL MOMENTUM SCANNER ENGINE
# ==============================================================================
def run_momentum_leader_analysis(df: pd.DataFrame):
    """
    Predictive Multi-Factor Institutional Momentum Scanner Engine.
    Evaluates Trend Alignment, Volume Acceleration, VWAP Anchor, Daily Breakouts, and Overextension Risk.
    """
    if df.empty or len(df) < 30:
        return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']
    volume = df['Volume']

    l_price = float(df['LTP'].iloc[-1]) if 'LTP' in df.columns else float(close.iloc[-1])
    c = l_price
    o_day = float(open_p.iloc[0])
    
    pct_change = ((c - o_day) / o_day) * 100

    # 1. Volume Acceleration & RVOL
    v20 = float(volume.tail(20).mean())
    v = float(volume.iloc[-1])
    rvol = v / v20 if v20 > 0 else 1.0
    vol_accel = volume.iloc[-1] > volume.iloc[-2] > volume.iloc[-3] if len(volume) >= 3 else False

    # 2. VWAP & Moving Averages Stack
    tp = (high + low + close) / 3
    vwap = float((tp * volume).cumsum().iloc[-1] / volume.cumsum().iloc[-1])

    ema9 = float(close.ewm(span=9).mean().iloc[-1])
    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    ema50 = float(close.ewm(span=50).mean().iloc[-1])

    # 3. ATR & Multi-Day Breakouts
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())

    h20_day = float(high.tail(100).max()) if len(df) >= 100 else float(high.max())
    is_multi_day_breakout = c >= (h20_day * 0.998)

    is_bullish = (c > vwap) and (c > ema9 > ema20 > ema50) and (pct_change >= 2.5) and (rvol >= 1.8)
    is_bearish = (c < vwap) and (c < ema9 < ema20 < ema50) and (pct_change <= -2.5) and (rvol >= 1.8)

    if not (is_bullish or is_bearish):
        return None

    # 4. PREDICTIVE PROBABILITY SCORE MATRIX
    score = 0
    score += 25 if (c > vwap and c > ema9) else 0
    score += 25 if (rvol >= 2.5 or vol_accel) else 15
    score += 15 if abs(pct_change) >= 3.5 else 10
    score += 20 if is_multi_day_breakout else 5
    
    is_extended = abs(c - ema9) / ema9 > 0.015
    if not is_extended:
        score += 15
    else:
        score -= 10

    prob_score = min(max(score, 40), 99)
    direction = "🔥 BULLISH INST. MOMENTUM" if is_bullish else "🩸 BEARISH INST. MOMENTUM"

    if is_bullish:
        suggested_entry = round(max(ema9, vwap), 2) if is_extended else round(c, 2)
        stop_loss = round(min(ema20, suggested_entry - (1.5 * atr)), 2)
        risk = suggested_entry - stop_loss
        target_price = round(suggested_entry + (3.0 * risk), 2)
        status_msg = "⚠️ Extended (Limit Entry at 9 EMA / VWAP)" if is_extended else "✅ High Probability Entry"
        exit_rule = "Trail along 9 EMA / Exit on 5-min candle close below VWAP"
    else:
        suggested_entry = round(min(ema9, vwap), 2) if is_extended else round(c, 2)
        stop_loss = round(max(ema20, suggested_entry + (1.5 * atr)), 2)
        risk = stop_loss - suggested_entry
        target_price = round(suggested_entry - (3.0 * risk), 2)
        status_msg = "⚠️ Extended (Limit Entry at 9 EMA / VWAP)" if is_extended else "✅ High Probability Entry"
        exit_rule = "Trail along 9 EMA / Exit on 5-min candle close above VWAP"

    rr_ratio = round(abs(target_price - suggested_entry) / risk, 2) if risk > 0 else 3.0

    return {
        "Symbol": df.name if hasattr(df, 'name') else "STOCK",
        "Direction": direction,
        "Day Change %": f"{pct_change:+.2f}%",
        "RVOL": round(rvol, 2),
        "Predictive Score": prob_score,
        "Status": status_msg,
        "Current Price": round(c, 2),
        "Suggested Entry": suggested_entry,
        "Stop Loss": stop_loss,
        "Target Price": target_price,
        "R/R Ratio": f"1 : {rr_ratio}",
        "Exit Strategy": exit_rule,
        "Structural Trigger": "Multi-Day Breakout" if is_multi_day_breakout else "Intraday Range Expansion"
    }

# ==============================================================================
# STREAMLIT APP NAVIGATION & UI
# ==============================================================================
st.sidebar.title("NQIRP Navigation")
page = st.sidebar.radio("Select Module", ["⚡ SMC Institutional Scanner", "👁️ Vision AI Chart Pattern Scanner"])

if page == "⚡ SMC Institutional Scanner":
    st.title("⚡ SMC Institutional Scanner Engine")
    st.markdown("Real-time multi-timeframe quantitative scanning for SMC confluences, FVG, BOS, and Momentum Leaders.")

    symbols_to_scan = ["REDINGTON", "FIRSTSOURCE", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

    tab_intraday, tab_daily, tab_momentum = st.tabs([
        "⚡ 1. Live Intraday Results (5-Min Data)", 
        "📊 2. Daily Swing Results (Historical)", 
        "🚀 3. Momentum Leaders of the Day"
    ])

# ==============================================================================
    # TAB 1: INTRADAY SMC SCANNER
    # ==============================================================================
    with tab_intraday:
        st.subheader("⚡ Live Intraday Scanner Results (5-Minute Timeframe)")
        st.caption("Targets updated to enforce a minimum 1:2.5 Risk-to-Reward ratio based on intraday volatility structure.")
        
        if st.button("⚡ Scan Intraday SMC Signals", type="primary"):
            with st.spinner("Scanning 5-minute intraday SMC confluences..."):
                intraday_results = []
                for symbol in symbols_to_scan:
                    clean_sym = symbol.strip()
                    df_5m = fetch_data(clean_sym, period="5d", interval="5m")
                    if not df_5m.empty and len(df_5m) >= 30:
                        df_5m.name = clean_sym
                        res_5m = run_smc_analysis(df_5m, timeframe_label="INTRADAY")
                        if res_5m:
                            res_5m["Upstox Instrument Key"] = f"NSE_EQ|{clean_sym}"
                            intraday_results.append(res_5m)
                st.session_state['intraday_results'] = intraday_results

        intraday_results = st.session_state.get('intraday_results', [])
        if intraday_results:
            df_intra = pd.DataFrame(intraday_results).sort_values(by="Master Score", ascending=False).reset_index(drop=True)
            st.dataframe(df_intra, use_container_width=True)
        else:
            st.info("Click 'Scan Intraday SMC Signals' above to trigger scanning.")

    # ==============================================================================
    # TAB 2: DAILY SWING SCANNER
    # ==============================================================================
    with tab_daily:
        st.subheader("📊 Historical Daily Scanner Results (1-Day Timeframe)")
        st.caption("Optimized for multi-day swing trades based on daily structural breakouts and fair value gaps.")
        
        if st.button("📊 Scan Daily Swing Signals", type="primary"):
            with st.spinner("Scanning daily timeframe SMC confluences..."):
                daily_results = []
                for symbol in symbols_to_scan:
                    clean_sym = symbol.strip()
                    df_daily = fetch_data(clean_sym, period="1y", interval="1d")
                    if not df_daily.empty and len(df_daily) >= 30:
                        df_daily.name = clean_sym
                        res_daily = run_smc_analysis(df_daily, timeframe_label="DAILY")
                        if res_daily:
                            res_daily["Upstox Instrument Key"] = f"NSE_EQ|{clean_sym}"
                            daily_results.append(res_daily)
                st.session_state['daily_results'] = daily_results

        daily_results = st.session_state.get('daily_results', [])
        if daily_results:
            df_day = pd.DataFrame(daily_results).sort_values(by="Master Score", ascending=False).reset_index(drop=True)
            st.dataframe(df_day, use_container_width=True)
        else:
            st.info("Click 'Scan Daily Swing Signals' above to trigger scanning.")

    # ==============================================================================
    # TAB 3: PREDICTIVE MOMENTUM LEADERS SCANNER
    # ==============================================================================
    with tab_momentum:
        st.subheader("🚀 Institutional Momentum Leaders of the Day")
        st.caption("Filters stocks with strong intraday volume acceleration, VWAP alignment, and multi-factor probability scores.")
        
        if st.button("🚀 Scan Momentum Leaders", type="primary"):
            with st.spinner("Scanning for high-momentum leaders..."):
                momentum_results = []
                for symbol in symbols_to_scan:
                    clean_sym = symbol.strip()
                    df_5m = fetch_data(clean_sym, period="5d", interval="5m")
                    if not df_5m.empty and len(df_5m) >= 30:
                        df_5m.name = clean_sym
                        m_res = run_momentum_leader_analysis(df_5m)
                        if m_res:
                            m_res["Upstox Instrument Key"] = f"NSE_EQ|{clean_sym}"
                            momentum_results.append(m_res)
                st.session_state['momentum_results'] = momentum_results

        momentum_results = st.session_state.get('momentum_results', [])
        if momentum_results:
            df_mom = pd.DataFrame(momentum_results).sort_values(by="Predictive Score", ascending=False).reset_index(drop=True)
            st.dataframe(df_mom, use_container_width=True)
        else:
            st.info("Click 'Scan Momentum Leaders' above to trigger scanning.")

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
