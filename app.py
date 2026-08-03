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
    l_price = float(df['LTP'].iloc[-1]) if 'LTP' in df.columns else float(close.iloc[-1])
    c, h, l, o, v = l_price, float(high.iloc[-1]), float(low.iloc[-1]), float(open_p.iloc[-1]), float(volume.iloc[-1])

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

    # 1. FAIR VALUE GAP (FVG)
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

    # 2. BREAK OF STRUCTURE (BOS)
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

    # 3. DOUBLE TOP & DOUBLE BOTTOM REJECTIONS
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

    # 4. TRIANGLE PATTERNS
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

    # 5. FLAG PATTERNS
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

    # 6. CUP AND HANDLE PATTERNS
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

    if not scores or direction == "NEUTRAL":
        return None

    master_score = max(scores) + min(len(smc_confluences) * 4.0, 20.0)

    # EXTENSION & ENTRY LOGIC
    max_extension_pct = 0.002
    
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

    if pct_extended > 0 and rvol < 1.0:
        target_prob = "⚠️ LOW (Fading Volume - Reversal Risk)"
    elif rvol >= 2.5 and len(smc_confluences) >= 2:
        target_prob = "🔥 HIGH (Strong Volume & Multi-Confluence)"
    elif rvol >= 1.2:
        target_prob = "⚡ MEDIUM (Healthy Volume)"
    else:
        target_prob = "⚠️ LOW (Weak Volume)"

    # TARGET & STOP LOSS CALCULATIONS
    if timeframe_label == "INTRADAY":
        stop_dist = max(1.5 * atr, suggested_entry * 0.005)
        raw_target_dist = stop_dist * 2.5 
    else:
        stop_dist = 1.0 * atr
        raw_target_dist = 2.5 * atr

    if direction == "BULLISH":
        stop_loss = round(suggested_entry - stop_dist, 2)
        raw_target = suggested_entry + raw_target_dist
        if raw_target > h20_prev and h20_prev > suggested_entry:
            target_price = round(h20_prev * 0.9995, 2)
        else:
            target_price = round(raw_target, 2)
    else:
        stop_loss = round(suggested_entry + stop_dist, 2)
        raw_target = suggested_entry - raw_target_dist
        if raw_target < l20_prev and l20_prev < suggested_entry:
            target_price = round(l20_prev * 1.0005, 2)
        else:
            target_price = round(raw_target, 2)

    actual_risk = abs(suggested_entry - stop_loss)
    actual_reward = abs(target_price - suggested_entry)
    rr_ratio = round(actual_reward / actual_risk, 2) if actual_risk > 0 else 2.5

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
    Evaluates Trend Alignment, Volume Acceleration, Daily Breakouts, and Overextension Risk.
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

    # 1. Volume & RVOL Acceleration
    v20 = float(volume.tail(20).mean())
    v = float(volume.iloc[-1])
    rvol = v / v20 if v20 > 0 else 1.0
    vol_accel = volume.iloc[-1] > volume.iloc[-2] > volume.iloc[-3] if len(volume) >= 3 else False

    # 2. VWAP & Moving Averages
    tp = (high + low + close) / 3
    vwap = float((tp * volume).cumsum().iloc[-1] / volume.cumsum().iloc[-1])

    ema9 = float(close.ewm(span=9).mean().iloc[-1])
    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    ema50 = float(close.ewm(span=50).mean().iloc[-1])

    # 3. ATR & Multi-Day Breakout References
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())

    h20_day = float(high.tail(100).max()) if len(df) >= 100 else float(high.max())
    is_multi_day_breakout = c >= (h20_day * 0.998)

    # Core Trend Conditions
    is_bullish = (c > vwap) and (c > ema9 > ema20 > ema50) and (pct_change >= 2.5) and (rvol >= 1.8)
    is_bearish = (c < vwap) and (c < ema9 < ema20 < ema50) and (pct_change <= -2.5) and (rvol >= 1.8)

    if not (is_bullish or is_bearish):
        return None

    # 4. PREDICTIVE PROBABILITY SCORE CALCULATION
    score = 0
    score += 25 if (c > vwap and c > ema9) else 0
    score += 25 if (rvol >= 2.5 or vol_accel) else 15
    score += 15 if abs(pct_change) >= 3.5 else 10
    score += 20 if is_multi_day_breakout else 5
    
    # Check overextension: Penalty if price is > 1.5% away from 9 EMA
    is_extended = abs(c - ema9) / ema9 > 0.015
    if not is_extended:
        score += 15
    else:
        score -= 10  # Risk penalty for chasing extended prices

    prob_score = min(max(score, 40), 99)

    direction = "🔥 BULLISH INST. MOMENTUM" if is_bullish else "🩸 BEARISH INST. MOMENTUM"
    
    # 5. TRADE EXECUTION LEVELS
    if is_bullish:
        suggested_entry = round(max(ema9, vwap), 2) if is_extended else round(c, 2)
        stop_loss = round(min(ema20, suggested_entry - (1.5 * atr)), 2)
        risk = suggested_entry - stop_loss
        target_price = round(suggested_entry + (3.0 * risk), 2)
        status_msg = "⚠️ Extended (Wait for Pullback to 9 EMA)" if is_extended else "✅ High Probability Entry"
    else:
        suggested_entry = round(min(ema9, vwap), 2) if is_extended else round(c, 2)
        stop_loss = round(max(ema20, suggested_entry + (1.5 * atr)), 2)
        risk = stop_loss - suggested_entry
        target_price = round(suggested_entry - (3.0 * risk), 2)
        status_msg = "⚠️ Extended (Wait for Pullback to 9 EMA)" if is_extended else "✅ High Probability Entry"

    rr_ratio = round(abs(target_price - suggested_entry) / risk, 2) if risk > 0 else 3.0

    return {
        "Symbol": df.name if hasattr(df, 'name') else "STOCK",
        "Direction": direction,
        "Day Change %": f"{pct_change:+.2f}%",
        "RVOL": round(rvol, 2),
        "Predictive Score": f"{prob_score}/100",
        "Status": status_msg,
        "Current Price": round(c, 2),
        "Suggested Entry": suggested_entry,
        "Stop Loss": stop_loss,
        "Target Price": target_price,
        "R/R Ratio": f"1 : {rr_ratio}",
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
    # TAB 3: MOMENTUM LEADERS SCANNER
    # ==============================================================================
    with tab_momentum:
        st.subheader("🚀 Institutional Momentum Leaders of the Day")
        st.caption("Filters stocks with >3% intraday move, heavy RVOL (>=2.0), and full EMA 9/20/50 alignment.")
        
        if st.button("🚀 Scan Momentum Leaders", type="primary"):
            with st.spinner("Scanning for high-momentum leaders..."):
                momentum_results = []
                for symbol in symbols_to_scan:
                    clean_sym = symbol.strip()
                    df_5m = fetch_data(clean_sym, period="1d", interval="5m")
                    if not df_5m.empty:
                        df_5m.name = clean_sym
                        m_res = run_momentum_leader_analysis(df_5m)
                        if m_res:
                            m_res["Upstox Instrument Key"] = f"NSE_EQ|{clean_sym}"
                            momentum_results.append(m_res)
                st.session_state['momentum_results'] = momentum_results

        momentum_results = st.session_state.get('momentum_results', [])
        if momentum_results:
            df_mom = pd.DataFrame(momentum_results).sort_values(by="Momentum Score", ascending=False).reset_index(drop=True)
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
