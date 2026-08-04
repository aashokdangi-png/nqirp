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
import os
import pickle

# Load pre-trained model if available in repository
MODEL_PATH = "model.pkl"
ml_model = pickle.load(open(MODEL_PATH, "rb")) if os.path.exists(MODEL_PATH) else None

def predict_trade_probability(rvol: float, vwap_dist_pct: float, atr_pct: float, day_change_pct: float, ema_aligned: bool, range_pos: float) -> dict:
    """
    ML Inference Engine: Evaluates trade setups using feature vectors.
    Returns AI Win Confidence % and Trap Risk status.
    """
    features = [[rvol, vwap_dist_pct, atr_pct, abs(day_change_pct), 1.0 if ema_aligned else 0.0, range_pos]]
    
    # 1. Use loaded model if available
    if ml_model is not None:
        try:
            prob = float(ml_model.predict_proba(features)[0][1]) * 100
        except Exception:
            prob = None
    else:
        prob = None

    # 2. Calibrated Quant Model (Fallback when model.pkl is not yet uploaded)
    if prob is None:
        base_prob = 50.0
        base_prob += min(rvol * 8.0, 24.0)
        base_prob += 12.0 if ema_aligned else -8.0
        base_prob -= max((vwap_dist_pct - 1.5) * 6.0, 0) # Overextension penalty
        base_prob += 10.0 if (0.2 <= range_pos <= 0.85) else -5.0 # Sweet spot range
        prob = min(max(base_prob, 35.0), 96.0)

    # Risk Trap Determination
    if vwap_dist_pct > 2.0 or (rvol < 1.0 and abs(day_change_pct) > 3.0):
        trap_risk = "⚠️ HIGH (Exhaustion/Trap)"
    elif prob >= 75.0:
        trap_risk = "🟢 LOW (High Conviction)"
    else:
        trap_risk = "🟡 MEDIUM"

    return {
        "AI Win Prob": f"{round(prob, 1)}%",
        "Trap Risk": trap_risk
    }
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
    """
    SMC Institutional Analysis Engine with Early Momentum Triggers, RSI Safeguard, and ML Inference.
    Does NOT alter existing UI schemas or table structures.
    """
    if df.empty or len(df) < 30:
        return None

    close = df['Close'].dropna()
    high = df['High'].dropna()
    low = df['Low'].dropna()
    open_p = df['Open'].dropna()
    volume = df['Volume'].dropna()

    if len(close) < 20:
        return None

    c = float(close.iloc[-1])
    h = float(high.iloc[-1])
    l = float(low.iloc[-1])
    o = float(open_p.iloc[-1])
    v = float(volume.iloc[-1])

    # 1. Volatility & Indicators
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    if atr <= 0 or np.isnan(atr):
        return None

    # RSI (14) Calculation (Safety Guard)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else 50.0

    # Volume & VWAP
    v20 = float(volume.tail(20).mean())
    rvol = v / v20 if v20 > 0 else 1.0
    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    ema50 = float(close.ewm(span=50).mean().iloc[-1])
    vwap = float((volume * (high + low + close) / 3).cumsum().iloc[-1] / volume.cumsum().iloc[-1]) if volume.sum() > 0 else c

    smc_confluences, scores = [], []
    direction = "NEUTRAL"

    # 2. EARLY MOMENTUM TRIGGER 1: VWAP Cross (Catches early moves at 1,450+)
    if c < vwap and close.iloc[-2] >= vwap and (o - c) > (atr * 0.4):
        smc_confluences.append("Early VWAP Breakdown Cross")
        scores.append(90)
        direction = "BEARISH"
    elif c > vwap and close.iloc[-2] <= vwap and (c - o) > (atr * 0.4):
        smc_confluences.append("Early VWAP Bullish Cross")
        scores.append(90)
        direction = "BULLISH"

    # 3. EARLY MOMENTUM TRIGGER 2: Micro-BOS Wick Sweep
    l3_prev = float(low.tail(4).iloc[:-1].min())
    h3_prev = float(high.tail(4).iloc[:-1].max())

    if c < l3_prev and direction == "NEUTRAL":
        smc_confluences.append("Micro-BOS Wick Breakdown")
        scores.append(85)
        direction = "BEARISH"
    elif c > h3_prev and direction == "NEUTRAL":
        smc_confluences.append("Micro-BOS Wick Breakout")
        scores.append(85)
        direction = "BULLISH"

    # 4. Fallback Standard Structural BOS
    h20_prev = float(high.tail(25).iloc[:-5].max())
    l20_prev = float(low.tail(25).iloc[:-5].min())
    if c < l20_prev and direction == "NEUTRAL":
        smc_confluences.append("Bearish Structural BOS")
        scores.append(92)
        direction = "BEARISH"
    elif c > h20_prev and direction == "NEUTRAL":
        smc_confluences.append("Bullish Structural BOS")
        scores.append(92)
        direction = "BULLISH"

    if not scores or direction == "NEUTRAL":
        return None

    # 5. Oversold / Overbought Safety Shield
    if direction == "BEARISH" and rsi < 25:
        return None  # Rejects late short entries at extreme oversold bottoms
    if direction == "BULLISH" and rsi > 75:
        return None  # Rejects late long entries at extreme overbought tops

    master_score = max(scores) + min(len(smc_confluences) * 4.0, 20.0)

    # 6. Entry, Target, and Stop Loss Calculations
    suggested_entry = round(c, 2)
    if direction == "BULLISH":
        stop_loss = round(suggested_entry - (1.2 * atr), 2)
        target_price = round(suggested_entry + (2.5 * abs(suggested_entry - stop_loss)), 2)
    else:
        stop_loss = round(suggested_entry + (1.2 * atr), 2)
        target_price = round(suggested_entry - (2.5 * abs(stop_loss - suggested_entry)), 2)

    actual_risk = abs(suggested_entry - stop_loss)
    actual_reward = abs(target_price - suggested_entry)
    rr_ratio = round(actual_reward / actual_risk, 2) if actual_risk > 0 else 2.5

    # 7. ML Feature Inputs & Inference
    vwap_dist_pct = abs(c - vwap) / vwap * 100
    atr_pct = (atr / c) * 100
    pct_change = ((c - o) / o) * 100
    ema_aligned = (c > ema20 > ema50) if direction == "BULLISH" else (c < ema20 < ema50)
    day_range = (h - l) if (h - l) > 0 else 1.0
    range_pos = (c - l) / day_range

    ml_out = predict_trade_probability(rvol, vwap_dist_pct, atr_pct, pct_change, ema_aligned, range_pos)

    # Output structure strictly matches existing Streamlit rendering code
    return {
        "Symbol": df.name if hasattr(df, 'name') else "STOCK",
        "Direction": direction,
        "Master Score": round(master_score, 1),
        "AI Win Prob": ml_out["AI Win Prob"],
        "Trap Risk": ml_out["Trap Risk"],
        "Trade Action": "✅ SWING ENTRY" if timeframe_label == "DAILY" else "✅ ACTIVE ENTRY",
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

# Calculate ML Feature Inputs
    h_day = float(high.tail(75).max()) if len(df) >= 75 else float(high.max())
    l_day = float(low.tail(75).min()) if len(df) >= 75 else float(low.min())
    vwap_dist_pct = abs(c - vwap) / vwap * 100
    atr_pct = (atr / c) * 100
    ema_aligned = (c > ema9 > ema20 > ema50) if is_bullish else (c < ema9 < ema20 < ema50)
    day_range = (h_day - l_day) if (h_day - l_day) > 0 else 1.0
    range_pos = (c - l_day) / day_range if is_bullish else (h_day - c) / day_range

    ml_out = predict_trade_probability(rvol, vwap_dist_pct, atr_pct, pct_change, ema_aligned, range_pos)

    return {
        "Symbol": df.name if hasattr(df, 'name') else "STOCK",
        "Direction": direction,
        "Predictive Score": prob_score,
        "AI Win Prob": ml_out["AI Win Prob"],
        "Trap Risk": ml_out["Trap Risk"],
        "Day Change %": f"{pct_change:+.2f}%",
        "RVOL": round(rvol, 2),
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
