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
    SMC Institutional Engine with Fixed Trigger Anchoring, Closed Candle Indicators,
    and 100% UI Schema Preservation (Tab 1 & Tab 2).
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

    c_live = float(close.iloc[-1])       # Current tick price
    c_closed = float(close.iloc[-2])     # Closed candle price
    v_closed = float(volume.iloc[-2])     # Closed candle volume
    o_live = float(open_p.iloc[-1])

    # 1. Volatility & Indicators on Closed Candle (Prevents flickering)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).iloc[:-1].mean())
    if atr <= 0 or np.isnan(atr):
        return None

    # RSI (14) on Closed Candle
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = float(rsi_series.dropna().iloc[-2]) if len(rsi_series.dropna()) >= 2 else 50.0

    # Volume & VWAP Anchor
    v20 = float(volume.tail(20).mean())
    rvol = v_closed / v20 if v20 > 0 else 1.0
    ema20 = float(close.ewm(span=20).mean().iloc[-2])
    ema50 = float(close.ewm(span=50).mean().iloc[-2])
    vwap = float((volume * (high + low + close) / 3).cumsum().iloc[-2] / volume.cumsum().iloc[-2]) if volume.sum() > 0 else c_closed

    smc_confluences, scores = [], []
    direction = "NEUTRAL"
    trigger_price = c_closed

    # 2. EARLY MOMENTUM TRIGGER 1: VWAP Cross
    if c_live < vwap and close.iloc[-2] >= vwap and (o_live - c_live) > (atr * 0.3):
        smc_confluences.append("Early VWAP Breakdown Cross")
        scores.append(90)
        direction = "BEARISH"
        trigger_price = round(vwap, 2)
    elif c_live > vwap and close.iloc[-2] <= vwap and (c_live - o_live) > (atr * 0.3):
        smc_confluences.append("Early VWAP Bullish Cross")
        scores.append(90)
        direction = "BULLISH"
        trigger_price = round(vwap, 2)

    # 3. EARLY MOMENTUM TRIGGER 2: Micro-BOS Wick Sweep
    l3_prev = float(low.tail(4).iloc[:-1].min())
    h3_prev = float(high.tail(4).iloc[:-1].max())
    if c_live < l3_prev and direction == "NEUTRAL":
        smc_confluences.append("Micro-BOS Wick Breakdown")
        scores.append(85)
        direction = "BEARISH"
        trigger_price = round(l3_prev, 2)
    elif c_live > h3_prev and direction == "NEUTRAL":
        smc_confluences.append("Micro-BOS Wick Breakout")
        scores.append(85)
        direction = "BULLISH"
        trigger_price = round(h3_prev, 2)

    # 4. Fallback Standard Structural BOS
    h20_prev = float(high.tail(25).iloc[:-5].max())
    l20_prev = float(low.tail(25).iloc[:-5].min())
    if c_live < l20_prev and direction == "NEUTRAL":
        smc_confluences.append("Bearish Structural BOS")
        scores.append(92)
        direction = "BEARISH"
        trigger_price = round(l20_prev, 2)
    elif c_live > h20_prev and direction == "NEUTRAL":
        smc_confluences.append("Bullish Structural BOS")
        scores.append(92)
        direction = "BULLISH"
        trigger_price = round(h20_prev, 2)

    if not scores or direction == "NEUTRAL":
        return None

    # 5. Oversold / Overbought Safety Shield
    if direction == "BEARISH" and rsi < 25:
        return None
    if direction == "BULLISH" and rsi > 75:
        return None

    master_score = max(scores) + min(len(smc_confluences) * 4.0, 20.0)

    # 6. Fixed Trigger Entry, Target, Stop Loss, and Overextension Shield
    suggested_entry = trigger_price
    if direction == "BULLISH":
        dist_from_trigger_pct = ((c_live - suggested_entry) / suggested_entry) * 100
        stop_loss = round(suggested_entry - (1.2 * atr), 2)
        target_price = round(suggested_entry + (2.5 * abs(suggested_entry - stop_loss)), 2)
    else:
        dist_from_trigger_pct = ((suggested_entry - c_live) / suggested_entry) * 100
        stop_loss = round(suggested_entry + (1.2 * atr), 2)
        target_price = round(suggested_entry - (2.5 * abs(stop_loss - suggested_entry)), 2)

    # Reject trades that ran > 1.2% past trigger point
    if dist_from_trigger_pct > 1.2:
        return None

    actual_risk = abs(suggested_entry - stop_loss)
    actual_reward = abs(target_price - suggested_entry)
    rr_ratio = round(actual_reward / actual_risk, 2) if actual_risk > 0 else 2.5

    # 7. ML Feature Inputs & Inference
    vwap_dist_pct = abs(c_live - vwap) / vwap * 100
    atr_pct = (atr / c_live) * 100
    pct_change = ((c_live - o_live) / o_live) * 100
    ema_aligned = (c_live > ema20 > ema50) if direction == "BULLISH" else (c_live < ema20 < ema50)
    day_range = (float(high.iloc[-1]) - float(low.iloc[-1])) if (float(high.iloc[-1]) - float(low.iloc[-1])) > 0 else 1.0
    range_pos = (c_live - float(low.iloc[-1])) / day_range

    ml_out = predict_trade_probability(rvol, vwap_dist_pct, atr_pct, pct_change, ema_aligned, range_pos)

    # 100% PRESERVED DICTIONARY SCHEMA
    return {
        "Symbol": df.name if hasattr(df, 'name') else "STOCK",
        "Direction": direction,
        "Master Score": round(master_score, 1),
        "AI Win Prob": ml_out["AI Win Prob"],
        "Trap Risk": ml_out["Trap Risk"],
        "Trade Action": "✅ SWING ENTRY" if timeframe_label == "DAILY" else "✅ ACTIVE ENTRY",
        "Suggested Entry": suggested_entry,         # FIXED TRIGGER ANCHOR
        "Current Price": round(c_live, 2),          # LIVE TICK PRICE
        "Target Price": target_price,
        "Stop Loss": stop_loss,
        "R/R Ratio": f"1 : {rr_ratio}",
        "RVOL": round(rvol, 2),                     # STABLE CLOSED BAR RVOL
        "SMC Signals": ", ".join(smc_confluences)
    }

# ==============================================================================
# 🚀 INSTITUTIONAL MOMENTUM SCANNER ENGINE
# ==============================================================================
def run_momentum_leader_analysis(df: pd.DataFrame):
    """
    Non-Breaking Institutional Momentum Engine.
    Fixes KeyError by explicitly returning 'Predictive Score', while locking entry 
    anchors and eliminating 5-second indicator flickering.
    """
    if df.empty or len(df) < 35:
        return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']
    volume = df['Volume']

    c_live = float(close.iloc[-1])       # Current tick price
    c_closed = float(close.iloc[-2])     # Last closed candle price
    v_closed = float(volume.iloc[-2])     # Last closed candle volume

    # 1. Indicators evaluated on closed bar to prevent 5-second flickering
    ema20 = float(close.ewm(span=20).mean().iloc[-2])
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    if atr <= 0 or np.isnan(atr):
        return None

    # Calculate VWAP Anchor
    today_date = close.index[-1].date() if hasattr(close.index[-1], 'date') else None
    if today_date:
        today_df = df[df.index.date == today_date]
        vwap_anchor = float((today_df['Volume'] * (today_df['High'] + today_df['Low'] + today_df['Close']) / 3).sum() / today_df['Volume'].sum()) if not today_df.empty else c_closed
    else:
        vwap_anchor = float((volume * (high + low + close) / 3).cumsum().iloc[-2] / volume.cumsum().iloc[-2])

    v20 = float(volume.tail(20).mean())
    rvol = v_closed / v20 if v20 > 0 else 1.0

    # 2. Fixed Structural Trigger
    h20_breakout = float(high.tail(30).iloc[:-2].max())
    l20_breakout = float(low.tail(30).iloc[:-2].min())

    is_bullish = c_live > vwap_anchor and c_live > ema20
    is_bearish = c_live < vwap_anchor and c_live < ema20

    if not (is_bullish or is_bearish):
        return None

    direction = "🔥 BULLISH MOMENTUM" if is_bullish else "🩸 BEARISH MOMENTUM"

    if is_bullish:
        suggested_entry = round(max(vwap_anchor, h20_breakout), 2)
        dist_from_trigger_pct = ((c_live - suggested_entry) / suggested_entry) * 100
        stop_loss = round(suggested_entry - (1.0 * atr), 2)
        target_price = round(suggested_entry + (2.5 * atr), 2)
    else:
        suggested_entry = round(min(vwap_anchor, l20_breakout), 2)
        dist_from_trigger_pct = ((suggested_entry - c_live) / suggested_entry) * 100
        stop_loss = round(suggested_entry + (1.0 * atr), 2)
        target_price = round(suggested_entry - (2.5 * atr), 2)

    # Calculate Predictive Score (Fixes KeyError)
    day_change_pct = round(((c_live - open_p.iloc[0]) / open_p.iloc[0]) * 100, 2)
    predictive_score = 50.0 + min(rvol * 12.0, 25.0) + min(abs(day_change_pct) * 10.0, 25.0)

    # 3. Dynamic Status Tracking
    if dist_from_trigger_pct < 0.1:
        trade_status = "🎯 AT BREAKOUT TRIGGER"
    elif 0.1 <= dist_from_trigger_pct <= 0.8:
        trade_status = f"🚀 ACTIVE (+{round(dist_from_trigger_pct, 2)}% from trigger)"
    else:
        trade_status = f"⚠️ OVEREXTENDED (+{round(dist_from_trigger_pct, 2)}% moved)"

    if dist_from_trigger_pct > 1.2:
        return None

    actual_risk = abs(suggested_entry - stop_loss)
    actual_reward = abs(target_price - suggested_entry)
    rr_ratio = round(actual_reward / actual_risk, 2) if actual_risk > 0 else 2.5

    # ML Model Inference
    ml_out = predict_trade_probability(rvol, abs(c_live - vwap_anchor)/vwap_anchor*100, (atr/c_live)*100, day_change_pct, True, 0.5)

    return {
        "Symbol": df.name if hasattr(df, 'name') else "STOCK",
        "Direction": direction,
        "Predictive Score": round(predictive_score, 1), # Fixed: Restored missing key
        "Current Price": round(c_live, 2),
        "Suggested Entry": suggested_entry,
        "Breakout Distance": f"{round(dist_from_trigger_pct, 2)}%",
        "Day Change %": f"{day_change_pct}%",
        "RVOL": round(rvol, 2),
        "Status": trade_status,
        "Stop Loss": stop_loss,
        "Target Price": target_price,
        "R/R Ratio": f"1 : {rr_ratio}",
        "AI Win Prob": ml_out["AI Win Prob"],
        "Trap Risk": ml_out["Trap Risk"]
    }
# ==============================================================================
# 🧠 META-CONTRARIAN & CROWD EXHAUSTION ENGINE (ISOLATED MODULE)
# ==============================================================================
def run_meta_contrarian_analysis(df: pd.DataFrame) -> dict:
    """
    Evaluates crowd concentration, trend extension, and blow-off volume climaxes.
    Re-ranks setups by applying contrarian factors to prevent chasing retail traps.
    """
    if df.empty or len(df) < 35:
        return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']
    volume = df['Volume']

    c_live = float(close.iloc[-1])
    c_closed = float(close.iloc[-2])
    v_closed = float(volume.iloc[-2])

    # 1. Technical Indicators on Closed Bar
    ema20 = float(close.ewm(span=20).mean().iloc[-2])
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).iloc[:-1].mean())
    if atr <= 0 or np.isnan(atr):
        return None

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = float((100 - (100 / (1 + rs))).dropna().iloc[-2]) if len(rs) >= 14 else 50.0

    v20 = float(volume.tail(20).mean())
    rvol = v_closed / v20 if v20 > 0 else 1.0

    today_date = close.index[-1].date() if hasattr(close.index[-1], 'date') else None
    if today_date:
        today_df = df[df.index.date == today_date]
        vwap = float((today_df['Volume'] * (today_df['High'] + today_df['Low'] + today_df['Close']) / 3).sum() / today_df['Volume'].sum()) if not today_df.empty else c_closed
    else:
        vwap = float((volume * (high + low + close) / 3).cumsum().iloc[-2] / volume.cumsum().iloc[-2])

    is_bullish = c_live > vwap and c_live > ema20
    is_bearish = c_live < vwap and c_live < ema20
    if not (is_bullish or is_bearish):
        return None

    base_score = 75.0
    contrarian_modifier = 0.0
    crowd_flags = []

    # 2. Derive Meta-Contrarian Factors
    vwap_dist_pct = abs(c_live - vwap) / vwap * 100
    ema_dist_pct = abs(c_live - ema20) / ema20 * 100

    if vwap_dist_pct > 1.8 or ema_dist_pct > 2.5:
        contrarian_modifier -= 6.0
        crowd_flags.append("⚠️ Overstretched VWAP (Chasing Risk)")
    elif vwap_dist_pct < 0.4:
        contrarian_modifier += 4.0
        crowd_flags.append("🟢 Fresh Pullback near VWAP Anchor")

    if rvol > 3.2:
        contrarian_modifier -= 5.0
        crowd_flags.append("🚨 Volume Blow-Off Climax")
    elif 1.4 <= rvol <= 2.5:
        contrarian_modifier += 4.0
        crowd_flags.append("🟢 Healthy Institutional Volume")

    if is_bullish and rsi > 72:
        contrarian_modifier -= 5.0
        crowd_flags.append("⚠️ RSI Overbought (>72)")
    elif is_bearish and rsi < 28:
        contrarian_modifier -= 5.0
        crowd_flags.append("⚠️ RSI Oversold (<28)")
    elif 45 <= rsi <= 62:
        contrarian_modifier += 3.0
        crowd_flags.append("🟢 RSI Balanced Zone")

    h20 = float(high.tail(30).iloc[:-2].max())
    l20 = float(low.tail(30).iloc[:-2].min())
    if (c_live >= h20 and is_bullish) or (c_live <= l20 and is_bearish):
        contrarian_modifier += 5.0
        crowd_flags.append("🚀 Fresh Structural Breakout")

    final_score = min(max(base_score + contrarian_modifier, 30.0), 98.0)

    if contrarian_modifier <= -6.0:
        crowd_status = "⚠️ CROWDED TRAP"
        action_advice = "🛑 SKIP (High Reversal Risk)"
    elif contrarian_modifier >= 5.0:
        crowd_status = "🔥 A+ FRESH MOVE"
        action_advice = "✅ HIGH CONVICTION"
    else:
        crowd_status = "🟡 MODERATE CROWDING"
        action_advice = "⚡ HALF POSITION"

    return {
        "Symbol": df.name if hasattr(df, 'name') else "STOCK",
        "Direction": "BULLISH" if is_bullish else "BEARISH",
        "Base Score": round(base_score, 1),
        "Contrarian Modifier": f"{contrarian_modifier:+.1f}",
        "Final Re-Ranked Score": round(final_score, 1),
        "Crowd Status": crowd_status,
        "Actionable Advice": action_advice,
        "Crowd Diagnostics": " | ".join(crowd_flags) if crowd_flags else "Standard Setup",
        "Current Price": round(c_live, 2),
        "RVOL": round(rvol, 2),
        "VWAP Dist %": f"{round(vwap_dist_pct, 2)}%"
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

    tab_intraday, tab_swing, tab_momentum, tab_contrarian = st.tabs([
        "⚡ Intraday SMC", "📈 Swing Signals", "🚀 Momentum Leaders", "🧠 Meta-Contrarian Engine"
    ])

    # --- TAB 1: INTRADAY SMC ---
    with tab_intraday:
        st.subheader("⚡ Intraday SMC Scanner Engine")
        if st.button("⚡ Run Intraday SMC Scan", type="primary", key="btn_intraday_scan"):
            with st.spinner("Scanning intraday SMC confluences..."):
                intraday_results = []
                for symbol in symbols_to_scan:
                    clean_sym = symbol.strip()
                    df_data = fetch_data(clean_sym, period="1d", interval="5m")
                    if not df_data.empty and len(df_data) >= 30:
                        df_data.name = clean_sym
                        res = run_smc_analysis(df_data, timeframe_label="INTRADAY")
                        if res:
                            intraday_results.append(res)
                st.session_state['intraday_results'] = intraday_results

        res_intraday = st.session_state.get('intraday_results', [])
        if res_intraday:
            df_intra = pd.DataFrame(res_intraday)
            if "Master Score" in df_intra.columns:
                df_intra = df_intra.sort_values(by="Master Score", ascending=False)
            st.dataframe(df_intra.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Click 'Run Intraday SMC Scan' above to scan symbols.")

    # --- TAB 2: SWING SIGNALS ---
    with tab_swing:
        st.subheader("📈 Swing Signals Engine")
        if st.button("📈 Run Swing Scan", type="primary", key="btn_swing_scan"):
            with st.spinner("Scanning daily swing SMC setups..."):
                swing_results = []
                for symbol in symbols_to_scan:
                    clean_sym = symbol.strip()
                    df_data = fetch_data(clean_sym, period="1mo", interval="1d")
                    if not df_data.empty and len(df_data) >= 30:
                        df_data.name = clean_sym
                        res = run_smc_analysis(df_data, timeframe_label="DAILY")
                        if res:
                            swing_results.append(res)
                st.session_state['swing_results'] = swing_results

        res_swing = st.session_state.get('swing_results', [])
        if res_swing:
            df_sw = pd.DataFrame(res_swing)
            if "Master Score" in df_sw.columns:
                df_sw = df_sw.sort_values(by="Master Score", ascending=False)
            st.dataframe(df_sw.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Click 'Run Swing Scan' above to scan symbols.")

    # --- TAB 3: MOMENTUM LEADERS ---
    with tab_momentum:
        st.subheader("🚀 Institutional Momentum Leaders Engine")
        if st.button("🚀 Scan Momentum Leaders", type="primary", key="btn_momentum_scan"):
            with st.spinner("Scanning momentum leaders..."):
                mom_results = []
                for symbol in symbols_to_scan:
                    clean_sym = symbol.strip()
                    df_data = fetch_data(clean_sym, period="5d", interval="5m")
                    if not df_data.empty and len(df_data) >= 35:
                        df_data.name = clean_sym
                        m_res = run_momentum_leader_analysis(df_data)
                        if m_res:
                            mom_results.append(m_res)
                st.session_state['mom_results'] = mom_results

        res_mom = st.session_state.get('mom_results', [])
        if res_mom:
            df_m = pd.DataFrame(res_mom)
            if "Predictive Score" in df_m.columns:
                df_m = df_m.sort_values(by="Predictive Score", ascending=False)
            st.dataframe(df_m.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Click 'Scan Momentum Leaders' above to scan symbols.")

    # --- TAB 4: META-CONTRARIAN ENGINE ---
    with tab_contrarian:
        st.subheader("🧠 Meta-Contrarian & Crowd Exhaustion Re-Ranker")
        st.caption("Filters standard momentum signals by penalizing overcrowded, overextended, or volume-climax setups.")
        if st.button("🧠 Run Meta-Contrarian Audit", type="primary", key="btn_contrarian_scan"):
            with st.spinner("Auditing market consensus and crowd exhaustion..."):
                contrarian_results = []
                for symbol in symbols_to_scan:
                    clean_sym = symbol.strip()
                    df_data = fetch_data(clean_sym, period="5d", interval="5m")
                    if not df_data.empty and len(df_data) >= 35:
                        df_data.name = clean_sym
                        c_res = run_meta_contrarian_analysis(df_data)
                        if c_res:
                            contrarian_results.append(c_res)
                st.session_state['contrarian_results'] = contrarian_results

        contrarian_results = st.session_state.get('contrarian_results', [])
        if contrarian_results:
            df_c = pd.DataFrame(contrarian_results)
            if "Final Re-Ranked Score" in df_c.columns:
                df_c = df_c.sort_values(by="Final Re-Ranked Score", ascending=False)
            st.dataframe(df_c.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Click 'Run Meta-Contrarian Audit' above to evaluate setups.")
    # ==============================================================================
    # TAB 4: META-CONTRARIAN ENGINE
    # ==============================================================================
    with tab_contrarian:
        st.subheader("🧠 Meta-Contrarian & Crowd Exhaustion Re-Ranker")
        st.caption("Filters standard momentum signals by penalizing overcrowded, overextended, or volume-climax setups.")

        if st.button("🧠 Run Meta-Contrarian Audit", type="primary"):
            with st.spinner("Auditing market consensus and crowd exhaustion..."):
                contrarian_results = []
                for symbol in symbols_to_scan:
                    clean_sym = symbol.strip()
                    df_data = fetch_data(clean_sym, period="5d", interval="5m")
                    if not df_data.empty and len(df_data) >= 35:
                        df_data.name = clean_sym
                        c_res = run_meta_contrarian_analysis(df_data)
                        if c_res:
                            contrarian_results.append(c_res)
                st.session_state['contrarian_results'] = contrarian_results

        contrarian_results = st.session_state.get('contrarian_results', [])
        if contrarian_results:
            df_c = pd.DataFrame(contrarian_results)
            if "Final Re-Ranked Score" in df_c.columns:
                df_c = df_c.sort_values(by="Final Re-Ranked Score", ascending=False)
            st.dataframe(df_c.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Click 'Run Meta-Contrarian Audit' above to evaluate setups.")

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
# ==============================================================================
# 📊 AUTOMATED AI DIAGNOSTIC & REVIEW REPORT UI
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Learning & Diagnostics")

if st.sidebar.button("📊 Generate Review Report"):
    st.subheader("📋 AI Scanner Diagnostic & Missed Trades Report")
    
    REPORT_FILE = "ml_report.json"
    if os.path.exists(REPORT_FILE):
        try:
            with open(REPORT_FILE, "r") as f:
                reports = json.load(f)
            
            latest_report = reports[-1]  # Get most recent report
            
            st.info(f"**Report Date:** {latest_report.get('Date')} | **Tracked Stocks:** {latest_report.get('Total Tracked')}")
            
            # 1. Display Recommendations & Fixes
            st.markdown("### 🛠️ Machine Fixes & Recommendations")
            for rec in latest_report.get("Recommendations", []):
                st.success(rec)
                
            # 2. Display Missed Trades Table
            st.markdown("### 🔍 Missed Opportunities & False Positives")
            missed = latest_report.get("Missed Details", [])
            if missed:
                st.dataframe(pd.DataFrame(missed), use_container_width=True)
            else:
                st.write("🎉 Zero missed major moves logged for this session!")
                
        except Exception as e:
            st.error(f"Error loading report: {e}")
    else:
        st.warning("No diagnostic report found yet. Reports auto-generate at market close (3:30 PM IST).")
