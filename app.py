import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import json

# Set page configuration
st.set_page_config(
    page_title="NQIRP Institutional Quant Engine",
    page_icon="⚡",
    layout="wide"
)

# ==============================================================================
# CORE DATA & HELPER FUNCTIONS
# ==============================================================================
@st.cache_data(ttl=60)
def fetch_data(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """Fetches historical market data via yfinance with multi-index cleanup."""
    try:
        ticker = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def predict_trade_probability(rvol, vwap_dist, atr_pct, day_change, is_bullish, sentiment_score=0.5):
    """Calculates AI win probability and trap risk metrics."""
    prob = 60.0 + min(rvol * 5.0, 15.0) - min(vwap_dist * 4.0, 15.0) + min(abs(day_change) * 2.0, 10.0)
    win_prob = round(min(max(prob, 35.0), 92.0), 1)
    trap_risk = "HIGH" if vwap_dist > 1.8 or rvol > 3.2 else ("MEDIUM" if vwap_dist > 1.0 else "LOW")
    return {"AI Win Prob": f"{win_prob}%", "Trap Risk": trap_risk}

def run_smc_analysis(df: pd.DataFrame, timeframe_label: str = "INTRADAY") -> dict:
    """Core Smart Money Concepts (SMC) structure scan engine."""
    if df.empty or len(df) < 30:
        return None
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    c_live = float(close.iloc[-1])
    
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    if atr <= 0 or np.isnan(atr):
        return None

    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    v20 = float(volume.tail(20).mean())
    rvol = float(volume.iloc[-1] / v20) if v20 > 0 else 1.0

    today_date = close.index[-1].date() if hasattr(close.index[-1], 'date') else None
    if today_date:
        today_df = df[df.index.date == today_date]
        vwap = float((today_df['Volume'] * (today_df['High'] + today_df['Low'] + today_df['Close']) / 3).sum() / today_df['Volume'].sum()) if not today_df.empty else c_live
    else:
        vwap = float((volume * (high + low + close) / 3).cumsum().iloc[-1] / volume.cumsum().iloc[-1])

    is_bullish = c_live > vwap and c_live > ema20
    is_bearish = c_live < vwap and c_live < ema20

    if not (is_bullish or is_bearish):
        return None

    direction = "BULLISH" if is_bullish else "BEARISH"
    sl = round(c_live - (1.2 * atr), 2) if is_bullish else round(c_live + (1.2 * atr), 2)
    tp = round(c_live + (2.4 * atr), 2) if is_bullish else round(c_live - (2.4 * atr), 2)

    return {
        "Symbol": getattr(df, 'name', "STOCK"),
        "Timeframe": timeframe_label,
        "Direction": direction,
        "Master Score": round(70.0 + min(rvol * 5.0, 20.0), 1),
        "Suggested Entry": round(c_live, 2),
        "Stop Loss": sl,
        "Target Price": tp,
        "SMC Signals": "VWAP Cross | Micro-BOS",
        "AI Win Prob": f"{round(65.0 + min(rvol * 4.0, 20.0), 1)}%",
        "Trade Action": "ACTIVE ENTRY"
    }

# ==============================================================================
# INSTITUTIONAL MOMENTUM SCANNER ENGINE
# ==============================================================================
def run_momentum_leader_analysis(df: pd.DataFrame):
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

    ema20 = float(close.ewm(span=20).mean().iloc[-2])

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    if atr <= 0 or np.isnan(atr):
        return None

    today_date = close.index[-1].date() if hasattr(close.index[-1], 'date') else None
    if today_date:
        today_df = df[df.index.date == today_date]
        vwap_anchor = float((today_df['Volume'] * (today_df['High'] + today_df['Low'] + today_df['Close']) / 3).sum() / today_df['Volume'].sum()) if not today_df.empty else c_closed
        today_open = float(today_df['Open'].iloc[0]) if not today_df.empty else float(open_p.iloc[0])
    else:
        vwap_anchor = float((volume * (high + low + close) / 3).cumsum().iloc[-2] / volume.cumsum().iloc[-2])
        today_open = float(open_p.iloc[0])

    v20 = float(volume.tail(20).mean())
    rvol = v_closed / v20 if v20 > 0 else 1.0

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

    day_change_pct = round(((c_live - today_open) / today_open) * 100, 2)
    predictive_score = 50.0 + min(rvol * 12.0, 25.0) + min(abs(day_change_pct) * 10.0, 25.0)

    if dist_from_trigger_pct < 0.1:
        trade_status = "🎯 AT BREAKOUT TRIGGER"
    elif 0.1 <= dist_from_trigger_pct <= 1.2:
        trade_status = f"🚀 ACTIVE (+{round(dist_from_trigger_pct, 2)}% from trigger)"
    else:
        trade_status = f"⚠️ OVEREXTENDED (+{round(dist_from_trigger_pct, 2)}% moved)"

    # Cap display threshold to 4.0% rather than 1.2% so momentum leaders aren't deleted
    if dist_from_trigger_pct > 4.0:
        return None

    actual_risk = abs(suggested_entry - stop_loss)
    actual_reward = abs(target_price - suggested_entry)
    rr_ratio = round(actual_reward / actual_risk, 2) if actual_risk > 0 else 2.5

    ml_out = predict_trade_probability(rvol, abs(c_live - vwap_anchor)/vwap_anchor*100, (atr/c_live)*100, day_change_pct, True, 0.5)

    return {
        "Symbol": getattr(df, 'name', "STOCK"),
        "Direction": direction,
        "Predictive Score": round(predictive_score, 1),
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
# META-CONTRARIAN & CROWD EXHAUSTION ENGINE
# ==============================================================================
def run_meta_contrarian_analysis(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 35:
        return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    c_live = float(close.iloc[-1])
    c_closed = float(close.iloc[-2])
    v_closed = float(volume.iloc[-2])

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
        "Symbol": getattr(df, 'name', "STOCK"),
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
# UNIFIED MASTER CONFLUENCE ENGINE
# ==============================================================================
def run_unified_master_scan(symbols: list) -> pd.DataFrame:
    """
    STRICT TRIPLE ENGINE AGREEMENT ENGINE
    Filters watchlist strictly for 'GEMS' where SMC, Momentum Leaders, 
    and Meta-Contrarian engines ALL trigger and agree on direction.
    """
    master_rows = []
    
    for symbol in symbols:
        df_data = fetch_data(symbol, period="5d", interval="5m")
        if df_data.empty or len(df_data) < 35:
            continue
        df_data.name = symbol

        # Run all three engines in parallel
        smc = run_smc_analysis(df_data, timeframe_label="INTRADAY")
        mom = run_momentum_leader_analysis(df_data)
        mc  = run_meta_contrarian_analysis(df_data)

        # RULE 1: TRIPLE ENGINE MANDATE (All 3 models must trigger)
        if not (smc and mom and mc):
            continue

        # RULE 2: ZERO TRAP TOLERANCE (Exclude crowded/extended signals)
        if "SKIP" in mc.get("Actionable Advice", "") or "TRAP" in mc.get("Crowd Status", ""):
            continue

        # RULE 3: STRICT DIRECTIONAL ALIGNMENT
        smc_dir = smc["Direction"].upper()
        mom_dir = "BULLISH" if "BULLISH" in mom["Direction"].upper() else "BEARISH"
        mc_dir  = mc["Direction"].upper()

        if not (smc_dir == mom_dir == mc_dir):
            continue  # Rejects split-direction signals

        # Determine Gem Quality Grade
        mc_advice = mc.get("Actionable Advice", "")
        if "HIGH CONVICTION" in mc_advice:
            grade = "💎 TRIPLE ENGINE A+ GEM"
        else:
            grade = "💎 TRIPLE ENGINE GEM"

        master_rows.append({
            "Symbol": symbol,
            "Grade": grade,
            "Direction": smc_dir,
            "SMC Entry": smc["Suggested Entry"],
            "Stop Loss": smc["Stop Loss"],
            "Target Price": smc["Target Price"],
            "Breakout Dist": mom["Breakout Distance"],
            "Contrarian Status": mc["Crowd Status"],
            "AI Win Prob": smc["AI Win Prob"],
            "Action": "🔥 HIGH CONVICTION ENTRY" if "A+" in grade else "✅ ACTIVE CONFLUENCE ENTRY"
        })

    df_res = pd.DataFrame(master_rows)
    if not df_res.empty and "Grade" in df_res.columns:
        grade_order = {"💎 TRIPLE ENGINE A+ GEM": 0, "💎 TRIPLE ENGINE GEM": 1}
        df_res["sort_key"] = df_res["Grade"].map(grade_order)
        df_res = df_res.sort_values(by="sort_key").drop(columns=["sort_key"])
    
    return df_res
# ==============================================================================
# QUANTITATIVE BACKTESTING ENGINE (FIX #1: SMC Logic, FIX #3: Limits, FIX #4: Open Trades)
# ==============================================================================
def run_quant_backtest(tickers: list, period: str = "60d", interval: str = "15m", risk_reward: float = 2.0):
    """
    Lookahead-free bar-by-bar backtesting aligned strictly with Institutional SMC Scanner logic.
    """
    all_trades = []

    for symbol in tickers:
        clean_sym = symbol.strip()
        df = fetch_data(clean_sym, period=period, interval=interval)
        
        if df.empty or len(df) < 50:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close']
        high = df['High']
        low = df['Low']

        in_trade = False
        current_trade = None

        for i in range(40, len(df)):
            curr_bar_time = df.index[i]
            h_price = float(high.iloc[i])
            l_price = float(low.iloc[i])
            c_price = float(close.iloc[i])

            # 1. Active Trade Management
            if in_trade and current_trade is not None:
                if current_trade["Direction"] == "BULLISH":
                    if h_price >= current_trade["Target Price"]:
                        current_trade["Exit Price"] = current_trade["Target Price"]
                        current_trade["Exit Time"] = curr_bar_time
                        current_trade["Result"] = "WIN 🎯"
                        current_trade["PnL %"] = round(((current_trade["Exit Price"] - current_trade["Entry Price"]) / current_trade["Entry Price"]) * 100, 2)
                        all_trades.append(current_trade)
                        in_trade, current_trade = False, None
                        continue
                    elif l_price <= current_trade["Stop Loss"]:
                        current_trade["Exit Price"] = current_trade["Stop Loss"]
                        current_trade["Exit Time"] = curr_bar_time
                        current_trade["Result"] = "LOSS 🛑"
                        current_trade["PnL %"] = round(((current_trade["Exit Price"] - current_trade["Entry Price"]) / current_trade["Entry Price"]) * 100, 2)
                        all_trades.append(current_trade)
                        in_trade, current_trade = False, None
                        continue

                elif current_trade["Direction"] == "BEARISH":
                    if l_price <= current_trade["Target Price"]:
                        current_trade["Exit Price"] = current_trade["Target Price"]
                        current_trade["Exit Time"] = curr_bar_time
                        current_trade["Result"] = "WIN 🎯"
                        current_trade["PnL %"] = round(((current_trade["Entry Price"] - current_trade["Exit Price"]) / current_trade["Entry Price"]) * 100, 2)
                        all_trades.append(current_trade)
                        in_trade, current_trade = False, None
                        continue
                    elif h_price >= current_trade["Stop Loss"]:
                        current_trade["Exit Price"] = current_trade["Stop Loss"]
                        current_trade["Exit Time"] = curr_bar_time
                        current_trade["Result"] = "LOSS 🛑"
                        current_trade["PnL %"] = round(((current_trade["Entry Price"] - current_trade["Exit Price"]) / current_trade["Entry Price"]) * 100, 2)
                        all_trades.append(current_trade)
                        in_trade, current_trade = False, None
                        continue

            # 2. Lookahead-Free SMC Signal Generation (Fix #1)
            if not in_trade and i < len(df) - 1:
                sub_df = df.iloc[:i+1].copy()
                sub_df.name = clean_sym
                
                smc_res = run_smc_analysis(sub_df, timeframe_label="INTRADAY")
                if smc_res:
                    entry = smc_res["Suggested Entry"]
                    sl = smc_res["Stop Loss"]
                    
                    # Adjust TP to user selected R/R ratio
                    risk = abs(entry - sl)
                    tp = round(entry + (risk_reward * risk), 2) if smc_res["Direction"] == "BULLISH" else round(entry - (risk_reward * risk), 2)

                    in_trade = True
                    current_trade = {
                        "Symbol": clean_sym,
                        "Direction": smc_res["Direction"],
                        "Entry Time": curr_bar_time,
                        "Entry Price": entry,
                        "Stop Loss": sl,
                        "Target Price": tp,
                        "Exit Time": "ACTIVE",
                        "Exit Price": "OPEN",
                        "Result": "OPEN ⏳",
                        "PnL %": 0.0,
                        "Signals": smc_res["SMC Signals"]
                    }

        # 3. Log Open Trades on Final Historical Bar (Fix #4)
        if in_trade and current_trade is not None:
            last_price = float(close.iloc[-1])
            current_trade["Exit Time"] = df.index[-1]
            current_trade["Exit Price"] = round(last_price, 2)
            current_trade["Result"] = "OPEN ⏳"
            pnl = ((last_price - current_trade["Entry Price"]) / current_trade["Entry Price"]) * 100 if current_trade["Direction"] == "BULLISH" else ((current_trade["Entry Price"] - last_price) / current_trade["Entry Price"]) * 100
            current_trade["PnL %"] = round(pnl, 2)
            all_trades.append(current_trade)

    return pd.DataFrame(all_trades)

# ==============================================================================
# STREAMLIT UI & MODULE ROUTING
# ==============================================================================
st.sidebar.title("NQIRP Navigation")
st.sidebar.subheader("🔍 Stock Universe Selector")

universe_choice = st.sidebar.selectbox(
    "Select Scanning Universe",
    ["Default Watchlist (7 Stocks)", "NIFTY 50 Expanded (50 Stocks)", "Custom Tickers"]
)

if universe_choice == "Default Watchlist (7 Stocks)":
    symbols_to_scan = ["REDINGTON", "FIRSTSOURCE", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
elif universe_choice == "NIFTY 50 Expanded (50 Stocks)":
    symbols_to_scan = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "LT", "AXISBANK", "SBIN", "BHARTIARTL",
        "ITC", "ASIANPAINT", "HCLTECH", "MARUTI", "SUNPHARMA", "TATAMOTORS", "TATASTEEL", "BAJFINANCE", "BAJAJFINSV",
        "WIPRO", "ULTRACEMCO", "TITAN", "POWERGRID", "NTPC", "ONGC", "COALINDIA", "ADANIENT", "ADANIPORTS", "GRASIM",
        "HINDALCO", "JSWSTEEL", "TECHM", "HEROMOTOCO", "EICHERMOT", "BPCL", "CIPLA", "DRREDDY", "DIVISLAB", "BRITANNIA",
        "TRENT", "BEL", "HAL", "VBL", "ZOMATO", "TATAELXSI", "FIRSTSOURCE", "REDINGTON", "PIDILITIND", "CHOLAFIN", "INDUSINDBK"
    ]
else:
    custom_input = st.sidebar.text_input("Enter Tickers (comma-separated)", "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK")
    symbols_to_scan = [s.strip().upper() for s in custom_input.split(",") if s.strip()]

page = st.sidebar.radio("Select Module", [
    "⚡ SMC Institutional Scanner",
    "🧪 Backtesting Engine",
    "👁️ Vision AI Chart Pattern Scanner"
])

# ------------------------------------------------------------------------------
# PAGE 1: SMC INSTITUTIONAL SCANNER
# ------------------------------------------------------------------------------
if page == "⚡ SMC Institutional Scanner":
    st.title("⚡ SMC Institutional Scanner Engine")
    st.markdown(f"Real-time multi-timeframe quantitative scanning across **{len(symbols_to_scan)} stocks** for SMC confluences, FVG, BOS, and Momentum Leaders.")

    tab_master, tab_intraday, tab_swing, tab_momentum, tab_contrarian = st.tabs([
        "🌟 Master Confluence", "⚡ Intraday SMC", "📈 Swing Signals", "🚀 Momentum Leaders", "🧠 Meta-Contrarian Engine"
    ])

    with tab_master:
        st.subheader("🌟 Unified Master Confluence Dashboard")
        st.caption("Single-click live scan across SMC, Momentum, and Meta-Contrarian engines to isolate A+ confluence trades.")
        
        if st.button("🌟 Run Unified Master Scan", type="primary", key="btn_master_scan"):
            with st.spinner(f"Running multi-engine confluence audit across {len(symbols_to_scan)} stocks..."):
                master_df = run_unified_master_scan(symbols_to_scan)
                st.session_state['master_results'] = master_df

        res_master = st.session_state.get('master_results', pd.DataFrame())
        if not res_master.empty:
            st.dataframe(res_master.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Click 'Run Unified Master Scan' above during market hours to evaluate all confluences in a single table.")

    with tab_intraday:
        st.subheader("⚡ Intraday SMC Scanner Engine")
        if st.button("⚡ Run Intraday SMC Scan", type="primary", key="btn_intraday_scan"):
            with st.spinner("Scanning intraday SMC confluences..."):
                intraday_results = []
                for symbol in symbols_to_scan:
                    df_data = fetch_data(symbol, period="5d", interval="5m")
                    if not df_data.empty and len(df_data) >= 30:
                        df_data.name = symbol
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

    with tab_swing:
        st.subheader("📈 Swing Signals Engine")
        if st.button("📈 Run Swing Scan", type="primary", key="btn_swing_scan"):
            with st.spinner("Scanning daily swing SMC setups..."):
                swing_results = []
                for symbol in symbols_to_scan:
                    df_data = fetch_data(symbol, period="1mo", interval="1d")
                    if not df_data.empty and len(df_data) >= 30:
                        df_data.name = symbol
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
            st.info("Click 'Run Swing Scan' above to evaluate daily charts.")

    with tab_momentum:
        st.subheader("🚀 Momentum Leaders Engine")
        if st.button("🚀 Run Momentum Scan", type="primary", key="btn_momentum_scan"):
            with st.spinner("Scanning momentum leaders..."):
                momentum_results = []
                for symbol in symbols_to_scan:
                    df_data = fetch_data(symbol, period="5d", interval="5m")
                    if not df_data.empty and len(df_data) >= 35:
                        df_data.name = symbol
                        res = run_momentum_leader_analysis(df_data)
                        if res:
                            momentum_results.append(res)
                st.session_state['momentum_results'] = momentum_results

        res_momentum = st.session_state.get('momentum_results', [])
        if res_momentum:
            df_mom = pd.DataFrame(res_momentum)
            if "Predictive Score" in df_mom.columns:
                df_mom = df_mom.sort_values(by="Predictive Score", ascending=False)
            st.dataframe(df_mom.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Click 'Run Momentum Scan' to rank momentum breakout leaders.")

    with tab_contrarian:
        st.subheader("🧠 Meta-Contrarian Engine")
        if st.button("🧠 Run Meta-Contrarian Scan", type="primary", key="btn_contrarian_scan"):
            with st.spinner("Analyzing crowd exhaustion and overextension..."):
                contrarian_results = []
                for symbol in symbols_to_scan:
                    df_data = fetch_data(symbol, period="5d", interval="5m")
                    if not df_data.empty and len(df_data) >= 35:
                        df_data.name = symbol
                        res = run_meta_contrarian_analysis(df_data)
                        if res:
                            contrarian_results.append(res)
                st.session_state['contrarian_results'] = contrarian_results

        res_contrarian = st.session_state.get('contrarian_results', [])
        if res_contrarian:
            df_mc = pd.DataFrame(res_contrarian)
            if "Final Re-Ranked Score" in df_mc.columns:
                df_mc = df_mc.sort_values(by="Final Re-Ranked Score", ascending=False)
            st.dataframe(df_mc.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Click 'Run Meta-Contrarian Scan' to check crowd crowding risks.")

# ------------------------------------------------------------------------------
# PAGE 2: BACKTESTING ENGINE
# ------------------------------------------------------------------------------
elif page == "🧪 Backtesting Engine":
    st.title("🧪 Quantitative Backtesting Engine")
    st.markdown("Bar-by-bar lookahead-free simulation directly testing the **Institutional SMC Strategy Engine**.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        interval = st.selectbox("Interval", ["5m", "15m", "1h", "1d"], index=1)
    
    # Intraday Limit Handling (Fix #3)
    max_period = "60d" if interval in ["5m", "15m"] else "1y"
    with col2:
        period = st.selectbox("Historical Lookback", ["7d", "30d", "60d", "1y", "2y"], index=2 if max_period == "60d" else 3)
        if interval in ["5m", "15m"] and period in ["1y", "2y"]:
            st.warning("⚠️ Intraday intervals auto-capped to 60d due to YFinance limits.")
            period = "60d"

    with col3:
        rr_input = st.number_input("Target Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    with col4:
        st.write("")
        run_bt = st.button("🧪 Run Backtest", type="primary")

    if run_bt:
        with st.spinner(f"Simulating SMC strategy across {len(symbols_to_scan)} stocks over {period}..."):
            bt_df = run_quant_backtest(symbols_to_scan, period=period, interval=interval, risk_reward=rr_input)
            st.session_state['bt_results'] = bt_df

    bt_results = st.session_state.get('bt_results', pd.DataFrame())
    if not bt_results.empty:
        total_trades = len(bt_results)
        wins = len(bt_results[bt_results['Result'] == 'WIN 🎯'])
        losses = len(bt_results[bt_results['Result'] == 'LOSS 🛑'])
        open_trades = len(bt_results[bt_results['Result'] == 'OPEN ⏳'])
        
        closed_trades = wins + losses
        win_rate = round((wins / closed_trades) * 100, 1) if closed_trades > 0 else 0.0
        total_pnl = round(bt_results['PnL %'].sum(), 2)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Trades", total_trades)
        m2.metric("Win Rate", f"{win_rate}%")
        m3.metric("Wins / Losses", f"{wins} / {losses}")
        m4.metric("Active / Open", open_trades)
        m5.metric("Net Strategy PnL", f"{total_pnl:+}%")

        st.markdown("### 📋 Trade Execution Log")
        st.dataframe(bt_results, use_container_width=True)
    else:
        st.info("Configure parameters and click 'Run Backtest' to execute simulation.")
        
