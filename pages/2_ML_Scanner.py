import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

st.set_page_config(page_title="NQIRP ML Scanner v3.0", page_icon="🤖", layout="wide")

st.title("🤖 AI & ML Strategy Scanner v3.0")
st.markdown("*Institutional-Grade: Zone Lifecycle Tracking, Dynamic SMC Targeting & AI VWAP Integration*")

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_ai_assets():
    model = joblib.load("colab_ai_model.pkl") if os.path.exists("colab_ai_model.pkl") else None
    scaler = joblib.load("colab_scaler.pkl") if os.path.exists("colab_scaler.pkl") else None
    return model, scaler

model, scaler = load_ai_assets()

if model is None:
    st.warning("Model file 'colab_ai_model.pkl' not found. AI probabilities will run in fallback simulation mode.")

EXACT_FEATURES = [
    'RVOL', 
    'ATR_Pct', 
    'RSI', 
    'Liquidity_Sweep_High', 
    'Liquidity_Sweep_Low', 
    'Bullish_FVG', 
    'Bullish_OB', 
    'Pattern_Flag_Breakout', 
    'Market_Sentiment'
]

expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else EXACT_FEATURES

st.sidebar.success(f"✅ AI Engine Configured: {len(expected_features)} Features")
st.sidebar.info("🎯 Dynamic SMC Zone Lifecycle & State Tracking Active.")

# --- 2. INDEX & SECTOR SENTIMENT FETCHING ---
SECTOR_MAP = {
    "Banking": "^NSEBANK",
    "IT": "^CNXIT",
    "Auto": "^CNXAUTO",
    "Energy": "^CNXENERGY",
    "FMCG": "^CNXFMCG",
    "Metal": "^CNXMETAL",
    "Infra": "^CNXINFRA",
    "Financials": "NIFTY_FIN_SERVICE.NS",
    "Telecom": "^NSEI",
    "Capital Goods": "^NSEI",
    "Healthcare": "^CNXPHARMA"
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
def fetch_market_sentiments():
    index_tickers = ["^NSEI", "^NSEMDCP50", "NIFTYSMALL100.NS", "^CNXSC", "^CNXSMLCAP"]
    sector_tickers = list(set(SECTOR_MAP.values()))
    fallback_tickers = [t for lst in SECTOR_CONSTITUENTS.values() for t in lst]
    
    all_tickers = list(set(index_tickers + sector_tickers + fallback_tickers))

    trends, returns = {}, {}
    try:
        data = yf.download(all_tickers, period="5d", interval="1d", progress=False)
        close_df = data["Close"] if "Close" in data else data

        def get_1d_return(ticker_symbol):
            if ticker_symbol in close_df:
                s = close_df[ticker_symbol].dropna()
                if len(s) >= 2:
                    return float((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2])
            return 0.0

        def get_group_avg_return(ticker_list):
            rets = [get_1d_return(t) for t in ticker_list if abs(get_1d_return(t)) > 1e-6]
            return float(np.mean(rets)) if rets else 0.0

        returns["Nifty_1D_Return"] = get_1d_return("^NSEI")
        returns["Midcap_1D_Return"] = get_1d_return("^NSEMDCP50")
        
        sml_ret = 0.0
        for sml_t in ["NIFTYSMALL100.NS", "^CNXSC", "^CNXSMLCAP"]:
            r = get_1d_return(sml_t)
            if abs(r) > 1e-5:
                sml_ret = r
                break
        if abs(sml_ret) < 1e-5:
            sml_ret = get_group_avg_return(SECTOR_CONSTITUENTS["Smallcap"])
        
        returns["Smallcap_1D_Return"] = sml_ret

        trends["^NSEI"] = f"{'+' if returns['Nifty_1D_Return'] >= 0 else ''}{returns['Nifty_1D_Return']*100:.2f}%"
        trends["^NSEMDCP50"] = f"{'+' if returns['Midcap_1D_Return'] >= 0 else ''}{returns['Midcap_1D_Return']*100:.2f}%"
        trends["Smallcap"] = f"{'+' if sml_ret >= 0 else ''}{sml_ret*100:.2f}%"

        for sector_name, sec_ticker in SECTOR_MAP.items():
            r = get_1d_return(sec_ticker)
            if abs(r) < 1e-5 and sector_name in SECTOR_CONSTITUENTS:
                r = get_group_avg_return(SECTOR_CONSTITUENTS[sector_name])
            returns[f"Sector_{sector_name}"] = r

    except Exception:
        pass
    return trends, returns

idx_trends, market_returns = fetch_market_sentiments()

col1, col2, col3 = st.columns(3)
col1.metric("Nifty 50 (Sentiment)", idx_trends.get("^NSEI", "Active"))
col2.metric("Nifty Midcap", idx_trends.get("^NSEMDCP50", "Active"))
col3.metric("Nifty Smallcap", idx_trends.get("Smallcap", "Active"))

st.markdown("---")

# --- 3. UNIVERSE SETUP & METADATA REGISTRY ---
STOCK_METADATA = {
    "RELIANCE": {"index": "Nifty 50", "sector": "Energy"},
    "TCS": {"index": "Nifty 50", "sector": "IT"},
    "HDFCBANK": {"index": "Nifty 50", "sector": "Banking"},
    "INFY": {"index": "Nifty 50", "sector": "IT"},
    "ICICIBANK": {"index": "Nifty 50", "sector": "Banking"},
    "SBIN": {"index": "Nifty 50", "sector": "Banking"},
    "ITC": {"index": "Nifty 50", "sector": "FMCG"},
    "BAJFINANCE": {"index": "Nifty 50", "sector": "Financials"},
    "MARUTI": {"index": "Nifty 50", "sector": "Auto"},
    "TATASTEEL": {"index": "Nifty 50", "sector": "Metal"},
    "FEDERALBNK": {"index": "Nifty Midcap", "sector": "Banking"},
    "POLYCAB": {"index": "Nifty Midcap", "sector": "Capital Goods"},
    "ASHOKLEY": {"index": "Nifty Midcap", "sector": "Auto"},
    "CDSL": {"index": "Nifty Smallcap", "sector": "Financials"},
    "SUZLON": {"index": "Nifty Smallcap", "sector": "Energy"},
    "BSOFT": {"index": "Nifty Smallcap", "sector": "IT"}
}

scan_category = st.selectbox("Select Universe", ["All Combined", "Nifty 50", "Nifty Midcap", "Nifty Smallcap"])

if scan_category == "Nifty 50":
    selected_tickers = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty 50"]
elif scan_category == "Nifty Midcap":
    selected_tickers = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty Midcap"]
elif scan_category == "Nifty Smallcap":
    selected_tickers = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty Smallcap"]
else:
    selected_tickers = list(STOCK_METADATA.keys())

def fetch_stock_data(ticker):
    if "upstox_client" in st.session_state and st.session_state.get("upstox_client"):
        try:
            upstox = st.session_state["upstox_client"]
            df_5m = upstox.get_ohlc(ticker, interval="5m")
            df_1d = upstox.get_ohlc(ticker, interval="1d")
            if df_5m is not None and not df_5m.empty and df_1d is not None and not df_1d.empty:
                return df_5m, df_1d
        except Exception:
            pass
    
    yf_symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    df_5m = yf.download(yf_symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
    df_1d = yf.download(yf_symbol, period="1mo", interval="1d", progress=False, auto_adjust=True)
    return df_5m, df_1d

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# --- 4. ZONE LIFECYCLE ENGINE (NEW DYNAMIC SMC LOGIC) ---
def track_smc_zones(df_5m, lookback=50):
    last_price = float(df_5m['Close'].iloc[-1])
    last_low = float(df_5m['Low'].iloc[-1])
    last_high = float(df_5m['High'].iloc[-1])
    
    active_zones = []
    actual_lookback = min(lookback, len(df_5m) - 3)
    
    for i in range(len(df_5m) - actual_lookback, len(df_5m) - 1):
        # 1. Bullish Order Block (OB): Down candle prior to strong up move
        if df_5m['Close'].iloc[i] < df_5m['Open'].iloc[i] and df_5m['Close'].iloc[i+1] > df_5m['Open'].iloc[i+1]:
            ob_top = float(df_5m['High'].iloc[i])
            ob_bottom = float(df_5m['Low'].iloc[i])
            
            # Check Mitigation: Has any subsequent candle closed below the OB?
            subsequent_closes = df_5m['Close'].iloc[i+1:-1]
            if not (subsequent_closes < ob_bottom).any():
                dist_pct = ((last_price - ob_top) / ob_top) * 100
                age = len(df_5m) - 1 - i
                time_str = df_5m.index[i].strftime('%H:%M')
                
                if ob_bottom <= last_price <= ob_top or ob_bottom <= last_low <= ob_top:
                    state = "🟢 ENTRY READY"
                    state_val = 2
                elif 0 < dist_pct <= 0.4:
                    state = "🟡 PULLBACK NEAR"
                    state_val = 1
                else:
                    state = "⏸️ ZONE CREATED"
                    state_val = 0
                    
                active_zones.append({
                    'type': 'Bullish OB', 'top': ob_top, 'bottom': ob_bottom,
                    'age': age, 'time': time_str, 'state': state, 'state_val': state_val
                })

        # 2. Bullish Fair Value Gap (FVG)
        if i < len(df_5m) - 2:
            if df_5m['Low'].iloc[i+2] > df_5m['High'].iloc[i] and df_5m['Close'].iloc[i+1] > df_5m['Open'].iloc[i+1]:
                fvg_top = float(df_5m['Low'].iloc[i+2])
                fvg_bottom = float(df_5m['High'].iloc[i])
                
                subsequent_lows = df_5m['Low'].iloc[i+2:-1]
                if not (subsequent_lows <= fvg_bottom).any():
                    dist_pct = ((last_price - fvg_top) / fvg_top) * 100
                    age = len(df_5m) - 1 - i
                    
                    if fvg_bottom <= last_low <= fvg_top or fvg_bottom <= last_price <= fvg_top:
                        state = "🟢 ENTRY READY"
                        state_val = 2
                    elif 0 < dist_pct <= 0.4:
                        state = "🟡 PULLBACK NEAR"
                        state_val = 1
                    else:
                        state = "⏸️ ZONE CREATED"
                        state_val = 0
                        
                    active_zones.append({
                        'type': 'Bullish FVG', 'top': fvg_top, 'bottom': fvg_bottom,
                        'age': age, 'time': df_5m.index[i].strftime('%H:%M'), 
                        'state': state, 'state_val': state_val
                    })

        # 3. Liquidity Sweep Low Rejection
        if i > 15:
            recent_min = float(df_5m['Low'].iloc[i-15:i].min())
            if df_5m['Low'].iloc[i] < recent_min and df_5m['Close'].iloc[i] > recent_min:
                sweep_lvl = recent_min
                
                subsequent_closes = df_5m['Close'].iloc[i+1:-1]
                if not (subsequent_closes < sweep_lvl).any():
                    dist_pct = ((last_price - sweep_lvl) / sweep_lvl) * 100
                    
                    if 0 <= dist_pct <= 0.25:
                        state = "🟢 ENTRY READY"
                        state_val = 2
                    elif 0.25 < dist_pct <= 0.6:
                        state = "🟡 PULLBACK NEAR"
                        state_val = 1
                    else:
                        state = "⏸️ ZONE CREATED"
                        state_val = 0
                        
                    active_zones.append({
                        'type': 'Sweep Low', 'top': sweep_lvl * 1.001, 'bottom': sweep_lvl,
                        'age': len(df_5m) - 1 - i, 'time': df_5m.index[i].strftime('%H:%M'), 
                        'state': state, 'state_val': state_val
                    })

    return active_zones

# --- 5. COMPOSITE SCORE CALCULATION ---
def calculate_composite_score(row):
    ai_prob = float(row.get("Raw_AI_Prob", 50.0))  
    score_ai = ai_prob * 0.40

    smc_str = str(row.get("SMC Structure", "")).upper()
    smc_score = 0.0
    if "ENTRY READY" in smc_str: smc_score += 20.0
    elif "PULLBACK NEAR" in smc_str: smc_score += 10.0
    elif "ZONE CREATED" in smc_str: smc_score += 5.0
    score_smc = min(smc_score, 20.0)

    try:
        rr_ratio = float(row.get("Tgt_Pct_Num", 1.0)) / float(row.get("SL_Pct_Num", 0.5)) if float(row.get("SL_Pct_Num", 0.5)) > 0 else 1.0
    except: rr_ratio = 1.0
    score_rr = min((rr_ratio / 3.0) * 15.0, 15.0)

    trend_points = 0.0
    if row.get("Day Trend", "") == "Uptrend" and float(row.get("Sector_Return_Val", 0.0)) > 0: trend_points += 7.5
    if row.get("Day Trend", "") == "Uptrend" and float(row.get("Index_Return_Val", 0.0)) > 0: trend_points += 7.5
    score_align = trend_points

    rvol_val = float(row.get("RVOL_Val", 1.0))
    score_vol = min((rvol_val / 2.0) * 10.0, 10.0)

    total_score = score_ai + score_smc + score_rr + score_align + score_vol

    # Penalties
    if float(row.get("VWAP_Dist_Pct", 0.0)) > 1.5: total_score -= 20.0
    
    return max(0.0, round(total_score, 2))

# --- 6. CORE SCANNER ENGINE ---
if "locked_results" not in st.session_state:
    st.session_state.locked_results = None

ctrl_col1, ctrl_col2 = st.columns([1, 3])
with ctrl_col1: lock_signals = st.checkbox("🔒 Lock Watchlist", value=False)
run_scan = st.button("🚀 Run AI Scan & Rank", type="primary")

if lock_signals and st.session_state.locked_results is not None:
    results_df = st.session_state.locked_results
elif run_scan:
    with st.spinner("Evaluating realistic zone lifecycles & structure..."):
        results = []
        market_sentiment = float(market_returns.get("Nifty_1D_Return", 0.0)) * 100

        for ticker in selected_tickers:
            try:
                df_5m, df_1d = fetch_stock_data(ticker)
                if df_5m is None or df_1d is None or df_5m.empty or df_1d.empty: continue

                if isinstance(df_5m.columns, pd.MultiIndex): df_5m.columns = df_5m.columns.get_level_values(0)
                if isinstance(df_1d.columns, pd.MultiIndex): df_1d.columns = df_1d.columns.get_level_values(0)

                close_5m, high_5m, low_5m, open_5m = df_5m["Close"].dropna(), df_5m["High"].dropna(), df_5m["Low"].dropna(), df_5m["Open"].dropna()
                vol_5m = df_5m["Volume"].dropna() if "Volume" in df_5m else pd.Series(1, index=close_5m.index)
                close_1d, high_1d, low_1d, open_1d = df_1d["Close"].dropna(), df_1d["High"].dropna(), df_1d["Low"].dropna(), df_1d["Open"].dropna()

                if len(close_5m) < 50 or len(close_1d) < 15: continue

                last_price = float(close_5m.iloc[-1])
                day_open = float(open_1d.iloc[-1])

                # Normalised RVOL & VWAP
                avg_vol = vol_5m.tail(20).mean()
                rvol = float(vol_5m.iloc[-1] / (avg_vol + 1e-5))
                typical_price = (high_5m + low_5m + close_5m) / 3
                current_vwap = float(((typical_price * vol_5m).cumsum() / (vol_5m.cumsum() + 1e-5)).iloc[-1])
                vwap_dist_pct = float(((last_price - current_vwap) / current_vwap) * 100)

                # Trend, ATR, RSI
                ema_20 = close_5m.ewm(span=20, adjust=False).mean().iloc[-1]
                day_trend = "Uptrend" if (last_price > ema_20 and last_price >= day_open) else "Downtrend"
                
                tr_1d = pd.concat([high_1d - low_1d, (high_1d - close_1d.shift(1)).abs(), (low_1d - close_1d.shift(1)).abs()], axis=1).max(axis=1)
                atr_14_val = float(tr_1d.tail(14).mean())
                atr_pct = float((atr_14_val / last_price) * 100)
                
                rsi_series = compute_rsi(close_5m, period=14)
                rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

                # Flag Pattern Fallback
                recent_range = (high_5m.tail(10).max() - low_5m.tail(10).min()) / last_price
                flag_breakout = 1 if ((close_5m.iloc[-1] - close_5m.iloc[-10]) / close_5m.iloc[-10] > 0.003 and recent_range < 0.015) else 0

                # --- APPLY ZONE LIFECYCLE TRACKER ---
                active_zones = track_smc_zones(df_5m, lookback=50)
                
                # Identify Best Zone for UI
                best_zone = None
                smc_ui_str = "Structure Clean / No Valid Zones"
                zone_context = "N/A"
                
                if active_zones:
                    best_zone = sorted(active_zones, key=lambda x: x['state_val'], reverse=True)[0]
                    smc_ui_str = f"[{best_zone['state']}] {best_zone['type']} (₹{best_zone['bottom']:.1f}-₹{best_zone['top']:.1f})"
                    zone_context = f"Formed {best_zone['age']} bars ago @ {best_zone['time']}"

                # AI Feature Mapping from Zones (Must convert states back to 1 or 0 for the AI model)
                bull_fvg_feat = 1 if any(z['type'] == 'Bullish FVG' and z['state_val'] >= 1 for z in active_zones) else 0
                bull_ob_feat = 1 if any(z['type'] == 'Bullish OB' and z['state_val'] >= 1 for z in active_zones) else 0
                sweep_low_feat = 1 if any(z['type'] == 'Sweep Low' and z['state_val'] >= 1 for z in active_zones) else 0
                
                feature_dict = {
                    'RVOL': rvol, 'ATR_Pct': atr_pct, 'RSI': rsi_val,
                    'Liquidity_Sweep_High': 0, 'Liquidity_Sweep_Low': sweep_low_feat,
                    'Bullish_FVG': bull_fvg_feat, 'Bullish_OB': bull_ob_feat,
                    'Pattern_Flag_Breakout': flag_breakout, 'Market_Sentiment': market_sentiment
                }

                X_df = pd.DataFrame([{f: feature_dict.get(f, 0) for f in expected_features}])
                
                prob = 0.5
                if model is not None:
                    X_inf = pd.DataFrame(scaler.transform(X_df), columns=expected_features) if scaler else X_df
                    if hasattr(model, "predict_proba"):
                        prob = float(model.predict_proba(X_inf)[0][1])
                    else:
                        prob = float(model.predict(X_inf)[0])
                score_pct = prob * 100

                # --- DYNAMIC TARGET & STOPLOSS RESOLUTION ---
                recent_swing_high = float(high_5m.tail(50).max())
                recent_swing_low = float(low_5m.tail(50).min())

                if day_trend == "Uptrend":
                    if best_zone and best_zone['state_val'] >= 1:
                        # Institutional Stop Loss based on Zone Invalidation
                        sl_price = best_zone['bottom'] - (0.1 * atr_14_val)
                        # Target based on Historical Liquidity Pool
                        tgt_price = recent_swing_high if recent_swing_high > last_price else last_price + (1.5 * atr_14_val)
                    else:
                        sl_price = recent_swing_low - (0.25 * atr_14_val)
                        tgt_price = last_price + (0.5 * atr_14_val)
                else: # Downtrend fallback
                    sl_price = recent_swing_high + (0.25 * atr_14_val)
                    tgt_price = last_price - (0.5 * atr_14_val)

                dyn_tgt_pct = abs((tgt_price - last_price) / last_price) * 100
                dyn_sl_pct = abs((last_price - sl_price) / last_price) * 100
                
                meta = STOCK_METADATA.get(ticker, {"index": "Unknown", "sector": "General"})
                
                item = {
                    "Stock": ticker, "Index Group": meta["index"], "Sector": meta["sector"],
                    "Last Price": f"₹{last_price:.2f}", "VWAP": f"₹{current_vwap:.2f}",
                    "Day Trend": day_trend, "Signal State": smc_ui_str, "Zone Context": zone_context,
                    "Dynamic Target": f"₹{tgt_price:.2f} ({dyn_tgt_pct:.1f}%)", 
                    "Dynamic Stoploss": f"₹{sl_price:.2f} ({dyn_sl_pct:.1f}%)",
                    "AI Probability": f"{score_pct:.1f}%",
                    "Raw_AI_Prob": score_pct, "SMC Structure": smc_ui_str,
                    "Tgt_Pct_Num": dyn_tgt_pct, "SL_Pct_Num": dyn_sl_pct,
                    "Index_Return_Val": market_returns.get(f"Sector_{meta['sector']}", 0.0),
                    "Sector_Return_Val": market_returns.get("Nifty_1D_Return", 0.0),
                    "VWAP_Dist_Pct": vwap_dist_pct, "RVOL_Val": rvol
                }
                item["Rank Score"] = calculate_composite_score(item)
                results.append(item)
            except Exception: continue

        if results:
            df_temp = pd.DataFrame(results).sort_values(by="Rank Score", ascending=False).reset_index(drop=True)
            df_temp["Rank"] = df_temp.index + 1
            results_df = df_temp
            st.session_state.locked_results = results_df

# --- 7. DISPLAY DASHBOARD ---
if "results_df" in locals() and results_df is not None:
    st.subheader("🎯 TOP 3 ACTIONABLE TRADES")
    
    top_3 = results_df.head(3)
    card_cols = st.columns(3)
    for idx, col in enumerate(card_cols):
        if idx < len(top_3):
            row = top_3.iloc[idx]
            with col:
                st.metric(label=f"#{row['Rank']} {row['Stock']} ({row['Sector']})", value=row["Last Price"], delta=f"Score: {row['Rank Score']}")
                st.write(f"**State:** `{row['Signal State']}`")
                st.write(f"**Context:** {row['Zone Context']}")
                st.write(f"**Target:** {row['Dynamic Target']}")
                st.write(f"**Stoploss:** {row['Dynamic Stoploss']}")

    st.markdown("---")
    st.subheader("📊 FULL WATCHLIST & LIFECYCLE STATUS")
    
    display_cols = [
        "Rank", "Stock", "Sector", "Rank Score", "Last Price", "Day Trend", 
        "Signal State", "Zone Context", "Dynamic Target", "Dynamic Stoploss", "AI Probability"
    ]
    st.dataframe(results_df[display_cols], use_container_width=True)
