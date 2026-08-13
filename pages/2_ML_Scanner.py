import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="NQIRP ML Scanner", page_icon="🤖", layout="wide")

st.title("🤖 AI & ML Strategy Scanner")
st.markdown("*PDF-Aligned Execution: Backtested Model + Dynamic SMC & ATR Targets*")

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_ai_assets():
    model = joblib.load("colab_ai_model.pkl") if os.path.exists("colab_ai_model.pkl") else None
    scaler = joblib.load("colab_scaler.pkl") if os.path.exists("colab_scaler.pkl") else None
    return model, scaler

model, scaler = load_ai_assets()

if model is None:
    st.error("Model file 'colab_ai_model.pkl' not found in repository.")
    st.stop()

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

if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    expected_features = EXACT_FEATURES

st.sidebar.success(f"✅ AI Engine Active: {len(expected_features)} Features")
st.sidebar.info("🎯 Dynamic ATR & Structural Swings applied for Targets/Stoploss.")

# --- 2. INDEX & MARKET SENTIMENT FETCHING ---
@st.cache_data(ttl=300)
def fetch_index_trends():
    tickers = ["^NSEI", "^NSEMDCP50", "NIFTYSMALL100.NS", "^CNXSMLCAP"]
    trends = {}
    returns = {"Nifty_1D_Return": 0.0, "Midcap_1D_Return": 0.0, "Smallcap_1D_Return": 0.0}
    try:
        data = yf.download(tickers, period="5d", interval="1d", progress=False)
        close_df = data["Close"] if "Close" in data else data
        
        # Nifty 50
        if "^NSEI" in close_df:
            s = close_df["^NSEI"].dropna()
            if len(s) >= 2:
                r = float((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2])
                returns["Nifty_1D_Return"] = r
                trends["^NSEI"] = f"{'+' if r >= 0 else ''}{r*100:.2f}%"

        # Nifty Midcap
        if "^NSEMDCP50" in close_df:
            s = close_df["^NSEMDCP50"].dropna()
            if len(s) >= 2:
                r = float((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2])
                returns["Midcap_1D_Return"] = r
                trends["^NSEMDCP50"] = f"{'+' if r >= 0 else ''}{r*100:.2f}%"

        # Nifty Smallcap (multi-fallback)
        sml_s = None
        for t in ["NIFTYSMALL100.NS", "^CNXSMLCAP"]:
            if t in close_df:
                s_cand = close_df[t].dropna()
                if len(s_cand) >= 2 and abs(float((s_cand.iloc[-1] - s_cand.iloc[-2]) / s_cand.iloc[-2])) > 1e-5:
                    sml_s = s_cand
                    break
        if sml_s is not None:
            r = float((sml_s.iloc[-1] - sml_s.iloc[-2]) / sml_s.iloc[-2])
            returns["Smallcap_1D_Return"] = r
            trends["Smallcap"] = f"{'+' if r >= 0 else ''}{r*100:.2f}%"
        else:
            trends["Smallcap"] = "0.00%"
    except Exception:
        pass
    return trends, returns

idx_trends, idx_returns = fetch_index_trends()

col1, col2, col3 = st.columns(3)
col1.metric("Nifty 50 (Sentiment)", idx_trends.get("^NSEI", "Active"))
col2.metric("Nifty Midcap", idx_trends.get("^NSEMDCP50", "Active"))
col3.metric("Nifty Smallcap", idx_trends.get("Smallcap", "Active"))

st.markdown("---")

# --- 3. UNIVERSE SETUP & METADATA REGISTRY ---
STOCK_METADATA = {
    # NIFTY 50
    "RELIANCE": {"index": "Nifty 50", "sector": "Energy"},
    "TCS": {"index": "Nifty 50", "sector": "IT"},
    "HDFCBANK": {"index": "Nifty 50", "sector": "Banking"},
    "INFY": {"index": "Nifty 50", "sector": "IT"},
    "ICICIBANK": {"index": "Nifty 50", "sector": "Banking"},
    "SBIN": {"index": "Nifty 50", "sector": "Banking"},
    "BHARTIARTL": {"index": "Nifty 50", "sector": "Telecom"},
    "ITC": {"index": "Nifty 50", "sector": "FMCG"},
    "LTIM": {"index": "Nifty 50", "sector": "IT"},
    "AXISBANK": {"index": "Nifty 50", "sector": "Banking"},
    "KOTAKBANK": {"index": "Nifty 50", "sector": "Banking"},
    "LT": {"index": "Nifty 50", "sector": "Infra"},
    "HINDUNILVR": {"index": "Nifty 50", "sector": "FMCG"},
    "BAJFINANCE": {"index": "Nifty 50", "sector": "Financials"},
    "MARUTI": {"index": "Nifty 50", "sector": "Auto"},
    "TATASTEEL": {"index": "Nifty 50", "sector": "Metal"},
    "NTPC": {"index": "Nifty 50", "sector": "Energy"},
    "M&M": {"index": "Nifty 50", "sector": "Auto"},

    # MIDCAP
    "TATAPOWER": {"index": "Nifty Midcap", "sector": "Energy"},
    "FEDERALBNK": {"index": "Nifty Midcap", "sector": "Banking"},
    "POLYCAB": {"index": "Nifty Midcap", "sector": "Capital Goods"},
    "PERSISTENT": {"index": "Nifty Midcap", "sector": "IT"},
    "COFORGE": {"index": "Nifty Midcap", "sector": "IT"},
    "ASHOKLEY": {"index": "Nifty Midcap", "sector": "Auto"},
    "MAXHEALTH": {"index": "Nifty Midcap", "sector": "Healthcare"},
    "VOLTAS": {"index": "Nifty Midcap", "sector": "Consumer Durables"},

    # SMALLCAP
    "CDSL": {"index": "Nifty Smallcap", "sector": "Financials"},
    "ANGELONE": {"index": "Nifty Smallcap", "sector": "Financials"},
    "KFINTECH": {"index": "Nifty Smallcap", "sector": "Financials"},
    "SUZLON": {"index": "Nifty Smallcap", "sector": "Energy"},
    "BSOFT": {"index": "Nifty Smallcap", "sector": "IT"},
    "HFCL": {"index": "Nifty Smallcap", "sector": "Infra"},
    "IEX": {"index": "Nifty Smallcap", "sector": "Financials"},
    "KEI": {"index": "Nifty Smallcap", "sector": "Capital Goods"}
}

NIFTY_50 = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty 50"]
MIDCAP_SAMPLES = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty Midcap"]
SMALLCAP_SAMPLES = [k for k, v in STOCK_METADATA.items() if v["index"] == "Nifty Smallcap"]

scan_category = st.selectbox("Select Universe", ["All Combined (32 Stocks)", "Nifty 50", "Nifty Midcap", "Nifty Smallcap"])

if scan_category == "Nifty 50":
    selected_tickers = NIFTY_50
elif scan_category == "Nifty Midcap":
    selected_tickers = MIDCAP_SAMPLES
elif scan_category == "Nifty Smallcap":
    selected_tickers = SMALLCAP_SAMPLES
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

# --- 4. COMPOSITE RANKING ENGINE (PDF + INDEX RELATIVE STRENGTH ALIGNED) ---
def calculate_composite_score(row):
    ai_prob = float(row.get("Raw_AI_Prob", 50.0))
    
    smc_str = str(row.get("SMC Structure", "")).upper()
    if "+" in smc_str:
        smc_mult = 1.25  # Multiple confluences
    elif smc_str in ["STRUCTURE CLEAN", "", "NONE"]:
        smc_mult = 0.80
    else:
        smc_mult = 1.10  # Single confluence

    try:
        tgt_val = float(row.get("Tgt_Pct_Num", 1.0))
        sl_val = float(row.get("SL_Pct_Num", 0.5))
        rr_ratio = tgt_val / sl_val if sl_val > 0 else 1.0
    except Exception:
        rr_ratio = 1.0

    day_trend = str(row.get("Day Trend", "")).strip()
    
    # Structural alignment with day trend
    is_bullish_smc = any(x in smc_str for x in ["BULLISH", "SWEEP LOW", "FLAG BREAKOUT"])
    is_bearish_smc = any(x in smc_str for x in ["BEARISH", "SWEEP HIGH", "FLAG BREAKOUT"])
    
    trend_align = 1.0
    if day_trend == "Uptrend" and is_bullish_smc:
        trend_align = 1.20
    elif day_trend == "Downtrend" and is_bearish_smc:
        trend_align = 1.20
    elif (day_trend == "Uptrend" and is_bearish_smc) or (day_trend == "Downtrend" and is_bullish_smc):
        trend_align = 0.60  # Discount when fighting intraday day trend

    # Parent Index Alignment Multiplier
    idx_ret = float(row.get("Index_Return_Val", 0.0))
    idx_align = 1.0
    if day_trend == "Uptrend" and idx_ret > 0:
        idx_align = 1.15
    elif day_trend == "Downtrend" and idx_ret < 0:
        idx_align = 1.15
    elif (day_trend == "Uptrend" and idx_ret < -0.003) or (day_trend == "Downtrend" and idx_ret > 0.003):
        idx_align = 0.85

    score = ai_prob * smc_mult * rr_ratio * trend_align * idx_align
    return round(score, 2)

# --- 5. CONTROLS & SESSION LOCK ---
if "locked_results" not in st.session_state:
    st.session_state.locked_results = None

ctrl_col1, ctrl_col2 = st.columns([1, 3])
with ctrl_col1:
    lock_signals = st.checkbox("🔒 Lock Watchlist (Freeze Live Flickering)", value=False)

run_scan = st.button("🚀 Run AI Scan & Rank", type="primary")

if lock_signals and st.session_state.locked_results is not None:
    st.info("🔒 Displaying Locked Watchlist. Uncheck to unlock live updates.")
    results_df = st.session_state.locked_results
elif run_scan:
    with st.spinner("Evaluating 32 stocks across Nifty 50, Midcap & Smallcap..."):
        results = []
        market_sentiment = float(idx_returns.get("Nifty_1D_Return", 0.0)) * 100

        for ticker in selected_tickers:
            try:
                df_5m, df_1d = fetch_stock_data(ticker)
                if df_5m is None or df_5m.empty or df_1d is None or df_1d.empty:
                    continue

                if isinstance(df_5m.columns, pd.MultiIndex):
                    df_5m.columns = df_5m.columns.get_level_values(0)
                if isinstance(df_1d.columns, pd.MultiIndex):
                    df_1d.columns = df_1d.columns.get_level_values(0)

                close_5m = df_5m["Close"].dropna()
                high_5m = df_5m["High"].dropna()
                low_5m = df_5m["Low"].dropna()
                open_5m = df_5m["Open"].dropna()
                vol_5m = df_5m["Volume"].dropna() if "Volume" in df_5m else pd.Series(1, index=close_5m.index)
                
                close_1d = df_1d["Close"].dropna()
                high_1d = df_1d["High"].dropna()
                low_1d = df_1d["Low"].dropna()
                open_1d = df_1d["Open"].dropna()

                if len(close_5m) < 20 or len(close_1d) < 15:
                    continue

                last_price = float(close_5m.iloc[-1])
                day_open = float(open_1d.iloc[-1])
                
                # Day Trend (Price vs Day Open)
                day_trend = "Uptrend" if last_price >= day_open else "Downtrend"

                rvol = float(vol_5m.iloc[-1] / (vol_5m.tail(20).mean() + 1e-5))

                tr_1d = pd.concat([
                    high_1d - low_1d, 
                    (high_1d - close_1d.shift(1)).abs(), 
                    (low_1d - close_1d.shift(1)).abs()
                ], axis=1).max(axis=1)
                
                atr_14_val = float(tr_1d.tail(14).mean())
                atr_pct = float((atr_14_val / last_price) * 100)

                rsi_series = compute_rsi(close_5m, period=14)
                rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

                recent_15_max = high_5m.iloc[-15:-1].max() if len(high_5m) >= 15 else high_5m.iloc[:-1].max()
                sweep_high = 1 if high_5m.iloc[-1] > recent_15_max else 0

                recent_15_min = low_5m.iloc[-15:-1].min() if len(low_5m) >= 15 else low_5m.iloc[:-1].min()
                sweep_low = 1 if low_5m.iloc[-1] < recent_15_min else 0

                bull_fvg = 1 if (len(high_5m) >= 3 and low_5m.iloc[-1] > high_5m.iloc[-3]) else 0
                bull_ob = 1 if (close_5m.iloc[-2] < open_5m.iloc[-2] and close_5m.iloc[-1] > high_5m.iloc[-2]) else 0

                recent_range = (high_5m.tail(10).max() - low_5m.tail(10).min()) / last_price
                price_chg = (close_5m.iloc[-1] - close_5m.iloc[-10]) / close_5m.iloc[-10]
                flag_breakout = 1 if (price_chg > 0.003 and recent_range < 0.015) else 0

                feature_dict = {
                    'RVOL': rvol,
                    'ATR_Pct': atr_pct,
                    'RSI': rsi_val,
                    'Liquidity_Sweep_High': sweep_high,
                    'Liquidity_Sweep_Low': sweep_low,
                    'Bullish_FVG': bull_fvg,
                    'Bullish_OB': bull_ob,
                    'Pattern_Flag_Breakout': flag_breakout,
                    'Market_Sentiment': market_sentiment
                }

                X_df = pd.DataFrame([{f: feature_dict.get(f, 0) for f in expected_features}])

                if scaler is not None:
                    scaled_array = scaler.transform(X_df)
                    X_inference = pd.DataFrame(scaled_array, columns=expected_features)
                else:
                    X_inference = X_df

                if hasattr(model, "predict_proba"):
                    raw_probs = model.predict_proba(X_inference)[0]
                    prob = float(raw_probs[1])
                    if prob < 0.3 and (bull_fvg or bull_ob or flag_breakout):
                        prob = float(raw_probs[0])
                else:
                    prob = float(model.predict(X_inference)[0])

                score_pct = prob * 100 if prob <= 1.0 else prob

                smc_signals = []
                if bull_fvg: smc_signals.append("Bullish FVG")
                if bull_ob: smc_signals.append("Bullish OB")
                if sweep_low: smc_signals.append("Sweep Low")
                if sweep_high: smc_signals.append("Sweep High")
                if flag_breakout: smc_signals.append("Flag Breakout")
                smc_str = " + ".join(smc_signals) if smc_signals else "Structure Clean"

                # Dynamic Target & SL (Daily ATR & Dynamic Swings)
                recent_swing_high = float(high_5m.tail(15).max())
                recent_swing_low = float(low_5m.tail(15).min())
                
                if day_trend == "Uptrend":
                    sl_price = recent_swing_low if (last_price - recent_swing_low) > (0.1 * atr_14_val) else (last_price - 0.25 * atr_14_val)
                    tgt_price = last_price + (0.5 * atr_14_val)
                    
                    dyn_tgt_pct = ((tgt_price - last_price) / last_price) * 100
                    dyn_sl_pct = ((last_price - sl_price) / last_price) * 100
                    
                    tgt_str = f"₹{tgt_price:.2f} (+{dyn_tgt_pct:.1f}%)"
                    sl_str = f"₹{sl_price:.2f} (-{dyn_sl_pct:.1f}%)"
                else:
                    sl_price = recent_swing_high if (recent_swing_high - last_price) > (0.1 * atr_14_val) else (last_price + 0.25 * atr_14_val)
                    tgt_price = last_price - (0.5 * atr_14_val)
                    
                    dyn_tgt_pct = ((last_price - tgt_price) / last_price) * 100
                    dyn_sl_pct = ((sl_price - last_price) / last_price) * 100
                    
                    tgt_str = f"₹{tgt_price:.2f} (-{dyn_tgt_pct:.1f}%)"
                    sl_str = f"₹{sl_price:.2f} (+{dyn_sl_pct:.1f}%)"

                meta = STOCK_METADATA.get(ticker, {"index": "Nifty 50", "sector": "General"})
                idx_group = meta["index"]
                sector_group = meta["sector"]

                if idx_group == "Nifty 50":
                    idx_ret_val = idx_returns.get("Nifty_1D_Return", 0.0)
                elif idx_group == "Nifty Midcap":
                    idx_ret_val = idx_returns.get("Midcap_1D_Return", 0.0)
                else:
                    idx_ret_val = idx_returns.get("Smallcap_1D_Return", 0.0)

                item = {
                    "Stock": ticker,
                    "Index Group": idx_group,
                    "Sector": sector_group,
                    "Last Price": f"₹{last_price:.2f}",
                    "Day Trend": day_trend,
                    "Daily ATR %": f"{atr_pct:.2f}%", 
                    "RSI (5m)": f"{rsi_val:.1f}",
                    "SMC Structure": smc_str,
                    "AI Probability": f"{score_pct:.1f}%",
                    "Dynamic Target": tgt_str,
                    "Dynamic Stoploss": sl_str,
                    "Raw_AI_Prob": score_pct,
                    "Tgt_Pct_Num": dyn_tgt_pct,
                    "SL_Pct_Num": dyn_sl_pct,
                    "Index_Return_Val": idx_ret_val
                }
                item["Rank Score"] = calculate_composite_score(item)
                results.append(item)
            except Exception:
                continue

        if results:
            df_temp = pd.DataFrame(results).sort_values(by="Rank Score", ascending=False).reset_index(drop=True)
            df_temp["Rank"] = df_temp.index + 1
            results_df = df_temp
            st.session_state.locked_results = results_df

# --- 6. DISPLAY SECTION ---
if "results_df" in locals() and results_df is not None:
    st.subheader("🎯 TOP 3 HIGH-CONVICTION TRADES")
    st.caption("Highest probability setups ranked by PDF Composite Score (AI Prob × SMC Confluence × Dynamic RR × Day Trend & Index Alignment).")
    
    top_3 = results_df.head(3)
    card_cols = st.columns(3)
    for idx, col in enumerate(card_cols):
        if idx < len(top_3):
            row = top_3.iloc[idx]
            with col:
                st.metric(
                    label=f"Rank #{row['Rank']} — {row['Stock']} ({row['Index Group']})", 
                    value=row["Last Price"], 
                    delta=f"Rank Score: {row['Rank Score']}"
                )
                st.write(f"**Sector:** {row['Sector']} | **Trend:** {row['Day Trend']}")
                st.write(f"**SMC:** {row['SMC Structure']}")
                st.write(f"**Target:** `{row['Dynamic Target']}`")
                st.write(f"**Stoploss:** `{row['Dynamic Stoploss']}`")
                st.write(f"**AI Prob:** {row['AI Probability']}")

    st.markdown("---")

    st.subheader("📊 ALL STOCKS — RANKED WATCHLIST WITH SECTOR MAPPING")
    
    display_cols = [
        "Rank", "Stock", "Index Group", "Sector", "Rank Score", "Last Price", "Day Trend", 
        "Daily ATR %", "RSI (5m)", "SMC Structure", "AI Probability", 
        "Dynamic Target", "Dynamic Stoploss"
    ]
    
    st.dataframe(results_df[display_cols], use_container_width=True)
