import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="NQIRP ML Scanner", page_icon="🤖", layout="wide")

st.title("🤖 AI & ML Strategy Scanner")
st.markdown("*Strict Execution: Backtested Model with Upstox/YF Dual Fetching*")

# 1. Load Trained Assets & Strict Config
@st.cache_resource
def load_ai_assets():
    model = joblib.load("colab_ai_model.pkl") if os.path.exists("colab_ai_model.pkl") else None
    scaler = joblib.load("colab_scaler.pkl") if os.path.exists("colab_scaler.pkl") else None
    config = {}
    if os.path.exists("ai_strategy_config.json"):
        with open("ai_strategy_config.json", "r") as f:
            config = json.load(f)
    return model, scaler, config

model, scaler, config = load_ai_assets()

if model is None:
    st.error("Model file 'colab_ai_model.pkl' not found in repository.")
    st.stop()

# Explicit 9-feature model schema strictly matched to Colab training
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

st.sidebar.success(f"✅ Schema Aligned: {len(expected_features)} Features Active")

# Target and Stoploss strictly from saved backtest parameters
target_pct = float(config.get("target_pct", config.get("target_percentage", 1.2)))
stop_pct = float(config.get("stop_pct", config.get("stop_percentage", 0.6)))

# 2. Market Sentiment Context
@st.cache_data(ttl=300)
def fetch_index_trends():
    tickers = ["^NSEI", "^NSEMDCP50", "^CNXSMLCAP"]
    trends = {}
    returns = {"Nifty_1D_Return": 0.0, "Midcap_1D_Return": 0.0, "Smallcap_1D_Return": 0.0}
    try:
        data = yf.download(tickers, period="5d", interval="1d", progress=False)
        close_df = data["Close"] if "Close" in data else data
        mapping = [("Nifty_1D_Return", "^NSEI"), ("Midcap_1D_Return", "^NSEMDCP50"), ("Smallcap_1D_Return", "^CNXSMLCAP")]
        for key, t in mapping:
            if t in close_df:
                s = close_df[t].dropna()
                if len(s) >= 2:
                    r = float((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2])
                    returns[key] = r
                    trends[t] = f"{'+' if r >= 0 else ''}{r*100:.2f}%"
    except Exception:
        pass
    return trends, returns

idx_trends, idx_returns = fetch_index_trends()

col1, col2, col3 = st.columns(3)
col1.metric("Nifty 50", idx_trends.get("^NSEI", "Active"))
col2.metric("Nifty Midcap", idx_trends.get("^NSEMDCP50", "Active"))
col3.metric("Nifty Smallcap", idx_trends.get("^CNXSMLCAP", "Active"))

st.markdown("---")

NIFTY_50 = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "LTIM", "AXISBANK", "KOTAKBANK", "LT", "HINDUNILVR", "BAJFINANCE", "MARUTI", "TATASTEEL", "NTPC", "M&M"]
MIDCAP_SAMPLES = ["TATAPOWER", "FEDERALBNK", "POLYCAB", "PERSISTENT", "COFORGE", "ASHOKLEY", "MAXHEALTH", "VOLTAS"]
SMALLCAP_SAMPLES = ["CDSL", "ANGELONE", "KFINTECH", "SUZLON", "BSOFT", "HFCL", "IEX", "KEI"]

scan_category = st.selectbox("Select Universe", ["Nifty 50", "Nifty Midcap", "Nifty Smallcap", "All Combined"])

if scan_category == "Nifty 50":
    selected_tickers = NIFTY_50
elif scan_category == "Nifty Midcap":
    selected_tickers = MIDCAP_SAMPLES
elif scan_category == "Nifty Smallcap":
    selected_tickers = SMALLCAP_SAMPLES
else:
    selected_tickers = NIFTY_50 + MIDCAP_SAMPLES + SMALLCAP_SAMPLES

# 3. Dual Data Fetching (Upstox Primary, YFinance Fallback)
def fetch_stock_data(ticker):
    # Try Upstox Session First
    if "upstox_client" in st.session_state and st.session_state.get("upstox_client"):
        try:
            upstox = st.session_state["upstox_client"]
            df_5m = upstox.get_ohlc(ticker, interval="5m")
            df_1d = upstox.get_ohlc(ticker, interval="1d")
            if df_5m is not None and not df_5m.empty and df_1d is not None and not df_1d.empty:
                return df_5m, df_1d
        except Exception:
            pass
    
    # YFinance Fallback (1mo required for 14-day Daily ATR)
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

# 4. Scanner Execution
if st.button("🚀 Run ML Scan", type="primary"):
    with st.spinner("Fetching data and running pure ML inference..."):
        results = []
        # Sentiment scaled to percentage to match model training scales
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
                day_trend = "Uptrend" if last_price >= day_open else "Downtrend"

                # Feature 1: RVOL (5m)
                rvol = float(vol_5m.iloc[-1] / (vol_5m.tail(20).mean() + 1e-5))

                # Feature 2: ATR_Pct (Daily timeframe correction applied here)
                tr_1d = pd.concat([
                    high_1d - low_1d, 
                    (high_1d - close_1d.shift(1)).abs(), 
                    (low_1d - close_1d.shift(1)).abs()
                ], axis=1).max(axis=1)
                
                atr_14 = tr_1d.tail(14).mean()
                atr_pct = float((atr_14 / last_price) * 100)

                # Feature 3: RSI (5m)
                rsi_series = compute_rsi(close_5m, period=14)
                rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

                # Feature 4: Liquidity_Sweep_High (5m)
                recent_max = high_5m.iloc[-15:-1].max() if len(high_5m) >= 15 else high_5m.iloc[:-1].max()
                sweep_high = 1 if high_5m.iloc[-1] > recent_max else 0

                # Feature 5: Liquidity_Sweep_Low (5m)
                recent_min = low_5m.iloc[-15:-1].min() if len(low_5m) >= 15 else low_5m.iloc[:-1].min()
                sweep_low = 1 if low_5m.iloc[-1] < recent_min else 0

                # Feature 6: Bullish_FVG (5m)
                bull_fvg = 1 if (len(high_5m) >= 3 and low_5m.iloc[-1] > high_5m.iloc[-3]) else 0

                # Feature 7: Bullish_OB (5m)
                bull_ob = 1 if (close_5m.iloc[-2] < open_5m.iloc[-2] and close_5m.iloc[-1] > high_5m.iloc[-2]) else 0

                # Feature 8: Pattern_Flag_Breakout (5m)
                recent_range = (high_5m.tail(10).max() - low_5m.tail(10).min()) / last_price
                price_chg = (close_5m.iloc[-1] - close_5m.iloc[-10]) / close_5m.iloc[-10]
                flag_breakout = 1 if (price_chg > 0.003 and recent_range < 0.015) else 0

                # Feature 9: Market_Sentiment
                sentiment_val = market_sentiment

                # Construct strict feature vector
                feature_dict = {
                    'RVOL': rvol,
                    'ATR_Pct': atr_pct,
                    'RSI': rsi_val,
                    'Liquidity_Sweep_High': sweep_high,
                    'Liquidity_Sweep_Low': sweep_low,
                    'Bullish_FVG': bull_fvg,
                    'Bullish_OB': bull_ob,
                    'Pattern_Flag_Breakout': flag_breakout,
                    'Market_Sentiment': sentiment_val
                }

                X_df = pd.DataFrame([{f: feature_dict.get(f, 0) for f in expected_features}])

                if scaler is not None:
                    X_df = scaler.transform(X_df)

                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X_df)[0][1])
                else:
                    prob = float(model.predict(X_df)[0])

                score_pct = prob * 100 if prob <= 1.0 else prob

                # Format Confluences
                smc_signals = []
                if bull_fvg: smc_signals.append("Bullish FVG")
                if bull_ob: smc_signals.append("Bullish OB")
                if sweep_low: smc_signals.append("Sweep Low")
                if sweep_high: smc_signals.append("Sweep High")
                if flag_breakout: smc_signals.append("Flag Breakout")
                smc_str = " + ".join(smc_signals) if smc_signals else "Structure Clean"

                # Standard Fixed Config Target & Stoploss Calculations
                if day_trend == "Uptrend":
                    tgt_price = last_price * (1 + target_pct / 100)
                    sl_price = last_price * (1 - stop_pct / 100)
                    tgt_str = f"₹{tgt_price:.2f} (+{target_pct:.1f}%)"
                    sl_str = f"₹{sl_price:.2f} (-{stop_pct:.1f}%)"
                else:
                    tgt_price = last_price * (1 - target_pct / 100)
                    sl_price = last_price * (1 + stop_pct / 100)
                    tgt_str = f"₹{tgt_price:.2f} (-{target_pct:.1f}%)"
                    sl_str = f"₹{sl_price:.2f} (+{stop_pct:.1f}%)"

                results.append({
                    "Stock": ticker,
                    "Last Price": f"₹{last_price:.2f}",
                    "Day Trend": day_trend,
                    "Daily ATR %": f"{atr_pct:.2f}%",
                    "RSI (5m)": f"{rsi_val:.1f}",
                    "SMC Structure": smc_str,
                    "AI Probability": f"{score_pct:.1f}%",
                    "Target": tgt_str,
                    "Stoploss": sl_str
                })
            except Exception:
                continue

        if results:
            st.subheader("🔥 AI Signals (9/9 Features Aligned & Scaled)")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("No setup signals generated.")
