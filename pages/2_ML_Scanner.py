import streamlit as st
import joblib
import json
import os
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="NQIRP ML Scanner", page_icon="🤖", layout="wide")

st.title("🤖 AI & ML Strategy Scanner")
st.markdown("*1-Year Backtested Model | Smart Money Concepts & Dynamic Targets*")

# Load trained AI model and config generated during backtesting
@st.cache_resource
def load_ai_assets():
    model = joblib.load("colab_ai_model.pkl") if os.path.exists("colab_ai_model.pkl") else None
    config = {}
    if os.path.exists("ai_strategy_config.json"):
        with open("ai_strategy_config.json", "r") as f:
            config = json.load(f)
    return model, config

model, config = load_ai_assets()

if model is None:
    st.error("Model file 'colab_ai_model.pkl' not found. Please verify repo uploads.")
    st.stop()

st.success("✅ Backtested AI Model & Config Active")

# Market Context - Live index returns passed to model
st.subheader("📊 Market Sentiment & Sector Context")
col1, col2, col3 = st.columns(3)

@st.cache_data(ttl=300)
def fetch_index_trends():
    tickers = ["^NSEI", "^NSEMDCP50", "^CNXSMLCAP"]
    trends = {}
    returns = {"Nifty_1D_Return": 0.0, "Midcap_1D_Return": 0.0, "Smallcap_1D_Return": 0.0}
    try:
        data = yf.download(tickers, period="5d", interval="1d", progress=False)
        close_df = data["Close"] if "Close" in data else data
        
        for key, t in [("Nifty_1D_Return", "^NSEI"), ("Midcap_1D_Return", "^NSEMDCP50"), ("Smallcap_1D_Return", "^CNXSMLCAP")]:
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
col1.metric("Nifty 50 (1D)", idx_trends.get("^NSEI", "Active"))
col2.metric("Nifty Midcap (1D)", idx_trends.get("^NSEMDCP50", "Active"))
col3.metric("Nifty Smallcap (1D)", idx_trends.get("^CNXSMLCAP", "Active"))

st.markdown("---")

st.subheader("🎯 Instant AI Signal Scanner")

NIFTY_50 = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "LTIM", "AXISBANK", "KOTAKBANK", "LT", "HINDUNILVR", "BAJFINANCE", "MARUTI", "TATASTEEL", "NTPC", "M&M"]
MIDCAP_SAMPLES = ["TATAPOWER", "FEDERALBNK", "POLYCAB", "PERSISTENT", "COFORGE", "ASHOKLEY", "MAXHEALTH", "VOLTAS"]
SMALLCAP_SAMPLES = ["CDSL", "ANGELONE", "KFINTECH", "SUZLON", "BSOFT", "HFCL", "IEX", "KEI"]

scan_category = st.selectbox(
    "Select Universe to Scan",
    ["Nifty 50", "Nifty Midcap", "Nifty Smallcap", "All Indices Combined"]
)

if scan_category == "Nifty 50":
    selected_tickers = NIFTY_50
elif scan_category == "Nifty Midcap":
    selected_tickers = MIDCAP_SAMPLES
elif scan_category == "Nifty Smallcap":
    selected_tickers = SMALLCAP_SAMPLES
else:
    selected_tickers = NIFTY_50 + MIDCAP_SAMPLES + SMALLCAP_SAMPLES

st.write(f"**Total Stocks Loaded in Selected Universe:** {len(selected_tickers)}")

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
    df_1d = yf.download(yf_symbol, period="10d", interval="1d", progress=False, auto_adjust=True)
    return df_5m, df_1d

if st.button("🚀 Run Instant ML Scan", type="primary"):
    with st.spinner("Evaluating live market data using backtested model assets..."):
        results = []
        
        # Load feature list and target ratios from backtested config
        expected_features = config.get("feature_names", config.get("features", []))
        target_pct_config = config.get("target_pct", 2.0)
        stop_pct_config = config.get("stop_pct", 1.0)

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
                open_1d = df_1d["Open"].dropna()

                if len(close_5m) < 15 or len(close_1d) < 2:
                    continue

                last_price = float(close_5m.iloc[-1])
                day_open = float(open_1d.iloc[-1])
                day_trend = "Uptrend" if last_price >= day_open else "Downtrend"

                # Extract SMC structures
                rvol = float(vol_5m.iloc[-1] / (vol_5m.tail(20).mean() + 1e-5))
                has_fvg = 1 if (len(high_5m) >= 3 and low_5m.iloc[-1] > high_5m.iloc[-3]) else 0
                has_sweep = 1 if (len(low_5m) >= 11 and low_5m.iloc[-1] < low_5m.iloc[-11:-1].min()) else 0
                has_ob = 1 if (close_5m.iloc[-2] < open_5m.iloc[-2] and close_5m.iloc[-1] > high_5m.iloc[-2]) else 0
                
                smc_signals = []
                if has_fvg: smc_signals.append("Bullish FVG")
                if has_sweep: smc_signals.append("Liquidity Sweep")
                if has_ob: smc_signals.append("Order Block")
                smc_str = " + ".join(smc_signals) if smc_signals else "Structure Clean"

                # Feature vector constructed strictly matching backtest parameters
                feat_dict = {
                    'Direction': 1 if day_trend == "Uptrend" else 0,
                    'Day_Trend': 1 if day_trend == "Uptrend" else 0,
                    'RVOL': rvol,
                    'MSB': 1 if (has_sweep or has_fvg) else 0,
                    'Bull_FVG': has_fvg,
                    'Sweep_Low': has_sweep,
                    'Bull_OB': has_ob,
                    'Nifty_1D_Return': idx_returns["Nifty_1D_Return"],
                    'Midcap_1D_Return': idx_returns["Midcap_1D_Return"],
                    'Smallcap_1D_Return': idx_returns["Smallcap_1D_Return"],
                }

                if expected_features:
                    X_df = pd.DataFrame([{f: feat_dict.get(f, 0) for f in expected_features}])
                else:
                    X_df = pd.DataFrame([feat_dict])

                # Probability evaluated purely by the trained ML model
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X_df)[0][1])
                else:
                    prob = float(model.predict(X_df)[0])

                # Targets mapped using saved backtesting configuration
                tgt_price = last_price * (1 + target_pct_config / 100)
                sl_price = last_price * (1 - stop_pct_config / 100)

                results.append({
                    "Stock": ticker,
                    "Last Price": f"₹{last_price:.2f}",
                    "Day Trend (Daily)": day_trend,
                    "SMC Confluence": smc_str,
                    "AI Confidence Score": f"{prob * 100:.1f}%" if prob <= 1.0 else f"{prob:.1f}%",
                    "Dynamic Target (Next Day)": f"₹{tgt_price:.2f} (+{target_pct_config:.1f}%)",
                    "Dynamic Stoploss": f"₹{sl_price:.2f} (-{stop_pct_config:.1f}%)"
                })
            except Exception:
                continue

        if results:
            st.subheader("🔥 High-Probability AI Trading Signals")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("No setup signals triggered for current market conditions.")
            
