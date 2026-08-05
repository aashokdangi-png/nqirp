import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import json
import itertools
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="NQIRP Institutional Quant Engine", page_icon="⚡", layout="wide")

INTRADAY_CFG = "intraday_config.json"
SWING_CFG = "swing_config.json"
INTRADAY_MODEL = "intraday_ml_model.pkl"
SWING_MODEL = "swing_ml_model.pkl"

# ==============================================================================
# DATA ENGINE WITH HIGH-EFFICIENCY CACHING
# ==============================================================================
@st.cache_data(ttl=300)
def fetch_data(symbol: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    try:
        ticker = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def load_config(mode="intraday"):
    filepath = INTRADAY_CFG if mode == "intraday" else SWING_CFG
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {"ema_span": 20, "atr_mult": 1.2, "rr_ratio": 2.0, "min_rvol": 1.0, "win_rate": 0.0} if mode == "intraday" else {"ema_span": 50, "atr_mult": 2.0, "rr_ratio": 2.5, "min_rvol": 1.0, "win_rate": 0.0}

def save_config(config, mode="intraday"):
    filepath = INTRADAY_CFG if mode == "intraday" else SWING_CFG
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)

# ==============================================================================
# MACHINE LEARNING ENGINE
# ==============================================================================
def train_ml_model(trades_df: pd.DataFrame, mode="intraday"):
    if trades_df.empty or len(trades_df) < 15:
        return False
    features = ["RVOL", "VWAP_Dist_Pct", "RSI", "ATR_Pct"]
    for col in features:
        if col not in trades_df.columns:
            trades_df[col] = 1.0
    X = trades_df[features].fillna(0)
    y = (trades_df["Result"] == "WIN 🎯").astype(int)
    if len(np.unique(y)) < 2:
        return False
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    model_path = INTRADAY_MODEL if mode == "intraday" else SWING_MODEL
    joblib.dump(model, model_path)
    return True

def predict_trade_prob(rvol, vwap_dist, rsi, atr_pct, mode="intraday"):
    model_path = INTRADAY_MODEL if mode == "intraday" else SWING_MODEL
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            prob = model.predict_proba([[rvol, vwap_dist, rsi, atr_pct]])[0][1] * 100
            trap = "HIGH" if vwap_dist > 1.8 or rvol > 3.0 else ("MEDIUM" if vwap_dist > 1.0 else "LOW")
            return f"{round(prob, 1)}%", trap
        except Exception:
            pass
    prob = round(min(max(55.0 + (rvol * 4) - (vwap_dist * 3), 35.0), 92.0), 1)
    trap = "HIGH" if vwap_dist > 1.8 else "LOW"
    return f"{prob}%", trap

# ==============================================================================
# INDICATORS CALCULATOR
# ==============================================================================
def calculate_indicators(df: pd.DataFrame, cfg: dict):
    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    if atr <= 0 or np.isnan(atr): atr = 1.0
    ema = float(close.ewm(span=cfg.get("ema_span", 20)).mean().iloc[-1])
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = float((100 - (100 / (1 + rs))).dropna().iloc[-1]) if not rs.empty else 50.0
    
    today_date = close.index[-1].date() if hasattr(close.index[-1], 'date') else None
    if today_date and len(df[df.index.date == today_date]) > 0:
        today_df = df[df.index.date == today_date]
        vwap = float((today_df['Volume'] * (today_df['High'] + today_df['Low'] + today_df['Close']) / 3).sum() / today_df['Volume'].sum())
    else:
        vwap = float((vol * (high + low + close) / 3).cumsum().iloc[-1] / vol.cumsum().iloc[-1])
        
    v20 = float(vol.tail(20).mean())
    rvol = float(vol.iloc[-1] / v20) if v20 > 0 else 1.0
    c_live = float(close.iloc[-1])
    vwap_dist = abs(c_live - vwap) / vwap * 100
    atr_pct = (atr / c_live) * 100
    
    return {"c_live": c_live, "atr": atr, "ema": ema, "vwap": vwap, "rsi": rsi, "rvol": rvol, "vwap_dist": vwap_dist, "atr_pct": atr_pct}

# ==============================================================================
# LIVE SCANNER ENGINES (USES LOADED PERMANENT CONFIGS)
# ==============================================================================
def run_smc_analysis(df: pd.DataFrame, timeframe_label="INTRADAY", mode="intraday"):
    if df.empty or len(df) < 30: return None
    cfg = load_config(mode)
    ind = calculate_indicators(df, cfg)
    
    is_bull = ind["c_live"] > ind["vwap"] and ind["c_live"] > ind["ema"] and ind["rvol"] >= cfg.get("min_rvol", 1.0)
    is_bear = ind["c_live"] < ind["vwap"] and ind["c_live"] < ind["ema"] and ind["rvol"] >= cfg.get("min_rvol", 1.0)
    if not (is_bull or is_bear): return None
    
    direction = "BULLISH" if is_bull else "BEARISH"
    sl_dist = cfg.get("atr_mult", 1.2) * ind["atr"]
    tp_dist = cfg.get("rr_ratio", 2.0) * sl_dist
    sl = round(ind["c_live"] - sl_dist if is_bull else ind["c_live"] + sl_dist, 2)
    tp = round(ind["c_live"] + tp_dist if is_bull else ind["c_live"] - tp_dist, 2)
    
    win_prob, trap_risk = predict_trade_prob(ind["rvol"], ind["vwap_dist"], ind["rsi"], ind["atr_pct"], mode)
    return {
        "Symbol": getattr(df, 'name', "STOCK"), "Timeframe": timeframe_label, "Direction": direction,
        "Suggested Entry": round(ind["c_live"], 2), "Stop Loss": sl, "Target Price": tp,
        "RVOL": round(ind["rvol"], 2), "RSI": round(ind["rsi"], 1), "AI Win Prob": win_prob, "Trap Risk": trap_risk, "Trade Action": "ACTIVE ENTRY"
    }

def run_momentum_analysis(df: pd.DataFrame, mode="intraday"):
    if df.empty or len(df) < 35: return None
    cfg = load_config(mode)
    ind = calculate_indicators(df, cfg)
    h20 = float(df['High'].tail(30).iloc[:-2].max())
    l20 = float(df['Low'].tail(30).iloc[:-2].min())
    
    is_bull = ind["c_live"] > ind["vwap"] and ind["c_live"] >= h20
    is_bear = ind["c_live"] < ind["vwap"] and ind["c_live"] <= l20
    if not (is_bull or is_bear): return None
    
    entry = round(max(ind["vwap"], h20) if is_bull else min(ind["vwap"], l20), 2)
    sl = round(entry - (cfg.get("atr_mult", 1.2) * ind["atr"]) if is_bull else entry + (cfg.get("atr_mult", 1.2) * ind["atr"]), 2)
    tp = round(entry + (cfg.get("rr_ratio", 2.0) * cfg.get("atr_mult", 1.2) * ind["atr"]) if is_bull else entry - (cfg.get("rr_ratio", 2.0) * cfg.get("atr_mult", 1.2) * ind["atr"]), 2)
    win_prob, trap_risk = predict_trade_prob(ind["rvol"], ind["vwap_dist"], ind["rsi"], ind["atr_pct"], mode)
    
    return {
        "Symbol": getattr(df, 'name', "STOCK"), "Direction": "🔥 BULLISH MOMENTUM" if is_bull else "🩸 BEARISH MOMENTUM",
        "Current Price": round(ind["c_live"], 2), "Suggested Entry": entry, "Stop Loss": sl, "Target Price": tp,
        "RVOL": round(ind["rvol"], 2), "R/R Ratio": f"1 : {cfg.get('rr_ratio', 2.0)}", "AI Win Prob": win_prob, "Trap Risk": trap_risk
    }

def run_meta_contrarian_analysis(df: pd.DataFrame, mode="intraday"):
    if df.empty or len(df) < 35: return None
    cfg = load_config(mode)
    ind = calculate_indicators(df, cfg)
    is_bull = ind["c_live"] > ind["vwap"]
    is_bear = ind["c_live"] < ind["vwap"]
    if not (is_bull or is_bear): return None
    
    score = 75.0
    flags = []
    if ind["vwap_dist"] > 1.8: score -= 8; flags.append("⚠️ Overstretched VWAP")
    elif ind["vwap_dist"] < 0.4: score += 5; flags.append("🟢 VWAP Anchor Pullback")
    if ind["rsi"] > 70: score -= 6; flags.append("⚠️ RSI Overbought")
    elif ind["rsi"] < 30: score -= 6; flags.append("⚠️ RSI Oversold")
    
    win_prob, _ = predict_trade_prob(ind["rvol"], ind["vwap_dist"], ind["rsi"], ind["atr_pct"], mode)
    return {
        "Symbol": getattr(df, 'name', "STOCK"), "Direction": "BULLISH" if is_bull else "BEARISH",
        "Re-Ranked Score": round(score, 1), "Crowd Diagnostics": " | ".join(flags) if flags else "Optimal Setup",
        "Current Price": round(ind["c_live"], 2), "RVOL": round(ind["rvol"], 2), "RSI": round(ind["rsi"], 1), "AI Win Prob": win_prob
    }

def run_master_confluence(symbols: list) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        df_5m = fetch_data(sym, period="5d", interval="5m")
        if df_5m.empty: continue
        df_5m.name = sym
        smc = run_smc_analysis(df_5m, timeframe_label="INTRADAY", mode="intraday")
        mom = run_momentum_analysis(df_5m, mode="intraday")
        mc = run_meta_contrarian_analysis(df_5m, mode="intraday")
        if smc and mom and mc:
            rows.append({
                "Symbol": sym, "Grade": "💎 TRIPLE ENGINE GEM", "Direction": smc["Direction"],
                "Entry": smc["Suggested Entry"], "Stop Loss": smc["Stop Loss"], "Target Price": smc["Target Price"],
                "AI Win Prob": smc["AI Win Prob"], "Trap Risk": smc["Trap Risk"], "Action": "🔥 HIGH CONVICTION ENTRY"
            })
    return pd.DataFrame(rows)

# ==============================================================================
# FAST IN-MEMORY BACKTESTER & STRATEGY DISCOVERY ENGINE
# ==============================================================================
def run_parameterized_backtest_cached(data_dict: dict, cfg: dict):
    all_trades = []
    for sym, df in data_dict.items():
        if df.empty or len(df) < 40: continue
        close, high, low = df['Close'], df['High'], df['Low']
        in_trade, current_trade = False, None
        
        for i in range(35, len(df)):
            curr_time = df.index[i]
            h_p, l_p, c_p = float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
            if in_trade and current_trade:
                if current_trade["Direction"] == "BULLISH":
                    if h_p >= current_trade["Target Price"]:
                        current_trade.update({"Exit Price": current_trade["Target Price"], "Exit Time": curr_time, "Result": "WIN 🎯", "PnL %": round(((current_trade["Target Price"] - current_trade["Entry Price"]) / current_trade["Entry Price"]) * 100, 2)})
                        all_trades.append(current_trade); in_trade, current_trade = False, None; continue
                    elif l_p <= current_trade["Stop Loss"]:
                        current_trade.update({"Exit Price": current_trade["Stop Loss"], "Exit Time": curr_time, "Result": "LOSS 🛑", "PnL %": round(((current_trade["Stop Loss"] - current_trade["Entry Price"]) / current_trade["Entry Price"]) * 100, 2)})
                        all_trades.append(current_trade); in_trade, current_trade = False, None; continue
                elif current_trade["Direction"] == "BEARISH":
                    if l_p <= current_trade["Target Price"]:
                        current_trade.update({"Exit Price": current_trade["Target Price"], "Exit Time": curr_time, "Result": "WIN 🎯", "PnL %": round(((current_trade["Entry Price"] - current_trade["Target Price"]) / current_trade["Entry Price"]) * 100, 2)})
                        all_trades.append(current_trade); in_trade, current_trade = False, None; continue
                    elif h_p >= current_trade["Stop Loss"]:
                        current_trade.update({"Exit Price": current_trade["Stop Loss"], "Exit Time": curr_time, "Result": "LOSS 🛑", "PnL %": round(((current_trade["Entry Price"] - current_trade["Stop Loss"]) / current_trade["Entry Price"]) * 100, 2)})
                        all_trades.append(current_trade); in_trade, current_trade = False, None; continue

            if not in_trade:
                sub_df = df.iloc[:i+1]
                ind = calculate_indicators(sub_df, cfg)
                is_bull = c_p > ind["vwap"] and c_p > ind["ema"] and ind["rvol"] >= cfg["min_rvol"]
                is_bear = c_p < ind["vwap"] and c_p < ind["ema"] and ind["rvol"] >= cfg["min_rvol"]
                if is_bull or is_bear:
                    direction = "BULLISH" if is_bull else "BEARISH"
                    sl_dist = cfg["atr_mult"] * ind["atr"]
                    tp_dist = cfg["rr_ratio"] * sl_dist
                    sl = round(c_p - sl_dist if is_bull else c_p + sl_dist, 2)
                    tp = round(c_p + tp_dist if is_bull else c_p - tp_dist, 2)
                    in_trade = True
                    current_trade = {
                        "Symbol": sym, "Direction": direction, "Entry Time": curr_time, "Entry Price": c_p,
                        "Stop Loss": sl, "Target Price": tp, "RVOL": round(ind["rvol"], 2),
                        "VWAP_Dist_Pct": round(ind["vwap_dist"], 2), "RSI": round(ind["rsi"], 1), "ATR_Pct": round(ind["atr_pct"], 2)
                    }
    return pd.DataFrame(all_trades)

def discover_best_strategies(tickers: list, mode="intraday"):
    period = "60d" if mode == "intraday" else "1y"
    interval = "5m" if mode == "intraday" else "1d"
    
    # 1. Pre-fetch historical data ONCE in-memory to prevent yfinance rate limits
    st.info("Pre-loading historical data into memory...")
    data_dict = {}
    for sym in tickers:
        df = fetch_data(sym, period=period, interval=interval)
        if not df.empty and len(df) >= 40:
            data_dict[sym] = df

    if not data_dict:
        return None, pd.DataFrame()

    param_grid = {
        "ema_span": [10, 20, 50],
        "atr_mult": [0.8, 1.2, 1.5, 2.0],
        "rr_ratio": [1.5, 2.0, 2.5, 3.0],
        "min_rvol": [1.0, 1.5]
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_cfg, best_win_rate, best_trades_df = None, 0.0, pd.DataFrame()
    
    # 2. Iterate offline over pre-fetched data
    progress = st.progress(0.0)
    for idx, cfg in enumerate(combinations):
        trades_df = run_parameterized_backtest_cached(data_dict, cfg)
        progress.progress((idx + 1) / len(combinations))
        if len(trades_df) < 15: continue
        win_rate = (sum(trades_df["Result"] == "WIN 🎯") / len(trades_df)) * 100
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_cfg = cfg
            best_cfg["win_rate"] = round(win_rate, 2)
            best_trades_df = trades_df

    progress.empty()
    if best_cfg:
        save_config(best_cfg, mode=mode)
        train_ml_model(best_trades_df, mode=mode)
    return best_cfg, best_trades_df

# ==============================================================================
# UI NAVIGATION & CONTROLS
# ==============================================================================
st.sidebar.title("NQIRP Quant Engine")
universe = st.sidebar.selectbox("Select Watchlist", ["Default Watchlist (7 Stocks)", "NIFTY 50 Expanded", "Custom Tickers"])
if universe == "Default Watchlist (7 Stocks)":
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "REDINGTON", "FIRSTSOURCE"]
elif universe == "NIFTY 50 Expanded":
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "LT", "AXISBANK"]
else:
    custom_in = st.sidebar.text_input("Enter Tickers (comma separated)", "RELIANCE, TCS, INFY")
    symbols = [s.strip().upper() for s in custom_in.split(",") if s.strip()]

intra_cfg = load_config("intraday")
swing_cfg = load_config("swing")
st.sidebar.markdown(f"**Intraday Config:** Win Rate `{intra_cfg.get('win_rate', 'N/A')}%` | R/R `1:{intra_cfg.get('rr_ratio', 2.0)}`")
st.sidebar.markdown(f"**Swing Config:** Win Rate `{swing_cfg.get('win_rate', 'N/A')}%` | R/R `1:{swing_cfg.get('rr_ratio', 2.5)}`")

page = st.sidebar.radio("Select Module", ["⚡ Multi-Tab Live Scanner", "🧪 AI Strategy Discovery & Backtester"])

if page == "⚡ Multi-Tab Live Scanner":
    st.title("⚡ Institutional Multi-Timeframe Scanner Engine")
    tab_master, tab_intraday, tab_momentum, tab_swing, tab_contrarian = st.tabs([
        "🌟 Master Confluence", "⚡ Intraday SMC (5m)", "🚀 Momentum Leaders (5m)", "📈 Swing Signals (1D Daily)", "🧠 Meta-Contrarian Engine"
    ])
    
    with tab_master:
        st.subheader("🌟 Unified Master Confluence Dashboard")
        if st.button("🌟 Run Unified Master Scan", type="primary"):
            with st.spinner("Executing triple-engine scan on 5m data..."):
                res = run_master_confluence(symbols)
                st.dataframe(res, use_container_width=True) if not res.empty else st.info("No confluences found currently.")
                
    with tab_intraday:
        st.subheader("⚡ Intraday SMC Scanner Engine (5-Minute Timeframe)")
        if st.button("⚡ Run Intraday Scan", type="primary"):
            with st.spinner("Scanning intraday 5m bars using saved intraday_config.json..."):
                results = [run_smc_analysis(fetch_data(s, "5d", "5m"), timeframe_label="5M INTRADAY", mode="intraday") for s in symbols]
                df_res = pd.DataFrame([r for r in results if r])
                st.dataframe(df_res, use_container_width=True) if not df_res.empty else st.info("No intraday setups found.")

    with tab_momentum:
        st.subheader("🚀 Momentum Leaders Engine (5-Minute Timeframe)")
        if st.button("🚀 Run Momentum Scan", type="primary"):
            with st.spinner("Scanning momentum leaders on 5m data..."):
                results = [run_momentum_analysis(fetch_data(s, "5d", "5m"), mode="intraday") for s in symbols]
                df_res = pd.DataFrame([r for r in results if r])
                st.dataframe(df_res, use_container_width=True) if not df_res.empty else st.info("No momentum leaders found.")

    with tab_swing:
        st.subheader("📈 Daily Swing Signals Engine (1D Daily Timeframe)")
        if st.button("📈 Run Daily Swing Scan", type="primary"):
            with st.spinner("Scanning 1-Year Daily candles using saved swing_config.json..."):
                results = [run_smc_analysis(fetch_data(s, "1y", "1d"), timeframe_label="1D DAILY SWING", mode="swing") for s in symbols]
                df_res = pd.DataFrame([r for r in results if r])
                st.dataframe(df_res, use_container_width=True) if not df_res.empty else st.info("No swing setups found.")

    with tab_contrarian:
        st.subheader("🧠 Meta-Contrarian Crowd Exhaustion Engine")
        if st.button("🧠 Run Meta-Contrarian Scan", type="primary"):
            with st.spinner("Scanning crowd traps and overextension..."):
                results = [run_meta_contrarian_analysis(fetch_data(s, "5d", "5m"), mode="intraday") for s in symbols]
                df_res = pd.DataFrame([r for r in results if r])
                st.dataframe(df_res, use_container_width=True) if not df_res.empty else st.info("No crowd traps detected.")

elif page == "🧪 AI Strategy Discovery & Backtester":
    st.title("🧪 Fast In-Memory Strategy Discovery Engine")
    st.caption("Runs fast in-memory strategy discovery, saves optimal parameters to JSON, and trains ML models.")
    
    tf_mode = st.radio("Select Target Timeframe Engine to Optimize", ["Intraday (5m / 60-Day Lookback)", "Swing Daily (1D / 1-Year Lookback)"])
    target_mode = "intraday" if "Intraday" in tf_mode else "swing"
    
    if st.button(f"🚀 Run Strategy Optimization ({target_mode.upper()})", type="primary"):
        with st.spinner(f"Pre-loading data & optimizing {target_mode.upper()} parameters..."):
            best_cfg, trades_df = discover_best_strategies(symbols, mode=target_mode)
            if best_cfg:
                st.success(f"🎉 Optimized Config Discovered! Win Rate: {best_cfg['win_rate']}%")
                st.json(best_cfg)
                st.subheader("Backtest Trade Logs Used for Machine Learning Training")
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.error("Could not find a high win-rate strategy permutation over the specified sample size.")
