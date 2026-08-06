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
# DATA ENGINE (FAST CACHED FETCHING)
# ==============================================================================
@st.cache_data(ttl=600)
def fetch_data(symbol: str, period: str = "1mo", interval: str = "5m") -> pd.DataFrame:
    try:
        ticker = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
        df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
        if df.empty: return pd.DataFrame()
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
    if trades_df.empty or len(trades_df) < 10:
        return False
    features = ["RVOL", "VWAP_Dist_Pct", "RSI", "ATR_Pct"]
    for col in features:
        if col not in trades_df.columns:
            trades_df[col] = 1.0
    X = trades_df[features].fillna(0)
    y = (trades_df["Result"] == "WIN 🎯").astype(int)
    if len(np.unique(y)) < 2:
        return False
    model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X, y)
    joblib.dump(model, INTRADAY_MODEL if mode == "intraday" else SWING_MODEL)
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
# VECTORIZED INDICATOR COMPUTATION (PRE-CALCULATED ONCE)
# ==============================================================================
def attach_vectorized_indicators(df: pd.DataFrame, ema_span: int):
    d = df.copy()
    close, high, low, vol = d['Close'], d['High'], d['Low'], d['Volume']
    
    # ATR
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    d['ATR'] = tr.rolling(14).mean().fillna(1.0)
    
    # EMA
    d['EMA'] = close.ewm(span=ema_span, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    d['RSI'] = (100 - (100 / (1 + rs))).fillna(50.0)
    
    # Cumulative VWAP
    tp = (high + low + close) / 3
    d['VWAP'] = (tp * vol).cumsum() / vol.cumsum().replace(0, 1e-9)
    
    # RVOL & Percentages
    v20 = vol.rolling(20).mean().replace(0, 1e-9)
    d['RVOL'] = (vol / v20).fillna(1.0)
    d['VWAP_Dist_Pct'] = (close - d['VWAP']).abs() / d['VWAP'] * 100
    d['ATR_Pct'] = (d['ATR'] / close) * 100
    
    return d

# ==============================================================================
# LIVE SCANNER ANALYZERS
# ==============================================================================
def run_smc_analysis(df: pd.DataFrame, timeframe_label="INTRADAY", mode="intraday"):
    if df.empty or len(df) < 30: return None
    cfg = load_config(mode)
    d = attach_vectorized_indicators(df, cfg.get("ema_span", 20))
    last = d.iloc[-1]
    
    c_live, vwap, ema, rvol = last['Close'], last['VWAP'], last['EMA'], last['RVOL']
    atr, atr_pct, rsi, vwap_dist = last['ATR'], last['ATR_Pct'], last['RSI'], last['VWAP_Dist_Pct']
    
    is_bull = c_live > vwap and c_live > ema and rvol >= cfg.get("min_rvol", 1.0)
    is_bear = c_live < vwap and c_live < ema and rvol >= cfg.get("min_rvol", 1.0)
    if not (is_bull or is_bear): return None
    
    direction = "BULLISH" if is_bull else "BEARISH"
    sl_dist = cfg.get("atr_mult", 1.2) * atr
    tp_dist = cfg.get("rr_ratio", 2.0) * sl_dist
    sl = round(c_live - sl_dist if is_bull else c_live + sl_dist, 2)
    tp = round(c_live + tp_dist if is_bull else c_live - tp_dist, 2)
    
    win_prob, trap_risk = predict_trade_prob(rvol, vwap_dist, rsi, atr_pct, mode)
    return {
        "Symbol": getattr(df, 'name', "STOCK"), "Timeframe": timeframe_label, "Direction": direction,
        "Suggested Entry": round(c_live, 2), "Stop Loss": sl, "Target Price": tp,
        "RVOL": round(rvol, 2), "RSI": round(rsi, 1), "AI Win Prob": win_prob, "Trap Risk": trap_risk, "Trade Action": "ACTIVE ENTRY"
    }

def run_momentum_analysis(df: pd.DataFrame, mode="intraday"):
    if df.empty or len(df) < 35: return None
    cfg = load_config(mode)
    d = attach_vectorized_indicators(df, cfg.get("ema_span", 20))
    last = d.iloc[-1]
    
    h20 = float(df['High'].tail(30).iloc[:-2].max())
    l20 = float(df['Low'].tail(30).iloc[:-2].min())
    c_live, vwap, atr = last['Close'], last['VWAP'], last['ATR']
    
    is_bull = c_live > vwap and c_live >= h20
    is_bear = c_live < vwap and c_live <= l20
    if not (is_bull or is_bear): return None
    
    entry = round(max(vwap, h20) if is_bull else min(vwap, l20), 2)
    sl = round(entry - (cfg.get("atr_mult", 1.2) * atr) if is_bull else entry + (cfg.get("atr_mult", 1.2) * atr), 2)
    tp = round(entry + (cfg.get("rr_ratio", 2.0) * cfg.get("atr_mult", 1.2) * atr) if is_bull else entry - (cfg.get("rr_ratio", 2.0) * cfg.get("atr_mult", 1.2) * atr), 2)
    win_prob, trap_risk = predict_trade_prob(last['RVOL'], last['VWAP_Dist_Pct'], last['RSI'], last['ATR_Pct'], mode)
    
    return {
        "Symbol": getattr(df, 'name', "STOCK"), "Direction": "🔥 BULLISH MOMENTUM" if is_bull else "🩸 BEARISH MOMENTUM",
        "Current Price": round(c_live, 2), "Suggested Entry": entry, "Stop Loss": sl, "Target Price": tp,
        "RVOL": round(last['RVOL'], 2), "R/R Ratio": f"1 : {cfg.get('rr_ratio', 2.0)}", "AI Win Prob": win_prob, "Trap Risk": trap_risk
    }

def run_meta_contrarian_analysis(df: pd.DataFrame, mode="intraday"):
    if df.empty or len(df) < 35: return None
    cfg = load_config(mode)
    d = attach_vectorized_indicators(df, cfg.get("ema_span", 20))
    last = d.iloc[-1]
    
    c_live, vwap = last['Close'], last['VWAP']
    is_bull, is_bear = c_live > vwap, c_live < vwap
    if not (is_bull or is_bear): return None
    
    score = 75.0
    flags = []
    if last['VWAP_Dist_Pct'] > 1.8: score -= 8; flags.append("⚠️ Overstretched VWAP")
    elif last['VWAP_Dist_Pct'] < 0.4: score += 5; flags.append("🟢 VWAP Anchor Pullback")
    if last['RSI'] > 70: score -= 6; flags.append("⚠️ RSI Overbought")
    elif last['RSI'] < 30: score -= 6; flags.append("⚠️ RSI Oversold")
    
    win_prob, _ = predict_trade_prob(last['RVOL'], last['VWAP_Dist_Pct'], last['RSI'], last['ATR_Pct'], mode)
    return {
        "Symbol": getattr(df, 'name', "STOCK"), "Direction": "BULLISH" if is_bull else "BEARISH",
        "Re-Ranked Score": round(score, 1), "Crowd Diagnostics": " | ".join(flags) if flags else "Optimal Setup",
        "Current Price": round(c_live, 2), "RVOL": round(last['RVOL'], 2), "RSI": round(last['RSI'], 1), "AI Win Prob": win_prob
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
# ULTRA-FAST VECTORIZED STRATEGY DISCOVERY ENGINE
# ==============================================================================
def run_fast_backtest(df: pd.DataFrame, sym: str, cfg: dict):
    trades = []
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    vwaps = df['VWAP'].values
    emas = df['EMA'].values
    rvols = df['RVOL'].values
    atrs = df['ATR'].values
    vwap_dists = df['VWAP_Dist_Pct'].values
    rsis = df['RSI'].values
    atr_pcts = df['ATR_Pct'].values
    times = df.index
    
    in_trade = False
    trade = {}
    
    for i in range(35, len(df)):
        c_p, h_p, l_p = closes[i], highs[i], lows[i]
        
        if in_trade:
            if trade["Direction"] == "BULLISH":
                if h_p >= trade["Target Price"]:
                    trade.update({"Exit Price": trade["Target Price"], "Exit Time": times[i], "Result": "WIN 🎯", "PnL %": round(((trade["Target Price"] - trade["Entry Price"]) / trade["Entry Price"]) * 100, 2)})
                    trades.append(trade); in_trade = False; continue
                elif l_p <= trade["Stop Loss"]:
                    trade.update({"Exit Price": trade["Stop Loss"], "Exit Time": times[i], "Result": "LOSS 🛑", "PnL %": round(((trade["Stop Loss"] - trade["Entry Price"]) / trade["Entry Price"]) * 100, 2)})
                    trades.append(trade); in_trade = False; continue
            elif trade["Direction"] == "BEARISH":
                if l_p <= trade["Target Price"]:
                    trade.update({"Exit Price": trade["Target Price"], "Exit Time": times[i], "Result": "WIN 🎯", "PnL %": round(((trade["Entry Price"] - trade["Target Price"]) / trade["Entry Price"]) * 100, 2)})
                    trades.append(trade); in_trade = False; continue
                elif h_p >= trade["Stop Loss"]:
                    trade.update({"Exit Price": trade["Stop Loss"], "Exit Time": times[i], "Result": "LOSS 🛑", "PnL %": round(((trade["Entry Price"] - trade["Stop Loss"]) / trade["Entry Price"]) * 100, 2)})
                    trades.append(trade); in_trade = False; continue

        if not in_trade:
            is_bull = c_p > vwaps[i] and c_p > emas[i] and rvols[i] >= cfg["min_rvol"]
            is_bear = c_p < vwaps[i] and c_p < emas[i] and rvols[i] >= cfg["min_rvol"]
            if is_bull or is_bear:
                direction = "BULLISH" if is_bull else "BEARISH"
                sl_dist = cfg["atr_mult"] * atrs[i]
                tp_dist = cfg["rr_ratio"] * sl_dist
                sl = round(c_p - sl_dist if is_bull else c_p + sl_dist, 2)
                tp = round(c_p + tp_dist if is_bull else c_p - tp_dist, 2)
                in_trade = True
                trade = {
                    "Symbol": sym, "Direction": direction, "Entry Time": times[i], "Entry Price": c_p,
                    "Stop Loss": sl, "Target Price": tp, "RVOL": round(rvols[i], 2),
                    "VWAP_Dist_Pct": round(vwap_dists[i], 2), "RSI": round(rsis[i], 1), "ATR_Pct": round(atr_pcts[i], 2)
                }
    return trades

def discover_best_strategies(tickers: list, mode="intraday"):
    period = "1mo" if mode == "intraday" else "1y"
    interval = "5m" if mode == "intraday" else "1d"
    
    st.info("Pre-loading historical market data...")
    raw_data = {}
    for sym in tickers:
        df = fetch_data(sym, period=period, interval=interval)
        if not df.empty and len(df) >= 40:
            raw_data[sym] = df

    if not raw_data:
        st.error("Unable to download data from Yahoo Finance. Please check internet connection or retry shortly.")
        return None, pd.DataFrame()

    param_grid = {
        "ema_span": [10, 20, 50],
        "atr_mult": [0.8, 1.2, 1.5, 2.0],
        "rr_ratio": [1.5, 2.0, 2.5, 3.0],
        "min_rvol": [1.0, 1.5]
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Pre-calculate indicator vectors for each EMA span
    st.info("Pre-calculating indicator vectors...")
    prepared_data = {ema: {sym: attach_vectorized_indicators(df, ema) for sym, df in raw_data.items()} for ema in [10, 20, 50]}
    
    best_cfg, best_win_rate, best_trades_df = None, 0.0, pd.DataFrame()
    progress = st.progress(0.0)
    
    for idx, cfg in enumerate(combinations):
        all_trades = []
        target_dict = prepared_data[cfg["ema_span"]]
        for sym, df in target_dict.items():
            all_trades.extend(run_fast_backtest(df, sym, cfg))
            
        progress.progress((idx + 1) / len(combinations))
        if len(all_trades) < 10: continue
        trades_df = pd.DataFrame(all_trades)
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
# STREAMLIT UI & DASHBOARD
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
    
    tf_mode = st.radio("Select Target Timeframe Engine to Optimize", ["Intraday (5m / 1-Month Lookback)", "Swing Daily (1D / 1-Year Lookback)"])
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
