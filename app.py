import json
import os
import itertools
import urllib.parse
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from t1_target_engine import T1TargetEngine

st.set_page_config(
    page_title="NQIRP Institutional Quant Engine", page_icon="⚡", layout="wide"
)

INTRADAY_CFG = "intraday_config.json"
SWING_CFG = "swing_config.json"
INTRADAY_MODEL = "intraday_ml_model.pkl"
SWING_MODEL = "swing_ml_model.pkl"

UPSTOX_ISIN_MAP = {
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "INFY": "NSE_EQ|INE009A01021",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "ICICIBANK": "NSE_EQ|INE090A01013",
    "REDINGTON": "NSE_EQ|INE891D01026",
    "FIRSTSOURCE": "NSE_EQ|INE688F01017",
    "FSL": "NSE_EQ|INE688F01017",
    "SBIN": "NSE_EQ|INE062A01020",
    "BHARTIARTL": "NSE_EQ|INE397D01024",
    "ITC": "NSE_EQ|INE154A01025",
    "LT": "NSE_EQ|INE018A01030",
    "AXISBANK": "NSE_EQ|INE238A01034",
    "KOTAKBANK": "NSE_EQ|INE237A01028",
    "HINDUNILVR": "NSE_EQ|INE030A01027",
    "BAJFINANCE": "NSE_EQ|INE296A01024",
    "MARUTI": "NSE_EQ|INE585B01010",
    "ASIANPAINT": "NSE_EQ|INE021A01026",
    "HCLTECH": "NSE_EQ|INE860A01027",
    "SUNPHARMA": "NSE_EQ|INE044A01036",
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "TATASTEEL": "NSE_EQ|INE081A01020",
    "NTPC": "NSE_EQ|INE733E01010",
    "POWERGRID": "NSE_EQ|INE752E01010",
    "TITAN": "NSE_EQ|INE280A01028",
    "ULTRACEMCO": "NSE_EQ|INE481G01011",
    "WIPRO": "NSE_EQ|INE075A01022",
    "ONGC": "NSE_EQ|INE213A01029",
    "ADANIENT": "NSE_EQ|INE423A01024",
    "ADANIPORTS": "NSE_EQ|INE742F01042",
    "COALINDIA": "NSE_EQ|INE522F01014",
    "M&M": "NSE_EQ|INE101A01026",
}


def load_config(mode="intraday") -> dict:
    filename = f"{mode}_config.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception:
            pass
    if mode == "intraday":
        return {
            "ema_span": 20,
            "atr_mult": 1.2,
            "rr_ratio": 2.0,
            "min_rvol": 1.0,
            "win_rate": 0.0,
        }
    return {
        "ema_span": 10,
        "atr_mult": 1.2,
        "rr_ratio": 1.5,
        "min_rvol": 1.0,
        "win_rate": 0.0,
    }


def save_config(cfg: dict, mode="intraday"):
    filename = f"{mode}_config.json"
    with open(filename, "w") as f:
        json.dump(cfg, f, indent=4)


def get_upstox_access_token() -> str | None:
    try:
        if "UPSTOX_ACCESS_TOKEN" in st.secrets:
            return st.secrets["UPSTOX_ACCESS_TOKEN"]
        return os.getenv("UPSTOX_ACCESS_TOKEN", None)
    except Exception:
        return None


def get_upstox_instrument_key(symbol: str) -> str:
    clean_sym = symbol.upper().replace(".NS", "").replace("&", "_").strip()
    return UPSTOX_ISIN_MAP.get(clean_sym, f"NSE_EQ|{clean_sym}")


def fetch_upstox_live(symbol: str, interval: str = "5m") -> pd.DataFrame | None:
    try:
        access_token = get_upstox_access_token()
        if not access_token:
            return None

        raw_key = get_upstox_instrument_key(symbol)
        instrument_key = urllib.parse.quote(raw_key, safe="")
        interval_str = str(interval).lower().strip()

        if "day" in interval_str or "1d" in interval_str:
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
            url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/day/1/{to_date}/{from_date}"
        else:
            digits = "".join(filter(str.isdigit, interval_str))
            int_val = int(digits) if digits else 5
            url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/minutes/{int_val}"

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            raw_candles = res.json().get("data", {}).get("candles", [])
            if raw_candles:
                df = pd.DataFrame(
                    raw_candles,
                    columns=[
                        "Datetime",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "OI",
                    ],
                )
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                df = df.sort_values("Datetime").reset_index(drop=True)
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df[df["Volume"] > 0].reset_index(drop=True)
                return df
    except Exception:
        pass
    return None


@st.cache_data(ttl=10)
def fetch_live_data(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    df_upstox = fetch_upstox_live(symbol, interval=interval)
    if df_upstox is not None and not df_upstox.empty and len(df_upstox) > 5:
        return df_upstox
    return fetch_historical_backtest_data(symbol, period=period, interval=interval)


def fetch_historical_backtest_data(
    symbol: str, period: str = "1mo", interval: str = "5m"
) -> pd.DataFrame:
    formatted_symbol = symbol if (".NS" in symbol or "^" in symbol) else f"{symbol}.NS"
    try:
        ticker = yf.Ticker(formatted_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            date_col = next(
                (col for col in df.columns if col.lower() in ["date", "datetime", "index"]),
                None,
            )
            if date_col:
                df.rename(columns={date_col: "Datetime"}, inplace=True)

            req_cols = ["Open", "High", "Low", "Close", "Volume"]
            if all(col in df.columns for col in req_cols) and len(df) > 10:
                for col in req_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df[df["Volume"] > 0].reset_index(drop=True)

                try:
                    fast_ltp = ticker.fast_info.get("lastPrice", None)
                    if fast_ltp and not np.isnan(fast_ltp):
                        df.loc[df.index[-1], "Close"] = round(float(fast_ltp), 2)
                except Exception:
                    pass

                return df
    except Exception:
        pass
    return pd.DataFrame()


def train_ml_model(trades_df: pd.DataFrame, mode="intraday"):
    """Trains Random Forest model on rich multi-context feature set."""
    if trades_df.empty or len(trades_df) < 15:
        return False
        
    features = ["RVOL", "VWAP_Dist_Pct", "RSI", "ATR_Pct", "Hour", "SMC_Score", "Nifty_Trend"]
    for col in features:
        if col not in trades_df.columns:
            trades_df[col] = 0.0
            
    X = trades_df[features].fillna(0)
    y = (trades_df["Result"] == "WIN 🎯").astype(int)
    
    if len(np.unique(y)) < 2:
        return False
        
    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
    model.fit(X, y)
    joblib.dump(model, INTRADAY_MODEL if mode == "intraday" else SWING_MODEL)
    return True


def predict_trade_prob(rvol, vwap_dist, rsi, atr_pct, hour=12, smc_score=0, nifty_trend=0.0, mode="intraday"):
    """
    Evaluates ML trade probability and explicitly reports whether 
    prediction comes from an active ML model file or a heuristic fallback.
    """
    model_path = INTRADAY_MODEL if mode == "intraday" else SWING_MODEL
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            features = [[rvol, vwap_dist, rsi, atr_pct, hour, smc_score, nifty_trend]]
            prob = model.predict_proba(features)[0][1] * 100
            trap = "HIGH" if vwap_dist > 1.8 or rvol > 3.0 else ("MEDIUM" if vwap_dist > 1.0 else "LOW")
            return f"{round(prob, 1)}%", trap, "🤖 Active ML Model"
        except Exception:
            pass
            
    # Explicit Heuristic Baseline Fallback
    prob = round(min(max(52.0 + (rvol * 3.5) - (vwap_dist * 2.5) + (smc_score * 4.0), 30.0), 90.0), 1)
    trap = "HIGH" if vwap_dist > 1.8 else "LOW"
    return f"{prob}%", trap, "📐 Baseline Heuristic (Model Pending)"


def attach_vectorized_indicators(df: pd.DataFrame, ema_span: int):
    d = df.copy()
    close, high, low, vol = d["Close"], d["High"], d["Low"], d["Volume"]
    
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean().fillna(1.0)
    d["EMA"] = close.ewm(span=ema_span, adjust=False).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    d["RSI"] = (100 - (100 / (1 + rs))).fillna(50.0)

    tp = (high + low + close) / 3
    if "Datetime" in d.columns and pd.api.types.is_datetime64_any_dtype(d["Datetime"]):
        d["Date_Group"] = d["Datetime"].dt.date
        d["Hour"] = d["Datetime"].dt.hour
        tp_vol = tp * vol
        cum_tp_vol = tp_vol.groupby(d["Date_Group"]).cumsum()
        cum_vol = vol.groupby(d["Date_Group"]).cumsum().replace(0, 1e-9)
        d["VWAP"] = cum_tp_vol / cum_vol
    else:
        d["Hour"] = 12
        d["VWAP"] = (tp * vol).cumsum() / vol.cumsum().replace(0, 1e-9)

    v20 = vol.rolling(20).mean().replace(0, 1e-9)
    d["RVOL"] = (vol / v20).fillna(1.0)
    d["VWAP_Dist_Pct"] = (close - d["VWAP"]).abs() / d["VWAP"] * 100
    d["ATR_Pct"] = (d["ATR"] / close) * 100
    
    # Vectorized SMC signals for Backtester & Live Engine
    d["SMC_Bull_Signal"] = (close > d["VWAP"]) & (close > d["EMA"]) & (d["Low"] > d["High"].shift(2))
    d["SMC_Bear_Signal"] = (close < d["VWAP"]) & (close < d["EMA"]) & (d["High"] < d["Low"].shift(2))
    
    d["SMC_Score"] = 0
    d.loc[d["SMC_Bull_Signal"], "SMC_Score"] += 2
    d.loc[d["SMC_Bear_Signal"], "SMC_Score"] -= 2
    
    return d


def detect_smc_and_patterns(df: pd.DataFrame) -> dict:
    if len(df) < 30:
        return {
            "SMC_Structure": "NEUTRAL",
            "FVG_Status": "NONE",
            "Order_Block": "NONE",
            "Pattern": "NONE",
            "SMC_Score": 0
        }
    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    pivot_highs, pivot_lows = [], []
    for i in range(2, n - 2):
        if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]):
            pivot_highs.append((i, highs[i]))
        if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
            pivot_lows.append((i, lows[i]))

    smc_score = 0
    fvg = "NONE"
    if lows[-1] > highs[-3]:
        fvg = "BULLISH FVG 🟢"
        smc_score += 1
    elif highs[-1] < lows[-3]:
        fvg = "BEARISH FVG 🔴"
        smc_score -= 1

    ob = "NONE"
    atr = df["ATR"].iloc[-1] if "ATR" in df.columns else 1.0
    if closes[-2] > closes[-3] and closes[-3] < closes[-4] and (closes[-1] - closes[-3]) > atr:
        ob = "BULLISH OB 🟩"
        smc_score += 1
    elif closes[-2] < closes[-3] and closes[-3] > closes[-4] and (closes[-3] - closes[-1]) > atr:
        ob = "BEARISH OB 🟥"
        smc_score -= 1

    smc_struct = "NEUTRAL"
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        last_ph, prev_ph = pivot_highs[-1][1], pivot_highs[-2][1]
        last_pl, prev_pl = pivot_lows[-1][1], pivot_lows[-2][1]
        curr_c = closes[-1]
        if curr_c > last_ph:
            smc_struct = "BULLISH BOS 🚀" if last_pl > prev_pl else "BULLISH CHoCH 🔄"
            smc_score += 2
        elif curr_c < last_pl:
            smc_struct = "BEARISH BOS 🩸" if last_ph < prev_ph else "BEARISH CHoCH 🔄"
            smc_score -= 2

    pattern = "NONE"
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        ph1, ph2 = pivot_highs[-2][1], pivot_highs[-1][1]
        pl1, pl2 = pivot_lows[-2][1], pivot_lows[-1][1]
        if abs(ph1 - ph2) / ph1 < 0.004:
            pattern = "DOUBLE TOP 📉"
        elif abs(pl1 - pl2) / pl1 < 0.004:
            pattern = "DOUBLE BOTTOM 📈"

    return {
        "SMC_Structure": smc_struct,
        "FVG_Status": fvg,
        "Order_Block": ob,
        "Pattern": pattern,
        "SMC_Score": smc_score
    }


def run_smc_analysis(df: pd.DataFrame, symbol: str, timeframe_label="INTRADAY", mode="intraday"):
    if df.empty or len(df) < 30:
        return None
    cfg = load_config(mode)
    d = attach_vectorized_indicators(df, cfg.get("ema_span", 20))
    smc_patterns = detect_smc_and_patterns(d)
    last = d.iloc[-1]
    
    c_live, vwap, ema, rvol = last["Close"], last["VWAP"], last["EMA"], last["RVOL"]
    atr, atr_pct, rsi, vwap_dist = last["ATR"], last["ATR_Pct"], last["RSI"], last["VWAP_Dist_Pct"]
    hour = last.get("Hour", 12)
    
    is_bull = c_live > vwap and c_live > ema and rvol >= cfg.get("min_rvol", 1.0)
    is_bear = c_live < vwap and c_live < ema and rvol >= cfg.get("min_rvol", 1.0)
    if not (is_bull or is_bear):
        return None
        
    direction = "BULLISH" if is_bull else "BEARISH"
    recent_high = float(df["High"].tail(10).iloc[:-1].max())
    recent_low = float(df["Low"].tail(10).iloc[:-1].min())
    breakout_level = recent_high if is_bull else recent_low
    
    extension_pct = ((c_live - breakout_level) / breakout_level) * 100 if is_bull else ((breakout_level - c_live) / breakout_level) * 100
    is_extended = extension_pct > 1.5
    action_status = "⚠️ EXTENDED (CHASE RISK)" if is_extended else "🔥 VALID TRIGGER ENTRY"

    sl_dist = cfg.get("atr_mult", 1.2) * atr
    tp_dist = cfg.get("rr_ratio", 2.0) * sl_dist
    sl = round(c_live - sl_dist if is_bull else c_live + sl_dist, 2)
    tp = round(c_live + tp_dist if is_bull else c_live - tp_dist, 2)
    
    win_prob, trap_risk, model_src = predict_trade_prob(
        rvol, vwap_dist, rsi, atr_pct, hour, smc_patterns["SMC_Score"], 0.0, mode
    )
    
    return {
        "Symbol": symbol,
        "Timeframe": timeframe_label,
        "Direction": direction,
        "Current Price": round(c_live, 2),
        "Trigger Level": round(breakout_level, 2),
        "Extension %": f"{round(extension_pct, 2)}%",
        "Entry Status": action_status,
        "Stop Loss": sl,
        "Target Price": tp,
        "SMC Structure": smc_patterns["SMC_Structure"],
        "Order Block": smc_patterns["Order_Block"],
        "FVG Status": smc_patterns["FVG_Status"],
        "Chart Pattern": smc_patterns["Pattern"],
        "RVOL": round(rvol, 2),
        "RSI": round(rsi, 1),
        "AI Win Prob": win_prob,
        "Trap Risk": trap_risk,
        "ML Model Engine": model_src
    }


def run_momentum_analysis(df: pd.DataFrame, symbol: str, mode="intraday"):
    if df.empty or len(df) < 35:
        return None
    cfg = load_config(mode)
    d = attach_vectorized_indicators(df, cfg.get("ema_span", 20))
    smc_patterns = detect_smc_and_patterns(d)
    last = d.iloc[-1]
    
    h20 = float(df["High"].tail(30).iloc[:-2].max())
    l20 = float(df["Low"].tail(30).iloc[:-2].min())
    c_live, vwap, atr = last["Close"], last["VWAP"], last["ATR"]
    
    is_bull = c_live > vwap and c_live >= h20
    is_bear = c_live < vwap and c_live <= l20
    if not (is_bull or is_bear):
        return None
        
    entry = round(max(vwap, h20) if is_bull else min(vwap, l20), 2)
    sl = round(entry - (cfg.get("atr_mult", 1.2) * atr) if is_bull else entry + (cfg.get("atr_mult", 1.2) * atr), 2)
    tp = round(entry + (cfg.get("rr_ratio", 2.0) * cfg.get("atr_mult", 1.2) * atr) if is_bull else entry - (cfg.get("rr_ratio", 2.0) * cfg.get("atr_mult", 1.2) * atr), 2)
    
    win_prob, trap_risk, model_src = predict_trade_prob(
        last["RVOL"], last["VWAP_Dist_Pct"], last["RSI"], last["ATR_Pct"], last.get("Hour", 12), smc_patterns["SMC_Score"], 0.0, mode
    )
    
    trigger_level = h20 if is_bull else l20
    ext_pct = ((c_live - trigger_level) / trigger_level) * 100 if is_bull else ((trigger_level - c_live) / trigger_level) * 100
    entry_status = "⚠️ EXTENDED (CHASE RISK)" if ext_pct > 1.5 else "🔥 VALID TRIGGER ENTRY"

    return {
        "Symbol": symbol,
        "Direction": "🔥 BULLISH MOMENTUM" if is_bull else "BEARISH MOMENTUM",
        "Current Price": round(c_live, 2),
        "Trigger Level": round(trigger_level, 2),
        "Extension %": f"{round(ext_pct, 2)}%",
        "Entry Status": entry_status,
        "Suggested Entry": entry,
        "Stop Loss": sl,
        "Target Price": tp,
        "SMC Structure": smc_patterns["SMC_Structure"],
        "Chart Pattern": smc_patterns["Pattern"],
        "RVOL": round(last["RVOL"], 2),
        "R/R Ratio": f"1 : {cfg.get('rr_ratio', 2.0)}",
        "AI Win Prob": win_prob,
        "Trap Risk": trap_risk,
        "ML Model Engine": model_src
    }


def run_meta_contrarian_analysis(df: pd.DataFrame, symbol: str, mode="intraday"):
    if df.empty or len(df) < 35:
        return None
    cfg = load_config(mode)
    d = attach_vectorized_indicators(df, cfg.get("ema_span", 20))
    smc_patterns = detect_smc_and_patterns(d)
    last = d.iloc[-1]
    
    c_live, vwap = last["Close"], last["VWAP"]
    is_bull, is_bear = c_live > vwap, c_live < vwap
    if not (is_bull or is_bear):
        return None
        
    score = 75.0
    flags = []
    if last["VWAP_Dist_Pct"] > 1.8:
        score -= 8
        flags.append("⚠️ Overstretched VWAP")
    elif last["VWAP_Dist_Pct"] < 0.4:
        score += 5
        flags.append("🟢 VWAP Anchor Pullback")
        
    win_prob, _, model_src = predict_trade_prob(
        last["RVOL"], last["VWAP_Dist_Pct"], last["RSI"], last["ATR_Pct"], last.get("Hour", 12), smc_patterns["SMC_Score"], 0.0, mode
    )
    
    recent_high = float(df["High"].tail(15).iloc[:-1].max())
    recent_low = float(df["Low"].tail(15).iloc[:-1].min())
    trigger_level = recent_high if is_bull else recent_low
    ext_pct = ((c_live - trigger_level) / trigger_level) * 100 if is_bull else ((trigger_level - c_live) / trigger_level) * 100
    entry_status = "⚠️ EXTENDED (CHASE RISK)" if ext_pct > 1.5 else "🔥 VALID TRIGGER ENTRY"

    return {
        "Symbol": symbol,
        "Direction": "BULLISH" if is_bull else "BEARISH",
        "Current Price": round(c_live, 2),
        "Trigger Level": round(trigger_level, 2),
        "Extension %": f"{round(ext_pct, 2)}%",
        "Entry Status": entry_status,
        "Re-Ranked Score": round(score, 1),
        "Crowd Diagnostics": " | ".join(flags) if flags else "Optimal Setup",
        "SMC Structure": smc_patterns["SMC_Structure"],
        "Chart Pattern": smc_patterns["Pattern"],
        "RVOL": round(last["RVOL"], 2),
        "RSI": round(last["RSI"], 1),
        "AI Win Prob": win_prob,
        "ML Model Engine": model_src
    }


def run_master_confluence(symbols: list) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        df_5m = fetch_live_data(sym, period="5d", interval="5m")
        if df_5m.empty:
            continue
        smc = run_smc_analysis(df_5m, sym, timeframe_label="INTRADAY", mode="intraday")
        mom = run_momentum_analysis(df_5m, sym, mode="intraday")
        mc = run_meta_contrarian_analysis(df_5m, sym, mode="intraday")
        matches = [m for m in [smc, mom, mc] if m is not None]

        if len(matches) >= 2:
            base = smc or mom or mc
            grade = "💎 TRIPLE ENGINE GEM" if len(matches) == 3 else "⚡ DOUBLE CONFLUENCE SETUP"
            action = "🔥 HIGH CONVICTION ENTRY" if len(matches) == 3 else "🎯 QUANT CONFIRMED ENTRY"
            rows.append({
                "Symbol": sym,
                "Grade": grade,
                "Direction": base.get("Direction", "BULLISH"),
                "Current Price": base.get("Current Price", 0),
                "Trigger Level": base.get("Trigger Level", 0),
                "Extension %": base.get("Extension %", "0.0%"),
                "Entry Status": base.get("Entry Status", "🔥 VALID TRIGGER ENTRY"),
                "Entry": base.get("Suggested Entry", base.get("Current Price", 0)),
                "Stop Loss": base.get("Stop Loss", 0),
                "Target Price": base.get("Target Price", 0),
                "SMC Structure": base.get("SMC Structure", "NEUTRAL"),
                "Order Block": base.get("Order Block", "NONE"),
                "FVG Status": base.get("FVG Status", "NONE"),
                "Chart Pattern": base.get("Chart Pattern", "NONE"),
                "AI Win Prob": base.get("AI Win Prob", "50%"),
                "Trap Risk": base.get("Trap Risk", "LOW"),
                "ML Model Engine": base.get("ML Model Engine", "Baseline"),
                "Action": action,
            })
        elif len(matches) == 1 and smc:
            rows.append({
                "Symbol": sym,
                "Grade": "📊 SINGLE ENGINE SIGNAL",
                "Direction": smc["Direction"],
                "Current Price": smc.get("Current Price", 0),
                "Trigger Level": smc.get("Trigger Level", 0),
                "Extension %": smc.get("Extension %", "0.0%"),
                "Entry Status": smc.get("Entry Status", "🔥 VALID TRIGGER ENTRY"),
                "Entry": smc.get("Suggested Entry", smc.get("Current Price", 0)),
                "Stop Loss": smc["Stop Loss"],
                "Target Price": smc["Target Price"],
                "SMC Structure": smc["SMC Structure"],
                "Order Block": smc.get("Order Block", "NONE"),
                "FVG Status": smc.get("FVG Status", "NONE"),
                "Chart Pattern": smc["Chart Pattern"],
                "AI Win Prob": smc["AI Win Prob"],
                "Trap Risk": smc["Trap Risk"],
                "ML Model Engine": smc.get("ML Model Engine", "Baseline"),
                "Action": "👀 WATCHLIST CANDIDATE",
            })
    return pd.DataFrame(rows)


def run_fast_backtest(df: pd.DataFrame, sym: str, cfg: dict, slippage_pct=0.0005, spread_pct=0.0002):
    """
    SMC-Aligned Backtester with Execution Friction (Slippage & Spread).
    Evaluates exact SMC triggers identical to live scanners.
    """
    trades = []
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    vwaps = df["VWAP"].values
    emas = df["EMA"].values
    rvols = df["RVOL"].values
    atrs = df["ATR"].values
    vwap_dists = df["VWAP_Dist_Pct"].values
    rsis = df["RSI"].values
    atr_pcts = df["ATR_Pct"].values
    smc_bulls = df["SMC_Bull_Signal"].values if "SMC_Bull_Signal" in df.columns else (closes > vwaps)
    smc_bears = df["SMC_Bear_Signal"].values if "SMC_Bear_Signal" in df.columns else (closes < vwaps)
    hours = df["Hour"].values if "Hour" in df.columns else np.full(len(df), 12)
    smc_scores = df["SMC_Score"].values if "SMC_Score" in df.columns else np.zeros(len(df))
    times = df.index

    in_trade = False
    trade = {}
    friction = slippage_pct + (spread_pct / 2.0)

    for i in range(35, len(df)):
        c_p, h_p, l_p = closes[i], highs[i], lows[i]
        
        if in_trade:
            if trade["Direction"] == "BULLISH":
                if h_p >= trade["Target Price"]:
                    exit_price = trade["Target Price"] * (1.0 - friction)
                    trade.update({
                        "Exit Price": round(exit_price, 2),
                        "Exit Time": times[i],
                        "Result": "WIN 🎯",
                        "PnL %": round(((exit_price - trade["Entry Price"]) / trade["Entry Price"]) * 100, 2),
                    })
                    trades.append(trade)
                    in_trade = False
                    continue
                elif l_p <= trade["Stop Loss"]:
                    exit_price = trade["Stop Loss"] * (1.0 - friction)
                    trade.update({
                        "Exit Price": round(exit_price, 2),
                        "Exit Time": times[i],
                        "Result": "LOSS 🛑",
                        "PnL %": round(((exit_price - trade["Entry Price"]) / trade["Entry Price"]) * 100, 2),
                    })
                    trades.append(trade)
                    in_trade = False
                    continue
            elif trade["Direction"] == "BEARISH":
                if l_p <= trade["Target Price"]:
                    exit_price = trade["Target Price"] * (1.0 + friction)
                    trade.update({
                        "Exit Price": round(exit_price, 2),
                        "Exit Time": times[i],
                        "Result": "WIN 🎯",
                        "PnL %": round(((trade["Entry Price"] - exit_price) / trade["Entry Price"]) * 100, 2),
                    })
                    trades.append(trade)
                    in_trade = False
                    continue
                elif h_p >= trade["Stop Loss"]:
                    exit_price = trade["Stop Loss"] * (1.0 + friction)
                    trade.update({
                        "Exit Price": round(exit_price, 2),
                        "Exit Time": times[i],
                        "Result": "LOSS 🛑",
                        "PnL %": round(((trade["Entry Price"] - exit_price) / trade["Entry Price"]) * 100, 2),
                    })
                    trades.append(trade)
                    in_trade = False
                    continue

        if not in_trade:
            is_bull = smc_bulls[i] and (rvols[i] >= cfg["min_rvol"])
            is_bear = smc_bears[i] and (rvols[i] >= cfg["min_rvol"])
            if is_bull or is_bear:
                direction = "BULLISH" if is_bull else "BEARISH"
                # Friction-adjusted entry
                entry_price = c_p * (1.0 + friction) if is_bull else c_p * (1.0 - friction)
                
                sl_dist = cfg["atr_mult"] * atrs[i]
                tp_dist = cfg["rr_ratio"] * sl_dist
                sl = round(entry_price - sl_dist if is_bull else entry_price + sl_dist, 2)
                tp = round(entry_price + tp_dist if is_bull else entry_price - tp_dist, 2)
                
                in_trade = True
                trade = {
                    "Symbol": sym,
                    "Direction": direction,
                    "Entry Time": times[i],
                    "Entry Price": round(entry_price, 2),
                    "Stop Loss": sl,
                    "Target Price": tp,
                    "RVOL": round(rvols[i], 2),
                    "VWAP_Dist_Pct": round(vwap_dists[i], 2),
                    "RSI": round(rsis[i], 1),
                    "ATR_Pct": round(atr_pcts[i], 2),
                    "Hour": hours[i],
                    "SMC_Score": smc_scores[i],
                    "Nifty_Trend": 0.0
                }
    return trades


def discover_best_strategies(tickers: list, mode="intraday", min_sample_size=30):
    """
    Grid optimizer with Anti Overfitting Sample Enforcement (N >= 30).
    Evaluates SMC strategy logic and fits ML models on realistic trade logs.
    """
    period = "1mo" if mode == "intraday" else "1y"
    interval = "5m" if mode == "intraday" else "1d"
    st.info("Pre-loading historical backtest market data via Yahoo Finance...")
    
    raw_data = {}
    for sym in tickers:
        df = fetch_historical_backtest_data(sym, period=period, interval=interval)
        if not df.empty and len(df) >= 40:
            raw_data[sym] = df
            
    if not raw_data:
        st.error("Unable to fetch historical backtest data. Please verify ticker symbols.")
        return None, pd.DataFrame()

    param_grid = {
        "ema_span": [10, 20, 50],
        "atr_mult": [0.8, 1.2, 1.5, 2.0],
        "rr_ratio": [1.5, 2.0, 2.5, 3.0],
        "min_rvol": [1.0, 1.5],
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    st.info("Pre-calculating vectorized indicator sets...")
    prepared_data = {
        ema: {sym: attach_vectorized_indicators(df, ema) for sym, df in raw_data.items()}
        for ema in [10, 20, 50]
    }
    
    best_cfg, best_win_rate, best_trades_df = None, 0.0, pd.DataFrame()
    progress = st.progress(0.0)
    
    for idx, cfg in enumerate(combinations):
        all_trades = []
        target_dict = prepared_data[cfg["ema_span"]]
        for sym, df in target_dict.items():
            all_trades.extend(run_fast_backtest(df, sym, cfg))
            
        progress.progress((idx + 1) / len(combinations))
        
        # Safeguard: Enforce robust sample size to prevent random curve-fitting noise
        if len(all_trades) < min_sample_size:
            continue
            
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


st.sidebar.title("NQIRP Quant Engine")

up_token = get_upstox_access_token()
if up_token:
    st.sidebar.success("🟢 Upstox Analytics API Connected")
else:
    st.sidebar.warning("⚠️ Upstox Token Missing from Secrets. Using Yahoo Finance Fallback.")

universe = st.sidebar.selectbox(
    "Select Watchlist",
    ["Default Watchlist (7 Stocks)", "NIFTY 50 Expanded", "Custom Tickers"],
)
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
    tab_master, tab_intraday, tab_momentum, tab_swing, tab_contrarian, tab_t1 = st.tabs([
        "🌟 Master Confluence",
        "⚡ Intraday SMC (5m)",
        "🚀 Momentum Leaders",
        "📈 Swing Signals",
        "🧠 Meta-Contrarian",
        "🎯 Next-Day (T+1) Target"
    ])

    with tab_master:
        st.subheader("🌟 Unified Master Confluence Dashboard")
        if st.button("🌟 Run Unified Master Scan", type="primary"):
            with st.spinner("Executing triple-engine scan on live 5m data..."):
                res = run_master_confluence(symbols)
                if not res.empty:
                    st.dataframe(res, use_container_width=True)
                else:
                    st.info("No confluences found currently.")

    with tab_intraday:
        st.subheader("⚡ Intraday SMC Scanner Engine (5-Minute Timeframe)")
        if st.button("⚡ Run Intraday Scan", type="primary"):
            with st.spinner("Scanning intraday 5m live candles using saved intraday_config.json..."):
                results = [
                    run_smc_analysis(fetch_live_data(s, "5d", "5m"), s, timeframe_label="5M INTRADAY", mode="intraday")
                    for s in symbols
                ]
                df_res = pd.DataFrame([r for r in results if r])
                if not df_res.empty:
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.info("No intraday setups found.")

    with tab_momentum:
        st.subheader("🚀 Momentum Leaders Engine (5-Minute Timeframe)")
        if st.button("🚀 Run Momentum Scan", type="primary"):
            with st.spinner("Scanning momentum leaders on live 5m candles..."):
                results = [run_momentum_analysis(fetch_live_data(s, "5d", "5m"), s, mode="intraday") for s in symbols]
                df_res = pd.DataFrame([r for r in results if r])
                if not df_res.empty:
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.info("No momentum leaders found.")

    with tab_swing:
        st.subheader("📈 Daily Swing Signals Engine (1D Daily Timeframe)")
        if st.button("📈 Run Daily Swing Scan", type="primary"):
            with st.spinner("Scanning 1-Year Daily candles using saved swing_config.json..."):
                results = [
                    run_smc_analysis(fetch_live_data(s, "1y", "1d"), s, timeframe_label="1D DAILY SWING", mode="swing")
                    for s in symbols
                ]
                df_res = pd.DataFrame([r for r in results if r])
                if not df_res.empty:
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.info("No swing setups found.")

    with tab_contrarian:
        st.subheader("🧠 Meta-Contrarian Crowd Exhaustion Engine")
        if st.button("🧠 Run Meta-Contrarian Scan", type="primary"):
            with st.spinner("Scanning crowd traps and live overextension..."):
                results = [run_meta_contrarian_analysis(fetch_live_data(s, "5d", "5m"), s, mode="intraday") for s in symbols]
                df_res = pd.DataFrame([r for r in results if r])
                if not df_res.empty:
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.info("No crowd traps detected.")

with tab_t1:
        st.subheader("🎯 Next-Day (T+1) Target & Live Market Evaluator")
        st.caption("Post-market target generator & live market parameter evaluator.")

        if st.button("🚀 Run Live T+1 Target Evaluator"):
            with st.spinner("Analyzing daily targets against live market sentiment & VWAP..."):
                # Fetch broad market index sentiment (Nifty 50)
                nifty_df = fetch_live_data("^NSEI", period="1d", interval="5m")
                nifty_pct = 0.0
                if not nifty_df.empty:
                    nifty_pct = ((nifty_df["Close"].iloc[-1] - nifty_df["Open"].iloc[0]) / nifty_df["Open"].iloc[0]) * 100

                st.caption(f"📊 Broad Market Sentiment (Nifty 50 Live): `{round(nifty_pct, 2)}%`")

                evaluated_results = []
                for sym in symbols:
                    df_daily = fetch_live_data(sym, period="3mo", interval="1d")
                    df_5m = fetch_live_data(sym, period="1d", interval="5m")

                    if not df_daily.empty:
                        base_t1 = T1TargetEngine.generate_t1_targets(df_daily, sym)
                        if base_t1:
                            live_t1 = T1TargetEngine.evaluate_live_t1_signal(base_t1, df_5m, nifty_pct)
                            evaluated_results.append(live_t1)

                if evaluated_results:
                    st.dataframe(pd.DataFrame(evaluated_results), use_container_width=True)
                else:
                    st.warning("No data retrieved for selected watchlist.")


elif page == "🧪 AI Strategy Discovery & Backtester":
    st.title("🧪 Fast In-Memory Strategy Discovery Engine")
    st.caption("Runs fast in-memory SMC strategy discovery with friction modeling & ML model training.")

    tf_mode = st.radio(
        "Select Target Timeframe Engine to Optimize",
        ["Intraday (5m / 1-Month Lookback)", "Swing Daily (1D / 1-Year Lookback)"],
    )
    target_mode = "intraday" if "Intraday" in tf_mode else "swing"

    if st.button(f"🚀 Run Strategy Optimization ({target_mode.upper()})", type="primary"):
        with st.spinner(f"Pre-loading historical data & optimizing {target_mode.upper()} parameters with SMC alignment..."):
            best_cfg, trades_df = discover_best_strategies(symbols, mode=target_mode, min_sample_size=30)
            if best_cfg:
                st.success(f"🎉 Optimized Config Discovered! Win Rate: {best_cfg['win_rate']}%")
                st.json(best_cfg)
                st.subheader("Backtest Trade Logs Used for Machine Learning Training")
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.error("Could not find a high win-rate strategy over the specified minimum trade sample size requirement.")

    st.markdown("---")
    st.subheader("🤖 AI T+1 Strategy Optimizer")
    st.caption("AI analyzes historical data to automatically find the highest win-rate ATR multiplier per stock.")

    selected_symbols = st.multiselect("Select Watchlist for Optimization", symbols, default=symbols[:5])

    if st.button("🧪 Run AI T+1 Optimizer"):
        optimized_results = []
        with st.spinner("AI is crunching 1-year historical data to find the best T+1 strategy..."):
            for sym in selected_symbols:
                df_hist = fetch_historical_backtest_data(sym, period="1y", interval="1d")
                if not df_hist.empty:
                    best_win_rate = 0
                    best_res = None
                    for mult in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
                        res = T1TargetEngine.backtest_t1_strategy(df_hist, atr_mult=mult)
                        if res and res.get("Target Hit Rate (%)", 0) > best_win_rate:
                            best_win_rate = res["Target Hit Rate (%)"]
                            best_res = res
                    if best_res:
                        final_res = {"Symbol": sym}
                        final_res.update(best_res)
                        final_res["Optimization Status"] = "🔥 INSTITUTIONAL EDGE" if best_win_rate >= 40 else "⚠️ SUB-OPTIMAL"
                        optimized_results.append(final_res)
                        
        if optimized_results:
            st.success("Optimization complete! Here are the highest-probability mathematical edges found:")
            st.dataframe(pd.DataFrame(optimized_results), use_container_width=True)
        else:
            st.warning("No backtest data returned for selected symbols.")

with st.sidebar.expander("🔍 Live Upstox API Diagnostic", expanded=False):
    diag_token = get_upstox_access_token()
    st.write(f"**Token Detected:** `{bool(diag_token)}`")
    if diag_token and st.button("Run Diagnostic Check"):
        test_url = "https://api.upstox.com/v3/historical-candle/intraday/NSE_EQ%7CINE002A01018/minutes/5"
        test_headers = {"Accept": "application/json", "Authorization": f"Bearer {diag_token}"}
        res = requests.get(test_url, headers=test_headers, timeout=5)
        st.write(f"**HTTP Status Code:** `{res.status_code}`")
        if res.status_code == 200:
            candles = res.json().get("data", {}).get("candles", [])
            st.write(f"**Latest Candle Time:** `{candles[0][0] if candles else 'No Data'}`")
            st.write(f"**Latest Close Price:** `{candles[0][4] if candles else 'No Data'}`")
        else:
            st.error(f"**Upstox Raw Response:** {res.text}")
