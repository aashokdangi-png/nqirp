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
    "INFY": "NSE_EQ|INE090A01021",
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
    """STEP 2 IMPLEMENTATION: Fetches token from secrets and strips whitespace."""
    for key in ["token", "UPSTOX_ANALYTICS_TOKEN", "UPSTOX_ACCESS_TOKEN"]:
        if key in st.secrets:
            val = st.secrets[key]
            if isinstance(val, str) and val.strip():
                return val.strip().replace('"', "").replace("'", "")
    return None


def get_upstox_instrument_key(symbol: str) -> str:
    clean_sym = symbol.replace(".NS", "").upper().strip()
    return UPSTOX_ISIN_MAP.get(clean_sym, f"NSE_EQ|{clean_sym}")


def fetch_upstox_live(symbol: str, interval: str = "5m") -> pd.DataFrame | None:
    try:
        access_token = get_upstox_access_token()
        if not access_token:
            return None

        instrument_key = get_upstox_instrument_key(symbol)
        encoded_key = urllib.parse.quote(instrument_key, safe="")

        if interval in ["day", "1d", "daily"]:
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/day/{to_date}/{from_date}"
        else:
            # Query 1minute candles (valid Upstox v2 interval)
            url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_key}/1minute"

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

                # Resample 1m Upstox data into 5m candles for scanner calculations
                if interval == "5m" and not df.empty:
                    df.set_index("Datetime", inplace=True)
                    df_5m = (
                        df.resample("5min")
                        .agg({
                            "Open": "first",
                            "High": "max",
                            "Low": "min",
                            "Close": "last",
                            "Volume": "sum",
                        })
                        .dropna()
                        .reset_index()
                    )
                    return df_5m

                return df
    except Exception:
        pass
    return None


@st.cache_data(ttl=10)
def fetch_live_data(
    symbol: str, period: str = "5d", interval: str = "5m"
) -> pd.DataFrame:
    df_upstox = fetch_upstox_live(symbol, interval=interval)
    if df_upstox is not None and not df_upstox.empty and len(df_upstox) > 15:
        return df_upstox

    return fetch_historical_backtest_data(symbol, period=period, interval=interval)


def fetch_historical_backtest_data(
    symbol: str, period: str = "1mo", interval: str = "5m"
) -> pd.DataFrame:
    formatted_symbol = symbol if (".NS" in symbol or "^" in symbol) else f"{symbol}.NS"
    try:
        ticker = yf.Ticker(formatted_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=False)
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
            trap = (
                "HIGH"
                if vwap_dist > 1.8 or rvol > 3.0
                else ("MEDIUM" if vwap_dist > 1.0 else "LOW")
            )
            return f"{round(prob, 1)}%", trap
        except Exception:
            pass
    prob = round(min(max(55.0 + (rvol * 4) - (vwap_dist * 3), 35.0), 92.0), 1)
    trap = "HIGH" if vwap_dist > 1.8 else "LOW"
    return f"{prob}%", trap


def attach_vectorized_indicators(df: pd.DataFrame, ema_span: int):
    d = df.copy()
    close, high, low, vol = d["Close"], d["High"], d["Low"], d["Volume"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["ATR"] = tr.rolling(14).mean().fillna(1.0)
    d["EMA"] = close.ewm(span=ema_span, adjust=False).mean()
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    d["RSI"] = (100 - (100 / (1 + rs))).fillna(50.0)

    tp = (high + low + close) / 3
    if "Datetime" in d.columns:
        d["Date_Group"] = pd.to_datetime(d["Datetime"]).dt.date
        tp_vol = tp * vol
        cum_tp_vol = tp_vol.groupby(d["Date_Group"]).cumsum()
        cum_vol = vol.groupby(d["Date_Group"]).cumsum().replace(0, 1e-9)
        d["VWAP"] = cum_tp_vol / cum_vol
    else:
        d["VWAP"] = (tp * vol).cumsum() / vol.cumsum().replace(0, 1e-9)

    v20 = vol.rolling(20).mean().replace(0, 1e-9)
    d["RVOL"] = (vol / v20).fillna(1.0)
    d["VWAP_Dist_Pct"] = (close - d["VWAP"]).abs() / d["VWAP"] * 100
    d["ATR_Pct"] = (d["ATR"] / close) * 100
    return d


def detect_smc_and_patterns(df: pd.DataFrame) -> dict:
    if len(df) < 30:
        return {
            "SMC_Structure": "NEUTRAL",
            "FVG_Status": "NONE",
            "Order_Block": "NONE",
            "Pattern": "NONE",
        }
    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    pivot_highs, pivot_lows = [], []
    for i in range(2, n - 2):
        if (
            highs[i] > highs[i - 1]
            and highs[i] > highs[i - 2]
            and highs[i] > highs[i + 1]
            and highs[i] > highs[i + 2]
        ):
            pivot_highs.append((i, highs[i]))
        if (
            lows[i] < lows[i - 1]
            and lows[i] < lows[i - 2]
            and lows[i] < lows[i + 1]
            and lows[i] < lows[i + 2]
        ):
            pivot_lows.append((i, lows[i]))

    fvg = "NONE"
    if lows[-1] > highs[-3]:
        fvg = "BULLISH FVG 🟢"
    elif highs[-1] < lows[-3]:
        fvg = "BEARISH FVG 🔴"

    ob = "NONE"
    atr = df["ATR"].iloc[-1] if "ATR" in df.columns else 1.0
    if (
        closes[-2] > closes[-3]
        and closes[-3] < closes[-4]
        and (closes[-1] - closes[-3]) > atr
    ):
        ob = "BULLISH OB 🟩"
    elif (
        closes[-2] < closes[-3]
        and closes[-3] > closes[-4]
        and (closes[-3] - closes[-1]) > atr
    ):
        ob = "BEARISH OB 🟥"

    smc_struct = "NEUTRAL"
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        last_ph, prev_ph = pivot_highs[-1][1], pivot_highs[-2][1]
        last_pl, prev_pl = pivot_lows[-1][1], pivot_lows[-2][1]
        curr_c = closes[-1]
        if curr_c > last_ph:
            smc_struct = (
                "BULLISH BOS 🚀" if last_pl > prev_pl else "BULLISH CHoCH 🔄"
            )
        elif curr_c < last_pl:
            smc_struct = (
                "BEARISH BOS 🩸" if last_ph < prev_ph else "BEARISH CHoCH 🔄"
            )

    pattern = "NONE"
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        ph1, ph2 = pivot_highs[-2][1], pivot_highs[-1][1]
        pl1, pl2 = pivot_lows[-2][1], pivot_lows[-1][1]
        if abs(ph1 - ph2) / ph1 < 0.004:
            pattern = "DOUBLE TOP 📉"
        elif abs(pl1 - pl2) / pl1 < 0.004:
            pattern = "DOUBLE BOTTOM 📈"
        elif ph2 < ph1 and pl2 > pl1:
            pattern = "SYMMETRICAL TRIANGLE 📐"
        elif abs(ph1 - ph2) / ph1 < 0.004 and pl2 > pl1:
            pattern = "ASCENDING TRIANGLE 📐"
        elif ph2 < ph1 and abs(pl1 - pl2) / pl1 < 0.004:
            pattern = "DESCENDING TRIANGLE 📐"
        elif len(pivot_lows) >= 3:
            pl0 = pivot_lows[-3][1]
            if pl1 < pl0 and pl1 < pl2 and closes[-1] > pl2:
                pattern = "CUP & HANDLE ☕"
    return {
        "SMC_Structure": smc_struct,
        "FVG_Status": fvg,
        "Order_Block": ob,
        "Pattern": pattern,
    }


def run_smc_analysis(
    df: pd.DataFrame, symbol: str, timeframe_label="INTRADAY", mode="intraday"
):
    if df.empty or len(df) < 30:
        return None
    cfg = load_config(mode)
    d = attach_vectorized_indicators(df, cfg.get("ema_span", 20))
    smc_patterns = detect_smc_and_patterns(d)
    last = d.iloc[-1]
    c_live, vwap, ema, rvol = (
        last["Close"],
        last["VWAP"],
        last["EMA"],
        last["RVOL"],
    )
    atr, atr_pct, rsi, vwap_dist = (
        last["ATR"],
        last["ATR_Pct"],
        last["RSI"],
        last["VWAP_Dist_Pct"],
    )
    is_bull = c_live > vwap and c_live > ema and rvol >= cfg.get("min_rvol", 1.0)
    is_bear = c_live < vwap and c_live < ema and rvol >= cfg.get("min_rvol", 1.0)
    if not (is_bull or is_bear):
        return None
        
    direction = "BULLISH" if is_bull else "BEARISH"
    
    # --- TRUE STRUCTURAL BREAKOUT / TRIGGER POINT ---
    recent_high = float(df["High"].tail(10).iloc[:-1].max())
    recent_low = float(df["Low"].tail(10).iloc[:-1].min())
    
    breakout_level = recent_high if is_bull else recent_low
    
    if is_bull:
        extension_pct = ((c_live - breakout_level) / breakout_level) * 100
    else:
        extension_pct = ((breakout_level - c_live) / breakout_level) * 100

    is_extended = extension_pct > 1.5
    action_status = "⚠️ EXTENDED (CHASE RISK)" if is_extended else "🔥 VALID TRIGGER ENTRY"
    # -----------------------------------------------

    sl_dist = cfg.get("atr_mult", 1.2) * atr
    tp_dist = cfg.get("rr_ratio", 2.0) * sl_dist
    sl = round(c_live - sl_dist if is_bull else c_live + sl_dist, 2)
    tp = round(c_live + tp_dist if is_bull else c_live - tp_dist, 2)
    
    win_prob, trap_risk = predict_trade_prob(rvol, vwap_dist, rsi, atr_pct, mode)
    
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
    sl = round(
        entry - (cfg.get("atr_mult", 1.2) * atr)
        if is_bull
        else entry + (cfg.get("atr_mult", 1.2) * atr),
        2,
    )
    tp = round(
        entry + (cfg.get("rr_ratio", 2.0) * cfg.get("atr_mult", 1.2) * atr)
        if is_bull
        else entry - (cfg.get("rr_ratio", 2.0) * cfg.get("atr_mult", 1.2) * atr),
        2,
    )
    win_prob, trap_risk = predict_trade_prob(
        last["RVOL"], last["VWAP_Dist_Pct"], last["RSI"], last["ATR_Pct"], mode
    )
    return {
        "Symbol": symbol,
        "Direction": (
            "🔥 BULLISH MOMENTUM" if is_bull else "BEARISH MOMENTUM"
        ),
        "Current Price": round(c_live, 2),
        "Suggested Entry": entry,
        "Stop Loss": sl,
        "Target Price": tp,
        "SMC Structure": smc_patterns["SMC_Structure"],
        "Chart Pattern": smc_patterns["Pattern"],
        "RVOL": round(last["RVOL"], 2),
        "R/R Ratio": f"1 : {cfg.get('rr_ratio', 2.0)}",
        "AI Win Prob": win_prob,
        "Trap Risk": trap_risk,
    }


def run_meta_contrarian_analysis(
    df: pd.DataFrame, symbol: str, mode="intraday"
):
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
    if last["RSI"] > 70:
        score -= 6
        flags.append("⚠️ RSI Overbought")
    elif last["RSI"] < 30:
        score -= 6
        flags.append("⚠️ RSI Oversold")
    win_prob, _ = predict_trade_prob(
        last["RVOL"], last["VWAP_Dist_Pct"], last["RSI"], last["ATR_Pct"], mode
    )
    return {
        "Symbol": symbol,
        "Direction": "BULLISH" if is_bull else "BEARISH",
        "Re-Ranked Score": round(score, 1),
        "Crowd Diagnostics": (
            " | ".join(flags) if flags else "Optimal Setup"
        ),
        "SMC Structure": smc_patterns["SMC_Structure"],
        "Chart Pattern": smc_patterns["Pattern"],
        "Current Price": round(c_live, 2),
        "RVOL": round(last["RVOL"], 2),
        "RSI": round(last["RSI"], 1),
        "AI Win Prob": win_prob,
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
            grade = (
                "💎 TRIPLE ENGINE GEM"
                if len(matches) == 3
                else "⚡ DOUBLE CONFLUENCE SETUP"
            )
            action = (
                "🔥 HIGH CONVICTION ENTRY"
                if len(matches) == 3
                else "🎯 QUANT CONFIRMED ENTRY"
            )
            rows.append({
                "Symbol": sym,
                "Grade": grade,
                "Direction": base.get("Direction", "BULLISH"),
                "Entry": base.get("Suggested Entry", base.get("Current Price", 0)),
                "Stop Loss": base.get("Stop Loss", 0),
                "Target Price": base.get("Target Price", 0),
                "SMC Structure": base.get("SMC Structure", "NEUTRAL"),
                "Order Block": base.get("Order Block", "NONE"),
                "FVG Status": base.get("FVG Status", "NONE"),
                "Chart Pattern": base.get("Chart Pattern", "NONE"),
                "AI Win Prob": base.get("AI Win Prob", "50%"),
                "Trap Risk": base.get("Trap Risk", "LOW"),
                "Action": action,
            })
        elif len(matches) == 1 and smc:
            rows.append({
                "Symbol": sym,
                "Grade": "📊 SINGLE ENGINE SIGNAL",
                "Direction": smc["Direction"],
                "Entry": smc["Suggested Entry"],
                "Stop Loss": smc["Stop Loss"],
                "Target Price": smc["Target Price"],
                "SMC Structure": smc["SMC Structure"],
                "Order Block": smc.get("Order Block", "NONE"),
                "FVG Status": smc.get("FVG Status", "NONE"),
                "Chart Pattern": smc["Chart Pattern"],
                "AI Win Prob": smc["AI Win Prob"],
                "Trap Risk": smc["Trap Risk"],
                "Action": "👀 WATCHLIST CANDIDATE",
            })
    return pd.DataFrame(rows)


def run_fast_backtest(df: pd.DataFrame, sym: str, cfg: dict):
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
    times = df.index
    in_trade = False
    trade = {}
    for i in range(35, len(df)):
        c_p, h_p, l_p = closes[i], highs[i], lows[i]
        if in_trade:
            if trade["Direction"] == "BULLISH":
                if h_p >= trade["Target Price"]:
                    trade.update({
                        "Exit Price": trade["Target Price"],
                        "Exit Time": times[i],
                        "Result": "WIN 🎯",
                        "PnL %": round(
                            (
                                (trade["Target Price"] - trade["Entry Price"])
                                / trade["Entry Price"]
                            )
                            * 100,
                            2,
                        ),
                    })
                    trades.append(trade)
                    in_trade = False
                    continue
                elif l_p <= trade["Stop Loss"]:
                    trade.update({
                        "Exit Price": trade["Stop Loss"],
                        "Exit Time": times[i],
                        "Result": "LOSS 🛑",
                        "PnL %": round(
                            (
                                (trade["Stop Loss"] - trade["Entry Price"])
                                / trade["Entry Price"]
                            )
                            * 100,
                            2,
                        ),
                    })
                    trades.append(trade)
                    in_trade = False
                    continue
            elif trade["Direction"] == "BEARISH":
                if l_p <= trade["Target Price"]:
                    trade.update({
                        "Exit Price": trade["Target Price"],
                        "Exit Time": times[i],
                        "Result": "WIN 🎯",
                        "PnL %": round(
                            (
                                (trade["Entry Price"] - trade["Target Price"])
                                / trade["Entry Price"]
                            )
                            * 100,
                            2,
                        ),
                    })
                    trades.append(trade)
                    in_trade = False
                    continue
                elif h_p >= trade["Stop Loss"]:
                    trade.update({
                        "Exit Price": trade["Stop Loss"],
                        "Exit Time": times[i],
                        "Result": "LOSS 🛑",
                        "PnL %": round(
                            (
                                (trade["Entry Price"] - trade["Stop Loss"])
                                / trade["Entry Price"]
                            )
                            * 100,
                            2,
                        ),
                    })
                    trades.append(trade)
                    in_trade = False
                    continue
        if not in_trade:
            is_bull = (
                c_p > vwaps[i] and c_p > emas[i] and rvols[i] >= cfg["min_rvol"]
            )
            is_bear = (
                c_p < vwaps[i] and c_p < emas[i] and rvols[i] >= cfg["min_rvol"]
            )
            if is_bull or is_bear:
                direction = "BULLISH" if is_bull else "BEARISH"
                sl_dist = cfg["atr_mult"] * atrs[i]
                tp_dist = cfg["rr_ratio"] * sl_dist
                sl = round(c_p - sl_dist if is_bull else c_p + sl_dist, 2)
                tp = round(c_p + tp_dist if is_bull else c_p - tp_dist, 2)
                in_trade = True
                trade = {
                    "Symbol": sym,
                    "Direction": direction,
                    "Entry Time": times[i],
                    "Entry Price": c_p,
                    "Stop Loss": sl,
                    "Target Price": tp,
                    "RVOL": round(rvols[i], 2),
                    "VWAP_Dist_Pct": round(vwap_dists[i], 2),
                    "RSI": round(rsis[i], 1),
                    "ATR_Pct": round(atr_pcts[i], 2),
                }
    return trades


def discover_best_strategies(tickers: list, mode="intraday"):
    period = "1mo" if mode == "intraday" else "1y"
    interval = "5m" if mode == "intraday" else "1d"
    st.info("Pre-loading historical backtest market data via Yahoo Finance...")
    raw_data = {}
    for sym in tickers:
        df = fetch_historical_backtest_data(sym, period=period, interval=interval)
        if not df.empty and len(df) >= 40:
            raw_data[sym] = df
    if not raw_data:
        st.error(
            "Unable to fetch historical backtest data. Please verify ticker symbols."
        )
        return None, pd.DataFrame()
    param_grid = {
        "ema_span": [10, 20, 50],
        "atr_mult": [0.8, 1.2, 1.5, 2.0],
        "rr_ratio": [1.5, 2.0, 2.5, 3.0],
        "min_rvol": [1.0, 1.5],
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    st.info("Pre-calculating indicator vectors...")
    prepared_data = {
        ema: {
            sym: attach_vectorized_indicators(df, ema)
            for sym, df in raw_data.items()
        }
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
        if len(all_trades) < 10:
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
    st.sidebar.warning(
        "⚠️ Upstox Token Missing from Secrets. Using Yahoo Finance Fallback."
    )

universe = st.sidebar.selectbox(
    "Select Watchlist",
    ["Default Watchlist (7 Stocks)", "NIFTY 50 Expanded", "Custom Tickers"],
)
if universe == "Default Watchlist (7 Stocks)":
    symbols = [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "REDINGTON",
        "FIRSTSOURCE",
    ]
elif universe == "NIFTY 50 Expanded":
    symbols = [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
        "BHARTIARTL",
        "ITC",
        "LT",
        "AXISBANK",
    ]
else:
    custom_in = st.sidebar.text_input(
        "Enter Tickers (comma separated)", "RELIANCE, TCS, INFY"
    )
    symbols = [s.strip().upper() for s in custom_in.split(",") if s.strip()]

intra_cfg = load_config("intraday")
swing_cfg = load_config("swing")
st.sidebar.markdown(
    f"**Intraday Config:** Win Rate `{intra_cfg.get('win_rate', 'N/A')}%` | R/R"
    f" `1:{intra_cfg.get('rr_ratio', 2.0)}`"
)
st.sidebar.markdown(
    f"**Swing Config:** Win Rate `{swing_cfg.get('win_rate', 'N/A')}%` | R/R"
    f" `1:{swing_cfg.get('rr_ratio', 2.5)}`"
)

page = st.sidebar.radio(
    "Select Module",
    ["⚡ Multi-Tab Live Scanner", "🧪 AI Strategy Discovery & Backtester"],
)

if page == "⚡ Multi-Tab Live Scanner":
    st.title("⚡ Institutional Multi-Timeframe Scanner Engine")
    tab_master, tab_intraday, tab_momentum, tab_swing, tab_contrarian = st.tabs([
        "🌟 Master Confluence",
        "⚡ Intraday SMC (5m)",
        "🚀 Momentum Leaders (5m)",
        "📈 Swing Signals (1D Daily)",
        "🧠 Meta-Contrarian Engine",
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
            with st.spinner(
                "Scanning intraday 5m live candles using saved intraday_config.json..."
            ):
                results = [
                    run_smc_analysis(
                        fetch_live_data(s, "5d", "5m"),
                        s,
                        timeframe_label="5M INTRADAY",
                        mode="intraday",
                    )
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
                results = [
                    run_momentum_analysis(
                        fetch_live_data(s, "5d", "5m"), s, mode="intraday"
                    )
                    for s in symbols
                ]
                df_res = pd.DataFrame([r for r in results if r])
                if not df_res.empty:
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.info("No momentum leaders found.")

    with tab_swing:
        st.subheader("📈 Daily Swing Signals Engine (1D Daily Timeframe)")
        if st.button("📈 Run Daily Swing Scan", type="primary"):
            with st.spinner(
                "Scanning 1-Year Daily candles using saved swing_config.json..."
            ):
                results = [
                    run_smc_analysis(
                        fetch_live_data(s, "1y", "1d"),
                        s,
                        timeframe_label="1D DAILY SWING",
                        mode="swing",
                    )
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
                results = [
                    run_meta_contrarian_analysis(
                        fetch_live_data(s, "5d", "5m"), s, mode="intraday"
                    )
                    for s in symbols
                ]
                df_res = pd.DataFrame([r for r in results if r])
                if not df_res.empty:
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.info("No crowd traps detected.")

elif page == "🧪 AI Strategy Discovery & Backtester":
    st.title("🧪 Fast In-Memory Strategy Discovery Engine")
    st.caption(
        "Runs fast in-memory strategy discovery on Yahoo Finance historical datasets,"
        " saves optimal parameters to JSON, and trains ML models."
    )

    tf_mode = st.radio(
        "Select Target Timeframe Engine to Optimize",
        [
            "Intraday (5m / 1-Month Lookback)",
            "Swing Daily (1D / 1-Year Lookback)",
        ],
    )
    target_mode = "intraday" if "Intraday" in tf_mode else "swing"

    if st.button(
        f"🚀 Run Strategy Optimization ({target_mode.upper()})", type="primary"
    ):
        with st.spinner(
            f"Pre-loading historical data & optimizing {target_mode.upper()} parameters..."
        ):
            best_cfg, trades_df = discover_best_strategies(symbols, mode=target_mode)
            if best_cfg:
                st.success(
                    f"🎉 Optimized Config Discovered! Win Rate: {best_cfg['win_rate']}%"
                )
                st.json(best_cfg)
                st.subheader("Backtest Trade Logs Used for Machine Learning Training")
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.error(
                    "Could not find a high win-rate strategy permutation over the"
                    " specified sample size."
                )
# --- TEMPORARY DIAGNOSTIC BLOCK ---
with st.sidebar.expander("🔍 Live Upstox API Diagnostic", expanded=True):
    diag_token = get_upstox_access_token()
    st.write(f"**Token Detected:** `{bool(diag_token)}`")
    if diag_token:
        # Test request directly to Upstox RELIANCE intraday endpoint
        test_url = "https://api.upstox.com/v2/historical-candle/intraday/NSE_EQ%7CINE002A01018/5minute"
        test_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {diag_token}",
        }
        res = requests.get(test_url, headers=test_headers, timeout=5)
        st.write(f"**HTTP Status Code:** `{res.status_code}`")
        if res.status_code == 200:
            candles = res.json().get("data", {}).get("candles", [])
            st.write(f"**Latest Candle Time:** `{candles[0][0] if candles else 'No Data'}`")
            st.write(f"**Latest Close Price:** `{candles[0][4] if candles else 'No Data'}`")
        else:
            st.error(f"**Upstox Raw Response:** {res.text}")
# -----------------------------------
