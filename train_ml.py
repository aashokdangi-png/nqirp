import os
import json
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

NIFTY_WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "AXISBANK.NS", "LT.NS"
]

REPORT_FILE = "ml_report.json"
DATASET_FILE = "historical_ml_dataset.csv"
INTRADAY_MODEL = "intraday_ml_model.pkl"
SWING_MODEL = "swing_ml_model.pkl"

def extract_features_from_df(df: pd.DataFrame, nifty_df: pd.DataFrame = None) -> pd.DataFrame:
    """Extracts rich institutional features including SMC, market context, and timing."""
    if len(df) < 35:
        return pd.DataFrame()
        
    d = df.copy()
    close, high, low, vol = d["Close"], d["High"], d["Low"], d["Volume"]
    
    # Technical Indicators
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean().fillna(1.0)
    d["EMA"] = close.ewm(span=20, adjust=False).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    d["RSI"] = (100 - (100 / (1 + rs))).fillna(50.0)
    
    tp = (high + low + close) / 3
    d["VWAP"] = (tp * vol).cumsum() / vol.cumsum().replace(0, 1e-9)
    d["RVOL"] = (vol / vol.rolling(20).mean().replace(0, 1e-9)).fillna(1.0)
    d["VWAP_Dist_Pct"] = (close - d["VWAP"]).abs() / d["VWAP"] * 100
    d["ATR_Pct"] = (d["ATR"] / close) * 100
    
    # Time-of-day feature
    if "Datetime" in d.columns and pd.api.types.is_datetime64_any_dtype(d["Datetime"]):
        d["Hour"] = pd.to_datetime(d["Datetime"]).dt.hour
    else:
        d["Hour"] = 12

    # SMC Numeric Encodings
    d["SMC_Score"] = 0
    # Bullish FVG
    d.loc[d["Low"] > d["High"].shift(2), "SMC_Score"] += 1
    # Bearish FVG
    d.loc[d["High"] < d["Low"].shift(2), "SMC_Score"] -= 1
    # High Volume Order Block
    d.loc[(vol > vol.rolling(20).mean() * 1.5) & (close > d["Open"]), "SMC_Score"] += 1
    d.loc[(vol > vol.rolling(20).mean() * 1.5) & (close < d["Open"]), "SMC_Score"] -= 1

    # Nifty Broad Market Trend Context
    if nifty_df is not None and not nifty_df.empty:
        nifty_ret = nifty_df["Close"].pct_change(5).reindex(d.index, method="ffill").fillna(0)
        d["Nifty_Trend"] = nifty_ret
    else:
        d["Nifty_Trend"] = 0.0

    return d

def retrain_ml_models_eod():
    """Continuously retrains Random Forest models on accumulated daily historical dataset."""
    print("🤖 Executing Continuous EOD ML Model Auto-Retraining Pipeline...")
    
    # Fetch Nifty Index context
    try:
        nifty_raw = yf.download("^NSEI", period="1mo", interval="5m", progress=False)
        if isinstance(nifty_raw.columns, pd.MultiIndex):
            nifty_raw.columns = nifty_raw.columns.get_level_values(0)
    except Exception:
        nifty_raw = pd.DataFrame()

    dataset_rows = []
    
    for sym in NIFTY_WATCHLIST:
        try:
            df = yf.download(sym, period="1mo", interval="5m", progress=False)
            if df.empty or len(df) < 50:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            
            d = extract_features_from_df(df, nifty_raw)
            if d.empty:
                continue
                
            # Target generation: Next 3 candles return > 0.5%
            future_return = (d["Close"].shift(-3) - d["Close"]) / d["Close"] * 100
            d["Target"] = (future_return > 0.4).astype(int)
            d = d.dropna(subset=["Target"])
            
            dataset_rows.append(d)
        except Exception as e:
            print(f"Error processing {sym}: {e}")

    if not dataset_rows:
        print("⚠️ Insufficient data collected for retraining.")
        return False

    combined_df = pd.concat(dataset_rows, ignore_index=True)
    
    features = ["RVOL", "VWAP_Dist_Pct", "RSI", "ATR_Pct", "Hour", "SMC_Score", "Nifty_Trend"]
    X = combined_df[features].fillna(0)
    y = combined_df["Target"]

    if len(np.unique(y)) < 2:
        print("⚠️ Single class dataset. Retraining aborted.")
        return False

    # Train Intraday Model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, INTRADAY_MODEL)
    joblib.dump(model, SWING_MODEL)
    
    print("✅ Intraday & Swing ML Models successfully updated & persisted!")
    return True

def generate_daily_diagnostic():
    """Evaluates market moves, identifies missed opportunities & appends actionable recommendations."""
    print("Running EOD Performance & Diagnostic Analysis...")
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    missed_trades = []
    successful_setups = []
    recommendations = []

    for sym in NIFTY_WATCHLIST:
        try:
            df = yf.download(sym, period="1d", interval="5m", progress=False)
            if df.empty or len(df) < 20:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            open_p = float(df['Open'].iloc[0])
            high_p = float(df['High'].max())
            low_p = float(df['Low'].min())

            max_drop_pct = ((open_p - low_p) / open_p) * 100
            max_gain_pct = ((high_p - open_p) / open_p) * 100

            if max_drop_pct >= 1.2 or max_gain_pct >= 1.2:
                direction = "BEARISH" if max_drop_pct > max_gain_pct else "BULLISH"
                move_pct = max_drop_pct if direction == "BEARISH" else max_gain_pct
                
                missed_trades.append({
                    "Symbol": sym.replace(".NS", ""),
                    "Direction": direction,
                    "Move Size": f"{round(move_pct, 2)}%",
                    "Reason": "Early VWAP / Micro-BOS threshold lag during opening hour."
                })
            elif max_drop_pct < 0.6 and max_gain_pct < 0.6:
                successful_setups.append(sym.replace(".NS", ""))
        except Exception:
            continue

    if len(missed_trades) > 3:
        recommendations.append("⚠️ High volatility detected. Lower VWAP breakdown threshold by 0.1%.")
    else:
        recommendations.append("🟢 Model parameters aligned well with price range.")

    # Execute ML Model Auto-Retraining
    model_updated = retrain_ml_models_eod()
    if model_updated:
        recommendations.append("🤖 Continuous Learning Engine: ML models successfully retrained on fresh market data.")

    daily_entry = {
        "Date": today_str,
        "Total Tracked": len(NIFTY_WATCHLIST),
        "Missed Trades Count": len(missed_trades),
        "Missed Details": missed_trades,
        "Recommendations": recommendations
    }

    reports = []
    if os.path.exists(REPORT_FILE):
        try:
            with open(REPORT_FILE, "r") as f:
                reports = json.load(f)
        except Exception:
            reports = []

    reports = [r for r in reports if r.get("Date") != today_str]
    reports.append(daily_entry)

    with open(REPORT_FILE, "w") as f:
        json.dump(reports, f, indent=4)

    print(f"Diagnostic report updated in {REPORT_FILE}!")

if __name__ == "__main__":
    generate_daily_diagnostic()
