import os
import pickle
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Nifty 50 Watchlist
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LTIM.NS", "AXISBANK.NS"
]

LOG_FILE = "historical_training_log.csv"
MODEL_FILE = "model.pkl"

def log_daily_scanner_performance(symbol_list):
    """
    1. Downloads EOD 5-min candle data for the watchlist.
    2. Identifies stocks that made major momentum moves (>= 1.2% drop/gain).
    3. Saves labeled features into historical_training_log.csv.
    """
    print("Fetching EOD market data...")
    records = []
    
    for sym in symbol_list:
        try:
            df = yf.download(sym, period="1d", interval="5m", progress=False)
            if df.empty or len(df) < 20:
                continue

            open_p = float(df['Open'].iloc[0])
            high_p = float(df['High'].max())
            low_p = float(df['Low'].min())
            close_p = float(df['Close'].iloc[-1])
            vol_mean = float(df['Volume'].mean())

            day_range = high_p - low_p
            max_drop_pct = ((open_p - low_p) / open_p) * 100
            max_gain_pct = ((high_p - open_p) / open_p) * 100
            
            # Label: 1 if stock moved >= 1.2% (Target Hit), else 0
            target_hit = 1 if (max_drop_pct >= 1.2 or max_gain_pct >= 1.2) else 0

            records.append({
                "Symbol": sym,
                "RVOL": round(vol_mean / 10000, 2),
                "VWAP_Dist": round(abs(close_p - open_p) / open_p * 100, 2),
                "ATR_Pct": round(day_range / close_p * 100, 2),
                "Day_Change": round(((close_p - open_p) / open_p) * 100, 2),
                "TargetHit": target_hit
            })
        except Exception:
            continue

    if records:
        new_df = pd.DataFrame(records)
        header_needed = not os.path.exists(LOG_FILE)
        new_df.to_csv(LOG_FILE, mode="a", header=header_needed, index=False)
        print(f"Logged {len(records)} stock setups to {LOG_FILE}.")

def retrain_model_pkl():
    """
    Reads historical_training_log.csv and retrains model.pkl
    """
    if not os.path.exists(LOG_FILE):
        print("No training log found. Run logging first.")
        return

    df = pd.read_csv(LOG_FILE)
    if len(df) < 15:
        print("Need at least 15 logged historical trades to train ML model.")
        return

    X = df[["RVOL", "VWAP_Dist", "ATR_Pct", "Day_Change"]]
    y = df["TargetHit"]

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    
    print("Successfully trained and updated model.pkl with market learning!")

if __name__ == "__main__":
    log_daily_scanner_performance(NIFTY_50)
    retrain_model_pkl()
