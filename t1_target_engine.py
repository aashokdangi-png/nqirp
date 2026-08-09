import pandas as pd
import numpy as np

class T1TargetEngine:
    @staticmethod
    def calculate_cpr(df_daily: pd.DataFrame) -> dict:
        """Calculates Central Pivot Range (CPR) for the next trading session."""
        last = df_daily.iloc[-1]
        p = (last["High"] + last["Low"] + last["Close"]) / 3.0
        bc = (last["High"] + last["Low"]) / 2.0
        tc = (p - bc) + p
        r1 = (2 * p) - last["Low"]
        s1 = (2 * p) - last["High"]
        return {
            "Pivot": round(p, 2),
            "TC": round(tc, 2),
            "BC": round(bc, 2),
            "R1": round(r1, 2),
            "S1": round(s1, 2)
        }

    @staticmethod
    def generate_t1_targets(df_daily: pd.DataFrame, symbol: str) -> dict:
        """Calculates post-market T+1 target estimation based on SMC, CPR, and ATR expansion."""
        if df_daily.empty or len(df_daily) < 5:
            return None
            
        cpr = T1TargetEngine.calculate_cpr(df_daily)
        close = df_daily["Close"].iloc[-1]
        
        # Calculate 14-day ATR if not present
        if "ATR" in df_daily.columns:
            atr = df_daily["ATR"].iloc[-1]
        else:
            high_low = df_daily["High"] - df_daily["Low"]
            atr = high_low.rolling(14).mean().iloc[-1]
            if pd.isna(atr):
                atr = close * 0.015

        # Single-day target calculation capped at R1/S1 and 1.2x ATR
        target_bullish = min(close + (1.2 * atr), cpr["R1"])
        target_bearish = max(close - (1.2 * atr), cpr["S1"])
        sl_bullish = round(close - (0.8 * atr), 2)
        sl_bearish = round(close + (0.8 * atr), 2)
        
        return {
            "Symbol": symbol,
            "Close Price": round(close, 2),
            "T+1 Bullish Target": round(target_bullish, 2),
            "Bullish SL": sl_bullish,
            "T+1 Bearish Target": round(target_bearish, 2),
            "Bearish SL": sl_bearish,
            "Tomorrow CPR (BC-TC)": f"{cpr['BC']} - {cpr['TC']}",
            "R1 / S1 Boundary": f"{cpr['R1']} / {cpr['S1']}",
            "Max Daily Volatility (1.2x ATR)": round(1.2 * atr, 2),
            "Status": "🔥 T+1 TARGET GENERATED"
        }
