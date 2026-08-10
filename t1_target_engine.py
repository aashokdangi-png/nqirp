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
        
        if "ATR" in df_daily.columns:
            atr = df_daily["ATR"].iloc[-1]
        else:
            high_low = df_daily["High"] - df_daily["Low"]
            atr = high_low.rolling(14).mean().iloc[-1]
            if pd.isna(atr):
                atr = close * 0.015

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

    @staticmethod
    def evaluate_live_t1_signal(t1_target: dict, df_live_5m: pd.DataFrame, nifty_change_pct: float = 0.0) -> dict:
        """Evaluates a static T+1 target against live 5m market action & sentiment."""
        if df_live_5m.empty or len(df_live_5m) < 3:
            t1_target["Live Action"] = "⏳ WAITING FOR DATA"
            return t1_target

        last = df_live_5m.iloc[-1]
        open_price = df_live_5m["Open"].iloc[0]
        c_live = last["Close"]
        vwap = last.get("VWAP", c_live)
        rvol = last.get("RVOL", 1.0)

        target_bull = t1_target["T+1 Bullish Target"]
        target_bear = t1_target["T+1 Bearish Target"]
        sl_bull = t1_target["Bullish SL"]
        sl_bear = t1_target["Bearish SL"]

        # Gap Exhaustion Filter
        if open_price >= target_bull:
            t1_target["Live Action"] = "⚠️ GAP EXHAUSTED (OVERTARGET)"
            t1_target["Signal Quality"] = "INVALID ❌"
            return t1_target

        # Live Bullish Trigger
        if c_live > open_price and c_live > vwap and rvol >= 1.2 and nifty_change_pct >= -0.2:
            if c_live < target_bull and c_live > sl_bull:
                t1_target["Live Action"] = "🔥 LIVE BULLISH ENTRY"
                t1_target["Signal Quality"] = "HIGH CONVICTION 🟢"
                return t1_target

        # Live Bearish Trigger
        if c_live < open_price and c_live < vwap and rvol >= 1.2 and nifty_change_pct <= 0.2:
            if c_live > target_bear and c_live < sl_bear:
                t1_target["Live Action"] = "🩸 LIVE BEARISH ENTRY"
                t1_target["Signal Quality"] = "HIGH CONVICTION 🔴"
                return t1_target

        # Invalidation
        if c_live <= sl_bull and c_live >= sl_bear:
            t1_target["Live Action"] = "🛑 STOP LOSS INVALIDATED"
            t1_target["Signal Quality"] = "EXIT ❌"
        else:
            t1_target["Live Action"] = "⏳ NO CLEAR DIRECTION"
            t1_target["Signal Quality"] = "NEUTRAL ⚪"

        return t1_target

    @staticmethod
    def backtest_t1_strategy(
        df_daily: pd.DataFrame, 
        atr_mult: float = 1.2, 
        sl_mult: float = 0.8,
        slippage_pct: float = 0.0005,
        spread_pct: float = 0.0002
    ) -> dict:
        """Simulates next-day (T+1) targets with execution friction and sample validation."""
        if len(df_daily) < 30:
            return None

        df = df_daily.copy().reset_index(drop=True)
        high_low = df["High"] - df["Low"]
        df["ATR"] = high_low.rolling(14).mean()

        total_trades = 0
        target_hits = 0
        sl_hits = 0
        pnl_list = []

        total_friction = slippage_pct + (spread_pct / 2.0)

        for i in range(15, len(df) - 1):
            raw_entry = df.loc[i, "Close"]
            atr = df.loc[i, "ATR"]
            if pd.isna(atr) or atr == 0:
                continue

            entry_close = raw_entry * (1.0 + total_friction)

            next_high = df.loc[i + 1, "High"]
            next_low = df.loc[i + 1, "Low"]

            target_price = entry_close + (atr_mult * atr)
            sl_price = entry_close - (sl_mult * atr)

            total_trades += 1

            if next_high >= target_price:
                target_hits += 1
                exit_price = target_price * (1.0 - total_friction)
                pnl_list.append((exit_price - entry_close) / entry_close * 100)
            elif next_low <= sl_price:
                sl_hits += 1
                exit_price = sl_price * (1.0 - total_friction)
                pnl_list.append((exit_price - entry_close) / entry_close * 100)
            else:
                next_close = df.loc[i + 1, "Close"]
                exit_price = next_close * (1.0 - total_friction)
                pnl_list.append((exit_price - entry_close) / entry_close * 100)

        if total_trades < 20:
            return None

        hit_rate = (target_hits / total_trades) * 100
        avg_pnl = np.mean(pnl_list) if pnl_list else 0
        wins = [p for p in pnl_list if p > 0]
        losses = [abs(p) for p in pnl_list if p < 0]
        profit_factor = (sum(wins) / sum(losses)) if sum(losses) > 0 else np.nan

        return {
            "ATR Multiplier": f"{atr_mult}x",
            "Total T+1 Sessions": total_trades,
            "Target Hit Rate (%)": round(hit_rate, 2),
            "SL Hit Rate (%)": round((sl_hits / total_trades) * 100, 2),
            "Avg Session PnL (%)": round(avg_pnl, 2),
            "Profit Factor": round(profit_factor, 2) if not np.isnan(profit_factor) else "N/A",
            "Friction Deduction": f"{round(total_friction*200, 2)}% roundtrip"
        }
