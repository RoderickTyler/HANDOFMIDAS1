"""
analysis.py
-----------
Turns raw data into the things that actually matter for a daily gold view:

  1. Merge market + FRED data onto one calendar
  2. Compute real yield level & trend
  3. Compute 2s10s curve spread & trend
  4. Compute rolling correlations (gold vs DXY, gold vs real yield)
  5. Flag DIVERGENCES — moments where the "textbook" relationship breaks,
     which is exactly where the senior-analyst judgment call matters most
"""

import pandas as pd
import numpy as np

import config


def merge_datasets(market_df: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
    """Combine market and FRED data onto a single daily index, forward-filled
    for non-trading days (FRED sometimes has different holidays than markets)."""
    if market_df.empty:
        return fred_df
    if fred_df.empty:
        return market_df

    combined = market_df.join(fred_df, how="outer")
    combined = combined.sort_index().ffill()
    return combined


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns: real yield trend, curve spread, rolling correlations."""
    out = df.copy()

    # Real yield trend (5-day and 20-day rate of change, in basis points)
    if "dfii10" in out.columns:
        out["real_yield_chg_5d_bps"] = out["dfii10"].diff(5) * 100
        out["real_yield_chg_20d_bps"] = out["dfii10"].diff(20) * 100

    # Curve spread — use FRED's precomputed T10Y2Y if present, else derive it
    if "t10y2y" in out.columns:
        out["curve_spread"] = out["t10y2y"]
    elif "dgs10" in out.columns and "dgs2" in out.columns:
        out["curve_spread"] = out["dgs10"] - out["dgs2"]

    if "curve_spread" in out.columns:
        out["curve_inverted"] = out["curve_spread"] < 0

    # Rolling correlations: gold vs DXY (expected negative), gold vs real yield (expected negative)
    if "gold_spot" in out.columns and "dxy" in out.columns:
        gold_ret = out["gold_spot"].pct_change()
        dxy_ret = out["dxy"].pct_change()
        out["corr_gold_dxy_30d"] = gold_ret.rolling(config.ROLLING_WINDOW_SHORT).corr(dxy_ret)
        out["corr_gold_dxy_90d"] = gold_ret.rolling(config.ROLLING_WINDOW_LONG).corr(dxy_ret)

    if "gold_spot" in out.columns and "dfii10" in out.columns:
        gold_ret = out["gold_spot"].pct_change()
        real_yield_chg = out["dfii10"].diff()
        out["corr_gold_realyield_30d"] = gold_ret.rolling(config.ROLLING_WINDOW_SHORT).corr(real_yield_chg)
        out["corr_gold_realyield_90d"] = gold_ret.rolling(config.ROLLING_WINDOW_LONG).corr(real_yield_chg)

    return out


def detect_divergences(df: pd.DataFrame) -> dict:
    """
    Flag the moments a senior analyst would actually care about: when the
    textbook relationship (gold vs DXY, gold vs real yields) has broken down
    over the recent window. This is where "something else is dominant" and
    needs a story, not just a chart.
    """
    flags = {}
    if df.empty:
        return flags

    latest = df.iloc[-1]

    # Gold vs DXY: expected negative correlation. Positive = divergence.
    corr_dxy = latest.get("corr_gold_dxy_30d", np.nan)
    if pd.notna(corr_dxy):
        if corr_dxy > config.CORR_DIVERGENCE_THRESHOLD:
            flags["gold_dxy_divergence"] = (
                f"30d gold/DXY correlation is {corr_dxy:.2f} (positive) — "
                f"the usual inverse relationship has broken down. "
                f"Likely explanation: simultaneous safe-haven demand for both, "
                f"or a structural theme (central bank buying, de-dollarization) "
                f"overriding the normal FX relationship."
            )

    # Gold vs real yields: expected negative correlation. Positive = divergence.
    corr_ry = latest.get("corr_gold_realyield_30d", np.nan)
    if pd.notna(corr_ry):
        if corr_ry > config.CORR_DIVERGENCE_THRESHOLD:
            flags["gold_realyield_divergence"] = (
                f"30d gold/real-yield correlation is {corr_ry:.2f} (positive) — "
                f"gold is rising even as real yields rise, which is NOT what the "
                f"textbook opportunity-cost model predicts. Investigate: central "
                f"bank buying, geopolitical risk premium, or a dollar-confidence story."
            )

    # Curve inversion — leading indicator flag
    if "curve_inverted" in df.columns:
        recent_inversions = df["curve_inverted"].tail(20).sum()
        if recent_inversions > 0:
            flags["curve_inversion"] = (
                f"2s10s curve has been inverted on {recent_inversions} of the last 20 "
                f"trading days — historically a recession leading indicator. Two gold "
                f"implications to weigh against each other: (1) recession fear = "
                f"risk-off = bullish gold now, (2) eventual Fed cuts = falling future "
                f"real yields = bullish gold later. Which is the market pricing today?"
            )

    return flags


def summarize_latest(df: pd.DataFrame) -> dict:
    """Pull the latest values of every tracked indicator into a clean dict for reporting."""
    if df.empty:
        return {}
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    def chg(col):
        if col in df.columns and pd.notna(latest.get(col)) and pd.notna(prev.get(col)):
            return latest[col] - prev[col]
        return None

    return {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "gold_spot": latest.get("gold_spot"),
        "gold_chg_1d": chg("gold_spot"),
        "dxy": latest.get("dxy"),
        "dxy_chg_1d": chg("dxy"),
        "vix": latest.get("vix"),
        "dgs10": latest.get("dgs10"),
        "dgs2": latest.get("dgs2"),
        "dfii10": latest.get("dfii10"),
        "real_yield_chg_5d_bps": latest.get("real_yield_chg_5d_bps"),
        "real_yield_chg_20d_bps": latest.get("real_yield_chg_20d_bps"),
        "curve_spread": latest.get("curve_spread"),
        "corr_gold_dxy_30d": latest.get("corr_gold_dxy_30d"),
        "corr_gold_realyield_30d": latest.get("corr_gold_realyield_30d"),
    }
