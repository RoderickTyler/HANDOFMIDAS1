"""
factor_attribution.py
----------------------
Scores which macro factors actually explain gold's moves over whatever
history is available: a daily-level regression (DXY / real yields / VIX),
a quarter-by-quarter dominant-factor breakdown, and a central-bank-buying
scorecard that separates real accumulation from pure gold-price effect.

Whatever data is available is used -- partial quarters are shown as-is
rather than withheld, since waiting for a full year isn't practical this
early in the system's life. Quarters with too few trading days are
skipped (MIN_DAYS_FOR_QUARTER) since a correlation from a handful of days
is closer to noise than signal.
"""

import numpy as np
import pandas as pd

import reserves_utils

MIN_DAYS_FOR_QUARTER = 15
MIN_DAYS_FOR_REGRESSION = 30


def build_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df = df.dropna(subset=["gold_spot", "dxy", "dfii10"])
    df["gold_ret"] = df["gold_spot"].pct_change()
    df["dxy_ret"] = df["dxy"].pct_change()
    df["real_yield_chg"] = df["dfii10"].diff()
    df["vix_chg"] = df["vix"].diff() if "vix" in df.columns else np.nan
    return df


def regression_scores(feat_df: pd.DataFrame):
    """Standardized-coefficient regression of gold's daily return against
    DXY return, real yield change, and VIX change. Returns None if there
    isn't enough data for this to mean anything."""
    cols = ["dxy_ret", "real_yield_chg", "vix_chg"]
    d = feat_df.dropna(subset=["gold_ret"] + cols)
    if len(d) < MIN_DAYS_FOR_REGRESSION:
        return None

    X_std = d[cols].copy()
    for c in cols:
        std = X_std[c].std()
        X_std[c] = (X_std[c] - X_std[c].mean()) / std if std else 0
    y_std = (d["gold_ret"] - d["gold_ret"].mean()) / d["gold_ret"].std()

    X = np.column_stack([np.ones(len(X_std)), X_std.values])
    beta, *_ = np.linalg.lstsq(X, y_std.values, rcond=None)

    y_pred = X @ beta
    ss_res = np.sum((y_std.values - y_pred) ** 2)
    ss_tot = np.sum((y_std.values - y_std.values.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0

    return {
        "n_days": len(d),
        "r2": r2,
        "scores": {
            "DXY return": beta[1],
            "Real yield change": beta[2],
            "VIX change": beta[3],
        },
    }


def quarterly_table(feat_df: pd.DataFrame):
    """Per-quarter gold return + correlation with each factor + which
    factor correlated most strongly. Shows whatever quarters exist, even
    partial ones, as long as they clear MIN_DAYS_FOR_QUARTER."""
    d = feat_df.dropna(subset=["gold_ret", "dxy_ret", "real_yield_chg"]).copy()
    if d.empty:
        return []
    d["quarter"] = d.index.to_period("Q")

    rows = []
    for q, group in d.groupby("quarter"):
        if len(group) < MIN_DAYS_FOR_QUARTER:
            continue
        gold_ret = (group["gold_spot"].iloc[-1] / group["gold_spot"].iloc[0] - 1) * 100
        corr_dxy = group["gold_ret"].corr(group["dxy_ret"])
        corr_ry = group["gold_ret"].corr(group["real_yield_chg"])
        corr_vix = group["gold_ret"].corr(group["vix_chg"]) if group["vix_chg"].notna().any() else np.nan

        corrs = {"DXY": corr_dxy, "Real yield": corr_ry, "VIX": corr_vix}
        valid = {k: v for k, v in corrs.items() if pd.notna(v)}
        dominant = max(valid, key=lambda k: abs(valid[k])) if valid else "n/a"

        rows.append({
            "Quarter": str(q),
            "Gold return": gold_ret,
            "corr(DXY)": corr_dxy,
            "corr(RealYield)": corr_ry,
            "corr(VIX)": corr_vix,
            "Dominant": dominant,
            "n_days": len(group),
        })
    return rows


def reserves_growth_table(reserves_df: pd.DataFrame, gold_price_start: float, gold_price_end: float):
    """Compares each country's reserve dollar-value growth against gold's
    own price appreciation over the same window -- separating genuine
    physical accumulation from a country's holdings simply being worth
    more because gold itself got more expensive."""
    if reserves_df is None or reserves_df.empty:
        return []

    filtered = reserves_utils.select_single_sector_per_country(reserves_df)
    country_col, period_col, value_col, _ = reserves_utils.get_columns(filtered)
    if not all([country_col, period_col, value_col]):
        return []

    filtered = filtered.copy()

    def _parse_period(s):
        try:
            return pd.Period(str(s).replace("M", ""), freq="M")
        except Exception:
            return None

    filtered["_period"] = filtered[period_col].apply(_parse_period)
    filtered = filtered.dropna(subset=["_period"])

    gold_price_growth = (
        (gold_price_end / gold_price_start - 1) * 100
        if gold_price_start else None
    )

    rows = []
    for country, group in filtered.groupby(country_col):
        group = group.sort_values("_period")
        if len(group) < 2:
            continue
        start_val = group[value_col].iloc[0]
        end_val = group[value_col].iloc[-1]
        if not start_val:
            continue
        growth = (end_val / start_val - 1) * 100
        excess = (growth - gold_price_growth) if gold_price_growth is not None else None

        if excess is None:
            read = "n/a (no gold price benchmark)"
        elif excess > 8:
            read = "Genuine accumulation"
        elif excess < -8:
            read = "Net seller / drawdown"
        else:
            read = "Flat (price effect only)"

        rows.append({
            "Country": country,
            "Value growth": growth,
            "vs. gold price": excess,
            "Read": read,
        })

    rows.sort(key=lambda r: -r["Value growth"])
    return rows


def synthesize_ranking(regression, quarterly_rows, reserves_rows):
    """Plain-English synthesis of what the tables above already show --
    not a new statistic, just a summary line."""
    lines = []

    if quarterly_rows:
        total = len(quarterly_rows)
        dxy_wins = sum(1 for r in quarterly_rows if r["Dominant"] == "DXY")
        vix_wins = sum(1 for r in quarterly_rows if r["Dominant"] == "VIX")
        ry_wins = sum(1 for r in quarterly_rows if r["Dominant"] == "Real yield")
        lines.append(
            f"DXY was the strongest correlate in {dxy_wins}/{total} quarter(s) shown; "
            f"VIX in {vix_wins}/{total}; real yields in {ry_wins}/{total}."
        )
        if total > 1 and quarterly_rows[-1]["Dominant"] != quarterly_rows[-2]["Dominant"]:
            lines.append(
                f"Note: the most recent quarter's dominant factor "
                f"({quarterly_rows[-1]['Dominant']}) differs from the prior quarter -- "
                f"possible regime shift, worth watching."
            )

    if reserves_rows:
        real_buyers = [r["Country"] for r in reserves_rows if r["Read"] == "Genuine accumulation"]
        if real_buyers:
            lines.append(f"Central bank buying beyond pure price effect: {', '.join(real_buyers)}.")
        else:
            lines.append("No country shows reserve growth clearly beyond gold's own price appreciation.")

    if regression and regression["r2"] < 0.2:
        lines.append(
            f"Caution: these factors only explain {regression['r2']*100:.0f}% of daily moves -- "
            f"most day-to-day variation comes from something outside this model."
        )

    return lines


def build_report(price_df: pd.DataFrame, reserves_df: pd.DataFrame = None):
    """Full pipeline. Returns None if there isn't enough data yet to say
    anything meaningful."""
    feat_df = build_features(price_df)
    if feat_df.empty or len(feat_df) < MIN_DAYS_FOR_REGRESSION:
        return None

    regression = regression_scores(feat_df)
    quarterly_rows = quarterly_table(feat_df)

    reserves_rows = []
    if reserves_df is not None and not reserves_df.empty:
        gold_start = feat_df["gold_spot"].iloc[0]
        gold_end = feat_df["gold_spot"].iloc[-1]
        reserves_rows = reserves_growth_table(reserves_df, gold_start, gold_end)

    ranking = synthesize_ranking(regression, quarterly_rows, reserves_rows)

    return {
        "regression": regression,
        "quarterly_rows": quarterly_rows,
        "reserves_rows": reserves_rows,
        "ranking": ranking,
        "n_days": len(feat_df),
    }
