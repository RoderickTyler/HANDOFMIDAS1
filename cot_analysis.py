"""
cot_analysis.py
----------------
INDEPENDENT MODULE (standalone, like xau_currency_score.py). Pulls the
CFTC's weekly Commitments of Traders (COT) report for COMEX gold futures
(Disaggregated Futures-Only report, contract code 088691) and tracks
Managed Money net positioning -- the standard "hot money"/speculative
sentiment proxy used across the industry.

Data source: CFTC's official Socrata Open Data API (free, public, no key
required for reasonable use):
    https://publicreporting.cftc.gov/resource/72hh-3qpy.json

Released every Friday 3:30pm ET, reflecting the prior Tuesday's positions
-- a real ~3-day lag baked into the data itself, and only one data point
per week (unlike your daily gold/DXY/yield pillars). Treat this the same
way as GPR: a slow-moving structural signal, not a daily trigger.

METHODOLOGY: the classic "COT Index" (Larry Williams' range-normalization
technique):
    COT_Index = (current_net - window_min) / (window_max - window_min) * 100
0 = most net-short in the window, 100 = most net-long. Shown across THREE
windows -- 1yr, 3yr, all-history -- rather than picking one, for the same
reason the regime model compares full-history vs recent-window: a single
window can hide whether "extreme" is extreme relative to the whole
picture or just relative to a possibly unusual recent stretch.

HONEST LIMITATION: the exact Socrata field names below (search terms in
_find_column calls) are based on research, not a live-tested connection
from this environment. Column discovery is deliberately defensive
(substring search, not hardcoded exact names) so a minor naming mismatch
fails loudly with the actual available columns printed, rather than
silently grabbing the wrong data.
"""

import numpy as np
import pandas as pd
import requests

CFTC_GOLD_CONTRACT_CODE = "088691"  # COMEX Gold -- confirmed via CFTC's own viewable report
CFTC_DISAGG_DATASET_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"


def fetch_cot_data(limit=5000):
    """
    Fetches the full available history of the Disaggregated Futures-Only
    report for COMEX gold via CFTC's Socrata API.
    """
    params = {
        "cftc_contract_market_code": CFTC_GOLD_CONTRACT_CODE,
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": limit,
    }
    try:
        resp = requests.get(CFTC_DISAGG_DATASET_URL, params=params, timeout=45)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[warn] CFTC COT request failed: {e}")
        print("       Manual fallback: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm")
        return pd.DataFrame()

    if not data:
        print("[warn] CFTC COT request returned no rows for gold (contract 088691).")
        return pd.DataFrame()

    return pd.DataFrame(data)


def _find_column(df, must_contain, must_not_contain=None):
    """Defensive column discovery -- searches by substring rather than a
    hardcoded exact name, since CFTC's precise Socrata field naming
    wasn't verified against a live connection from this environment."""
    must_not_contain = must_not_contain or []
    for col in df.columns:
        col_lower = col.lower()
        if all(term in col_lower for term in must_contain) and not any(term in col_lower for term in must_not_contain):
            return col
    return None


def build_managed_money_series(raw_df: pd.DataFrame):
    """
    Extracts report date, open interest, and Managed Money long/short from
    the raw CFTC response, computes net position, returns a clean
    DataFrame. Prints available columns if expected fields aren't found,
    so a schema mismatch is immediately diagnosable rather than silent.
    """
    if raw_df.empty:
        return pd.DataFrame()

    date_col = _find_column(raw_df, ["report_date"])
    oi_col = _find_column(raw_df, ["open_interest"], must_not_contain=["change", "pct", "percent"])
    mm_long_col = _find_column(raw_df, ["m_money", "long"], must_not_contain=["short", "spread", "change", "pct"])
    mm_short_col = _find_column(raw_df, ["m_money", "short"], must_not_contain=["long", "spread", "change", "pct"])

    missing = [name for name, col in [("date", date_col), ("open interest", oi_col),
                                       ("managed money long", mm_long_col),
                                       ("managed money short", mm_short_col)] if col is None]
    if missing:
        print(f"[warn] Could not find expected COT columns: {missing}.")
        print(f"       CFTC may have changed their schema. Available columns: {list(raw_df.columns)}")
        return pd.DataFrame()

    df = pd.DataFrame({
        "date": pd.to_datetime(raw_df[date_col]),
        "open_interest": pd.to_numeric(raw_df[oi_col], errors="coerce"),
        "mm_long": pd.to_numeric(raw_df[mm_long_col], errors="coerce"),
        "mm_short": pd.to_numeric(raw_df[mm_short_col], errors="coerce"),
    }).set_index("date").sort_index()

    df["mm_net"] = df["mm_long"] - df["mm_short"]
    return df.dropna()


def cot_index(series: pd.Series, window=None):
    """
    Larry Williams' COT Index: where the CURRENT value sits within its own
    trailing range, 0-100. window=None uses the full series (all-history).
    """
    if window is not None:
        series = series.tail(window)
    if len(series) < 2:
        return None
    current = series.iloc[-1]
    lo, hi = series.min(), series.max()
    if hi == lo:
        return 50.0
    return (current - lo) / (hi - lo) * 100


def position_changes(df: pd.DataFrame, windows=(4, 12, 26, 52)):
    """
    Change in Managed Money net position over several trailing windows
    (in weeks). Per the "positioning surprise" critique: the RATE of
    change is often more informative than the absolute level -- someone
    rapidly building exposure tells a different story than someone who's
    been at the same level for months, even if the current level looks
    identical in both cases.
    """
    if df.empty or "mm_net" not in df.columns:
        return {}
    current = df["mm_net"].iloc[-1]
    changes = {}
    for w in windows:
        if len(df) > w:
            past = df["mm_net"].iloc[-1 - w]
            changes[w] = current - past
        else:
            changes[w] = None
    return changes


def positioning_surprise_flag(changes: dict, df: pd.DataFrame, surprise_threshold_std=1.5):
    """
    Flags whether the most recent short-window change (4wk) is unusually
    LARGE relative to this series' own typical week-to-week volatility --
    a genuine "surprise," not just "positioning changed some amount."
    """
    if df.empty or len(df) < 20 or changes.get(4) is None:
        return None
    weekly_diffs = df["mm_net"].diff().dropna()
    typical_4wk_move_std = weekly_diffs.rolling(4).sum().std()
    if not typical_4wk_move_std or typical_4wk_move_std == 0:
        return None
    z = changes[4] / typical_4wk_move_std
    return {
        "change_4wk": changes[4],
        "z_score": z,
        "is_surprise": abs(z) >= surprise_threshold_std,
        "direction": "building longs" if z > 0 else "building shorts" if z < 0 else "flat",
    }


def classify_level(index_3yr):
    """Buckets the 3yr COT Index into 5 plain-language tiers."""
    if index_3yr is None:
        return None, None
    if index_3yr >= 90:
        return "extreme", "long"
    elif index_3yr >= 70:
        return "elevated", "long"
    elif index_3yr <= 10:
        return "extreme", "short"
    elif index_3yr <= 30:
        return "elevated", "short"
    else:
        return "neutral", None


def classify_change(surprise):
    """Buckets the 4-week positioning change (via its z-score) into plain-
    language magnitude tiers, plus direction."""
    if surprise is None:
        return None, None
    z = abs(surprise["z_score"])
    if z >= 2.5:
        magnitude = "extreme"
    elif z >= 1.5:
        magnitude = "large"
    elif z >= 0.75:
        magnitude = "moderate"
    else:
        magnitude = "slight"

    # Direction doesn't meaningfully matter for genuinely slight changes --
    # near-zero movement isn't really "toward" anything. Keeping direction
    # as None here matches change_desc's single "slight" entry below.
    if magnitude == "slight":
        direction = None
    else:
        direction = "long" if surprise["z_score"] > 0 else "short" if surprise["z_score"] < 0 else None
    return magnitude, direction


def build_narrative(index_3yr, surprise):
    """
    Combines positioning LEVEL (how stretched vs. history) and positioning
    CHANGE (how fast it's moving right now) into a plain-English narrative
    -- the two dimensions tell very different stories depending on whether
    they agree or disagree, which a single number can't convey.

    Returns a dict with a short headline tag and a 2-3 sentence explanation
    written for someone unfamiliar with COT mechanics.
    """
    level_tier, level_dir = classify_level(index_3yr)
    change_tier, change_dir = classify_change(surprise)

    if level_tier is None and change_tier is None:
        return {"headline": "Not enough data", "narrative": "Not enough COT history yet to interpret."}

    level_desc = {
        ("extreme", "long"): "at a historical extreme on the long side",
        ("elevated", "long"): "historically elevated on the long side",
        ("neutral", None): "near its historical middle -- nothing notable",
        ("elevated", "short"): "historically elevated on the short side",
        ("extreme", "short"): "at a historical extreme on the short side",
    }.get((level_tier, level_dir), "at an unclear level")

    change_desc = {
        ("extreme", "long"): "moving extremely fast toward more longs",
        ("large", "long"): "moving quickly toward more longs",
        ("moderate", "long"): "moderately adding to longs",
        ("slight", None): "barely changing",
        ("moderate", "short"): "moderately adding to shorts",
        ("large", "short"): "moving quickly toward more shorts",
        ("extreme", "short"): "moving extremely fast toward more shorts",
    }.get((change_tier, change_dir), "changing at an unclear pace")

    # The "so what": does the recent CHANGE agree with the existing LEVEL
    # (reinforcing an already-stretched position) or move against/toward
    # it (an early-stage move, or a potential unwind)?
    same_direction = (level_dir is not None and change_dir is not None and level_dir == change_dir)
    opposite_direction = (level_dir is not None and change_dir is not None and level_dir != change_dir)
    is_big_change = change_tier in ("large", "extreme")
    is_stretched_level = level_tier in ("elevated", "extreme")

    if is_stretched_level and same_direction and is_big_change:
        headline = "⚠ Stretched AND still building"
        why = (f"Positioning is already {level_desc}, and it's {change_desc} on top of that. "
               f"This is the combination worth paying the most attention to: the crowd is still piling in "
               f"even though there's historically little room left for fresh buyers (or sellers) to join. "
               f"That imbalance -- lots of people already positioned the same way, with room shrinking for "
               f"more to follow -- is what historically sets up sharper-than-usual reversals if sentiment turns.")
    elif is_stretched_level and not is_big_change:
        headline = "⚠ Stretched, but stalling"
        why = (f"Positioning is {level_desc}, but it's {change_desc} recently. "
               f"The crowd hasn't grown much lately -- the extreme is old, not fresh. Worth watching whether "
               f"this starts to unwind, since a stretched position that stops attracting new participants "
               f"can be more fragile than one still actively building.")
    elif is_stretched_level and opposite_direction:
        headline = "↻ Stretched, and starting to unwind"
        why = (f"Positioning is still {level_desc}, but the recent move is {change_desc} -- the opposite "
               f"direction from the existing extreme. This often reflects the crowd beginning to exit an "
               f"already-stretched position (short-covering if the extreme was short, profit-taking if it "
               f"was long), which can be an early sign the extreme itself is starting to resolve.")
    elif not is_stretched_level and is_big_change:
        headline = "→ Fresh move building"
        why = (f"Positioning is {level_desc} today, but it's {change_desc} right now. This looks like a new "
               f"move getting underway rather than a stretched, crowded one -- there's historically more room "
               f"for this to continue before positioning itself becomes a headwind to the trade.")
    else:
        headline = "✓ Unremarkable"
        why = ("Funds are positioned near their historical middle, and recent activity has been "
               "unremarkable. Current positioning is unlikely to materially amplify or constrain "
               "the existing trend.")

    return {
        "headline": headline,
        "level_tier": level_tier, "level_dir": level_dir,
        "change_tier": change_tier, "change_dir": change_dir,
        "narrative": why,
    }


def interpret_level(index_3yr):
    """
    Describes the STATE, deliberately without implying an outcome --
    "historically elevated" describes positioning, it does not claim
    positioning this elevated always reverts. Uses the 3yr window as the
    primary read (a middle ground between the noisier 1yr and the slower
    all-history).
    """
    if index_3yr is None:
        return "Not enough data yet"
    if index_3yr >= 80:
        return "Historically elevated (long side)"
    elif index_3yr <= 20:
        return "Historically depressed (long side, i.e. elevated short)"
    else:
        return "Near historical median -- no notable extreme"


def build_report(limit=5000):
    """Full standalone pipeline: fetch -> build series -> compute 1yr/3yr/all-history COT Index."""
    raw = fetch_cot_data(limit=limit)
    df = build_managed_money_series(raw)
    if df.empty:
        return None

    latest = df.iloc[-1]
    windows = {"1yr": 52, "3yr": 156, "all-history": None}
    indices = {label: cot_index(df["mm_net"], window=w) for label, w in windows.items()}
    changes = position_changes(df)
    surprise = positioning_surprise_flag(changes, df)
    level_interpretation = interpret_level(indices.get("3yr"))
    narrative = build_narrative(indices.get("3yr"), surprise)

    return {
        "df": df,
        "latest_date": df.index[-1],
        "latest_mm_long": latest["mm_long"],
        "latest_mm_short": latest["mm_short"],
        "latest_mm_net": latest["mm_net"],
        "latest_open_interest": latest["open_interest"],
        "cot_index": indices,
        "position_changes": changes,
        "surprise": surprise,
        "level_interpretation": level_interpretation,
        "narrative": narrative,
        "n_weeks": len(df),
    }


def print_report(limit=5000, recent_weeks=12):
    from tabulate import tabulate

    result = build_report(limit=limit)
    print("\n--- COT Report: Managed Money Positioning, COMEX Gold (CFTC, weekly) ---")
    print("  *** WEEKLY STRUCTURAL INDICATOR -- NOT FOR TIMING INTRADAY/SWING ENTRIES ***")
    print("  (Data is 3-6 days old by design: reflects Tuesday's positions, released Friday.")
    print("   Use this to gauge how crowded the current macro narrative already is, not to time trades.)")
    if result is None:
        print("  Could not fetch/parse COT data this run.")
        return

    print(f"\n  Latest report date: {result['latest_date'].date()}")
    print(f"  Managed Money: {result['latest_mm_long']:,.0f} long / "
          f"{result['latest_mm_short']:,.0f} short  ->  net {result['latest_mm_net']:+,.0f} contracts")
    print(f"  Open interest: {result['latest_open_interest']:,.0f} contracts")
    print(f"  ({result['n_weeks']} weekly reports available)")

    print(f"\n  Positioning state (3yr window): {result['level_interpretation']}")
    print("  (This describes current positioning relative to its own history -- it does NOT")
    print("   predict what happens next. Historically elevated positioning can persist for months.)")

    narrative = result.get("narrative")
    if narrative:
        print(f"\n  {narrative['headline']}")
        print(f"  {narrative['narrative']}")

    print("\n  COT Index by window (0 = most net-short in window, 100 = most net-long in window):")
    for label, val in result["cot_index"].items():
        if val is None:
            print(f"    {label:12s}: not enough data yet")
        else:
            print(f"    {label:12s}: {val:.1f}")

    print("\n  Position change (contracts, current vs N weeks ago):")
    for w, chg in result["position_changes"].items():
        if chg is None:
            print(f"    {w:2d}wk: not enough data yet")
        else:
            direction = "more long" if chg > 0 else "more short" if chg < 0 else "unchanged"
            print(f"    {w:2d}wk: {chg:+,.0f}  ({direction})")

    surprise = result["surprise"]
    if surprise:
        print(f"\n  Positioning surprise check (is the recent move unusually large for this series?):")
        print(f"    4wk change: {surprise['change_4wk']:+,.0f} contracts ({surprise['direction']})")
        print(f"    z-score vs typical 4wk move: {surprise['z_score']:+.2f}")
        if surprise["is_surprise"]:
            print(f"    -> YES, this is an unusually large positioning shift for this series.")
        else:
            print(f"    -> No, this is within normal week-to-week variation.")

    recent = result["df"].tail(recent_weeks)
    table = [[d.date(), f"{r['mm_long']:,.0f}", f"{r['mm_short']:,.0f}", f"{r['mm_net']:+,.0f}",
              f"{r['open_interest']:,.0f}"] for d, r in recent.iterrows()]
    print(f"\n  Last {recent_weeks} weekly reports:")
    print(tabulate(table, headers=["Date", "MM Long", "MM Short", "MM Net", "Open Interest"], tablefmt="simple"))


if __name__ == "__main__":
    print_report()
