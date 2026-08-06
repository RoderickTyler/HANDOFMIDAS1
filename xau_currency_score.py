"""
xau_currency_score.py
----------------------
INDEPENDENT MODULE. Decomposes gold's USD price move into a "USD strength"
component and a "gold basket strength" component, using a small basket of
majors (EUR, GBP, JPY, CHF) weighted to roughly match DXY's real
composition -- so the USD-strength leg computed here should track real
DXY closely (a built-in sanity check).

Only 5 series are fetched (gold via GC=F futures, EURUSD, GBPUSD, USDJPY,
USDCHF via spot FX -- all from yfinance, no key needed). Every cross
(XAUEUR, XAUGBP, XAUJPY, XAUCHF) is DERIVED via triangulation, not fetched
separately -- those crosses are mathematically determined by the majors
above, so fetching them independently would risk tiny provider-specific
inconsistencies with the arbitrage-implied value.

THE CORE IDENTITY (exact algebra, not a regression/approximation):
    XAUUSD_return = Gold_basket_return - USD_basket_return
    =>  Gold_basket_return = XAUUSD_return + USD_basket_return

This holds exactly (log returns, matching weights on both sides) --
verified every run via a self-check that computes the gold basket return
TWO independent ways and confirms they match.

Interpretation:
    - Gold_basket_return >> XAUUSD_return  -> USD weakness inflated the
      USD gold price; the "real" gold move was smaller than XAUUSD alone
      suggests.
    - Gold_basket_return << XAUUSD_return  -> gold rose in USD terms
      DESPITE a strengthening dollar -- stronger evidence of genuine gold
      demand than XAUUSD alone shows.
    - Gold_basket_return ~= XAUUSD_return  -> USD was roughly flat against
      the basket; XAUUSD is a fair read of gold's "real" move today.
"""

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = {
    "xauusd": "GC=F",  # XAUUSD=X isn't a valid Yahoo ticker (confirmed 404).
    # GC=F is COMEX gold futures -- the same ticker the rest of this system
    # already uses successfully. It sits ~1% away from true spot XAUUSD
    # (a known futures/spot basis, not a bug -- we found this gap earlier
    # comparing to a broker chart). That basis is roughly stable day to
    # day, so it barely affects RETURNS (which is all this module uses) --
    # it would matter much more if this module reported price LEVELS.
    "eurusd": "EURUSD=X",
    "gbpusd": "GBPUSD=X",
    "usdjpy": "USDJPY=X",
    "usdchf": "USDCHF=X",
}

# DXY's real component weights, renormalized to just these 4 currencies.
# DXY also includes CAD (~9.1%) and SEK (~4.2%), deliberately excluded --
# CAD is oil-correlated and SEK is thin/illiquid; both would add noise
# unrelated to gold/USD dynamics rather than clean signal.
_RAW_WEIGHTS = {"eur": 57.6, "jpy": 13.6, "gbp": 11.9, "chf": 3.6}
_TOTAL = sum(_RAW_WEIGHTS.values())
WEIGHTS = {k: v / _TOTAL for k, v in _RAW_WEIGHTS.items()}


def fetch_data(period="1y"):
    """Fetch the 5 raw spot series. Returns a DataFrame indexed by date."""
    frames = {}
    for name, ticker in TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period=period, interval="1d")
            if hist.empty:
                print(f"[warn] No data returned for {name} ({ticker})")
                continue
            frames[name] = hist["Close"]
        except Exception as e:
            print(f"[warn] Failed to fetch {name} ({ticker}): {e}")

    if not frames:
        raise RuntimeError("No FX/gold data could be fetched -- check internet connection.")

    df = pd.DataFrame(frames)
    # Same normalize + dedupe fix applied elsewhere in this system --
    # yfinance timestamps can carry a non-midnight time component that
    # silently doubles rows when merged/processed otherwise.
    df.index = df.index.tz_localize(None).normalize()
    df = df.groupby(df.index).last()
    return df


def build_scores(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Derives triangulated crosses, daily log returns, the USD basket
    return, the gold basket return, and the identity self-check."""
    df = raw_df.dropna(subset=list(TICKERS.keys())).copy()

    # Triangulated cross LEVELS (informational)
    df["xaueur"] = df["xauusd"] / df["eurusd"]
    df["xaugbp"] = df["xauusd"] / df["gbpusd"]
    df["xaujpy"] = df["xauusd"] * df["usdjpy"]
    df["xauchf"] = df["xauusd"] * df["usdchf"]

    # Daily LOG returns (required for the additive identity to hold exactly)
    for col in ["xauusd", "eurusd", "gbpusd", "usdjpy", "usdchf"]:
        df[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))

    # USD strength vs each currency, sign-normalized so POSITIVE = USD stronger.
    # EURUSD/GBPUSD quote USD per foreign unit (foreign currency is base),
    # so USD strengthening = EURUSD/GBPUSD FALLING -> flip sign.
    # USDJPY/USDCHF quote foreign currency per USD (USD is base),
    # so USD strengthening = USDJPY/USDCHF RISING -> no sign flip.
    df["usd_str_eur"] = -df["eurusd_ret"]
    df["usd_str_gbp"] = -df["gbpusd_ret"]
    df["usd_str_jpy"] = df["usdjpy_ret"]
    df["usd_str_chf"] = df["usdchf_ret"]

    df["usd_basket_ret"] = (
        WEIGHTS["eur"] * df["usd_str_eur"]
        + WEIGHTS["gbp"] * df["usd_str_gbp"]
        + WEIGHTS["jpy"] * df["usd_str_jpy"]
        + WEIGHTS["chf"] * df["usd_str_chf"]
    )

    df["gold_basket_ret"] = df["xauusd_ret"] + df["usd_basket_ret"]

    # Flag implausibly large single-day gold moves. GC=F (futures) can show
    # an artificial jump on contract-roll dates that doesn't reflect a real
    # price move -- we found exactly this pattern before (an "11% single-day
    # drop" in gold that turned out to be a roll artifact, not a real event).
    # Real gold essentially never moves >6% in a single day outside a
    # genuine historic crisis, so treat anything past that as suspect.
    ROLL_ARTIFACT_THRESHOLD = 0.06
    df["likely_roll_artifact"] = df["xauusd_ret"].abs() > ROLL_ARTIFACT_THRESHOLD

    # Self-check: compute the gold basket return the OTHER way (direct
    # weighted average of gold priced in each currency) and confirm they match.
    for col in ["xaueur", "xaugbp", "xaujpy", "xauchf"]:
        df[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))

    df["gold_basket_ret_check"] = (
        WEIGHTS["eur"] * df["xaueur_ret"]
        + WEIGHTS["gbp"] * df["xaugbp_ret"]
        + WEIGHTS["jpy"] * df["xaujpy_ret"]
        + WEIGHTS["chf"] * df["xauchf_ret"]
    )
    df["identity_gap"] = (df["gold_basket_ret"] - df["gold_basket_ret_check"]).abs()

    return df.dropna(subset=["xauusd_ret", "usd_basket_ret", "gold_basket_ret"])


def daily_table(scored_df: pd.DataFrame, n_days=20):
    """Readable recent-days table: date, XAUUSD return, USD basket return,
    gold basket return, and which force was dominant that day."""
    recent = scored_df.tail(n_days)
    rows = []
    for date, row in recent.iterrows():
        dominant = "Gold-driven" if abs(row["gold_basket_ret"]) > abs(row["usd_basket_ret"]) else "USD-driven"
        if row.get("likely_roll_artifact"):
            dominant = "[ROLL ARTIFACT?] " + dominant
        rows.append({
            "Date": date.date(),
            "XAUUSD ret": row["xauusd_ret"] * 100,
            "USD basket ret": row["usd_basket_ret"] * 100,
            "Gold basket ret": row["gold_basket_ret"] * 100,
            "Dominant": dominant,
        })
    return rows


def raw_pairs_table(scored_df: pd.DataFrame, n_days=10):
    """
    Shows each RAW fetched series (gold futures + 4 FX majors) with level
    and daily % change, so you can see exactly what each individual leg
    is doing before it gets folded into the composite basket scores.
    Returns (headers, rows) for direct use with tabulate.
    """
    recent = scored_df.tail(n_days)
    headers = ["Date", "XAUUSD(fut)", "chg%", "EURUSD", "chg%", "GBPUSD", "chg%", "USDJPY", "chg%", "USDCHF", "chg%"]
    rows = []
    for date, row in recent.iterrows():
        rows.append([
            date.date(),
            f"{row['xauusd']:.2f}", f"{row['xauusd_ret']*100:+.2f}",
            f"{row['eurusd']:.4f}", f"{row['eurusd_ret']*100:+.2f}",
            f"{row['gbpusd']:.4f}", f"{row['gbpusd_ret']*100:+.2f}",
            f"{row['usdjpy']:.2f}", f"{row['usdjpy_ret']*100:+.2f}",
            f"{row['usdchf']:.4f}", f"{row['usdchf_ret']*100:+.2f}",
        ])
    return headers, rows


def cross_pairs_table(scored_df: pd.DataFrame, n_days=10):
    """
    Shows each DERIVED (triangulated) gold cross with level and daily %
    change -- "what is gold doing, priced in each currency."
    Returns (headers, rows) for direct use with tabulate.
    """
    recent = scored_df.tail(n_days)
    headers = ["Date", "XAUEUR", "chg%", "XAUGBP", "chg%", "XAUJPY", "chg%", "XAUCHF", "chg%"]
    rows = []
    for date, row in recent.iterrows():
        rows.append([
            date.date(),
            f"{row['xaueur']:.2f}", f"{row['xaueur_ret']*100:+.2f}",
            f"{row['xaugbp']:.2f}", f"{row['xaugbp_ret']*100:+.2f}",
            f"{row['xaujpy']:.0f}", f"{row['xaujpy_ret']*100:+.2f}",
            f"{row['xauchf']:.2f}", f"{row['xauchf_ret']*100:+.2f}",
        ])
    return headers, rows


def leg_agreement(scored_df: pd.DataFrame, n_days=20):
    """
    Checks whether the 4 currency legs actually agree on direction, or
    whether the basket average is being driven by one dominant leg (EUR
    carries ~66% of the weight) while the others sit flat/opposite.

    For each day: how many of the 4 legs share the SAME SIGN as that day's
    overall usd_basket_ret, and what usd_basket_ret would have been WITHOUT
    the EUR leg (the largest single contributor) -- if excluding EUR flips
    the sign or magnitude a lot, that day's "USD strength" reading is
    really more of an "EUR weakness" reading wearing a basket costume.
    """
    recent = scored_df.tail(n_days).copy()
    legs = ["usd_str_eur", "usd_str_gbp", "usd_str_jpy", "usd_str_chf"]

    def _agree_count(row):
        basket_sign = np.sign(row["usd_basket_ret"])
        return sum(1 for leg in legs if np.sign(row[leg]) == basket_sign and basket_sign != 0)

    recent["agree_count"] = recent.apply(_agree_count, axis=1)

    # usd_basket_ret with EUR excluded, weights renormalized among the other 3
    non_eur_weight = WEIGHTS["gbp"] + WEIGHTS["jpy"] + WEIGHTS["chf"]
    recent["usd_basket_ex_eur"] = (
        WEIGHTS["gbp"] * recent["usd_str_gbp"]
        + WEIGHTS["jpy"] * recent["usd_str_jpy"]
        + WEIGHTS["chf"] * recent["usd_str_chf"]
    ) / non_eur_weight

    rows = []
    for date, row in recent.iterrows():
        eur_dependent = (
            np.sign(row["usd_basket_ret"]) != np.sign(row["usd_basket_ex_eur"])
            or abs(row["usd_basket_ret"]) > 0 and abs(row["usd_basket_ex_eur"]) / max(abs(row["usd_basket_ret"]), 1e-9) < 0.4
        )
        rows.append({
            "Date": date.date(),
            "USD basket": row["usd_basket_ret"] * 100,
            "Legs agreeing (of 4)": int(row["agree_count"]),
            "Ex-EUR basket": row["usd_basket_ex_eur"] * 100,
            "EUR-dependent?": "YES" if eur_dependent else "no",
        })
    return rows


def fetch_real_dxy(period="1y"):
    """Fetches the REAL DXY index for external validation of our derived
    usd_basket_ret -- our basket uses different (fewer, differently
    weighted) currencies than real DXY, so this checks how close our
    simplified proxy actually tracks the real thing."""
    try:
        hist = yf.Ticker("DX-Y.NYB").history(period=period, interval="1d")
        if hist.empty:
            print("[warn] No data returned for real DXY (DX-Y.NYB)")
            return None
        s = hist["Close"]
        s.index = s.index.tz_localize(None).normalize()
        s = s.groupby(s.index).last()
        return s
    except Exception as e:
        print(f"[warn] Failed to fetch real DXY: {e}")
        return None


def validate_against_real_dxy(scored_df: pd.DataFrame, period="1y"):
    """
    Compares our derived usd_basket_ret against the REAL DXY index's own
    daily return. High correlation = our simplified 4-currency proxy is a
    trustworthy stand-in for real USD strength. Low correlation = our
    basket is missing something real DXY captures (e.g. CAD, SEK, or the
    geometric-vs-arithmetic weighting difference).
    """
    real_dxy = fetch_real_dxy(period=period)
    if real_dxy is None:
        return None

    real_dxy_ret = np.log(real_dxy / real_dxy.shift(1)).rename("real_dxy_ret")
    merged = scored_df[["usd_basket_ret"]].join(real_dxy_ret, how="inner").dropna()

    if len(merged) < 10:
        print("[warn] Not enough overlapping data to validate against real DXY.")
        return None

    corr = merged["usd_basket_ret"].corr(merged["real_dxy_ret"])
    return {
        "n_days": len(merged),
        "correlation": corr,
        "our_cum_pct": merged["usd_basket_ret"].sum() * 100,
        "real_dxy_cum_pct": merged["real_dxy_ret"].sum() * 100,
        "merged": merged,
    }


def risk_sentiment_scores(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    EUR/GBP = 'risk' currencies (rate-differential / growth sensitive).
    JPY/CHF = classic safe-haven currencies (strengthen in flight-to-safety).

    risk_on_score:    average strength of EUR + GBP (positive = risk currencies gaining)
    safehaven_score:  average strength of JPY + CHF (positive = havens gaining)
    risk_spread:      risk_on_score - safehaven_score
                       (positive = "risk-on" day, risk currencies beating havens;
                        negative = "risk-off" day, havens beating risk currencies)
    """
    df = scored_df.copy()
    df["risk_on_score"] = -(df["usd_str_eur"] + df["usd_str_gbp"]) / 2
    df["safehaven_score"] = -(df["usd_str_jpy"] + df["usd_str_chf"]) / 2
    df["risk_spread"] = df["risk_on_score"] - df["safehaven_score"]
    return df
    """Rolling summary over the trailing window."""
    recent = scored_df.tail(n_days)
    dominant_days = recent.apply(
        lambda r: "Gold" if abs(r["gold_basket_ret"]) > abs(r["usd_basket_ret"]) else "USD", axis=1
    )
    return {
        "n_days": len(recent),
        "cum_xauusd_pct": recent["xauusd_ret"].sum() * 100,
        "cum_usd_basket_pct": recent["usd_basket_ret"].sum() * 100,
        "cum_gold_basket_pct": recent["gold_basket_ret"].sum() * 100,
        "gold_driven_days": int((dominant_days == "Gold").sum()),
        "usd_driven_days": int((dominant_days == "USD").sum()),
        "max_identity_gap_pct": recent["identity_gap"].max() * 100 if not recent.empty else None,
        "roll_artifact_days": int(recent["likely_roll_artifact"].sum()),
    }


def run_report(period="1y", n_days=20, detail_days=10):
    """Full standalone pipeline: fetch -> score -> print."""
    raw = fetch_data(period=period)
    scored = build_scores(raw)

    if scored.empty:
        print("[warn] Not enough data to compute XAU currency scores.")
        return

    from tabulate import tabulate

    print(f"\n--- Raw currency pairs (last {detail_days} days) ---")
    print("(What each individual leg is actually doing, before combining into scores)\n")
    headers, rows = raw_pairs_table(scored, n_days=detail_days)
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    print(f"\n--- Derived gold crosses (last {detail_days} days) ---")
    print("(Gold's price triangulated into each currency -- 'what is gold doing in EUR/GBP/JPY/CHF terms')\n")
    headers, rows = cross_pairs_table(scored, n_days=detail_days)
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    print(f"\n--- XAU Currency Strength Decomposition (last {n_days} days) ---")
    print("(Gold-driven = basket return exceeds USD basket return that day; USD-driven = the reverse)\n")

    rows = daily_table(scored, n_days=n_days)
    table = [[r["Date"], f"{r['XAUUSD ret']:+.2f}%", f"{r['USD basket ret']:+.2f}%",
              f"{r['Gold basket ret']:+.2f}%", r["Dominant"]] for r in rows]
    print(tabulate(table, headers=["Date", "XAUUSD", "USD basket", "Gold basket", "Dominant"], tablefmt="simple"))

    print(f"\n--- Leg agreement / dispersion check (last {detail_days} days) ---")
    print("(Is USD strength broad-based across all 4 currencies, or driven mainly by EUR alone?)\n")
    rows = leg_agreement(scored, n_days=detail_days)
    table = [[r["Date"], f"{r['USD basket']:+.2f}%", r["Legs agreeing (of 4)"],
              f"{r['Ex-EUR basket']:+.2f}%", r["EUR-dependent?"]] for r in rows]
    print(tabulate(table, headers=["Date", "USD basket", "Legs agreeing", "Ex-EUR basket", "EUR-dependent?"], tablefmt="simple"))

    print(f"\n--- Validation against REAL DXY (not just our derived basket) ---")
    validation = validate_against_real_dxy(scored, period=period)
    if validation:
        print(f"  Correlation (our basket vs. real DXY, {validation['n_days']} days): {validation['correlation']:.4f}")
        print(f"  Cumulative return -- our basket: {validation['our_cum_pct']:+.2f}%  "
              f"real DXY: {validation['real_dxy_cum_pct']:+.2f}%")
        if validation['correlation'] > 0.85:
            print(f"  -> Our simplified 4-currency basket tracks real DXY well.")
        else:
            print(f"  -> Notable divergence from real DXY -- CAD/SEK (excluded from our basket) "
                  f"or the weighting formula difference may matter more than expected.")
    else:
        print("  Could not validate against real DXY this run.")

    s = summary(scored, n_days=n_days)
    print(f"\nOver the last {s['n_days']} days:")
    print(f"  Cumulative XAUUSD return:     {s['cum_xauusd_pct']:+.2f}%")
    print(f"  Cumulative USD basket return: {s['cum_usd_basket_pct']:+.2f}%")
    print(f"  Cumulative Gold basket return:{s['cum_gold_basket_pct']:+.2f}%")
    print(f"  Days gold-driven vs USD-driven: {s['gold_driven_days']} vs {s['usd_driven_days']}")
    if s["max_identity_gap_pct"] is not None:
        print(f"  Max identity self-check gap: {s['max_identity_gap_pct']:.5f}% "
              f"(should be ~0 -- confirms the math checks out)")
    if s["roll_artifact_days"] > 0:
        print(f"  [!] {s['roll_artifact_days']} day(s) flagged as likely futures contract-roll "
              f"artifacts (>6% single-day move) -- treat those specific days' classification "
              f"with skepticism; the cumulative totals above may be distorted by them.")


if __name__ == "__main__":
    run_report()
