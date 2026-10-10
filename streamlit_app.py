"""
streamlit_app.py
-----------------
Web dashboard for the Gold Macro System. Wraps the existing CLI modules
(fetch_data, analysis, hmm_regime, cot_analysis, factor_attribution,
trading_mode, econ_calendar, journal) in a Streamlit UI instead of
terminal text, so the same analysis can be deployed and viewed online.

Run locally:
    streamlit run streamlit_app.py

Deploy free on Streamlit Community Cloud: point it at this file, add
FRED_API_KEY under App settings -> Secrets. (The live XAUUSD spot quote
uses gold-api.com, which needs no key at all.)
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

import config
import fetch_data
import analysis
import hmm_regime
import factor_attribution
import trading_mode
import econ_calendar
import cot_analysis
import reserves_utils
import journal
import spot_gold
import gex_engine
import gold_comparison
import live_quotes
import gex_snapshot
import gex_alt
import gld_basis
import spot_history
import macro_confidence

st.set_page_config(
    page_title="Gold Macro Dashboard",
    page_icon="\U0001F4C8",
    layout="wide",
)

CACHE_TTL = 15 * 60  # 15 minutes -- avoid hammering free APIs on every rerun


# ---------------------------------------------------------------------------
# Cached data fetchers (thin wrappers around the existing modules)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_core_data(period: str, quick: bool = True):
    data = fetch_data.fetch_all(period=period, quick=quick)
    merged = analysis.merge_datasets(data["market"], data["fred"])
    if merged.empty:
        return None
    signals = analysis.compute_signals(merged)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    signals.to_csv(config.PRICE_CACHE)  # keep CLI-compatible cache on disk
    return {"raw": data, "signals": signals}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_spot():
    try:
        return spot_gold.fetch_xauusd_spot()
    except Exception:
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_live_quotes(feed: str):
    """
    ADDITIVE: standalone comparison table for VIX/DXY/Gold/EURUSD/GBPUSD/
    USDCHF/USDJPY, toggleable between yfinance and Dukascopy via the
    sidebar. Does NOT feed into the regime model, factor attribution, the
    main Overview metrics above it, or any other tab -- purely a separate
    comparison view. VIX always falls back to yfinance regardless of feed
    (confirmed unavailable via Dukascopy -- see live_quotes.py docstring).
    """
    rows = []
    for sym in live_quotes.YFINANCE_TICKERS.keys():
        try:
            val, source_used, note = live_quotes.get_quote(sym, feed=feed)
        except Exception as e:
            val, source_used, note = None, "error", str(e)
        rows.append({"Symbol": sym, "Value": val, "Source used": source_used, "Note": note})
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_spot_history(period: str):
    try:
        series = fetch_data.get_xauusd_spot_history(period=period)
        return series if not series.empty else None
    except Exception:
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_regime(_signals: pd.DataFrame):
    return hmm_regime.analyze_regime(_signals)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_cot():
    try:
        return cot_analysis.build_report()
    except Exception:
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_calendar(days_ahead=14):
    cal = econ_calendar.get_upcoming_calendar(days_ahead=days_ahead)
    powell = econ_calendar.get_upcoming_powell_speeches(days_ahead=days_ahead)
    return cal, powell


GEX_CACHE_TTL = 10 * 60  # options data is more time-sensitive than macro data

REGIME_LOG_URL = "https://raw.githubusercontent.com/RoderickTyler/HANDOFMIDAS1/data/logs/regime_log.csv"
REGIME_LOG_CACHE_TTL = 5 * 60  # the log itself only updates every 15 min; no need to refetch more than this


@st.cache_data(ttl=REGIME_LOG_CACHE_TTL, show_spinner=False)
def load_regime_log():
    """
    Fetches the regime history log over plain HTTP from the 'data' branch --
    deliberately NOT read from the local deployment checkout, so that the
    GitHub Actions cron committing to that branch every 15 minutes never
    triggers a Streamlit Cloud redeploy (which only watches 'main').
    Returns None if the branch/file doesn't exist yet (e.g. before the
    first Action run) or the fetch fails for any reason.
    """
    try:
        resp = requests.get(REGIME_LOG_URL, timeout=15)
        if resp.status_code != 200 or not resp.text.strip():
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        if df.empty:
            return None
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, format="ISO8601", errors="coerce")
        df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
        return df if not df.empty else None
    except Exception:
        return None


def compute_feature_separation(regime_result):
    """
    ADDITIVE diagnostic: for each of the model's 4 inputs, how much does
    its STANDARDIZED value differ across the fitted states? A bigger
    spread means that feature does more of the work distinguishing states
    from each other -- an emergent property of the fitted model, not
    something set by hand anywhere in the code. Uses the SAME mean/std the
    model itself was standardized with (returned by hmm_regime.analyze_regime),
    so scores are genuinely comparable across features despite very
    different raw units (a return % vs a yield-change vs a vol measure).

    Returns a dict {feature_name: separation_score}, sorted descending, or
    None if the model result doesn't have what's needed (e.g. too few
    states/observations for state_characteristics to have been computed).
    """
    mean = regime_result.get("mean")
    std = regime_result.get("std")
    characteristics = regime_result.get("state_characteristics")
    if mean is None or std is None or not characteristics:
        return None

    feature_cols = hmm_regime.FEATURE_COLS
    mean_arr = np.asarray(mean).flatten()
    std_arr = np.asarray(std).flatten()

    scores = {}
    for i, feat in enumerate(feature_cols):
        if i >= len(std_arr) or std_arr[i] == 0:
            continue
        standardized_state_means = []
        for row in characteristics:
            raw_mean = row.get(f"{feat}_mean")
            if raw_mean is not None and raw_mean == raw_mean:  # not NaN
                standardized_state_means.append((raw_mean - mean_arr[i]) / std_arr[i])
        if len(standardized_state_means) >= 2:
            scores[feat] = max(standardized_state_means) - min(standardized_state_means)

    if not scores:
        return None
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# EXPERIMENTAL, ADDITIVE ONLY: two ways of digging into the gold_vol_5d
# dominance finding, per user request. Both are self-contained -- neither
# touches the existing regime_result, mode, or any display above them.
# ---------------------------------------------------------------------------

def compute_volatility_context(regime_result, lookback_days=252):
    """
    OPTION 1: instead of asking "which feature wins" (constant -- always
    gold_vol_5d, per user observation), ask the more informative question:
    is CURRENT gold_vol_5d elevated or calm relative to ITS OWN recent
    history? That's the part that actually varies day to day and is
    genuinely useful for stop-sizing.
    """
    feat_df = regime_result.get("feat_df")
    if feat_df is None or feat_df.empty or "gold_vol_5d" not in feat_df.columns:
        return None

    series = feat_df["gold_vol_5d"].dropna()
    if len(series) < 20:
        return None

    window = series.tail(lookback_days)
    current = series.iloc[-1]
    percentile = (window < current).sum() / len(window) * 100

    lookback_for_trend = min(10, len(series) - 1)
    prior = series.iloc[-1 - lookback_for_trend]
    trend = "rising" if current > prior else "falling" if current < prior else "flat"
    trend_pct = ((current - prior) / prior * 100) if prior != 0 else None

    return {
        "current": current,
        "percentile_in_own_history": percentile,
        "window_days": len(window),
        "trend": trend,
        "trend_pct": trend_pct,
        "trend_lookback_days": lookback_for_trend,
    }


def build_smoothed_feat_df(signals, window=5):
    """
    OPTION 2: build a "fairer" feature set to test whether gold_vol_5d's
    dominance is genuine economic signal or a smoothing artifact (a rolling
    stat is inherently more persistent than raw daily changes, which an HMM
    naturally favors when building sticky states, independent of which
    feature is more economically meaningful). Applies the SAME 5-period
    rolling treatment to the 3 raw-change features (rolling MEAN, matching
    gold_vol_5d's own rolling-window persistence) while leaving gold_vol_5d
    itself unchanged (it's already smoothed). Reuses hmm_regime.build_features'
    exact raw-feature construction first, then smooths on top -- so this
    stays consistent with the production pipeline rather than reinventing it.
    """
    df = signals.copy()
    required = {"gold_spot", "dxy", "dfii10"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(subset=["gold_spot", "dxy", "dfii10"])
    df["gold_ret"] = df["gold_spot"].pct_change()
    df["dxy_ret"] = df["dxy"].pct_change()
    df["real_yield_chg"] = df["dfii10"].diff()
    df["gold_vol_5d"] = df["gold_ret"].rolling(5).std()

    smoothed = pd.DataFrame(index=df.index)
    smoothed["gold_ret"] = df["gold_ret"].rolling(window).mean()
    smoothed["dxy_ret"] = df["dxy_ret"].rolling(window).mean()
    smoothed["real_yield_chg"] = df["real_yield_chg"].rolling(window).mean()
    smoothed["gold_vol_5d"] = df["gold_vol_5d"]  # already smoothed, left as-is

    return smoothed[hmm_regime.FEATURE_COLS].dropna()


def compute_regime_episodes(df):
    """
    Collapses consecutive same-state rows into episodes (start, end,
    duration, state) -- the basis for dwell-time stats and streak length.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["state", "start", "end", "duration_minutes"])
    d = df.sort_values("timestamp_utc").reset_index(drop=True)
    d["group"] = (d["top_state"] != d["top_state"].shift()).cumsum()
    episodes = d.groupby("group").agg(
        state=("top_state", "first"),
        start=("timestamp_utc", "first"),
        end=("timestamp_utc", "last"),
    ).reset_index(drop=True)
    episodes["duration_minutes"] = (episodes["end"] - episodes["start"]).dt.total_seconds() / 60
    return episodes


def compute_regime_confidence(
    log_df, episodes, live_top_prob=None,
    choppy_window_hours=4, choppy_flip_threshold=2, prob_threshold=0.65,
    min_sticky_streak_hours=1.0,
):
    """
    ADDITIVE helper -- does not alter any existing regime computation.
    Combines regime PERSISTENCE (how long the current state has held, and
    how it compares to that state's typical dwell time this week) with the
    HMM's own instantaneous confidence (top_prob) into one tier: sticky /
    neutral / choppy. The intent is to distinguish "this regime read is
    backed by a stable trend" from "this is one recent flip that may not
    hold" -- a single flip shouldn't carry the same weight as a state
    that's persisted well past its usual duration.

    - choppy: 2+ state changes within the last `choppy_window_hours` --
      treat the current regime as noisy, not a confirming macro layer.
    - sticky: NOT choppy, AND current streak has cleared an absolute floor
      (min_sticky_streak_hours) AND at least one PRIOR occurrence of this
      state exists in the week's log to judge "typical" against (a state
      on its very first-ever appearance has nothing to compare to -- the
      comparison would be trivially true against itself, which is exactly
      the "one recent flip looks stable" trap this is meant to avoid),
      AND current streak >= that state's typical weekly dwell time,
      AND live model probability >= prob_threshold.
    - neutral: everything else.

    Returns None if there isn't enough logged history yet to judge.
    """
    if log_df is None or log_df.empty or episodes is None or episodes.empty:
        return None

    current_episode = episodes.iloc[-1]
    current_state = current_episode["state"]
    current_streak_hours = current_episode["duration_minutes"] / 60

    same_state_episodes = episodes[episodes["state"] == current_state]
    has_prior_occurrence = len(same_state_episodes) >= 2  # current + at least 1 earlier
    typical_dwell_hours = (
        same_state_episodes["duration_minutes"].mean() / 60
        if not same_state_episodes.empty else None
    )

    cutoff = log_df["timestamp_utc"].max() - pd.Timedelta(hours=choppy_window_hours)
    flips_recent = int((episodes["start"] >= cutoff).sum())

    if live_top_prob is None:
        live_top_prob = log_df.iloc[-1]["top_prob"]

    is_choppy = flips_recent >= choppy_flip_threshold
    is_sticky = (
        not is_choppy
        and current_streak_hours >= min_sticky_streak_hours
        and has_prior_occurrence
        and (typical_dwell_hours is None or current_streak_hours >= typical_dwell_hours)
        and live_top_prob >= prob_threshold
    )
    tier = "choppy" if is_choppy else ("sticky" if is_sticky else "neutral")

    return {
        "tier": tier,
        "current_state": current_state,
        "current_streak_hours": current_streak_hours,
        "typical_dwell_hours": typical_dwell_hours,
        "has_prior_occurrence": has_prior_occurrence,
        "flips_recent": flips_recent,
        "choppy_window_hours": choppy_window_hours,
        "live_top_prob": live_top_prob,
    }


@st.cache_data(ttl=GEX_CACHE_TTL, show_spinner=False)
def load_gex_expirations(ticker: str):
    import yfinance as yf
    return list(yf.Ticker(ticker).options)


@st.cache_data(ttl=GEX_CACHE_TTL, show_spinner=False)
def load_gex_assessment(ticker: str, expiration, min_oi, use_marketdata: bool):
    if use_marketdata and config.MARKETDATA_API_KEY:
        chain, spot = gex_engine.fetch_gld_chain_marketdata_app(
            api_key=config.MARKETDATA_API_KEY,
            underlying=ticker,
            min_open_interest=min_oi or None,
        )
    else:
        chain, spot = gex_engine.fetch_gld_chain_yfinance(
            ticker=ticker, expiration=expiration, min_open_interest=min_oi or None
        )
    result = gex_engine.run_assessment(chain, spot=spot, underlying=ticker)
    return result


@st.cache_data(ttl=GEX_CACHE_TTL, show_spinner=False)
def load_gex_live_refs():
    live_gld = None
    live_xau = None
    try:
        live_gld = gex_engine.fetch_live_gld_spot()
    except Exception:
        pass
    try:
        spot_result = spot_gold.fetch_xauusd_spot()
        if spot_result is not None:
            live_xau = spot_result["price"]
    except Exception:
        pass
    return live_gld, live_xau


@st.cache_data(ttl=5 * 60, show_spinner=False)
def load_remote_gex_store():
    """ADDITIVE: last-good GEX snapshots written by the scheduled logger to the
    `data` branch (empty dict if none yet / unreachable)."""
    return gex_snapshot.fetch_remote()


@st.cache_data(ttl=GEX_CACHE_TTL, show_spinner=False)
def load_cboe_chain(symbol: str):
    return gex_alt.fetch_cboe_chain(symbol, timeout=12)


@st.cache_data(ttl=GEX_CACHE_TTL, show_spinner=False)
def load_gvz():
    return gex_alt.fetch_gvz()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_risk_free():
    return gex_alt.fetch_risk_free()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_spot_history_chain(period: str, _reference, live_price):
    """ADDITIVE: validated TRUE-spot daily history (yfinance -> Dukascopy -> Stooq), checked against the GC=F
    reference so a wrong series is rejected rather than plotted. Display only -- nothing else uses it."""
    try:
        return spot_history.get_spot_history(period, reference=_reference, live_price=live_price)
    except Exception as e:
        return {"series": None, "source": None, "notes": [f"spot history chain error: {e}"],
                "shift_days": 0, "corr": None, "ratio": None, "live_point": False}


BASIS_CACHE_TTL = 15 * 60  # GLD/XAUUSD basis is re-captured every 15 min (matches the regime logger cadence)


@st.cache_data(ttl=BASIS_CACHE_TTL, show_spinner=False)
def load_basis_live():
    """FAST cadence: GLD / XAUUSD / GC=F basis point. Never raises."""
    return gld_basis.fetch_basis_snapshot()


@st.cache_data(ttl=5 * 60, show_spinner=False)
def load_basis_history():
    """Logged 15-min basis points written by the scheduled job to the data branch."""
    return gld_basis.fetch_remote_points()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_avg_volume_10d(symbol: str):
    return gex_alt.fetch_avg_volume_10d(symbol)


def load_stable_oz_per_share():
    """The existing 24h-cached oz/share (gold_comparison), or None. Not cached here on purpose --
    gold_comparison does its own 24h caching."""
    try:
        return float(gold_comparison.get_oz_per_share())
    except Exception:
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_smoothed_regime(_signals: pd.DataFrame):
    """ADDITIVE: the same fair-smoothing refit the Regime tab's Option 2 runs
    on a button, cached so the Macro Confidence tab can reuse it."""
    feat = build_smoothed_feat_df(_signals)
    if feat.empty:
        return None
    return hmm_regime.analyze_regime(_signals, precomputed_feat_df=feat)


def find_nearest_gex_clusters(gex_by_strike, spot, window_pct=0.08, magnitude_frac=0.20, top_n=6):
    """
    Finds strikes CLOSE to spot that ALSO stand out visually (a real
    concentration of GEX, not just noise) -- different from
    result.call_walls/put_walls, which rank purely by size everywhere in
    the chain regardless of distance from spot. This answers "what's
    nearby AND large enough to matter", not "what's biggest anywhere in
    the chain" (that's call_walls/put_walls) and not "everything within
    reach regardless of size" (too lenient a filter just adds noise like
    a $1-2M strike sitting next to a $25M one).

    Ranks by GROSS exposure (|CallGEX| + |PutGEX|) rather than NetGEX --
    a strike with large, roughly-offsetting call and put GEX looks small
    on a net basis but is still visually one of the busiest strikes on the
    chart (tall bars both up and down), so gross is what actually matches
    "how big does this bar look", not net.

    window_pct: how far from spot to look (fraction of spot price).
    magnitude_frac: a strike must have gross GEX at least this fraction of
        the largest gross GEX within the window (default 20%) to count as
        a real concentration rather than noise.
    Returns a DataFrame sorted by distance from spot (nearest first), or
    an empty DataFrame if nothing in the window clears the bar.
    """
    if gex_by_strike is None or gex_by_strike.empty:
        return pd.DataFrame()
    window = gex_by_strike[
        (gex_by_strike.index >= spot * (1 - window_pct))
        & (gex_by_strike.index <= spot * (1 + window_pct))
    ].copy()
    if window.empty:
        return pd.DataFrame()
    window["GrossGEX"] = window["CallGEX"].abs() + window["PutGEX"].abs()
    threshold = window["GrossGEX"].max() * magnitude_frac
    candidates = window[window["GrossGEX"] >= threshold].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates["DistFromSpot"] = candidates.index - spot
    candidates["AbsDistFromSpot"] = candidates["DistFromSpot"].abs()
    return candidates.sort_values("AbsDistFromSpot").head(top_n)


def load_factor_attribution(signals, reserves_df):
    try:
        return factor_attribution.build_report(signals, reserves_df)
    except Exception as e:
        st.warning(f"Factor attribution failed this run: {e}")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Gold Macro System")
st.sidebar.caption("Free, no-paid-API macro dashboard for gold")

period = st.sidebar.selectbox("Lookback window", ["6mo", "1y", "2y", "5y"], index=1)

if "quick_load" not in st.session_state:
    st.session_state.quick_load = True  # default: quick load, per user request

if st.sidebar.button("\u26a1 Quick load (skip reserves/GPR)"):
    st.session_state.quick_load = True
    st.cache_data.clear()

if st.sidebar.button("\U0001F504 Refresh data now"):
    st.session_state.quick_load = False
    st.cache_data.clear()

st.sidebar.caption(
    "Quick load is the default \u2014 skips Central Bank Reserves and the "
    "Geopolitical Risk Index for a faster load (both are the slowest, "
    "flakiest fetches). For the full dataset including those, click "
    "'Refresh data now' above."
)

if not config.FRED_API_KEY:
    st.sidebar.error(
        "No FRED_API_KEY found. Add it as a local .env value or, if deployed "
        "on Streamlit Cloud, under **Settings -> Secrets**. Yields, real "
        "yield trend, curve spread, and the econ calendar will show as n/a "
        "until this is set."
    )
else:
    st.sidebar.success("FRED_API_KEY loaded.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Yahoo Finance (gold/DXY/VIX), FRED (yields), IMF (reserves), "
    "CFTC (COT), Caldara-Iacoviello GPR index, gold-api.com (XAUUSD spot). "
    "All free, no key needed except FRED."
)

st.sidebar.markdown("---")
live_quotes_feed_choice = st.sidebar.radio(
    "Live Quotes feed (Overview tab)",
    ["yfinance (default)", "Dukascopy"],
    index=0,
    help="Toggles the source for the separate 'Live Quotes Comparison' table on the "
         "Overview tab only. Does NOT affect the main Gold/DXY/VIX metrics above it, "
         "the regime model, or any other tab. VIX always falls back to yfinance -- "
         "confirmed unavailable via Dukascopy.",
)
live_quotes_feed = "dukascopy" if live_quotes_feed_choice == "Dukascopy" else "yfinance"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("\U0001F4C8 Gold Macro Daily Briefing")
st.caption(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} \u2014 lookback: {period}")

try:
    with st.spinner("Fetching market data, yields, reserves, and risk index..."):
        core = load_core_data(period, quick=st.session_state.quick_load)
except Exception as e:
    core = None
    st.error(
        f"Data fetch failed: {e}\n\n"
        "This usually means the app can't reach Yahoo Finance / FRED right now "
        "(temporary outage or rate limit), or FRED_API_KEY is missing/invalid. "
        "Check the sidebar, then hit Refresh."
    )

if core is None:
    st.stop()

signals = core["signals"]
raw = core["raw"]
summary = analysis.summarize_latest(signals)
flags = analysis.detect_divergences(signals)

spot_result = load_spot()
if spot_result is not None:
    summary["xauusd_spot"] = spot_result["price"]

_all_tabs = st.tabs([
    "Overview", "Regime (HMM)", "Macro Confidence", "COT Positioning", "Factor Attribution",
    "Econ Calendar", "Central Bank Reserves", "Journal", "GEX (Options)",
    "GEX (Alternative)", "Regime History",
])
# ADDITIVE: two new tabs were inserted into the display order. `tabs[0..8]`
# below keep their ORIGINAL meaning (so no existing `with tabs[n]:` block
# had to change); the new tabs are addressed by name.
tabs = [_all_tabs[i] for i in (0, 1, 3, 4, 5, 6, 7, 8, 10)]
macro_conf_tab = _all_tabs[2]
gex_alt_tab = _all_tabs[9]

# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    gold_val = summary.get("gold_spot")
    col1.metric(
        "Gold (GC=F futures)",
        f"{gold_val:,.2f}" if gold_val is not None else "n/a",
        f"{summary.get('gold_chg_1d'):+.2f}" if summary.get("gold_chg_1d") is not None else None,
    )
    xauusd = summary.get("xauusd_spot")
    col2.metric("Gold (XAUUSD spot)", f"{xauusd:,.2f}" if xauusd is not None else "n/a")
    dxy_val = summary.get("dxy")
    col3.metric(
        "DXY",
        f"{dxy_val:,.2f}" if dxy_val is not None else "n/a",
        f"{summary.get('dxy_chg_1d'):+.2f}" if summary.get("dxy_chg_1d") is not None else None,
    )
    vix_val = summary.get("vix")
    col4.metric("VIX", f"{vix_val:,.2f}" if vix_val is not None else "n/a")

    st.markdown(f"#### Live Quotes Comparison (feed: {live_quotes_feed_choice})")
    st.caption(
        "Standalone comparison view, toggled from the sidebar -- does NOT feed into the "
        "regime model, factor attribution, or the Gold/DXY/VIX metrics above. VIX always "
        "falls back to yfinance (confirmed unavailable via Dukascopy). DXY via Dukascopy "
        "is their own dollar-index construction, not verified identical to ICE DXY."
    )
    try:
        live_quotes_df = load_live_quotes(live_quotes_feed)
        display_df = live_quotes_df.copy()
        display_df["Value"] = display_df["Value"].apply(lambda v: f"{v:,.4f}" if v is not None else "n/a")
        st.table(display_df.set_index("Symbol"))
    except Exception as e:
        st.warning(f"Live Quotes Comparison unavailable this run: {e}")

    col5, col6, col7, col8 = st.columns(4)
    dgs10 = summary.get("dgs10")
    col5.metric("10Y Nominal Yield", f"{dgs10:.2f}%" if dgs10 is not None else "n/a")
    dfii10 = summary.get("dfii10")
    col6.metric("10Y Real Yield (TIPS)", f"{dfii10:.2f}%" if dfii10 is not None else "n/a")
    curve = summary.get("curve_spread")
    col7.metric("2s10s Curve Spread", f"{curve:.2f}%" if curve is not None else "n/a")
    corr_dxy = summary.get("corr_gold_dxy_30d")
    col8.metric("30d Corr Gold/DXY", f"{corr_dxy:+.2f}" if corr_dxy is not None else "n/a", help="Expected negative")

    st.markdown("#### Price history")
    _live_px = spot_result["price"] if spot_result is not None else None
    _ref = signals["gold_spot"] if "gold_spot" in signals.columns else None
    spot_info = load_spot_history_chain(period, _ref, _live_px)
    spot_series = spot_info["series"]
    if spot_series is not None:
        chart_df = pd.DataFrame(index=signals.index.union(spot_series.index))
        chart_df["Gold Spot (XAUUSD)"] = spot_series
        gold_chart_note = (
            f"Showing true spot gold (XAUUSD), not the futures contract. Source: {spot_info['source']}"
            + ("; last point = live pulled spot" if spot_info["live_point"] else "")
            + (f"; validated against GC=F (return correlation {spot_info['corr']:.3f}, median spot/futures ratio {spot_info['ratio']:.3f})"
               if spot_info.get("corr") is not None else "") + "."
        )
    else:
        chart_df = pd.DataFrame(index=signals.index)
        chart_df["Gold (GC=F futures)"] = signals["gold_spot"] if "gold_spot" in signals.columns else None
        gold_chart_note = (
            "No spot-gold history source passed validation this run, so this is showing GC=F futures instead "
            "\u2014 they track closely but aren't identical. Tried: " + " | ".join(spot_info["notes"] or ["nothing returned"])
        )
    if "dxy" in signals.columns:
        chart_df["DXY"] = signals["dxy"]
    chart_df = chart_df.dropna(how="all")
    if not chart_df.empty:
        st.line_chart(chart_df)
    st.caption(gold_chart_note)
    if spot_series is not None and spot_info["notes"]:
        st.caption("Spot sources skipped: " + " | ".join(spot_info["notes"]))

    st.markdown("#### Divergence flags")
    st.caption("Where the textbook gold-vs-DXY / gold-vs-real-yield relationship may be breaking down.")
    if flags:
        for key, msg in flags.items():
            st.warning(msg)
    else:
        st.success("None triggered today \u2014 relationships holding roughly as expected.")

    with st.expander("Full indicator table"):
        table_rows = {
            "Gold (GC=F futures)": summary.get("gold_spot"),
            "Gold (XAUUSD spot)": summary.get("xauusd_spot"),
            "DXY": summary.get("dxy"),
            "VIX": summary.get("vix"),
            "10Y Nominal Yield": summary.get("dgs10"),
            "2Y Nominal Yield": summary.get("dgs2"),
            "10Y Real Yield (TIPS)": summary.get("dfii10"),
            "Real yield chg (5d, bps)": summary.get("real_yield_chg_5d_bps"),
            "Real yield chg (20d, bps)": summary.get("real_yield_chg_20d_bps"),
            "2s10s Curve Spread": summary.get("curve_spread"),
            "30d Corr Gold vs DXY": summary.get("corr_gold_dxy_30d"),
            "30d Corr Gold vs Real Yield": summary.get("corr_gold_realyield_30d"),
        }
        st.table(pd.DataFrame(table_rows.items(), columns=["Indicator", "Value"]).set_index("Indicator"))

    if not raw["gpr"].empty:
        with st.expander("Geopolitical Risk Index (latest)"):
            st.dataframe(raw["gpr"].tail(6))

# ---------------------------------------------------------------------------
# Tab: Regime (HMM)
# ---------------------------------------------------------------------------

with tabs[1]:
    st.subheader("3-state macro regime model (Declining / Range / Rising)")
    with st.spinner("Fitting HMM regime model..."):
        try:
            regime_result = load_regime(signals)
        except Exception as e:
            regime_result = None
            st.warning(f"Regime model failed this run (non-fatal): {e}")

    cot_result = load_cot()

    if regime_result is None:
        st.info("Not enough data yet to fit the regime model.")
    else:
        if not regime_result["model_healthy"]:
            st.warning("Model health check flagged possible overfitting -- treat this read with extra skepticism.")

        # --- ADDITIVE: compact regime confidence badge (sticky/neutral/choppy) ---
        # Moved to the top of the tab for visibility. Uses the SAME logged
        # history as the Regime History tab, but the LIVE model probability from
        # this tab's own regime_result (more current than the last logged snapshot).
        hist_log_df = load_regime_log()
        if hist_log_df is not None and regime_result is not None:
            hist_episodes = compute_regime_episodes(hist_log_df)
            live_prob = max(regime_result["current_probs"].values())
            confidence = compute_regime_confidence(hist_log_df, hist_episodes, live_top_prob=live_prob)
            if confidence is not None:
                tier_display = {
                    "sticky": ("\U0001F7E2", "Sticky / high confidence"),
                    "neutral": ("\U0001F7E1", "Neutral"),
                    "choppy": ("\U0001F534", "Choppy / low confidence"),
                }
                emoji, label = tier_display[confidence["tier"]]
                st.markdown(f"**Regime confidence (from {confidence['choppy_window_hours']}h/weekly history): {emoji} {label}**")
                st.caption(
                    f"{confidence['flips_recent']} flip(s) in the last {confidence['choppy_window_hours']}h \u2014 "
                    f"current streak {confidence['current_streak_hours']:.1f}h \u2014 "
                    f"full detail in the Regime History tab."
                )
                st.markdown("---")

        probs_df = pd.DataFrame(
            {"state": list(regime_result["current_probs"].keys()),
             "probability": [v * 100 for v in regime_result["current_probs"].values()]}
        ).set_index("state")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Current state probabilities**")
            st.bar_chart(probs_df)
        with c2:
            st.markdown("**Share of days in each state (full history)**")
            freq_df = pd.DataFrame(
                {"state": list(regime_result["state_frequency"].keys()),
                 "pct_of_days": list(regime_result["state_frequency"].values())}
            ).set_index("state")
            st.bar_chart(freq_df)

        with st.expander("Why this regime? (feature separation)"):
            st.caption(
                "For each of the model's 4 inputs: how much does its standardized "
                "value differ across the fitted states? A bigger bar means that "
                "feature does more of the work distinguishing states from each "
                "other -- this falls out of the historical data, nothing here is "
                "manually weighted."
            )
            separation = compute_feature_separation(regime_result)
            if separation is None:
                st.caption("Not available this run (need more fitted states/observations).")
            else:
                sep_df = pd.DataFrame(
                    {"feature": list(separation.keys()), "separation_score": list(separation.values())}
                ).set_index("feature")
                st.bar_chart(sep_df)
                top_feature = next(iter(separation))
                st.caption(
                    f"Highest this run: **{top_feature}** -- the feature currently doing the "
                    f"most to distinguish Declining/Range/Rising from each other."
                )

        st.markdown("**Full-history transition matrix**")
        st.dataframe(regime_result["transition_matrix"].round(3))

        if regime_result.get("recent_matrix") is not None:
            st.markdown(f"**Recent-window transition matrix** (last {regime_result['recent_window_days']} trading days)")
            st.dataframe(regime_result["recent_matrix"].round(3))

        try:
            mode = trading_mode.determine_mode(regime_result, cot_result=cot_result)
        except Exception as e:
            mode = None
            st.warning(f"Trading mode failed this run: {e}")

        if mode:
            st.markdown("### Today's Trading Mode (filter, not an entry signal)")
            m1, m2, m3 = st.columns(3)
            m1.metric("State", mode["top_state"], f"{mode['top_prob']*100:.1f}% confidence")
            m2.metric("Mode", mode["strategy_name"])
            m3.metric("Suggested size", mode["size"].split(" -- ")[0])
            st.caption(mode["strategy_note"])
            st.info(f"**Overall assessment:** {mode['overall_assessment']}")
            if cot_result is not None:
                st.caption(f"COT positioning check: {mode.get('cot_note', 'n/a')}")
                if mode.get("cot_level_interpretation"):
                    st.caption(f"COT state (3yr): {mode['cot_level_interpretation']}")

        st.markdown("---")
        st.caption(
            "Everything below this line is EXPERIMENTAL -- exploring the "
            "gold_vol_5d dominance finding. Nothing above this line is "
            "affected by either of these; both are self-contained side "
            "analyses, collapsed by default."
        )

        with st.expander("\U0001F9EA Option 1: volatility vs. its own recent history"):
            st.caption(
                "Since gold_vol_5d wins the feature-separation check every run, "
                "'which feature dominates' isn't actually informative day to "
                "day. This asks the question that DOES vary: is current "
                "volatility elevated or calm relative to its own recent range?"
            )
            vol_context = compute_volatility_context(regime_result)
            if vol_context is None:
                st.caption("Not enough history this run to compute this.")
            else:
                vc1, vc2, vc3 = st.columns(3)
                vc1.metric("Current gold_vol_5d", f"{vol_context['current']:.5f}")
                vc2.metric(
                    "Percentile vs. own history",
                    f"{vol_context['percentile_in_own_history']:.0f}th",
                    help=f"Over the last {vol_context['window_days']} trading days",
                )
                trend_arrow = "\u2191" if vol_context["trend"] == "rising" else "\u2193" if vol_context["trend"] == "falling" else "\u2192"
                trend_label = f"{trend_arrow} {vol_context['trend']}"
                if vol_context["trend_pct"] is not None:
                    trend_label += f" ({vol_context['trend_pct']:+.1f}%)"
                vc3.metric(f"Trend (vs {vol_context['trend_lookback_days']}d ago)", trend_label)
                if vol_context["percentile_in_own_history"] >= 80:
                    st.warning("Volatility is elevated relative to its own recent range -- wider stop buffers / smaller size would be consistent with this.")
                elif vol_context["percentile_in_own_history"] <= 20:
                    st.info("Volatility is calm relative to its own recent range.")

        with st.expander("\U0001F9EA Option 2: does gold_vol_5d still dominate with fair smoothing?"):
            st.caption(
                "Tests whether gold_vol_5d's dominance is genuine signal, or "
                "an artifact of comparing one smoothed statistic against three "
                "raw daily-change features (HMMs naturally favor whichever "
                "feature is most persistent when building sticky states). "
                "Refits a SEPARATE model with the same 5-day rolling treatment "
                "applied to all 4 features. Does not affect the model or "
                "trading mode used anywhere else in this app."
            )
            if st.button("Run the fair-smoothing comparison"):
                with st.spinner("Refitting with smoothed features..."):
                    try:
                        smoothed_feat_df = build_smoothed_feat_df(signals)
                        smoothed_result = hmm_regime.analyze_regime(signals, precomputed_feat_df=smoothed_feat_df) if not smoothed_feat_df.empty else None
                    except Exception as e:
                        smoothed_result = None
                        st.warning(f"Smoothed refit failed this run: {e}")

                if smoothed_result is None:
                    st.caption("Not enough smoothed history this run to compute this.")
                else:
                    smoothed_separation = compute_feature_separation(smoothed_result)
                    if smoothed_separation is None:
                        st.caption("Could not compute separation scores for the smoothed model.")
                    else:
                        sep_df = pd.DataFrame(
                            {"feature": list(smoothed_separation.keys()), "separation_score": list(smoothed_separation.values())}
                        ).set_index("feature")
                        st.bar_chart(sep_df)
                        top_feature = next(iter(smoothed_separation))
                        if top_feature == "gold_vol_5d":
                            st.warning(
                                "gold_vol_5d STILL dominates even with comparable smoothing on every "
                                "feature -- points toward genuine signal, not just a smoothing artifact."
                            )
                        else:
                            st.info(
                                f"With fair smoothing, **{top_feature}** takes over as the top "
                                f"separator instead -- consistent with the original dominance being "
                                f"at least partly a smoothing artifact, not purely economic signal."
                            )

                    st.markdown("##### Direction regime: original vs. smoothed model")
                    st.caption(
                        "The smoothed model was fit with gold_ret (not gold_vol_5d) as its "
                        "dominant separator, so its current-state read is arguably a more "
                        "direction-driven regime call. Comparing it against the original here "
                        "-- not replacing it. Neither trading mode nor anything else in this "
                        "app uses the smoothed model's read; this is diagnostic only."
                    )
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.markdown("**Original (volatility-dominant) model**")
                        orig_probs = regime_result["current_probs"]
                        orig_top_state, orig_top_prob = next(iter(orig_probs.items()))
                        st.metric("Current state", orig_top_state, f"{orig_top_prob*100:.1f}% confidence")
                        for state, prob in orig_probs.items():
                            st.caption(f"{state}: {prob*100:.1f}%")
                    with dc2:
                        st.markdown("**Smoothed (direction-dominant) model**")
                        smoothed_probs = smoothed_result["current_probs"]
                        smoothed_top_state, smoothed_top_prob = next(iter(smoothed_probs.items()))
                        st.metric("Current state", smoothed_top_state, f"{smoothed_top_prob*100:.1f}% confidence")
                        for state, prob in smoothed_probs.items():
                            st.caption(f"{state}: {prob*100:.1f}%")

                    if orig_top_state == smoothed_top_state:
                        st.success(
                            f"Both models agree: **{orig_top_state}**. The volatility-driven and "
                            f"direction-driven reads point the same way -- a more robust signal "
                            f"than either alone."
                        )
                    else:
                        st.warning(
                            f"Models DISAGREE: original says **{orig_top_state}**, smoothed says "
                            f"**{smoothed_top_state}**. Worth treating the original's directional "
                            f"framing with extra caution right now -- its dominant feature "
                            f"(gold_vol_5d) may be classifying on volatility character, not "
                            f"direction, exactly like the Aug 13-14 case."
                        )

# ---------------------------------------------------------------------------
# Tab: COT Positioning
# ---------------------------------------------------------------------------

with tabs[2]:
    st.subheader("CFTC Commitments of Traders \u2014 Managed Money, COMEX gold")
    with st.spinner("Fetching COT data from CFTC..."):
        cot_result = load_cot()

    if cot_result is None:
        st.info("COT data unavailable this run.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest week", str(cot_result["latest_date"].date()))
        c2.metric("Managed Money net", f"{cot_result['latest_mm_net']:,.0f}")
        c3.metric("Open interest", f"{cot_result['latest_open_interest']:,.0f}")
        idx3 = cot_result["cot_index"].get("3yr")
        c4.metric("COT Index (3yr)", f"{idx3:.0f}" if idx3 is not None else "n/a")

        st.markdown("**COT Index by window**")
        idx_df = pd.DataFrame(cot_result["cot_index"].items(), columns=["window", "index_0_100"]).set_index("window")
        st.bar_chart(idx_df)

        st.info(cot_result["narrative"])
        st.caption(f"Positioning level: {cot_result['level_interpretation']}")

        st.markdown("**Managed Money net positioning, recent history**")
        st.line_chart(cot_result["df"][["mm_net"]].tail(104))

# ---------------------------------------------------------------------------
# Tab: Factor Attribution
# ---------------------------------------------------------------------------

with tabs[3]:
    st.subheader("What's actually moving gold: DXY / real yields / VIX / central bank buying")
    with st.spinner("Scoring factor attribution..."):
        fa = load_factor_attribution(signals, raw["reserves"])

    if fa is None:
        st.info("Not enough clean daily history yet for a meaningful attribution.")
    else:
        reg = fa.get("regression")
        if reg:
            st.markdown(f"**Daily-move regression** (last {reg['n_days']} days) \u2014 R\u00b2 = {reg['r2']:.3f}")
            reg_df = pd.DataFrame(reg["scores"].items(), columns=["Factor", "Standardized score"]).set_index("Factor")
            st.bar_chart(reg_df)
        if fa.get("quarterly_rows"):
            st.markdown("**Quarterly regime shifts**")
            st.dataframe(pd.DataFrame(fa["quarterly_rows"]))
        if fa.get("reserves_rows"):
            st.markdown("**Central bank buying vs. gold's own price move**")
            st.dataframe(pd.DataFrame(fa["reserves_rows"]))
        if fa.get("ranking"):
            st.markdown("**Summary**")
            for line in fa["ranking"]:
                st.write(f"- {line}")

# ---------------------------------------------------------------------------
# Tab: Econ Calendar
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Upcoming releases: CPI, NFP, GDP, ISM PMI, FOMC")
    days_ahead = st.slider("Days ahead", 7, 30, 14)
    with st.spinner("Fetching FRED release calendar..."):
        try:
            cal, powell = load_calendar(days_ahead=days_ahead)
        except Exception as e:
            cal, powell = {}, []
            st.warning(f"Calendar fetch failed: {e}")

    if not cal:
        st.info("No upcoming primary releases found in this window (or FRED_API_KEY not set).")
    else:
        rows = []
        today = datetime.now().date()
        for label, dates in cal.items():
            for d in dates:
                rows.append({"Date": d.strftime("%Y-%m-%d (%a)"), "Release": label, "Today?": "\u2190 TODAY" if d == today else ""})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if powell:
        st.caption(f"Possible Fed Chair speech date(s) (best-effort, verify manually): {', '.join(powell)}")
    else:
        st.caption(
            "No Fed Chair speech detected via best-effort check -- verify directly: "
            "https://www.federalreserve.gov/newsevents/speeches.htm"
        )

# ---------------------------------------------------------------------------
# Tab: Central Bank Reserves
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Reserves pivot helper (mirrors main.py's CLI table, but returns a real
# DataFrame instead of a preformatted string, so it renders as an actual
# table with all countries as columns -- not just a tail() of one long list)
# ---------------------------------------------------------------------------

def _parse_imf_period(period_str):
    s = str(period_str).replace("M", "")
    try:
        return pd.Period(s, freq="M")
    except Exception:
        return None


def pivot_reserves_by_country(reserves_df, n_periods=12, value_fmt="usd_billions"):
    country_col, period_col, value_col, _ = reserves_utils.get_columns(reserves_df)
    if not all([country_col, period_col, value_col]):
        return None

    filtered = reserves_utils.select_single_sector_per_country(reserves_df).copy()
    filtered["_period"] = filtered[period_col].apply(_parse_imf_period)
    filtered = filtered.dropna(subset=["_period"])

    pivot = filtered.pivot_table(index="_period", columns=country_col, values=value_col, aggfunc="last")
    pivot = pivot.sort_index().tail(n_periods)
    pivot.index = pivot.index.astype(str)

    if value_fmt == "usd_billions":
        pivot = pivot.apply(lambda col: col.map(lambda v: v / 1e9 if pd.notna(v) else None))
    return pivot


with tabs[5]:
    st.subheader("Central bank gold reserves (IMF IRFCL)")
    reserves_df = raw.get("reserves")
    reserves_vol_df = raw.get("reserves_volume")
    reserves_vol_unit = raw.get("reserves_volume_unit")

    if reserves_df is None or reserves_df.empty:
        st.info("Reserves data unavailable this run (IMF API can be flaky). Manual fallback: "
                "https://www.gold.org/goldhub/data/gold-reserves-by-country")
    else:
        st.caption("USD VALUE, $ billions (mark-to-market) \u2014 moves from both actual buying/selling AND gold's own price.")
        usd_pivot = pivot_reserves_by_country(reserves_df, n_periods=12, value_fmt="usd_billions")
        if usd_pivot is not None:
            st.dataframe(usd_pivot.style.format("${:,.2f}B", na_rep="-"), use_container_width=True)
        else:
            st.warning("Could not identify country/period/value columns in the reserves data.")

    if reserves_vol_df is not None and not reserves_vol_df.empty:
        st.caption(f"PHYSICAL QUANTITY ({reserves_vol_unit or 'unit not identified'}) \u2014 immune to price moves.")
        vol_pivot = pivot_reserves_by_country(reserves_vol_df, n_periods=12, value_fmt="plain")
        if vol_pivot is not None:
            st.dataframe(vol_pivot.style.format("{:,.1f}", na_rep="-"), use_container_width=True)
        else:
            st.warning("Could not identify country/period/value columns in the volume data.")
    else:
        st.info("Could not fetch physical quantity (volume) gold reserves this run.")

# ---------------------------------------------------------------------------
# Tab: Journal
# ---------------------------------------------------------------------------

with tabs[6]:
    st.subheader("Weekly thesis journal")
    st.caption(
        "Log a falsifiable prediction each week (\"gold breaks $2,450 if real yields fall "
        "below 1.8%\" -- not \"gold will be volatile\"), then score it later to build your "
        "own track record by factor."
    )

    with st.form("thesis_form"):
        dominant_factor = st.selectbox("Dominant factor", journal.DOMINANT_FACTOR_OPTIONS)
        thesis = st.text_area("Thesis (why)")
        prediction = st.text_input("Falsifiable prediction (specific, checkable)")
        check_days = st.number_input("Check back in how many days?", min_value=1, max_value=90, value=14)
        submitted = st.form_submit_button("Log thesis")
        if submitted:
            if not thesis or not prediction:
                st.error("Thesis and prediction are both required.")
            else:
                journal.add_entry(dominant_factor, thesis, prediction, check_in_days=int(check_days))
                st.success("Thesis logged.")
                st.cache_data.clear()

    due = journal.entries_due_for_review()
    if due:
        st.markdown("### Past theses due for review")
        for row in due:
            with st.expander(f"[{row['entry_date']}] {row['dominant_factor']}"):
                st.write(f"**Thesis:** {row['thesis']}")
                st.write(f"**Prediction:** {row['falsifiable_prediction']}")
                col1, col2 = st.columns(2)
                note = st.text_input("Outcome note", key=f"note_{row['entry_date']}")
                if col1.button("Mark correct", key=f"correct_{row['entry_date']}"):
                    journal.score_entry(row["entry_date"], True, note)
                    st.rerun()
                if col2.button("Mark incorrect", key=f"incorrect_{row['entry_date']}"):
                    journal.score_entry(row["entry_date"], False, note)
                    st.rerun()

    st.markdown("### Hit-rate scoreboard")
    st.text(journal.hit_rate_summary())

# ---------------------------------------------------------------------------
# Tab: GEX (Options) -- independent engine, own product/data source
# ---------------------------------------------------------------------------

with tabs[7]:
    st.subheader("Dealer Gamma Exposure (GEX) \u2014 GLD options")
    st.caption(
        "Separate engine from the macro system above: pulls a live options chain "
        "for ONE product (GLD by default) and estimates dealer positioning from it. "
        "**Key assumption, unverified against real dealer books:** customers are "
        "assumed net long calls and net long puts, dealers net short both \u2014 the "
        "standard simplifying convention most public GEX approaches use, not a fact "
        "about this specific chain. Treat everything below as a diagnostic overlay, "
        "not a signal on its own."
    )

    gcol1, gcol2, gcol3 = st.columns([1, 1, 1])
    ticker = gcol1.text_input("Ticker", value="GLD")
    min_oi_input = gcol2.number_input("Min open interest filter", min_value=0, value=0, step=10)
    use_md = False
    if config.MARKETDATA_API_KEY:
        use_md = gcol3.toggle("Use MarketData.app", value=True, help="More reliable OI/Greeks than the free yfinance fallback.")
    else:
        gcol3.caption("Using yfinance (free). Add MARKETDATA_API_KEY as a secret for more reliable OI/Greeks.")

    expiration_choice = None
    if not use_md:
        try:
            expirations = load_gex_expirations(ticker)
        except Exception as e:
            expirations = []
            st.warning(f"Could not list expirations for {ticker}: {e}")
        if expirations:
            exp_label = st.selectbox(
                "Expiration",
                ["Nearest with usable open interest (auto)"] + expirations,
            )
            expiration_choice = None if exp_label.startswith("Nearest") else exp_label

    run_gex = st.button("\U0001F504 Run / refresh GEX assessment")

    gex_state_key = f"gex_result_{ticker}_{expiration_choice}_{min_oi_input}_{use_md}"
    gex_note_key = gex_state_key + "_note"
    if run_gex or gex_state_key not in st.session_state:
        st.session_state[gex_note_key] = None
        live_result, live_source = None, None
        errors_seen = []
        try:
            with st.spinner(f"Fetching {ticker} options chain and computing GEX..."):
                live_result = load_gex_assessment(ticker, expiration_choice, min_oi_input, use_md)
            live_source = "MarketData.app" if use_md else "yfinance"
        except Exception as e:
            errors_seen.append(f"{'MarketData.app' if use_md else 'yfinance'}: {e}")
            # ADDITIVE fallback 1: MarketData.app failed (rate limit etc.) -> try free yfinance
            if use_md:
                try:
                    with st.spinner("MarketData.app failed -- trying yfinance instead..."):
                        live_result = load_gex_assessment(ticker, expiration_choice, min_oi_input, False)
                    live_source = "yfinance (MarketData.app failed)"
                    st.session_state[gex_note_key] = (
                        "warning",
                        f"MarketData.app failed ({errors_seen[0]}). Showing a LIVE yfinance pull instead.",
                    )
                except Exception as e2:
                    errors_seen.append(f"yfinance: {e2}")

        if live_result is not None:
            st.session_state[gex_state_key] = live_result
            # ADDITIVE: remember this as the last good pull (never raises)
            gex_snapshot.save_payload(
                f"{ticker.upper()}|main",
                gex_snapshot.payload_from_result(live_result, expiration_choice, live_source),
            )
        else:
            st.session_state[gex_state_key] = None
            # ADDITIVE fallback 2: last successful pull (local file or data branch)
            snap, origin = gex_snapshot.load_best_payload(f"{ticker.upper()}|main", load_remote_gex_store())
            if snap is not None:
                try:
                    st.session_state[gex_state_key] = gex_snapshot.result_from_payload(snap)
                    when, age = gex_snapshot.describe_age(snap.get("saved_utc"))
                    st.session_state[gex_note_key] = (
                        "error",
                        f"LIVE FETCH FAILED -- showing the LAST SUCCESSFUL pull instead: "
                        f"**{when}** ({age}; source {snap.get('source', 'unknown')}; stored {origin}). "
                        f"Nothing below is live. Failure detail: {' | '.join(errors_seen)}",
                    )
                except Exception as e3:
                    errors_seen.append(f"snapshot unreadable: {e3}")
            if st.session_state[gex_state_key] is None:
                st.error(
                    f"GEX assessment failed: {' | '.join(errors_seen)}\n\n"
                    "Common causes: no listed options for this ticker, no expiration with "
                    "usable open interest right now, or (for the free yfinance path) Yahoo "
                    "rate-limiting. Try a different expiration, or lower/remove the min "
                    "open interest filter. No saved snapshot exists yet for this ticker. "
                    "The GEX (Alternative) tab uses a different data source and may still work."
                )

    result = st.session_state.get(gex_state_key)
    _gex_note = st.session_state.get(gex_note_key)
    if _gex_note and result is not None:
        (st.error if _gex_note[0] == "error" else st.warning)(_gex_note[1])

    if result is not None:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Spot", f"{result.spot:.2f}")
        m2.metric("Net GEX", f"{result.net_gex:,.0f}")
        m3.metric("Gamma Flip", f"{result.gamma_flip:.2f}" if result.gamma_flip else "not found")
        m4.metric("Dealer Delta", f"{result.dealer_delta:,.0f}")
        m5.metric("Regime", result.regime.split(" (")[0])
        st.caption(result.regime)

        st.markdown("#### GEX by strike (near spot)")
        gbs = result.gex_by_strike.copy()
        window = gbs[(gbs.index >= result.spot * 0.85) & (gbs.index <= result.spot * 1.15)]
        if not window.empty:
            st.bar_chart(window[["CallGEX", "PutGEX"]])
        else:
            st.bar_chart(gbs[["CallGEX", "PutGEX"]])
        st.caption("Chart zoomed to \u00b115% of spot for readability; raw walls below cover the full chain.")

        st.markdown("##### Self-calculated nearest levels for reference")
        st.caption(
            "The closest-to-spot strikes that still show a real GEX concentration "
            "(not just any strike within reach) \u2014 filters out small/noise levels "
            "so the list favors visually significant bars near spot, not every bar."
        )

        # Pull live reference EARLY so this section can use the actual current
        # price for "% from spot" / above-below framing, instead of the chain
        # snapshot's spot -- which can go stale fast if there's been a big
        # move since the chain was fetched (options chains refresh far less
        # often than the underlying's own live price does).
        live_gld, live_xau = load_gex_live_refs()
        reference_spot = result.spot
        deviation_pct = None
        if ticker.upper() == "GLD" and live_gld:
            reference_spot = live_gld
            deviation_pct = abs(live_gld - result.spot) / result.spot * 100
        elif ticker.upper() != "GLD":
            # For a non-GLD ticker we don't have a matching "live" quote source
            # wired up (load_gex_live_refs() is GLD/XAUUSD-specific) -- fall
            # back to the chain's own spot rather than mixing tickers.
            reference_spot = result.spot

        DEVIATION_WARN_THRESHOLD_PCT = 0.5
        if deviation_pct is not None and deviation_pct >= DEVIATION_WARN_THRESHOLD_PCT:
            st.warning(
                f"\u26a0\ufe0f Live {ticker} ({live_gld:.2f}) has moved {deviation_pct:.2f}% from this chain "
                f"snapshot's spot ({result.spot:.2f}) \u2014 a real move since the options data was fetched, "
                f"not just refresh lag. The levels below are computed relative to the LIVE price so the "
                f"'% from spot' / above-below framing stays accurate; the wall strikes themselves are "
                f"unaffected (options strikes don't move), only which side of price they're currently on can "
                f"shift after a big move like this."
            )
        elif deviation_pct is not None:
            st.caption(f"Live {ticker} vs. chain snapshot: {deviation_pct:.2f}% apart \u2014 using live price as the reference below.")

        nearest = find_nearest_gex_clusters(gbs, reference_spot)
        oz_for_nearest = None
        if ticker.upper() == "GLD":
            try:
                oz_for_nearest = gold_comparison.get_oz_per_share()
            except Exception:
                oz_for_nearest = None
        if nearest.empty:
            st.caption("Nothing near spot cleared the meaningful-size threshold this run.")
        else:
            for strike, row in nearest.iterrows():
                dist = row["DistFromSpot"]
                side = "above spot (resistance-leaning)" if dist > 0 else "below spot (support-leaning)" if dist < 0 else "essentially AT spot (pin risk)"
                call_gex, put_gex = row["CallGEX"], row["PutGEX"]
                gross_gex = row["GrossGEX"]
                net_gex = call_gex + put_gex
                if abs(call_gex) > abs(put_gex) * 1.3:
                    lean = "call-dominant"
                elif abs(put_gex) > abs(call_gex) * 1.3:
                    lean = "put-dominant"
                elif gross_gex > 0 and abs(net_gex) < gross_gex * 0.3:
                    lean = "two-sided / battleground (large call AND put both, net mostly cancels)"
                else:
                    lean = "mixed call/put"
                pct_away = (dist / reference_spot) * 100
                line = (
                    f"**{strike:.2f}** ({pct_away:+.2f}% from spot, {side}, {lean}) \u2014 "
                    f"Gross GEX {gross_gex:,.0f} (Net {net_gex:,.0f})"
                )
                if oz_for_nearest:
                    spot_equiv = strike / oz_for_nearest
                    line += f"  \u2192 spot XAUUSD equiv \u2248 **{spot_equiv:,.0f}** (oz/share used {oz_for_nearest:.6f})"
                st.markdown(line)

        st.markdown("#### Contextual levels (spot-aware support/resistance)")
        try:
            ctx = result.contextual_levels()
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**ATM pin:** {ctx['atm_pin'][0]:.2f}  (GEX {ctx['atm_pin'][1]:,.0f})")
                st.write(
                    f"**Largest wall overall:** {ctx['largest_wall_overall']['strike']:.2f}  "
                    f"(GEX {ctx['largest_wall_overall']['gex']:,.0f}, "
                    f"{ctx['largest_wall_overall']['side']})"
                )
                st.write("**Resistance (above spot):**")
                if ctx["resistance"]:
                    st.table(pd.DataFrame(ctx["resistance"], columns=["Strike", "GEX"]))
                else:
                    st.caption("None found above spot in this chain.")
            with c2:
                st.write("**Support (below spot):**")
                if ctx["support"]:
                    st.table(pd.DataFrame(ctx["support"], columns=["Strike", "GEX"]))
                else:
                    st.caption("None found below spot in this chain.")
        except Exception as e:
            st.warning(f"Contextual levels unavailable: {e}")

        st.markdown("#### Dealer positioning context")
        try:
            conc = gex_engine.compute_gex_concentration(result)
            st.write(
                f"**GEX concentration:** net/gross ratio {conc['ratio']:.1%} "
                f"(gross {conc['gross_gex']:,.0f}, net {conc['net_gex']:,.0f})"
            )
            st.caption(conc["narrative"])
        except Exception as e:
            st.warning(f"GEX concentration unavailable: {e}")

        try:
            delta_ctx = gex_engine.compute_dealer_delta_context(result)
            st.write(
                f"**Dealer delta vs. 10-day avg volume:** {delta_ctx['ratio']:.1%} "
                f"({delta_ctx['dealer_delta']:,.0f} vs {delta_ctx['avg_volume_10d']:,.0f})"
            )
            st.caption(delta_ctx["size_narrative"])
            st.caption(delta_ctx["direction_narrative"])
        except Exception as e:
            st.caption(f"Dealer delta / volume comparison unavailable this run: {e}")

        st.markdown("#### Live reference (independent of chain snapshot age)")
        lc1, lc2 = st.columns(2)
        lc1.metric(f"Live {ticker}", f"{live_gld:.2f}" if live_gld is not None else "n/a")
        lc2.metric("Live XAUUSD spot (gold-api.com)", f"{live_xau:.2f}" if live_xau is not None else "n/a")
        st.caption(
            "Compare against 'Spot' above \u2014 if these differ noticeably, the chain "
            "the GEX numbers were computed from is an older snapshot than right now."
        )

        if ticker.upper() == "GLD":
            st.markdown("#### Spot Gold Equivalent (proxy)")
            st.caption(
                "GLD's structure translated into $/oz using GLD price = OzPerShare \u00d7 "
                "Spot. This is a PROXY \u2014 GLD's own gamma walls converted to spot units "
                "\u2014 not independently observed COMEX/spot options data. The oz/share "
                "ratio drifts slowly (fund expenses), so it's re-derived live from "
                "Live GLD / Live spot each run (cached ~24h so it doesn't jitter) "
                "rather than pulled from a fixed constant."
            )
            try:
                proxy = gold_comparison.convert_result_to_spot_gold_terms(result)
                p1, p2, p3 = st.columns(3)
                p1.metric("Spot Gold Equiv", f"{proxy['spot_gold_equivalent']:.2f}")
                p2.metric(
                    "Gamma Flip Equiv",
                    f"{proxy['gamma_flip_spot_equivalent']:.2f}"
                    if proxy["gamma_flip_spot_equivalent"] is not None else "not found",
                )
                p3.metric("oz/share used", f"{proxy['oz_per_share_used']:.6f}")

                pc1, pc2 = st.columns(2)
                with pc1:
                    st.write("**Resistance (call walls, spot-equivalent):**")
                    st.table(pd.DataFrame(proxy["call_walls_spot_equivalent"], columns=["Spot Level", "GEX"]))
                with pc2:
                    st.write("**Support (put walls, spot-equivalent):**")
                    st.table(pd.DataFrame(proxy["put_walls_spot_equivalent"], columns=["Spot Level", "GEX"]))

                try:
                    ctx = result.contextual_levels()
                    oz = proxy["oz_per_share_used"]
                    atm_strike, atm_gex = ctx["atm_pin"]
                    largest = ctx["largest_wall_overall"]
                    st.write(
                        f"**ATM Pin (spot-equivalent):** {atm_strike / oz:.2f}  (GEX {atm_gex:,.0f})"
                    )
                    st.write(
                        f"**Largest wall overall (spot-equivalent):** {largest['strike'] / oz:.2f}  "
                        f"(GEX {largest['gex']:,.0f}, {largest['side']})"
                    )
                    scx1, scx2 = st.columns(2)
                    with scx1:
                        st.write("**Resistance (spot-equivalent, above spot):**")
                        if ctx["resistance"]:
                            st.table(pd.DataFrame(
                                [(s / oz, g) for s, g in ctx["resistance"]],
                                columns=["Spot Level", "GEX"],
                            ))
                        else:
                            st.caption("None found above spot in this chain.")
                    with scx2:
                        st.write("**Support (spot-equivalent, below spot):**")
                        if ctx["support"]:
                            st.table(pd.DataFrame(
                                [(s / oz, g) for s, g in ctx["support"]],
                                columns=["Spot Level", "GEX"],
                            ))
                        else:
                            st.caption("None found below spot in this chain.")
                except Exception as e:
                    st.caption(f"Spot-equivalent contextual levels unavailable: {e}")
            except Exception as e:
                st.warning(f"Spot gold equivalent conversion unavailable this run: {e}")
        else:
            st.caption(
                f"Spot Gold Equivalent proxy only applies to GLD (converts GLD's share-price "
                f"structure to $/oz) \u2014 not shown for {ticker}."
            )

        with st.expander("Raw walls (unfiltered by spot, diagnostic)"):
            st.write("**Call walls (largest CallGEX):**")
            st.table(pd.DataFrame(result.call_walls, columns=["Strike", "GEX"]))
            st.write("**Put walls (largest |PutGEX|):**")
            st.table(pd.DataFrame(result.put_walls, columns=["Strike", "GEX"]))
    else:
        st.info("Click 'Run / refresh GEX assessment' above to fetch a live chain and compute GEX.")

# ---------------------------------------------------------------------------
# Tab: Regime History -- reads the log the GitHub Actions cron maintains
# ---------------------------------------------------------------------------

with tabs[8]:
    st.subheader("Regime stability over the past week")
    st.caption(
        "Logged every 15 minutes, weekdays only, by a separate scheduled job \u2014 "
        "not the live dashboard's own regime read, which can differ slightly if "
        "market data has moved since the last log entry. Stored in UTC, shown "
        "here converted to Singapore time (SGT, UTC+8)."
    )

    log_df = load_regime_log()

    if log_df is None:
        st.info(
            "No regime history yet. This fills in automatically once the "
            "regime-logger GitHub Action has run at least once (every 15 min, "
            "weekdays) \u2014 check back after that, or trigger it manually from "
            "the repo's Actions tab."
        )
    else:
        log_df = log_df.copy()
        log_df["timestamp_sgt"] = log_df["timestamp_utc"].dt.tz_convert("Asia/Singapore")

        episodes = compute_regime_episodes(log_df)  # duration math is tz-agnostic, uses timestamp_utc internally
        episodes["start_sgt"] = episodes["start"].dt.tz_convert("Asia/Singapore")
        episodes["end_sgt"] = episodes["end"].dt.tz_convert("Asia/Singapore")

        # --- ADDITIVE: regime confidence tier (sticky/neutral/choppy) ---
        # Purely additive on top of everything above -- doesn't change any
        # existing metric, chart, or table in this tab.
        confidence = compute_regime_confidence(log_df, episodes)
        if confidence is not None:
            tier_display = {
                "sticky": ("\U0001F7E2", "Sticky / high confidence"),
                "neutral": ("\U0001F7E1", "Neutral"),
                "choppy": ("\U0001F534", "Choppy / low confidence"),
            }
            emoji, label = tier_display[confidence["tier"]]
            st.markdown(f"#### {emoji} {label}")
            dwell_str = (
                f"{confidence['typical_dwell_hours']:.1f}h"
                if confidence["typical_dwell_hours"] is not None and confidence["has_prior_occurrence"]
                else "not enough history yet for this state"
            )
            st.caption(
                f"{confidence['current_state']}, {confidence['current_streak_hours']:.1f}h streak "
                f"(this state's typical dwell time this week: {dwell_str}) \u2014 "
                f"{confidence['flips_recent']} flip(s) in the last {confidence['choppy_window_hours']}h \u2014 "
                f"model probability {confidence['live_top_prob']*100:.0f}%"
            )
            if confidence["tier"] == "choppy":
                st.warning(
                    "Regime has flipped multiple times recently \u2014 treat the current state as noisy, "
                    "not a confirming macro layer for a setup right now."
                )
            elif confidence["tier"] == "sticky":
                st.success(
                    "Current state has held longer than its typical duration this week, with high model "
                    "confidence \u2014 reasonable to lean on this as a supporting macro layer."
                )
            else:
                st.caption("Neither clearly stable nor clearly choppy \u2014 treat as background context, not a confirming or disqualifying signal either way.")

        latest = log_df.iloc[-1]
        current_episode = episodes.iloc[-1] if not episodes.empty else None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current state (last logged)", latest["top_state"])
        m1.caption(f"as of {latest['timestamp_sgt'].strftime('%Y-%m-%d %H:%M')} SGT")
        if current_episode is not None:
            streak_hrs = current_episode["duration_minutes"] / 60
            m2.metric("Current streak", f"{streak_hrs:.1f}h")
        today_sgt = pd.Timestamp.now(tz="Asia/Singapore").normalize()
        changes_today = int((episodes["start_sgt"].dt.normalize() == today_sgt).sum()) if not episodes.empty else 0
        m3.metric("Changes today", changes_today)
        if not episodes.empty:
            avg_dwell_hrs = episodes["duration_minutes"].mean() / 60
            m4.metric("Avg dwell time (week)", f"{avg_dwell_hrs:.1f}h")

        st.markdown("#### Timeline (color = regime state)")
        try:
            import altair as alt
            heat_df = log_df.copy()
            heat_df["date"] = heat_df["timestamp_sgt"].dt.strftime("%Y-%m-%d (%a)")
            heat_df["time_of_day"] = heat_df["timestamp_sgt"].dt.strftime("%H:%M")
            chart = alt.Chart(heat_df).mark_rect().encode(
                x=alt.X("time_of_day:O", title="Time of day (SGT)", sort=None),
                y=alt.Y("date:O", title=None, sort=None),
                color=alt.Color(
                    "top_state:N",
                    title="Regime",
                    scale=alt.Scale(
                        domain=["Declining", "Range", "Rising"],
                        range=["#E24B4A", "#888780", "#639922"],
                    ),
                ),
                tooltip=["timestamp_sgt:T", "top_state:N", "top_prob:Q"],
            ).properties(height=28 * heat_df["date"].nunique() + 40)
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.warning(f"Timeline chart unavailable: {e}")

        st.markdown("#### Changes per day")
        if not episodes.empty:
            episodes_by_day = episodes.copy()
            episodes_by_day["day"] = episodes_by_day["start_sgt"].dt.strftime("%Y-%m-%d (%a)")
            changes_per_day = episodes_by_day.groupby("day").size().rename("Regime changes")
            st.bar_chart(changes_per_day)
        else:
            st.caption("Not enough history yet to compute changes per day.")

        with st.expander("Raw log (last 100 rows)"):
            display_df = log_df[["timestamp_sgt", "top_state", "top_prob", "prob_declining", "prob_range", "prob_rising"]].tail(100).copy()
            display_df["timestamp_sgt"] = display_df["timestamp_sgt"].dt.strftime("%Y-%m-%d %H:%M SGT")
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# ===========================================================================
# ADDITIVE: Macro Confidence tab
# ===========================================================================

with macro_conf_tab:
    st.subheader("Macro Confidence")
    st.caption(
        "One place to see how much corroboration sits behind today's trading mode. It pulls together "
        "checks the app already runs elsewhere (HMM conviction, model health, recent-vs-full persistence, "
        "the logged sticky/choppy tier, the smoothed direction-regime cross-check, COT crowding, "
        "volatility percentile). **It is a rule-based checklist, not a calibrated probability, and the "
        "layers are not independent** (several come from the same HMM), so '5 of 6 agree' is weaker "
        "evidence than it sounds. It does not change the mode or size. GEX is deliberately excluded — "
        "that is a separate dealer-positioning overlay, not part of the macro regime."
    )

    try:
        mc_regime = load_regime(signals)
    except Exception as e:
        mc_regime = None
        st.warning(f"Regime model failed this run (non-fatal): {e}")

    try:
        mc_cot = load_cot()
    except Exception:
        mc_cot = None

    if mc_regime is None:
        st.info("Not enough data yet to fit the regime model.")
    else:
        try:
            mc_mode = trading_mode.determine_mode(mc_regime, cot_result=mc_cot)
        except Exception as e:
            mc_mode = None
            st.warning(f"Trading mode failed this run: {e}")

        if mc_mode:
            # Logged persistence tier (same helpers the Regime tab badge uses)
            mc_persist = None
            try:
                _log = load_regime_log()
                if _log is not None:
                    _eps = compute_regime_episodes(_log)
                    mc_persist = compute_regime_confidence(
                        _log, _eps, live_top_prob=max(mc_regime["current_probs"].values())
                    )
            except Exception:
                mc_persist = None

            # Smoothed (direction-dominant) model: opt-in, because it's an extra HMM fit
            st.markdown("##### Direction-regime cross-check")
            if st.button("Run / refresh the smoothed-model cross-check", key="mc_run_smoothed"):
                st.session_state["mc_smoothed_requested"] = True
            mc_smoothed = None
            if st.session_state.get("mc_smoothed_requested"):
                with st.spinner("Refitting with fair smoothing (cached for 15 min)..."):
                    try:
                        mc_smoothed = load_smoothed_regime(signals)
                        if mc_smoothed is None:
                            st.caption("Not enough smoothed history this run.")
                    except Exception as e:
                        st.warning(f"Smoothed refit failed this run: {e}")
            else:
                st.caption(
                    "Not run yet — it refits a second HMM, so it is opt-in. Until you run it, that layer "
                    "shows as n/a and is excluded from the tally."
                )

            try:
                mc_vol = compute_volatility_context(mc_regime)
            except Exception:
                mc_vol = None

            layers = macro_confidence.build_layers(mc_regime, mc_mode, mc_persist, mc_smoothed, mc_vol)
            tally = macro_confidence.summarize(layers)

            st.markdown("---")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Regime", mc_mode["top_state"], f"{mc_mode['top_prob']*100:.0f}% (HMM)")
            k2.metric("Mode", mc_mode["strategy_name"].split(" (")[0].title())
            k3.metric("Suggested size", mc_mode["size"].split(" -- ")[0])
            k4.metric("Layer agreement", tally["label"],
                      f"{tally['supports']}/{tally['evaluable']} support · {tally['conflicts']} conflict")
            st.info(f"**Overall assessment:** {mc_mode['overall_assessment']}")

            st.markdown("##### Checklist")
            chk = pd.DataFrame([
                {"": macro_confidence.ICON[l["status"]], "Layer": l["layer"],
                 "Status": l["status"], "Detail": l["detail"]} for l in layers
            ])
            st.dataframe(chk, hide_index=True, use_container_width=True)
            st.caption(
                f"{tally['supports']} support · {tally['caution']} caution · {tally['conflicts']} conflict "
                f"· {tally['not_evaluable']} not evaluable / neutral (excluded). "
                "Label rule: Aligned = no caution or conflict; Conflicted = 2+ conflicts or ≥40% of evaluable "
                "layers; otherwise Mixed. Those cut-offs are a convention, not statistically derived."
            )

            if mc_smoothed is not None:
                st.markdown("##### Original vs smoothed model")
                oc, sc = st.columns(2)
                for col, title, res in ((oc, "Original (volatility-dominant)", mc_regime),
                                        (sc, "Smoothed (direction-dominant)", mc_smoothed)):
                    with col:
                        st.markdown(f"**{title}**")
                        pr = res["current_probs"]
                        top = max(pr, key=pr.get)
                        st.metric("Current state", top, f"{pr[top]*100:.1f}%")
                        st.bar_chart(pd.DataFrame({"probability": [v * 100 for v in pr.values()]}, index=list(pr.keys())))

            if mc_cot is not None and mc_mode.get("cot_surprise_note"):
                st.caption(mc_mode["cot_surprise_note"])
            if mc_persist is not None:
                st.caption("Persistence detail and the flip timeline live in the Regime History tab.")


# ===========================================================================
# ADDITIVE: GEX (Alternative) tab -- CBOE + GVZ, no MarketData.app
#
# Two cadences, kept separate on purpose:
#   * option chain / GEX snapshot -> original GEX timing (GEX_CACHE_TTL, 10 min in-app;
#     the scheduled job also stores one every ~30 min as the fallback)
#   * GLD<->spot basis            -> every 15 min (BASIS_CACHE_TTL + scheduled job),
#     used only to translate GLD levels into XAUUSD terms
# ===========================================================================

def _fmt_level(x):
    return "not found" if x is None else f"{x:,.2f}"


def _to_xau(level, factor):
    return None if (level is None or factor is None) else level * factor


def _basis_panel(basis_choice, live_pt, history, stable_oz, chain_spot=None):
    """Shows the basis in use, how it compares with the 24h-stable oz/share, and its recent history."""
    st.markdown("#### GLD → spot-gold basis (15-minute capture)")
    st.caption(
        "Strikes don't move, but the factor that turns a GLD level into a $/oz level does. The original tab uses "
        "an oz/share ratio cached for 24h (stable on purpose). This panel also captures the LIVE GLD ÷ XAUUSD ratio "
        "every 15 minutes, so you can see whether the stable ratio has drifted from the market, and how many dollars "
        "of $/oz that is worth. GLD trades US hours only while spot trades ~24h, so a ratio taken while GLD is closed "
        "would show a fake basis move — those points are stored but never used to convert."
    )
    pt = basis_choice.get("point") or live_pt or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GLD", f"{pt['gld']:,.2f}" if pt.get("gld") else "n/a")
    c2.metric("XAUUSD (spot)", f"{pt['xau']:,.2f}" if pt.get("xau") else "n/a")
    c3.metric("Live oz/share (GLD÷XAU)", f"{pt['ratio']:.6f}" if pt.get("ratio") else "n/a")
    c4.metric("24h-stable oz/share", f"{stable_oz:.6f}" if stable_oz else "n/a")
    gcb = pt.get("gc_basis_pct")
    c5.metric("GC=F − spot basis", f"{gcb:+.2f}%" if gcb is not None else "n/a")

    st.markdown("**Spot price: pulled vs self-calculated, with the deviation**")
    st.caption(
        "PULLED = XAUUSD straight from the gold-api feed. SELF-CALCULATED = GLD price ÷ the 24h-stable oz/share "
        "(what the original GEX tab uses). The deviation columns are measured against the pulled spot. A self-calculated "
        "row is withheld while GLD isn't trading, because GLD's stale price against a live spot would be a fake gap."
    )
    _rr = pd.DataFrame(gld_basis.reconcile_rows(pt, stable_oz, chain_spot))
    st.dataframe(_rr.round(2), hide_index=True, use_container_width=True)
    _sc = _rr[_rr["Measure"].str.contains("SELF")]
    if len(_sc) and _sc.iloc[0]["Value"] == _sc.iloc[0]["Value"] and _sc.iloc[0]["Value"] is not None:
        _d = _sc.iloc[0]["Dev vs pulled ($/oz)"]
        st.caption(
            f"Self-calculated spot is {_d:+.2f} $/oz ({_sc.iloc[0]['Dev vs pulled (bps)']:+.1f} bps) from the pulled spot right now. "
            f"Any level converted with the stable ratio carries roughly that much error at today's price."
        )

    src = basis_choice["source"]
    if src == "none":
        st.warning("No basis available — spot-equivalent levels are unavailable this run.")
    else:
        when, age = (gex_snapshot.describe_age(basis_choice["point"]["ts_utc"]) if basis_choice.get("point") else ("n/a", "n/a"))
        st.info(f"Converting with: **{basis_choice['note']}** — captured {when} ({age}).")
        if pt.get("ratio") and stable_oz:
            diff_bps = (pt["ratio"] / stable_oz - 1) * 1e4
            lvl = pt.get("xau") or 0
            st.caption(
                f"Live ratio vs 24h-stable: {diff_bps:+.1f} bps → about ${abs(diff_bps) / 1e4 * lvl:,.2f}/oz on a level at "
                f"{lvl:,.0f}. If that is small the stable ratio is fine; if it grows, the faster basis is doing real work."
            )
    if live_pt and not live_pt.get("ratio_valid"):
        st.caption("Live pairing not valid right now (GLD hasn't printed recently — market closed or halted), so the last valid point is used.")
    if live_pt and live_pt.get("errors"):
        st.caption("Basis capture notes: " + " | ".join(live_pt["errors"]))

    valid = [p for p in (history or []) if p.get("ratio_valid") and p.get("ratio")]
    if len(valid) >= 2:
        hdf = pd.DataFrame({"oz/share (live)": [p["ratio"] for p in valid]},
                           index=pd.to_datetime([p["ts_utc"] for p in valid], utc=True).tz_convert("Asia/Singapore"))
        st.markdown("**Logged live oz/share (SGT)**")
        st.line_chart(hdf)
        gcs = [(p["ts_utc"], p["gc_basis_pct"]) for p in (history or []) if p.get("gc_basis_pct") is not None]
        if len(gcs) >= 2:
            st.markdown("**Logged GC=F − spot basis, % (SGT)**")
            st.line_chart(pd.DataFrame({"GC basis %": [v for _, v in gcs]},
                                       index=pd.to_datetime([t for t, _ in gcs], utc=True).tz_convert("Asia/Singapore")))
        _ds = gld_basis.deviation_series(history, stable_oz)
        if len(_ds) >= 2:
            st.markdown("**Logged deviation: self-calculated − pulled spot, $/oz (SGT)**")
            st.line_chart(pd.DataFrame({"self-calculated − pulled ($/oz)": [v for _, v in _ds]},
                                       index=pd.to_datetime([t for t, _ in _ds], utc=True).tz_convert("Asia/Singapore")))
            _vals = [v for _, v in _ds]
            st.caption(f"Over {len(_vals)} logged points: mean {sum(_vals) / len(_vals):+.2f}, range {min(_vals):+.2f} to {max(_vals):+.2f} $/oz.")
    else:
        st.caption("Basis history will appear here once the scheduled job has logged a few 15-minute points.")


def _render_gex_alt_view(view, spot, gvz, sym, contracts=None, basis=None, stable_oz=None, avg_volume=None):
    """One DTE selection. `view` is gex_alt.compute_view output (live) or its JSON round-trip (snapshot).
    basis = gld_basis.choose_basis(...) result; spot-equivalent columns appear when it has a factor."""
    gx = view["gex"] if isinstance(view["gex"], pd.DataFrame) else gex_snapshot.df_from_json(view["gex"])
    t_years = view["t_years"]
    f = (basis or {}).get("spot_per_gld")                       # $/oz per GLD $
    f_stable = (1.0 / stable_oz) if stable_oz else None
    has_narr = "regime_own" in view                              # old snapshots predate the narratives

    # the expected-move used for "is this wall in play"
    move, move_label = None, ""
    if view.get("straddle"):
        move, move_label = view["straddle"]["one_sigma"], "ATM straddle 1σ"
    elif view.get("atm_iv_own"):
        move, move_label = gex_alt.expected_move(spot, view["atm_iv_own"], t_years), "own ATM-IV 1σ"
    elif gvz is not None:
        move, move_label = gex_alt.expected_move(spot, gvz / 100.0, t_years), "GVZ 1σ"

    # ---- 1. What it adds up to ------------------------------------------------
    if has_narr:
        st.markdown("#### What this adds up to")
        st.caption("Reading aids built from the numbers below — heuristics, not signals. Same stance as the original GEX tab.")
        for it in gex_alt.build_read(view, spot, move, move_label):
            (st.warning if it["level"] == "warn" else st.markdown)(it["text"] if it["level"] == "warn" else f"- {it['text']}")

        st.markdown("#### Gamma regime")
        rc1, rc2 = st.columns(2)
        for col, title, key, flipkey in ((rc1, "CBOE Greeks", "regime_cboe", "flip_cboe_iv"),
                                         (rc2, "Self-calculated Greeks", "regime_own", "flip_own_iv")):
            with col:
                r_ = view[key]
                st.markdown(f"**{title}**")
                st.metric("Regime", r_["label"].split(" (")[0])
                st.caption(r_["label"])
                st.caption(r_["meaning"])
                if r_.get("flip_text"):
                    st.caption(r_["flip_text"])
                    if (f or f_stable) and view.get(flipkey):
                        _p = f"{view[flipkey] * f:,.0f}" if f else "n/a"
                        _c = f"{view[flipkey] * f_stable:,.0f}" if f_stable else "n/a"
                        st.caption(f"Flip in spot-gold terms: ≈ {_p} (pulled basis) | {_c} (self-calculated) XAUUSD")

    # ---- 2. Expected range ----------------------------------------------------
    st.markdown("#### Expected range for this DTE")
    st.caption(
        "1σ move = Spot × vol × √T (T in years to the expiry, or to the cumulative DTE). GVZ is CBOE's 30-day "
        "constant-maturity implied vol on GLD options, so it is a *30-day* number applied to a shorter horizon; the "
        "ATM rows are term-specific. The straddle row is read straight off traded prices (ATM call + put ≈ 0.798σ√T·S), "
        "so it needs no vol or time convention — if it disagrees with the IV rows, trust it over them. Implied vol usually "
        "runs above realised vol, so these ranges tend to be wider than what actually happens; 1σ ≈ 68% only for "
        "normal returns, and gold's tails are fatter."
    )
    sm = view.get("straddle")
    rt = gex_alt.range_table(spot, t_years, gvz, view.get("atm_iv_cboe"), view.get("atm_iv_own"),
                             None if not sm else sm["one_sigma"])
    for c_ in ("1σ low", "1σ high"):
        if f:
            rt[c_ + " (XAUUSD, pulled basis)"] = rt[c_] * f
        if f_stable:
            rt[c_ + " (XAUUSD, self-calc)"] = rt[c_] * f_stable
    st.dataframe(rt.round(2), hide_index=True, use_container_width=True)
    if sm:
        st.caption(f"Straddle uses the {sm['strike']:,.2f} strike: premium {sm['straddle']:.2f} → 1σ ≈ {sm['one_sigma']:.2f} GLD $.")

    # ---- 3. Side-by-side GEX --------------------------------------------------
    st.markdown("#### Dealer gamma: CBOE Greeks vs self-calculated Greeks")
    st.caption(
        "Same open interest and the same sign convention (calls +, puts −; dealers assumed short both — an "
        "assumption, not observed dealer books). Only the Greeks differ. CBOE publishes no gamma flip, so "
        "BOTH flips below are model-based (our Black-Scholes, fed with our IV vs CBOE's IV)."
    )
    lo, hi = spot * 0.92, spot * 1.08
    cc, oc = st.columns(2)
    for col, title, tag, flipkey, ddkey, lvkey in ((cc, "CBOE Greeks", "cboe", "flip_cboe_iv", "dealer_delta_cboe", "levels_cboe"),
                                                    (oc, "Self-calculated Greeks", "own", "flip_own_iv", "dealer_delta_own", "levels_own")):
        with col:
            st.markdown(f"**{title}**")
            st.metric("Net GEX", f"{view['net_gex_' + tag]:,.0f}")
            st.metric("Gamma flip (our model, " + ("CBOE IV" if tag == "cboe" else "our IV") + ")", _fmt_level(view[flipkey]))
            st.metric("Dealer delta (shares)", f"{view[ddkey]:,.0f}")
            lv = view[lvkey]
            st.write(f"Call wall: **{_fmt_level(lv['call_wall'])}**  ·  Put wall: **{_fmt_level(lv['put_wall'])}**  ·  ATM pin: **{_fmt_level(lv['atm_pin'])}**")
            st.write("Resistance (above spot): " + (", ".join(f"{k:,.2f}" for k, _ in lv["resistance"]) or "none"))
            st.write("Support (below spot): " + (", ".join(f"{k:,.2f}" for k, _ in lv["support"]) or "none"))

    base = view["net_gex_cboe"]
    if base:
        diff_pct = (view["net_gex_own"] - base) / abs(base) * 100
        agree_sign = (view["net_gex_own"] >= 0) == (view["net_gex_cboe"] >= 0)
        msg = f"Net GEX differs by {diff_pct:+.1f}% (own vs CBOE); sign {'agrees' if agree_sign else '**DISAGREES** — treat the regime label as unreliable'}."
        (st.caption if agree_sign else st.warning)(msg)
    f1, f2 = view["flip_cboe_iv"], view["flip_own_iv"]
    if (f1 is None) != (f2 is None):
        st.warning("Gamma flip exists under one IV input but not the other — the flip is fragile here; don't lean on it.")
    elif f1 is not None and f2 is not None and abs(f1 - f2) / spot > 0.005:
        st.caption(f"The two flips are {abs(f1 - f2):.2f} apart ({abs(f1 - f2) / spot * 100:.2f}% of spot) — flip level is sensitive to the IV input.")

    win = gx[(gx.index >= lo) & (gx.index <= hi)]
    if not win.empty:
        st.markdown("**Net GEX by strike (±8% of spot)**")
        st.bar_chart(win[["NetGEX_cboe", "NetGEX_own"]].rename(
            columns={"NetGEX_cboe": "CBOE Greeks", "NetGEX_own": "Self-calculated"}))
        with st.expander("Call / put split by strike"):
            st.dataframe(win[["CallGEX_cboe", "PutGEX_cboe", "CallGEX_own", "PutGEX_own", "CallOI", "PutOI"]].round(0),
                         use_container_width=True)

    # ---- 4. Nearest significant clusters -------------------------------------
    if has_narr:
        st.markdown("#### Nearest significant GEX clusters")
        st.caption(
            "The closest-to-spot strikes that still show a real GEX concentration (≥20% of the largest gross GEX within ±8% of spot), "
            "ranked by gross exposure — filters out small/noise strikes so the list favours visually significant bars near spot. "
            "Based on the self-calculated Greeks; the CBOE-Greeks version is in the expander."
        )

        def _cluster_lines(recs):
            if not recs:
                st.caption("Nothing near spot cleared the meaningful-size threshold this run.")
            for c_ in recs:
                line = (f"**{c_['strike']:.2f}** ({c_['pct_away']:+.2f}% from spot, {c_['side']}, {c_['lean']}) — "
                        f"Gross GEX {c_['gross_gex']:,.0f} (Net {c_['net_gex']:,.0f})")
                if f or f_stable:
                    _p = f"{c_['strike'] * f:,.0f}" if f else "n/a"
                    _c = f"{c_['strike'] * f_stable:,.0f}" if f_stable else "n/a"
                    line += f"  → XAUUSD ≈ **{_p}** (pulled basis) | **{_c}** (self-calculated)"
                st.markdown(line)
        _cluster_lines(view["clusters_own"])
        with st.expander("Same list using CBOE's Greeks"):
            _cluster_lines(view["clusters_cboe"])

    # ---- 5. Levels in spot-gold terms: PULLED basis vs SELF-CALCULATED, side by side ----------
    st.markdown("#### Spot Gold Equivalent (proxy) — pulled basis vs self-calculated")
    st.caption(
        "Every GLD level is converted two ways and shown next to each other. PULLED BASIS = level × (pulled XAUUSD ÷ live GLD), "
        "i.e. anchored to the spot price actually pulled from the feed. SELF-CALCULATED = level ÷ the 24h-stable oz/share. "
        "Δ is pulled − calculated in $/oz: that is how far the two methods disagree about where the wall sits in spot terms. "
        "This is a PROXY — GLD's own levels converted to spot units — not observed COMEX/spot options data."
    )
    if not f and not f_stable:
        st.caption("No basis or oz/share available this run, so the conversion is skipped.")
    else:
        def _conv(level):
            p_ = _to_xau(level, f)
            c_ = _to_xau(level, f_stable)
            d_ = (p_ - c_) if (p_ is not None and c_ is not None) else None
            b_ = (d_ / c_ * 1e4) if (d_ is not None and c_) else None
            return p_, c_, d_, b_
        rows = []
        def add(name, own, cb, g_own=None, g_cb=None):
            for tag, lvl, g_ in (("Self-calc Greeks", own, g_own), ("CBOE Greeks", cb, g_cb)):
                if lvl is None:
                    continue
                p_, c_, d_, b_ = _conv(lvl)
                rows.append({"Level": name, "Greeks": tag, "GLD": lvl,
                             "XAUUSD — pulled basis": p_, "XAUUSD — self-calculated": c_,
                             "Δ pulled − calc ($/oz)": d_, "Δ (bps)": b_, "GEX": g_})
        lo_, lc_ = view["levels_own"], view["levels_cboe"]
        add("Spot (chain)", spot, spot)
        add("Gamma flip", view["flip_own_iv"], view["flip_cboe_iv"])
        add("ATM pin", lo_["atm_pin"], lc_["atm_pin"])
        add("Call wall", lo_["call_wall"], lc_["call_wall"])
        add("Put wall", lo_["put_wall"], lc_["put_wall"])
        for i_ in range(max(len(lo_["resistance"]), len(lc_["resistance"]))):
            o_ = lo_["resistance"][i_] if i_ < len(lo_["resistance"]) else (None, None)
            c_ = lc_["resistance"][i_] if i_ < len(lc_["resistance"]) else (None, None)
            add(f"Resistance {i_ + 1}", o_[0], c_[0], o_[1], c_[1])
        for i_ in range(max(len(lo_["support"]), len(lc_["support"]))):
            o_ = lo_["support"][i_] if i_ < len(lo_["support"]) else (None, None)
            c_ = lc_["support"][i_] if i_ < len(lc_["support"]) else (None, None)
            add(f"Support {i_ + 1}", o_[0], c_[0], o_[1], c_[1])
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)
        if f and f_stable:
            st.caption(
                f"The two factors differ by {(f / f_stable - 1) * 1e4:+.1f} bps → about ${abs(f / f_stable - 1) * spot * f:,.2f}/oz at spot. "
                f"If that is small, either column is fine; if it is large, trust the pulled basis (it is anchored to the live spot)."
            )
        elif not f_stable:
            st.caption("Self-calculated column unavailable: no oz/share value this run.")
        elif not f:
            st.caption("Pulled-basis column unavailable: no valid live/logged basis this run.")

    # ---- 6. Dealer positioning context ---------------------------------------
    if has_narr:
        st.markdown("#### Dealer positioning context")
        cn_o, cn_c = view["concentration_own"], view["concentration_cboe"]
        st.write(f"**GEX concentration:** net/gross ratio {cn_o['ratio']:.1%} (gross {cn_o['gross_gex']:,.0f}, net {cn_o['net_gex']:,.0f}) — self-calculated Greeks")
        st.caption(cn_o["narrative"])
        if abs(cn_o["ratio"] - cn_c["ratio"]) > 0.10:
            st.caption(f"CBOE-Greeks ratio is {cn_c['ratio']:.1%} — the two Greek sets disagree on how lopsided the book is.")
        dctx = gex_alt.dealer_delta_context(view["dealer_delta_own"], avg_volume)
        if dctx["ratio"] is not None:
            st.write(f"**Dealer delta vs. 10-day avg volume:** {dctx['ratio']:.1%} ({dctx['dealer_delta']:,.0f} vs {dctx['avg_volume_10d']:,.0f})")
            st.caption(dctx["size_narrative"])
        else:
            st.caption("Dealer delta / volume comparison unavailable this run (no volume data).")
        st.caption(dctx["direction_narrative"])
        st.caption("Selection-specific: this covers only the expiries you picked above, not the whole chain.")

    # ---- 7. Deviation --------------------------------------------------------
    st.markdown("#### How far apart are the Greeks?")
    dv = view.get("deviation") or {}
    if not dv.get("n"):
        st.caption("No contracts with both Greek sets in this selection.")
    else:
        def _nf(x, fmt):
            return "n/a" if x is None or x != x else format(x, fmt)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Median |ΔIV|", _nf(dv.get("median_abs_dIV_pts"), ".2f") + " vol pts")
        d2.metric("Median |ΔDelta|", _nf(dv.get("median_abs_dDelta"), ".4f"))
        d3.metric("Median |ΔGamma|", _nf(dv.get("median_abs_dGamma_pct"), ".1f") + "%")
        d4.metric("OI-wtd Gamma own/CBOE", _nf(dv.get("oi_wtd_gamma_ratio_own_over_cboe"), ".3f"))
        st.caption(
            f"{dv['n']} contracts with open interest. The OI-weighted ratio is what matters for GEX; the median |ΔGamma| is "
            f"inflated by illiquid wings whose bid/ask mid is unreliable (see Spread % in the table). A constant IV offset "
            f"across liquid strikes alongside matching gamma points to a time-to-expiry convention difference rather than noise."
        )

    if contracts is not None and not contracts.empty:
        st.markdown("**Contract detail — both IVs and both Greek sets (top 40 by open interest)**")
        cols = ["Expiration", "Strike", "OptionType", "OpenInterest", "Bid", "Ask", "Spread %", "IV_cboe", "IV_own",
                "dIV", "IV_used_source", "Delta_cboe", "Delta_own", "dDelta", "Gamma_cboe", "Gamma_own", "dGamma_pct"]
        cols = [c_ for c_ in cols if c_ in contracts.columns]
        cd = contracts.sort_values("OpenInterest", ascending=False).head(40)[cols].copy()
        cd = cd.rename(columns={"dIV": "ΔIV (pts)", "dDelta": "ΔDelta", "dGamma_pct": "ΔGamma %",
                                "IV_used_source": "Own Greeks used IV"})
        st.dataframe(cd.round(4), hide_index=True, use_container_width=True)
    elif contracts is None:
        st.caption("Contract-level table needs live data; not stored in the snapshot.")

    # ---- 8. Raw walls --------------------------------------------------------
    if has_narr and view.get("raw_walls_own"):
        with st.expander("Raw walls (unfiltered by spot, diagnostic)"):
            st.caption("Can surface the at-the-money strike on both sides, since ATM gamma is naturally largest — real information, just not directional.")
            rw = view["raw_walls_own"]
            w1, w2 = st.columns(2)
            with w1:
                st.write("**Call walls (largest CallGEX):**")
                st.table(pd.DataFrame(rw["call_walls"], columns=["Strike", "GEX"]))
            with w2:
                st.write("**Put walls (largest |PutGEX|):**")
                st.table(pd.DataFrame(rw["put_walls"], columns=["Strike", "GEX"]))


with gex_alt_tab:
    st.subheader("GEX (Alternative) — CBOE delayed quotes + GVZ")
    st.caption(
        "Independent of the GEX (Options) tab and of MarketData.app: option chain, open interest and Greeks come "
        "from CBOE's free delayed-quote feed; expected range from GVZ. CBOE's own Greeks and our Black-Scholes "
        "Greeks are shown separately so you can see where they deviate. **Data limits:** quotes are delayed "
        "(~15 min) and open interest is the *previous day's* figure, so intraday GEX is an estimate. "
        "**Cadence:** the option-chain/GEX snapshot follows the original GEX timing (10-min cache; refresh button); "
        "the GLD→spot basis is captured separately every 15 min."
    )

    ac1, ac2, ac3 = st.columns([1, 2, 1])
    alt_sym = ac1.text_input("Underlying", value="GLD", key="galt_sym").upper().strip() or "GLD"
    _BASIS_OPTS = {
        "Own IV (backed out of bid/ask mid)": "mid",
        "Hybrid: own IV where spread ≤ 10% of mid, else CBOE's IV": "hybrid",
        "CBOE's IV (isolates formula / time / rate differences)": "cboe",
    }
    alt_basis_label = ac2.radio("Self-calculated Greeks use", list(_BASIS_OPTS), key="galt_basis")
    alt_basis = _BASIS_OPTS[alt_basis_label]
    ac3.write("")
    if ac3.button("🔄 Refresh chain (GEX cadence)", key="galt_refresh"):
        load_cboe_chain.clear()
        load_gvz.clear()
        st.session_state.pop("galt_fail_until", None)
    if ac3.button("🔄 Refresh basis (15-min)", key="galt_refresh_basis"):
        load_basis_live.clear()
        load_basis_history.clear()

    # ---- basis (fast cadence) -------------------------------------------------
    try:
        live_pt = load_basis_live()
    except Exception:
        live_pt = None
    try:
        basis_hist = load_basis_history()
    except Exception:
        basis_hist = []
    stable_oz = load_stable_oz_per_share() if alt_sym == "GLD" else None
    basis_choice = (gld_basis.choose_basis(live_pt, basis_hist, stable_oz) if alt_sym == "GLD"
                    else {"spot_per_gld": None, "source": "none", "point": None, "note": "spot-gold conversion only applies to GLD"})

    # ---- chain (GEX cadence); failures remembered briefly so clicks don't each wait on a dead host
    alt_live, alt_err = None, None
    _now = datetime.now().timestamp()
    if _now < st.session_state.get("galt_fail_until", 0):
        alt_err = st.session_state.get("galt_fail_msg", "recent fetch failed")
    else:
        try:
            with st.spinner("Fetching CBOE delayed option chain..."):
                _raw, _meta = load_cboe_chain(alt_sym)
            alt_live = (_raw, _meta)
            st.session_state.pop("galt_fail_until", None)
        except Exception as e:
            alt_err = str(e)
            st.session_state["galt_fail_until"] = _now + 120
            st.session_state["galt_fail_msg"] = alt_err

    if alt_live is not None:
        raw_df, meta = alt_live
        spot_alt = meta["spot"]
        rf, rf_note = load_risk_free()
        gvz_val = load_gvz()
        avg_vol = load_avg_volume_10d(alt_sym)
        chain_alt = gex_alt.add_time_to_expiry(raw_df, meta.get("asof_utc"))
        enr = gex_alt.enrich_chain(chain_alt, spot_alt, rf, alt_basis)
        exp_tbl = gex_alt.expiry_table(enr)

        asof_iso = meta["asof_utc"].isoformat() if meta.get("asof_utc") else None
        when, age = gex_snapshot.describe_age(asof_iso) if asof_iso else ("unknown", "unknown age")
        st.success(f"Live CBOE chain — stamped **{when}** ({age}); source: {meta.get('url')}")
        if not meta.get("has_cboe_greeks"):
            st.warning("This CBOE response contained no Greeks — the 'CBOE Greeks' side will be empty/zero.")
        if meta.get("iv_rescaled"):
            st.caption("CBOE IVs looked like percentages and were rescaled to decimals.")

        # live price vs chain snapshot (same idea and 0.5% threshold as the original tab)
        _lp = (live_pt or {}).get("gld") if (live_pt or {}).get("gld_age_min") is not None and live_pt["gld_age_min"] <= gld_basis.FRESH_MINUTES else None
        if alt_sym == "GLD" and _lp:
            dev = abs(_lp - spot_alt) / spot_alt * 100
            if dev >= 0.5:
                st.warning(
                    f"⚠️ Live {alt_sym} ({_lp:.2f}) has moved {dev:.2f}% from this chain snapshot's spot ({spot_alt:.2f}) — "
                    f"a real move since the options data was stamped, not just refresh lag. Strikes don't move, but which side "
                    f"of price each wall now sits on can have changed; read the levels relative to the live price."
                )
            else:
                st.caption(f"Live {alt_sym} vs chain snapshot: {dev:.2f}% apart — within normal delay.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{alt_sym} (CBOE)", f"{spot_alt:,.2f}")
        m2.metric("GVZ", f"{gvz_val:.2f}" if gvz_val is not None else "n/a")
        m3.metric("CBOE iv30 (raw, as published)", f"{float(meta['iv30']):.4g}" if meta.get("iv30") not in (None, "") else "n/a")
        m4.metric("Risk-free used", f"{rf*100:.2f}%", help=rf_note)

        with st.expander("GLD → spot-gold basis (15-minute capture) and spot reconciliation", expanded=True):
            _basis_panel(basis_choice, live_pt, basis_hist, stable_oz, chain_spot=spot_alt)

        if exp_tbl.empty:
            st.warning("No unexpired expirations in the CBOE response.")
        else:
            st.markdown("#### Choose the DTE")
            dte_mode = st.radio(
                "Selection", ["Single expiry", "Cumulative (all expiries up to N days)"],
                key="galt_mode", horizontal=True,
            )
            alt_view = None
            if dte_mode == "Single expiry":
                labels = [f"{r_.Expiration}  —  {r_.DTE_label}  —  OI {r_.TotalOI:,.0f}" for r_ in exp_tbl.itertuples()]
                pick = st.selectbox("Expiry", labels, index=0, key="galt_exp")
                chosen = exp_tbl.iloc[labels.index(pick)]
                alt_view = gex_alt.compute_view(enr, spot_alt, rf, "single", expiry=chosen["Expiration"])
            else:
                max_avail = float(exp_tbl["DTE"].max())
                opts = [d_ for d_ in (1, 2, 3, 5, 7, 14, 21, 30, 45, 60, 90) if d_ <= max_avail + 1] or [int(max(max_avail, 1))]
                n_days = st.select_slider("Up to N days to expiry", options=opts, value=7 if 7 in opts else opts[-1], key="galt_cum")
                alt_view = gex_alt.compute_view(enr, spot_alt, rf, "cumulative", max_dte=n_days)

            with st.expander("All expiries (DTE, open interest)"):
                st.dataframe(exp_tbl[["Expiration", "DTE_label", "TotalOI", "Contracts"]].rename(
                    columns={"DTE_label": "DTE"}), hide_index=True, use_container_width=True)

            # remember this pull as the last good one (once per CBOE timestamp / basis choice)
            _save_tag = f"{alt_sym}|{asof_iso}|{alt_basis}"
            if st.session_state.get("galt_saved_for") != _save_tag:
                try:
                    _payload = gex_alt.build_snapshot_payload(
                        alt_sym, spot_alt, meta, enr, rf, rf_note, gvz_val, alt_basis,
                        avg_volume=avg_vol, basis=(basis_choice.get("point") if basis_choice else None))
                    if gex_snapshot.save_payload(f"{alt_sym}|alt", _payload):
                        st.session_state["galt_saved_for"] = _save_tag
                except Exception as e:
                    st.caption(f"(Could not store a last-good snapshot this run: {e})")

            if alt_view is None:
                st.warning("No contracts with open interest in that selection.")
            else:
                st.caption(f"Selection: **{alt_view['label']}** — {alt_view['n_contracts']} contracts, open interest {alt_view['total_oi']:,.0f}.")
                _render_gex_alt_view(alt_view, spot_alt, gvz_val, alt_sym, contracts=alt_view["selection_df"],
                                     basis=basis_choice, stable_oz=stable_oz, avg_volume=avg_vol)

    else:
        # --- fallback: last successful CBOE pull (chain), with the CURRENT basis (fast cadence)
        snap, origin = gex_snapshot.load_best_payload(f"{alt_sym}|alt", load_remote_gex_store())
        st.error(f"Live CBOE fetch failed: {alt_err}")
        if snap is None or not snap.get("views"):
            st.info(
                "No saved last-good snapshot yet for this underlying. Once one live pull succeeds (or the "
                "scheduled logger has run), it will appear here automatically."
            )
        else:
            when, age = gex_snapshot.describe_age(snap.get("saved_utc"))
            cb_when, _ = gex_snapshot.describe_age(snap.get("asof_cboe_utc")) if snap.get("asof_cboe_utc") else ("unknown", "")
            st.warning(
                f"Showing the LAST SUCCESSFUL chain pull, saved **{when}** ({age}); CBOE data stamp {cb_when}; "
                f"stored {origin}. The chain is not live. Spot in this snapshot: {snap['spot']:,.2f}."
            )
            with st.expander("GLD → spot-gold basis (15-minute capture) and spot reconciliation", expanded=True):
                _basis_panel(basis_choice, live_pt, basis_hist, stable_oz)
            vkeys = list(snap["views"].keys())
            vlabels = [snap["views"][k]["label"] for k in vkeys]
            vpick = st.selectbox("Stored view", vlabels, index=0, key="galt_snap_view")
            sv = snap["views"][vkeys[vlabels.index(vpick)]]
            _render_gex_alt_view(sv, snap["spot"], snap.get("gvz"), alt_sym, contracts=None,
                                 basis=basis_choice, stable_oz=stable_oz, avg_volume=snap.get("avg_volume_10d"))
