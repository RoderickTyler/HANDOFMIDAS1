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

import pandas as pd
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
def load_core_data(period: str):
    data = fetch_data.fetch_all(period=period)
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

    window_pct: how far from spot to look (fraction of spot price).
    magnitude_frac: a strike must have |NetGEX| at least this fraction of
        the largest |NetGEX| within the window (default 20%) to count as
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
    window["AbsNetGEX"] = window["NetGEX"].abs()
    threshold = window["AbsNetGEX"].max() * magnitude_frac
    candidates = window[window["AbsNetGEX"] >= threshold].copy()
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

if st.sidebar.button("\U0001F504 Refresh data now"):
    st.cache_data.clear()

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

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("\U0001F4C8 Gold Macro Daily Briefing")
st.caption(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} \u2014 lookback: {period}")

try:
    with st.spinner("Fetching market data, yields, reserves, and risk index..."):
        core = load_core_data(period)
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

tabs = st.tabs([
    "Overview", "Regime (HMM)", "COT Positioning", "Factor Attribution",
    "Econ Calendar", "Central Bank Reserves", "Journal", "GEX (Options)",
])

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
    spot_history = load_spot_history(period)
    chart_df = pd.DataFrame(index=signals.index)
    if spot_history is not None:
        chart_df["Gold Spot (XAUUSD)"] = spot_history.reindex(chart_df.index)
        gold_chart_note = "Showing true spot gold (XAUUSD), not the futures contract."
    else:
        chart_df["Gold (GC=F futures)"] = signals["gold_spot"] if "gold_spot" in signals.columns else None
        gold_chart_note = (
            "Couldn't fetch true spot gold (XAUUSD=X) this run, so this is showing "
            "GC=F futures instead \u2014 they track closely but aren't identical."
        )
    if "dxy" in signals.columns:
        chart_df["DXY"] = signals["dxy"]
    chart_df = chart_df.dropna(how="all")
    if not chart_df.empty:
        st.line_chart(chart_df)
    st.caption(gold_chart_note)

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
    if run_gex or gex_state_key not in st.session_state:
        try:
            with st.spinner(f"Fetching {ticker} options chain and computing GEX..."):
                st.session_state[gex_state_key] = load_gex_assessment(
                    ticker, expiration_choice, min_oi_input, use_md
                )
        except Exception as e:
            st.session_state[gex_state_key] = None
            st.error(
                f"GEX assessment failed: {e}\n\n"
                "Common causes: no listed options for this ticker, no expiration with "
                "usable open interest right now, or (for the free yfinance path) Yahoo "
                "rate-limiting. Try a different expiration, or lower/remove the min "
                "open interest filter."
            )

    result = st.session_state.get(gex_state_key)

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
        nearest = find_nearest_gex_clusters(gbs, result.spot)
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
                if abs(call_gex) > abs(put_gex) * 1.3:
                    lean = "call-dominant"
                elif abs(put_gex) > abs(call_gex) * 1.3:
                    lean = "put-dominant"
                else:
                    lean = "mixed call/put"
                pct_away = (dist / result.spot) * 100
                line = (
                    f"**{strike:.2f}** ({pct_away:+.2f}% from spot, {side}, {lean}) \u2014 "
                    f"NetGEX {row['NetGEX']:,.0f}"
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
        live_gld, live_xau = load_gex_live_refs()
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
