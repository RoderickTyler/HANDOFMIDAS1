"""
main.py
-------
CLI entry point.

Usage:
    python main.py                 # run the daily briefing
    python main.py --log-thesis    # interactively log this week's thesis
    python main.py --hit-rate      # show your running accuracy scoreboard
    python main.py --period 2y     # fetch a longer lookback window
    python main.py --factors       # re-run factor attribution on cached data, no live fetch

First-time setup:
    1. pip install -r requirements.txt
    2. Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
    3. Create a .env file next to this script:
           FRED_API_KEY=your_key_here
    4. python main.py
"""

import argparse
import os
from datetime import datetime
import pandas as pd

import fetch_data
import analysis
import report
import journal
import econ_calendar
import hmm_regime
import factor_attribution
import reserves_utils
import trading_mode
import xau_currency_score
import validation
import cot_analysis
import spot_gold


def run_daily_briefing(period="1y"):
    data = fetch_data.fetch_all(period=period)
    merged = analysis.merge_datasets(data["market"], data["fred"])

    if merged.empty:
        print("\nNo data could be assembled. Check your internet connection and FRED_API_KEY.")
        return

    signals = analysis.compute_signals(merged)
    summary = analysis.summarize_latest(signals)
    flags = analysis.detect_divergences(signals)

    # cache for later inspection / your own further analysis
    signals.to_csv(fetch_data.config.PRICE_CACHE)

    required_cols = ["gold_spot", "dxy", "dfii10"]
    missing_cols = [c for c in required_cols if c not in signals.columns]
    if missing_cols:
        print(f"\n[warn] Cannot compute complete-data range -- missing column(s) {missing_cols} "
              f"entirely (likely FRED_API_KEY isn't set, so no yield data was fetched at all). "
              f"See the warning above for how to fix this. Continuing with what's available.")
    else:
        complete_rows = signals.dropna(subset=required_cols)
        if not complete_rows.empty:
            print(f"\nComplete-data range (gold + DXY + real yield all present): "
                  f"{complete_rows.index.min().date()} to {complete_rows.index.max().date()} "
                  f"({len(complete_rows)} days). This is the range models below actually use.")

    print("Fetching real spot XAUUSD (gold-api.com, free, no key needed -- for comparison against GC=F)...")
    try:
        xauusd_result = spot_gold.fetch_xauusd_spot()
        if xauusd_result is not None:
            summary["xauusd_spot"] = xauusd_result["price"]
        else:
            print("  [info] Spot fetch returned nothing this run -- skipping.")
    except Exception as e:
        print(f"[warn] Spot XAUUSD fetch failed this run (non-fatal): {e}")

    print()
    report.print_full_briefing(summary, flags)

    print("\nFetching economic calendar (CPI, NFP, GDP, ISM PMI, FOMC)...")
    _print_econ_calendar()

    print("\nFitting macro regime model (3-state HMM)...")
    regime_result = None
    try:
        regime_result = hmm_regime.analyze_regime(signals)
        _print_regime(regime_result)
    except Exception as e:
        print(f"[warn] Regime model failed this run (non-fatal, rest of briefing continues): {e}")

    print("\nFetching COT positioning (CFTC, weekly -- for the trading mode confidence check)...")
    cot_result = None
    try:
        cot_result = cot_analysis.build_report()
        if cot_result is None:
            print("  [info] COT data unavailable this run -- trading mode will run without it.")
    except Exception as e:
        print(f"[warn] COT fetch failed this run (non-fatal): {e}")

    mode = None
    try:
        mode = _print_trading_mode(regime_result, cot_result=cot_result)
    except Exception as e:
        print(f"[warn] Trading mode failed this run (non-fatal): {e}")

    if not data["gpr"].empty:
        print("\n--- Geopolitical Risk Index (latest available) ---")
        print(data["gpr"].tail(3))

    if not data["reserves"].empty:
        print("\n--- Central Bank Gold Reserves (IMF): recent trend by country ---")
        print(_summarize_reserves_by_country(data["reserves"]))

        history_path = _save_reserves_history(data["reserves"])
        print(f"\n  Full monthly history (back to 2015, all countries/sectors) "
              f"saved to: {history_path}")

        print("\n--- Central Bank Gold Reserves: last 12 months by country (USD) ---")
        print(_pivot_reserves_history(data["reserves"], n_periods=12))
        print("\n  [!] REMINDER: the table above is USD VALUE (mark-to-market). It moves both")
        print("  from actual buying/selling AND from gold's own price moving -- it does NOT")
        print("  by itself tell you whether a country actually bought or sold. See the")
        print("  PHYSICAL QUANTITY table below for that.")

    if data.get("reserves_volume") is not None and not data["reserves_volume"].empty:
        unit = data.get("reserves_volume_unit", "unit not identified -- verify manually")
        print(f"\n--- Central Bank Gold Reserves: PHYSICAL QUANTITY, last 12 months ({unit}) ---")
        print("(This is immune to gold price moves -- a change here means the country actually")
        print(" added or removed physical gold, not just a price effect.)\n")
        print(_pivot_reserves_history(data["reserves_volume"], n_periods=12, value_fmt="plain"))
    else:
        print("\n[warn] Could not fetch physical quantity (volume) gold reserves this run -- "
              "only the USD-value table above is available, which conflates price and quantity "
              "effects. Manual fallback for physical holdings: "
              "https://www.gold.org/goldhub/data/gold-reserves-by-country")

    print("\nScoring factor attribution (DXY / real yields / VIX / central bank buying)...")
    try:
        _print_factor_attribution(signals, data["reserves"])
    except Exception as e:
        print(f"[warn] Factor attribution failed this run (non-fatal): {e}")

    try:
        _print_macro_summary(mode, cot_result=cot_result)
    except Exception as e:
        print(f"[warn] Macro summary failed this run (non-fatal): {e}")


def _print_factor_attribution(price_df, reserves_df):
    from tabulate import tabulate

    result = factor_attribution.build_report(price_df, reserves_df)
    print("\n--- Factor Attribution: what's actually moving gold ---")
    if result is None:
        print("  Not enough clean daily history yet for a meaningful attribution.")
        return

    reg = result["regression"]
    if reg:
        print(f"\n  Daily-move regression (last {reg['n_days']} days):")
        table = [[k, f"{v:+.3f}"] for k, v in reg["scores"].items()]
        print(tabulate(table, headers=["Factor", "Standardized score"], tablefmt="simple"))
        print(f"  Overall R²: {reg['r2']:.3f} "
              f"(~{reg['r2']*100:.0f}% of daily moves explained by these factors combined)")
    else:
        print("  Not enough data yet for the daily regression.")

    if result["quarterly_rows"]:
        print("\n  Quarterly regime shifts (partial quarters included):")
        table = [[
            r["Quarter"], f"{r['Gold return']:+.1f}%",
            f"{r['corr(DXY)']:+.2f}" if pd.notna(r["corr(DXY)"]) else "n/a",
            f"{r['corr(RealYield)']:+.2f}" if pd.notna(r["corr(RealYield)"]) else "n/a",
            f"{r['corr(VIX)']:+.2f}" if pd.notna(r["corr(VIX)"]) else "n/a",
            r["Dominant"], r["n_days"],
        ] for r in result["quarterly_rows"]]
        print(tabulate(table, headers=["Quarter", "Gold ret", "corr(DXY)", "corr(RY)", "corr(VIX)", "Dominant", "n days"], tablefmt="simple"))
    else:
        print("\n  Not enough history yet for a quarterly breakdown.")

    if result["reserves_rows"]:
        print("\n  Central bank buying: value growth vs. gold's own price gain over the same window:")
        table = [[
            r["Country"], f"{r['Value growth']:+.1f}%",
            f"{r['vs. gold price']:+.0f}pp" if r["vs. gold price"] is not None else "n/a",
            r["Read"],
        ] for r in result["reserves_rows"]]
        print(tabulate(table, headers=["Country", "Value growth", "vs. gold price", "Read"], tablefmt="simple"))

    if result["ranking"]:
        print("\n  Summary:")
        for line in result["ranking"]:
            print(f"    - {line}")


def _print_trading_mode(regime_result, cot_result=None):
    print("\n--- Trading Mode (today's filter, not an entry signal) ---")
    mode = trading_mode.determine_mode(regime_result, cot_result=cot_result)
    if mode is None:
        print("  No regime data available -- skipping.")
        return None

    print(f"  State: {mode['top_state']} ({mode['top_prob']*100:.1f}% confidence, {mode['confidence']})")
    print(f"  Mode:  {mode['strategy_name']}")
    print(f"         {mode['strategy_note']}")
    print(f"  Full vs. recent-window agreement: {mode['consistency_note']}")
    print(f"  Suggested size: {mode['size']}")
    if cot_result is not None:
        print(f"  COT positioning check: {mode.get('cot_note', 'n/a')}")
        if mode.get("cot_level_interpretation"):
            print(f"  COT state (3yr): {mode['cot_level_interpretation']}")
        if mode.get("cot_surprise_note"):
            print(f"  {mode['cot_surprise_note']}")
    else:
        print("  COT positioning check: not run this session (use --cot to fetch it, or wire it into "
              "the daily briefing if you want this automatically each run)")
    print("  Reminder: this sets your MODE and SIZE for the day. Entries, stops, and")
    print("  targets still come from your own 5m/15m/1h/4h read -- this doesn't replace that.")

    print("\n  Overall Assessment:")
    print(f"  {mode['overall_assessment']}")

    return mode


def _print_macro_summary(mode, cot_result=None):
    """The final one-glance takeaway, printed at the very end of the daily
    briefing -- everything above is the detail; this is the headline."""
    print("\n" + "=" * 70)
    print("TODAY'S MACRO SUMMARY")
    print("=" * 70)
    if mode is None:
        print("  Not enough data this run to produce a summary.")
        return

    print(f"  - Regime: {mode['top_state']} ({mode['confidence']} confidence, {mode['top_prob']*100:.0f}%)")

    if cot_result is not None and mode.get("cot_level_interpretation"):
        print(f"  - Positioning: {mode['cot_level_interpretation']}")
    else:
        print("  - Positioning: not available this run")

    bias = {"Rising": "Bullish", "Declining": "Bearish", "Range": "Neutral/Range"}.get(mode["top_state"], "Unclear")
    print(f"  - Macro bias: {bias}")
    print(f"  - Conviction/Size: {mode['size'].split(' -- ')[0]}")

    if mode.get("cot_crowding_detected") is True:
        print(f"  - Structural warning: Bullish/bearish trend remains intact, but positioning is "
              f"historically stretched in the same direction -- reduced conviction.")
    elif not mode.get("consistency_ok", True):
        print(f"  - Structural warning: Recent regime behavior has diverged from the longer-run "
              f"pattern -- treat today's read with extra caution.")
    else:
        print(f"  - Structural warning: None")
    print("=" * 70)


def _print_econ_calendar(days_ahead=14):
    calendar = econ_calendar.get_upcoming_calendar(days_ahead=days_ahead)
    print(f"\n--- Economic Calendar: next {days_ahead} days ---")
    if not calendar:
        print("  No upcoming primary releases found in this window (or FRED_API_KEY not set).")
    else:
        today = datetime.now().date()
        for label, dates in calendar.items():
            for d in dates:
                tag = " <- TODAY" if d == today else ""
                print(f"  {d.strftime('%Y-%m-%d (%a)')}  {label}{tag}")

    powell = econ_calendar.get_upcoming_powell_speeches(days_ahead=days_ahead)
    if powell:
        print(f"  Possible Fed Chair speech date(s) found (best-effort, verify manually): {', '.join(powell)}")
    else:
        print("  No Fed Chair speech detected via best-effort check -- "
              "verify directly if this matters today: "
              "https://www.federalreserve.gov/newsevents/speeches.htm")


def _print_regime(result):
    print("\n--- Macro Regime (3-state HMM: Declining / Range / Rising) ---")
    if result is None:
        print("  Not enough data yet to fit the regime model.")
        return

    if not result["model_healthy"]:
        print("  [!] Model health check flagged possible overfitting -- treat this read with extra skepticism.")

    print("  Current state probabilities:")
    for label, prob in result["current_probs"].items():
        print(f"    {label}: {prob*100:.1f}%")

    print("\n  How often each state actually occurs in the sample (a state name doesn't")
    print("  imply rarity -- check this against the name yourself):")
    for label, pct in result["state_frequency"].items():
        print(f"    {label}: {pct}% of days")

    if result["small_sample_states"]:
        print(f"  [!] Small-sample state(s), transition odds unreliable: "
              f"{', '.join(result['small_sample_states'])}")

    print(f"\n  FULL-HISTORY transition matrix ({result['n_observations']} days, "
          f"{result['labeled_df'].index.min().date()} to {result['labeled_df'].index.max().date()}):")
    print(result["transition_matrix"].round(3).to_string())
    print("  Caution: this is a single average over the WHOLE period. If early and recent")
    print("  history behaved differently, this can look nothing like current conditions --")
    print("  always check the recent-window matrix below before trusting this one alone.")

    if result["recent_matrix"] is not None:
        print(f"\n  RECENT-WINDOW transition matrix (last {result['recent_window_days']} trading days, "
              f"~3 months: {result['recent_window_start'].date()} to {result['recent_window_end'].date()}):")
        print(result["recent_matrix"].round(3).to_string())
    else:
        print(f"\n  Not enough history yet for a separate recent-window matrix (need ~73+ days).")


def _parse_period(period_str):
    """Parse IMF period strings like '2026-M06' or '2026-06' into a pandas Period (monthly)."""
    s = str(period_str).replace("M", "")
    try:
        return pd.Period(s, freq="M")
    except Exception:
        return None


def _nearest_row(group, target_period):
    """Find the row in `group` whose period is closest to target_period."""
    diffs = (group["_period"] - target_period).map(lambda d: abs(d.n) if d is not None else float("inf"))
    idx = diffs.idxmin()
    return group.loc[idx]


def _save_reserves_history(reserves_df):
    """
    Saves the FULL monthly history IMF returned (back to 2015, all
    countries, all sectors) to a CSV -- this is fetched every run but was
    previously discarded after computing just the summary table. This is
    your raw material for any deeper backtesting/charting you want to do
    yourself later.
    """
    import os
    path = os.path.join(fetch_data.config.DATA_DIR, "reserves_history.csv")
    reserves_df.to_csv(path, index=False)
    return path


def _pivot_reserves_history(reserves_df, n_periods=12, value_fmt="usd_billions"):
    """
    A readable period-by-country table for the most recent N months, using
    the SAME per-country sector selection logic (via reserves_utils) as
    the summary table -- so numbers here match what you see there, rather
    than mixing institutional sub-sectors within a country/column.

    value_fmt: "usd_billions" (default, formats as $X.XXB) or "plain"
    (formats as a comma-separated number, for non-USD series like troy
    ounces where a $ prefix would be misleading).
    """
    country_col, period_col, value_col, _ = reserves_utils.get_columns(reserves_df)
    if not all([country_col, period_col, value_col]):
        return "Could not identify country/period/value columns."

    filtered = reserves_utils.select_single_sector_per_country(reserves_df).copy()
    filtered["_period"] = filtered[period_col].apply(_parse_period)
    filtered = filtered.dropna(subset=["_period"])

    pivot = filtered.pivot_table(index="_period", columns=country_col, values=value_col, aggfunc="last")
    pivot = pivot.sort_index().tail(n_periods)
    pivot.index = pivot.index.astype(str)

    # Format as readable $ billions instead of raw floats (which pandas
    # renders in scientific notation for numbers this large, e.g.
    # "2.439850e+11" -- unreadable at a glance). NaN stays blank rather
    # than becoming "nan" text.
    def _fmt(v):
        if pd.isna(v):
            return "-"
        if value_fmt == "plain":
            return f"{v:,.1f}"
        return f"${v/1e9:,.2f}B"

    formatted = pivot.apply(lambda col: col.map(_fmt))

    return formatted.to_string()


def _summarize_reserves_by_country(reserves_df, lookbacks_months=(1, 3, 6)):
    """
    IMF's IRFCL data comes back long-format (one row per country per period),
    not wide-format. This pivots it into a recent-trend view per country:
    latest reading vs. ~1/3/6 months ago (nearest available data point to
    each target), so you can see near-term accumulation/drawdown rather than
    a decade-long structural trend.
    """
    df = reserves_df.copy()

    country_col, period_col, value_col, sector_col = reserves_utils.get_columns(df)

    if not all([country_col, period_col, value_col]):
        return "Could not identify country/period/value columns to summarize (raw data was returned instead)."

    filtered = reserves_utils.select_single_sector_per_country(df).copy()
    filtered["_period"] = filtered[period_col].apply(_parse_period)
    filtered = filtered.dropna(subset=["_period"])

    rows = []
    for country, group in filtered.groupby(country_col):
        group = group.sort_values("_period")
        if group.empty:
            continue
        sector_used = group[sector_col].iloc[0] if sector_col else "n/a"
        latest = group.iloc[-1]
        latest_val = latest[value_col]
        latest_period = latest["_period"]

        row = {
            "Country": country,
            "Sector": sector_used if sector_used else "n/a",
            "Latest period": latest[period_col],
            "Latest value": f"{latest_val:,.0f}",
        }

        for months_back in lookbacks_months:
            target = latest_period - months_back
            # only compare against real earlier data, not the latest row itself
            earlier_group = group[group["_period"] <= target]
            if earlier_group.empty:
                row[f"{months_back}mo chg"] = "n/a"
                continue
            comp = _nearest_row(earlier_group, target)
            comp_val = comp[value_col]
            pct_chg = (
                f"{((latest_val - comp_val) / comp_val * 100):+.1f}%"
                if comp_val not in (None, 0)
                else "n/a"
            )
            row[f"{months_back}mo chg"] = pct_chg

        rows.append(row)

    if not rows:
        return "No rows to summarize."

    out_df = pd.DataFrame(rows).sort_values("Country")
    return out_df.to_string(index=False)


def log_thesis_interactive():
    journal.print_weekly_prompts()
    print("\nEnter your weekly thesis (Ctrl+C to cancel):")
    print(f"Dominant factor options: {', '.join(journal.DOMINANT_FACTOR_OPTIONS)}")
    dominant_factor = input("Dominant factor: ").strip()
    thesis = input("Thesis (why): ").strip()
    prediction = input("Falsifiable prediction (specific, checkable): ").strip()
    days = input("Check back in how many days? [14]: ").strip()
    days = int(days) if days else 14
    journal.add_entry(dominant_factor, thesis, prediction, check_in_days=days)


def run_extended():
    """
    Fetches the FULL available history (bounded by DFII10's real start
    date, 2003-01-02 -- the actual data-availability ceiling, not an
    arbitrary choice), re-runs the state-count validation on this much
    larger, more heterogeneous sample (spanning multiple real macro
    regimes: 2008 crisis, 2013 taper tantrum, 2015-18 hiking, 2020 COVID,
    2022 hiking, 2023-24 disinflation, and the current regime), and
    presents it SIDE BY SIDE against your existing 1-year cached model --
    this never replaces or modifies the short-window model, it's purely a
    comparison layer to test whether the current model's structure and
    conclusions hold up at scale, or are specific to the current regime.
    """
    from tabulate import tabulate

    EXTENDED_START = "2003-01-02"  # DFII10's actual start date -- the real ceiling

    print(f"Fetching extended history from {EXTENDED_START} (this covers ~23 years and "
          f"will take noticeably longer than a normal run)...")
    market = fetch_data.get_market_data(period="max")
    fred = fetch_data.get_fred_data(start=EXTENDED_START)
    merged = analysis.merge_datasets(market, fred)
    extended_signals = analysis.compute_signals(merged)

    if merged.empty:
        print("[error] Could not fetch extended history.")
        return

    extended_complete = extended_signals.dropna(subset=["gold_spot", "dxy", "dfii10"])
    print(f"\nExtended complete-data range: {extended_complete.index.min().date()} to "
          f"{extended_complete.index.max().date()} ({len(extended_complete)} days)")

    feat_df = hmm_regime.build_features(extended_signals)
    if len(feat_df) < 200:
        print("[error] Not enough extended data to run a meaningful comparison.")
        return

    print("\n--- Data quality check: futures contract-roll artifacts ---")
    print("(GC=F is futures data; roll events can produce implausible single-day spikes that")
    print(" aren't real market moves -- we already found and confirmed one such case before.")
    print(f" Over ~23 years this dataset could contain many roll events, previously unscreened.)\n")
    flagged_df = hmm_regime.flag_roll_artifacts(feat_df)
    n_flagged = int(flagged_df["likely_roll_artifact"].sum())
    print(f"  Flagged {n_flagged} day(s) out of {len(feat_df)} as likely roll artifacts "
          f"(single-day gold move > {hmm_regime.ROLL_ARTIFACT_THRESHOLD*100:.0f}%).")
    if n_flagged > 0:
        flagged_dates = flagged_df[flagged_df["likely_roll_artifact"]].index
        sample = flagged_dates[:10]
        print(f"  Sample dates: {[d.date().isoformat() for d in sample]}"
              f"{' (+more)' if n_flagged > 10 else ''}")

    feat_df_clean = flagged_df[~flagged_df["likely_roll_artifact"]][hmm_regime.FEATURE_COLS]
    feat_df_raw = feat_df  # preserve explicitly -- `feat_df` gets reassigned below

    print("\n--- Step 1: Does more history support a different number of states? ---")
    print("(Run twice -- WITH and WITHOUT flagged artifact days -- to check whether the")
    print(" state-count conclusion actually depends on them, or is robust either way.)\n")

    comparison = hmm_regime.compare_state_counts(feat_df, candidates=(2, 3, 4, 5, 6))
    print("  WITH artifacts (unfiltered):")
    table = []
    for n, r in sorted(comparison.items()):
        if "error" in r:
            table.append([n, "ERROR", "", "", ""])
        else:
            table.append([n, f"{r['train_ll']:.2f}", f"{r['test_ll']:.2f}", f"{r['gap']:.2f}",
                          f"{r['n_states_used']}/{n}"])
    print(tabulate(table, headers=["n_states", "Train LL", "Test LL", "Gap", "States used"], tablefmt="simple"))
    recommended_n_raw = hmm_regime.recommend_state_count(comparison)

    if n_flagged > 0 and len(feat_df_clean) >= 200:
        comparison_clean = hmm_regime.compare_state_counts(feat_df_clean, candidates=(2, 3, 4, 5, 6))
        print("\n  WITHOUT artifacts (filtered):")
        table_clean = []
        for n, r in sorted(comparison_clean.items()):
            if "error" in r:
                table_clean.append([n, "ERROR", "", "", ""])
            else:
                table_clean.append([n, f"{r['train_ll']:.2f}", f"{r['test_ll']:.2f}", f"{r['gap']:.2f}",
                                     f"{r['n_states_used']}/{n}"])
        print(tabulate(table_clean, headers=["n_states", "Train LL", "Test LL", "Gap", "States used"], tablefmt="simple"))
        recommended_n_clean = hmm_regime.recommend_state_count(comparison_clean)

        print(f"\n  Candidate state count WITH artifacts:    {recommended_n_raw}")
        print(f"  Candidate state count WITHOUT artifacts: {recommended_n_clean}")
        if recommended_n_raw == recommended_n_clean:
            print(f"  -> MATCH. The state-count conclusion does not appear to depend on the "
                  f"flagged artifacts -- reassuring.")
        else:
            print(f"  -> MISMATCH. The state-count conclusion CHANGES depending on whether artifacts "
                  f"are included -- a real reason for caution. Proceeding with the ARTIFACT-FREE "
                  f"result below, since that's the more trustworthy read.")

        # Use the clean data and clean recommendation going forward -- the
        # more trustworthy choice given we know these artifacts are real
        # data glitches, not real market moves.
        feat_df = feat_df_clean
        comparison = comparison_clean
        recommended_n = recommended_n_clean
    else:
        if n_flagged > 0:
            print(f"\n  [warn] Too few clean observations after removing artifacts to re-run the "
                  f"comparison -- proceeding with unfiltered data.")
        recommended_n = recommended_n_raw

    print("\n--- Step 1b: Threshold sensitivity -- does the artifact threshold itself matter? ---")
    print("(Re-running the filter+comparison at several thresholds around the 6% default.")
    print(" If the recommended state count stays the same across all of them, that's a")
    print(" stronger, more robust conclusion than relying on one arbitrarily-chosen cutoff.)\n")
    threshold_sweep = {}
    for thresh in (0.05, 0.06, 0.07, 0.08):
        flagged_at_thresh = hmm_regime.flag_roll_artifacts(feat_df_raw, threshold=thresh)
        clean_at_thresh = flagged_at_thresh[~flagged_at_thresh["likely_roll_artifact"]][hmm_regime.FEATURE_COLS]
        n_at_thresh = int(flagged_at_thresh["likely_roll_artifact"].sum())
        if len(clean_at_thresh) >= 200:
            comp_at_thresh = hmm_regime.compare_state_counts(clean_at_thresh, candidates=(2, 3, 4, 5, 6))
            rec_at_thresh = hmm_regime.recommend_state_count(comp_at_thresh)
        else:
            rec_at_thresh = None
        threshold_sweep[thresh] = {"n_flagged": n_at_thresh, "recommended_n": rec_at_thresh}

    sweep_table = [[f"{t*100:.0f}%", d["n_flagged"], d["recommended_n"] if d["recommended_n"] else "n/a"]
                   for t, d in sorted(threshold_sweep.items())]
    print(tabulate(sweep_table, headers=["Threshold", "Days flagged", "Recommended n_states"], tablefmt="simple"))
    sweep_ns = [d["recommended_n"] for d in threshold_sweep.values() if d["recommended_n"] is not None]
    threshold_robust = len(set(sweep_ns)) <= 1 if sweep_ns else False
    if threshold_robust:
        print(f"\n  -> ROBUST: recommended state count ({sweep_ns[0] if sweep_ns else 'n/a'}) is unchanged "
              f"across every threshold tested (5-8%). Not sensitive to exactly where the cutoff is drawn.")
    else:
        print(f"\n  -> SENSITIVE: recommended state count changes depending on the exact threshold used "
              f"({sweep_ns}). Treat the specific state count with more caution.")

    original_n = hmm_regime.N_STATES
    print(f"\nCandidate state count on extended data: {recommended_n}")
    print("(This improves statistical fit -- it does NOT by itself mean these states are")
    print(" economically meaningful or decision-useful. The HMM optimizes likelihood, not")
    print(" usefulness. See the archetype table below and judge for yourself whether these")
    print(" states represent genuinely different market behavior worth acting on.)")
    if recommended_n == original_n:
        print(f"-> MATCHES the original {original_n}-state model. The original choice holds up at scale.")
    else:
        print(f"-> DIFFERS from the original {original_n}-state model. The extended history suggests "
              f"the current model's structure may be too coarse/fine -- worth investigating further "
              f"before trusting the short-window model's exact state definitions.")

    print(f"\n--- Step 2: Fitting candidate model with n_states={recommended_n} ---")
    if n_flagged > 0 and len(feat_df_clean) >= 200:
        print(f"  (Using ARTIFACT-FILTERED features -- {n_flagged} flagged day(s) excluded from fitting)")
        extended_result = hmm_regime.analyze_regime(
            extended_signals, n_states=recommended_n, precomputed_feat_df=feat_df
        )

        print(f"\n--- Step 2b: State/archetype/persistence stability -- raw vs. artifact-filtered fit ---")
        print("(Fitting a SECOND model on the RAW (unfiltered) data at the SAME state count, purely")
        print(" to compare against the clean fit -- same state count matching doesn't guarantee the")
        print(" states themselves, or how persistent they are, stayed the same.)\n")
        extended_result_raw = hmm_regime.analyze_regime(
            extended_signals, n_states=recommended_n, precomputed_feat_df=feat_df_raw
        )

        print("  Artifact log (exact days excluded from fitting):")
        artifact_rows = flagged_df[flagged_df["likely_roll_artifact"]]
        log_table = [[d.date().isoformat(), f"{r['gold_ret']*100:+.2f}%"] for d, r in artifact_rows.iterrows()]
        print(tabulate(log_table, headers=["Date", "Gold return"], tablefmt="simple"))

        print("\n  State frequency: raw fit vs. filtered fit (matched by rank, low-to-high return):")
        raw_freqs = sorted(extended_result_raw["state_frequency"].items(),
                            key=lambda kv: next(c["gold_ret_mean"] for c in extended_result_raw["state_characteristics"] if c["label"] == kv[0]))
        clean_freqs = sorted(extended_result["state_frequency"].items(),
                              key=lambda kv: next(c["gold_ret_mean"] for c in extended_result["state_characteristics"] if c["label"] == kv[0]))
        freq_stab_table = []
        max_freq_diff = 0
        for i in range(min(len(raw_freqs), len(clean_freqs))):
            raw_lbl, raw_pct = raw_freqs[i]
            clean_lbl, clean_pct = clean_freqs[i]
            diff = abs(raw_pct - clean_pct)
            max_freq_diff = max(max_freq_diff, diff)
            freq_stab_table.append([f"Rank {i}", f"{raw_lbl}: {raw_pct}%", f"{clean_lbl}: {clean_pct}%", f"{diff:+.1f}pp"])
        print(tabulate(freq_stab_table, headers=["", "Raw fit", "Filtered fit", "Diff"], tablefmt="simple"))

        print("\n  Archetype means: raw fit vs. filtered fit (matched by rank):")
        raw_chars = sorted(extended_result_raw["state_characteristics"], key=lambda c: c["gold_ret_mean"])
        clean_chars = sorted(extended_result["state_characteristics"], key=lambda c: c["gold_ret_mean"])
        char_stab_table = []
        for i in range(min(len(raw_chars), len(clean_chars))):
            rc, cc = raw_chars[i], clean_chars[i]
            char_stab_table.append([
                f"Rank {i}",
                f"{rc.get('gold_ret_mean', 0)*100:+.3f}%", f"{cc.get('gold_ret_mean', 0)*100:+.3f}%",
                f"{rc.get('gold_vol_5d_mean', 0)*100:.3f}%", f"{cc.get('gold_vol_5d_mean', 0)*100:.3f}%",
            ])
        print(tabulate(char_stab_table, headers=["", "Raw ret", "Filtered ret", "Raw vol", "Filtered vol"], tablefmt="simple"))

        print("\n  Self-persistence (diagonal of transition matrix): raw fit vs. filtered fit:")
        raw_tm = extended_result_raw["transition_matrix"]
        clean_tm = extended_result["transition_matrix"]
        pers_table = []
        max_persist_diff = 0
        for i in range(min(len(raw_freqs), len(clean_freqs))):
            raw_lbl = raw_freqs[i][0]
            clean_lbl = clean_freqs[i][0]
            raw_persist = raw_tm.loc[raw_lbl, raw_lbl] * 100 if raw_lbl in raw_tm.index else None
            clean_persist = clean_tm.loc[clean_lbl, clean_lbl] * 100 if clean_lbl in clean_tm.index else None
            if raw_persist is not None and clean_persist is not None:
                diff = abs(raw_persist - clean_persist)
                max_persist_diff = max(max_persist_diff, diff)
                pers_table.append([f"Rank {i}", f"{raw_persist:.1f}%", f"{clean_persist:.1f}%", f"{diff:+.1f}pp"])
        print(tabulate(pers_table, headers=["", "Raw fit", "Filtered fit", "Diff"], tablefmt="simple"))

        stability_robust = max_freq_diff <= 5 and max_persist_diff <= 10
        if stability_robust:
            print(f"\n  -> STABLE: max frequency shift {max_freq_diff:.1f}pp, max persistence shift "
                  f"{max_persist_diff:.1f}pp -- both small. Filtering artifacts changed the state count "
                  f"story very little beyond removing the artifacts themselves.")
        else:
            print(f"\n  -> SENSITIVE: max frequency shift {max_freq_diff:.1f}pp, max persistence shift "
                  f"{max_persist_diff:.1f}pp -- at least one is large. The artifacts were doing more than "
                  f"just adding noise; they were meaningfully shaping state composition or persistence.")
    else:
        extended_result = hmm_regime.analyze_regime(extended_signals, n_states=recommended_n)
        stability_robust = None
        max_freq_diff = None
        max_persist_diff = None

    print(f"\n--- Step 3: Loading existing 1-year cached model for comparison ---")
    price_path = fetch_data.config.PRICE_CACHE
    short_result = None
    if os.path.exists(price_path):
        short_price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
        short_result = hmm_regime.analyze_regime(short_price_df)
    else:
        print(f"[warn] No cached short-window model found at {price_path} -- "
              f"run 'python main.py' first for a full side-by-side comparison.")

    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON: 1-YEAR MODEL vs. EXTENDED-HISTORY MODEL")
    print("=" * 70)

    if short_result and extended_result:
        comp_table = [
            ["Date range",
             f"{short_result['labeled_df'].index.min().date()} to {short_result['labeled_df'].index.max().date()}",
             f"{extended_result['labeled_df'].index.min().date()} to {extended_result['labeled_df'].index.max().date()}"],
            ["Observations", len(short_result['labeled_df']), len(extended_result['labeled_df'])],
            ["States used", hmm_regime.N_STATES, recommended_n],
            ["Model health (gap)",
             "OK" if short_result['model_healthy'] else "OVERFIT WARNING",
             "OK" if extended_result['model_healthy'] else "OVERFIT WARNING"],
        ]
        print(tabulate(comp_table, headers=["", "1-Year Model", "Extended Model"], tablefmt="simple"))

        print("\nCurrent state probabilities:")
        short_probs = short_result["current_probs"]
        ext_probs = extended_result["current_probs"]
        all_labels = sorted(set(short_probs) | set(ext_probs))
        probs_table = [[lbl, f"{short_probs.get(lbl, 0)*100:.1f}%", f"{ext_probs.get(lbl, 0)*100:.1f}%"]
                        for lbl in all_labels]
        print(tabulate(probs_table, headers=["State", "1-Year Model", "Extended Model"], tablefmt="simple"))

        print("\nState frequency (share of all days):")
        short_freq = short_result["state_frequency"]
        ext_freq = extended_result["state_frequency"]
        all_freq_labels = sorted(set(short_freq) | set(ext_freq))
        freq_table = [[lbl, f"{short_freq.get(lbl, 0)}%", f"{ext_freq.get(lbl, 0)}%"] for lbl in all_freq_labels]
        print(tabulate(freq_table, headers=["State", "1-Year Model", "Extended Model"], tablefmt="simple"))

        print("\n--- What does each EXTENDED state actually look like? ---")
        print("(Sorted by mean gold return, ascending -- this is what 'State 0', 'State 1', etc. mean)")
        print("(Archetype names are a RULE-BASED heuristic -- return/vol rank relative to the other")
        print(" states in THIS fit, not a validated economic classification. Treat as a readable label,")
        print(" not a rigorous claim.)\n")
        all_chars = extended_result["state_characteristics"]
        char_table = []
        for c in all_chars:
            archetype_name, archetype_desc = hmm_regime.name_state_archetype(c, all_chars)
            char_table.append([
                c["label"], archetype_name, c["n"],
                f"{c.get('gold_ret_mean', 0)*100:+.3f}%", f"{c.get('gold_ret_std', 0)*100:.3f}%",
                f"{c.get('dxy_ret_mean', 0)*100:+.3f}%",
                f"{c.get('real_yield_chg_mean', 0)*100:+.4f}pp",
                f"{c.get('gold_vol_5d_mean', 0)*100:.3f}%",
            ])
        print(tabulate(char_table, headers=["State", "Archetype", "n", "Gold ret (mean)", "Gold ret (std)",
                                              "DXY ret (mean)", "Real yield chg (mean)", "5d vol (mean)"],
                        tablefmt="simple"))

        print("\n--- Which extended archetype do TODAY's actual conditions resemble? ---")
        print("(Scores today's most recent short-window data against the extended model's fitted")
        print(" state distributions -- a bridge between your daily 3-state model and the richer")
        print(" extended taxonomy, even though today's model only knows 3 states itself.)\n")
        if short_result.get("feat_df") is not None and not short_result["feat_df"].empty:
            today_row = short_result["feat_df"].iloc[-1]

            print("  Raw feature vector being scored (today's actual, UNSTANDARDIZED values --")
            print("  same raw numbers your short model uses, standardized here using the EXTENDED")
            print("  model's own mean/std, not the short model's):")
            for col in hmm_regime.FEATURE_COLS:
                print(f"    {col}: {today_row[col]:.6f}")

            # SEQUENCE-AWARE (primary, trustworthy): appends today onto the
            # extended model's OWN real historical sequence, so the fitted
            # transition matrix and yesterday's likely state properly
            # inform today's read -- this is the methodologically correct
            # answer, verified to exactly reconstruct analyze_regime's own
            # current-state computation when tested against known data.
            today_scores_seq = hmm_regime.score_feature_vector_sequence_aware(
                today_row, extended_result["feat_df"], extended_result["model"],
                extended_result["mean"], extended_result["std"], extended_result["label_map"]
            )
            # ISOLATED (secondary, shown for comparison only): scores
            # today's row with NO history/transition context at all -- can
            # disagree sharply with the sequence-aware read, especially
            # when states have very different fitted variances. Do not
            # treat this as "today's state" on its own.
            today_scores_isolated = hmm_regime.score_feature_vector_isolated(
                today_row, extended_result["model"], extended_result["mean"],
                extended_result["std"], extended_result["label_map"]
            )

            top_label = max(today_scores_seq, key=today_scores_seq.get)
            top_prob = today_scores_seq[top_label]

            print(f"\n  SEQUENCE-AWARE result (primary -- accounts for the extended model's own")
            print(f"  transition history, i.e. 'what state were we likely in yesterday'):")
            for lbl, p in today_scores_seq.items():
                print(f"    {lbl}: {p:.10f}")

            top_char = next((c for c in all_chars if c["label"] == top_label), None)
            if top_char:
                top_name, top_desc = hmm_regime.name_state_archetype(top_char, all_chars)
                print(f"\n  Best match (sequence-aware): {top_label} -- \"{top_name}\" ({top_prob*100:.4f}% probability)")
                print(f"  {top_desc}")

            print(f"\n  ISOLATED result (secondary -- for comparison ONLY, no transition context,")
            print(f"  ignores what state we were likely in yesterday):")
            for lbl, p in today_scores_isolated.items():
                print(f"    {lbl}: {p:.10f}")
            isolated_top = max(today_scores_isolated, key=today_scores_isolated.get)
            if isolated_top != top_label:
                print(f"\n  [!] Sequence-aware and isolated methods DISAGREE on today's best match "
                      f"({top_label} vs. {isolated_top}). This is a real, meaningful divergence, not "
                      f"noise -- trust the sequence-aware result above; the isolated one ignores "
                      f"today's actual position in the historical sequence.")
            if top_prob > 0.999999 or today_scores_isolated[isolated_top] > 0.999999:
                print(f"  [!] At least one method is EXTREMELY close to exact 1.0 -- worth sanity-checking")
                print(f"      that no state's fitted variance has collapsed to near-zero (see std values below).")

            print("\n  Full breakdown, sequence-aware (4 decimal places):")
            score_table = [[lbl, f"{p*100:.4f}%"] for lbl, p in today_scores_seq.items()]
            print(tabulate(score_table, headers=["Extended State", "Probability"], tablefmt="simple"))

            print("\n  Sanity check -- each state's fitted volatility (std), to rule out a collapsed")
            print("  covariance artificially inflating one state's likelihood:")
            std_table = [[c["label"], f"{c.get('gold_ret_std', 0)*100:.4f}%"] for c in all_chars]
            print(tabulate(std_table, headers=["State", "Gold ret std"], tablefmt="simple"))
            std_values = [c.get("gold_ret_std", 0) for c in all_chars]
            if min(std_values) > 0 and max(std_values) / min(std_values) > 20:
                print(f"  [!] Wide spread in fitted volatility across states (max/min ratio "
                      f"{max(std_values)/min(std_values):.0f}x) -- this alone can make one state's "
                      f"likelihood dominate even for modest feature differences. Not necessarily wrong, "
                      f"but worth knowing this is a contributing factor to how decisive the match looks.")
        else:
            print("  Could not score today's conditions (short-window feature data unavailable).")

        print("\n--- Conclusions ---")
        if recommended_n == hmm_regime.N_STATES:
            print(f"  - State count ({hmm_regime.N_STATES}) is CONSISTENT across both windows.")
        else:
            print(f"  - State count DIFFERS: 1-year model uses {hmm_regime.N_STATES}, extended "
                  f"history supports {recommended_n}. Treat the 1-year model's exact state "
                  f"definitions with caution until this is reconciled.")

        short_top_state = max(short_probs, key=short_probs.get)
        common_states = set(short_freq) & set(ext_freq)
        if common_states:
            freq_diffs = {s: abs(short_freq.get(s, 0) - ext_freq.get(s, 0)) for s in common_states}
            max_diff_state = max(freq_diffs, key=freq_diffs.get)
            if freq_diffs[max_diff_state] > 15:
                print(f"  - '{max_diff_state}' occurs at very different rates in each window "
                      f"({short_freq.get(max_diff_state,0)}% vs {ext_freq.get(max_diff_state,0)}%) -- "
                      f"a real sign the 1-year window's regime is NOT representative of the longer history.")
            else:
                print(f"  - State frequencies are broadly similar between the two windows -- "
                      f"no dramatic sign the current period is a statistical outlier vs. the full history.")

        print(f"\n  Reminder: this comparison does NOT tell you which model is 'right' for trading --")
        print(f"  it tells you whether the current 1-year model's structure and behavior generalizes.")
        print(f"  A real divergence here is valuable information, not a failure.")
    else:
        print("[warn] Could not build full comparison -- missing one of the two models.")

    print("\n" + "=" * 70)
    print("RESEARCH INTEGRITY SUMMARY")
    print("=" * 70)
    print(f"  Roll artifacts detected:                    {n_flagged} day(s)")
    print(f"  Artifacts excluded from fitting:             {'Yes' if n_flagged > 0 and len(feat_df_clean) >= 200 else 'N/A -- none detected or too few clean obs remained'}")
    print(f"  State-count changed after filtering:         "
          f"{'N/A' if n_flagged == 0 else ('No' if recommended_n_raw == recommended_n else 'Yes')}")
    print(f"  Robust across artifact threshold (5-8%):     "
          f"{'Yes' if threshold_robust else 'No' if sweep_ns else 'N/A'}")
    print(f"  State/archetype/persistence stability:       "
          f"{'Stable' if stability_robust else ('Sensitive' if stability_robust is False else 'N/A')}")
    overall_robust = (
        (n_flagged == 0 or recommended_n_raw == recommended_n)
        and (threshold_robust if sweep_ns else True)
        and (stability_robust if stability_robust is not None else True)
    )
    print(f"\n  Overall conclusion: "
          f"{'ROBUST to detected roll artifacts.' if overall_robust else 'SENSITIVE to roll artifacts -- treat extended conclusions with extra caution.'}")
    print("=" * 70)


def run_validation(min_train_days=90, refit_every=10):
    """
    The validation layer: does the regime model actually have predictive
    value, tested honestly (walk-forward, no look-ahead)? Plus multi-day
    transition forecasting and exact Shapley factor decomposition.

    Uses your cached price_history.csv (same data your daily briefing
    uses). This can take a minute or two -- it refits the HMM many times
    to stay honest about not using future data.
    """
    from tabulate import tabulate

    price_path = fetch_data.config.PRICE_CACHE
    if not os.path.exists(price_path):
        print(f"[error] No cached price data found at {price_path}. "
              f"Run 'python main.py' first to build the cache.")
        return

    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)

    print("--- Walk-Forward Validation ---")
    print("(Refitting the regime model periodically using ONLY past data at each point --")
    print(" no look-ahead. This is slower than a normal run; that's expected.)\n")

    wf_labels = validation.walk_forward_regime_labels(
        price_df, min_train_days=min_train_days, refit_every=refit_every
    )
    if wf_labels is None or wf_labels.empty:
        print("[warn] Not enough data for walk-forward validation yet "
              f"(need at least {min_train_days + 20} days).")
    else:
        fwd_rets = validation.forward_returns(price_df, horizons=(1, 3, 5, 10))
        analysis = validation.regime_conditioned_analysis(wf_labels, fwd_rets, horizons=(1, 3, 5, 10))

        if analysis:
            for h, res in analysis.items():
                print(f"\n{h}-day forward returns (baseline: {res['baseline_mean_pct']:+.3f}%, "
                      f"n={res['baseline_n']}):")
                table = [[r["regime"], r["n"], f"{r['mean_fwd_ret_pct']:+.3f}%",
                          f"{r['std_fwd_ret_pct']:.3f}%",
                          f"{r['z_vs_baseline']:+.2f}" if pd.notna(r['z_vs_baseline']) else "n/a"]
                         for r in res["by_regime"]]
                print(tabulate(table, headers=["Regime", "n", "Mean fwd ret", "Std", "z vs baseline"], tablefmt="simple"))
            print("\n  Reading the z-score: |z| > ~2 is a rough signal the regime's forward")
            print("  returns differ meaningfully from the unconditional baseline. With this")
            print("  little data per state, treat this as suggestive, not conclusive --")
            print("  the honest answer to 'would this have made money' needs more history.")
        else:
            print("[warn] Could not compute regime-conditioned forward returns.")

    # Multi-day transition forecast, using the regular (non-walk-forward) fit
    print("\n--- Multi-Day Transition Forecast ---")
    regime_result = hmm_regime.analyze_regime(price_df)
    if regime_result:
        forecast = validation.multi_day_transition_forecast(
            regime_result["transition_matrix"], regime_result["current_probs"], n_days_list=(1, 3, 5, 10)
        )
        table = []
        for n_days, probs in forecast.items():
            row = [n_days] + [f"{probs.get(s, 0)*100:.1f}%" for s in regime_result["transition_matrix"].index]
            table.append(row)
        print(tabulate(table, headers=["Days ahead"] + list(regime_result["transition_matrix"].index), tablefmt="simple"))
    else:
        print("[warn] Could not compute multi-day forecast (no regime fit available).")

    # Exact Shapley decomposition of the daily factor regression
    print("\n--- Exact Shapley Decomposition (DXY / Real Yield / VIX contribution to gold's daily moves) ---")
    feat_df = factor_attribution.build_features(price_df)
    shapley = validation.shapley_decomposition(feat_df)
    if shapley:
        table = [[k, f"{v:.4f}", f"{v/shapley['full_r2']*100:.1f}%" if shapley['full_r2'] else "n/a"]
                 for k, v in shapley["shapley_values"].items()]
        print(tabulate(table, headers=["Factor", "Shapley R² share", "% of explained variance"], tablefmt="simple"))
        print(f"\n  Full model R²: {shapley['full_r2']:.4f}  "
              f"(sum of Shapley values: {shapley['sum_check']:.4f} -- exact by construction)")
        print(f"  n={shapley['n_obs']} days")
    else:
        print("[warn] Not enough data for Shapley decomposition.")


def run_flow_check(fx_period="1y"):
    """
    Cross-validates the HMM regime model against REAL currency risk-flow
    behavior -- these are two independently-derived signals (the regime
    model only ever sees gold/DXY/real yields; this only sees EUR/GBP/JPY/
    CHF), so if they agree, that's genuine external confirmation rather
    than the model just agreeing with itself.

    Checks: when the regime model says "Declining" (a sell-off state),
    do safe-haven currencies (JPY/CHF) actually outperform risk currencies
    (EUR/GBP) that day, as classic risk-off flows would predict? And the
    reverse for "Rising"?

    Uses your CACHED price_history.csv for the regime side (same data your
    regular daily briefing uses -- so the regime read here matches what
    you saw there), rather than an independent fresh fetch. This means no
    year-over-year split is possible with only ~1 year cached -- that
    section is skipped/noted rather than shown misleadingly. The FX side
    still does a live fetch (no FX cache currently exists to reuse).
    """
    price_path = fetch_data.config.PRICE_CACHE
    if not os.path.exists(price_path):
        print(f"[error] No cached price data found at {price_path}. "
              f"Run 'python main.py' first to build the cache.")
        return

    print(f"Using cached price data from {price_path} (matches your daily briefing)...")
    signals = pd.read_csv(price_path, index_col=0, parse_dates=True)

    regime_result = hmm_regime.analyze_regime(signals)
    if regime_result is None:
        print("[error] Could not fit regime model from cached data -- not enough data.")
        return

    print(f"Fetching FX data for risk-sentiment scoring ({fx_period}, live)...")
    raw_fx = xau_currency_score.fetch_data(period=fx_period)
    scored_fx = xau_currency_score.build_scores(raw_fx)
    scored_fx = xau_currency_score.risk_sentiment_scores(scored_fx)

    # Join regime labels onto the FX risk-sentiment data by date
    regime_labels = regime_result["labeled_df"][["label"]].rename(columns={"label": "regime"})
    joined = scored_fx[["risk_on_score", "safehaven_score", "risk_spread"]].join(regime_labels, how="inner")

    if len(joined) < 20:
        print(f"[warn] Only {len(joined)} overlapping days between cached regime data and live FX "
              f"data -- too few for a reliable check. This can happen if your cache is older than "
              f"the FX lookback window; try running 'python main.py' again first to refresh it.")
        return

    print(f"\n--- Risk-Flow Sanity Check: does real FX behavior match the regime model? ---")
    print(f"({len(joined)} overlapping days, {joined.index.min().date()} to {joined.index.max().date()})\n")

    from tabulate import tabulate
    grouped = joined.groupby("regime")[["risk_on_score", "safehaven_score", "risk_spread"]].mean() * 100
    counts = joined["regime"].value_counts()

    table = []
    for regime in grouped.index:
        n = counts.get(regime, 0)
        row = grouped.loc[regime]
        expected = ("Risk-OFF expected (havens up)" if regime == "Declining"
                    else "Risk-ON expected (risk ccys up)" if regime == "Rising"
                    else "No strong expectation")
        matches = (
            (regime == "Declining" and row["risk_spread"] < 0) or
            (regime == "Rising" and row["risk_spread"] > 0) or
            (regime == "Range")
        )
        table.append([
            regime, int(n), f"{row['risk_on_score']:+.3f}%", f"{row['safehaven_score']:+.3f}%",
            f"{row['risk_spread']:+.3f}%", expected, "YES" if matches else "NO -- unexpected"
        ])
    print(tabulate(table, headers=["Regime", "Days", "Risk-on avg", "Safehaven avg", "Spread", "Expected", "Matches?"], tablefmt="simple"))

    # Year-over-year split, if enough history
    joined["year"] = joined.index.year
    years = sorted(joined["year"].unique())
    if len(years) >= 2:
        print(f"\n--- Same check, split by year (does the relationship hold up over time?) ---")
        for year in years:
            year_data = joined[joined["year"] == year]
            if len(year_data) < 20:
                print(f"\n{year}: only {len(year_data)} days, too few to break out separately.")
                continue
            print(f"\n{year} ({len(year_data)} days):")
            year_grouped = year_data.groupby("regime")[["risk_spread"]].mean() * 100
            year_counts = year_data["regime"].value_counts()
            yr_table = [[r, int(year_counts.get(r, 0)), f"{year_grouped.loc[r, 'risk_spread']:+.3f}%"]
                        for r in year_grouped.index]
            print(tabulate(yr_table, headers=["Regime", "Days", "Avg risk_spread"], tablefmt="simple"))
    else:
        print(f"\n[info] Only {len(years)} calendar year of data in the current cache -- "
              f"no year-over-year split possible yet. This will become available "
              f"automatically as your cache accumulates more history over time.")


def run_factors_only():
    """
    Re-runs just the factor attribution using CACHED data already on disk
    (data/price_history.csv, data/reserves_history.csv) -- no live fetch.
    Useful for re-checking the analysis without waiting on network calls,
    e.g. right after a daily run, or to try a different lookback slice.
    """
    price_path = fetch_data.config.PRICE_CACHE
    reserves_path = os.path.join(fetch_data.config.DATA_DIR, "reserves_history.csv")

    if not os.path.exists(price_path):
        print(f"[error] No cached price data found at {price_path}. "
              f"Run 'python main.py' first to build the cache.")
        return

    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)

    reserves_df = pd.DataFrame()
    if os.path.exists(reserves_path):
        reserves_df = pd.read_csv(reserves_path)
    else:
        print(f"[info] No cached reserves history found at {reserves_path} -- "
              f"central bank buying section will be skipped. Run 'python main.py' "
              f"at least once to build it.")

    _print_factor_attribution(price_df, reserves_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Macro Daily Briefing System")
    parser.add_argument("--log-thesis", action="store_true", help="Log this week's thesis")
    parser.add_argument("--hit-rate", action="store_true", help="Show your running accuracy")
    parser.add_argument("--period", default=None,
                         help="Lookback window for market data (e.g. 6mo, 1y, 2y). Default: 1y.")
    parser.add_argument("--factors", action="store_true",
                         help="Re-run factor attribution on cached data only, no live fetch")
    parser.add_argument("--fx-score", action="store_true",
                         help="Run the independent XAU currency strength decomposition "
                              "(USD-driven vs gold-driven), live fetch, standalone")
    parser.add_argument("--fx-days", type=int, default=20,
                         help="Number of recent days to show in --fx-score (default 20)")
    parser.add_argument("--fx-detail-days", type=int, default=10,
                         help="Number of recent days to show in the raw pairs/crosses tables (default 10)")
    parser.add_argument("--flow-check", action="store_true",
                         help="Cross-validate the regime model (from your cached price data, matching "
                              "your daily briefing) against real FX risk-on/risk-off flows (live fetch)")
    parser.add_argument("--validate", action="store_true",
                         help="Walk-forward validation: does the regime model actually predict forward "
                              "returns, tested honestly (no look-ahead)? Plus multi-day transition "
                              "forecast and exact Shapley factor decomposition. Uses cached data, no live fetch.")
    parser.add_argument("--extended", action="store_true",
                         help="Fetch FULL available history (back to 2003, DFII10's real start date), "
                              "re-validate the state count on this larger sample, and compare side by "
                              "side against your existing 1-year cached model. Live fetch, takes a while.")
    parser.add_argument("--cot", action="store_true",
                         help="Standalone COT (Commitments of Traders) report: Managed Money positioning "
                              "in COMEX gold, with 1yr/3yr/all-history COT Index. Live fetch from CFTC, "
                              "free, no key needed.")
    args = parser.parse_args()

    if args.log_thesis:
        log_thesis_interactive()
    elif args.hit_rate:
        print(journal.hit_rate_summary())
    elif args.factors:
        run_factors_only()
    elif args.fx_score:
        xau_currency_score.run_report(period=args.period or "1y", n_days=args.fx_days, detail_days=args.fx_detail_days)
    elif args.flow_check:
        run_flow_check(fx_period=args.period or "1y")
    elif args.validate:
        run_validation()
    elif args.extended:
        run_extended()
    elif args.cot:
        cot_analysis.print_report()
    else:
        run_daily_briefing(period=args.period or "1y")
