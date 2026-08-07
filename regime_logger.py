"""
regime_logger.py
-----------------
Standalone script run by a GitHub Actions cron (NOT part of the Streamlit
app itself, and not imported by it) that takes one regime snapshot and
appends it to a rolling 7-day log.

Deliberately decoupled from git/branch mechanics: this script only reads
and writes plain CSV files on local disk. The GitHub Actions workflow
(.github/workflows/regime_logger.yml) handles fetching the existing log
from the 'data' branch beforehand and committing the result back to it
afterward -- keeping the *logging* commits off the 'main' branch that
Streamlit Cloud watches, so the live dashboard never gets rebooted by a
15-minute cron tick.

Usage:
    python regime_logger.py --existing path/to/old_log.csv --output path/to/new_log.csv

If --existing is omitted, doesn't exist, or is empty, starts a fresh log.
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

import fetch_data
import analysis
import hmm_regime

RETENTION_DAYS = 7
COLUMNS = [
    "timestamp_utc", "top_state", "top_prob",
    "prob_declining", "prob_range", "prob_rising",
]


def compute_current_regime():
    """
    Runs the same fetch -> signals -> HMM pipeline the dashboard uses, and
    returns one row (dict) describing the CURRENT regime read.
    """
    data = fetch_data.fetch_all(period="2y")  # HMM needs real history to fit properly
    merged = analysis.merge_datasets(data["market"], data["fred"])
    if merged.empty:
        raise RuntimeError("No merged market/FRED data available this run.")
    signals = analysis.compute_signals(merged)

    result = hmm_regime.analyze_regime(signals)
    if result is None:
        raise RuntimeError("hmm_regime.analyze_regime returned None (not enough data to fit).")

    probs = result["current_probs"]  # dict, already sorted descending by probability
    top_state, top_prob = next(iter(probs.items()))

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "top_state": top_state,
        "top_prob": round(top_prob, 4),
        "prob_declining": round(probs.get("Declining", float("nan")), 4),
        "prob_range": round(probs.get("Range", float("nan")), 4),
        "prob_rising": round(probs.get("Rising", float("nan")), 4),
    }
    return row


def load_existing(path):
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            df = pd.read_csv(path)
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA
            return df[COLUMNS]
        except Exception as e:
            print(f"[warn] Could not read existing log ({e}) -- starting fresh.")
    return pd.DataFrame(columns=COLUMNS)


def prune_old_rows(df, retention_days=RETENTION_DAYS):
    if df.empty:
        return df
    df = df.copy()
    # format="ISO8601" parses each row's own ISO variant independently --
    # without it, pandas infers ONE strict format from the first row and
    # silently turns every other row into NaT (which then gets dropped
    # below), if timestamp strings ever differ even slightly (e.g. with vs
    # without microseconds). Caught this via testing before it shipped.
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, format="ISO8601", errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    df = df[df["timestamp_utc"] >= cutoff].copy()
    df["timestamp_utc"] = df["timestamp_utc"].apply(lambda t: t.isoformat())
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", default=None, help="Path to existing log CSV, if any")
    parser.add_argument("--output", required=True, help="Path to write the updated log CSV")
    args = parser.parse_args()

    existing_df = load_existing(args.existing)

    try:
        new_row = compute_current_regime()
    except Exception as e:
        print(f"[error] Could not compute regime this run: {e}")
        # Still write out the pruned existing log so 7-day retention keeps
        # working even on a run where live computation failed (e.g. a
        # transient network blip) -- better than losing history over that.
        pruned = prune_old_rows(existing_df)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        pruned.to_csv(args.output, index=False)
        raise

    new_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
    new_df = prune_old_rows(new_df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    new_df.to_csv(args.output, index=False)
    print(
        f"Logged regime: {new_row['top_state']} ({new_row['top_prob']:.1%}) "
        f"-- {len(new_df)} rows retained (last {RETENTION_DAYS} days)"
    )


if __name__ == "__main__":
    main()
