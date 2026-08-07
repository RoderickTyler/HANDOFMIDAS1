"""
basis_monitor.py

Purpose
-------
Tracks the structural disparity between the three "gold" instruments feeding
the assessment engine:

    1. Spot gold      (XAU/USD)
    2. COMEX GC futures (the underlying for your options/GEX layer)
    3. GLD ETF         (the underlying for your GLD-options layer)

It produces two independent, auditable checks:

    - GC Basis        = GC_futures_close - Spot_close        (cost-of-carry gap)
    - GLD Tracking Dev = (GLD_close * oz_per_share) - Spot_close   (ETF tracking error)

...plus a roll-period detector for GC (based on open-interest crossover between
the front-month and next-month contracts), since basis behaves differently
in the days around a roll.

Design principles (per the spec discussed):
    - Never overwrite an existing confidence/probability number. Every flag is
      appended as a separate field alongside the original, never mutating it.
    - No invented discount percentages. Until a flag's historical impact has
      been validated (see `validate_flag_impact`), the adjusted-confidence
      field is left as None and only the flag + reason is shown.
    - GC basis and GLD tracking deviation are tracked and flagged SEPARATELY.
      They are never blended into one "disparity" number.

Expected input data
--------------------
This module does NOT fetch live data (no market-data API is reachable from
this environment). It expects you to supply daily OHLC/OI history as CSVs,
pulled from your chosen sources (e.g. CME DataMine for GC, a broker/data API
or Yahoo-style feed for GLD, and an XAU spot feed). See `load_data()` for the
exact expected columns, and `_demo_dataset()` at the bottom for a synthetic
example you can run immediately to see the full pipeline work end-to-end.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    spot_csv: str,
    gc_front_csv: str,
    gc_next_csv: str,
    gld_csv: str,
) -> pd.DataFrame:
    """
    Load and merge the four required daily series on Date.

    Expected columns per file:
        spot_csv:     Date, Close                      (XAU/USD spot close)
        gc_front_csv: Date, Close, OpenInterest         (front-month GC contract)
        gc_next_csv:  Date, Close, OpenInterest         (next-month GC contract)
        gld_csv:      Date, Close, OzPerShare           (GLD close + that day's
                                                          published gold ounces
                                                          per share, from the
                                                          SPDR daily fact sheet
                                                          NAV data — this value
                                                          drifts slowly over
                                                          time as the trust
                                                          accrues expenses, so
                                                          do not hardcode a
                                                          constant)

    Returns a single DataFrame indexed by Date with all series aligned.
    """
    spot = pd.read_csv(spot_csv, parse_dates=["Date"]).rename(columns={"Close": "Spot"})
    gc_f = pd.read_csv(gc_front_csv, parse_dates=["Date"]).rename(
        columns={"Close": "GC_Front", "OpenInterest": "OI_Front"}
    )
    gc_n = pd.read_csv(gc_next_csv, parse_dates=["Date"]).rename(
        columns={"Close": "GC_Next", "OpenInterest": "OI_Next"}
    )
    gld = pd.read_csv(gld_csv, parse_dates=["Date"]).rename(
        columns={"Close": "GLD", "OzPerShare": "GLD_OzPerShare"}
    )

    df = (
        spot[["Date", "Spot"]]
        .merge(gc_f[["Date", "GC_Front", "OI_Front"]], on="Date", how="inner")
        .merge(gc_n[["Date", "GC_Next", "OI_Next"]], on="Date", how="inner")
        .merge(gld[["Date", "GLD", "GLD_OzPerShare"]], on="Date", how="inner")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_gc_basis(df: pd.DataFrame) -> pd.DataFrame:
    """GC futures (front-month) minus spot, in $/oz."""
    df = df.copy()
    df["GC_Basis"] = df["GC_Front"] - df["Spot"]
    return df


def compute_gld_tracking_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    GLD's spot-gold-equivalent price minus actual spot, in $/oz.

    GLD share price = OzPerShare * Spot, so the spot-equivalent is
    GLD price DIVIDED BY OzPerShare (not multiplied — that was an earlier
    bug in this function, caught via a real-data sanity check where the
    multiplied version produced a ~$4000 "deviation," an obvious red flag).
    """
    df = df.copy()
    df["GLD_SpotEquivalent"] = df["GLD"] / df["GLD_OzPerShare"]
    df["GLD_TrackingDeviation"] = df["GLD_SpotEquivalent"] - df["Spot"]
    return df


def compute_rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """
    For each point, the percentile rank (0-100) of that value within the
    trailing `window` observations. Used to judge whether today's basis /
    tracking deviation is unusual relative to its own recent history, rather
    than against an arbitrary fixed threshold.
    """
    def pct_rank(x: np.ndarray) -> float:
        current = x[-1]
        return 100.0 * (x < current).sum() / (len(x) - 1) if len(x) > 1 else np.nan

    return series.rolling(window, min_periods=max(20, window // 4)).apply(
        pct_rank, raw=True
    )


def detect_roll_period(
    df: pd.DataFrame,
    lead_days: int = 10,
    lag_days: int = 3,
) -> pd.Series:
    """
    Flags days as being inside a "roll window" using open-interest crossover:
    the point where OI_Next first exceeds OI_Front marks the effective roll.
    Everything from `lead_days` before that crossover through `lag_days`
    after it is flagged True — this is the period during which the
    front-month contract's OI is actively decaying and its GEX/wall
    readings should be treated as less reliable.

    If OI_Next never crosses OI_Front in the given data (e.g. still early
    in the contract's life), no flag is raised.
    """
    crossed = df["OI_Next"] > df["OI_Front"]
    if not crossed.any():
        return pd.Series(False, index=df.index)

    crossover_idx = crossed.idxmax()  # first True
    start = max(0, crossover_idx - lead_days)
    end = min(len(df) - 1, crossover_idx + lag_days)

    flag = pd.Series(False, index=df.index)
    flag.iloc[start : end + 1] = True
    return flag


# ---------------------------------------------------------------------------
# Flag assembly (non-destructive: original values untouched)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DisparityFlag:
    name: str
    active: bool
    detail: str
    validated_adjustment_pct: Optional[float] = None  # None until backtested


@dataclasses.dataclass
class DisparitySnapshot:
    date: pd.Timestamp
    gc_basis: float
    gc_basis_percentile: float
    gld_tracking_deviation: float
    gld_tracking_percentile: float
    roll_flag: DisparityFlag
    gc_basis_flag: DisparityFlag
    gld_tracking_flag: DisparityFlag

    def apply_to(self, confidence_label: str, original_pct: float) -> str:
        """
        Renders a display line for an existing confidence/probability field,
        appending flag notes WITHOUT altering the original number. This is
        the function your assessment-output layer should call for any field
        (e.g. 'Probability Support Holds', wall 'Confidence', etc.) that is
        derived even partly from GC-options or GLD-options data.
        """
        notes = []
        for flag in (self.roll_flag, self.gc_basis_flag, self.gld_tracking_flag):
            if flag.active:
                adj = (
                    f" -> adjusted: {flag.validated_adjustment_pct:.0f}%"
                    if flag.validated_adjustment_pct is not None
                    else " -> historical adjustment not yet validated"
                )
                notes.append(f"[WARNING] {flag.name}: {flag.detail}{adj}")

        line = f"{confidence_label}: {original_pct:.0f}%"
        if notes:
            line += "   " + " | ".join(notes)
        return line


def build_snapshot(
    df: pd.DataFrame,
    percentile_window: int = 252,
    basis_hi_pct: float = 80.0,
    basis_lo_pct: float = 20.0,
    roll_lead_days: int = 10,
    roll_lag_days: int = 3,
) -> pd.DataFrame:
    """
    Runs the full pipeline and returns the dataframe with all derived columns
    plus boolean flag columns, ready to be sliced for the latest date or
    iterated for backtesting.
    """
    df = compute_gc_basis(df)
    df = compute_gld_tracking_deviation(df)

    df["GC_Basis_Percentile"] = compute_rolling_percentile(
        df["GC_Basis"], window=percentile_window
    )
    df["GLD_Tracking_Percentile"] = compute_rolling_percentile(
        df["GLD_TrackingDeviation"], window=percentile_window
    )

    df["Roll_Flag"] = detect_roll_period(
        df, lead_days=roll_lead_days, lag_days=roll_lag_days
    )
    df["GC_Basis_Flag"] = (
        df["GC_Basis_Percentile"] >= basis_hi_pct
    ) | (df["GC_Basis_Percentile"] <= basis_lo_pct)
    df["GLD_Tracking_Flag"] = (
        df["GLD_Tracking_Percentile"] >= basis_hi_pct
    ) | (df["GLD_Tracking_Percentile"] <= basis_lo_pct)

    return df


def latest_snapshot(df: pd.DataFrame) -> DisparitySnapshot:
    """Convenience accessor: pulls today's flags in the DisparitySnapshot shape."""
    row = df.iloc[-1]

    roll_flag = DisparityFlag(
        name="Roll period active",
        active=bool(row["Roll_Flag"]),
        detail="GC front-month OI decaying vs. next contract — wall/GEX reliability reduced",
    )
    gc_flag = DisparityFlag(
        name="GC basis unusual",
        active=bool(row["GC_Basis_Flag"]),
        detail=f"basis at {row['GC_Basis_Percentile']:.0f}th percentile (trailing window)",
    )
    gld_flag = DisparityFlag(
        name="GLD tracking deviation unusual",
        active=bool(row["GLD_Tracking_Flag"]),
        detail=f"deviation at {row['GLD_Tracking_Percentile']:.0f}th percentile (trailing window)",
    )

    return DisparitySnapshot(
        date=row["Date"],
        gc_basis=row["GC_Basis"],
        gc_basis_percentile=row["GC_Basis_Percentile"],
        gld_tracking_deviation=row["GLD_TrackingDeviation"],
        gld_tracking_percentile=row["GLD_Tracking_Percentile"],
        roll_flag=roll_flag,
        gc_basis_flag=gc_flag,
        gld_tracking_flag=gld_flag,
    )


# ---------------------------------------------------------------------------
# Historical validation (this is what should EVENTUALLY populate
# validated_adjustment_pct — never guess this number, derive it)
# ---------------------------------------------------------------------------

def validate_flag_impact(
    df: pd.DataFrame,
    flag_col: str,
    outcome_col: str,
) -> dict:
    """
    Compares the historical hit-rate of `outcome_col` (e.g. a boolean column
    you log elsewhere: "did this support level hold?") between days where
    `flag_col` was True vs. False. Returns the empirical difference, which is
    the only legitimate source for a validated_adjustment_pct — not a guess.

    This requires you to have logged outcome_col alongside the flags over
    time (part of the Phase 1 logging you're already building for GEX/wall
    validation). Until that history exists, this function has nothing to
    compute and validated_adjustment_pct should stay None.
    """
    flagged = df[df[flag_col] == True][outcome_col]      # noqa: E712
    unflagged = df[df[flag_col] == False][outcome_col]    # noqa: E712

    if len(flagged) < 20 or len(unflagged) < 20:
        return {
            "sufficient_data": False,
            "note": "Need at least ~20 historical instances of each condition "
                    "before trusting this comparison.",
        }

    return {
        "sufficient_data": True,
        "hit_rate_flagged": flagged.mean(),
        "hit_rate_unflagged": unflagged.mean(),
        "difference_pct_points": 100 * (flagged.mean() - unflagged.mean()),
        "n_flagged": len(flagged),
        "n_unflagged": len(unflagged),
    }


# ---------------------------------------------------------------------------
# Demo — synthetic dataset so the pipeline is runnable end-to-end right now,
# without waiting on real CME/broker data. Replace with load_data(...) once
# you have real CSVs.
# ---------------------------------------------------------------------------

def _demo_dataset(n_days: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-06-01", periods=n_days)

    spot = 2300 + np.cumsum(rng.normal(0, 6, n_days))

    # GC front-month: spot + carry premium that drifts with a slow rate cycle,
    # widening noticeably in the last stretch of each contract's life.
    carry = 8 + 4 * np.sin(np.linspace(0, 6 * np.pi, n_days)) + rng.normal(0, 1.2, n_days)
    gc_front = spot + carry

    # GC next-month: similar but carries a bit more premium (further dated)
    gc_next = spot + carry + 6 + rng.normal(0, 1.0, n_days)

    # Open interest: front month declines, next month builds, crossing
    # roughly every ~42 trading days (simulating a bi-monthly cycle chunk)
    cycle_len = 42
    oi_front = []
    oi_next = []
    for i in range(n_days):
        phase = i % cycle_len
        oi_front.append(max(5000, 200000 - phase * 4000 + rng.normal(0, 3000)))
        oi_next.append(min(220000, phase * 5200 + rng.normal(0, 3000)))

    gld_oz_per_share = np.linspace(0.0929, 0.0925, n_days)  # slow expense drift
    gld = (spot + rng.normal(0, 0.8, n_days)) / gld_oz_per_share / 100  # rough share price scale
    # (scaling above is illustrative only — replace with real GLD closes)

    return pd.DataFrame(
        {
            "Date": dates,
            "Spot": spot,
            "GC_Front": gc_front,
            "OI_Front": oi_front,
            "GC_Next": gc_next,
            "OI_Next": oi_next,
            "GLD": gld,
            "GLD_OzPerShare": gld_oz_per_share,
        }
    )


if __name__ == "__main__":
    demo = _demo_dataset()
    result = build_snapshot(demo)
    snap = latest_snapshot(result)

    print(f"Snapshot date: {snap.date.date()}")
    print()
    print(snap.apply_to("Strongest Support Confidence", 95.0))
    print(snap.apply_to("Probability Support Holds", 81.0))
    print()
    print("Raw values:")
    print(f"  GC Basis: {snap.gc_basis:+.2f}  (percentile: {snap.gc_basis_percentile:.0f})")
    print(
        f"  GLD Tracking Deviation: {snap.gld_tracking_deviation:+.2f}  "
        f"(percentile: {snap.gld_tracking_percentile:.0f})"
    )
    print(f"  Roll period active: {snap.roll_flag.active}")
