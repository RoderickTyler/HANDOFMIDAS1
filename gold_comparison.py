"""
gold_comparison.py

Free (yfinance, no signup) puller for the three "gold" instruments, feeding
directly into basis_monitor.py's existing GC-basis / GLD-tracking-deviation
functions with real historical data.

Tickers used (Yahoo Finance):
    XAUUSD=X   spot gold
    GC=F       COMEX gold futures (active/front contract)
    GLD        SPDR Gold Shares ETF

Note on GLD_OzPerShare: GLD's gold-per-share backing drifts slowly downward
over time as the trust accrues expenses (roughly ~0.40%/yr). The constant
below is an approximation — for precision, check SPDR's daily published NAV
figures (spdrgoldshares.com) rather than relying on a hardcoded number,
especially for anything beyond a rough sanity check.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from basis_monitor import (
    compute_gc_basis,
    compute_gld_tracking_deviation,
    compute_rolling_percentile,
)

GLD_OZ_PER_SHARE_APPROX = 0.091881  # implied from GLD/accurate-spot on 2026-07-31 (gold-api.com); still verify against spdrgoldshares.com — this drifts slowly over time

# ---------------------------------------------------------------------------
# Auto-refreshing oz-per-share (hybrid: cached + auto-derive + manual override)
# ---------------------------------------------------------------------------

import json
import os
from datetime import datetime, timezone

_DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gld_oz_per_share_cache.json"
)


def _read_cache(cache_path: str) -> Optional[dict]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(cache_path: str, value: float, source: str) -> None:
    data = {
        "oz_per_share": value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


def set_oz_per_share(value: float, cache_path: str = _DEFAULT_CACHE_PATH) -> None:
    """
    MANUAL OVERRIDE: forces the cached oz-per-share to a specific value you
    provide (e.g. SPDR's officially published NAV-per-share figure, if you
    want ground truth rather than a market-price-derived estimate). This
    becomes the value get_oz_per_share() returns until it goes stale again
    or you override it once more.
    """
    _write_cache(cache_path, value, source="manual override")
    print(f"oz-per-share manually set to {value:.6f} and cached.")


def get_oz_per_share(
    cache_path: str = _DEFAULT_CACHE_PATH,
    max_staleness_hours: float = 24.0,
    force_refresh: bool = False,
) -> float:
    """
    Returns the current best oz-per-share figure, following this order:

      1. If a cached value exists and is younger than max_staleness_hours
         (and force_refresh is False), return it as-is — avoids jittering
         a number that should move slowly, by not re-deriving it from noisy
         live quotes on every single call.
      2. Otherwise, try to derive a fresh value from live prices you're
         already pulling elsewhere (fetch_live_gld_spot + 
         fetch_live_spot_gold_api) — free, no extra API cost, since both
         are already called for the Live Reference Snapshot section.
      3. If that live derivation fails (network issue, etc.), fall back to
         the last cached value even if stale, with a warning.
      4. If there's no cache at all AND live derivation fails, fall back to
         the hardcoded GLD_OZ_PER_SHARE_APPROX constant, with a warning.

    Use set_oz_per_share() to manually pin a specific value (e.g. from
    SPDR's official published NAV) instead of the market-derived estimate.
    """
    cached = _read_cache(cache_path)

    if cached is not None and not force_refresh:
        cached_time = datetime.fromisoformat(cached["timestamp"])
        age_hours = (datetime.now(timezone.utc) - cached_time).total_seconds() / 3600.0
        if age_hours < max_staleness_hours:
            return cached["oz_per_share"]

    # Cache missing or stale (or force_refresh) — try to derive fresh.
    try:
        from gex_engine import fetch_live_gld_spot
        live_gld = fetch_live_gld_spot()
        live_spot = fetch_live_spot_gold_api()
        fresh_value = live_gld / live_spot
        _write_cache(cache_path, fresh_value, source="live-derived")
        print(f"oz-per-share refreshed: {fresh_value:.6f} "
              f"(from live GLD {live_gld:.2f} / live spot {live_spot:.2f})")
        return fresh_value
    except Exception as e:
        if cached is not None:
            print(f"Could not refresh oz-per-share live ({e}); "
                  f"using last cached value ({cached['oz_per_share']:.6f}, "
                  f"cached {cached['timestamp']}).")
            return cached["oz_per_share"]
        print(f"Could not refresh oz-per-share live ({e}) and no cache exists; "
              f"falling back to hardcoded constant ({GLD_OZ_PER_SHARE_APPROX:.6f}).")
        return GLD_OZ_PER_SHARE_APPROX


def fetch_live_spot_gold_api() -> float:
    """
    Pulls the current live spot gold price from gold-api.com — free, no API
    key, no signup, CORS-open. This is a real-time SNAPSHOT only (their free
    tier is built around the current price, not a historical series), so
    use this to get today's accurate spot price, and combine it with
    yfinance's GC=F / GLD history for everything else.

    NOTE: the exact endpoint path was inferred from gold-api.com's stated
    pattern (their confirmed /symbols endpoint follows base_url + /symbols,
    so price very likely follows base_url + /price/{symbol}). If this
    returns a 404, check https://gold-api.com/docs directly for the exact
    path and update GOLD_API_PRICE_URL below accordingly.
    """
    import requests

    url = "https://api.gold-api.com/price/XAU"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(
            f"gold-api.com request failed ({resp.status_code}): {resp.text[:200]}. "
            "Check https://gold-api.com/docs for the current exact endpoint path."
        )
    data = resp.json()
    # Response shape not fully confirmed — try common key names defensively.
    for key in ("price", "rate", "value"):
        if key in data:
            return float(data[key])
    raise KeyError(
        f"Unexpected response shape from gold-api.com: {data}. "
        "Inspect the actual JSON keys and adjust this function."
    )


def _pull_spot_gold(period: str) -> pd.DataFrame:
    """
    Yahoo doesn't expose a working spot-gold ticker via yfinance (confirmed
    by testing — both XAU=X and XAUUSD=X 404). This builds a Spot column
    using GC futures history for the shape/trend, then overrides only the
    MOST RECENT value with a real live spot quote from gold-api.com — so
    historical Spot values are still GC-based (slightly basis-shifted) but
    today's value is accurate. Good enough for correlation/trend checks;
    if you need historically-accurate spot for every past day, you'd need
    a paid historical spot data source.
    """
    import yfinance as yf

    gc_hist = yf.download("GC=F", period=period, progress=False)[["Close"]].rename(
        columns={"Close": "Spot"}
    )

    try:
        live_spot = fetch_live_spot_gold_api()
        gc_hist.iloc[-1, gc_hist.columns.get_loc("Spot")] = live_spot
        print(f"Live spot gold from gold-api.com: {live_spot:.2f} "
              f"(overriding most recent row only; earlier history is GC-futures-based)")
    except Exception as e:
        print(f"Could not fetch live spot from gold-api.com ({e}); "
              "falling back to GC futures for all rows, including the most recent.")

    return gc_hist


def pull_gold_comparison_data(period: str = "6mo") -> pd.DataFrame:
    """
    Pulls daily closes for spot, GC futures, and GLD over the given period
    (yfinance period strings: '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'),
    aligns them on date, and returns a single DataFrame ready for
    compute_gc_basis() / compute_gld_tracking_deviation().
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is not installed. Run: pip install yfinance"
        ) from e

    spot = _pull_spot_gold(period)
    gc = yf.download("GC=F", period=period, progress=False)[["Close"]].rename(
        columns={"Close": "GC_Front"}
    )
    gld = yf.download("GLD", period=period, progress=False)[["Close"]].rename(
        columns={"Close": "GLD"}
    )

    df = spot.join(gc, how="inner").join(gld, how="inner")
    df = df.dropna().reset_index()
    df = df.rename(columns={"Date": "Date"})
    df["GLD_OzPerShare"] = GLD_OZ_PER_SHARE_APPROX  # flat approximation, see note above

    # flatten any yfinance multiindex columns if present
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]

    return df


def summarize_disparity(df: pd.DataFrame, percentile_window: int = 60) -> dict:
    """
    Runs the existing basis_monitor calculations on real pulled data and
    returns the latest readings plus rolling correlations — the practical
    "how differently or similarly do these move" answer.
    """
    df = compute_gc_basis(df)
    df = compute_gld_tracking_deviation(df)

    window = min(percentile_window, max(20, len(df) - 1))
    df["GC_Basis_Percentile"] = compute_rolling_percentile(df["GC_Basis"], window=window)
    df["GLD_Tracking_Percentile"] = compute_rolling_percentile(
        df["GLD_TrackingDeviation"], window=window
    )

    # Daily return correlations — the most direct "do they move together" check
    returns = df[["Spot", "GC_Front", "GLD"]].pct_change().dropna()
    corr_gc = returns["Spot"].corr(returns["GC_Front"])
    corr_gld = returns["Spot"].corr(returns["GLD"])

    latest = df.iloc[-1]

    return {
        "date": latest["Date"],
        "spot": latest["Spot"],
        "gc_front": latest["GC_Front"],
        "gld": latest["GLD"],
        "gc_basis": latest["GC_Basis"],
        "gc_basis_percentile": latest["GC_Basis_Percentile"],
        "gld_tracking_deviation": latest["GLD_TrackingDeviation"],
        "gld_tracking_percentile": latest["GLD_Tracking_Percentile"],
        "daily_return_corr_spot_vs_gc": corr_gc,
        "daily_return_corr_spot_vs_gld": corr_gld,
        "n_days": len(df),
    }


def convert_result_to_spot_gold_terms(result, oz_per_share: Optional[float] = None) -> dict:
    """
    Converts a gex_engine.AssessmentResult computed on GLD share prices into
    spot-gold-equivalent ($/oz) terms, using the same GLD price = OzPerShare
    x Spot relationship established in basis_monitor.py.

    oz_per_share: if None (default), auto-resolves via get_oz_per_share()
    (cached, auto-refreshing, live-derived). Pass a specific value to
    override for this call only.

    Only price-like fields are converted (spot, gamma flip, wall strikes).
    GEX values stay in dollars — they represent dealer dollar-exposure, not
    a price, so they don't need unit conversion.

    This is a PROXY, not independently observed data: it tells you what
    price level GLD's structure implies in spot terms, assuming the current
    GLD-to-spot ratio holds. It is not the same as pulling actual COMEX GC
    options data — treat it as a translation of GLD's structure into
    familiar spot-price units, not as direct evidence about spot-gold
    dealer positioning specifically.
    """
    if oz_per_share is None:
        oz_per_share = get_oz_per_share()

    def to_spot(gld_price: float) -> float:
        return gld_price / oz_per_share

    return {
        "underlying": f"{result.underlying} (converted to spot-gold-equivalent proxy)",
        "spot_gold_equivalent": to_spot(result.spot),
        "net_gex": result.net_gex,  # dollar exposure, no conversion needed
        "gamma_flip_spot_equivalent": (
            to_spot(result.gamma_flip) if result.gamma_flip is not None else None
        ),
        "dealer_delta": result.dealer_delta,
        "regime": result.regime,
        "call_walls_spot_equivalent": [
            (to_spot(strike), gex) for strike, gex in result.call_walls
        ],
        "put_walls_spot_equivalent": [
            (to_spot(strike), gex) for strike, gex in result.put_walls
        ],
        "oz_per_share_used": oz_per_share,
    }


def print_spot_gold_proxy_summary(result, oz_per_share: Optional[float] = None) -> None:
    """Pretty-prints the spot-gold-equivalent proxy view alongside a clear caveat.
    oz_per_share=None (default) auto-resolves via get_oz_per_share()."""
    proxy = convert_result_to_spot_gold_terms(result, oz_per_share)

    print(f"Underlying:        {proxy['underlying']}")
    print(f"Spot Gold Equiv:   {proxy['spot_gold_equivalent']:.2f}")
    print(f"Net GEX:           {proxy['net_gex']:,.0f}")
    if proxy["gamma_flip_spot_equivalent"] is not None:
        print(f"Gamma Flip Equiv:  {proxy['gamma_flip_spot_equivalent']:.2f}")
    else:
        print("Gamma Flip Equiv:  not found in range")
    print(f"Dealer Delta:      {proxy['dealer_delta']:,.0f}")
    print(f"Regime:            {proxy['regime']}")
    print()
    print("Resistance candidates (call walls, spot-equivalent):")
    for strike, gex in proxy["call_walls_spot_equivalent"]:
        print(f"  {strike:.2f}   GEX {gex:,.0f}")
    print()
    print("Support candidates (put walls, spot-equivalent):")
    for strike, gex in proxy["put_walls_spot_equivalent"]:
        print(f"  {strike:.2f}   GEX {gex:,.0f}")
    print()
    print(f"(oz/share used: {proxy['oz_per_share_used']:.6f} — a snapshot conversion, drifts slowly over "
          f"time; re-derive periodically from real GLD/spot rather than trusting it long-term. "
          f"This is GLD's structure translated into spot units — a proxy, not independently "
          f"observed COMEX/spot options data.)")


def print_spot_gold_contextual_levels(result, oz_per_share: Optional[float] = None) -> None:
    """
    Spot-gold-equivalent version of AssessmentResult.print_contextual_levels() —
    same true support/resistance/ATM-pin/largest-wall breakdown, but with
    strikes converted to $/oz via the same GLD-price = OzPerShare x Spot
    relationship used elsewhere in this module.

    oz_per_share: if None (default), auto-resolves via get_oz_per_share().
    """
    if oz_per_share is None:
        oz_per_share = get_oz_per_share()

    ctx = result.contextual_levels()

    def to_spot(gld_price: float) -> float:
        return gld_price / oz_per_share

    spot_equiv = to_spot(result.spot)
    print(f"Spot Gold Equiv:   {spot_equiv:.2f}")

    atm_strike, atm_gex = ctx["atm_pin"]
    print(f"ATM Pin:           {to_spot(atm_strike):.2f}   GEX {atm_gex:,.0f}  "
          f"(nearest strike to spot — consolidation/pin signal, not a directional level)")
    print()

    largest = ctx["largest_wall_overall"]
    print(f"Largest wall overall: {to_spot(largest['strike']):.2f}   "
          f"GEX {largest['gex']:,.0f}   [{largest['side']}]")
    print()

    print("Resistance (spot-equivalent, strikes ABOVE spot only, ranked by GEX):")
    if ctx["resistance"]:
        for strike, gex in ctx["resistance"]:
            print(f"  {to_spot(strike):.2f}   GEX {gex:,.0f}")
    else:
        print("  none found above spot in this chain")
    print()

    print("Support (spot-equivalent, strikes BELOW spot only, ranked by GEX):")
    if ctx["support"]:
        for strike, gex in ctx["support"]:
            print(f"  {to_spot(strike):.2f}   GEX {gex:,.0f}")
    else:
        print("  none found below spot in this chain")
    print()
    print(f"(oz/share used: {oz_per_share:.6f} — see caveat in print_spot_gold_proxy_summary above.)")


if __name__ == "__main__":
    data = pull_gold_comparison_data(period="6mo")
    result = summarize_disparity(data)

    print(f"As of {result['date'].date()}  (n={result['n_days']} trading days)")
    print()
    print(f"Spot:              {result['spot']:.2f}")
    print(f"GC Futures:        {result['gc_front']:.2f}")
    print(f"GLD:               {result['gld']:.2f}  (approx spot-equiv: "
          f"{result['gld'] / GLD_OZ_PER_SHARE_APPROX:.2f}, oz/share is approximate)")
    print()
    print(f"GC Basis:          {result['gc_basis']:+.2f}  "
          f"({result['gc_basis_percentile']:.0f}th percentile, trailing window)")
    print(f"GLD Tracking Dev:  {result['gld_tracking_deviation']:+.2f}  "
          f"({result['gld_tracking_percentile']:.0f}th percentile, trailing window)")
    print()
    print(f"Daily return correlation, Spot vs GC:   {result['daily_return_corr_spot_vs_gc']:.4f}")
    print(f"Daily return correlation, Spot vs GLD:  {result['daily_return_corr_spot_vs_gld']:.4f}")
