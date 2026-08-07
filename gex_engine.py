"""
gex_engine.py

The core dealer-structure engine. Takes a single options chain (for ONE
underlying — GLD or GC, never blended) and produces:

    - Net Gamma Exposure (GEX), by strike and in aggregate
    - Gamma flip level (zero-crossing of aggregate GEX vs. hypothetical spot)
    - Net Dealer Delta
    - Ranked call/put walls (largest |Gamma x OI| concentrations)
    - Regime classification: Positive Gamma / Negative Gamma / Near-Flip
    - A plain-language assessment block, matching the output format agreed
      on earlier in the design (Direction / Continuation / Support /
      Resistance / etc.)

Kept deliberately to ONE product per call. If you're running both GLD and
GC, call this twice and compare the two AssessmentResult objects as two
independent votes (per the earlier design) rather than merging chains.

Dealer-side convention (explicit assumption, stated per the "don't treat
estimates as fact" principle from the design discussion):
    Customers are assumed net long calls / net long puts, dealers net
    short calls / net short puts. This is the standard simplifying
    assumption used by most public GEX methodologies. It is NOT verified
    against your actual instrument and should be treated as an estimate.

Greeks: if your data source doesn't supply delta/gamma directly (common for
futures options data), this module computes them via Black-Scholes from
implied volatility. If your source DOES supply Greeks (common for GLD via
broker APIs), pass them directly and skip the BS calculation.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes Greeks (used only when the data source doesn't supply them)
# ---------------------------------------------------------------------------

def bs_delta_gamma(
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    r: float = 0.05,
    option_type: str = "call",
) -> tuple[float, float]:
    """
    Returns (delta, gamma) via Black-Scholes. `gamma` is identical for calls
    and puts; `delta` differs by type. t_years must be > 0.
    """
    if t_years <= 0 or iv <= 0:
        return (0.0, 0.0)

    d1 = (np.log(spot / strike) + (r + 0.5 * iv**2) * t_years) / (iv * np.sqrt(t_years))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(t_years))

    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0

    return (delta, gamma)


def fill_missing_greeks(chain: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
    """
    For any row missing Delta/Gamma, compute via Black-Scholes using IV,
    Strike, TimeToExpiryYears, and OptionType. Rows that already have
    Delta/Gamma populated are left untouched.

    Expected columns: Strike, OptionType ('call'/'put'), OpenInterest,
    TimeToExpiryYears, IV, and optionally Delta, Gamma.
    """
    chain = chain.copy()
    if "Delta" not in chain.columns:
        chain["Delta"] = np.nan
    if "Gamma" not in chain.columns:
        chain["Gamma"] = np.nan

    needs_calc = chain["Delta"].isna() | chain["Gamma"].isna()
    for idx in chain[needs_calc].index:
        row = chain.loc[idx]
        delta, gamma = bs_delta_gamma(
            spot=spot,
            strike=row["Strike"],
            t_years=row["TimeToExpiryYears"],
            iv=row["IV"],
            r=r,
            option_type=row["OptionType"],
        )
        chain.loc[idx, "Delta"] = delta
        chain.loc[idx, "Gamma"] = gamma

    return chain


# ---------------------------------------------------------------------------
# Data ingestion — yfinance (GLD) — free, no signup, no API key
# ---------------------------------------------------------------------------

def scan_expirations_for_liquidity(ticker: str = "GLD", max_to_check: int = 20) -> None:
    """
    Diagnostic helper: for each of the next `max_to_check` expirations,
    prints how many contracts have real (>0) open interest, and whether
    those contracts actually cluster near the current spot price. Run this
    BEFORE fetch_gld_chain_yfinance() when the default pull comes back with
    only deep ITM/OTM junk strikes — it'll show you which expiration
    actually has usable near-the-money data right now.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    spot = float(hist["Close"].iloc[-1]) if not hist.empty else float("nan")

    expirations = tk.options[:max_to_check]
    print(f"Spot: {spot:.2f}\n")
    print(f"{'Expiration':<12} {'#calls w/OI':<12} {'#puts w/OI':<12} {'nearest strike w/OI':<20}")

    for exp in expirations:
        try:
            chain = tk.option_chain(exp)
        except Exception as e:
            print(f"{exp:<12} error: {e}")
            continue

        calls_oi = chain.calls[chain.calls["openInterest"] > 0]
        puts_oi = chain.puts[chain.puts["openInterest"] > 0]

        all_strikes_with_oi = pd.concat([calls_oi["strike"], puts_oi["strike"]])
        if len(all_strikes_with_oi) > 0:
            nearest = all_strikes_with_oi.iloc[
                (all_strikes_with_oi - spot).abs().argsort()[:1]
            ].values[0]
            distance = abs(nearest - spot)
            nearest_str = f"{nearest:.2f} (Δ{distance:.2f})"
        else:
            nearest_str = "none"

        print(f"{exp:<12} {len(calls_oi):<12} {len(puts_oi):<12} {nearest_str:<20}")


def fetch_gld_chain_yfinance(
    ticker: str = "GLD",
    expiration: Optional[str] = None,
    min_open_interest: Optional[int] = None,
) -> tuple[pd.DataFrame, float]:
    """
    Pulls today's live-ish options chain from Yahoo Finance via the
    (unofficial) yfinance library — no API key, no signup, free.

    Caveats worth knowing before relying on this (see accompanying notes):
      - yfinance scrapes an undocumented Yahoo endpoint, not a licensed API.
        It can break or get rate-limited without notice; treat it as a
        starting point, not a guaranteed-stable production dependency.
      - No Delta/Gamma are supplied — this function leaves those columns
        empty and run_assessment() -> fill_missing_greeks() will compute
        them via Black-Scholes from the IV Yahoo does supply.
      - Only the current chain is available — no historical chains, so
        this covers your day-to-day pipeline but NOT the Phase 1
        validation backtest, which needs history from elsewhere.

    Requires: pip install yfinance

    Returns (chain_df, spot_price).
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is not installed. Run: pip install yfinance --break-system-packages"
        ) from e

    tk = yf.Ticker(ticker)

    # Spot price: use the most recent close from a short history pull,
    # since yfinance's `info` dict is slower and less reliable for this.
    hist = tk.history(period="1d")
    if hist.empty:
        raise ValueError(f"Could not retrieve a current price for {ticker}.")
    spot = float(hist["Close"].iloc[-1])

    available_expirations = tk.options
    if not available_expirations:
        raise ValueError(f"No listed option expirations found for {ticker}.")

    if expiration is not None:
        if expiration not in available_expirations:
            raise ValueError(
                f"Expiration {expiration} not available. Choices: {available_expirations}"
            )
        expirations_to_try = [expiration]
    else:
        # Try nearest-dated first, then walk outward if it has no usable
        # data (common right after a weekend, or for very-near-dated
        # expirations with thin/stale open interest).
        expirations_to_try = list(available_expirations)

    last_error = None
    for exp in expirations_to_try:
        try:
            chain, spot_out = _fetch_single_expiration(
                tk, exp, spot, min_open_interest
            )
            if exp != expirations_to_try[0]:
                print(f"Note: nearest expiration(s) had no usable open interest; "
                      f"using {exp} instead.")
            return chain, spot_out
        except ValueError as e:
            last_error = e
            continue

    raise ValueError(
        f"No expiration among {expirations_to_try[:5]}{'...' if len(expirations_to_try) > 5 else ''} "
        f"returned usable contracts. Last error: {last_error}"
    )


def _fetch_single_expiration(
    tk, expiration: str, spot: float, min_open_interest: Optional[int]
) -> tuple[pd.DataFrame, float]:
    """Helper: pulls and parses one expiration's chain. Raises ValueError if empty."""
    opt_chain = tk.option_chain(expiration)

    exp_date = pd.Timestamp(expiration)
    days_to_expiry = max((exp_date - pd.Timestamp.today()).days, 1)
    t_years = days_to_expiry / 365.0

    rows = []
    for opt_type, df in (("call", opt_chain.calls), ("put", opt_chain.puts)):
        for _, r in df.iterrows():
            oi = r.get("openInterest")
            if pd.isna(oi) or oi <= 0:
                continue
            if min_open_interest is not None and oi < min_open_interest:
                continue
            iv = r.get("impliedVolatility")
            if pd.isna(iv) or iv <= 0:
                continue
            rows.append(
                {
                    "Strike": float(r["strike"]),
                    "OptionType": opt_type,
                    "OpenInterest": float(oi),
                    "TimeToExpiryYears": t_years,
                    "IV": float(iv),
                    "Delta": np.nan,
                    "Gamma": np.nan,
                }
            )

    chain = pd.DataFrame(rows)
    if chain.empty:
        raise ValueError(f"No usable contracts for expiration {expiration}.")

    print(
        f"[fetch_gld_chain_yfinance diagnostics] expiration={expiration}  "
        f"contracts={len(chain)}  "
        f"OI range=[{chain['OpenInterest'].min():.0f}, {chain['OpenInterest'].max():.0f}]  "
        f"IV range=[{chain['IV'].min():.4f}, {chain['IV'].max():.4f}]  "
        f"strike range=[{chain['Strike'].min():.2f}, {chain['Strike'].max():.2f}]  "
        f"days_to_expiry={days_to_expiry}"
    )

    return chain, spot


# ---------------------------------------------------------------------------
# Data ingestion — MarketData.app (GLD)
# ---------------------------------------------------------------------------

def fetch_gld_chain_marketdata_app(
    api_key: str,
    underlying: str = "GLD",
    dte_min: Optional[int] = None,
    dte_max: Optional[int] = None,
    min_open_interest: Optional[int] = None,
) -> tuple[pd.DataFrame, float]:
    """
    Pulls a live (or 24h-delayed, on the free tier) options chain from
    MarketData.app and reshapes it into the `chain` DataFrame schema
    `run_assessment()` expects: Strike, OptionType, OpenInterest,
    TimeToExpiryYears, IV, Delta, Gamma.

    IMPORTANT (documented MarketData.app quirk): the API can return HTTP 203
    (Non-Authoritative Information, served from cache) instead of 200 on a
    successful request. Both are treated as success below — many HTTP
    clients only check for 200 by default, which silently breaks in
    production if you copy this pattern elsewhere.

    Returns (chain_df, spot_price).
    """
    url = f"https://api.marketdata.app/v1/options/chain/{underlying}/"
    params = {"format": "json"}
    if dte_min is not None:
        params["dte"] = f">{dte_min}"
    if min_open_interest is not None:
        params["min_open_interest"] = min_open_interest

    resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, params=params)

    if resp.status_code not in (200, 203):
        raise RuntimeError(
            f"MarketData.app request failed: {resp.status_code} {resp.text[:300]}"
        )

    data = resp.json()

    # Response fields are parallel arrays (one array per field, not one
    # object per contract) — reshape into row-per-contract first.
    required_fields = [
        "strike", "side", "dte", "openInterest", "iv",
        "delta", "gamma", "underlyingPrice",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Unexpected response shape, missing fields: {missing}")

    n = len(data["strike"])
    rows = []
    for i in range(n):
        oi = data["openInterest"][i]
        if oi is None or (min_open_interest is not None and oi < min_open_interest):
            continue
        rows.append(
            {
                "Strike": data["strike"][i],
                "OptionType": data["side"][i],  # 'call' / 'put'
                "OpenInterest": float(oi),
                "TimeToExpiryYears": max(data["dte"][i], 0) / 365.0,
                "IV": data["iv"][i],
                # Pre-computed Greeks from the API — pass through directly.
                # fill_missing_greeks() will only recompute rows where these
                # are None (e.g. illiquid/stale contracts the API didn't
                # price), so Black-Scholes is a fallback, not the default.
                "Delta": data["delta"][i],
                "Gamma": data["gamma"][i],
            }
        )

    chain = pd.DataFrame(rows)
    if chain.empty:
        raise ValueError("No contracts returned after filtering — check dte/OI filters.")

    spot = float(data["underlyingPrice"][0])
    return chain, spot


def get_freshest_spot_price(
    marketdata_spot: float,
    marketdata_updated_ts: Optional[float] = None,
    max_staleness_minutes: float = 20.0,
) -> dict:
    """
    Staleness check / routing helper: MarketData.app's free tier can be
    delayed up to 24h, which is fine for the chain (OI/Greeks don't move
    fast) but NOT fine for the spot price used in the gamma-flip and wall
    calculations, where you want the most current reference price you can
    get.

    This does not fetch anything itself (Barchart is not on this
    environment's network allowlist, so live calls have to happen in your
    own runtime). It's a decision helper: it tells you whether the
    MarketData.app spot is fresh enough to trust as-is, or whether you
    should override `spot` in run_assessment() with a fresher quote pulled
    from Barchart (or your broker) before computing GEX/flip/walls.

    Usage:
        result = get_freshest_spot_price(md_spot, md_updated_ts)
        if result["use_alternate_source"]:
            spot = pull_spot_from_barchart(...)  # implement against your
                                                   # Barchart account/plan
        else:
            spot = result["spot"]
    """
    stale = False
    if marketdata_updated_ts is not None:
        import time
        age_minutes = (time.time() - marketdata_updated_ts) / 60.0
        stale = age_minutes > max_staleness_minutes

    return {
        "spot": marketdata_spot,
        "is_stale": stale,
        "use_alternate_source": stale,
        "note": (
            "MarketData.app quote is stale beyond threshold — pull a fresher "
            "spot from Barchart (or your broker feed) and pass it into "
            "run_assessment() instead of this value."
            if stale
            else "MarketData.app quote is within freshness threshold — safe to use directly."
        ),
    }


# ---------------------------------------------------------------------------
# GEX / Delta exposure
# ---------------------------------------------------------------------------

def compute_oi_by_strike(chain: pd.DataFrame) -> pd.DataFrame:
    """
    Raw open interest per strike (NOT GEX-weighted) — needed separately
    from compute_gex_by_strike(), since that function only outputs the
    Greek-weighted dollar exposure and loses the underlying OI figures.
    This is what lets you track "which strikes gained OI" over time,
    independent of gamma/GEX changes.

    Returns a DataFrame indexed by Strike with columns: CallOI, PutOI, TotalOI.
    """
    grouped = chain.groupby(["Strike", "OptionType"])["OpenInterest"].sum().unstack(fill_value=0)
    grouped = grouped.rename(columns={"call": "CallOI", "put": "PutOI"})
    for col in ("CallOI", "PutOI"):
        if col not in grouped.columns:
            grouped[col] = 0.0
    grouped["TotalOI"] = grouped["CallOI"] + grouped["PutOI"]
    return grouped.sort_index()


def compute_gex_by_strike(chain: pd.DataFrame, spot: float, contract_multiplier: int = 100) -> pd.DataFrame:
    """
    Dollar gamma exposure per strike, using the standard convention:
        GEX = OI * Gamma * contract_multiplier * spot^2 * 0.01
    Calls contribute positively, puts negatively (dealer-short-gamma-on-puts
    convention — see module docstring for the assumption being made).

    Returns a DataFrame indexed by Strike with columns: CallGEX, PutGEX, NetGEX.
    """
    df = chain.copy()
    df["SignedGEX"] = np.where(
        df["OptionType"] == "call",
        df["OpenInterest"] * df["Gamma"] * contract_multiplier * spot**2 * 0.01,
        -df["OpenInterest"] * df["Gamma"] * contract_multiplier * spot**2 * 0.01,
    )

    grouped = df.groupby(["Strike", "OptionType"])["SignedGEX"].sum().unstack(fill_value=0)
    grouped = grouped.rename(columns={"call": "CallGEX", "put": "PutGEX"})
    for col in ("CallGEX", "PutGEX"):
        if col not in grouped.columns:
            grouped[col] = 0.0
    grouped["NetGEX"] = grouped["CallGEX"] + grouped["PutGEX"]
    return grouped.sort_index()


def compute_dealer_delta(chain: pd.DataFrame, contract_multiplier: int = 100) -> float:
    """
    Aggregate dealer delta exposure across the whole chain. Positive means
    dealers are net long delta from their positioning (before hedging);
    interpretation of directional lean should be read alongside this sign.
    """
    df = chain.copy()
    signed_delta = np.where(
        df["OptionType"] == "call",
        -df["OpenInterest"] * df["Delta"] * contract_multiplier,   # dealer short the call
        -df["OpenInterest"] * df["Delta"] * contract_multiplier,   # dealer short the put too (short both sides)
    )
    return float(np.sum(signed_delta))


def compute_gamma_flip(
    chain: pd.DataFrame,
    spot: float,
    price_range_pct: float = 0.08,
    n_points: int = 161,
    r: float = 0.05,
    contract_multiplier: int = 100,
) -> Optional[float]:
    """
    Recomputes aggregate NetGEX across a grid of hypothetical spot prices
    (holding OI fixed, letting Gamma vary with spot via Black-Scholes) and
    finds where NetGEX crosses zero. Returns None if no crossing is found
    in the tested range (widen price_range_pct if that happens).
    """
    grid = np.linspace(spot * (1 - price_range_pct), spot * (1 + price_range_pct), n_points)
    net_gex_curve = []

    for hyp_spot in grid:
        total = 0.0
        for _, row in chain.iterrows():
            _, gamma = bs_delta_gamma(
                spot=hyp_spot,
                strike=row["Strike"],
                t_years=row["TimeToExpiryYears"],
                iv=row["IV"],
                r=r,
                option_type=row["OptionType"],
            )
            sign = 1.0 if row["OptionType"] == "call" else -1.0
            total += sign * row["OpenInterest"] * gamma * contract_multiplier * hyp_spot**2 * 0.01
        net_gex_curve.append(total)

    net_gex_curve = np.array(net_gex_curve)
    sign_changes = np.where(np.diff(np.sign(net_gex_curve)) != 0)[0]

    if len(sign_changes) == 0:
        return None

    # Take the crossing nearest current spot
    crossing_idx = min(sign_changes, key=lambda i: abs(grid[i] - spot))
    x0, x1 = grid[crossing_idx], grid[crossing_idx + 1]
    y0, y1 = net_gex_curve[crossing_idx], net_gex_curve[crossing_idx + 1]
    # linear interpolation for the zero crossing
    flip = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    return float(flip)


def rank_walls(gex_by_strike: pd.DataFrame, top_n: int = 3) -> dict:
    """
    Returns the strongest call wall(s) and put wall(s), ranked by absolute
    GEX concentration. Call wall = largest positive CallGEX strikes
    (resistance candidates). Put wall = largest negative PutGEX magnitude
    strikes (support candidates).

    NOTE: this is deliberately RAW and unfiltered by spot — it can surface
    the at-the-money strike as a top "wall" on both sides, since ATM gamma
    is naturally the largest. That's real, useful diagnostic information,
    just not directly actionable as a directional level. For a spot-aware
    view (true support/resistance vs. the ATM pin, separated), use
    AssessmentResult.contextual_levels() instead — it's built to sit
    alongside this raw ranking, not replace it.
    """
    call_walls = gex_by_strike["CallGEX"].sort_values(ascending=False).head(top_n)
    put_walls = gex_by_strike["PutGEX"].abs().sort_values(ascending=False).head(top_n)
    return {
        "call_walls": [(strike, val) for strike, val in call_walls.items()],
        "put_walls": [(strike, val) for strike, val in put_walls.items()],
    }


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(net_gex: float, spot: float, gamma_flip: Optional[float]) -> str:
    """
    Three-state classification, deliberately simple (see design discussion
    on keeping the state space small enough to estimate transitions from).
    """
    if gamma_flip is not None and abs(spot - gamma_flip) / spot < 0.005:
        return "Near-Flip (Transition Zone)"
    if net_gex > 0:
        return "Positive Gamma (Mean-Reversion Bias)"
    return "Negative Gamma (Continuation Bias)"


# ---------------------------------------------------------------------------
# Assembled assessment output
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AssessmentResult:
    underlying: str
    spot: float
    net_gex: float
    gamma_flip: Optional[float]
    dealer_delta: float
    regime: str
    call_walls: list          # raw, unfiltered top-N by GEX (unchanged — kept as diagnostic)
    put_walls: list           # raw, unfiltered top-N by GEX (unchanged — kept as diagnostic)
    gex_by_strike: Optional[pd.DataFrame] = None  # full per-strike table, needed for contextual_levels()
    oi_by_strike: Optional[pd.DataFrame] = None    # raw OI per strike, needed for OI-change tracking

    def summary(self) -> str:
        lines = [
            f"Underlying:        {self.underlying}",
            f"Spot:              {self.spot:.2f}",
            f"Net GEX:           {self.net_gex:,.0f}",
            f"Gamma Flip:        {self.gamma_flip:.2f}" if self.gamma_flip else "Gamma Flip:        not found in range",
            f"Dealer Delta:      {self.dealer_delta:,.0f}",
            f"Regime:            {self.regime}",
            "",
            "Resistance candidates (call walls, RAW — unfiltered by spot, see contextual_levels() for a spot-aware view):",
        ]
        for strike, val in self.call_walls:
            lines.append(f"  {strike:.2f}   GEX {val:,.0f}")
        lines.append("")
        lines.append("Support candidates (put walls, RAW — unfiltered by spot):")
        for strike, val in self.put_walls:
            lines.append(f"  {strike:.2f}   GEX {val:,.0f}")
        return "\n".join(lines)

    def contextual_levels(self, top_n: int = 3) -> dict:
        """
        Spot-aware layer on top of the raw walls above. Answers the actual
        trading question — "what's above me, what's below me, and what's
        the single biggest thing on the board regardless of side" —
        rather than just the largest GEX concentration irrespective of
        whether it's above or below current price (which is what the raw
        call_walls/put_walls can conflate, since the at-the-money strike
        usually has the largest gamma of all and can dominate BOTH lists).

        Returns:
            atm_pin: the strike nearest spot and its GEX — reported
                separately since it's a pinning/consolidation signal, not
                a directional level to trade against.
            resistance: strikes ABOVE spot only, ranked by GEX — genuine
                resistance candidates.
            support: strikes BELOW spot only, ranked by GEX — genuine
                support candidates.
            largest_wall_overall: the single biggest GEX concentration on
                the whole board, whichever side it's on — so you always
                know "what's the biggest thing here" even if it turns out
                to be the ATM pin rather than a directional wall.
        """
        if self.gex_by_strike is None:
            raise ValueError(
                "This AssessmentResult wasn't built with gex_by_strike populated "
                "(older result object) — re-run run_assessment() to get contextual_levels()."
            )

        df = self.gex_by_strike.copy()
        df["AbsGEX"] = df["NetGEX"].abs()
        df["DistanceFromSpot"] = (df.index.to_series() - self.spot).abs()

        atm_idx = df["DistanceFromSpot"].idxmin()
        atm_pin = (float(atm_idx), float(df.loc[atm_idx, "NetGEX"]))

        above = df[df.index > self.spot].sort_values("AbsGEX", ascending=False).head(top_n)
        below = df[df.index < self.spot].sort_values("AbsGEX", ascending=False).head(top_n)

        resistance = [(float(strike), float(row["NetGEX"])) for strike, row in above.iterrows()]
        support = [(float(strike), float(row["NetGEX"])) for strike, row in below.iterrows()]

        overall_idx = df["AbsGEX"].idxmax()
        overall_strike = float(overall_idx)
        overall_gex = float(df.loc[overall_idx, "NetGEX"])
        overall_side = (
            "ATM pin" if overall_idx == atm_idx
            else ("resistance (above spot)" if overall_strike > self.spot else "support (below spot)")
        )

        return {
            "atm_pin": atm_pin,
            "resistance": resistance,
            "support": support,
            "largest_wall_overall": {
                "strike": overall_strike,
                "gex": overall_gex,
                "side": overall_side,
            },
        }

    def print_contextual_levels(self, top_n: int = 3) -> None:
        ctx = self.contextual_levels(top_n=top_n)
        print(f"Spot:              {self.spot:.2f}")
        print(f"ATM Pin:           {ctx['atm_pin'][0]:.2f}   GEX {ctx['atm_pin'][1]:,.0f}  "
              f"(nearest strike to spot — consolidation/pin signal, not a directional level)")
        print()
        print(f"Largest wall overall: {ctx['largest_wall_overall']['strike']:.2f}   "
              f"GEX {ctx['largest_wall_overall']['gex']:,.0f}   "
              f"[{ctx['largest_wall_overall']['side']}]")
        print()
        print("Resistance (strikes ABOVE spot only, ranked by GEX):")
        if ctx["resistance"]:
            for strike, gex in ctx["resistance"]:
                print(f"  {strike:.2f}   GEX {gex:,.0f}")
        else:
            print("  none found above spot in this chain")
        print()
        print("Support (strikes BELOW spot only, ranked by GEX):")
        if ctx["support"]:
            for strike, gex in ctx["support"]:
                print(f"  {strike:.2f}   GEX {gex:,.0f}")
        else:
            print("  none found below spot in this chain")


def run_assessment(
    chain: pd.DataFrame,
    spot: float,
    underlying: str,
    r: float = 0.05,
    contract_multiplier: int = 100,
    verbose: bool = False,
) -> AssessmentResult:
    """
    Full pipeline entry point. `chain` must have columns:
        Strike, OptionType ('call'/'put'), OpenInterest, TimeToExpiryYears, IV
    and optionally Delta, Gamma (filled in automatically if missing).

    Set verbose=True to print diagnostics at each stage — useful for
    tracing exactly where values collapse to zero if that happens.
    """
    chain = fill_missing_greeks(chain, spot=spot, r=r)

    if verbose:
        print(f"[run_assessment diagnostics] after fill_missing_greeks: "
              f"Delta NaN count={chain['Delta'].isna().sum()}  "
              f"Gamma NaN count={chain['Gamma'].isna().sum()}  "
              f"Gamma range=[{chain['Gamma'].min():.6f}, {chain['Gamma'].max():.6f}]  "
              f"Delta range=[{chain['Delta'].min():.4f}, {chain['Delta'].max():.4f}]")

    gex_by_strike = compute_gex_by_strike(chain, spot=spot, contract_multiplier=contract_multiplier)
    oi_by_strike = compute_oi_by_strike(chain)

    if verbose:
        print(f"[run_assessment diagnostics] NetGEX per-strike sum check: "
              f"{gex_by_strike['NetGEX'].abs().sum():,.0f} (should be well above 0)")

    net_gex = float(gex_by_strike["NetGEX"].sum())
    dealer_delta = compute_dealer_delta(chain, contract_multiplier=contract_multiplier)
    gamma_flip = compute_gamma_flip(chain, spot=spot, r=r, contract_multiplier=contract_multiplier)
    regime = classify_regime(net_gex, spot, gamma_flip)
    walls = rank_walls(gex_by_strike)

    return AssessmentResult(
        underlying=underlying,
        spot=spot,
        net_gex=net_gex,
        gamma_flip=gamma_flip,
        dealer_delta=dealer_delta,
        regime=regime,
        call_walls=walls["call_walls"],
        put_walls=walls["put_walls"],
        gex_by_strike=gex_by_strike,
        oi_by_strike=oi_by_strike,
    )


# ---------------------------------------------------------------------------
# Demo — synthetic chain so this runs end-to-end right now
# ---------------------------------------------------------------------------

def _demo_chain(spot: float = 2440.0, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strikes = np.arange(spot - 100, spot + 100, 5)
    rows = []
    for k in strikes:
        for opt_type in ("call", "put"):
            # OI concentrated near round numbers and near-the-money, per typical
            # real-world clustering
            base_oi = 4000 * np.exp(-((k - spot) ** 2) / (2 * 35**2))
            if k % 25 == 0:
                base_oi *= 1.8  # round-number clustering
            oi = max(50, base_oi + rng.normal(0, 300))

            iv = 0.16 + 0.02 * abs(k - spot) / 100 + rng.normal(0, 0.005)
            rows.append(
                {
                    "Strike": k,
                    "OptionType": opt_type,
                    "OpenInterest": oi,
                    "TimeToExpiryYears": 30 / 365,
                    "IV": max(0.05, iv),
                }
            )
    return pd.DataFrame(rows)


def run_full_assessment_with_spot_proxy(
    expiration: Optional[str] = None,
    min_open_interest: Optional[int] = None,
    oz_per_share: Optional[float] = None,
    verbose: bool = False,
    marketdata_api_key: Optional[str] = None,
) -> None:
    """
    One-call convenience wrapper: fetches the GLD chain, runs the GEX
    assessment, prints it, then prints the same result translated into
    spot-gold-equivalent ($/oz) terms underneath — clearly labeled as a
    SELF-CALCULATED PROXY (a unit conversion of GLD's own structure), not
    independently observed COMEX/spot options data.

    Data source: pass marketdata_api_key to use MarketData.app (real OI
    and pre-computed Greeks). If omitted, falls back to yfinance (free,
    no key, but confirmed unreliable OI for GLD as of this build).

    Usage:
        run_full_assessment_with_spot_proxy(marketdata_api_key="YOUR_TOKEN")
        # or, sticking with yfinance:
        run_full_assessment_with_spot_proxy(expiration="2026-08-14")
    """
    from gold_comparison import (
        print_spot_gold_proxy_summary,
        print_spot_gold_contextual_levels,
    )

    if marketdata_api_key is not None:
        chain, spot = fetch_gld_chain_marketdata_app(
            api_key=marketdata_api_key, min_open_interest=min_open_interest
        )
    else:
        chain, spot = fetch_gld_chain_yfinance(
            expiration=expiration, min_open_interest=min_open_interest
        )

    result = run_assessment(chain, spot=spot, underlying="GLD", verbose=verbose)

    print()
    print("=" * 60)
    print("GLD (native, as-traded share price)")
    print("=" * 60)
    print(result.summary())

    print()
    print("=" * 60)
    print("GLD — CONTEXTUAL LEVELS (spot-aware: true support/resistance)")
    print("=" * 60)
    result.print_contextual_levels()

    print()
    print("=" * 60)
    print("SPOT GOLD EQUIVALENT — SELF-CALCULATED PROXY")
    print("(GLD structure converted to $/oz using GLD price = OzPerShare x Spot; ")
    print(" NOT independently observed spot-options data — see caveat below)")
    print("=" * 60)
    print_spot_gold_proxy_summary(result, oz_per_share=oz_per_share)
    print()
    print("=" * 60)
    print("SPOT GOLD EQUIVALENT — CONTEXTUAL LEVELS")
    print("=" * 60)
    print_spot_gold_contextual_levels(result, oz_per_share=oz_per_share)

    print()
    print("=" * 60)
    print("LIVE REFERENCE SNAPSHOT (NEW — independent of chain snapshot age)")
    print("=" * 60)
    print_live_reference_snapshot()

    print()
    print("=" * 60)
    print("DEALER POSITIONING CONTEXT (NEW — net/gross GEX + delta vs volume)")
    print("=" * 60)
    print_dealer_positioning_context(result)


def print_live_reference_snapshot() -> None:
    """
    NEW SECTION — live reference prices, independent of whatever the chain
    source's (possibly delayed) snapshot spot says. Pulls:
      - Live spot gold (XAU/USD) from gold-api.com
      - Live GLD share price from yfinance
    and prints both with a Singapore-time timestamp, so you always know
    exactly how current (or not) the numbers above this section actually are.

    This does NOT change anything about the chain/GEX/walls computation —
    it's a separate, additive "what's the market doing right now" readout.
    """
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        sgt_now = datetime.now(ZoneInfo("Asia/Singapore"))
    except Exception:
        # Fallback if zoneinfo/tzdata isn't available on this system —
        # Singapore is a fixed UTC+8 offset, no daylight saving.
        from datetime import timezone, timedelta
        sgt_now = datetime.now(timezone(timedelta(hours=8)))

    timestamp_str = sgt_now.strftime("%Y-%m-%d %H:%M:%S SGT")

    print(f"Timestamp:         {timestamp_str}")

    try:
        from gold_comparison import fetch_live_spot_gold_api
        live_spot = fetch_live_spot_gold_api()
        print(f"Live Spot Gold:    {live_spot:.2f}   (source: gold-api.com)")
    except Exception as e:
        print(f"Live Spot Gold:    unavailable ({e})")

    try:
        live_gld = fetch_live_gld_spot()
        print(f"Live GLD:          {live_gld:.2f}   (source: yfinance)")
    except Exception as e:
        print(f"Live GLD:          unavailable ({e})")

    print()
    print("(Compare these against the 'Spot' figures printed above — if they")
    print(" differ noticeably, the chain/walls above are from an older snapshot")
    print(" than right now. Chain data itself is NOT re-fetched here — this is")
    print(" a live price check only, to show how stale the analysis above is.)")


def fetch_live_gld_spot() -> float:
    """
    Live-ish GLD share price via yfinance (not any delayed chain-source
    snapshot). Used both as a live reference and as an optional override
    for a stale `spot` value elsewhere.
    """
    import yfinance as yf
    hist = yf.Ticker("GLD").history(period="1d")
    if hist.empty:
        raise ValueError("Could not retrieve a current GLD price from yfinance.")
    return float(hist["Close"].iloc[-1])


# ---------------------------------------------------------------------------
# NEW SECTION — Dealer Positioning Context
# (net-vs-gross GEX concentration + dealer delta vs. real trading volume,
#  both with narrative). Purely additive — does not alter any existing
#  function, output, or field above this point.
# ---------------------------------------------------------------------------

def compute_gex_concentration(result: "AssessmentResult") -> dict:
    """
    Compares net GEX against gross (absolute-value-summed) GEX across all
    strikes. A small net relative to a large gross means calls and puts are
    largely cancelling each other out — a two-sided, moderate-conviction
    book — even if the net figure alone looks directional. A net figure
    that's a large fraction of gross means the book is genuinely lopsided.

    Thresholds are simple, fixed cutoffs for narrative purposes only — not
    statistically derived. Treat the narrative as a starting heuristic, not
    a validated signal.
    """
    if result.gex_by_strike is None:
        raise ValueError("AssessmentResult has no gex_by_strike — re-run run_assessment().")

    gross = float(result.gex_by_strike["NetGEX"].abs().sum())
    net = result.net_gex
    ratio = abs(net) / gross if gross > 0 else 0.0

    if ratio < 0.15:
        narrative = (
            "Highly balanced book — calls and puts largely offset each other. "
            "The net regime label is real but reflects a modest tilt on top of "
            "a mostly two-sided position. Low conviction from this layer alone."
        )
    elif ratio < 0.35:
        narrative = (
            "Moderately balanced book — some net tilt, but a large share of "
            "gross exposure is offsetting. Treat the regime label as a "
            "moderate-confidence secondary input, not a dominant signal."
        )
    else:
        narrative = (
            "Lopsided book — net GEX represents a large share of gross exposure. "
            "This is a higher-conviction directional/regime reading than the "
            "typical case."
        )

    return {"gross_gex": gross, "net_gex": net, "ratio": ratio, "narrative": narrative}


def print_gex_concentration(result: "AssessmentResult") -> None:
    ctx = compute_gex_concentration(result)
    print(f"Gross GEX (sum |NetGEX| across strikes): {ctx['gross_gex']:,.0f}")
    print(f"Net GEX:                                 {ctx['net_gex']:,.0f}")
    print(f"Net / Gross ratio:                       {ctx['ratio']:.1%}")
    print()
    print(ctx["narrative"])


def fetch_gld_10day_avg_volume() -> float:
    """Live 10-day average GLD share volume via yfinance, no key required."""
    import yfinance as yf
    hist = yf.Ticker("GLD").history(period="10d")
    if hist.empty or "Volume" not in hist.columns:
        raise ValueError("Could not retrieve GLD volume history from yfinance.")
    return float(hist["Volume"].mean())


def compute_dealer_delta_context(result: "AssessmentResult") -> dict:
    """
    Compares Dealer Delta (a share-equivalent figure) against GLD's live
    10-day average trading volume, to judge whether the dealer positioning
    is large enough, relative to normal market size, to matter.

    Thresholds (<10% / 10-40% / >40%) are simple fixed cutoffs for
    narrative purposes, not statistically derived — same caveat as
    compute_gex_concentration().
    """
    avg_volume = fetch_gld_10day_avg_volume()
    dealer_delta = result.dealer_delta
    ratio = abs(dealer_delta) / avg_volume if avg_volume > 0 else 0.0

    if ratio < 0.10:
        size_narrative = (
            "Small relative to normal trading size — background noise, "
            "unlikely to be a meaningful factor on its own."
        )
    elif ratio < 0.40:
        size_narrative = (
            "Moderate relative to normal trading size — worth weighting as a "
            "secondary confirming/contradicting check against your macro view, "
            "not a standalone signal."
        )
    else:
        size_narrative = (
            "Large relative to normal trading size — this dealer positioning "
            "is substantial enough to be a meaningful secondary input."
        )

    direction_narrative = (
        "Dealers are net long delta (positive) — a mild bullish-leaning tilt "
        "in the book. Static exposure, not an active hedging force by itself "
        "(that's gamma's job) — read it as a confirming/contradicting check "
        "against your macro thesis, not a trigger."
        if dealer_delta > 0 else
        "Dealers are net short delta (negative) — a mild bearish-leaning tilt "
        "in the book. Static exposure, not an active hedging force by itself "
        "(that's gamma's job) — read it as a confirming/contradicting check "
        "against your macro thesis, not a trigger."
    )

    return {
        "dealer_delta": dealer_delta,
        "avg_volume_10d": avg_volume,
        "ratio": ratio,
        "size_narrative": size_narrative,
        "direction_narrative": direction_narrative,
    }


def print_dealer_delta_context(result: "AssessmentResult") -> None:
    ctx = compute_dealer_delta_context(result)
    print(f"Dealer Delta:                {ctx['dealer_delta']:,.0f}")
    print(f"GLD 10-day avg volume:       {ctx['avg_volume_10d']:,.0f}")
    print(f"Dealer Delta / avg volume:   {ctx['ratio']:.1%}")
    print()
    print(ctx["size_narrative"])
    print()
    print(ctx["direction_narrative"])


def print_dealer_positioning_context(result: "AssessmentResult") -> None:
    """
    Combined new section: net-vs-gross GEX concentration, then dealer delta
    vs. real trading volume. Call this in addition to anything else you're
    already printing — it doesn't require or alter the existing summary,
    contextual_levels, or spot-proxy output.
    """
    print("--- GEX concentration (net vs. gross) ---")
    print_gex_concentration(result)
    print()
    print("--- Dealer delta vs. 10-day average volume ---")
    try:
        print_dealer_delta_context(result)
    except Exception as e:
        print(f"Could not fetch volume comparison ({e}); showing dealer delta only: "
              f"{result.dealer_delta:,.0f}")


if __name__ == "__main__":
    print("gex_engine.py loaded. Functions are now available in this session.")
    print()
    print("To run on real data, call one of:")
    print('  run_full_assessment_with_spot_proxy(marketdata_api_key="YOUR_TOKEN")')
    print('  run_full_assessment_with_spot_proxy(expiration="YYYY-MM-DD")  # yfinance path')
    print()
    print("To see the synthetic demo (no real data, just to sanity-check the math):")
    print("  from gex_engine import _demo_chain, run_assessment")
    print('  chain = _demo_chain(spot=2440.0)')
    print('  print(run_assessment(chain, spot=2440.0, underlying="demo").summary())')
