"""
gex_alt.py
-----------
INDEPENDENT module behind the "GEX (Alternative)" tab. It does NOT use
MarketData.app and does NOT import gex_engine, so the original GEX tab and
this one can fail (or be rate-limited) independently of each other.

Data sources (all free, no key):
    - CBOE delayed quotes JSON  (cdn.cboe.com)  -> GLD option chain with
      CBOE's own IV / Delta / Gamma / open interest / bid / ask
    - GVZ via yfinance (^GVZ)                   -> CBOE Gold ETF Volatility Index
    - ^IRX via yfinance                         -> 13-week T-bill, risk-free proxy

What it computes, side by side and labelled separately:
    - CBOE Greeks (as supplied)            vs
    - Self-calculated Black-Scholes Greeks (own IV backed out of the option's
      bid/ask mid, or CBOE's IV -- caller's choice)
    - a deviation column for IV / Delta / Gamma
    - GEX per strike under BOTH Greek sets, net GEX, gamma flip, walls
    - GVZ-based and ATM-IV-based expected range for the selected DTE

VERIFICATION STATUS (be honest about it):
    The CBOE JSON layout below (data.options[*].option / iv / delta / gamma /
    open_interest / bid / ask, data.current_price, top-level timestamp) is
    written from knowledge of that public feed, NOT confirmed against a live
    response from the environment this was built in (the host was blocked
    there). The parser is defensive and reports exactly what it could not
    find, but the first live run on your side is the real test.

Dealer-side convention is the same simplifying assumption the original GEX
tab uses (customers long calls and puts, dealers short both). It is an
assumption, not an observation of real dealer books.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

ET = ZoneInfo("America/New_York")
SGT = ZoneInfo("Asia/Singapore")

CBOE_URL_TEMPLATES = [
    "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json",
    "https://cdn.cboe.com/api/global/delayed_quotes/options/_{sym}.json",
]
_HEADERS = {"User-Agent": "Mozilla/5.0 (HandOfMidas GEX-alt)", "Accept": "application/json"}

# OCC-style symbol: ROOT + YYMMDD + C/P + strike*1000 (8 digits)
_OCC_RE = re.compile(r"^(?P<root>.+?)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<k>\d{8})$")

MIN_T_YEARS = 1.0 / (365.0 * 24.0)  # floor at ~1 hour so 0DTE doesn't blow up
HYBRID_MAX_SPREAD_PCT = 10.0       # "hybrid" mode trusts the mid-derived IV only when (ask-bid)/mid <= this


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

class CboeFetchError(RuntimeError):
    pass


def _parse_asof(ts) -> Optional[datetime]:
    """CBOE's top-level timestamp is a naive 'YYYY-MM-DD HH:MM:SS' string.
    ASSUMPTION: it is US/Eastern exchange time. Returns tz-aware UTC or None."""
    if not ts:
        return None
    try:
        dt = pd.to_datetime(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize(ET)
        return dt.tz_convert("UTC").to_pydatetime()
    except Exception:
        return None


def parse_cboe_payload(payload: dict) -> tuple[pd.DataFrame, dict]:
    """Turns the raw CBOE JSON into a tidy DataFrame + meta dict. Raises
    CboeFetchError with a specific reason if the layout isn't what we expect."""
    if not isinstance(payload, dict) or "data" not in payload:
        raise CboeFetchError(f"Unexpected CBOE JSON: top-level keys = {list(payload)[:8] if isinstance(payload, dict) else type(payload)}")
    data = payload["data"]
    opts = data.get("options")
    if not opts:
        raise CboeFetchError(f"CBOE JSON has no 'options' list (data keys = {list(data)[:12]})")

    spot = data.get("current_price")
    if spot is None:
        spot = data.get("close") or data.get("last_trade_price")
    asof = _parse_asof(payload.get("timestamp"))

    rows = []
    for o in opts:
        m = _OCC_RE.match(str(o.get("option", "")))
        if not m:
            continue
        exp = datetime(2000 + int(m["yy"]), int(m["mm"]), int(m["dd"])).date()
        rows.append({
            "Expiration": exp,
            "Strike": int(m["k"]) / 1000.0,
            "OptionType": "call" if m["cp"] == "C" else "put",
            "OpenInterest": o.get("open_interest"),
            "Volume": o.get("volume"),
            "Bid": o.get("bid"),
            "Ask": o.get("ask"),
            "Last": o.get("last_trade_price"),
            "IV_cboe": o.get("iv"),
            "Delta_cboe": o.get("delta"),
            "Gamma_cboe": o.get("gamma"),
        })
    if not rows:
        raise CboeFetchError("CBOE options list present but no OCC symbols could be parsed.")

    df = pd.DataFrame(rows)
    for c in ["OpenInterest", "Volume", "Bid", "Ask", "Last", "IV_cboe", "Delta_cboe", "Gamma_cboe"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["OpenInterest"] = df["OpenInterest"].fillna(0.0)

    if spot is None or not np.isfinite(float(spot)):
        raise CboeFetchError("CBOE JSON has no usable underlying price (current_price/close).")

    # CBOE quotes option IV as a decimal (0.25). If the median looks like a percent
    # (>3 would mean 300% vol), rescale rather than silently producing garbage.
    iv_rescaled = False
    med_iv = df["IV_cboe"].median()
    if np.isfinite(med_iv) and med_iv > 3.0:
        df["IV_cboe"] = df["IV_cboe"] / 100.0
        iv_rescaled = True

    meta = {
        "iv_rescaled": iv_rescaled,
        "spot": float(spot),
        "asof_utc": asof,
        "asof_raw": payload.get("timestamp"),
        "iv30": data.get("iv30"),
        "n_contracts": int(len(df)),
        "has_cboe_greeks": bool(df["Gamma_cboe"].notna().any()),
    }
    return df, meta


def fetch_cboe_chain(symbol: str = "GLD", timeout: int = 25) -> tuple[pd.DataFrame, dict]:
    """Tries each known URL pattern; returns (chain_df, meta). meta['url'] is the
    one that worked. Raises CboeFetchError listing every attempt on failure."""
    errors = []
    for tmpl in CBOE_URL_TEMPLATES:
        url = tmpl.format(sym=symbol.upper())
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            if r.status_code != 200:
                errors.append(f"{url} -> HTTP {r.status_code}")
                continue
            df, meta = parse_cboe_payload(r.json())
            meta["url"] = url
            return df, meta
        except CboeFetchError as e:
            errors.append(f"{url} -> {e}")
        except Exception as e:
            errors.append(f"{url} -> {type(e).__name__}: {e}")
    raise CboeFetchError("CBOE fetch failed. " + " | ".join(errors))


def fetch_gvz() -> Optional[float]:
    """Latest GVZ close (index points = annualised implied vol in %)."""
    try:
        import yfinance as yf
        h = yf.Ticker("^GVZ").history(period="7d")
        if h.empty:
            return None
        return float(h["Close"].dropna().iloc[-1])
    except Exception:
        return None


def fetch_risk_free(default: float = 0.04) -> tuple[float, str]:
    """13-week T-bill yield (^IRX, quoted in percent) as a decimal. Falls back
    to `default` and says so."""
    try:
        import yfinance as yf
        h = yf.Ticker("^IRX").history(period="7d")
        v = float(h["Close"].dropna().iloc[-1])
        if 0 <= v < 30:
            return v / 100.0, "^IRX (13-week T-bill, yfinance)"
    except Exception:
        pass
    return default, f"fallback constant {default:.2%} (^IRX unavailable)"


# ---------------------------------------------------------------------------
# Black-Scholes (vectorised, European, q = 0 -- GLD pays no dividend)
# ---------------------------------------------------------------------------

def _d1d2(S, K, T, sig, r):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * sqrtT)
    return d1, d1 - sig * sqrtT


def bs_price(S, K, T, sig, r, cp):
    """cp: +1 call / -1 put (array or scalar)."""
    S, K, T, sig, cp = (np.asarray(x, dtype=float) for x in (S, K, T, sig, cp))
    d1, d2 = _d1d2(S, K, T, sig, r)
    return cp * (S * norm.cdf(cp * d1) - K * np.exp(-r * T) * norm.cdf(cp * d2))


def bs_delta_gamma(S, K, T, sig, r, cp):
    S, K, T, sig, cp = (np.asarray(x, dtype=float) for x in (S, K, T, sig, cp))
    d1, _ = _d1d2(S, K, T, sig, r)
    delta = np.where(cp > 0, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = norm.pdf(d1) / (S * sig * np.sqrt(T))
    return delta, gamma


def implied_vol(price, S, K, T, r, cp, lo=1e-4, hi=6.0, iters=70):
    """Vectorised bisection. Returns NaN where the price is outside no-arbitrage
    bounds (below intrinsic or above the sigma=hi price)."""
    price, K, T, cp = (np.asarray(x, dtype=float) for x in (price, K, T, cp))
    S = np.broadcast_to(np.asarray(S, dtype=float), price.shape)
    lo_a = np.full(price.shape, lo)
    hi_a = np.full(price.shape, hi)
    p_lo = bs_price(S, K, T, lo_a, r, cp)
    p_hi = bs_price(S, K, T, hi_a, r, cp)
    valid = np.isfinite(price) & (price >= p_lo - 1e-9) & (price <= p_hi + 1e-9) & (T > 0)
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        p_mid = bs_price(S, K, T, mid, r, cp)
        up = p_mid < price
        lo_a = np.where(up, mid, lo_a)
        hi_a = np.where(up, hi_a, mid)
    out = 0.5 * (lo_a + hi_a)
    return np.where(valid, out, np.nan)


# ---------------------------------------------------------------------------
# Chain enrichment: T, own IV, own Greeks, deviations
# ---------------------------------------------------------------------------

def add_time_to_expiry(df: pd.DataFrame, asof_utc: Optional[datetime]) -> pd.DataFrame:
    """T measured from the data's own as-of time (so it's consistent with how
    CBOE's delayed Greeks were stamped), to 16:00 ET on the expiry date."""
    out = df.copy()
    asof = asof_utc or datetime.now(timezone.utc)
    exp_dt = [datetime(e.year, e.month, e.day, 16, 0, tzinfo=ET).astimezone(timezone.utc) for e in out["Expiration"]]
    secs = np.array([(x - asof).total_seconds() for x in exp_dt])
    out["T_years_raw"] = secs / (365.0 * 24 * 3600)
    out["TimeToExpiryYears"] = np.maximum(out["T_years_raw"], MIN_T_YEARS)
    out["DTE"] = (secs / 86400.0)
    return out


def enrich_chain(df: pd.DataFrame, spot: float, r: float, own_iv_basis: str = "mid") -> pd.DataFrame:
    """
    Adds IV_own (backed out of mid price), then Delta_own / Gamma_own.
    own_iv_basis:
        "mid"   -> Greeks use IV_own where available, else CBOE IV (flagged per row)
        "cboe"  -> Greeks use CBOE's IV (isolates formula / time / rate differences)
        "hybrid"-> IV_own only when the bid/ask spread is <= HYBRID_MAX_SPREAD_PCT of mid, else CBOE IV
    """
    out = df.copy()
    cp = np.where(out["OptionType"] == "call", 1.0, -1.0)
    T = out["TimeToExpiryYears"].to_numpy()
    K = out["Strike"].to_numpy()

    bid, ask = out["Bid"].to_numpy(dtype=float), out["Ask"].to_numpy(dtype=float)
    mid = np.where((bid > 0) & (ask > 0) & (ask >= bid), 0.5 * (bid + ask), np.nan)
    out["Mid"] = mid
    out["IV_own"] = implied_vol(mid, spot, K, T, r, cp)

    iv_c = out["IV_cboe"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread_pct = np.where(np.isfinite(mid) & (mid > 0), (ask - bid) / mid * 100.0, np.nan)
    out["Spread %"] = spread_pct
    if own_iv_basis == "cboe":
        iv_used = iv_c.copy()
        src = np.where(np.isfinite(iv_c), "cboe", "none")
    elif own_iv_basis == "hybrid":
        # own IV only where the quote is tight enough to trust the mid; else CBOE's IV
        tight = np.isfinite(out["IV_own"].to_numpy()) & np.isfinite(spread_pct) & (spread_pct <= HYBRID_MAX_SPREAD_PCT)
        iv_used = np.where(tight, out["IV_own"].to_numpy(), iv_c)
        src = np.where(tight, "own(mid)", np.where(np.isfinite(iv_c), "cboe(wide/none spread)", "none"))
    else:
        iv_used = np.where(np.isfinite(out["IV_own"]), out["IV_own"], iv_c)
        src = np.where(np.isfinite(out["IV_own"]), "own(mid)", np.where(np.isfinite(iv_c), "cboe(fallback)", "none"))
    out["IV_used"] = iv_used
    out["IV_used_source"] = src

    ok = np.isfinite(iv_used) & (iv_used > 0) & (T > 0)
    d = np.full(len(out), np.nan)
    g = np.full(len(out), np.nan)
    if ok.any():
        dd, gg = bs_delta_gamma(spot, K[ok], T[ok], iv_used[ok], r, cp[ok])
        d[ok], g[ok] = dd, gg
    out["Delta_own"], out["Gamma_own"] = d, g

    out["dIV"] = (out["IV_own"] - out["IV_cboe"]) * 100.0          # vol points
    out["dDelta"] = out["Delta_own"] - out["Delta_cboe"]
    out["dGamma"] = out["Gamma_own"] - out["Gamma_cboe"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dGamma_pct"] = np.where(out["Gamma_cboe"] > 0, out["dGamma"] / out["Gamma_cboe"] * 100.0, np.nan)
    return out


def deviation_summary(df: pd.DataFrame) -> dict:
    """Aggregate view of how far self-calculated Greeks sit from CBOE's,
    weighted toward contracts that actually carry open interest."""
    d = df[(df["OpenInterest"] > 0) & df["Gamma_own"].notna() & df["Gamma_cboe"].notna()]
    if d.empty:
        return {"n": 0}
    w = d["OpenInterest"].to_numpy()
    wsum = w.sum()

    def wavg(x):
        x = x.to_numpy(dtype=float)
        m = np.isfinite(x)
        return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.any() else float("nan")

    ivd = d["dIV"].abs().dropna()
    return {
        "n": int(len(d)),
        "median_abs_dIV_pts": float(ivd.median()) if len(ivd) else float("nan"),
        "oi_wtd_dIV_pts": wavg(d["dIV"]),
        "median_abs_dDelta": float(d["dDelta"].abs().median()),
        "median_abs_dGamma_pct": float(d["dGamma_pct"].abs().median()),
        "oi_wtd_gamma_ratio_own_over_cboe": float(np.sum(d["Gamma_own"] * w) / np.sum(d["Gamma_cboe"] * w)),
        "oi_share_covered": float(wsum / max(df["OpenInterest"].sum(), 1.0)),
    }


# ---------------------------------------------------------------------------
# DTE selection
# ---------------------------------------------------------------------------

def expiry_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per expiry: DTE, OI, call/put OI -- so the DTE choice is visible."""
    g = df.groupby("Expiration").agg(
        DTE=("DTE", "first"),
        TotalOI=("OpenInterest", "sum"),
        Contracts=("Strike", "size"),
    ).reset_index()
    g = g[g["DTE"] > 0].sort_values("Expiration").reset_index(drop=True)
    g["DTE_label"] = g["DTE"].map(lambda x: f"{x:.1f}d" if x < 10 else f"{x:.0f}d")
    return g


def select_contracts(df: pd.DataFrame, mode: str, expiry=None, max_dte: Optional[float] = None,
                     spot: Optional[float] = None, strike_band: float = 0.35) -> pd.DataFrame:
    """mode 'single' -> one expiry; 'cumulative' -> every expiry with 0 < DTE <= max_dte."""
    d = df[df["DTE"] > 0]
    if mode == "single":
        d = d[d["Expiration"] == expiry]
    else:
        d = d[d["DTE"] <= float(max_dte)]
    if spot:
        d = d[(d["Strike"] >= spot * (1 - strike_band)) & (d["Strike"] <= spot * (1 + strike_band))]
    return d.copy()


# ---------------------------------------------------------------------------
# GEX under both Greek sets
# ---------------------------------------------------------------------------

def _gex_col(oi, gamma, spot, sign, mult=100):
    return sign * oi * gamma * mult * spot**2 * 0.01


def gex_by_strike_both(sel: pd.DataFrame, spot: float, mult: int = 100) -> pd.DataFrame:
    """Per-strike Call/Put/Net GEX using CBOE gamma and using own gamma, separately."""
    d = sel.copy()
    sign = np.where(d["OptionType"] == "call", 1.0, -1.0)
    d["gex_cboe"] = _gex_col(d["OpenInterest"], d["Gamma_cboe"].fillna(0.0), spot, sign, mult)
    d["gex_own"] = _gex_col(d["OpenInterest"], d["Gamma_own"].fillna(0.0), spot, sign, mult)
    d["is_call"] = d["OptionType"] == "call"
    rows = {}
    for tag in ("cboe", "own"):
        col = f"gex_{tag}"
        rows[f"CallGEX_{tag}"] = d[d["is_call"]].groupby("Strike")[col].sum()
        rows[f"PutGEX_{tag}"] = d[~d["is_call"]].groupby("Strike")[col].sum()
    out = pd.DataFrame(rows).fillna(0.0).sort_index()
    for tag in ("cboe", "own"):
        out[f"NetGEX_{tag}"] = out[f"CallGEX_{tag}"] + out[f"PutGEX_{tag}"]
    oi = d.groupby(["Strike", "OptionType"])["OpenInterest"].sum().unstack(fill_value=0.0)
    out["CallOI"] = oi.get("call", 0.0)
    out["PutOI"] = oi.get("put", 0.0)
    return out.fillna(0.0)


def gamma_flip(sel: pd.DataFrame, spot: float, r: float, iv_col: str,
               rng: float = 0.10, n: int = 201, mult: int = 100) -> Optional[float]:
    """Zero-crossing of aggregate net GEX over a grid of hypothetical spots
    (OI and IV held fixed, gamma re-evaluated with Black-Scholes). Model-based:
    CBOE supplies no flip, so both flips here use OUR formula -- one fed with
    our IV, one with CBOE's IV. Returns the crossing nearest spot, else None."""
    d = sel[(sel["OpenInterest"] > 0) & sel[iv_col].notna() & (sel[iv_col] > 0)]
    if d.empty:
        return None
    K = d["Strike"].to_numpy()[None, :]
    T = d["TimeToExpiryYears"].to_numpy()[None, :]
    iv = d[iv_col].to_numpy()[None, :]
    oi = d["OpenInterest"].to_numpy()[None, :]
    sign = np.where(d["OptionType"].to_numpy() == "call", 1.0, -1.0)[None, :]
    grid = np.linspace(spot * (1 - rng), spot * (1 + rng), n)
    S = grid[:, None]
    _, g = bs_delta_gamma(S, K, T, iv, r, 1.0)
    curve = (sign * oi * g * mult * S**2 * 0.01).sum(axis=1)
    idx = np.where(np.diff(np.sign(curve)) != 0)[0]
    if len(idx) == 0:
        return None
    i = min(idx, key=lambda j: abs(grid[j] - spot))
    x0, x1, y0, y1 = grid[i], grid[i + 1], curve[i], curve[i + 1]
    return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))


def dealer_delta(sel: pd.DataFrame, delta_col: str, mult: int = 100) -> float:
    d = sel[sel[delta_col].notna()]
    return float(-(d["OpenInterest"] * d[delta_col]).sum() * mult)


def levels_from_gex(gx: pd.DataFrame, spot: float, tag: str, top_n: int = 3) -> dict:
    """Spot-aware walls: resistance = largest |net| strikes above spot, support below."""
    net = gx[f"NetGEX_{tag}"]
    above = net[net.index > spot].abs().sort_values(ascending=False).head(top_n)
    below = net[net.index < spot].abs().sort_values(ascending=False).head(top_n)
    pin = float(net.index[np.argmin(np.abs(net.index.to_numpy() - spot))]) if len(net) else None
    return {
        "resistance": [(float(k), float(net.loc[k])) for k in above.index],
        "support": [(float(k), float(net.loc[k])) for k in below.index],
        "atm_pin": pin,
        "call_wall": float(gx[f"CallGEX_{tag}"].idxmax()) if len(gx) else None,
        "put_wall": float(gx[f"PutGEX_{tag}"].idxmin()) if len(gx) else None,
    }


# ---------------------------------------------------------------------------
# Expected range
# ---------------------------------------------------------------------------

def expected_move(spot: float, vol_decimal: float, t_years: float) -> float:
    """1-sigma move = Spot x vol x sqrt(T). vol as a decimal (GVZ 20 -> 0.20)."""
    return float(spot * vol_decimal * np.sqrt(max(t_years, 0.0)))


def atm_iv(sel: pd.DataFrame, spot: float, col: str) -> Optional[float]:
    """Average IV of the two strikes bracketing spot (calls and puts pooled)."""
    d = sel[sel[col].notna() & (sel[col] > 0)]
    if d.empty:
        return None
    strikes = np.sort(d["Strike"].unique())
    below = strikes[strikes <= spot]
    above = strikes[strikes >= spot]
    pick = []
    if len(below):
        pick.append(below[-1])
    if len(above):
        pick.append(above[0])
    v = d[d["Strike"].isin(pick)][col]
    return float(v.mean()) if len(v) else None


def atm_straddle_move(sel: pd.DataFrame, spot: float) -> Optional[dict]:
    """Price-implied 1-sigma move from the ATM straddle: for an at-the-money option,
    call + put premium ~= 0.798 x sigma x sqrt(T) x S, so 1sigma ~= straddle / 0.798.
    Needs no vol or time-to-expiry convention at all -- it is read straight off traded
    prices -- which makes it a useful cross-check on the IV-based rows."""
    d = sel[sel["Mid"].notna() & (sel["Mid"] > 0)]
    if d.empty:
        return None
    strikes = np.sort(d["Strike"].unique())
    k = float(strikes[np.argmin(np.abs(strikes - spot))])
    c = d[(d["Strike"] == k) & (d["OptionType"] == "call")]["Mid"]
    p = d[(d["Strike"] == k) & (d["OptionType"] == "put")]["Mid"]
    if c.empty or p.empty:
        return None
    straddle = float(c.iloc[0] + p.iloc[0])
    return {"strike": k, "straddle": straddle, "one_sigma": straddle / 0.7979}


def range_table(spot: float, t_years: float, gvz: Optional[float], atm_cboe: Optional[float],
                atm_own: Optional[float], straddle_move: Optional[float] = None) -> pd.DataFrame:
    rows = []
    empty = {"Vol %": np.nan, "1σ move": np.nan, "1σ low": np.nan, "1σ high": np.nan,
             "2σ low": np.nan, "2σ high": np.nan}
    for label, vol in (("GVZ (30-day constant maturity)", None if gvz is None else gvz / 100.0),
                       ("ATM IV - CBOE (selected expiry)", atm_cboe),
                       ("ATM IV - own (selected expiry)", atm_own)):
        if vol is None or not np.isfinite(vol):
            rows.append({"Vol source": label, **empty})
            continue
        m = expected_move(spot, vol, t_years)
        rows.append({"Vol source": label, "Vol %": vol * 100, "1σ move": m,
                     "1σ low": spot - m, "1σ high": spot + m,
                     "2σ low": spot - 2 * m, "2σ high": spot + 2 * m})
    label = "ATM straddle ÷ 0.798 (price-implied)"
    if straddle_move is None or not np.isfinite(straddle_move):
        rows.append({"Vol source": label, **empty})
    else:
        m = float(straddle_move)
        rows.append({"Vol source": label, "Vol %": m / (spot * np.sqrt(max(t_years, 1e-12))) * 100,
                     "1σ move": m, "1σ low": spot - m, "1σ high": spot + m,
                     "2σ low": spot - 2 * m, "2σ high": spot + 2 * m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Narratives -- the same readings the original GEX (Options) tab gives, rebuilt
# on this tab's data so both tabs speak the same language. Thresholds and wording
# deliberately mirror gex_engine / streamlit_app.py (parity-tested in tests/).
# ---------------------------------------------------------------------------

NEAR_FLIP_PCT = 0.005   # same 0.5% band as gex_engine.classify_regime


def classify_regime(net_gex: float, spot: float, gamma_flip: Optional[float]) -> str:
    if gamma_flip is not None and abs(spot - gamma_flip) / spot < NEAR_FLIP_PCT:
        return "Near-Flip (Transition Zone)"
    if net_gex > 0:
        return "Positive Gamma (Mean-Reversion Bias)"
    return "Negative Gamma (Continuation Bias)"


_REGIME_MEANING = {
    "Positive": (
        "Under this convention dealers are net LONG gamma around the current price: to stay hedged they "
        "sell into rallies and buy dips, which dampens moves. Expect range / mean-reversion behaviour and "
        "pinning toward large strikes; breakouts tend to fade unless flow overwhelms the hedging."
    ),
    "Negative": (
        "Under this convention dealers are net SHORT gamma around the current price: hedging adds to moves "
        "(buy as price rises, sell as it falls), so moves can extend and accelerate. Expect continuation and "
        "larger intraday swings; support/resistance levels are less reliable than usual."
    ),
    "Near-Flip": (
        "Spot sits within 0.5% of the gamma flip, so a small move can switch the hedging regime. Treat the "
        "label as unstable \u2014 the flip level matters more than the sign of net GEX right now."
    ),
}


def regime_interpretation(label: str, net_gex: float, spot: float, gamma_flip: Optional[float]) -> dict:
    """Plain-language meaning + where spot sits relative to the flip."""
    key = label.split(" ")[0].split("-")[0] if not label.startswith("Near") else "Near-Flip"
    meaning = _REGIME_MEANING.get(key, "")
    flip_text = None
    if gamma_flip is None:
        flip_text = ("No gamma flip found within \u00b110% of spot for this selection \u2014 net GEX keeps one sign "
                     "across the whole tested range, so the regime read is not close to changing on a normal move.")
    else:
        dist = (gamma_flip - spot) / spot * 100.0
        if net_gex > 0 and dist < 0:
            flip_text = (f"Flip at {gamma_flip:,.2f} is {abs(dist):.2f}% BELOW spot: that is the cushion before "
                         f"the regime would turn from positive to negative gamma on a decline.")
        elif net_gex < 0 and dist > 0:
            flip_text = (f"Flip at {gamma_flip:,.2f} is {dist:.2f}% ABOVE spot: price needs to rise that far "
                         f"to get back into positive gamma.")
        else:
            flip_text = (f"Flip at {gamma_flip:,.2f} is {abs(dist):.2f}% {'above' if dist > 0 else 'below'} spot, "
                         f"but net GEX has the opposite sign to what that would imply \u2014 the curve is close to flat "
                         f"around spot, so treat both the flip and the label as low-confidence.")
    return {"label": label, "meaning": meaning, "flip_text": flip_text}


def concentration(gx: pd.DataFrame, tag: str) -> dict:
    """Net vs gross GEX (same 15% / 35% cut-offs and wording as gex_engine)."""
    net_by = gx[f"NetGEX_{tag}"]
    gross = float(net_by.abs().sum())
    net = float(net_by.sum())
    ratio = abs(net) / gross if gross > 0 else 0.0
    if ratio < 0.15:
        text = ("Highly balanced book \u2014 calls and puts largely offset each other. The net regime label is real "
                "but reflects a modest tilt on top of a mostly two-sided position. Low conviction from this layer alone.")
    elif ratio < 0.35:
        text = ("Moderately balanced book \u2014 some net tilt, but a large share of gross exposure is offsetting. "
                "Treat the regime label as a moderate-confidence secondary input, not a dominant signal.")
    else:
        text = ("Lopsided book \u2014 net GEX represents a large share of gross exposure. This is a higher-conviction "
                "directional/regime reading than the typical case.")
    return {"gross_gex": gross, "net_gex": net, "ratio": ratio, "narrative": text}


def dealer_delta_context(dealer_delta_value: float, avg_volume_10d: Optional[float]) -> dict:
    """Dealer delta vs GLD's 10-day average volume (same <10% / 10-40% / >40% wording)."""
    out = {"dealer_delta": dealer_delta_value, "avg_volume_10d": avg_volume_10d, "ratio": None, "size_narrative": None}
    if avg_volume_10d and avg_volume_10d > 0:
        ratio = abs(dealer_delta_value) / avg_volume_10d
        out["ratio"] = ratio
        if ratio < 0.10:
            out["size_narrative"] = ("Small relative to normal trading size \u2014 background noise, unlikely to be a "
                                     "meaningful factor on its own.")
        elif ratio < 0.40:
            out["size_narrative"] = ("Moderate relative to normal trading size \u2014 worth weighting as a secondary "
                                     "confirming/contradicting check against your macro view, not a standalone signal.")
        else:
            out["size_narrative"] = ("Large relative to normal trading size \u2014 this dealer positioning is substantial "
                                     "enough to be a meaningful secondary input.")
    out["direction_narrative"] = (
        "Dealers are net long delta (positive) \u2014 a mild bullish-leaning tilt in the book. Static exposure, not an "
        "active hedging force by itself (that's gamma's job) \u2014 read it as a confirming/contradicting check against "
        "your macro thesis, not a trigger."
        if dealer_delta_value > 0 else
        "Dealers are net short delta (negative) \u2014 a mild bearish-leaning tilt in the book. Static exposure, not an "
        "active hedging force by itself (that's gamma's job) \u2014 read it as a confirming/contradicting check against "
        "your macro thesis, not a trigger."
    )
    return out


def nearest_clusters(gx: pd.DataFrame, spot: float, tag: str, window_pct: float = 0.08,
                     magnitude_frac: float = 0.20, top_n: int = 6) -> pd.DataFrame:
    """Same algorithm as find_nearest_gex_clusters in the app: strikes close to spot that
    ALSO carry a real concentration (>= magnitude_frac of the window's largest gross GEX)."""
    if gx is None or gx.empty:
        return pd.DataFrame()
    w = gx[(gx.index >= spot * (1 - window_pct)) & (gx.index <= spot * (1 + window_pct))].copy()
    if w.empty:
        return pd.DataFrame()
    w["CallGEX"] = w[f"CallGEX_{tag}"]
    w["PutGEX"] = w[f"PutGEX_{tag}"]
    w["GrossGEX"] = w["CallGEX"].abs() + w["PutGEX"].abs()
    cand = w[w["GrossGEX"] >= w["GrossGEX"].max() * magnitude_frac].copy()
    if cand.empty:
        return pd.DataFrame()
    cand["DistFromSpot"] = cand.index - spot
    cand["AbsDistFromSpot"] = cand["DistFromSpot"].abs()
    return cand.sort_values("AbsDistFromSpot").head(top_n)[["CallGEX", "PutGEX", "GrossGEX", "DistFromSpot"]]


def cluster_lean(call_gex: float, put_gex: float, gross_gex: float) -> str:
    net = call_gex + put_gex
    if abs(call_gex) > abs(put_gex) * 1.3:
        return "call-dominant"
    if abs(put_gex) > abs(call_gex) * 1.3:
        return "put-dominant"
    if gross_gex > 0 and abs(net) < gross_gex * 0.3:
        return "two-sided / battleground (large call AND put both, net mostly cancels)"
    return "mixed call/put"


def cluster_side(dist: float) -> str:
    if dist > 0:
        return "above spot (resistance-leaning)"
    if dist < 0:
        return "below spot (support-leaning)"
    return "essentially AT spot (pin risk)"


def clusters_as_records(gx: pd.DataFrame, spot: float, tag: str) -> list:
    c = nearest_clusters(gx, spot, tag)
    out = []
    for strike, row in c.iterrows():
        out.append({
            "strike": float(strike), "dist": float(row["DistFromSpot"]),
            "pct_away": float(row["DistFromSpot"] / spot * 100.0),
            "call_gex": float(row["CallGEX"]), "put_gex": float(row["PutGEX"]),
            "gross_gex": float(row["GrossGEX"]), "net_gex": float(row["CallGEX"] + row["PutGEX"]),
            "side": cluster_side(float(row["DistFromSpot"])),
            "lean": cluster_lean(float(row["CallGEX"]), float(row["PutGEX"]), float(row["GrossGEX"])),
        })
    return out


def raw_walls(gx: pd.DataFrame, tag: str, top_n: int = 3) -> dict:
    """Unfiltered by spot (can surface the ATM strike on both sides) -- diagnostic, like the original."""
    cw = gx[f"CallGEX_{tag}"].sort_values(ascending=False).head(top_n)
    pw = gx[f"PutGEX_{tag}"].abs().sort_values(ascending=False).head(top_n)
    return {"call_walls": [(float(k), float(v)) for k, v in cw.items()],
            "put_walls": [(float(k), float(v)) for k, v in pw.items()]}


def range_context(levels: dict, spot: float, move: Optional[float], move_label: str) -> list:
    """Which walls sit inside today's expected range -- i.e. which are 'in play'."""
    if move is None or not np.isfinite(move) or move <= 0:
        return []
    out = []
    for name, lst in (("resistance", levels.get("resistance", [])), ("support", levels.get("support", []))):
        if not lst:
            out.append(f"No {name} wall found on that side of spot in this selection.")
            continue
        k = lst[0][0]
        dist = abs(k - spot)
        sig = dist / move
        pct = (k - spot) / spot * 100.0
        if sig <= 1.0:
            verdict = f"INSIDE the 1\u03c3 band ({sig:.2f}\u03c3) \u2014 realistically in play for this horizon"
        elif sig <= 2.0:
            verdict = f"between 1\u03c3 and 2\u03c3 ({sig:.2f}\u03c3) \u2014 needs a larger-than-expected move"
        else:
            verdict = f"beyond 2\u03c3 ({sig:.2f}\u03c3) \u2014 unlikely to matter for this horizon"
        out.append(f"Nearest {name} wall {k:,.2f} ({pct:+.2f}% from spot) is {verdict} [band: {move_label}].")
    return out


def build_read(view: dict, spot: float, move: Optional[float], move_label: str) -> list:
    """The 'what this adds up to' list. Each item: {'level': 'good'|'info'|'warn', 'text': str}.
    Heuristic reading aids, not signals -- same stance as the original tab."""
    items = []
    rc, ro = view["regime_cboe"], view["regime_own"]
    if rc["label"] == ro["label"]:
        items.append({"level": "info", "text": f"Both Greek sets agree: **{ro['label']}**. {ro['meaning']}"})
    else:
        items.append({"level": "warn", "text":
                      f"The Greek sets DISAGREE on the regime \u2014 CBOE Greeks: **{rc['label']}**; self-calculated: "
                      f"**{ro['label']}**. Don't lean on the label until that is explained (see the deviation block)."})
        items.append({"level": "info", "text": f"Self-calculated reading: {ro['meaning']}"})
    for who, r_ in (("CBOE-IV", rc), ("own-IV", ro)):
        if r_.get("flip_text"):
            items.append({"level": "info", "text": f"Flip ({who} model): {r_['flip_text']}"})
            break
    conc = view["concentration_own"]
    items.append({"level": "info", "text": f"Book shape: net/gross {conc['ratio']:.0%}. {conc['narrative']}"})
    dd = view["dealer_delta_own"]
    items.append({"level": "info", "text": ("Net dealer delta is " + ("long" if dd > 0 else "short") +
                                            f" ({dd:,.0f} shares) \u2014 static tilt, not a trigger.")})
    for t in range_context(view["levels_own"], spot, move, move_label):
        items.append({"level": "info", "text": t})
    return items


# ---------------------------------------------------------------------------
# Views (one DTE selection -> one self-contained result) and snapshot payload
# ---------------------------------------------------------------------------

def view_key(mode: str, expiry=None, max_dte=None) -> str:
    return f"exp:{expiry}" if mode == "single" else f"cum:{int(max_dte)}"


def compute_view(enriched: pd.DataFrame, spot: float, r: float, mode: str,
                 expiry=None, max_dte: Optional[float] = None) -> Optional[dict]:
    """Everything the tab shows for ONE DTE selection. Returns None if the
    selection has no contracts."""
    sel = select_contracts(enriched, mode, expiry=expiry, max_dte=max_dte, spot=spot)
    sel = sel[sel["OpenInterest"] > 0] if (sel["OpenInterest"] > 0).any() else sel
    if sel.empty:
        return None
    gx = gex_by_strike_both(sel, spot)

    if mode == "single":
        t_years = float(sel["TimeToExpiryYears"].iloc[0])
        dte = float(sel["DTE"].iloc[0])
        label = f"{expiry} ({dte:.1f} DTE)"
        atm_src = sel
    else:
        dte = float(max_dte)
        t_years = max(dte / 365.0, MIN_T_YEARS)
        label = f"All expiries ≤ {int(max_dte)} DTE"
        far = sel["Expiration"].max()
        atm_src = sel[sel["Expiration"] == far]

    out = {
        "key": view_key(mode, expiry, max_dte),
        "label": label,
        "mode": mode,
        "expiry": None if expiry is None else str(expiry),
        "max_dte": None if max_dte is None else float(max_dte),
        "dte": dte,
        "t_years": t_years,
        "n_contracts": int(len(sel)),
        "total_oi": float(sel["OpenInterest"].sum()),
        "gex": gx,
        "net_gex_own": float(gx["NetGEX_own"].sum()),
        "net_gex_cboe": float(gx["NetGEX_cboe"].sum()),
        "flip_own_iv": gamma_flip(sel, spot, r, "IV_used"),
        "flip_cboe_iv": gamma_flip(sel, spot, r, "IV_cboe"),
        "dealer_delta_own": dealer_delta(sel, "Delta_own"),
        "dealer_delta_cboe": dealer_delta(sel, "Delta_cboe"),
        "levels_own": levels_from_gex(gx, spot, "own"),
        "levels_cboe": levels_from_gex(gx, spot, "cboe"),
        "atm_iv_cboe": atm_iv(atm_src, spot, "IV_cboe"),
        "atm_iv_own": atm_iv(atm_src, spot, "IV_own"),
        "deviation": deviation_summary(sel),
        "straddle": atm_straddle_move(atm_src, spot),
        "selection_df": sel,   # not serialised
    }
    fo, fc = out["flip_own_iv"], out["flip_cboe_iv"]
    out["regime_own"] = regime_interpretation(classify_regime(out["net_gex_own"], spot, fo), out["net_gex_own"], spot, fo)
    out["regime_cboe"] = regime_interpretation(classify_regime(out["net_gex_cboe"], spot, fc), out["net_gex_cboe"], spot, fc)
    out["concentration_own"] = concentration(gx, "own")
    out["concentration_cboe"] = concentration(gx, "cboe")
    out["clusters_own"] = clusters_as_records(gx, spot, "own")
    out["clusters_cboe"] = clusters_as_records(gx, spot, "cboe")
    out["raw_walls_own"] = raw_walls(gx, "own")
    out["raw_walls_cboe"] = raw_walls(gx, "cboe")
    return out


def standard_view_specs(exp_table: pd.DataFrame, n_single: int = 6, cum_days=(7, 14, 30)) -> list[dict]:
    specs = [{"mode": "single", "expiry": row.Expiration} for row in exp_table.head(n_single).itertuples()]
    max_avail = float(exp_table["DTE"].max()) if len(exp_table) else 0.0
    for c in cum_days:
        if max_avail >= min(c, 1):
            specs.append({"mode": "cumulative", "max_dte": c})
    return specs


def fetch_avg_volume_10d(symbol: str = "GLD") -> Optional[float]:
    """10-day average share volume (yfinance) for the dealer-delta-vs-volume narrative."""
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period="10d")
        return float(h["Volume"].mean()) if not h.empty else None
    except Exception:
        return None


def build_snapshot_payload(symbol: str, spot: float, meta: dict, enriched: pd.DataFrame, r: float,
                           r_note: str, gvz: Optional[float], own_iv_basis: str,
                           avg_volume: Optional[float] = None, basis: Optional[dict] = None) -> dict:
    """Compact, JSON-safe record of the standard views -- what gets stored as
    the 'last good' fallback for this tab."""
    import gex_snapshot as gs
    exp_table = expiry_table(enriched)
    views = {}
    for spec in standard_view_specs(exp_table):
        v = compute_view(enriched, spot, r, **spec)
        if v is None:
            continue
        v = dict(v)
        v.pop("selection_df", None)
        v["gex"] = gs.df_to_json(v["gex"])
        views[v["key"]] = v
    return {
        "kind": "alt",
        "symbol": symbol,
        "spot": float(spot),
        "asof_cboe_utc": None if meta.get("asof_utc") is None else meta["asof_utc"].isoformat(),
        "asof_cboe_raw": meta.get("asof_raw"),
        "gvz": gvz,
        "cboe_iv30": meta.get("iv30"),
        "r": r,
        "r_note": r_note,
        "own_iv_basis": own_iv_basis,
        "source_url": meta.get("url"),
        "avg_volume_10d": avg_volume,
        "basis_at_save": basis,
        "views": views,
    }
