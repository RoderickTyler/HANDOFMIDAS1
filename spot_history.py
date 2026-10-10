"""
spot_history.py
----------------
TRUE spot-gold (XAUUSD) daily history for the Overview "Price history" chart, so the
chart shows spot instead of silently falling back to the GC=F futures contract.

Why a chain: Yahoo's spot tickers (XAUUSD=X / XAU=X) are flaky or missing on some
days, which is what pushed the old chart onto GC=F. This tries several free sources
in order and uses the first one that passes validation:

    1. yfinance  XAUUSD=X, then XAU=X
    2. Dukascopy (dukascopy-python, instrument XAU/USD, daily, BID)   [needs: pip install dukascopy-python]
    3. Stooq     xauusd daily CSV (keyless; may require an api key now -- reported if so)

VALIDATION (a wrong series is worse than none): a candidate is accepted only if, against
the GC=F reference over their overlap, it has
    - at least MIN_ROWS rows,
    - daily-return correlation >= MIN_CORR (spot and futures should move almost 1:1),
    - median price ratio within +/-MAX_REL_GAP of 1 (the real futures-spot basis is ~1-1.5%).
Because sources stamp daily bars differently (Dukascopy bars open ~21:00-22:00 UTC the
evening before), each candidate is also tried shifted by -1/0/+1 day and the best-
correlating alignment is kept; the shift used is reported.

The last point is optionally replaced/extended with the live pulled spot (Mon-Fri), so the
chart ends at "now" rather than yesterday's close.

Nothing here feeds the regime model, factors or any signal -- display only.
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

MIN_ROWS = 20
MIN_CORR = 0.90
MAX_REL_GAP = 0.06
STOOQ_URL = "https://stooq.com/q/d/l/?s=xauusd&i=d&d1={d1}&d2={d2}"


def period_to_days(period: str) -> int:
    p = (period or "1y").strip().lower()
    if p in ("max", "all"):
        return 3653
    m = re.fullmatch(r"(\d+)\s*(d|wk|w|mo|m|y)", p)
    if not m:
        return 366
    n, u = int(m.group(1)), m.group(2)
    return int(n * {"d": 1, "w": 7, "wk": 7, "mo": 31, "m": 31, "y": 366}[u]) + 5


def _daily(series: pd.Series, name="xauusd_spot") -> pd.Series:
    s = series.dropna().astype(float)
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize() if s.index.tz is None else s.index.tz_convert(None).normalize()
    s = s.groupby(s.index).last().sort_index()
    s.name = name
    return s


# ---------------------------------------------------------------------------
# Fetchers: each returns a daily Series (tz-naive date index) or raises
# ---------------------------------------------------------------------------

def fetch_yfinance(start: datetime, end: datetime) -> pd.Series:
    import yfinance as yf
    last_err = None
    for tk in ("XAUUSD=X", "XAU=X"):
        try:
            h = yf.Ticker(tk).history(start=start.date().isoformat(), end=(end + timedelta(days=1)).date().isoformat(), interval="1d")
            if h is not None and not h.empty:
                return _daily(h["Close"])
            last_err = f"{tk}: empty"
        except Exception as e:
            last_err = f"{tk}: {e}"
    raise RuntimeError(last_err or "no data")


def fetch_dukascopy(start: datetime, end: datetime) -> pd.Series:
    import dukascopy_python as dk
    from dukascopy_python import instruments as ins
    df = dk.fetch(ins.INSTRUMENT_FX_METALS_XAU_USD, dk.INTERVAL_DAY_1, dk.OFFER_SIDE_BID, start, end)
    if df is None or df.empty or "close" not in df.columns:
        raise RuntimeError("empty Dukascopy response")
    s = df["close"].copy()
    # Bars are stamped at their OPEN in UTC (often 21:00/22:00 the previous evening). +12h puts every
    # plausible convention on the calendar day the bar mostly covers; the alignment check fine-tunes it.
    idx = pd.DatetimeIndex(s.index)
    idx = (idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")) + pd.Timedelta(hours=12)
    s.index = idx.tz_convert(None)
    return _daily(s)


def fetch_stooq(start: datetime, end: datetime) -> pd.Series:
    import requests
    url = STOOQ_URL.format(d1=start.strftime("%Y%m%d"), d2=end.strftime("%Y%m%d"))
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    txt = r.text.strip()
    if not txt.lower().startswith("date"):
        raise RuntimeError(f"unexpected Stooq response: {txt[:80]!r}")
    df = pd.read_csv(io.StringIO(txt), parse_dates=["Date"]).set_index("Date")
    return _daily(df["Close"])


DEFAULT_FETCHERS = [("yfinance XAUUSD=X", fetch_yfinance), ("Dukascopy XAU/USD", fetch_dukascopy), ("Stooq XAUUSD", fetch_stooq)]


# ---------------------------------------------------------------------------
# Validation / alignment
# ---------------------------------------------------------------------------

def validate_against_reference(series: pd.Series, reference: Optional[pd.Series]) -> dict:
    """Returns {ok, reason, shift_days, corr, ratio, aligned (Series)}. With no usable reference
    we cannot validate, so the result says so and ok=True only when `allow_unvalidated` upstream."""
    out = {"ok": False, "reason": "", "shift_days": 0, "corr": None, "ratio": None, "aligned": series, "validated": False}
    if series is None or len(series) < MIN_ROWS:
        out["reason"] = f"only {0 if series is None else len(series)} rows (< {MIN_ROWS})"
        return out
    if (series <= 0).any():
        out["reason"] = "non-positive prices"
        return out
    if reference is None or len(reference.dropna()) < MIN_ROWS:
        out.update(ok=True, reason="no reference available to validate against", validated=False)
        return out
    ref = _daily(reference, "ref")
    best = None
    for shift in (0, -1, 1):
        s2 = series.copy()
        s2.index = s2.index + pd.Timedelta(days=shift)
        j = pd.concat([s2.rename("s"), ref], axis=1, join="inner").dropna()
        if len(j) < MIN_ROWS:
            continue
        c = j.pct_change().dropna().corr().iloc[0, 1]
        if np.isfinite(c) and (best is None or c > best[0]):
            best = (c, shift, j, s2)
    if best is None:
        out["reason"] = "too little overlap with the reference"
        return out
    corr, shift, j, s2 = best
    ratio = float((j["s"] / j["ref"]).median())
    out.update(corr=float(corr), shift_days=shift, ratio=ratio, aligned=s2, validated=True)
    if corr < MIN_CORR:
        out["reason"] = f"daily-return correlation with GC=F only {corr:.2f} (< {MIN_CORR})"
    elif abs(ratio - 1) > MAX_REL_GAP:
        out["reason"] = f"median price ratio to GC=F is {ratio:.3f} (outside ±{MAX_REL_GAP:.0%})"
    else:
        out["ok"] = True
    return out


def apply_live_point(series: pd.Series, live_price: Optional[float], now_utc: Optional[datetime] = None) -> tuple[pd.Series, bool]:
    """Replace/extend the final point with the live pulled spot (Mon-Fri only: on weekends the
    pulled price is just Friday's close and would add a bogus bar)."""
    if live_price is None or not np.isfinite(live_price) or live_price <= 0:
        return series, False
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.weekday() >= 5:
        return series, False
    day = pd.Timestamp(now_utc.date())
    s = series.copy()
    if day < s.index.max():
        return s, False
    s.loc[day] = float(live_price)
    return s.sort_index(), True


def get_spot_history(period: str = "1y", reference: Optional[pd.Series] = None,
                     live_price: Optional[float] = None, fetchers=None,
                     now_utc: Optional[datetime] = None, allow_unvalidated: bool = False) -> dict:
    """Returns {series, source, notes, shift_days, corr, ratio, live_point}. series is None when no
    source passed validation (callers then fall back to GC=F WITH a visible label)."""
    now_utc = now_utc or datetime.now(timezone.utc)
    end = now_utc.replace(tzinfo=None)
    start = end - timedelta(days=period_to_days(period))
    notes = []
    for name, fn in (fetchers or DEFAULT_FETCHERS):
        try:
            raw = fn(start, end)
        except Exception as e:
            notes.append(f"{name}: failed ({type(e).__name__}: {str(e)[:120]})")
            continue
        v = validate_against_reference(raw, reference)
        if not v["ok"] or (not v["validated"] and not allow_unvalidated):
            notes.append(f"{name}: rejected — {v['reason'] or 'unvalidated'}")
            continue
        series, live = apply_live_point(v["aligned"], live_price, now_utc)
        if v["shift_days"]:
            notes.append(f"{name}: dates shifted {v['shift_days']:+d} day to align with GC=F")
        return {"series": series, "source": name, "notes": notes, "shift_days": v["shift_days"],
                "corr": v["corr"], "ratio": v["ratio"], "live_point": live}
    return {"series": None, "source": None, "notes": notes, "shift_days": 0, "corr": None, "ratio": None, "live_point": False}
