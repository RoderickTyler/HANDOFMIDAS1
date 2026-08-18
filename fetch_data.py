"""
fetch_data.py
-------------
Pulls all the raw data needed for the daily gold macro view, from entirely
free sources:

  - yfinance          -> gold, DXY, VIX          (no key needed)
  - FRED (fredapi)    -> yields, real yields      (free key needed)
  - IMF SDMX API      -> central bank reserves    (no key needed)
  - Iacoviello GPR    -> geopolitical risk index  (no key needed)

Every function returns a pandas DataFrame or Series so analysis.py can work
with clean, typed data rather than re-parsing raw responses everywhere.
"""

import io
import sys
from datetime import datetime, timedelta
import pandas as pd
import requests
import yfinance as yf

import config

try:
    from fredapi import Fred
except ImportError:
    Fred = None

try:
    import imfp
except ImportError:
    imfp = None


# ---------------------------------------------------------------------------
# Market data (yfinance)
# ---------------------------------------------------------------------------

def period_to_start_date(period: str) -> str:
    """
    Converts a yfinance-style period string ('1y', '6mo', '2y', etc.) into
    an ISO start date, so FRED's yield data can be fetched starting from
    the SAME point market data starts -- instead of the two sources having
    mismatched lookback windows and leaving early rows in the cache with
    real yields but blank gold/DXY/VIX (or vice versa).
    """
    today = datetime.now()
    period = period.lower().strip()

    if period == "max":
        return "2000-01-01"  # generous floor; FRED/yfinance will just return what they have
    if period == "ytd":
        return datetime(today.year, 1, 1).strftime("%Y-%m-%d")

    try:
        num = int("".join(ch for ch in period if ch.isdigit()))
        unit = "".join(ch for ch in period if ch.isalpha())
    except ValueError:
        return "2020-01-01"  # fallback to the old default if parsing fails

    if unit in ("d", "day", "days"):
        delta = timedelta(days=num)
    elif unit in ("mo", "month", "months"):
        delta = timedelta(days=num * 31)
    elif unit in ("y", "year", "years"):
        delta = timedelta(days=num * 366)
    else:
        return "2020-01-01"

    return (today - delta).strftime("%Y-%m-%d")


def get_market_data(period="1y", interval="1d"):
    """
    Fetch gold, DXY, VIX daily history.
    Returns a single DataFrame indexed by date with columns:
    gold_spot, dxy, vix
    """
    frames = {}
    for name, ticker in config.YF_TICKERS.items():
        if name == "us10y_yield_quote":
            continue  # cross-check only, handled separately if needed
        try:
            hist = yf.Ticker(ticker).history(period=period, interval=interval)
            if hist.empty:
                print(f"[warn] No data returned for {name} ({ticker})")
                continue
            frames[name] = hist["Close"]
        except Exception as e:
            print(f"[warn] Failed to fetch {name} ({ticker}): {e}")

    if not frames:
        raise RuntimeError("No market data could be fetched — check internet connection.")

    df = pd.DataFrame(frames)
    df.index = df.index.tz_localize(None)  # drop tz for easy merging with FRED
    df.index = df.index.normalize()  # zero out time-of-day (e.g. 05:00:00 -> 00:00:00)
    # so a trading day here aligns with FRED's date-only index instead of
    # creating two separate rows for "the same day"
    df = df.groupby(df.index).last()  # in case normalizing created any duplicate index entries
    return df


def get_xauusd_spot_history(period="1y", interval="1d"):
    """
    TRUE spot gold (XAU/USD) daily history, via Yahoo Finance's "XAUUSD=X"
    ticker -- this is genuinely different from get_market_data()'s
    "gold_spot" column, which (despite the name) is actually GC=F, the
    COMEX gold FUTURES front-month contract. Futures and spot track each
    other closely but are not identical (contango/backwardation, roll
    effects), so this exists purely to give an honest, correctly-labeled
    real-spot series for display -- it does NOT feed into any of the
    existing signals/correlations/regime model, which all deliberately
    continue to use GC=F as before.

    Returns a pandas Series of Close prices indexed by date (tz-naive,
    normalized to match the rest of the pipeline), or an empty Series if
    the fetch fails for any reason -- callers should treat that as "fall
    back to futures for display" rather than a hard error.
    """
    try:
        hist = yf.Ticker("XAUUSD=X").history(period=period, interval=interval)
        if hist.empty:
            print("[warn] No data returned for XAUUSD=X (spot gold) -- yfinance may not carry this ticker right now.")
            return pd.Series(dtype=float, name="xauusd_spot")
        series = hist["Close"]
        series.index = series.index.tz_localize(None).normalize()
        series = series.groupby(series.index).last()
        series.name = "xauusd_spot"
        return series
    except Exception as e:
        print(f"[warn] Failed to fetch XAUUSD=X (spot gold) history: {e}")
        return pd.Series(dtype=float, name="xauusd_spot")


# ---------------------------------------------------------------------------
# Rates / real yields (FRED)
# ---------------------------------------------------------------------------

def get_fred_data(start="2020-01-01"):
    """
    Fetch nominal yields, real (TIPS) yield, curve spread, and inflation
    expectations from FRED. Requires a free API key in .env (FRED_API_KEY).
    Returns a DataFrame indexed by date.
    """
    if not config.FRED_API_KEY:
        print(
            "[warn] No FRED_API_KEY set. Skipping yield data.\n"
            "       Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "       Then add it to a .env file as FRED_API_KEY=your_key_here"
        )
        return pd.DataFrame()

    if Fred is None:
        print("[warn] fredapi not installed. Run: pip install fredapi")
        return pd.DataFrame()

    fred = Fred(api_key=config.FRED_API_KEY)
    series = {}
    for name, series_id in config.FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start)
            series[name] = s
        except Exception as e:
            print(f"[warn] Failed to fetch FRED series {series_id}: {e}")

    if not series:
        return pd.DataFrame()

    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Geopolitical Risk Index (Caldara & Iacoviello, free .xls download)
# ---------------------------------------------------------------------------

def get_gpr_index():
    """
    Downloads the monthly Geopolitical Risk Index (GPR) built by Fed
    economists Dario Caldara and Matteo Iacoviello. Free, no key.
    Returns a DataFrame with columns: gpr, gpr_threat, gpr_act (if available).
    """
    try:
        resp = requests.get(config.GPR_DATA_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content))
        # The published file's exact column names shift occasionally;
        # normalize what we can find.
        df.columns = [c.strip().upper() for c in df.columns]
        date_col = next((c for c in df.columns if "DATE" in c or "MONTH" in c), None)
        if date_col:
            df = df.assign(date=pd.to_datetime(df[date_col])).set_index("date").copy()
        return df
    except Exception as e:
        print(f"[warn] Failed to fetch GPR index: {e}")
        print("       Manual fallback: https://www.matteoiacoviello.com/gpr.htm")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Central bank gold reserves (IMF, via the `imfp` package)
# ---------------------------------------------------------------------------
# IMF fully decommissioned dataservices.imf.org in Nov 2025 in favor of a new
# SDMX 3.0 API at api.imf.org with a different URL structure. Rather than
# hand-track IMF's endpoint changes ourselves, we use `imfp`
# (https://pypi.org/project/imfp/), a maintained package built specifically
# to track IMF's current API. If IMF changes things again, updating imfp
# (`pip install --upgrade imfp`) is the fix, not editing this file.

def _select_gold_indicator(gold_rows: pd.DataFrame, prefer: str):
    """
    Picks a gold indicator row from IRFCL's indicator list.

    prefer='usd': the USD-value series (what get_imf_gold_reserves uses --
    this is a MARK-TO-MARKET dollar value: it moves both when a country's
    actual gold holdings change AND whenever gold's price itself moves,
    even with zero buying/selling. A month-over-month "decline" in this
    series does NOT by itself mean a country sold gold -- it can be pure
    price effect.

    prefer='volume': the PHYSICAL QUANTITY series (troy ounces / tonnes --
    whatever unit IMF reports), which only changes when a country actually
    adds or removes physical gold. This is the series that answers "did
    they actually buy or sell," independent of price.
    """
    def _is_usd(row):
        code, desc = str(row["input_code"]).upper(), str(row["description"]).upper()
        return "USD" in code or "USD" in desc

    def _is_other_currency(row):
        code, desc = str(row["input_code"]).upper(), str(row["description"]).upper()
        return any(cur in code or cur in desc for cur in ("EUR", "GBP", "JPY", "SDR"))

    def _is_volume(row):
        code, desc = str(row["input_code"]).upper(), str(row["description"]).upper()
        return any(term in desc for term in ("OUNCE", "TROY", "FINE GOLD", "METRIC TON", "TONNE", "KILOGRAM"))

    if prefer == "volume":
        volume_rows = gold_rows[gold_rows.apply(_is_volume, axis=1)]
        if not volume_rows.empty:
            return volume_rows.iloc[0]
        return None  # no volume series found; caller should handle gracefully

    # prefer == 'usd' (existing behavior, unchanged)
    usd_rows = gold_rows[gold_rows.apply(_is_usd, axis=1)]
    if not usd_rows.empty:
        return usd_rows.iloc[0]
    non_currency_rows = gold_rows[~gold_rows.apply(_is_other_currency, axis=1) & ~gold_rows.apply(_is_volume, axis=1)]
    return non_currency_rows.iloc[0] if not non_currency_rows.empty else gold_rows.iloc[0]


def _fetch_irfcl_gold_series(prefer: str, countries=None):
    """
    Shared fetch logic for both the USD-value and physical-quantity gold
    series -- discovers indicator/area/freq/sector parameters from IMF's
    own metadata (not hardcoded), fetches, and attaches sector descriptions.
    `prefer` is 'usd' or 'volume', see _select_gold_indicator() above.
    """
    countries = countries or config.IMF_COUNTRIES

    if imfp is None:
        print(
            "[warn] imfp package not installed. Run: pip install imfp\n"
            "       Manual fallback: https://www.gold.org/goldhub/data/gold-reserves-by-country"
        )
        return pd.DataFrame(), None

    try:
        # imfp renamed this function from imf_app_name -> set_imf_app_name when
        # it rewrote itself for IMF's new SDMX 3.0 API (the old one no longer
        # exists). Try the current name first, fall back to the old one for
        # anyone pinned to an older imfp version.
        if hasattr(imfp, "set_imf_app_name"):
            imfp.set_imf_app_name("gold_macro_daily_briefing")
        elif hasattr(imfp, "imf_app_name"):
            imfp.imf_app_name("gold_macro_daily_briefing")
    except Exception:
        pass

    try:
        params = imfp.imf_parameters("IRFCL")
    except Exception as e:
        print(f"[warn] Could not fetch IRFCL parameters from IMF: {e}")
        return pd.DataFrame(), None

    indicator_key = next((k for k in params if "indicator" in k.lower()), None)
    area_key = next((k for k in params if any(t in k.lower() for t in ("area", "country", "ref_area"))), None)
    freq_key = next((k for k in params if "freq" in k.lower()), None)

    if not indicator_key:
        print("[warn] Could not locate an indicator parameter in IRFCL's structure.")
        return pd.DataFrame(), None

    ind_df = params[indicator_key]
    gold_rows = ind_df[ind_df["description"].str.contains("gold", case=False, na=False)]
    if gold_rows.empty:
        print("[warn] No gold-related indicator found in IRFCL parameters.")
        return pd.DataFrame(), None

    chosen = _select_gold_indicator(gold_rows, prefer=prefer)
    if chosen is None:
        print(f"[warn] No '{prefer}' gold indicator found among IRFCL's gold-related series.")
        return pd.DataFrame(), None

    print(f"[info] Using IMF gold indicator ({prefer}): {chosen['input_code']} ({chosen['description']})")
    unit_description = chosen["description"]

    query_params = {indicator_key: ind_df[ind_df["input_code"] == chosen["input_code"]]}

    if area_key:
        area_df = params[area_key]
        country_names = config.IMF_COUNTRY_NAMES
        name_pattern = "|".join(country_names.values())
        matched_areas = area_df[area_df["description"].str.contains(name_pattern, case=False, na=False, regex=True)]
        if matched_areas.empty:
            matched_areas = area_df[area_df["input_code"].isin(countries)]
        if not matched_areas.empty:
            query_params[area_key] = matched_areas
        else:
            print(f"[warn] Could not match countries against IMF's area codes; fetching all areas.")

    if freq_key:
        freq_df = params[freq_key]
        monthly = freq_df[freq_df["input_code"] == "M"]
        query_params[freq_key] = monthly if not monthly.empty else freq_df.iloc[[0]]

    sector_key = next((k for k in params if "sector" in k.lower()), None)
    sector_lookup = {}
    if sector_key:
        sector_df = params[sector_key]
        sector_lookup = dict(zip(sector_df["input_code"], sector_df["description"]))

    try:
        # imfp doesn't expose its own timeout parameter, so we bound the
        # underlying network call at the socket level instead -- this is
        # global for the process, so we carefully reset it right after,
        # even if the call fails, to avoid affecting other fetches
        # (yfinance, FRED, GPR) that run elsewhere in this system.
        import socket
        IMF_TIMEOUT_SECONDS = 45
        original_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(IMF_TIMEOUT_SECONDS)
            df = imfp.imf_dataset(database_id="IRFCL", parameters=query_params, start_year=2015)
        finally:
            socket.setdefaulttimeout(original_timeout)
    except Exception as e:
        print(f"[warn] IMF dataset request failed or timed out after {IMF_TIMEOUT_SECONDS}s: {e}")
        print("       This can happen if IMF's server is slow for this particular series --")
        print("       safe to retry later. Manual fallback: "
              "https://www.gold.org/goldhub/data/gold-reserves-by-country")
        return pd.DataFrame(), None

    if df is None or df.empty:
        print("[warn] IMF reserve data unavailable this run.")
        return pd.DataFrame(), None

    if sector_lookup:
        sector_col = next((c for c in df.columns if "sector" in c.lower()), None)
        if sector_col:
            df["sector_description"] = df[sector_col].map(sector_lookup).fillna("Unknown")

    return df, unit_description


def get_imf_gold_reserves(countries=None):
    """
    Pulls official gold reserve holdings in USD VALUE (mark-to-market) from
    the IMF's IRFCL dataset. IMPORTANT: this series moves with BOTH actual
    buying/selling AND gold's own price -- a month-over-month change here
    conflates the two. For physical quantity only (immune to price
    effects), use get_imf_gold_reserves_volume() instead.

    Note: if this fails, the manual fallback is the World Gold Council's
    quarterly report: https://www.gold.org/goldhub/data/gold-reserves-by-country
    """
    df, _ = _fetch_irfcl_gold_series(prefer="usd", countries=countries)
    return df


def get_imf_gold_reserves_volume(countries=None):
    """
    Pulls official gold reserve holdings in PHYSICAL QUANTITY (troy ounces
    or whatever unit IMF reports -- check the printed [info] line each run
    for the exact unit, since IMF's own wording is the source of truth).

    This series only changes when a country actually adds/removes physical
    gold -- it is NOT affected by gold's price moving. This is the correct
    series to answer "did they actually buy or sell," as opposed to the
    USD-value series in get_imf_gold_reserves() which conflates real
    buying/selling with pure price appreciation/depreciation.

    Returns (DataFrame, unit_description) -- the unit_description tells you
    exactly what unit IMF reported this in (e.g. "millions of fine troy
    ounces"), since guessing the unit wrong would be worse than not showing
    this at all.
    """
    return _fetch_irfcl_gold_series(prefer="volume", countries=countries)


# ---------------------------------------------------------------------------
# Convenience: pull everything at once
# ---------------------------------------------------------------------------

def fetch_all(period="1y", quick=False):
    """
    quick=True skips the Geopolitical Risk Index and IMF central bank
    reserves fetches (both USD-value and physical-quantity) -- these are
    consistently the slowest calls (GPR is an Excel file fetch from an
    external researcher's site; IMF's SDMX API can be flaky/slow even when
    it works) and aren't needed for most day-to-day dashboard views.
    Returns empty DataFrames for the skipped keys, which every downstream
    consumer already handles gracefully (the same shape as "this fetch
    failed on a normal, non-quick run" -- an already-handled case).
    """
    print("Fetching market data (gold, DXY, VIX)...")
    market = get_market_data(period=period)

    fred_start = period_to_start_date(period)
    print(f"Fetching Treasury yield data from FRED (from {fred_start}, matching market data lookback)...")
    fred = get_fred_data(start=fred_start)

    if quick:
        print("[quick load] Skipping Geopolitical Risk Index and IMF central bank reserves.")
        gpr = pd.DataFrame()
        reserves = pd.DataFrame()
        reserves_volume = pd.DataFrame()
        reserves_volume_unit = None
    else:
        print("Fetching Geopolitical Risk Index...")
        gpr = get_gpr_index()

        print("Fetching central bank gold reserves from IMF (USD value)...")
        reserves = get_imf_gold_reserves()

        print("Fetching central bank gold reserves from IMF (physical quantity -- to separate real "
              "buying/selling from pure price effect)...")
        reserves_volume, reserves_volume_unit = get_imf_gold_reserves_volume()

    return {
        "market": market,
        "fred": fred,
        "gpr": gpr,
        "reserves": reserves,
        "reserves_volume": reserves_volume,
        "reserves_volume_unit": reserves_volume_unit,
    }


if __name__ == "__main__":
    data = fetch_all()
    for name, df in data.items():
        print(f"\n=== {name} ===")
        print(df.tail() if not df.empty else "(no data)")
