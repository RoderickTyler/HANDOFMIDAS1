"""
config.py
---------
Central place for API keys, tickers, and tunable thresholds.

FRED_API_KEY: Get a FREE key at https://fred.stlouisfed.org/docs/api/api_key.html
(takes ~2 minutes, no cost, no credit card). Put it in a .env file next to this
script as:

    FRED_API_KEY=your_key_here

Everything else (yfinance, IMF SDMX) needs no key at all.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present (local / CLI use)


def _get_secret(name: str, default: str = "") -> str:
    """
    Resolve a secret from, in order: environment variable / .env (local &
    most hosts), then Streamlit's secrets manager (st.secrets) if running
    inside Streamlit and the key was configured there -- this is how
    Streamlit Community Cloud injects secrets (Settings -> Secrets), since
    that platform doesn't support a checked-in .env file.
    """
    val = os.getenv(name, "")
    if val:
        return val
    try:
        import streamlit as st  # only import if actually available
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return default


FRED_API_KEY = _get_secret("FRED_API_KEY")

# Real spot XAUUSD (for comparison against the GC=F futures price the rest
# of this system uses) comes from gold-api.com, which needs no key at all
# -- see spot_gold.py. No config needed here anymore.

# ---- yfinance tickers (no key needed) ----
YF_TICKERS = {
    "gold_spot": "GC=F",      # COMEX gold futures (front month) - good proxy for spot
    "dxy": "DX-Y.NYB",        # US Dollar Index
    "vix": "^VIX",            # CBOE Volatility Index
    "us10y_yield_quote": "^TNX",  # quoted as yield*10, backup/cross-check vs FRED
}

# ---- FRED series IDs (need free API key) ----
FRED_SERIES = {
    "dgs2": "DGS2",        # 2-Year Treasury Constant Maturity Rate (nominal)
    "dgs10": "DGS10",      # 10-Year Treasury Constant Maturity Rate (nominal)
    "dfii10": "DFII10",    # 10-Year Treasury Inflation-Indexed (real) Yield
    "t10y2y": "T10Y2Y",    # 10Y minus 2Y spread, precomputed by FRED
    "t5yifr": "T5YIFR",    # 5yr, 5yr forward inflation expectation rate
}

# ---- IMF gold reserves (via the `imfp` package, no key needed) ----
# imfp tracks IMF's current API internally; if IMF changes endpoints again,
# `pip install --upgrade imfp` is the fix rather than editing this file.
# IMF's ref_area codes aren't always plain ISO2, so we match by country
# NAME against IMF's own description text instead of assuming a code format.
IMF_COUNTRY_NAMES = {
    "CN": "China",
    "IN": "India",
    "TR": "Turkey",
    "RU": "Russia",
    "PL": "Poland",
    "US": "United States",
    "DE": "Germany",
}
IMF_COUNTRIES = list(IMF_COUNTRY_NAMES.keys())  # ISO2 codes, kept as a fallback match

# ---- Geopolitical Risk Index (Caldara & Iacoviello, free, updated monthly) ----
GPR_DATA_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"

# ---- Analysis thresholds ----
ROLLING_WINDOW_SHORT = 30   # trading days, ~6 weeks
ROLLING_WINDOW_LONG = 90    # trading days, ~1 quarter
CORR_DIVERGENCE_THRESHOLD = 0.2  # if rolling corr crosses this far from expected sign

# ---- File paths ----
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRICE_CACHE = os.path.join(DATA_DIR, "price_history.csv")
JOURNAL_FILE = os.path.join(DATA_DIR, "thesis_journal.csv")
