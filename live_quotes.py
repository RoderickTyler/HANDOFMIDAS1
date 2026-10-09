"""
live_quotes.py
---------------
INDEPENDENT MODULE. Live quote view for VIX / DXY / Gold / EURUSD / GBPUSD
/ USDCHF / USDJPY, toggleable between two free data sources:

    --feed yfinance   (default -- same source the rest of this system uses)
    --feed dukascopy  (Dukascopy Bank SA's free data feed)

This does NOT touch the main daily briefing pipeline (regime model,
factor attribution, etc.) -- it's a standalone comparison view, so
switching feeds here can never destabilize anything already validated
elsewhere in the system.

COVERAGE, CONFIRMED BY INSPECTING THE ACTUAL PACKAGE (not assumed):
    - EURUSD, GBPUSD, USDCHF, USDJPY, Gold: all genuinely available via
      Dukascopy (Gold is REAL spot XAUUSD via Dukascopy, vs. the GC=F
      futures the rest of this system uses -- a real upgrade for this
      one instrument specifically, if you want it).
    - VIX: NOT available via Dukascopy at all (confirmed by listing every
      instrument constant in the package -- none match). When
      --feed dukascopy is selected, VIX always falls back to yfinance
      with an explicit note, since there's no Dukascopy alternative.
    - DXY: Dukascopy has a dollar-index instrument, but it's Dukascopy's
      OWN construction -- not verified to be identical to the official
      ICE DXY (DX-Y.NYB) the rest of this system uses. Treat it as "a"
      dollar index for comparison, not a guaranteed match.
"""

from datetime import datetime, timedelta

import yfinance as yf

try:
    import dukascopy_python
    from dukascopy_python import instruments as dk_instruments
except ImportError:
    dukascopy_python = None
    dk_instruments = None

YFINANCE_TICKERS = {
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "Gold": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDCHF": "USDCHF=X",
    "USDJPY": "USDJPY=X",
}

# (instrument constant name, caveat note or None) -- VIX deliberately absent
DUKASCOPY_INSTRUMENTS = {
    "DXY": ("INSTRUMENT_IDX_AMERICA_DOLLAR_IDX_USD",
            "Dukascopy's own dollar-index construction -- NOT verified identical to ICE DXY"),
    "Gold": ("INSTRUMENT_FX_METALS_XAU_USD", "Real spot XAUUSD (not futures, unlike GC=F)"),
    "EURUSD": ("INSTRUMENT_FX_MAJORS_EUR_USD", None),
    "GBPUSD": ("INSTRUMENT_FX_MAJORS_GBP_USD", None),
    "USDCHF": ("INSTRUMENT_FX_MAJORS_USD_CHF", None),
    "USDJPY": ("INSTRUMENT_FX_MAJORS_USD_JPY", None),
}


def fetch_yfinance_quote(ticker_symbol):
    """Most recent daily close via yfinance. Returns None on any failure --
    never raises, so one bad ticker can't take down the whole comparison."""
    try:
        hist = yf.Ticker(ticker_symbol).history(period="2d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"[warn] yfinance fetch failed for {ticker_symbol}: {e}")
        return None


def fetch_dukascopy_quote(instrument_const_name, lookback_days=5):
    """
    Most recent daily close via Dukascopy. Returns None if the package
    isn't installed, the instrument constant doesn't exist, or the fetch
    fails for any reason -- always fails gracefully.
    """
    if dukascopy_python is None:
        print("[warn] dukascopy_python not installed. Run: pip install dukascopy-python")
        return None

    instrument = getattr(dk_instruments, instrument_const_name, None)
    if instrument is None:
        print(f"[warn] Dukascopy instrument constant {instrument_const_name} not found in this "
              f"version of the package -- their instrument list may have changed.")
        return None

    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    try:
        df = dukascopy_python.fetch(
            instrument, dukascopy_python.INTERVAL_DAY_1, dukascopy_python.OFFER_SIDE_BID,
            start, end,
        )
        if df is None or df.empty or "close" not in df.columns:
            return None
        return float(df["close"].iloc[-1])
    except Exception as e:
        print(f"[warn] Dukascopy fetch failed for {instrument_const_name}: {e}")
        return None


def get_quote(symbol, feed="yfinance"):
    """
    Fetches a single symbol's latest quote from the requested feed.
    Returns (value, source_used, note) -- source_used may differ from the
    requested feed (e.g. VIX always falls back to yfinance under
    --feed dukascopy), and note explains why/any caveat, so the display
    layer never has to guess what actually happened.
    """
    if feed == "dukascopy":
        if symbol == "VIX":
            value = fetch_yfinance_quote(YFINANCE_TICKERS["VIX"])
            return value, "yfinance", "VIX unavailable via Dukascopy -- fell back to yfinance"

        if symbol in DUKASCOPY_INSTRUMENTS:
            const_name, note = DUKASCOPY_INSTRUMENTS[symbol]
            value = fetch_dukascopy_quote(const_name)
            if value is not None:
                return value, "dukascopy", note or ""
            # Dukascopy fetch failed -- fall back to yfinance rather than showing nothing
            fallback = fetch_yfinance_quote(YFINANCE_TICKERS[symbol])
            return fallback, "yfinance (dukascopy fetch failed)", ""

    # default / explicit yfinance feed
    value = fetch_yfinance_quote(YFINANCE_TICKERS[symbol])
    return value, "yfinance", ""


def print_live_quotes(feed="yfinance"):
    from tabulate import tabulate

    print(f"\n--- Live Quotes (feed: {feed}) ---")
    if feed == "dukascopy" and dukascopy_python is None:
        print("  [warn] dukascopy_python not installed -- every symbol will fall back to yfinance.")
        print("         Run: pip install dukascopy-python")

    rows = []
    for symbol in YFINANCE_TICKERS.keys():
        value, source_used, note = get_quote(symbol, feed=feed)
        val_str = f"{value:,.4f}" if value is not None else "n/a"
        rows.append([symbol, val_str, source_used, note])

    print(tabulate(rows, headers=["Symbol", "Value", "Source used", "Note"], tablefmt="simple"))
    print()
    print("  Reminder: this is a standalone comparison view -- it does NOT feed into the")
    print("  regime model, factor attribution, or anything else in the daily briefing.")


if __name__ == "__main__":
    print_live_quotes()
