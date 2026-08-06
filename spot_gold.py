"""
spot_gold.py
------------
Fetches a REAL spot XAUUSD quote, purely to compare directly against the
GC=F futures price the rest of this system uses for everything else. This
does NOT replace GC=F anywhere -- it's an additional, clearly-labeled data
point so you can see both side by side and know exactly which one you're
looking at.

Source: gold-api.com's free `/price/{symbol}` endpoint -- no signup, no
API key, no rate limit on this endpoint (per their published docs:
https://gold-api.com/llms.txt). This replaces an earlier version that used
GoldAPI.io, which required a free key; that's no longer needed.
"""

import requests

SPOT_URL = "https://api.gold-api.com/price/XAU"


def fetch_xauusd_spot(timeout=15):
    """
    Fetches the real, current XAU/USD spot quote from gold-api.com.
    Returns a dict with price/currency/symbol/updated_at, or None if the
    request fails -- always fails gracefully, never raises, never crashes
    the caller.
    """
    try:
        resp = requests.get(SPOT_URL, headers={"Accept": "application/json"}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "price" not in data:
            print(f"[warn] gold-api.com response missing expected 'price' field: {data}")
            return None
        return {
            "price": data.get("price"),
            "currency": data.get("currency", "USD"),
            "symbol": data.get("symbol", "XAU"),
            "updated_at": data.get("updatedAt"),
        }
    except Exception as e:
        print(f"[warn] gold-api.com spot fetch failed (non-fatal, GC=F price below still works): {e}")
        return None
