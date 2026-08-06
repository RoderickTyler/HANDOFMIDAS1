"""
spot_gold.py
------------
Fetches a REAL spot XAUUSD quote via GoldAPI.io, purely to compare
directly against the GC=F futures price the rest of this system uses for
everything else. This does NOT replace GC=F anywhere -- it's an
additional, clearly-labeled data point so you can see both side by side
and know exactly which one you're looking at.

Free tier requires a free signup (no card required) -- unlike FRED/CFTC/
IMF, which need no key at all. Get one at https://www.goldapi.io, then
add it to your .env file as:
    GOLDAPI_KEY=your_key_here

If this key isn't set, this feature is skipped gracefully with a one-line
note -- everything else in the system keeps working exactly as before.

Endpoint and response format confirmed directly against GoldAPI.io's own
published examples (JS/Python/Node.js official repos + docs), not guessed.
"""

import requests

import config

GOLDAPI_URL = "https://www.goldapi.io/api/XAU/USD"


def fetch_xauusd_spot(timeout=15):
    """
    Fetches the real, current XAU/USD spot quote from GoldAPI.io.
    Returns a dict with price/bid/ask/exchange/symbol/timestamp, or None
    if the key isn't set or the request fails -- always fails gracefully,
    never raises, never crashes the caller.
    """
    api_key = getattr(config, "GOLDAPI_KEY", "")
    if not api_key:
        return None

    try:
        resp = requests.get(
            GOLDAPI_URL,
            headers={"x-access-token": api_key, "Accept": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "price" not in data:
            print(f"[warn] GoldAPI.io response missing expected 'price' field: {data}")
            return None
        return {
            "price": data.get("price"),
            "bid": data.get("bid"),
            "ask": data.get("ask"),
            "exchange": data.get("exchange"),
            "symbol": data.get("symbol"),
            "timestamp": data.get("timestamp"),
        }
    except Exception as e:
        print(f"[warn] GoldAPI.io spot fetch failed (non-fatal, GC=F price below still works): {e}")
        return None
