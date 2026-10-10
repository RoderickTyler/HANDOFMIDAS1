"""
gld_basis.py
-------------
The GLD <-> spot-gold "basis" snapshot, captured on its OWN (faster) cadence,
separate from the options-chain snapshot.

Why separate: strikes in a chain don't move, but the number that turns a GLD
strike into a spot-gold ($/oz) level does -- spot per GLD dollar =
XAUUSD / GLD. The existing app caches oz-per-share for 24h on purpose (so it
doesn't jitter). That is fine for a slow ratio, but any premium/discount drift
or a timing mismatch between the delayed chain and live spot goes unseen. This
module records the LIVE ratio every ~15 minutes (same cadence as the regime
logger) so both can be compared, and so the history of the ratio is visible.

Captured per snapshot:
    gld            GLD last price (yfinance, 1-minute bars when available)
    xau            XAUUSD spot (api.gold-api.com, keyless -- same source as spot_gold.py)
    gc             GC=F front future (yfinance)
    ratio          gld / xau            (= oz per share, live)
    spot_per_gld   xau / gld            (multiply a GLD level by this -> $/oz)
    gc_basis_pct   (gc - xau) / xau * 100
    ratio_valid    False when GLD hasn't traded recently (market closed / halted):
                   spot trades ~24h but GLD does not, so a ratio taken then would
                   mix a stale GLD price with a live spot and show a FAKE basis move.
                   Invalid points are stored (for the record) but never used to convert.

Storage: logs/gld_basis.json on the repo's `data` branch (written by gex_logger.py
in the scheduled job), newest-last, trimmed to the latest `KEEP` points.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Callable, Optional

import requests

REMOTE_URL = "https://raw.githubusercontent.com/RoderickTyler/HANDOFMIDAS1/data/logs/gld_basis.json"
LOCAL_PATH = os.path.join(tempfile.gettempdir(), "handofmidas_gld_basis.json")
SPOT_URL = "https://api.gold-api.com/price/XAU"
KEEP = 300                 # ~3 trading days at 15-minute spacing
FRESH_MINUTES = 20         # GLD/GC print older than this => not a live pairing


def _yf_last(symbol: str):
    """Returns (price, age_minutes) from the freshest 1-minute bar; falls back to the
    last daily close with age=None (age unknown => treated as not live)."""
    import yfinance as yf
    import pandas as pd
    tk = yf.Ticker(symbol)
    try:
        h = tk.history(period="1d", interval="1m")
        h = h.dropna(subset=["Close"])
        if not h.empty:
            last_ts = h.index[-1]
            age = (pd.Timestamp.now(tz=last_ts.tz) - last_ts).total_seconds() / 60.0
            return float(h["Close"].iloc[-1]), float(age)
    except Exception:
        pass
    d = tk.history(period="5d").dropna(subset=["Close"])
    if d.empty:
        raise ValueError(f"no price for {symbol}")
    return float(d["Close"].iloc[-1]), None


def _spot_xau(timeout: int = 10) -> float:
    r = requests.get(SPOT_URL, headers={"Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    return float(r.json()["price"])


def fetch_basis_snapshot(
    get_last: Callable = _yf_last,
    get_spot: Callable = _spot_xau,
    now: Optional[datetime] = None,
) -> dict:
    """Never raises. Missing pieces come back as None with a note in `errors`."""
    now = now or datetime.now(timezone.utc)
    errors = []
    gld = gld_age = gc = gc_age = xau = None
    try:
        gld, gld_age = get_last("GLD")
    except Exception as e:
        errors.append(f"GLD: {e}")
    try:
        gc, gc_age = get_last("GC=F")
    except Exception as e:
        errors.append(f"GC=F: {e}")
    try:
        xau = get_spot()
    except Exception as e:
        errors.append(f"XAUUSD spot: {e}")

    pt = {
        "ts_utc": now.isoformat(),
        "gld": gld, "gld_age_min": gld_age,
        "xau": xau, "gc": gc, "gc_age_min": gc_age,
        "ratio": None, "spot_per_gld": None, "ratio_valid": False,
        "gc_basis_pct": None, "errors": errors,
    }
    if gld and xau and gld > 0 and xau > 0:
        pt["ratio"] = gld / xau
        pt["spot_per_gld"] = xau / gld
        pt["ratio_valid"] = gld_age is not None and gld_age <= FRESH_MINUTES
    if gc and xau and xau > 0 and gc_age is not None and gc_age <= FRESH_MINUTES:
        pt["gc_basis_pct"] = (gc - xau) / xau * 100.0
    return pt


# ---------------------------------------------------------------------------
# History file (list of points, newest last)
# ---------------------------------------------------------------------------

def append_point(path: str, point: dict, keep: int = KEEP) -> list:
    pts = read_points(path)
    pts.append(point)
    pts = pts[-keep:]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"points": pts}, f)
    os.replace(tmp, path)
    return pts


def read_points(path: str) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        pts = obj.get("points", []) if isinstance(obj, dict) else []
        return pts if isinstance(pts, list) else []
    except Exception:
        return []


def fetch_remote_points(timeout: int = 10) -> list:
    try:
        r = requests.get(REMOTE_URL, timeout=timeout)
        if r.status_code != 200 or not r.text.strip():
            return []
        obj = r.json()
        pts = obj.get("points", []) if isinstance(obj, dict) else []
        return pts if isinstance(pts, list) else []
    except Exception:
        return []


def latest_valid(points: list) -> Optional[dict]:
    for p in reversed(points or []):
        if p.get("ratio_valid") and p.get("spot_per_gld"):
            return p
    return None


def age_minutes(point: dict, now: Optional[datetime] = None) -> Optional[float]:
    try:
        ts = datetime.fromisoformat(point["ts_utc"])
        return ((now or datetime.now(timezone.utc)) - ts).total_seconds() / 60.0
    except Exception:
        return None


def choose_basis(live: Optional[dict], logged: list, stable_oz: Optional[float]) -> dict:
    """Decides which ratio converts GLD levels to spot terms, and says why.
    Order: live valid point -> latest valid LOGGED point -> 24h-stable oz/share.
    Returns {spot_per_gld, source, point, note}."""
    if live and live.get("ratio_valid"):
        return {"spot_per_gld": live["spot_per_gld"], "source": "live", "point": live,
                "note": "live GLD/XAUUSD pairing"}
    lg = latest_valid(logged)
    if lg:
        return {"spot_per_gld": lg["spot_per_gld"], "source": "logged", "point": lg,
                "note": "latest valid logged 15-min point (live pairing not valid right now)"}
    if stable_oz:
        return {"spot_per_gld": 1.0 / stable_oz, "source": "stable", "point": None,
                "note": "24h-stable oz/share (no valid live or logged basis)"}
    return {"spot_per_gld": None, "source": "none", "point": None, "note": "no basis available"}


# ---------------------------------------------------------------------------
# Reconciliation: pulled vs self-calculated spot, and the deviation between them
# ---------------------------------------------------------------------------

def reconcile_rows(pt: Optional[dict], stable_oz: Optional[float], chain_spot: Optional[float] = None) -> list:
    """Spot price three ways, each compared with the PULLED spot (gold-api XAUUSD):
        pulled           -- XAUUSD straight from the feed
        self-calculated  -- GLD price / oz-per-share (the ratio the original GEX tab uses)
        GC=F             -- the futures contract (its gap to spot is the futures basis)
        chain-time       -- the CBOE chain's own (delayed) GLD spot / oz-per-share, so the
                            staleness of the chain shows up in $/oz
    A row whose inputs aren't trustworthy right now (GLD not live, no oz/share) carries
    value None and an explanatory note instead of a misleading number."""
    pt = pt or {}
    xau = pt.get("xau")
    rows = []

    def row(measure, how, value, note=""):
        dev = (value - xau) if (value is not None and xau) else None
        bps = (dev / xau * 1e4) if dev is not None else None
        rows.append({"Measure": measure, "How": how, "Value": value, "Dev vs pulled ($/oz)": dev,
                     "Dev vs pulled (bps)": bps, "Note": note})

    row("XAUUSD spot — PULLED", "gold-api.com, live", xau, "" if xau else "pull failed")
    if pt.get("gld") and stable_oz and pt.get("ratio_valid"):
        row("XAUUSD spot — SELF-CALCULATED", "GLD ÷ 24h-stable oz/share", pt["gld"] / stable_oz, "")
    else:
        why = ("GLD not live (market closed/halted)" if pt.get("gld") and not pt.get("ratio_valid")
               else "no oz/share available" if not stable_oz else "no GLD price")
        row("XAUUSD spot — SELF-CALCULATED", "GLD ÷ 24h-stable oz/share", None, why)
    if pt.get("gc") and pt.get("gc_basis_pct") is not None:
        row("GC=F futures (reference)", "yfinance front month", pt["gc"], "gap to spot = futures basis")
    else:
        row("GC=F futures (reference)", "yfinance front month", None, "not fresh right now")
    if chain_spot and stable_oz:
        row("Chain-time spot-equivalent", "CBOE chain GLD ÷ oz/share", chain_spot / stable_oz,
            "gap includes the chain's ~15-min delay")
    return rows


def deviation_series(points: list, stable_oz: Optional[float]) -> list:
    """[(ts_utc, self_calculated - pulled)] over logged VALID points -- how the stable oz/share
    has tracked the live market through time."""
    if not stable_oz:
        return []
    out = []
    for p in points or []:
        if p.get("ratio_valid") and p.get("gld") and p.get("xau"):
            out.append((p["ts_utc"], p["gld"] / stable_oz - p["xau"]))
    return out
