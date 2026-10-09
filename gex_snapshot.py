"""
gex_snapshot.py
----------------
"Last good GEX pull" store, so a rate-limited / failed live fetch shows the
most recent SUCCESSFUL result -- clearly labelled with when it was taken --
instead of an empty tab.

Two places a snapshot can live, and the newest one wins:
    1. LOCAL  -- a JSON file in the OS temp dir of the running app. Written
       every time a live pull succeeds. Survives Streamlit reruns and
       sessions, but NOT a Streamlit Cloud reboot/redeploy (ephemeral disk).
    2. REMOTE -- logs/gex_last_good.json on the repo's `data` branch, written
       by the optional scheduled job (gex_logger.py + gex_log.yml). Survives
       everything, and is fetched over plain HTTP exactly like the regime log
       (so committing to it never triggers a redeploy).

Payloads are keyed, e.g. "GLD|main" (the original GEX tab) and "GLD|alt"
(the GEX (Alternative) tab). Nothing here touches the network except
`fetch_remote` (short timeout, never raises).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

SGT = ZoneInfo("Asia/Singapore")
LOCAL_PATH = os.path.join(tempfile.gettempdir(), "handofmidas_gex_last_good.json")
REMOTE_URL = "https://raw.githubusercontent.com/RoderickTyler/HANDOFMIDAS1/data/logs/gex_last_good.json"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _clean(o):
    """Make numpy / pandas / dates JSON-safe; NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (datetime, pd.Timestamp)):
        return o.isoformat()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if isinstance(o, pd.DataFrame):
        return {"__df__": True, "index": _clean(list(o.index)), "columns": list(map(str, o.columns)),
                "data": _clean(o.to_numpy().tolist())}
    return o


def df_to_json(df: pd.DataFrame) -> dict:
    return _clean(df)


def df_from_json(d: dict) -> pd.DataFrame:
    df = pd.DataFrame(d["data"], columns=d["columns"], index=d["index"])
    try:
        df.index = pd.to_numeric(df.index)
    except Exception:
        pass
    return df.apply(pd.to_numeric, errors="coerce")


def _read_file(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_file(path: str, obj: dict) -> bool:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_clean(obj), f)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_payload(key: str, payload: dict, path: Optional[str] = None) -> bool:
    """Stores `payload` under `key`, stamping it with the save time (UTC).
    Merges into whatever else is in the file. Never raises."""
    path = path or LOCAL_PATH
    store = _read_file(path)
    entry = dict(payload)
    entry["saved_utc"] = datetime.now(timezone.utc).isoformat()
    store[key] = entry
    return _write_file(path, store)


def fetch_remote(timeout: int = 10) -> dict:
    try:
        r = requests.get(REMOTE_URL, timeout=timeout)
        if r.status_code != 200 or not r.text.strip():
            return {}
        obj = r.json()
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_ts(s) -> Optional[datetime]:
    try:
        dt = pd.to_datetime(s, utc=True)
        return dt.to_pydatetime()
    except Exception:
        return None


def load_best_payload(key: str, remote_store: Optional[dict] = None, path: Optional[str] = None):
    """Returns (payload, origin) where origin is 'local' or 'data-branch',
    choosing whichever copy was saved most recently; (None, None) if neither."""
    path = path or LOCAL_PATH
    cands = []
    loc = _read_file(path).get(key)
    if isinstance(loc, dict) and loc.get("saved_utc"):
        cands.append((_parse_ts(loc["saved_utc"]), loc, "local"))
    rem = (remote_store or {}).get(key)
    if isinstance(rem, dict) and rem.get("saved_utc"):
        cands.append((_parse_ts(rem["saved_utc"]), rem, "data-branch"))
    cands = [c for c in cands if c[0] is not None]
    if not cands:
        return None, None
    cands.sort(key=lambda c: c[0], reverse=True)
    return cands[0][1], cands[0][2]


# ---------------------------------------------------------------------------
# Time display (SGT, with age)
# ---------------------------------------------------------------------------

def describe_age(saved_utc: str, now: Optional[datetime] = None) -> tuple[str, str]:
    """Returns ('2026-10-10 02:41 SGT', '3h 12m ago')."""
    dt = _parse_ts(saved_utc)
    if dt is None:
        return "unknown time", "unknown age"
    now = now or datetime.now(timezone.utc)
    mins = max(int((now - dt).total_seconds() // 60), 0)
    if mins < 60:
        age = f"{mins}m ago"
    elif mins < 48 * 60:
        age = f"{mins // 60}h {mins % 60}m ago"
    else:
        age = f"{mins // (24 * 60)}d ago"
    return dt.astimezone(SGT).strftime("%Y-%m-%d %H:%M SGT"), age


# ---------------------------------------------------------------------------
# Main (original) GEX tab: AssessmentResult <-> payload
# ---------------------------------------------------------------------------

def payload_from_result(result, expiration, source: str, band: float = 0.30) -> dict:
    """Serialises a gex_engine.AssessmentResult. Per-strike tables are trimmed
    to +/-band of spot to keep the file small."""
    spot = float(result.spot)

    def trim(df):
        if df is None:
            return None
        d = df[(df.index >= spot * (1 - band)) & (df.index <= spot * (1 + band))]
        return df_to_json(d if not d.empty else df)

    return {
        "kind": "main",
        "underlying": result.underlying,
        "spot": spot,
        "net_gex": float(result.net_gex),
        "gamma_flip": None if result.gamma_flip is None else float(result.gamma_flip),
        "dealer_delta": float(result.dealer_delta),
        "regime": result.regime,
        "call_walls": [[float(a), float(b)] for a, b in result.call_walls],
        "put_walls": [[float(a), float(b)] for a, b in result.put_walls],
        "gex_by_strike": trim(result.gex_by_strike),
        "oi_by_strike": trim(result.oi_by_strike),
        "expiration": None if expiration is None else str(expiration),
        "source": source,
    }


def result_from_payload(payload: dict):
    """Rebuilds a gex_engine.AssessmentResult so the existing tab code renders
    it unchanged. Imported lazily so this module has no hard dependency."""
    import gex_engine
    gbs = df_from_json(payload["gex_by_strike"]) if payload.get("gex_by_strike") else None
    obs = df_from_json(payload["oi_by_strike"]) if payload.get("oi_by_strike") else None
    return gex_engine.AssessmentResult(
        underlying=payload["underlying"],
        spot=float(payload["spot"]),
        net_gex=float(payload["net_gex"]),
        gamma_flip=payload.get("gamma_flip"),
        dealer_delta=float(payload["dealer_delta"]),
        regime=payload["regime"],
        call_walls=[tuple(x) for x in payload.get("call_walls", [])],
        put_walls=[tuple(x) for x in payload.get("put_walls", [])],
        gex_by_strike=gbs,
        oi_by_strike=obs,
    )
