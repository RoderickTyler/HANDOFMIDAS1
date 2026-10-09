"""
gex_logger.py
--------------
Run by the scheduled GitHub Action (gex_log.yml) -- NOT by Streamlit.
Pulls GLD options data from sources that don't need an API key and writes the
"last good" JSON the dashboard falls back to when its own live fetch fails.

    python gex_logger.py --out data_branch/logs/gex_last_good.json

Rules:
  - A source that fails this run leaves its previous entry untouched (we never
    overwrite a good snapshot with nothing).
  - Exits 0 even if both sources fail, so a transient outage doesn't turn the
    scheduled job red; it just prints what happened.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import gex_alt
import gex_snapshot as gs


def log_main(store: dict) -> bool:
    try:
        import gex_engine
        chain, spot = gex_engine.fetch_gld_chain_yfinance(ticker="GLD", expiration=None, min_open_interest=None)
        result = gex_engine.run_assessment(chain, spot=spot, underlying="GLD")
        payload = gs.payload_from_result(result, None, "yfinance (scheduled logger)")
        payload["saved_utc"] = datetime.now(timezone.utc).isoformat()
        store["GLD|main"] = gs._clean(payload)
        print(f"[main] ok spot={spot:.2f} net_gex={result.net_gex:,.0f}")
        return True
    except Exception as e:
        print(f"[main] FAILED, keeping previous entry: {type(e).__name__}: {e}")
        return False


def log_alt(store: dict) -> bool:
    try:
        df, meta = gex_alt.fetch_cboe_chain("GLD")
        spot = meta["spot"]
        r, r_note = gex_alt.fetch_risk_free()
        gvz = gex_alt.fetch_gvz()
        df = gex_alt.add_time_to_expiry(df, meta.get("asof_utc"))
        enriched = gex_alt.enrich_chain(df, spot, r, "mid")
        payload = gex_alt.build_snapshot_payload("GLD", spot, meta, enriched, r, r_note, gvz, "mid")
        payload["saved_utc"] = datetime.now(timezone.utc).isoformat()
        store["GLD|alt"] = gs._clean(payload)
        print(f"[alt] ok spot={spot:.2f} views={len(payload['views'])} gvz={gvz}")
        return True
    except Exception as e:
        print(f"[alt] FAILED, keeping previous entry: {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    store = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                store = json.load(f)
        except Exception:
            store = {}

    ok_main = log_main(store)
    ok_alt = log_alt(store)
    if not (ok_main or ok_alt):
        print("Nothing new to write.")
        return 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(store, f)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
