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
import gld_basis


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
        payload = gex_alt.build_snapshot_payload("GLD", spot, meta, enriched, r, r_note, gvz, "mid",
                                                 avg_volume=gex_alt.fetch_avg_volume_10d("GLD"))
        payload["saved_utc"] = datetime.now(timezone.utc).isoformat()
        store["GLD|alt"] = gs._clean(payload)
        print(f"[alt] ok spot={spot:.2f} views={len(payload['views'])} gvz={gvz}")
        return True
    except Exception as e:
        print(f"[alt] FAILED, keeping previous entry: {type(e).__name__}: {e}")
        return False


def log_basis(path: str) -> bool:
    """Fast cadence (~15 min): GLD/XAUUSD/GC basis point appended to a rolling history."""
    try:
        pt = gld_basis.fetch_basis_snapshot()
        if pt["gld"] is None and pt["xau"] is None:
            print(f"[basis] FAILED, nothing captured: {pt['errors']}")
            return False
        gld_basis.append_point(path, pt)
        print(f"[basis] ok ratio={pt['ratio']} valid={pt['ratio_valid']} gc_basis_pct={pt['gc_basis_pct']} errors={pt['errors']}")
        return True
    except Exception as e:
        print(f"[basis] FAILED: {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path of gex_last_good.json")
    ap.add_argument("--basis-out", default=None, help="path of gld_basis.json (default: next to --out)")
    ap.add_argument("--mode", choices=["auto", "basis", "full"], default="auto",
                    help="auto = basis every run, GEX chain only in the first 15 min of each half hour")
    args = ap.parse_args()
    basis_path = args.basis_out or os.path.join(os.path.dirname(args.out) or ".", "gld_basis.json")

    do_gex = args.mode == "full" or (args.mode == "auto" and datetime.now(timezone.utc).minute % 30 < 15)
    wrote = False

    if args.mode in ("auto", "basis", "full"):
        wrote |= log_basis(basis_path)

    if do_gex:
        store = {}
        if os.path.exists(args.out):
            try:
                with open(args.out, "r", encoding="utf-8") as f:
                    store = json.load(f)
            except Exception:
                store = {}
        ok_main = log_main(store)
        ok_alt = log_alt(store)
        if ok_main or ok_alt:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(store, f)
            print(f"Wrote {args.out}")
            wrote = True
    else:
        print("[gex] skipped this run (chain snapshot runs on the slower half-hour cadence)")

    if not wrote:
        print("Nothing new to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
