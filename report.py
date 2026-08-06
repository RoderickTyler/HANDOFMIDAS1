"""
report.py
---------
Formats everything into a single readable daily briefing — the kind of
one-pager a junior analyst would send to a desk lead.
"""

from tabulate import tabulate

import journal


def fmt(val, decimals=2, suffix=""):
    if val is None:
        return "n/a"
    try:
        return f"{val:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def build_report(summary: dict, flags: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"GOLD MACRO DAILY BRIEFING — {summary.get('date', 'n/a')}")
    lines.append("=" * 70)

    table = [
        ["Gold (GC=F futures)", fmt(summary.get("gold_spot")), fmt(summary.get("gold_chg_1d"), 2, " (1d)")
         + " -- COMEX futures, NOT true spot (~0.5-1.5% basis gap is normal)"],
    ]

    xauusd = summary.get("xauusd_spot")
    if xauusd is not None:
        gc_f = summary.get("gold_spot")
        diff_note = ""
        if gc_f is not None:
            diff = xauusd - gc_f
            diff_pct = (diff / gc_f * 100) if gc_f else 0
            diff_note = f" -- {diff:+.2f} ({diff_pct:+.2f}%) vs. GC=F above"
        table.append(["Gold (XAUUSD spot, gold-api.com)", fmt(xauusd), f"REAL spot quote, live API{diff_note}"])
    else:
        table.append(["Gold (XAUUSD spot)", "n/a", "Spot fetch failed this run (rare -- source needs no key)"])

    table += [
        ["DXY", fmt(summary.get("dxy")), fmt(summary.get("dxy_chg_1d"), 2, " (1d)")],
        ["VIX", fmt(summary.get("vix")), ""],
        ["10Y Nominal Yield", fmt(summary.get("dgs10"), 2, "%"), ""],
        ["2Y Nominal Yield", fmt(summary.get("dgs2"), 2, "%"), ""],
        ["10Y Real Yield (TIPS)", fmt(summary.get("dfii10"), 2, "%"), ""],
        ["Real yield chg (5d)", fmt(summary.get("real_yield_chg_5d_bps"), 1, " bps"), ""],
        ["Real yield chg (20d)", fmt(summary.get("real_yield_chg_20d_bps"), 1, " bps"), ""],
        ["2s10s Curve Spread", fmt(summary.get("curve_spread"), 2, "%"), ""],
        ["30d Corr: Gold vs DXY", fmt(summary.get("corr_gold_dxy_30d"), 2), "(expect negative)"],
        ["30d Corr: Gold vs Real Yield", fmt(summary.get("corr_gold_realyield_30d"), 2), "(expect negative)"],
    ]
    lines.append(tabulate(table, headers=["Indicator", "Value", "Note"], tablefmt="simple"))
    lines.append("")
    lines.append("  Quick reference -- which price to use:")
    lines.append("  * GC=F futures: what every other calculation in this system uses (regime model,")
    lines.append("    factor attribution, divergence flags, etc.) -- consistent across the whole system,")
    lines.append("    but not identical to a live spot quote you'd see on a broker's chart.")
    lines.append("  * XAUUSD spot: pulled fresh from gold-api.com's live API each run -- this is the number")
    lines.append("    to compare against your actual trading platform's spot price.")

    lines.append("\n--- DIVERGENCE FLAGS (where the textbook model may be breaking) ---")
    if flags:
        for key, msg in flags.items():
            lines.append(f"  [!] {msg}")
    else:
        lines.append("  None triggered today — relationships holding roughly as expected.")

    return "\n".join(lines)


def print_full_briefing(summary: dict, flags: dict):
    print(build_report(summary, flags))
    journal.print_reflection_prompts()

    due = journal.entries_due_for_review()
    if due:
        print("\n--- Past theses due for review ---")
        for row in due:
            print(f"  [{row['entry_date']}] {row['dominant_factor']}: {row['thesis']}")
            print(f"      Prediction: {row['falsifiable_prediction']}")
            print(f"      -> Score this with: journal.score_entry('{row['entry_date']}', True/False, 'note')")
