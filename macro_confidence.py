"""
macro_confidence.py
--------------------
Pure logic behind the "Macro Confidence" tab. It gathers the checks the app
ALREADY computes in different places (regime probability, model health,
recent-vs-full persistence, logged persistence tier, direction-regime
cross-check, COT crowding, volatility percentile) into one checklist.

IMPORTANT -- what this is and is not:
    - It is a RULE-BASED CHECKLIST of agreeing / cautioning / conflicting
      layers. It is NOT a calibrated probability, and the layers are not
      independent (several derive from the same HMM), so "5 of 6 agree" is
      weaker evidence than it sounds. The tab says so.
    - Nothing here changes the trading mode or size. determine_mode() stays
      the single source of truth for that; this only explains how much
      corroboration sits behind it.
    - GEX is deliberately excluded: it is a separate dealer-positioning
      overlay, not part of the macro regime read.
"""

from __future__ import annotations

from typing import Optional

SUPPORTS, CAUTION, CONFLICTS, NEUTRAL, NA = "supports", "caution", "conflicts", "neutral", "n/a"

ICON = {SUPPORTS: "✅", CAUTION: "⚠️", CONFLICTS: "❌", NEUTRAL: "➖", NA: "❔"}


def _layer(name: str, status: str, detail: str) -> dict:
    return {"layer": name, "status": status, "detail": detail}


def build_layers(
    regime_result: Optional[dict],
    mode: Optional[dict],
    persistence: Optional[dict] = None,
    smoothed_result: Optional[dict] = None,
    vol_context: Optional[dict] = None,
) -> list[dict]:
    layers: list[dict] = []
    if regime_result is None or mode is None:
        return layers

    top_state, top_prob = mode["top_state"], float(mode["top_prob"])

    # 1. HMM conviction
    if top_prob >= 0.70:
        s = SUPPORTS
    elif top_prob >= 0.50:
        s = CAUTION
    else:
        s = CONFLICTS
    layers.append(_layer("HMM state conviction", s,
                         f"{top_state} at {top_prob*100:.0f}% (≥70% supports, 50–70% caution, <50% conflicts)."))

    # 2. Model health / sample size
    healthy = bool(regime_result.get("model_healthy", True))
    small = top_state in (regime_result.get("small_sample_states") or [])
    if not healthy:
        layers.append(_layer("Model health", CONFLICTS, "Overfitting check flagged this fit."))
    elif small:
        layers.append(_layer("Model health", CAUTION, f"{top_state} is a small-sample state; its probability is less trustworthy."))
    else:
        layers.append(_layer("Model health", SUPPORTS, "Health check passed; state has adequate sample."))

    # 3. Recent vs full-history persistence
    if mode.get("consistency_ok") is None:
        layers.append(_layer("Recent vs full-history persistence", NA, "Not enough data for a recent-window comparison."))
    else:
        layers.append(_layer("Recent vs full-history persistence",
                             SUPPORTS if mode["consistency_ok"] else CONFLICTS,
                             mode.get("consistency_note", "")))

    # 4. Logged persistence tier (sticky / neutral / choppy)
    if persistence is None:
        layers.append(_layer("Logged persistence (weekly log)", NA, "Regime log not available yet."))
    else:
        tier = persistence["tier"]
        status = {"sticky": SUPPORTS, "neutral": CAUTION, "choppy": CONFLICTS}.get(tier, NA)
        layers.append(_layer(
            "Logged persistence (weekly log)", status,
            f"{tier}: {persistence['flips_recent']} flip(s) in last {persistence['choppy_window_hours']}h, "
            f"current streak {persistence['current_streak_hours']:.1f}h."))

    # 5. Direction-regime cross-check (smoothed model)
    if smoothed_result is None:
        layers.append(_layer("Direction-regime cross-check (smoothed model)", NA, "Not run yet — press the button above."))
    else:
        sp = smoothed_result["current_probs"]
        s_state = max(sp, key=sp.get)
        s_prob = sp[s_state]
        if s_state == top_state:
            layers.append(_layer("Direction-regime cross-check (smoothed model)", SUPPORTS,
                                 f"Both models say {top_state} (smoothed: {s_prob*100:.0f}%)."))
        else:
            layers.append(_layer("Direction-regime cross-check (smoothed model)", CONFLICTS,
                                 f"Original says {top_state}, smoothed says {s_state} ({s_prob*100:.0f}%). "
                                 f"The original is mostly a volatility call; the smoothed one leans on direction."))

    # 6. COT crowding
    crowd = mode.get("cot_crowding_detected")
    if crowd is None:
        layers.append(_layer("COT positioning", NA, mode.get("cot_note", "COT not available.")))
    elif crowd:
        layers.append(_layer("COT positioning", CAUTION, mode.get("cot_note", "") + " (size already stepped down)"))
    else:
        layers.append(_layer("COT positioning", SUPPORTS, mode.get("cot_note", "")))

    # 7. Volatility context
    if vol_context is None:
        layers.append(_layer("Volatility vs own history", NA, "Not enough history."))
    else:
        pct = vol_context["percentile_in_own_history"]
        if pct >= 80:
            layers.append(_layer("Volatility vs own history", CAUTION,
                                 f"{pct:.0f}th percentile — elevated; wider stops / smaller size consistent."))
        elif pct <= 20:
            layers.append(_layer("Volatility vs own history", NEUTRAL, f"{pct:.0f}th percentile — calm."))
        else:
            layers.append(_layer("Volatility vs own history", NEUTRAL, f"{pct:.0f}th percentile — unremarkable."))

    return layers


def summarize(layers: list[dict]) -> dict:
    """Tally evaluable layers (NEUTRAL and n/a are excluded from the count)."""
    ev = [l for l in layers if l["status"] in (SUPPORTS, CAUTION, CONFLICTS)]
    n_sup = sum(l["status"] == SUPPORTS for l in ev)
    n_cau = sum(l["status"] == CAUTION for l in ev)
    n_con = sum(l["status"] == CONFLICTS for l in ev)
    total = len(ev)
    if total == 0:
        label = "Insufficient information"
    elif n_con == 0 and n_cau == 0:
        label = "Aligned"
    elif n_con >= 2 or (total and n_con / total >= 0.4):
        label = "Conflicted"
    else:
        label = "Mixed"
    return {"supports": n_sup, "caution": n_cau, "conflicts": n_con, "evaluable": total,
            "not_evaluable": len(layers) - total, "label": label}
