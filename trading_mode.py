"""
trading_mode.py
----------------
Translates the HMM regime output into a concrete, actionable line: which
strategy family to favor today (mean reversion vs. trend following), how
confident that call is, and a size recommendation -- the missing link
between "here's the state" and "here's what I do with it at 8am."

This does NOT replace your 5m/15m/1h/4h chart work. It sets your MODE and
SIZE for the day; your actual entries, stops, and targets still come from
your own intraday technical read, exactly as before. Think of this as a
filter you check once each morning, not a signal you check intraday.

COT integration (optional `cot_result` parameter): per the design agreed
after review -- COT modulates CONFIDENCE, never DIRECTION. If the regime
model says "Rising" (long bias) and COT shows positioning already
historically elevated on the long side, that's a real reason to reduce
size (a crowded trade has more room to unwind against you), but it never
flips the mode to short. Only ONE clean, defensible mechanism (same-
direction historical extremity) actually adjusts size -- other COT
signals (positioning surprise, opposite-direction positioning) are
surfaced as information for your own judgment, not silently baked into
the size number, since that mapping hasn't been validated yet (see the
planned Regime x COT hit-rate test in validation.py).
"""

STRATEGY_MAP = {
    "Range": ("MEAN REVERSION", "Fade extremes toward support/resistance on 15m/1h. "
                                  "Avoid chasing breakouts; expect reversion, not continuation."),
    "Rising": ("TREND FOLLOWING (LONG BIAS)", "Buy dips/pullbacks on 5m/15m within the uptrend. "
                                                "Trail stops, let winners run. Avoid fading strength."),
    "Declining": ("TREND FOLLOWING (SHORT BIAS)", "Sell rallies/pullbacks on 5m/15m within the downtrend. "
                                                     "Trail stops, let winners run. Avoid fading weakness."),
}

SIZE_TIERS = ["MINIMAL", "REDUCED", "NORMAL-TO-REDUCED", "NORMAL"]


def _cot_alignment_check(top_state, cot_result):
    """
    The ONE size-affecting COT mechanism: does historically-extreme
    positioning sit on the SAME side as the regime's current directional
    bias? That's the one case with a clear, defensible rationale (a
    crowded trade has more room to unwind against you) -- everything else
    COT tells you gets surfaced as information, not folded into size.
    """
    if cot_result is None:
        return None, "COT data not available this run"

    index_3yr = cot_result.get("cot_index", {}).get("3yr")
    if index_3yr is None:
        return None, "Not enough COT history for a 3yr read"

    bias_direction = {"Rising": "long", "Declining": "short"}.get(top_state)
    if bias_direction is None:
        return None, f"No directional bias for state '{top_state}' -- COT alignment n/a"

    if bias_direction == "long" and index_3yr >= 80:
        return True, (f"Same-direction crowding: regime bias is LONG, positioning is historically "
                       f"elevated ({index_3yr:.0f}/100 on 3yr COT Index).")
    if bias_direction == "short" and index_3yr <= 20:
        return True, (f"Same-direction crowding: regime bias is SHORT, positioning is historically "
                       f"depressed/elevated-short ({index_3yr:.0f}/100 on 3yr COT Index).")

    return False, (f"Positioning is not historically stretched in the direction of the current trend "
                    f"(3yr COT Index: {index_3yr:.0f}/100).")


def _downgrade_size(size_label):
    """Steps size down exactly one tier, floored at REDUCED -- COT is a
    secondary confirmatory signal, not treated as strong enough alone to
    push all the way down to MINIMAL."""
    base = size_label.split(" ")[0] if size_label else "NORMAL"
    if base not in SIZE_TIERS:
        return size_label
    idx = SIZE_TIERS.index(base)
    new_idx = max(idx - 1, SIZE_TIERS.index("REDUCED"))
    return SIZE_TIERS[new_idx]


def determine_mode(regime_result, cot_result=None):
    """
    Takes the dict returned by hmm_regime.analyze_regime() (and optionally
    cot_analysis.build_report()) and produces a concrete mode + confidence
    + size recommendation.
    """
    if regime_result is None:
        return None

    current_probs = regime_result["current_probs"]
    top_state = max(current_probs, key=current_probs.get)
    top_prob = current_probs[top_state]

    strategy_name, strategy_note = STRATEGY_MAP.get(top_state, ("UNKNOWN", "State not recognized."))

    # Confidence tier from how dominant the top state is
    if top_prob >= 0.70:
        confidence = "HIGH"
    elif top_prob >= 0.50:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    # Consistency check: does the recent-window matrix agree with the
    # full-history matrix on how persistent THIS state is? Large
    # disagreement is exactly the kind of thing that burned us earlier --
    # a stale full-history read masking a real recent regime change.
    consistency_note = "n/a -- not enough data for a recent-window comparison"
    consistency_ok = True
    full_matrix = regime_result.get("transition_matrix")
    recent_matrix = regime_result.get("recent_matrix")

    if recent_matrix is not None and top_state in full_matrix.index and top_state in recent_matrix.index:
        full_persist = full_matrix.loc[top_state, top_state]
        recent_persist = recent_matrix.loc[top_state, top_state]
        diff = abs(full_persist - recent_persist)
        if diff > 0.25:
            consistency_ok = False
            consistency_note = (f"DISAGREE -- full-history says {top_state} persists "
                                 f"{full_persist*100:.0f}% of the time, recent 3mo says "
                                 f"{recent_persist*100:.0f}%. Treat today's read with caution.")
        else:
            consistency_note = (f"agree -- full-history {full_persist*100:.0f}% vs. "
                                 f"recent 3mo {recent_persist*100:.0f}% persistence for {top_state}.")

    # Size recommendation combines confidence + consistency. Small-sample
    # states get an automatic downgrade regardless of how confident the
    # probability looks, since that confidence isn't well-supported.
    is_small_sample = top_state in regime_result.get("small_sample_states", [])

    if is_small_sample:
        size = "MINIMAL -- small-sample state, don't trust this confidence level"
    elif confidence == "HIGH" and consistency_ok:
        size = "NORMAL"
    elif confidence == "LOW" or not consistency_ok:
        size = "REDUCED -- state unclear or full/recent history disagree"
    else:
        size = "NORMAL-TO-REDUCED -- moderate confidence"

    # COT modulation: only the same-direction-crowding case adjusts size,
    # by exactly one tier, and never below REDUCED from this signal alone.
    cot_crowding, cot_note = _cot_alignment_check(top_state, cot_result)
    if cot_crowding:
        downgraded = _downgrade_size(size)
        size = f"{downgraded} -- downgraded from COT crowding: {cot_note}"

    result = {
        "top_state": top_state,
        "top_prob": top_prob,
        "confidence": confidence,
        "strategy_name": strategy_name,
        "strategy_note": strategy_note,
        "consistency_ok": consistency_ok,
        "consistency_note": consistency_note,
        "size": size,
        "cot_crowding_detected": cot_crowding,
        "cot_note": cot_note,
    }

    # Surface other COT info transparently -- informational only, does NOT
    # affect size (see module docstring: this mapping isn't validated yet).
    if cot_result is not None:
        result["cot_level_interpretation"] = cot_result.get("level_interpretation")
        surprise = cot_result.get("surprise")
        if surprise and surprise.get("is_surprise"):
            result["cot_surprise_note"] = (
                f"Note: large recent positioning surprise detected ({surprise['direction']}, "
                f"4wk change {surprise['change_4wk']:+,.0f} contracts, z={surprise['z_score']:+.2f}). "
                f"Not factored into size above -- shown for your own judgment."
            )

    result["overall_assessment"] = build_overall_assessment(result)
    return result


def build_overall_assessment(mode: dict):
    """
    Synthesizes regime state, confidence, persistence-consistency, and COT
    positioning into ONE readable paragraph, instead of leaving the reader
    to mentally combine four separate lines themselves. This doesn't add
    a new number/score -- it's purely a plain-English summary of what the
    other fields already say, written the way a human analyst would.
    """
    state = mode["top_state"]
    confidence = mode["confidence"]
    prob = mode["top_prob"]

    bias_phrase = {
        "Rising": "supportive of a long bias",
        "Declining": "supportive of a short bias",
        "Range": "directionless -- no trend bias favored",
    }.get(state, "unclear")

    confirm_phrase = {
        "HIGH": f"confirmed with high confidence ({prob*100:.0f}%)",
        "MODERATE": f"moderately confirmed ({prob*100:.0f}%)",
        "LOW": f"not clearly confirmed ({prob*100:.0f}%, low conviction)",
    }.get(confidence, "unclear")

    consistency_phrase = ("recent behavior agrees with the longer-run pattern"
                           if mode.get("consistency_ok")
                           else "recent behavior has DIVERGED from the longer-run pattern -- treat with caution")

    if mode.get("cot_crowding_detected") is True:
        positioning_phrase = "positioning IS historically stretched in the direction of this trend"
        conviction_note = "Conviction reduced -- upside/downside may be increasingly dependent on new participants joining a crowded trade."
    elif mode.get("cot_crowding_detected") is False:
        positioning_phrase = "positioning is NOT historically stretched in the direction of this trend"
        conviction_note = "No positioning-based reason to reduce conviction."
    else:
        positioning_phrase = "positioning data wasn't available this run"
        conviction_note = "Size recommendation reflects regime/persistence only."

    paragraph = (
        f"Macro backdrop is {bias_phrase}. Trend strength is {confirm_phrase} by regime analysis, "
        f"and {consistency_phrase}. On the positioning side, {positioning_phrase}. {conviction_note}"
    )
    return paragraph
