"""
journal.py
----------
This is the piece that actually builds the "junior -> senior" skill over
time: a falsifiable-prediction log.

Every week, you write down (1) which factor you think is dominant, and
(2) a SPECIFIC, checkable prediction that would prove you wrong. A week or
two later, the system shows you that entry again next to what actually
happened, so you score your own reasoning instead of just reading charts.

This is the single highest-leverage habit mentioned in the senior-analyst
discussion — most of the "skill" is really just accumulated, reviewed
track record.
"""

import os
import csv
from datetime import datetime, timedelta

import config

FIELDS = [
    "entry_date", "dominant_factor", "thesis", "falsifiable_prediction",
    "check_by_date", "outcome", "was_correct", "notes",
]

DOMINANT_FACTOR_OPTIONS = [
    "real_yields", "dxy", "central_bank_buying", "geopolitical_risk",
    "curve/recession_signal", "other",
]


def _ensure_file():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    if not os.path.exists(config.JOURNAL_FILE):
        with open(config.JOURNAL_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()


def add_entry(dominant_factor, thesis, falsifiable_prediction, check_in_days=14):
    """Log a new weekly thesis + falsifiable prediction."""
    _ensure_file()
    entry = {
        "entry_date": datetime.now().strftime("%Y-%m-%d"),
        "dominant_factor": dominant_factor,
        "thesis": thesis,
        "falsifiable_prediction": falsifiable_prediction,
        "check_by_date": (datetime.now() + timedelta(days=check_in_days)).strftime("%Y-%m-%d"),
        "outcome": "",
        "was_correct": "",
        "notes": "",
    }
    with open(config.JOURNAL_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(entry)
    print(f"Logged thesis for {entry['entry_date']}. Check back by {entry['check_by_date']}.")


def entries_due_for_review():
    """Return past entries whose check_by_date has arrived and aren't yet scored."""
    _ensure_file()
    due = []
    today = datetime.now().date()
    with open(config.JOURNAL_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["was_correct"] == "" and row["check_by_date"]:
                check_date = datetime.strptime(row["check_by_date"], "%Y-%m-%d").date()
                if check_date <= today:
                    due.append(row)
    return due


def score_entry(entry_date, was_correct: bool, outcome_note: str = ""):
    """Update a past entry with the actual outcome, once you've checked it."""
    _ensure_file()
    rows = []
    with open(config.JOURNAL_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    updated = False
    for row in rows:
        if row["entry_date"] == entry_date and row["was_correct"] == "":
            row["was_correct"] = "yes" if was_correct else "no"
            row["outcome"] = outcome_note
            updated = True

    if updated:
        with open(config.JOURNAL_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Scored entry from {entry_date}.")
    else:
        print(f"No unscored entry found for {entry_date}.")


def hit_rate_summary():
    """Compute your running accuracy — the actual senior-skill scoreboard."""
    _ensure_file()
    with open(config.JOURNAL_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["was_correct"] in ("yes", "no")]

    if not rows:
        return "No scored entries yet. Log a thesis, then come back after check_by_date."

    correct = sum(1 for r in rows if r["was_correct"] == "yes")
    total = len(rows)
    by_factor = {}
    for r in rows:
        f = r["dominant_factor"]
        by_factor.setdefault(f, [0, 0])
        by_factor[f][1] += 1
        if r["was_correct"] == "yes":
            by_factor[f][0] += 1

    lines = [f"Overall hit rate: {correct}/{total} ({correct/total:.0%})"]
    for factor, (c, t) in by_factor.items():
        lines.append(f"  {factor}: {c}/{t} ({c/t:.0%})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in Socratic prompts — the actual "senior perspective" questions,
# surfaced automatically alongside the data every day.
# ---------------------------------------------------------------------------

DAILY_REFLECTION_PROMPTS = [
    "If I had to bet which ONE factor explains 70% of today's move, which "
    "would it be — and what single data point would prove me wrong?",
    "Is today's dollar move driven by rate differentials, or by risk-off "
    "flows? (Both push DXY up, but imply opposite things for gold.)",
    "Did the correlation between gold and real yields hold today, or "
    "diverge? If it diverged, what's the competing story?",
    "Is any geopolitical headline today already priced in, or genuinely new "
    "information the market hasn't digested yet?",
    "Has anything in the 2yr yield or curve shifted the near-term Fed path "
    "in a way that changes the carry cost of holding gold?",
]

WEEKLY_THESIS_PROMPTS = [
    "Which factor (real yields / DXY / central bank buying / geopolitical "
    "risk / curve-recession signal) do you think will dominate gold over "
    "the next 1-2 weeks?",
    "Write ONE specific, checkable prediction that would be FALSE if you're "
    "wrong (e.g. '10y real yield falls below X and gold breaks $Y' — not "
    "'gold will be volatile').",
    "What would the senior desk view push back on in your thesis right now?",
]


def print_reflection_prompts():
    print("\n--- Daily reflection (answer these yourself, not with the data) ---")
    for p in DAILY_REFLECTION_PROMPTS:
        print(f"  - {p}")


def print_weekly_prompts():
    print("\n--- Weekly thesis prompts ---")
    for p in WEEKLY_THESIS_PROMPTS:
        print(f"  - {p}")
