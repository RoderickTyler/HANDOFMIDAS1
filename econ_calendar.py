"""
econ_calendar.py
-----------------
Pulls upcoming economic release dates from FRED's own release calendar --
no new API/key needed, just the FRED_API_KEY you already have.

PRIMARY (always shown): CPI, NFP (Employment Situation), GDP, ISM
Manufacturing PMI.

SECONDARY (shown only if it's an FOMC meeting/statement date, or a Fed
Chair speech): everything else Fed-related is intentionally left out --
routine regional Fed speakers are noise for this purpose.

Honest limitation: FRED's calendar covers scheduled data RELEASES and FOMC
meeting dates -- it does NOT track individual speech schedules (no free
structured source does). The Chair-speech check below is a best-effort
scrape of the Fed's own speeches page and can break if the Fed changes
their site; if it returns nothing, check manually:
https://www.federalreserve.gov/newsevents/speeches.htm
"""

from datetime import datetime, timedelta

import requests

import config

FRED_BASE = "https://api.stlouisfed.org/fred"

# Primary releases we always want to see, matched by name against FRED's
# own release list (discovered dynamically, not hardcoded IDs -- FRED has
# renumbered releases before, so matching by name is more durable).
PRIMARY_RELEASES = {
    "CPI": {"exact": ["Consumer Price Index"], "contains": ["Consumer Price Index"]},
    "NFP (Employment Situation)": {"exact": ["Employment Situation"], "contains": ["Employment Situation"]},
    "GDP": {"exact": ["Gross Domestic Product"], "contains": ["Gross Domestic Product"]},
    "ISM Manufacturing PMI": {"exact": ["ISM Report on Business Manufacturing", "ISM Manufacturing"],
                               "contains": ["ISM", "Manufacturing"]},
}

# Secondary: only FOMC meeting/statement dates, not routine Fed speakers.
FOMC_RELEASE = {"exact": ["FOMC Press Release"], "contains": ["FOMC"]}


def _fred_get(endpoint, **params):
    if not config.FRED_API_KEY:
        return None
    params["api_key"] = config.FRED_API_KEY
    params["file_type"] = "json"
    try:
        resp = requests.get(f"{FRED_BASE}/{endpoint}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[warn] FRED calendar request failed ({endpoint}): {e}")
        return None


def _list_all_releases():
    data = _fred_get("releases", limit=1000)
    if not data:
        return []
    return data.get("releases", [])


def _find_release_id(releases, exact_names, contains_terms):
    """Prefer an exact name match; fall back to a release whose name
    contains ALL of the given terms (to avoid grabbing an unrelated
    sub-release, e.g. 'Consumer Price Index by ZIP Code')."""
    by_name = {r["name"]: r["id"] for r in releases}
    for name in exact_names:
        if name in by_name:
            return by_name[name], name

    for r in releases:
        name_lower = r["name"].lower()
        if all(term.lower() in name_lower for term in contains_terms):
            return r["id"], r["name"]

    return None, None


def _upcoming_dates_for_release(release_id, days_ahead=14):
    today = datetime.now().date()
    data = _fred_get(
        "release/dates",
        release_id=release_id,
        include_release_dates_with_no_data="true",
        sort_order="asc",
        realtime_start=today.isoformat(),
    )
    if not data:
        return []

    cutoff = today + timedelta(days=days_ahead)
    dates = []
    for d in data.get("release_dates", []):
        try:
            release_date = datetime.strptime(d["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        if today <= release_date <= cutoff:
            dates.append(release_date)
    return dates


def get_upcoming_calendar(days_ahead=14):
    """
    Returns a dict: {label: [dates]} for primary releases (CPI, NFP, GDP,
    ISM PMI) plus FOMC meeting dates, all within the next `days_ahead` days.
    """
    if not config.FRED_API_KEY:
        print("[warn] No FRED_API_KEY set -- skipping economic calendar.")
        return {}

    releases = _list_all_releases()
    if not releases:
        print("[warn] Could not fetch FRED's release list -- skipping economic calendar.")
        return {}

    results = {}

    for label, matchers in PRIMARY_RELEASES.items():
        release_id, matched_name = _find_release_id(releases, matchers["exact"], matchers["contains"])
        if release_id is None:
            print(f"[warn] Could not find a FRED release matching '{label}'.")
            continue
        dates = _upcoming_dates_for_release(release_id, days_ahead=days_ahead)
        dates = _sanity_filter(label, matched_name, dates, days_ahead)
        if dates:
            results[label] = dates

    fomc_id, fomc_name = _find_release_id(releases, FOMC_RELEASE["exact"], FOMC_RELEASE["contains"])
    if fomc_id is not None:
        dates = _upcoming_dates_for_release(fomc_id, days_ahead=days_ahead)
        dates = _sanity_filter("FOMC meeting/statement", fomc_name, dates, days_ahead)
        if dates:
            results["FOMC meeting/statement"] = dates

    return results


def _sanity_filter(label, matched_name, dates, days_ahead, max_plausible=4):
    """
    None of CPI/NFP/GDP/ISM/FOMC legitimately fire more than a handful of
    times in a 2-week window. If a "match" returns way more dates than
    that, it almost certainly grabbed the wrong FRED release (e.g. a daily
    interest-rate series instead of the actual meeting calendar) rather
    than genuinely reflecting reality. Refuse to display it rather than
    show something misleading, and print what actually got matched so it
    can be debugged/fixed rather than silently wrong.
    """
    if len(dates) > max_plausible:
        print(
            f"[warn] '{label}' matched FRED release '{matched_name}', which returned "
            f"{len(dates)} dates in a {days_ahead}-day window -- implausible for this "
            f"release type, so this is almost certainly the wrong release. Suppressing "
            f"it rather than showing misleading dates. Check manually if this matters: "
            f"https://www.federalreserve.gov/newsevents/calendar.htm"
        )
        return []
    return dates


# Current Fed Chair name, used for the speech-detection scrape below.
# IMPORTANT: update this when the Fed Chair changes (last updated: Chairman
# Warsh confirmed as of July 2026, per a July 29, 2026 FOMC statement search
# -- succeeded Chair Powell). We search for BOTH the current and prior name
# during any transition period so this doesn't silently go stale the same
# way it did before.
CURRENT_FED_CHAIR_SEARCH_TERMS = ["Chairman Warsh", "Chair Warsh", "Chair Powell"]


def get_upcoming_powell_speeches(days_ahead=14):
    """
    Best-effort scrape of the Fed's own speeches listing page for entries
    mentioning the Chair (see CURRENT_FED_CHAIR_SEARCH_TERMS above -- update
    that list, not this function, when the Fed Chair changes). This is
    fragile (HTML scrape, no structured API exists for this anywhere free)
    and may return nothing even when speeches exist. Treat an empty result
    as inconclusive, not "none scheduled" -- always check the source page
    directly for anything time-sensitive:
    https://www.federalreserve.gov/newsevents/speeches.htm
    """
    url = "https://www.federalreserve.gov/newsevents/speeches.htm"
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"[info] Could not check Fed speeches page (non-critical): {e}")
        return []

    # Very lightweight, tolerant scrape: look for the Chair's name near a
    # date-shaped string. This will break if the Fed redesigns their page;
    # that's expected and why this is explicitly best-effort.
    import re
    matches = []
    for name in CURRENT_FED_CHAIR_SEARCH_TERMS:
        pattern = r"(\d{1,2}/\d{1,2}/\d{4}).{0,200}?" + re.escape(name)
        for m in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
            matches.append(m.group(1))

    return sorted(set(matches))[:5]  # dedupe, cap; rough signal, not a full calendar
