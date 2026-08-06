"""
reserves_utils.py
------------------
Shared logic for picking a single, consistent institutional sector per
country from IMF's IRFCL data (which reports multiple sectors -- Monetary
Authorities, Central Government, etc. -- per country per period).

This was previously duplicated in two places inside main.py (the trend
summary and the historical pivot table); factor_attribution.py needs the
same logic a third time, so it's centralized here to avoid the three
places drifting out of sync with each other.
"""

import pandas as pd


def get_columns(df):
    """Identify the country/period/value/sector columns regardless of
    exact naming, since IMF/imfp column names can vary."""
    country_col = next((c for c in df.columns if c.lower() in ("country", "ref_area", "area")), None)
    period_col = next((c for c in df.columns if "period" in c.lower()), None)
    value_col = next((c for c in df.columns if "value" in c.lower() or c.lower() == "obs_value"), None)
    sector_col = next((c for c in df.columns if c == "sector_description"), None) \
        or next((c for c in df.columns if "sector" in c.lower()), None)
    return country_col, period_col, value_col, sector_col


def select_single_sector_per_country(reserves_df):
    """
    Returns a filtered copy of reserves_df where each country contributes
    rows from exactly ONE institutional sector: Monetary Authorities
    preferred (the standard reserve-holding institution), then Total, then
    whichever sector has the largest latest value as a last resort.

    Never mixes sectors within a single country's time series -- that was
    the root cause of the implausible month-to-month swings we found
    earlier (different sub-entities getting compared as if they were the
    same continuous series).
    """
    country_col, period_col, value_col, sector_col = get_columns(reserves_df)
    if not all([country_col, value_col]) or not sector_col:
        return reserves_df.copy()

    filtered_rows = []
    for country, group in reserves_df.groupby(country_col):
        available = group[sector_col].unique()
        if len(available) == 1:
            filtered_rows.append(group)
            continue

        monetary_auth = [s for s in available if "monetary authorit" in str(s).lower()]
        total_sector = [s for s in available if "total" in str(s).lower()]

        if monetary_auth:
            chosen = monetary_auth[0]
        elif total_sector:
            chosen = total_sector[0]
        else:
            latest_by_sector = group.groupby(sector_col)[value_col].last()
            chosen = latest_by_sector.idxmax()

        filtered_rows.append(group[group[sector_col] == chosen])

    return pd.concat(filtered_rows) if filtered_rows else reserves_df.copy()
