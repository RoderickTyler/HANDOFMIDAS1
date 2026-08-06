# Gold Macro System Plus Calendar Extension and COT v3

A free, no-paid-API system that pulls the four indicators we discussed —
real yields, DXY, central bank reserves, geopolitical risk — into one daily
briefing, plus a built-in habit-forming journal to help you build the
"which factor is dominant right now" judgment that separates a junior
analyst from a senior one.

Two ways to use it:
- **CLI** (`main.py`) — the original terminal briefing, see "Daily use" below.
- **Web dashboard** (`streamlit_app.py`) — the same analysis as an
  interactive site with charts, tabs, and a clickable journal. See
  **[Web dashboard](#web-dashboard)** below to run it locally or deploy it
  free on Streamlit Community Cloud.

## What it does

Every time you run it, you get:

1. **A data table**: gold, DXY, VIX, 2Y/10Y nominal yields, 10Y real (TIPS)
   yield, real yield trend (5d/20d), 2s10s curve spread, and rolling
   30-day correlations of gold vs DXY and gold vs real yields.
2. **Divergence flags**: automatic alerts when the "textbook" relationship
   (gold should move opposite DXY and opposite real yields) has broken
   down — this is exactly where the senior-analyst judgment call matters.
3. **Economic calendar**: upcoming CPI, NFP (jobs report), GDP, and ISM
   Manufacturing PMI release dates for the next 2 weeks, pulled from FRED's
   own release calendar — plus FOMC meeting/statement dates. Routine Fed
   speakers are intentionally excluded; only FOMC meetings and (best-effort)
   Chair Powell speeches are flagged.
4. **Daily reflection prompts**: five questions printed every run, meant
   for *you* to answer, not the data — this is the Socratic layer.
4. **A weekly thesis journal**: log a falsifiable prediction each week
   ("gold breaks $2,450 if real yields fall below 1.8%" — not "gold will
   be volatile"), and the system flags it for review later and tracks
   your hit rate over time, broken out by which factor you bet on.

## Setup (one-time, ~5 minutes)

1. Install Python 3.9+ if you don't have it.
2. In this folder, install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Get a **free** FRED API key (no cost, no credit card):
   https://fred.stlouisfed.org/docs/api/api_key.html
4. Create a file named `.env` in this same folder with:
   ```
   FRED_API_KEY=your_key_here
   ```
   (Everything else — yfinance for gold/DXY/VIX, IMF's SDMX API for
   central bank reserves, CFTC's COT data, and the Iacoviello
   Geopolitical Risk Index — needs no key at all.)

5. **Optional:** for a real spot XAUUSD quote (to compare against the
   GC=F futures price the rest of the system uses), sign up free at
   https://www.goldapi.io (no card required), then add to the same
   `.env` file:
   ```
   GOLDAPI_KEY=your_key_here
   ```
   If you skip this, the briefing still works exactly the same — it just
   shows "n/a" for the spot row instead of a live quote.

## Daily use

Run the briefing:
```
python main.py
```

Log this week's thesis (does this interactively, asks you the Socratic
questions, then stores your answer):
```
python main.py --log-thesis
```

Check your running accuracy scoreboard:
```
python main.py --hit-rate
```

Pull a longer lookback window (default is 1 year):
```
python main.py --period 2y
```

## Web dashboard

The same system, as a browser dashboard instead of terminal text — tabs for
Overview, Regime (HMM), COT Positioning, Factor Attribution, Econ Calendar,
Central Bank Reserves, and the thesis Journal (log/score from the browser).

### Run it locally

```
pip install -r requirements.txt
cp .env.example .env        # then fill in FRED_API_KEY (and GOLDAPI_KEY if you have it)
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

### Deploy it for free (Streamlit Community Cloud)

1. Push this repo to GitHub (see below if you haven't yet).
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**.
3. Pick this repo, branch `main`, and set **Main file path** to
   `streamlit_app.py`.
4. Before or after the first deploy, open **Settings -> Secrets** on the app
   and paste:
   ```
   FRED_API_KEY = "your_key_here"
   GOLDAPI_KEY = "your_key_here"
   ```
   (`GOLDAPI_KEY` is optional — leave it out and the spot-price row just
   shows n/a.) `config.py` reads secrets from the environment first, then
   falls back to `st.secrets`, so this works without any code changes.
5. Deploy. You'll get a public URL like `https://<your-app>.streamlit.app`.

The dashboard caches each data source for 15 minutes (`st.cache_data`, see
`CACHE_TTL` in `streamlit_app.py`) so normal page views/reruns don't
re-hit Yahoo Finance/FRED/CFTC/IMF on every click — use the **Refresh data
now** button in the sidebar to force a re-fetch.

### Push this repo to GitHub yourself

From this project folder:

```
git init
git add .
git commit -m "Gold macro system: CLI + web dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

(Create the empty repo on GitHub first — github.com/new — without a
README/license so there's no merge conflict on first push.)

Note: `.env`, `.streamlit/secrets.toml`, and everything in `data/` are
git-ignored on purpose — they hold your API keys and personal journal, and
should never be committed. `.env.example` and
`.streamlit/secrets.toml.example` show the expected format.

## Reviewing past theses

The daily briefing automatically shows you any logged thesis whose
"check by" date has arrived. Once you've checked what actually happened,
score it from the Python shell or a script:

```python
import journal
journal.score_entry("2026-07-14", True, "Gold broke 2450 as predicted when real yields fell")
```

Over weeks and months, `python main.py --hit-rate` becomes your actual
track record, broken down by which factor you tend to call correctly
(real yields vs DXY vs central bank buying vs geopolitical risk vs
curve/recession signal) — this is the concrete version of the "senior
judgment" skill: knowing which of your own instincts to trust.

## Data sources (all free)

| Data | Source | Key needed? |
|---|---|---|
| Gold, DXY, VIX | Yahoo Finance (`yfinance`) | No |
| 2Y/10Y nominal yields, 10Y real (TIPS) yield, curve spread | FRED (`fredapi`) | Yes, free |
| Central bank gold reserves | IMF SDMX public API | No |
| Geopolitical Risk Index | Caldara & Iacoviello (Fed economists), matteoiacoviello.com | No |

## Notes / known limitations

- **IMF reserves data** updates quarterly and its API structure has
  changed before — if `get_imf_gold_reserves()` returns nothing, this
  won't affect your daily indicators; check
  https://www.gold.org/goldhub/data/gold-reserves-by-country manually.
- **GPR index** is a monthly series, so day-to-day it won't move; treat
  it as a slow-moving confirmation of risk regime, not a daily signal.
- The rolling correlation windows (30-day, 90-day) and the divergence
  threshold (0.2) are starting defaults in `config.py` — worth tuning
  once you've watched the output for a few weeks and get a feel for
  what counts as a "real" divergence versus noise.
- `yfinance` occasionally rate-limits or changes its response format
  without notice since it scrapes Yahoo Finance rather than using an
  official API — if it breaks, check for a `pip install --upgrade
  yfinance` first.

## File overview

```
config.py       -> API keys, tickers, thresholds
fetch_data.py   -> pulls all raw data from the four free sources
analysis.py     -> merges data, computes real yield trend, curve spread,
                   rolling correlations, and divergence flags
journal.py      -> weekly thesis logging + hit-rate scoreboard +
                   the built-in Socratic prompts
report.py       -> formats everything into the daily briefing text
main.py         -> CLI entry point
streamlit_app.py -> web dashboard entry point (`streamlit run streamlit_app.py`)
data/           -> price_history.csv (cache) and thesis_journal.csv
                   (your prediction log) get created here on first run
```
