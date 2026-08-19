# Fantasy Trade Calculator — Web App

Public-facing layer on top of the modeling pipeline documented in
`../README.md`: a Streamlit app with a trade calculator, a filterable rankings
table, and a Sleeper-synced league dashboard. No visitor to the live site ever
triggers a FantasyCalc/KTC/CFBD pull — a weekly GitHub Actions cron job
re-runs the model and writes static JSON (`data/`), which the app reads. The
only *live* per-visitor calls are to Sleeper's free public API, and only when
someone actively syncs a league.

## Setup — two manual steps only you can do

### 1. GitHub Actions secret

The weekly snapshot job needs `CFBD_API_KEY` as a **repository secret**:
Settings → Secrets and variables → Actions → New repository secret → name
`CFBD_API_KEY`, value your CollegeFootballData API key.

Still worth setting before the first run: without it, `compute_rookie_college_
features()` now detects the missing key up front, prints a clear warning, and
falls back to draft-capital-only rookie valuation for the whole run — it no
longer crashes `generate_snapshot.py` (this was a real gap, found during this
build and fixed; verified by unsetting the key post-import and confirming a
clean, correctly-shaped fallback result with no exception). A configured key
that's simply out of its monthly call quota degrades the same way. Either way,
board files still get written — setting the secret just gets you real rookie
college-production signal instead of the draft-capital-only fallback.

### 2. Streamlit Community Cloud deployment

Go to [share.streamlit.io](https://share.streamlit.io), connect this GitHub
repo, and set:
- **Main file path:** `Dynasty_Utility_Calculator/webapp/app.py`
- **Requirements file:** `Dynasty_Utility_Calculator/webapp/requirements.txt`
  (the lean one — `streamlit`/`pandas`/`requests` only; the deployed app never
  re-runs the model, so it must NOT use `requirements-pipeline.txt`, which
  would needlessly install scikit-learn/nflreadpy/etc. on every deploy).

## Running locally

```
cd Dynasty_Utility_Calculator/webapp
streamlit run app.py
```

Reads directly from `data/*.json` — run `python generate_snapshot.py` first
(from this directory, with `CFBD_API_KEY` set in your environment or a
`.env` file one level up) if you want fresh data rather than whatever is
already checked in.

## Known limitations (v1)

- **`sleeper_id` coverage: 92.8%** of assets — some players/picks won't be
  matchable to a synced Sleeper roster. **`team` coverage: 93.3%.**
- **CFBD rookie college-production data is currently near 0% coverage** — the
  free tier's monthly call quota is exhausted as of this build; it resets
  monthly. Both an exhausted quota and a missing key degrade gracefully to
  draft-capital-only rookie valuation in the meantime.
- **Non-10/12-team leagues** get rounded to the nearest supported team size
  for valuation purposes, with a visible note in the UI — this is an
  approximation, not an exact fit.
- **Tab 3b ("Five targets") has no team-need matching** — it's a league-wide
  ranking by positive model-vs-market divergence only, explicitly labeled as
  such in the UI. Matching targets to your own team's positional needs is a
  planned v2 enhancement, not built here.
- Zero-CFBD-data rookies (`college_seasons_found == 0`) are excluded from
  residual-sorted views (Tab 2's "biggest buy-lows" sort, Tab 3b) for the same
  reason they're excluded from the notebook's divergence tables — until real
  college-production data is available for them, their divergence is a
  predictable artifact of missing features, not a genuine signal.
