# Fantasy Trade Calculator

A dynasty/redraft fantasy football trade calculator that trains supervised gradient-boosting
models to predict market consensus value (FantasyCalc, blended with KeepTradeCut) from player
fundamentals, producing one unified value board per format that puts every player and draft pick
on the same directly-comparable scale — plus a public Streamlit website (trade calculator,
filterable rankings, Sleeper-synced league dashboard) built on top of it.

## Methodology & scope

**Consensus value is the training target, not a benchmark to beat.** These models are trained to
reflect the fantasy trade market as closely as possible — agreement with FantasyCalc/KeepTradeCut
consensus is the design goal, not an independent finding. Where a player's model-predicted value
diverges from their current market price, that divergence is reported as a residual — a candidate
worth investigating (fundamentals the model weighs differently than the market currently prices
them), never a claim that the model is right and the market is wrong.

**Six boards:** three formats (Superflex dynasty, 1QB dynasty, redraft) each fit at two league
sizes (10-team and 12-team). Dynasty boards include the next three years of rookie draft picks
(2027–2029, rounds 1–3, Early/Mid/Late tiers) priced from real market data where available and a
constructed year-discount/class-strength/format-multiplier formula elsewhere; redraft boards have
no pick assets.

**Two consensus sources, blended:** FantasyCalc is primary (it's the only source with real
per-team-size pricing); KeepTradeCut is blended in at reduced weight to reduce single-source bias.

**Sample-weighted training (top-of-board tail compression fix):** `HistGradientBoostingRegressor`'s
default `min_samples_leaf=20`, combined with this project's ~379-row training pool, structurally
forces the top ~15-20 "elite tier" players — in any position, any format — into leaves blended with
lower-value neighbors, since every leaf must average at least 20 samples. That systematically
compresses predictions for the most valuable assets (first surfaced as superflex elite-QB
underprediction, then confirmed as a general effect across all positions). The fix: every training
row is weighted by its own percentile rank in that format's `consensus_value` distribution
(`weight = 1 + percentile**3`, position- and format-blind — no player, position, or format is
special-cased) combined with a lower `min_samples_leaf` (10, down from 20), so the loss function is
pushed to resolve the top of the distribution instead of averaging it away. Verified across all 6
formats with a multi-seed robustness check; Spearman/MAE are flat-to-improved everywhere, not traded
off. (Two earlier fixes were tried and honestly ruled out first: isolating QB into its own model made
predictions worse — too little data to resolve the tail at all — and an "entrenched starter" feature
didn't discriminate between players. A board-assembly-layer blend toward consensus for established SF
QBs was deployed as an interim fix and has since been fully replaced by this training-level approach,
since it also fixes the smaller RB/WR/TE compression the blend never touched.)

**Known exclusions:**
- Kickers, defenses (DST), and IDP are not modeled.
- TE premium scoring is not modeled — FantasyCalc's API doesn't expose TE-premium values.
- Rookie college-production features (dominator rating, breakout age) depend on the free-tier
  CollegeFootballData API, which has a hard monthly call quota. When that quota is exhausted, those
  features degrade gracefully to `NaN` and affected rookies are valued on draft capital, age, and
  (for RBs) combine speed score alone — the underlying model handles missing features natively, so
  this is a quality degradation for the affected rookies, not a failure.

## Tech Stack
Python, pandas, NumPy, scikit-learn (`HistGradientBoostingRegressor`), nflreadpy, requests,
Streamlit (web app), Sleeper API (live league sync)

## Key Takeaway
Framing the modeling problem as "predict the market" rather than "build an independent valuation
and penalize its drift from the market" resolves the central tension that made the prior version
of this project's rankings hard to trust — the model no longer has to choose between agreeing with
consensus and being useful. Divergence from consensus becomes a free byproduct (a residual) instead
of an error signal to chase, and the buy-low/sell-high candidates it surfaces are grounded in a
model that's already been validated against the real market it's trying to describe.

---

## Web app

Public-facing layer on top of the modeling pipeline above: a Streamlit app with a trade
calculator, a filterable rankings table, and a Sleeper-synced league dashboard. No visitor to the
live site ever triggers a FantasyCalc/KTC/CFBD pull — a weekly GitHub Actions cron job re-runs the
model and writes static JSON (`data/`), which the app reads. The only *live* per-visitor calls are
to Sleeper's free public API, and only when someone actively syncs a league.

### Setup — two manual steps only you can do

#### 1. GitHub Actions secret

The weekly snapshot job needs `CFBD_API_KEY` as a **repository secret**:
Settings → Secrets and variables → Actions → New repository secret → name
`CFBD_API_KEY`, value your CollegeFootballData API key (get a free one at
[collegefootballdata.com/key](https://collegefootballdata.com/key)).

Still worth setting before the first run: without it, `compute_rookie_college_
features()` detects the missing key up front, prints a clear warning, and
falls back to draft-capital-only rookie valuation for the whole run — it does
not crash `generate_snapshot.py` (this was a real gap, found during this
build and fixed; verified by unsetting the key post-import and confirming a
clean, correctly-shaped fallback result with no exception). A configured key
that's simply out of its monthly call quota degrades the same way. Either way,
board files still get written — setting the secret just gets you real rookie
college-production signal instead of the draft-capital-only fallback.

#### 2. Streamlit Community Cloud deployment

Go to [share.streamlit.io](https://share.streamlit.io), connect this GitHub
repo, and set:
- **Main file path:** `Fantasy_Trade_Calculator/app.py`
- **Requirements file:** `Fantasy_Trade_Calculator/requirements.txt`
  (the lean one — `streamlit`/`pandas`/`requests` only; the deployed app never
  re-runs the model, so it must NOT use `requirements-pipeline.txt`, which
  would needlessly install scikit-learn/nflreadpy/etc. on every deploy).

### Running locally

```
cd Fantasy_Trade_Calculator
streamlit run app.py
```

Reads directly from `data/*.json` — run `python generate_snapshot.py` first
(from this same directory, with `CFBD_API_KEY` set in your environment or a
`.env` file in this directory) if you want fresh data rather than whatever is
already checked in.

### Known limitations (v1)

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
