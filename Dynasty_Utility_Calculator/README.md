# Fantasy Trade Calculator

A dynasty/redraft fantasy football trade calculator that trains supervised gradient-boosting
models to predict market consensus value (FantasyCalc, blended with KeepTradeCut) from player
fundamentals, producing one unified value board per format that puts every player and draft pick
on the same directly-comparable scale.

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

## Setup

Get a free CollegeFootballData API key at [collegefootballdata.com](https://collegefootballdata.com/key)
(email signup, ~1,000 calls/month free tier), then create a `.env` file in this directory:

```
CFBD_API_KEY=your_key_here
```

`.env` is gitignored — never commit it or hardcode the key in source.

## Tech Stack
Python, pandas, NumPy, scikit-learn (`HistGradientBoostingRegressor`), nflreadpy, requests

## Key Takeaway
Framing the modeling problem as "predict the market" rather than "build an independent valuation
and penalize its drift from the market" resolves the central tension that made the prior version
of this project's rankings hard to trust — the model no longer has to choose between agreeing with
consensus and being useful. Divergence from consensus becomes a free byproduct (a residual) instead
of an error signal to chase, and the buy-low/sell-high candidates it surfaces are grounded in a
model that's already been validated against the real market it's trying to describe.
