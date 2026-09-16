# Profiling Notes

## Main Leagues (stg_main)
- 87,315 rows across 11 leagues. Row counts look healthy (E0 highest at
  9,921; SC0 lowest at 5,912) - no evidence of a download gap.
- HomeTeam is NULL in 1,059 rows (1.21%) - blank trailing rows in some
  source files. Dropped naturally by `WHERE HomeTeam IS NOT NULL` in
  normalization.
- Auto-named columns (column27, column28, column29, column121) are
  100% null - artifacts of malformed rows in a small number of source
  files. Confirmed no real data lives in them.

## Extra Leagues (stg_extra)
- 63,185 rows across 16 leagues, all present.
- Season column has two formats: plain year and split year "YYYY/YYYY".
  ARG and JPN show inconsistent formatting even within their own history.
  CHN has shorter coverage (starts 2014).
- Odds columns: B365CH/CD/CA are 92.49% null; PSCH/CD/CA are only ~6%
  null. Normalization uses PSCH/PSCD/PSCA (Pinnacle) as the primary
  odds source.

## Phase 5 pivot (city/coordinate mapping)
- Attempted: fuzzy-matching Wikidata SPARQL results against all 968
  historical team names - abandoned (query timeouts at scale, and club
  names are poorly indexed as searchable Wikidata entities).
- Attempted: Nominatim (OpenStreetMap) free-text geocoding of all 968
  raw team names - abandoned. Result: only ~22% resolved even with
  country constraints, because most football clubs are not tagged as
  searchable places in OSM's index. Several "resolved" matches were
  also geographically wrong (e.g. a Belgian club matched to a Brazilian
  village), confirming the approach is fundamentally unreliable for
  this data, not just under-tuned.
- SCOPE DECISION: switched to only current top-flight teams per league,
  sourced from each league's current-season Wikipedia article, which
  lists Club + City directly in a table. This is a much smaller list
  (~20 clubs per league x 27 leagues) and is accurate by construction
  rather than by geocoding. Historical/relegated teams outside the
  current top flight are excluded from the distance-based analyses
  (though still included in simple win/loss home-edge counts, which
  don't require city data).
