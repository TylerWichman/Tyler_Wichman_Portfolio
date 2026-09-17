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
- Some HomeTeam/Home values have trailing whitespace (confirmed in N1:
  'Ajax ', 'Feyenoord ', 'Groningen ', 'Heracles ', 'Roda ', 'Utrecht ',
  'Vitesse ', 'Willem II ' all exist as separate values from their
  trimmed counterparts). ALL team-name joins in Phase 7 normalization
  must use TRIM(HomeTeam)/TRIM(Home), never raw string equality.

## Extra Leagues (stg_extra)
- 63,185 rows across 16 leagues, all present.
- Season column has two formats: plain year and split year "YYYY/YYYY".
  ARG and JPN show inconsistent formatting even within their own history.
  CHN has shorter coverage (starts 2014).
- Odds columns: B365CH/CD/CA are 92.49% null; PSCH/CD/CA are only ~6%
  null. Normalization uses PSCH/PSCD/PSCA (Pinnacle) as the primary
  odds source, not B365.

## Phase 5: team -> city mapping (COMPLETE)
- Abandoned approaches (documented for the record, not repeated):
  Wikidata SPARQL (query timeouts at scale, poor coverage of clubs as
  searchable entities); Nominatim free-text geocoding of raw club names
  (only ~22% resolved even with country constraints, and several
  "resolved" matches were geographically wrong, e.g. a Belgian club
  matched to a Brazilian village) - club names are not well-indexed as
  places, so geocoding them directly does not work.
- WORKING approach: scoped to each league's CURRENT top-flight roster
  only (not full history). For each of the 27 leagues, pulled the
  current-season Wikipedia article's Club/Location table, then
  fuzzy-matched (rapidfuzz) each Wikipedia club name against the
  actual raw team names in stg_main/stg_extra (both are real club
  names in different spellings - a much easier matching problem than
  geocoding a bare name as if it were a place).
- Automated pipeline: build_team_cities.py (searches Wikipedia, parses
  the location table, fuzzy-matches against staged data, saves
  incrementally so a crash never loses more than one league).
- Manual overrides: ref_team_cities_manual.csv, used for leagues/teams
  the automated pipeline couldn't parse (I1, N1, SC0, D1, E0 done fully
  by hand before the automation existed; IRL/JPN/SWZ/USA needed manual
  fixing after automation partially failed; ~30 additional diacritic/
  abbreviation fixes across ARG, POL, ROU, SWE, T1, B1, P1, BRA, DNK,
  MEX, FIN, SP1 after fuzzy-match scores came in below the 80 threshold).
- Final merge: consolidate_cities.py combines ref_team_cities_manual.csv
  (always wins on conflict) with auto-matched rows scoring >= 80,
  producing ref_team_cities.csv.
- FINAL RESULT: 463 team-city mappings across all 27 leagues, verified
  against raw staged team names (zero "bad mapping" / phantom entries
  remaining as of the last check).
- KNOWN LIMITATION (state this in the README): city mapping covers only
  each league's CURRENT top-flight roster, not full 2000-2026 history.
  This means: (a) many historical/relegated teams have no city and are
  excluded from travel-distance analyses (though still usable for
  simple win/loss home-edge counts, which need no city data at all);
  (b) this may bias the mapped sample toward larger, more financially
  stable clubs, since those are more likely to still be in the top
  flight today. Row-count-weighted coverage per league ranges from
  ~36% (ROU, T1) to ~99% (USA) of all historical matches - still tens
  of thousands of matches overall, more than sufficient for the
  planned analyses.
