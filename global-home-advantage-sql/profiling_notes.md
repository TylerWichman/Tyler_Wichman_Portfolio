# Profiling Notes

## Main Leagues (stg_main)
- 87,315 rows across 11 leagues. Row counts look healthy (E0 highest at
  9,921; SC0 lowest at 5,912) — no evidence of a download gap.
- HomeTeam is NULL in 1,059 rows (1.21%) — blank trailing rows in some
  source files. These are dropped naturally by the `WHERE HomeTeam IS
  NOT NULL` filter in normalization; no fix needed.
- Auto-named columns (column27, column28, column29, column121) are
  100% null — artifacts of malformed rows in a small number of source
  files (stray commas). Confirmed no real data lives in them.
- Team name variant check deferred to Phase 9 validation (staged-vs-
  loaded row count comparison), which will surface any alias mismatches
  with exact counts.

## Extra Leagues (stg_extra)
- 63,185 rows across 16 leagues, all present and accounted for.
- Season column has two formats: plain year (FIN, IRL, SWE, USA, BRA,
  NOR) and split year "YYYY/YYYY" (POL, RUS, AUT, DNK, MEX, ROU, SWZ).
  ARG and JPN show inconsistent formatting even within their own
  history. CHN has shorter coverage (starts 2014, not 2012).
  Normalization (Phase 7) will parse both formats into a single
  start_year integer.
- Odds columns: B365CH/CD/CA are 92.49% null; BFECH/CD/CA are 85%+
  null; PSCH/CD/CA are only ~6% null. Extra League normalization will
  use PSCH/PSCD/PSCA (Pinnacle closing odds) as the primary odds
  source, not B365.
- One stray non-numeric value ("x") found in AvgCH — TRY_CAST will
  convert this to NULL safely.
