# Profiling Notes

## Main leagues (stg_main)
- 90,252 rows from 297 files (11 leagues x 27 seasons).
- Early-2000s files have rows missing trailing empty columns. Staging uses
  `null_padding`; without it ~2,900 rows were silently dropped.
- Older files store team names in HT/AT instead of HomeTeam/AwayTeam
  (948 rows). Normalization coalesces the two.
- 111 fully blank rows (no date, no teams) are dropped in normalization.
- Season comes from the file name (`0405` -> 2004/2005).
- Dates mix dd/mm/yy and dd/mm/yyyy; parsed by string length.
- Team names have trailing whitespace in places ('Ajax ' vs 'Ajax'); all
  joins use TRIM().
- Auto-named columns (column24..column121) are empty artifacts.

## Extra leagues (stg_extra)
- 63,185 rows from 16 files. China starts in 2014, the rest in 2012.
- Season is either "YYYY" (calendar-year leagues) or "YYYY/YYYY";
  start_year = first four characters.

## Odds
- Only closing odds are loaded (PSC*, AvgC*, B365C*), so odds-based
  analysis covers 2012 onward.
- Pinnacle closing odds stop partway through 2025; the analysis falls
  back to average closing odds after that.
- 26 average-odds rows with impossible overrounds (<1.0 or >1.25) are
  removed, mostly April-May 2026.

## Known data notes
- One ARG 2013/14 match was played in January 2015 (rescheduled); kept.
- G1 2001/02 had 14 teams (182 matches), which is a complete season.
