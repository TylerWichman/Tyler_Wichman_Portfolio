-- Rebuilds reference + fact tables from staged data and reference CSVs. Rerunnable.
DELETE FROM odds; DELETE FROM matches; DELETE FROM team_aliases;
DELETE FROM teams; DELETE FROM cities; DELETE FROM leagues;

-- 1. Reference tables
INSERT INTO leagues
SELECT TRIM(league_id), TRIM(league_name), TRIM(country), TRIM(data_source)
FROM read_csv('leagues_seed.csv', header = true, all_varchar = true);

INSERT INTO cities
SELECT replace(TRIM(city), ' ', '_') || '_' || TRIM(iso2), TRIM(city), TRIM(iso2),
       CAST(latitude AS DOUBLE), CAST(longitude AS DOUBLE)
FROM read_csv('ref_cities_coords.csv', header = true, all_varchar = true);

-- 2. Normalize both sources
CREATE OR REPLACE TABLE stg_all_raw AS
SELECT 'main' AS src,
       TRIM(league_code) AS league_id,
       '20' || LEFT(season_code, 2) || '/20' || RIGHT(season_code, 2) AS season_label,
       2000 + TRY_CAST(LEFT(season_code, 2) AS INTEGER) AS start_year,
       CASE LENGTH(TRIM("Date"))
            WHEN 8 THEN TRY_STRPTIME(TRIM("Date"), '%d/%m/%y')
            ELSE        TRY_STRPTIME(TRIM("Date"), '%d/%m/%Y') END::DATE AS match_date,
       COALESCE(NULLIF(TRIM(HomeTeam), ''), NULLIF(TRIM("HT"), '')) AS home_raw,
       COALESCE(NULLIF(TRIM(AwayTeam), ''), NULLIF(TRIM("AT"), '')) AS away_raw,
       TRY_CAST(TRY_CAST(TRIM(FTHG) AS DOUBLE) AS INTEGER) AS home_goals,
       TRY_CAST(TRY_CAST(TRIM(FTAG) AS DOUBLE) AS INTEGER) AS away_goals,
       TRY_CAST(TRIM(PSCH)   AS DOUBLE) AS pinnacle_close_h,
       TRY_CAST(TRIM(PSCD)   AS DOUBLE) AS pinnacle_close_d,
       TRY_CAST(TRIM(PSCA)   AS DOUBLE) AS pinnacle_close_a,
       TRY_CAST(TRIM(AvgCH)  AS DOUBLE) AS avg_close_h,
       TRY_CAST(TRIM(AvgCD)  AS DOUBLE) AS avg_close_d,
       TRY_CAST(TRIM(AvgCA)  AS DOUBLE) AS avg_close_a,
       TRY_CAST(TRIM(B365CH) AS DOUBLE) AS b365_close_h,
       TRY_CAST(TRIM(B365CD) AS DOUBLE) AS b365_close_d,
       TRY_CAST(TRIM(B365CA) AS DOUBLE) AS b365_close_a
FROM stg_main
UNION ALL
SELECT 'extra', TRIM(league_code), TRIM(Season),
       TRY_CAST(LEFT(TRIM(Season), 4) AS INTEGER),
       TRY_STRPTIME(TRIM("Date"), '%d/%m/%Y')::DATE,
       NULLIF(TRIM(Home), ''), NULLIF(TRIM(Away), ''),
       TRY_CAST(TRY_CAST(TRIM(HG) AS DOUBLE) AS INTEGER),
       TRY_CAST(TRY_CAST(TRIM(AG) AS DOUBLE) AS INTEGER),
       TRY_CAST(TRIM(PSCH) AS DOUBLE), TRY_CAST(TRIM(PSCD) AS DOUBLE), TRY_CAST(TRIM(PSCA) AS DOUBLE),
       TRY_CAST(TRIM(AvgCH) AS DOUBLE), TRY_CAST(TRIM(AvgCD) AS DOUBLE), TRY_CAST(TRIM(AvgCA) AS DOUBLE),
       TRY_CAST(TRIM(B365CH) AS DOUBLE), TRY_CAST(TRIM(B365CD) AS DOUBLE), TRY_CAST(TRIM(B365CA) AS DOUBLE)
FROM stg_extra;

CREATE OR REPLACE TABLE stg_all AS
SELECT md5(concat_ws('|', league_id, match_date, home_raw, away_raw)) AS match_id, *
FROM stg_all_raw
WHERE match_date IS NOT NULL AND start_year IS NOT NULL
  AND home_raw IS NOT NULL AND away_raw IS NOT NULL
  AND home_goals IS NOT NULL AND away_goals IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY league_id, match_date, home_raw, away_raw
                           ORDER BY src, season_label) = 1;

-- 3. Teams and aliases (spelling variants only; renamed clubs stay separate)
CREATE OR REPLACE TEMP TABLE ref_map AS
WITH ref AS (
    SELECT TRIM(league_code) AS league_code, TRIM(team) AS team,
           TRIM(city) AS city, TRIM(iso2) AS iso2
    FROM read_csv('ref_team_cities.csv', header = true, all_varchar = true)
),
alias_merge(league_code, alias, canonical) AS (VALUES
    ('ARG', 'Colon Santa FE',   'Colon Santa Fe'),
    ('N1',  'Roda',             'Roda JC'),
    ('SWE', 'Oster',            'Osters'),
    ('G1',  'Yiannina',         'Giannina'),
    ('G1',  'Kalithea',         'Kallithea'),
    ('G1',  'Athens Kallithea', 'Kallithea'),
    ('POL', 'Ruch',             'Ruch Chorzow'),
    ('NOR', 'Ham-Kam',          'HamKam')
)
SELECT r.league_code, r.team,
       COALESCE(a.canonical, r.team) AS canonical,
       r.league_code || ':' || COALESCE(a.canonical, r.team) AS team_id,
       replace(r.city, ' ', '_') || '_' || r.iso2 AS city_id
FROM ref r
LEFT JOIN alias_merge a ON a.league_code = r.league_code AND a.alias = r.team;

INSERT INTO teams
SELECT DISTINCT team_id, league_code, canonical, city_id FROM ref_map;

INSERT INTO team_aliases
SELECT league_code, team, team_id FROM ref_map;

-- 4. Matches
INSERT INTO matches
SELECT s.match_id, s.league_id, s.season_label, s.start_year, s.match_date,
       ah.team_id, aa.team_id, s.home_goals, s.away_goals
FROM stg_all s
JOIN team_aliases ah ON ah.league_id = s.league_id AND ah.alias = s.home_raw
JOIN team_aliases aa ON aa.league_id = s.league_id AND aa.alias = s.away_raw;

-- 5. Closing odds, then drop impossible overrounds
INSERT INTO odds SELECT match_id, 'pinnacle_close', pinnacle_close_h, pinnacle_close_d, pinnacle_close_a
FROM stg_all WHERE pinnacle_close_h > 1 AND pinnacle_close_d > 1 AND pinnacle_close_a > 1;
INSERT INTO odds SELECT match_id, 'avg_close', avg_close_h, avg_close_d, avg_close_a
FROM stg_all WHERE avg_close_h > 1 AND avg_close_d > 1 AND avg_close_a > 1;
INSERT INTO odds SELECT match_id, 'b365_close', b365_close_h, b365_close_d, b365_close_a
FROM stg_all WHERE b365_close_h > 1 AND b365_close_d > 1 AND b365_close_a > 1;

DELETE FROM odds
WHERE 1/odds_home + 1/odds_draw + 1/odds_away NOT BETWEEN 1.0 AND 1.25;
