CREATE OR REPLACE VIEW v_distance_bands AS
WITH b AS (
    SELECT *,
           goal_diff - AVG(goal_diff) OVER (PARTITION BY league_id, season_label) AS gd_vs_league,
           CASE WHEN same_city        THEN '0 same city'
                WHEN travel_km < 100  THEN '1 <100 km'
                WHEN travel_km < 300  THEN '2 100-300'
                WHEN travel_km < 600  THEN '3 300-600'
                WHEN travel_km < 1200 THEN '4 600-1200'
                WHEN travel_km < 2500 THEN '5 1200-2500'
                ELSE                       '6 2500+' END AS band
    FROM match_facts
)
SELECT band, COUNT(*) AS n,
       AVG((result='H')::INT) AS home_win,
       AVG(gd_vs_league) AS gd_vs_league,
       STDDEV(gd_vs_league) / SQRT(COUNT(*)) AS se_gd,
       AVG(prob_home_fair) AS mkt_home_prob,
       AVG((result='H')::INT) FILTER (WHERE prob_home_fair IS NOT NULL) - AVG(prob_home_fair) AS home_win_minus_mkt
FROM b GROUP BY band;

CREATE OR REPLACE VIEW v_league_hfa AS
WITH t AS (
    SELECT *, match_date BETWEEN DATE '2020-03-15' AND DATE '2021-06-30' AS is_covid
    FROM match_facts WHERE start_year >= 2012
)
SELECT league_id, league_country, data_source,
       AVG(travel_km) FILTER (WHERE NOT is_covid)                   AS avg_km,
       COUNT(*) FILTER (WHERE NOT is_covid)                         AS n_normal,
       AVG(goal_diff) FILTER (WHERE NOT is_covid)                   AS hfa_normal,
       COUNT(*) FILTER (WHERE is_covid)                             AS n_covid,
       AVG(goal_diff) FILTER (WHERE is_covid)                       AS hfa_covid,
       AVG(goal_diff) FILTER (WHERE NOT is_covid)
         - AVG(goal_diff) FILTER (WHERE is_covid)                   AS covid_drop,
       STDDEV(goal_diff) FILTER (WHERE is_covid)
         / SQRT(COUNT(*) FILTER (WHERE is_covid))                   AS se_covid
FROM t GROUP BY ALL;

CREATE OR REPLACE VIEW v_market_by_period AS
SELECT CASE WHEN match_date <  DATE '2020-03-15' THEN '0 pre-COVID'
            WHEN match_date <  DATE '2020-09-01' THEN '1 Mar-Aug 2020'
            WHEN match_date <  DATE '2021-01-01' THEN '2 Sep-Dec 2020'
            WHEN match_date <= DATE '2021-06-30' THEN '3 Jan-Jun 2021'
            ELSE '4 post-COVID' END AS period,
       COUNT(*) AS n_odds,
       AVG((result='H')::INT) AS home_win,
       AVG(prob_home_fair) AS mkt_home_prob,
       AVG((result='H')::INT) - AVG(prob_home_fair) AS win_minus_mkt,
       1.96 * SQRT(AVG(prob_home_fair * (1 - prob_home_fair)) / COUNT(*)) AS ci95
FROM match_facts
WHERE start_year >= 2012 AND prob_home_fair IS NOT NULL
GROUP BY 1;
