CREATE OR REPLACE VIEW match_facts AS
WITH o AS (
    SELECT m.match_id,
           COALESCE(p.odds_home, a.odds_home) AS oh,
           COALESCE(p.odds_draw, a.odds_draw) AS od,
           COALESCE(p.odds_away, a.odds_away) AS oa,
           CASE WHEN p.match_id IS NOT NULL THEN 'pinnacle_close'
                WHEN a.match_id IS NOT NULL THEN 'avg_close' END AS odds_source
    FROM matches m
    LEFT JOIN odds p ON p.match_id = m.match_id AND p.bookmaker = 'pinnacle_close'
    LEFT JOIN odds a ON a.match_id = m.match_id AND a.bookmaker = 'avg_close'
)
SELECT
    m.match_id, m.league_id, l.country AS league_country, l.data_source,
    m.season_label, m.start_year, m.match_date,
    m.home_team_id, th.canonical_name AS home_team, ch.city_name AS home_city, ch.country AS home_iso2,
    m.away_team_id, ta.canonical_name AS away_team, ca.city_name AS away_city, ca.country AS away_iso2,
    m.home_goals, m.away_goals,
    m.home_goals - m.away_goals AS goal_diff,
    CASE WHEN m.home_goals > m.away_goals THEN 'H'
         WHEN m.home_goals = m.away_goals THEN 'D' ELSE 'A' END AS result,
    -- Haversine distance, Earth radius 6371 km
    2 * 6371 * ASIN(SQRT(
        POWER(SIN(RADIANS(ca.latitude - ch.latitude) / 2), 2)
      + COS(RADIANS(ch.latitude)) * COS(RADIANS(ca.latitude))
      * POWER(SIN(RADIANS(ca.longitude - ch.longitude) / 2), 2)
    )) AS travel_km,
    th.city_id = ta.city_id AS same_city,
    ch.country <> ca.country AS cross_border,
    o.odds_source,
    -- Fair probabilities (margin removed by normalizing the implied probabilities)
    (1/o.oh) / (1/o.oh + 1/o.od + 1/o.oa) AS prob_home_fair,
    (1/o.od) / (1/o.oh + 1/o.od + 1/o.oa) AS prob_draw_fair,
    (1/o.oa) / (1/o.oh + 1/o.od + 1/o.oa) AS prob_away_fair
FROM matches m
JOIN leagues l  ON l.league_id = m.league_id
JOIN teams th   ON th.team_id  = m.home_team_id
JOIN cities ch  ON ch.city_id  = th.city_id
JOIN teams ta   ON ta.team_id  = m.away_team_id
JOIN cities ca  ON ca.city_id  = ta.city_id
LEFT JOIN o     ON o.match_id  = m.match_id;
