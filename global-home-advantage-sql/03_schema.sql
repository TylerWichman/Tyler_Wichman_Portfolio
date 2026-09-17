DROP TABLE IF EXISTS odds;
DROP TABLE IF EXISTS matches;
DROP TABLE IF EXISTS team_aliases;
DROP TABLE IF EXISTS teams;
DROP TABLE IF EXISTS cities;
DROP TABLE IF EXISTS leagues;

CREATE TABLE leagues (
    league_id   VARCHAR PRIMARY KEY,
    league_name VARCHAR NOT NULL,
    country     VARCHAR NOT NULL,
    data_source VARCHAR NOT NULL CHECK (data_source IN ('main', 'extra'))
);

CREATE TABLE cities (
    city_id   VARCHAR PRIMARY KEY,      -- e.g. 'Toronto_CA'
    city_name VARCHAR NOT NULL,
    country   VARCHAR NOT NULL,         -- ISO2 of the city itself, not the league
    latitude  DOUBLE NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude DOUBLE NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    UNIQUE (city_name, country)
);

CREATE TABLE teams (
    team_id        VARCHAR PRIMARY KEY,   -- e.g. 'E0:Arsenal' (league-scoped by design)
    league_id      VARCHAR NOT NULL REFERENCES leagues (league_id),
    canonical_name VARCHAR NOT NULL,
    city_id        VARCHAR NOT NULL REFERENCES cities (city_id)
);

CREATE TABLE team_aliases (
    league_id VARCHAR NOT NULL REFERENCES leagues (league_id),
    alias     VARCHAR NOT NULL,
    team_id   VARCHAR NOT NULL REFERENCES teams (team_id),
    PRIMARY KEY (league_id, alias)
);

CREATE TABLE matches (
    match_id     VARCHAR PRIMARY KEY,
    league_id    VARCHAR NOT NULL REFERENCES leagues (league_id),
    season_label VARCHAR NOT NULL,
    start_year   INTEGER NOT NULL CHECK (start_year BETWEEN 1990 AND 2030),
    match_date   DATE NOT NULL,
    home_team_id VARCHAR NOT NULL REFERENCES teams (team_id),
    away_team_id VARCHAR NOT NULL REFERENCES teams (team_id),
    home_goals   INTEGER NOT NULL CHECK (home_goals >= 0),
    away_goals   INTEGER NOT NULL CHECK (away_goals >= 0),
    CHECK (home_team_id <> away_team_id),
    UNIQUE (league_id, match_date, home_team_id, away_team_id)
);

-- Fair probabilities are computed in the Phase 8 view, not stored.
CREATE TABLE odds (
    match_id  VARCHAR NOT NULL REFERENCES matches (match_id),
    bookmaker VARCHAR NOT NULL,
    odds_home DOUBLE CHECK (odds_home > 1),
    odds_draw DOUBLE CHECK (odds_draw > 1),
    odds_away DOUBLE CHECK (odds_away > 1),
    PRIMARY KEY (match_id, bookmaker)
);
