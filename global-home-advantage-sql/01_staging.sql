-- null_padding fills rows that omit trailing empty columns (common in 2001-2005 files).
CREATE OR REPLACE TABLE stg_main AS
SELECT
    regexp_extract(filename, 'main_([A-Z0-9]+)_(\d{4})\.csv', 1) AS league_code,
    regexp_extract(filename, 'main_([A-Z0-9]+)_(\d{4})\.csv', 2) AS season_code,
    *
FROM read_csv('data/main_*.csv', union_by_name = true, filename = true,
              all_varchar = true, null_padding = true, ignore_errors = true);

CREATE OR REPLACE TABLE stg_extra AS
SELECT
    regexp_extract(filename, 'extra_([A-Z0-9]+)\.csv', 1) AS league_code,
    *
FROM read_csv('data/extra_*.csv', union_by_name = true, filename = true,
              all_varchar = true, null_padding = true, ignore_errors = true);
