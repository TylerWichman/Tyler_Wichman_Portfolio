import duckdb
from pathlib import Path

DB_PATH = "/content/hfa.duckdb"


def get_con():
    """Open (or create) the DuckDB database file."""
    return duckdb.connect(DB_PATH)


def run_sql(path, con=None):
    """Run every statement in a .sql file. Return the last statement's result as a DataFrame."""
    close_after = con is None
    con = con or get_con()
    sql = Path(path).read_text()
    result = con.execute(sql)
    try:
        df = result.df()
    except Exception:
        df = None  # the last statement returned no rows (e.g. CREATE TABLE)
    if close_after:
        con.close()
    return df
