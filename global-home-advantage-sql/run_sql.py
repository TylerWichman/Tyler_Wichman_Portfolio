import duckdb
from pathlib import Path

DB_PATH = "/content/hfa_global.duckdb"

def get_con():
    return duckdb.connect(DB_PATH)

def run_sql(path, con=None):
    close_after = con is None
    con = con or get_con()
    sql = Path(path).read_text()
    result = con.execute(sql)
    try:
        df = result.df()
    except Exception:
        df = None
    if close_after:
        con.close()
    return df
