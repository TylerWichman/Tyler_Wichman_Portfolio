"""
Automated league -> Wikipedia -> current top-flight team/city extraction,
matched against actual raw team names in stg_main/stg_extra.

Run this once per league (or loop through all remaining leagues).
Requires: pandas, requests, rapidfuzz, lxml
"""
import re
import time

import pandas as pd
import requests
from rapidfuzz import fuzz, process

HEADERS = {"User-Agent": "tyler-portfolio-project/1.0 (personal analysis)"}


def wiki_search_title(query):
    """Find the best-matching Wikipedia page title for a search query."""
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": 1,
        },
        headers=HEADERS, timeout=15,
    )
    results = r.json().get("query", {}).get("search", [])
    if not results:
        return None
    return results[0]["title"]


def fetch_tables(title):
    """Fetch a Wikipedia page and return all HTML tables as DataFrames."""
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    try:
        return pd.read_html(r.text), url
    except ValueError:
        return [], url


def find_team_location_table(tables):
    """Find the table listing clubs and their city/location."""
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        has_team = any(k in " ".join(cols) for k in ["team", "club"])
        has_loc = any(k in " ".join(cols) for k in ["location", "city"])
        if has_team and has_loc and len(t) >= 8:  # a real league table, not a small sidebar
            return t
    return None


def clean_cell(x):
    """Strip footnote markers, parenthetical district notes, extra whitespace."""
    if pd.isna(x):
        return None
    x = re.sub(r"\[.*?\]", "", str(x))          # footnotes like [1]
    x = re.sub(r"\(.*?\)", "", x)                # parenthetical district notes
    return x.strip()


def get_raw_team_names(con, league_code, is_extra):
    """Pull actual team names as they appear in the staged data, trimmed."""
    if is_extra:
        q = f"SELECT DISTINCT TRIM(Home) AS team FROM stg_extra WHERE league_code = '{league_code}'"
    else:
        q = f"SELECT DISTINCT TRIM(HomeTeam) AS team FROM stg_main WHERE league_code = '{league_code}'"
    return con.execute(q).df()["team"].dropna().unique().tolist()


def build_league(con, league_code, search_query, is_extra, match_threshold=80):
    """Full pipeline for one league: search -> fetch -> parse -> fuzzy match."""
    title = wiki_search_title(search_query)
    if title is None:
        print(f"[{league_code}] No Wikipedia page found for '{search_query}'")
        return pd.DataFrame()

    tables, url = fetch_tables(title)
    table = find_team_location_table(tables)
    if table is None:
        print(f"[{league_code}] Found page '{title}' ({url}) but no team/location table")
        return pd.DataFrame()

    team_col = next(c for c in table.columns if "team" in str(c).lower() or "club" in str(c).lower())
    loc_col = next(c for c in table.columns if "location" in str(c).lower() or "city" in str(c).lower())

    wiki_teams = table[[team_col, loc_col]].copy()
    wiki_teams.columns = ["wiki_team", "city"]
    wiki_teams["wiki_team"] = wiki_teams["wiki_team"].apply(clean_cell)
    wiki_teams["city"] = wiki_teams["city"].apply(clean_cell)
    wiki_teams = wiki_teams.dropna()

    raw_names = get_raw_team_names(con, league_code, is_extra)
    if not raw_names:
        print(f"[{league_code}] WARNING: no raw team names found in staged data — check league_code")
        return pd.DataFrame()

    rows = []
    for _, row in wiki_teams.iterrows():
        match = process.extractOne(row["wiki_team"], raw_names, scorer=fuzz.WRatio)
        matched_name, score = (match[0], match[1]) if match else (None, 0)
        rows.append({
            "league_code": league_code,
            "wiki_team": row["wiki_team"],
            "matched_raw_team": matched_name,
            "city": row["city"],
            "match_score": score,
        })

    result = pd.DataFrame(rows)
    n_good = (result["match_score"] >= match_threshold).sum()
    print(f"[{league_code}] {title}: {len(result)} clubs, {n_good} matched >= {match_threshold}, source: {url}")
    return result


# ---- League definitions: code -> (search query, is_extra_format) ----
LEAGUES_TO_BUILD = {
    "SP1": ("2026-27 La Liga", False),
    "F1":  ("2026-27 Ligue 1", False),
    "B1":  ("2026-27 Belgian Pro League", False),
    "P1":  ("2026-27 Primeira Liga", False),
    "T1":  ("2026-27 Super Lig", False),
    "G1":  ("2026-27 Super League Greece", False),
    "ARG": ("2026 Argentine Primera Division", True),
    "AUT": ("2026-27 Austrian Football Bundesliga", True),
    "BRA": ("2026 Campeonato Brasileiro Serie A", True),
    "CHN": ("2026 Chinese Super League", True),
    "DNK": ("2026-27 Danish Superliga", True),
    "FIN": ("2026 Veikkausliiga", True),
    "IRL": ("2026 League of Ireland Premier Division", True),
    "JPN": ("2026 J1 League", True),
    "MEX": ("2026-27 Liga MX season", True),
    "NOR": ("2026 Eliteserien", True),
    "POL": ("2026-27 Ekstraklasa", True),
    "ROU": ("2026-27 Liga I", True),
    "RUS": ("2026-27 Russian Premier League", True),
    "SWE": ("2026 Allsvenskan", True),
    "SWZ": ("2026-27 Swiss Super League", True),
    "USA": ("2026 Major League Soccer season", True),
}


def run_all(con, leagues=None, threshold=80):
    """Run the full pipeline for all leagues (or a subset) and return combined results."""
    targets = leagues or LEAGUES_TO_BUILD
    all_results = []
    for code, (query, is_extra) in targets.items():
        result = build_league(con, code, query, is_extra, threshold)
        if not result.empty:
            all_results.append(result)
        time.sleep(1)  # be polite to Wikipedia
    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)

def wiki_search_title_safe(query, max_retries=3):
    """Wrapped version with retries for transient failures."""
    for attempt in range(max_retries):
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
                headers=HEADERS, timeout=15,
            )
            if r.status_code == 200:
                results = r.json().get("query", {}).get("search", [])
                return results[0]["title"] if results else None
        except Exception as e:
            print(f"  retry {attempt+1}/{max_retries} for search '{query}': {e}")
        time.sleep(3 * (attempt + 1))
    return None


def build_league_safe(con, league_code, search_query, is_extra, match_threshold=80):
    """Resilient wrapper: never raises, always returns a DataFrame (possibly empty)."""
    try:
        title = wiki_search_title_safe(search_query)
        if title is None:
            print(f"[{league_code}] No page found (after retries) for '{search_query}'")
            return pd.DataFrame()
        tables, url = fetch_tables(title)
        table = find_team_location_table(tables)
        if table is None:
            print(f"[{league_code}] Found page '{title}' ({url}) but no team/location table")
            return pd.DataFrame()
        team_col = next(c for c in table.columns if "team" in str(c).lower() or "club" in str(c).lower())
        loc_col = next(c for c in table.columns if "location" in str(c).lower() or "city" in str(c).lower())
        wiki_teams = table[[team_col, loc_col]].copy()
        wiki_teams.columns = ["wiki_team", "city"]
        wiki_teams["wiki_team"] = wiki_teams["wiki_team"].apply(clean_cell)
        wiki_teams["city"] = wiki_teams["city"].apply(clean_cell)
        wiki_teams = wiki_teams.dropna()
        raw_names = get_raw_team_names(con, league_code, is_extra)
        if not raw_names:
            print(f"[{league_code}] WARNING: no raw team names in staged data")
            return pd.DataFrame()
        rows = []
        for _, row in wiki_teams.iterrows():
            match = process.extractOne(row["wiki_team"], raw_names, scorer=fuzz.WRatio)
            matched_name, score = (match[0], match[1]) if match else (None, 0)
            rows.append({"league_code": league_code, "wiki_team": row["wiki_team"],
                         "matched_raw_team": matched_name, "city": row["city"], "match_score": score})
        result = pd.DataFrame(rows)
        n_good = (result["match_score"] >= match_threshold).sum()
        print(f"[{league_code}] {title}: {len(result)} clubs, {n_good} matched >= {match_threshold}")
        return result
    except Exception as e:
        print(f"[{league_code}] FAILED with error: {e}")
        return pd.DataFrame()


def run_remaining(con, leagues_to_run, threshold=80, pause=2):
    """Run only the specified leagues, resilient to individual failures."""
    all_results = []
    for code in leagues_to_run:
        query, is_extra = LEAGUES_TO_BUILD[code]
        result = build_league_safe(con, code, query, is_extra, threshold)
        if not result.empty:
            all_results.append(result)
        time.sleep(pause)
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

def run_all_safe(con, leagues=None, threshold=80, pause=2, save_path='team_cities_auto_raw.csv'):
    """Runs every league, saving incrementally after each one so a crash never loses prior progress."""
    import os
    targets = leagues or LEAGUES_TO_BUILD
    all_results = []
    # Resume support: if a partial file already exists, load it and skip completed leagues
    already_done = set()
    if os.path.exists(save_path):
        existing = pd.read_csv(save_path)
        all_results.append(existing)
        already_done = set(existing['league_code'].unique())
        print(f"Resuming: {len(already_done)} leagues already saved: {sorted(already_done)}")

    for code, (query, is_extra) in targets.items():
        if code in already_done:
            continue
        result = build_league_safe(con, code, query, is_extra, threshold)
        if not result.empty:
            all_results.append(result)
            # Save after EVERY league, not just at the end
            pd.concat(all_results, ignore_index=True).to_csv(save_path, index=False)
        time.sleep(pause)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
