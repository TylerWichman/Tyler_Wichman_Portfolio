"""Download Main League (season-by-season) and Extra League (all-history) files."""
import time
from pathlib import Path

import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MAIN_LEAGUES = ["E0", "SC0", "D1", "I1", "SP1", "F1", "N1", "B1", "P1", "T1", "G1"]
FIRST_SEASON_START = 2000
LAST_SEASON_START = 2026

EXTRA_LEAGUES = [
    "ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN",
    "MEX", "NOR", "POL", "ROU", "RUS", "SWE", "SWZ", "USA",
]


def season_codes(first, last):
    return [f"{str(y)[-2:]}{str(y + 1)[-2:]}" for y in range(first, last + 1)]


def save_as_utf8(content, path):
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    path.write_text(text, encoding="utf-8")


def download(url, path):
    if path.exists():
        print(f"skip (exists): {path.name}")
        return
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        print(f"FAILED {resp.status_code}: {url}")
        return
    save_as_utf8(resp.content, path)
    print(f"saved: {path.name}")
    time.sleep(1)


def main():
    for league in MAIN_LEAGUES:
        for code in season_codes(FIRST_SEASON_START, LAST_SEASON_START):
            url = f"https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
            download(url, DATA_DIR / f"main_{league}_{code}.csv")

    for code in EXTRA_LEAGUES:
        url = f"https://football-data.co.uk/new/{code}.csv"
        download(url, DATA_DIR / f"extra_{code}.csv")

    r = requests.get("https://www.football-data.co.uk/notes.txt", timeout=30)
    (DATA_DIR / "notes.txt").write_text(r.text, encoding="utf-8")


if __name__ == "__main__":
    main()
