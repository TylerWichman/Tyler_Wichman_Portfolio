"""
Merge the automated (Wikipedia fuzzy-match) results with the manually-built
rows into one final ref_team_cities.csv, applying the match-confidence
threshold and reporting exactly what's left to review.
"""
import pandas as pd

THRESHOLD = 80

# 1. Load automated results
auto = pd.read_csv('team_cities_auto_raw.csv')
auto_good = auto[auto['match_score'] >= THRESHOLD].copy()
auto_good = auto_good.rename(columns={'matched_raw_team': 'team'})[['league_code', 'team', 'city']]

auto_review = auto[auto['match_score'] < THRESHOLD].copy()

# 2. Load manually-built rows
manual = pd.read_csv('ref_team_cities_manual.csv')

# 3. Combine, de-duplicate on (league_code, team)
final = pd.concat([manual, auto_good], ignore_index=True)
final = final.drop_duplicates(subset=['league_code', 'team'])
final.to_csv('ref_team_cities.csv', index=False)

# 4. Coverage report
print("=== Final ref_team_cities.csv ===")
print(f"{len(final)} total team-city mappings across {final['league_code'].nunique()} leagues")
print(final['league_code'].value_counts().sort_index())

print("\n=== Needs manual review (match_score < 80) ===")
if len(auto_review) > 0:
    print(auto_review[['league_code', 'wiki_team', 'matched_raw_team', 'city', 'match_score']].to_string(index=False))
else:
    print("None")
