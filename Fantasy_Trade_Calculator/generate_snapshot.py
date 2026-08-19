"""Weekly static-snapshot generator for the Fantasy Trade Calculator website.

Run by a GitHub Actions cron job. Re-fits the model exactly as the notebook does and
writes static JSON output under data/ -- the Streamlit app reads only these files at
request time, so no site visitor ever triggers a live FantasyCalc/KTC/CFBD pull. This
script is the only place those live pulls happen.
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# trade_calculator_pipeline.py lives alongside this script in the same directory.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import trade_calculator_pipeline as tcp  # noqa: E402
import nflreadpy as nfl  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _clean_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of JSON-safe dicts (NaN/NaT/numpy scalars -> None/native types)."""
    return json.loads(df.where(pd.notna(df), None).to_json(orient="records"))


def _team_and_sleeper_lookup() -> pd.DataFrame:
    """gsis_id -> (sleeper_id, team), pulled directly from nflreadpy's raw player-id
    table. build_crosswalk() trims this column out, so it's fetched again here rather
    than modifying that (verified, in-use-elsewhere) function.
    """
    raw = nfl.load_ff_playerids().to_pandas()
    return raw[["gsis_id", "sleeper_id", "team"]].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")


def run() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Building crosswalk...")
    crosswalk = tcp.build_crosswalk()

    print("Pulling FantasyCalc (6 format-teamsize combos)...")
    fc_all = tcp.pull_all_fantasycalc()

    print("Pulling KTC dynasty + redraft...")
    ktc_dyn = tcp.pull_ktc_dynasty()
    ktc_redraft = tcp.pull_ktc_redraft()

    print("Feature assembly: draft capital, weekly features, aging curves, combine, CFBD rookie...")
    draft_capital = tcp.load_draft_capital(crosswalk)
    weekly = tcp.load_weekly_features()
    curve_params, peak_ages, aged = tcp.fit_aging_curves(weekly, draft_capital)
    combine_feats = tcp.load_combine_features()
    rookie_college = tcp.compute_rookie_college_features(draft_capital, draft_year=2026)

    print("Draft pick valuation...")
    pick_universe = tcp.build_pick_universe(fc_all, ktc_dyn)

    print("Unified feature table...")
    active_ids = tcp.active_asset_gsis_ids(fc_all, crosswalk)
    snapshot = tcp.build_player_snapshot(aged, draft_capital, peak_ages, active_ids=active_ids)
    feature_table = tcp.build_feature_table(snapshot, draft_capital, combine_feats, rookie_college)

    print("Consensus targets for all 6 format-teamsize combos...")
    consensus_targets = tcp.build_all_consensus_targets(fc_all, ktc_dyn, ktc_redraft, crosswalk)

    print("Fitting all 6 models...")
    fit_results = tcp.fit_all_format_models(feature_table, consensus_targets)

    print("Board assembly...")
    boards = tcp.build_all_boards(feature_table, crosswalk, consensus_targets, pick_universe, fit_results)

    team_sleeper = _team_and_sleeper_lookup()

    search_rows = []
    player_count = 0
    pick_count = 0

    for (fmt, team_size), board in boards.items():
        b = board.merge(team_sleeper, on="gsis_id", how="left")
        b["residual"] = b["model_value"] - b["consensus_value"]
        b["residual_pct"] = np.where(
            b["consensus_value"] != 0, b["residual"] / b["consensus_value"] * 100, 0.0
        )
        b = b.rename(columns={"gsis_id": "asset_id"})
        out_cols = [
            "asset_id", "sleeper_id", "name", "position", "team", "asset_type",
            "model_value", "consensus_value", "residual", "residual_pct",
            "current_age", "college_seasons_found", "year", "round", "tier",
        ]
        records = _clean_records(b[out_cols])
        with open(os.path.join(DATA_DIR, f"{fmt}_{team_size}.json"), "w") as f:
            json.dump(records, f, separators=(",", ":"))

        if (fmt, team_size) == ("sf_dynasty", 12):
            player_count = int((b["asset_type"] == "player").sum())
            pick_count = int((b["asset_type"] == "pick").sum())
            search_rows.append(b[["asset_id", "sleeper_id", "name", "position", "team", "asset_type"]])

    search_index = (
        pd.concat(search_rows, ignore_index=True)
        .drop_duplicates(subset=["asset_id"])
        .sort_values("name")
    )
    with open(os.path.join(DATA_DIR, "search_index.json"), "w") as f:
        json.dump(_clean_records(search_index), f, separators=(",", ":"))

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": tcp.CURRENT_SEASON,
        "formats": [f"{fmt}_{ts}" for fmt, ts in boards.keys()],
        "player_count": player_count,
        "pick_count": pick_count,
    }
    with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote {len(boards)} board files + search_index.json + metadata.json to {DATA_DIR}")
    for (fmt, team_size), board in boards.items():
        print(f"  {fmt}_{team_size}.json: {len(board)} rows")
    print(f"  search_index.json: {len(search_index)} unique assets")

    sleeper_cov = search_index["sleeper_id"].notna().mean()
    team_cov = search_index["team"].notna().mean()
    print(f"  sleeper_id coverage in search index: {sleeper_cov:.1%}")
    print(f"  team coverage in search index: {team_cov:.1%}")


if __name__ == "__main__":
    run()
