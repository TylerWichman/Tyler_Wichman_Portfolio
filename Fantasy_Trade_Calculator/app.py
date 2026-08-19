"""Fantasy Trade Calculator -- Streamlit app.

Reads static weekly snapshot JSON (data/, written by generate_snapshot.py -- never
pulled live here). The only live calls this app makes are to Sleeper's public API,
and only when a user actively syncs a league in Tab 3.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import sleeper

DATA_DIR = Path(__file__).parent / "data"

FORMATS = {"Superflex Dynasty": "sf_dynasty", "1QB Dynasty": "oneqb_dynasty", "Redraft": "redraft"}
TEAM_SIZES = [10, 12]
TRADE_TOLERANCE_PCT = 5.0  # within this %, a trade is called "roughly even"


@st.cache_data
def load_board(format_key: str, team_size: int) -> pd.DataFrame:
    path = DATA_DIR / f"{format_key}_{team_size}.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sleeper_id_norm"] = df["sleeper_id"].apply(sleeper.normalize_sleeper_id)
    return df


@st.cache_data
def load_metadata() -> dict:
    path = DATA_DIR / "metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def nearest_team_size(n: int) -> tuple[int, bool]:
    """Round a real league's roster count to our nearest supported size (10 or 12)."""
    if n in TEAM_SIZES:
        return n, False
    nearest = min(TEAM_SIZES, key=lambda t: abs(t - n))
    return nearest, True


def format_value(v) -> str:
    return f"{v:,.0f}" if pd.notna(v) else "--"


# ---------------------------------------------------------------------------
# Sidebar -- format/team-size selection, shared across all 3 tabs
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Fantasy Trade Calculator", layout="wide")
st.title("Fantasy Trade Calculator")

meta = load_metadata()
if meta.get("generated_at"):
    st.caption(f"Board data last updated: {meta['generated_at']}")

with st.sidebar:
    st.header("League format")
    format_label = st.selectbox("Format", list(FORMATS.keys()))
    format_key = FORMATS[format_label]
    team_size = st.selectbox("Team size", TEAM_SIZES)
    is_dynasty = format_key != "redraft"

board = load_board(format_key, team_size)
if board.empty:
    st.error(f"No snapshot data found for {format_label} / {team_size}-team. "
             f"Run generate_snapshot.py first.")
    st.stop()

players_df = board[board["asset_type"] == "player"]
picks_df = board[board["asset_type"] == "pick"] if is_dynasty else board.iloc[0:0]

synced = st.session_state.get("synced_league")

tab1, tab2, tab3 = st.tabs(["Trade Calculator", "Rankings", "League Dashboard"])

# ---------------------------------------------------------------------------
# Tab 1 -- Trade Calculator
# ---------------------------------------------------------------------------

with tab1:
    st.subheader("Trade Calculator")
    st.caption(f"Values from the {format_label} / {team_size}-team board.")

    asset_pool = pd.concat([players_df, picks_df]) if is_dynasty else players_df
    if not is_dynasty:
        st.caption("Redraft has no draft-pick assets -- only players are selectable.")

    def asset_label(row) -> str:
        if row["asset_type"] == "pick":
            return f"{row['name']} (pick)"
        team = row.get("team") or "FA"
        return f"{row['name']} ({row['position']}, {team})"

    asset_pool = asset_pool.copy()
    asset_pool["label"] = asset_pool.apply(asset_label, axis=1)
    label_to_value = dict(zip(asset_pool["label"], asset_pool["model_value"]))

    roster_filter_options = ["Full board"]
    roster_map: dict[str, list[str]] = {}
    if synced:
        roster_filter_options += [f"My roster ({synced['my_team_name']})"] + [
            f"{t['team_name']}'s roster" for t in synced["rosters"] if t["roster_id"] != synced["my_roster_id"]
        ]
        for t in synced["rosters"]:
            key = (f"My roster ({synced['my_team_name']})" if t["roster_id"] == synced["my_roster_id"]
                   else f"{t['team_name']}'s roster")
            roster_map[key] = t["sleeper_ids_norm"]

    col_a, col_b = st.columns(2)

    def side_selector(col, side_label: str, key_prefix: str):
        with col:
            st.markdown(f"**Side {side_label}**")
            filt = st.selectbox("Filter options to", roster_filter_options, key=f"{key_prefix}_filter") \
                if synced else "Full board"
            pool = asset_pool
            if filt != "Full board":
                ids = set(roster_map.get(filt, []))
                pool = asset_pool[asset_pool["sleeper_id_norm"].isin(ids)]
            picks = st.multiselect(f"Assets -- Side {side_label}", pool["label"].tolist(), key=f"{key_prefix}_assets")
            total = sum(label_to_value.get(p, 0) for p in picks)
            st.metric(f"Side {side_label} total", format_value(total))
            return total

    total_a = side_selector(col_a, "A", "side_a")
    total_b = side_selector(col_b, "B", "side_b")

    st.divider()
    if total_a == 0 and total_b == 0:
        st.info("Add assets to both sides to see a verdict.")
    else:
        bigger = max(total_a, total_b)
        diff_pct = abs(total_a - total_b) / bigger * 100 if bigger else 0
        if diff_pct <= TRADE_TOLERANCE_PCT:
            st.success(f"Roughly even trade (within {TRADE_TOLERANCE_PCT:.0f}%).")
        elif total_a > total_b:
            st.success(f"Side A favored by {diff_pct:.1f}% ({format_value(total_a - total_b)} value).")
        else:
            st.success(f"Side B favored by {diff_pct:.1f}% ({format_value(total_b - total_a)} value).")

# ---------------------------------------------------------------------------
# Tab 2 -- Rankings
# ---------------------------------------------------------------------------

with tab2:
    st.subheader("Rankings")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pos_filter = st.selectbox("Position", ["All", "QB", "RB", "WR", "TE"])
    with c2:
        if is_dynasty:
            type_filter = st.selectbox("Asset type", ["Both", "Players", "Picks"])
        else:
            type_filter = "Players"
    with c3:
        sort_by = st.selectbox("Sort by", ["Model value", "Residual (buy-low)"])
    with c4:
        ascending = st.checkbox("Ascending", value=False)

    view = board.copy()
    if not is_dynasty:
        view = view[view["asset_type"] == "player"]
    elif type_filter == "Players":
        view = view[view["asset_type"] == "player"]
    elif type_filter == "Picks":
        view = view[view["asset_type"] == "pick"]

    if pos_filter != "All":
        view = view[view["position"] == pos_filter]

    if sort_by == "Residual (buy-low)":
        before = len(view)
        view = view[~(view["college_seasons_found"] == 0)]
        excluded = before - len(view)
        if excluded:
            st.caption(f"Excluding {excluded} zero-snap rookie(s) with no matched college-production "
                       f"data -- their divergence isn't independent signal, it's a known modeling gap.")
        sort_col = "residual"
    else:
        sort_col = "model_value"

    view = view.sort_values(sort_col, ascending=ascending)
    display_cols = ["name", "position", "team", "asset_type", "model_value", "consensus_value",
                     "residual", "residual_pct", "current_age"]
    st.dataframe(view[display_cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 3 -- League Dashboard
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("League Dashboard")

    with st.form("sync_form"):
        sync_input = st.text_input("Sleeper username or league ID")
        submitted = st.form_submit_button("Sync league")

    if submitted and sync_input:
        league_id = None
        candidate_leagues = []
        user_id = sleeper.resolve_username(sync_input)
        if user_id:
            season = str(meta.get("season", 2026))
            candidate_leagues = sleeper.list_user_leagues(user_id, season)
            if not candidate_leagues:
                st.warning(f"Found user '{sync_input}' but no leagues for season {season}.")
            elif len(candidate_leagues) == 1:
                league_id = candidate_leagues[0]["league_id"]
            else:
                st.session_state["_candidate_leagues"] = candidate_leagues
                st.session_state["_sync_user_id"] = user_id
        else:
            lg = sleeper.get_league(sync_input)
            if lg:
                league_id = sync_input
            else:
                st.error("Couldn't resolve that as a Sleeper username or a league ID.")

        if league_id:
            st.session_state["_pending_league_id"] = league_id

    if st.session_state.get("_candidate_leagues"):
        options = {f"{l['name']} ({l['season']})": l["league_id"]
                   for l in st.session_state["_candidate_leagues"]}
        choice = st.selectbox("Multiple leagues found -- pick one", list(options.keys()))
        if st.button("Use this league"):
            st.session_state["_pending_league_id"] = options[choice]
            del st.session_state["_candidate_leagues"]

    if st.session_state.get("_pending_league_id"):
        lid = st.session_state.pop("_pending_league_id")
        lg = sleeper.get_league(lid)
        rosters_raw = sleeper.get_rosters(lid)
        users_raw = sleeper.get_users(lid)
        traded_picks = sleeper.get_traded_picks(lid)

        if not lg or not rosters_raw:
            st.error("Couldn't load rosters for that league.")
        else:
            real_size = lg.get("total_rosters", len(rosters_raw))
            snap_size, approximated = nearest_team_size(real_size)
            if approximated:
                st.info(f"This league has {real_size} teams -- no exact snapshot for that size, "
                        f"using the {snap_size}-team board as the closest approximation.")

            uid_to_name = {u["user_id"]: (u.get("metadata", {}) or {}).get("team_name") or u.get("display_name")
                           for u in users_raw}
            b = load_board(format_key, snap_size)
            b_players = b[b["asset_type"] == "player"]
            b_picks = b[b["asset_type"] == "pick"] if is_dynasty else b.iloc[0:0]

            rosters = []
            for r in rosters_raw:
                sleeper_ids_norm = [sleeper.normalize_sleeper_id(p) for p in (r.get("players") or [])]
                rosters.append({
                    "roster_id": r["roster_id"],
                    "owner_id": r.get("owner_id"),
                    "team_name": uid_to_name.get(r.get("owner_id"), f"Roster {r['roster_id']}"),
                    "sleeper_ids_norm": sleeper_ids_norm,
                })

            my_user_id = st.session_state.get("_sync_user_id") or sleeper.resolve_username(sync_input)
            my_roster = next((r for r in rosters if r["owner_id"] == my_user_id), rosters[0])

            st.session_state["synced_league"] = {
                "league_id": lid, "league_name": lg.get("name"), "snap_size": snap_size,
                "rosters": rosters, "my_roster_id": my_roster["roster_id"],
                "my_team_name": my_roster["team_name"], "traded_picks": traded_picks,
            }
            st.success(f"Synced '{lg.get('name')}' -- {len(rosters)} teams.")
            synced = st.session_state["synced_league"]

    if not synced:
        st.info("Sync a league above to see your roster, buy-low targets, and the team value histogram.")
    else:
        b = load_board(format_key, synced["snap_size"])
        b_players = b[b["asset_type"] == "player"]
        b_picks = b[b["asset_type"] == "pick"] if is_dynasty else b.iloc[0:0]

        my_ids = set(next(r for r in synced["rosters"] if r["roster_id"] == synced["my_roster_id"])["sleeper_ids_norm"])
        my_assets = b_players[b_players["sleeper_id_norm"].isin(my_ids)]

        st.markdown(f"### Your highest-valued assets -- {synced['my_team_name']}")
        top_n = st.slider("Show top N", 5, 25, 10)
        st.dataframe(
            my_assets.sort_values("model_value", ascending=False).head(top_n)
            [["name", "position", "team", "model_value", "consensus_value"]],
            use_container_width=True, hide_index=True,
        )

        st.markdown("### Five targets")
        st.caption("Top undervalued assets league-wide (positive model-vs-market divergence), "
                   "not matched to your specific team needs -- that's a planned future enhancement, not v1.")
        other_ids = set()
        for r in synced["rosters"]:
            if r["roster_id"] != synced["my_roster_id"]:
                other_ids |= set(r["sleeper_ids_norm"])
        targets = b_players[b_players["sleeper_id_norm"].isin(other_ids)]
        targets = targets[~(targets["college_seasons_found"] == 0)]
        targets = targets[targets["residual"] > 0].sort_values("residual", ascending=False).head(5)
        st.dataframe(targets[["name", "position", "team", "model_value", "consensus_value", "residual"]],
                     use_container_width=True, hide_index=True)

        st.markdown("### Roster Value Histogram")
        st.caption("Players + owned draft picks (un-traded future picks valued at the league-average "
                   "\"Mid\" tier, since draft order isn't knowable pre-draft).")

        def mid_pick_value(year: int, rnd: int) -> float:
            if not is_dynasty:
                return 0.0
            row = b_picks[(b_picks["year"] == year) & (b_picks["round"] == rnd) & (b_picks["tier"] == "Mid")]
            return float(row["model_value"].iloc[0]) if len(row) else 0.0

        pick_years = sorted(set(int(y) for y in b_picks["year"].dropna().unique())) if is_dynasty else []
        pick_rounds = sorted(set(int(r) for r in b_picks["round"].dropna().unique())) if is_dynasty else []
        roster_ids_all = [r["roster_id"] for r in synced["rosters"]]

        # Every roster has exactly one natural pick per (year, round); a traded_picks
        # entry reassigns one specific (season, round, original owner) slot to a new
        # current owner. Default (no entry) = the original owner still has it.
        pick_owner = {
            (int(tp["season"]), int(tp["round"]), tp["roster_id"]): tp["owner_id"]
            for tp in synced["traded_picks"]
        }

        team_totals = []
        for r in synced["rosters"]:
            ids = set(r["sleeper_ids_norm"])
            player_val = b_players[b_players["sleeper_id_norm"].isin(ids)]["model_value"].sum()
            pick_val = 0.0
            for y in pick_years:
                for rnd in pick_rounds:
                    for original_roster in roster_ids_all:
                        current_owner = pick_owner.get((y, rnd, original_roster), original_roster)
                        if current_owner == r["roster_id"]:
                            pick_val += mid_pick_value(y, rnd)
            team_totals.append({"team_name": r["team_name"], "roster_id": r["roster_id"],
                                 "total_value": player_val + pick_val})

        totals_df = pd.DataFrame(team_totals).sort_values("total_value", ascending=False).reset_index(drop=True)
        totals_df["rank"] = totals_df.index + 1
        my_row = totals_df[totals_df["roster_id"] == synced["my_roster_id"]]
        if len(my_row):
            st.markdown(f"**You are {int(my_row['rank'].iloc[0])} of {len(totals_df)}** "
                        f"({format_value(my_row['total_value'].iloc[0])} total value).")

        chart_df = totals_df.set_index("team_name")[["total_value"]]
        st.bar_chart(chart_df)
