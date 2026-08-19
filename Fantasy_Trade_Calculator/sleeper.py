"""Thin client for Sleeper's public API (no auth required).

https://docs.sleeper.com/ -- all endpoints below are public GET requests.
Every function returns None/[] on failure instead of raising, so the Streamlit
UI can decide what to show rather than crashing on a bad username/league id.
"""
import requests

BASE = "https://api.sleeper.app/v1"
TIMEOUT = 15


def _get(path: str):
    try:
        r = requests.get(f"{BASE}{path}", timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.RequestException:
        return None


def resolve_username(username: str) -> str | None:
    """Username -> Sleeper user_id, or None if not found."""
    data = _get(f"/user/{username}")
    if not data or "user_id" not in data:
        return None
    return data["user_id"]


def list_user_leagues(user_id: str, season: str) -> list[dict]:
    """All leagues (any format) a user belongs to for a given season."""
    data = _get(f"/user/{user_id}/leagues/nfl/{season}")
    return data if isinstance(data, list) else []


def get_league(league_id: str) -> dict | None:
    """League settings -- total_rosters, roster_positions, scoring_settings, etc."""
    data = _get(f"/league/{league_id}")
    return data if isinstance(data, dict) else None


def get_rosters(league_id: str) -> list[dict]:
    """Each roster: roster_id, owner_id, players (list of Sleeper player-id strings)."""
    data = _get(f"/league/{league_id}/rosters")
    return data if isinstance(data, list) else []


def get_users(league_id: str) -> list[dict]:
    """Each user: user_id, display_name, metadata.team_name if set."""
    data = _get(f"/league/{league_id}/users")
    return data if isinstance(data, list) else []


def get_traded_picks(league_id: str) -> list[dict]:
    """Each entry: season, round, roster_id (original owner), owner_id (current owner),
    previous_owner_id. Empty list if the league has no trades or endpoint is empty."""
    data = _get(f"/league/{league_id}/traded_picks")
    return data if isinstance(data, list) else []


def normalize_sleeper_id(x) -> str | None:
    """Our snapshot's sleeper_id serializes as float (e.g. 9221.0) due to a pandas
    upcast; Sleeper's own API returns player ids as plain strings (e.g. "9221").
    Normalize both sides through this before comparing/joining."""
    if x is None:
        return None
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x)
