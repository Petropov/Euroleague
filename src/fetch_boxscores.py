"""Fetch/cached boxscores and build player_game dataset."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable

import pandas as pd
import requests

from src.config import BOXSCORE_ENDPOINT, CURATED_DIR, DEFAULT_SEASON_CODE, LOG_DIR, RAW_DIR, REQUEST_DELAY_SECONDS, TEAMS
from src.fetch_results import run as run_fetch_results
from src.utils_http import build_session, get_json, get_logger, read_json_cache, write_json_cache
from src.utils_parse import safe_float, safe_int, safe_text, slugify


TEAM_CODE_KEYS = ["teamcode", "code", "teamCode", "shortName", "tricode"]
PLAYER_LIST_KEYS = ["players", "PlayerStats", "playerStats", "roster", "athletes"]

STAT_KEYS = {
    "min": ["min", "minutes", "timePlayed", "MIN"],
    "pts": ["pts", "points", "PTS"],
    "reb": ["reb", "totReb", "rebounds", "TR"],
    "ast": ["ast", "assists", "AS"],
    "stl": ["stl", "steals", "ST"],
    "blk": ["blk", "blocks", "BK"],
    "to": ["to", "turnovers", "TO"],
    "pf": ["pf", "fouls", "PF"],
    "fgm": ["fgm", "fgMade", "FGM"],
    "fga": ["fga", "fgAttempted", "FGA"],
    "tpm": ["tpm", "tpMade", "3PM", "fg3Made"],
    "tpa": ["tpa", "tpAttempted", "3PA", "fg3Attempted"],
    "ftm": ["ftm", "ftMade", "FTM"],
    "fta": ["fta", "ftAttempted", "FTA"],
}


def pick_value(payload: dict, aliases: list[str], default=None):
    for alias in aliases:
        if alias in payload:
            return payload[alias]
    lowered = {str(k).lower(): v for k, v in payload.items()}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return default


def iter_team_nodes(obj) -> Iterable[dict]:
    if isinstance(obj, dict):
        code = pick_value(obj, TEAM_CODE_KEYS)
        if isinstance(code, str) and code.upper() in TEAMS:
            yield obj
        for val in obj.values():
            yield from iter_team_nodes(val)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_team_nodes(item)


def iter_player_nodes(team_node: dict) -> Iterable[dict]:
    for key in PLAYER_LIST_KEYS:
        candidate = team_node.get(key)
        if isinstance(candidate, list):
            for player in candidate:
                if isinstance(player, dict):
                    yield player


def resolve_opponent(game_row: pd.Series, team_code: str) -> tuple[str, str]:
    if game_row["home_team_code"] == team_code:
        return game_row["away_team_code"], "home"
    return game_row["home_team_code"], "away"


def parse_player_rows(game_row: pd.Series, payload: dict, season_code: str, logger) -> list[dict]:
    rows: list[dict] = []
    seen = set()

    for team_node in iter_team_nodes(payload):
        team_code = safe_text(pick_value(team_node, TEAM_CODE_KEYS)).upper()
        if team_code not in TEAMS:
            continue
        opponent_code, home_away = resolve_opponent(game_row, team_code)
        for player in iter_player_nodes(team_node):
            player_name = safe_text(
                pick_value(
                    player,
                    ["player", "name", "fullname", "fullName", "PLAYER", "personName"],
                    default="Unknown",
                )
            )
            player_id = safe_text(pick_value(player, ["playercode", "playerId", "id", "code"]))
            if not player_id:
                player_id = f"{slugify(player_name)}_{team_code.lower()}_{season_code.lower()}"

            row = {
                "season_code": season_code,
                "gamecode_num": safe_int(game_row["gamecode_num"]),
                "date": game_row["date"],
                "team_code": team_code,
                "opponent_code": opponent_code,
                "home_away": home_away,
                "player_id": player_id,
                "player_name": player_name,
            }
            for stat, aliases in STAT_KEYS.items():
                value = pick_value(player, aliases, default=0)
                row[stat] = safe_float(value, default=0.0)
            key = (row["season_code"], row["gamecode_num"], row["team_code"], row["player_id"])
            if key not in seen:
                rows.append(row)
                seen.add(key)

    if not rows:
        logger.warning("No player rows parsed for game %s", game_row["gamecode_full"])
    return rows


def run(season_code: str = DEFAULT_SEASON_CODE) -> pd.DataFrame:
    logger = get_logger(LOG_DIR / "pipeline.log")
    games_path = CURATED_DIR / "games.parquet"
    if games_path.exists():
        games_df = pd.read_parquet(games_path)
        if not games_df.empty and "season_code" in games_df.columns:
            games_df = games_df[games_df["season_code"] == season_code].copy()
    else:
        games_df = run_fetch_results(season_code=season_code)

    if games_df.empty:
        logger.warning("No games available for boxscore fetching in %s", season_code)
        player_df = pd.DataFrame()
        player_df.to_parquet(CURATED_DIR / "player_game.parquet", index=False)
        player_df.to_csv(CURATED_DIR / "player_game.csv", index=False)
        return player_df

    session = build_session()
    fetched = 0
    cached = 0
    rows: list[dict] = []

    for _, game in games_df.iterrows():
        gamecode_num = safe_int(game["gamecode_num"])
        if gamecode_num is None:
            logger.warning("Skipping game without numeric code: %s", game.get("gamecode_full"))
            continue

        cache_path = RAW_DIR / "boxscore" / season_code / f"{game['gamecode_full']}.json"
        payload = read_json_cache(cache_path)
        if payload is None:
            try:
                payload = get_json(
                    session,
                    BOXSCORE_ENDPOINT,
                    params={"gamecode": str(gamecode_num), "seasoncode": season_code},
                )
                write_json_cache(cache_path, payload)
                fetched += 1
                time.sleep(REQUEST_DELAY_SECONDS)
            except requests.RequestException as exc:
                logger.warning("Failed to fetch boxscore for %s: %s", game["gamecode_full"], exc)
                continue
        else:
            cached += 1

        rows.extend(parse_player_rows(game, payload, season_code, logger))

    player_df = pd.DataFrame(rows)
    if not player_df.empty:
        for col in [*STAT_KEYS.keys()]:
            player_df[col] = pd.to_numeric(player_df[col], errors="coerce").fillna(0.0)
        denom = 2 * (player_df["fga"] + 0.44 * player_df["fta"])
        player_df["ts"] = (player_df["pts"] / denom).where(denom > 0, 0.0)
        player_df["efg"] = ((player_df["fgm"] + 0.5 * player_df["tpm"]) / player_df["fga"]).where(
            player_df["fga"] > 0, 0.0
        )
        player_df["usage_proxy"] = player_df["fga"] + 0.44 * player_df["fta"] + player_df["to"]
        player_df = player_df.sort_values(["player_id", "date", "gamecode_num"], kind="stable").reset_index(drop=True)

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    player_df.to_parquet(CURATED_DIR / "player_game.parquet", index=False)
    player_df.to_csv(CURATED_DIR / "player_game.csv", index=False)

    logger.info("Boxscores fetched=%s cached=%s for %s", fetched, cached, season_code)
    logger.info("player_game rows written: %s", len(player_df))
    return player_df


def main():
    parser = argparse.ArgumentParser(description="Fetch EuroLeague boxscores and build player_game table")
    parser.add_argument("--season", dest="season_code", default=DEFAULT_SEASON_CODE)
    args = parser.parse_args()
    run(season_code=args.season_code)


if __name__ == "__main__":
    main()
