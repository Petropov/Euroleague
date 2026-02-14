"""Fetch/cached boxscores and build player_game dataset."""

from __future__ import annotations

import argparse
import time

import pandas as pd
import requests

from src.config import BOXSCORE_ENDPOINT, CURATED_DIR, DEFAULT_SEASON_CODE, LOG_DIR, RAW_DIR, REQUEST_DELAY_SECONDS
from src.fetch_results import run as run_fetch_results
from src.utils_http import build_session, get_json, get_logger, read_json_cache, write_json_cache
from src.utils_parse import safe_float, safe_int, safe_text, slugify


STAT_KEYS = ["min", "pts", "reb", "ast", "stl", "blk", "to", "pf", "fgm", "fga", "tpm", "tpa", "ftm", "fta"]


def parse_minutes(value) -> float:
    text = safe_text(value)
    if not text:
        return 0.0
    if ":" not in text:
        return safe_float(text, default=0.0)
    mins, secs = text.split(":", maxsplit=1)
    return safe_float(mins, default=0.0) + safe_float(secs, default=0.0) / 60.0


def resolve_opponent(game_row: pd.Series, team_code: str) -> tuple[str, str]:
    if game_row["home_team_code"] == team_code:
        return game_row["away_team_code"], "home"
    return game_row["home_team_code"], "away"


def parse_player_rows(game_row: pd.Series, payload: dict, season_code: str, logger) -> list[dict]:
    rows: list[dict] = []
    seen = set()

    statblocks = payload.get("Stats") if isinstance(payload, dict) else None
    if not isinstance(statblocks, list):
        logger.warning("Boxscore payload missing Stats for game %s", game_row["gamecode_full"])
        return rows

    home_code = safe_text(game_row["home_team_code"]).upper()
    away_code = safe_text(game_row["away_team_code"]).upper()

    for team_node in statblocks:
        if not isinstance(team_node, dict):
            continue

        block_team_code = safe_text(team_node.get("Team") or team_node.get("team") or team_node.get("Code")).upper()
        players = team_node.get("PlayersStats")
        if not isinstance(players, list):
            continue

        for player in players:
            if not isinstance(player, dict):
                continue

            team_code = safe_text(player.get("Team") or block_team_code).upper()
            if team_code not in {home_code, away_code}:
                continue

            opponent_code, home_away = resolve_opponent(game_row, team_code)
            player_name = safe_text(player.get("Player"), default="Unknown")
            player_id = safe_text(player.get("Player_ID")) or slugify(player_name)

            fgm2 = safe_float(player.get("FieldGoalsMade2"), default=0.0)
            fga2 = safe_float(player.get("FieldGoalsAttempted2"), default=0.0)
            tpm = safe_float(player.get("FieldGoalsMade3"), default=0.0)
            tpa = safe_float(player.get("FieldGoalsAttempted3"), default=0.0)
            ftm = safe_float(player.get("FreeThrowsMade"), default=0.0)
            fta = safe_float(player.get("FreeThrowsAttempted"), default=0.0)
            reb = safe_float(player.get("ReboundsTotal"), default=None)
            if reb is None:
                reb = safe_float(player.get("ReboundsOffensive"), default=0.0) + safe_float(
                    player.get("ReboundsDefensive"), default=0.0
                )

            row = {
                "season_code": season_code,
                "gamecode_num": safe_int(game_row["gamecode_num"]),
                "date": game_row["date"],
                "team_code": team_code,
                "opponent_code": opponent_code,
                "home_away": home_away,
                "player_id": player_id,
                "player_name": player_name,
                "min": parse_minutes(player.get("Minutes")),
                "pts": safe_float(player.get("Points"), default=0.0),
                "ast": safe_float(player.get("Assists"), default=0.0),
                "reb": safe_float(reb, default=0.0),
                "stl": safe_float(player.get("Steals"), default=0.0),
                "blk": safe_float(player.get("BlocksFavour"), default=0.0),
                "to": safe_float(player.get("Turnovers"), default=0.0),
                "pf": safe_float(player.get("FoulsCommited"), default=0.0),
                "fgm": fgm2 + tpm,
                "fga": fga2 + tpa,
                "tpm": tpm,
                "tpa": tpa,
                "ftm": ftm,
                "fta": fta,
            }
            key = (row["season_code"], row["gamecode_num"], row["team_code"], row["player_id"])
            if key not in seen:
                rows.append(row)
                seen.add(key)

    has_playersstats = any(
        isinstance(block, dict) and isinstance(block.get("PlayersStats"), list) and block.get("PlayersStats")
        for block in statblocks
    )
    if has_playersstats and not rows:
        seen_keys = sorted(
            {
                key
                for block in statblocks
                if isinstance(block, dict)
                for player in block.get("PlayersStats", [])
                if isinstance(player, dict)
                for key in player.keys()
            }
        )
        raise RuntimeError(
            f"Boxscore parser produced 0 rows for {game_row['gamecode_full']} despite PlayersStats. Keys: {seen_keys}"
        )
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
        for col in STAT_KEYS:
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
