"""Build game-over-game and rolling metrics per player."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import CURATED_DIR, DEFAULT_SEASON_CODE, LOG_DIR
from src.utils_http import get_logger

TARGET_STATS = ["min", "pts", "reb", "ast", "stl", "blk", "to", "pf", "ts", "efg", "usage_proxy"]


def run(season_code: str = DEFAULT_SEASON_CODE) -> pd.DataFrame:
    logger = get_logger(LOG_DIR / "pipeline.log")
    player_path = CURATED_DIR / "player_game.parquet"
    if not player_path.exists():
        raise FileNotFoundError("player_game.parquet not found. Run fetch_boxscores first.")

    df = pd.read_parquet(player_path)
    if "season_code" in df.columns:
        df = df[df["season_code"] == season_code].copy()

    if df.empty:
        logger.warning("No player_game rows for %s; writing empty player_gog.", season_code)
        out = pd.DataFrame()
        out.to_parquet(CURATED_DIR / "player_gog.parquet", index=False)
        out.to_csv(CURATED_DIR / "player_gog.csv", index=False)
        return out

    df = df.sort_values(["player_id", "date", "gamecode_num"], kind="stable").reset_index(drop=True)

    for stat in TARGET_STATS:
        if stat not in df.columns:
            df[stat] = 0.0
        group = df.groupby("player_id", sort=False)[stat]
        df[f"gog_{stat}"] = group.diff().fillna(0.0)
        df[f"r5_{stat}"] = group.transform(lambda s: s.rolling(5, min_periods=1).mean())

    df.to_parquet(CURATED_DIR / "player_gog.parquet", index=False)
    df.to_csv(CURATED_DIR / "player_gog.csv", index=False)
    logger.info("player_gog rows written: %s", len(df))
    return df


def main():
    parser = argparse.ArgumentParser(description="Build player game-over-game and rolling metrics")
    parser.add_argument("--season", dest="season_code", default=DEFAULT_SEASON_CODE)
    args = parser.parse_args()
    run(season_code=args.season_code)


if __name__ == "__main__":
    main()
