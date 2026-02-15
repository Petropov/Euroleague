"""Build leak-free pre-game team rolling features and training matrix."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import CURATED_DIR, DEFAULT_SEASON_CODE

ROLL_COLS = ["team_ts", "team_efg", "team_to", "margin", "usage_conc_top3"]


def run(season_code: str = DEFAULT_SEASON_CODE) -> pd.DataFrame:
    team = pd.read_csv(CURATED_DIR / "team_game.csv")
    if "season_code" in team.columns:
        team = team[team["season_code"] == season_code].copy()

    team["date"] = pd.to_datetime(team["date"], errors="coerce", utc=True)
    team = team.sort_values(["team_code", "date", "gamecode_num"], kind="stable")

    for col in ROLL_COLS:
        team[col] = pd.to_numeric(team[col], errors="coerce")
        team[f"r5_{col}_pre"] = (
            team.groupby("team_code", sort=False)[col].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    team["games_played_pre"] = team.groupby("team_code", sort=False).cumcount()

    keep = [
        "season_code",
        "gamecode_num",
        "date",
        "team_code",
        "home_away",
        "win_flag",
        "games_played_pre",
    ] + [f"r5_{col}_pre" for col in ROLL_COLS]
    team_roll = team[keep].copy()

    home = team_roll[team_roll["home_away"] == "home"].copy()
    away = team_roll[team_roll["home_away"] == "away"].copy()

    merged = home.merge(
        away,
        on=["season_code", "gamecode_num"],
        suffixes=("_home", "_away"),
        how="inner",
    )

    feature_map = {
        "r5_team_ts_pre": "d_r5_ts_pre",
        "r5_team_efg_pre": "d_r5_efg_pre",
        "r5_team_to_pre": "d_r5_to_pre",
        "r5_margin_pre": "d_r5_margin_pre",
        "r5_usage_conc_top3_pre": "d_r5_usage_conc_pre",
    }

    for source_col, out_col in feature_map.items():
        merged[out_col] = merged[f"{source_col}_home"].fillna(0.0) - merged[f"{source_col}_away"].fillna(0.0)

    train = pd.DataFrame(
        {
            "season_code": merged["season_code"],
            "date": merged["date_home"],
            "gamecode_num": merged["gamecode_num"],
            "home_team_code": merged["team_code_home"],
            "away_team_code": merged["team_code_away"],
            "home": 1,
            "y": merged["win_flag_home"].astype(int),
            "home_games_played_pre": merged["games_played_pre_home"],
            "away_games_played_pre": merged["games_played_pre_away"],
        }
    )
    for out_col in feature_map.values():
        train[out_col] = merged[out_col]

    train = train.sort_values(["date", "gamecode_num"], kind="stable").reset_index(drop=True)
    train.to_csv(CURATED_DIR / "model_train.csv", index=False)
    return train


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model_train.csv from team_game.csv")
    parser.add_argument("--season", dest="season_code", default=DEFAULT_SEASON_CODE)
    args = parser.parse_args()
    run(args.season_code)


if __name__ == "__main__":
    main()
