"""Build all-team team_game features from player boxscores."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import CURATED_DIR, DEFAULT_SEASON_CODE


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den).where(den > 0, 0.0)


def run(season_code: str = DEFAULT_SEASON_CODE) -> pd.DataFrame:
    player = pd.read_csv(CURATED_DIR / "player_game_all.csv")
    games = pd.read_csv(CURATED_DIR / "games_all.csv")

    if "season_code" in player.columns:
        player = player[player["season_code"] == season_code].copy()
    if "season_code" in games.columns:
        games = games[games["season_code"] == season_code].copy()

    num_cols = ["pts", "fga", "fgm", "tpa", "tpm", "fta", "ftm", "to", "usage_proxy"]
    for col in num_cols:
        player[col] = pd.to_numeric(player[col], errors="coerce").fillna(0.0)

    grouped = (
        player.groupby(["season_code", "gamecode_num", "date", "team_code", "opponent_code", "home_away"], as_index=False)[num_cols]
        .sum()
        .rename(
            columns={
                "pts": "team_pts",
                "fga": "team_fga",
                "fgm": "team_fgm",
                "tpa": "team_tpa",
                "tpm": "team_tpm",
                "fta": "team_fta",
                "ftm": "team_ftm",
                "to": "team_to",
                "usage_proxy": "team_usage",
            }
        )
    )

    grouped["team_ts"] = _safe_div(grouped["team_pts"], 2 * (grouped["team_fga"] + 0.44 * grouped["team_fta"]))
    grouped["team_efg"] = _safe_div(grouped["team_fgm"] + 0.5 * grouped["team_tpm"], grouped["team_fga"])

    top3_usage = (
        player.sort_values(["season_code", "gamecode_num", "team_code", "usage_proxy"], ascending=[True, True, True, False])
        .groupby(["season_code", "gamecode_num", "team_code"], as_index=False)
        .head(3)
        .groupby(["season_code", "gamecode_num", "team_code"], as_index=False)["usage_proxy"]
        .sum()
        .rename(columns={"usage_proxy": "top3_usage_sum"})
    )

    grouped = grouped.merge(top3_usage, on=["season_code", "gamecode_num", "team_code"], how="left")
    grouped["top3_usage_sum"] = grouped["top3_usage_sum"].fillna(0.0)
    grouped["usage_conc_top3"] = _safe_div(grouped["top3_usage_sum"], grouped["team_usage"])

    games_keep = games[["season_code", "gamecode_num", "home_team_code", "away_team_code", "home_score", "away_score", "date"]].copy()
    games_keep["home_score"] = pd.to_numeric(games_keep["home_score"], errors="coerce")
    games_keep["away_score"] = pd.to_numeric(games_keep["away_score"], errors="coerce")
    grouped = grouped.merge(games_keep, on=["season_code", "gamecode_num"], how="left", suffixes=("", "_game"))

    grouped["is_home"] = grouped["team_code"] == grouped["home_team_code"]
    grouped["team_score"] = grouped["home_score"].where(grouped["is_home"], grouped["away_score"])
    grouped["opp_score"] = grouped["away_score"].where(grouped["is_home"], grouped["home_score"])
    grouped["margin"] = grouped["team_score"] - grouped["opp_score"]
    grouped["win_flag"] = (grouped["margin"] > 0).astype(int)

    grouped = grouped.sort_values(["date", "gamecode_num", "team_code"], kind="stable").reset_index(drop=True)
    grouped.to_csv(CURATED_DIR / "team_game.csv", index=False)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build team_game.csv from player_game_all.csv")
    parser.add_argument("--season", dest="season_code", default=DEFAULT_SEASON_CODE)
    args = parser.parse_args()
    run(args.season_code)


if __name__ == "__main__":
    main()
