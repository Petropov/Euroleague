"""Run the full EuroLeague PAN/OLY pipeline."""

from __future__ import annotations

import argparse

from src.build_player_gog import run as run_player_gog
from src.config import DEFAULT_SEASON_CODE, LOG_DIR, TEAMS
from src.fetch_boxscores import run as run_fetch_boxscores
from src.fetch_results import run as run_fetch_results
from src.utils_http import get_logger


def run(season_code: str = DEFAULT_SEASON_CODE):
    logger = get_logger(LOG_DIR / "pipeline.log")
    games_df = run_fetch_results(season_code=season_code)

    total_games = len(games_df)
    if total_games == 0:
        raise RuntimeError(f"No games were parsed for {season_code}; failing fast.")

    pan_oly_count = int(
        ((games_df["home_team_code"].isin(TEAMS)) | (games_df["away_team_code"].isin(TEAMS))).sum()
    )
    if pan_oly_count == 0:
        logger.warning("No PAN/OLY games found for %s", season_code)

    run_fetch_boxscores(season_code=season_code)
    run_player_gog(season_code=season_code)


def main():
    parser = argparse.ArgumentParser(description="Run complete PAN/OLY EuroLeague dataset pipeline")
    parser.add_argument("--season", dest="season_code", default=DEFAULT_SEASON_CODE)
    args = parser.parse_args()
    run(season_code=args.season_code)


if __name__ == "__main__":
    main()
