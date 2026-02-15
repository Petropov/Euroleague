"""Run the full EuroLeague data pipeline."""

from __future__ import annotations

import argparse

from src.build_team_features import run as run_team_features
from src.build_team_game import run as run_team_game
from src.config import DEFAULT_SEASON_CODE
from src.fetch_boxscores import run as run_fetch_boxscores
from src.fetch_results import run as run_fetch_results
from src.train_win_model import run as run_train_win_model


def run(season_code: str = DEFAULT_SEASON_CODE):
    run_fetch_results(season_code=season_code)
    run_fetch_boxscores(season_code=season_code)
    run_team_game(season_code=season_code)
    run_team_features(season_code=season_code)
    run_train_win_model(season_code=season_code)


def main():
    parser = argparse.ArgumentParser(description="Run full EuroLeague dataset pipeline")
    parser.add_argument("--season", dest="season_code", default=DEFAULT_SEASON_CODE)
    args = parser.parse_args()
    run(season_code=args.season_code)


if __name__ == "__main__":
    main()
