"""Run the core EuroLeague PAN/OLY pipeline."""

from __future__ import annotations

import argparse

from src.build_player_gog import run as run_build_player_gog
from src.config import DEFAULT_SEASON_CODE
from src.fetch_boxscores import run as run_fetch_boxscores
from src.fetch_results import run as run_fetch_results


def run(season_code: str = DEFAULT_SEASON_CODE):
    run_fetch_results(season_code=season_code)
    run_fetch_boxscores(season_code=season_code)
    run_build_player_gog(season_code=season_code)


def main():
    parser = argparse.ArgumentParser(description="Run core EuroLeague dataset pipeline")
    parser.add_argument("--season", dest="season_code", default=DEFAULT_SEASON_CODE)
    args = parser.parse_args()
    run(season_code=args.season_code)


if __name__ == "__main__":
    main()
