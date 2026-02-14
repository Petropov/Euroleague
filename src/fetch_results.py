"""Fetch and curate EuroLeague game list XML feed."""

from __future__ import annotations

import argparse

import pandas as pd
import requests

from src.config import CURATED_DIR, DEFAULT_SEASON_CODE, LOG_DIR, RAW_DIR, RESULTS_ENDPOINT, TEAMS
from src.utils_http import build_session, get_logger, get_text, write_text_cache
from src.utils_parse import parse_results_xml


def run(season_code: str = DEFAULT_SEASON_CODE) -> pd.DataFrame:
    logger = get_logger(LOG_DIR / "pipeline.log")
    session = build_session()

    raw_path = RAW_DIR / f"results_{season_code}.xml"
    try:
        xml_text = get_text(session, RESULTS_ENDPOINT, params={"seasonCode": season_code})
        write_text_cache(raw_path, xml_text)
    except requests.RequestException as exc:
        if raw_path.exists():
            logger.warning("Using cached results XML for %s due to fetch error: %s", season_code, exc)
            xml_text = raw_path.read_text(encoding="utf-8")
        else:
            raise

    games = parse_results_xml(xml_text)
    games_df = pd.DataFrame(games)
    if games_df.empty:
        logger.warning("No games parsed from results feed for %s", season_code)
    else:
        games_df["season_code"] = season_code
        games_df = games_df[
            games_df["home_team_code"].isin(TEAMS) | games_df["away_team_code"].isin(TEAMS)
        ].copy()
        games_df = games_df.sort_values(["date", "gamecode_num"], kind="stable").reset_index(drop=True)

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    games_df.to_parquet(CURATED_DIR / "games.parquet", index=False)
    games_df.to_csv(CURATED_DIR / "games.csv", index=False)
    logger.info("Found %s PAN/OLY games for %s", len(games_df), season_code)
    return games_df


def main():
    parser = argparse.ArgumentParser(description="Fetch EuroLeague results feed and curate PAN/OLY games")
    parser.add_argument("--season", dest="season_code", default=DEFAULT_SEASON_CODE)
    args = parser.parse_args()
    run(season_code=args.season_code)


if __name__ == "__main__":
    main()
