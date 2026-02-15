"""Fetch and curate EuroLeague game list XML feed."""

from __future__ import annotations

import argparse

import pandas as pd
import requests

from src.config import CURATED_DIR, DEFAULT_SEASON_CODE, LOG_DIR, RAW_DIR, RESULTS_ENDPOINT, TEAMS
from src.utils_http import build_session, get_logger, write_text_cache
from src.utils_parse import parse_results_xml


def _to_iso_datetime(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    try:
        return pd.to_datetime(text, utc=True).isoformat()
    except Exception:
        return text


def _require_parsed_games(xml_text: str, season_code: str, status_code: int) -> list[dict]:
    games = parse_results_xml(xml_text)
    if games:
        return games
    preview = xml_text[:200].replace("\n", " ")
    raise RuntimeError(
        "No games parsed from EuroLeague results feed. "
        f"status_code={status_code}; response_preview='{preview}'; "
        "hint: seasonCode should look like 'E2025'."
    )


def run(season_code: str = DEFAULT_SEASON_CODE) -> pd.DataFrame:
    logger = get_logger(LOG_DIR / "pipeline.log")
    session = build_session()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / f"results_{season_code}.xml"
    status_code = -1

    try:
        response = session.get(RESULTS_ENDPOINT, params={"seasonCode": season_code}, timeout=30)
        status_code = response.status_code
        response.raise_for_status()
        xml_text = response.text
        write_text_cache(raw_path, xml_text)
    except requests.RequestException as exc:
        if raw_path.exists():
            logger.warning("Using cached results XML for %s due to fetch error: %s", season_code, exc)
            xml_text = raw_path.read_text(encoding="utf-8")
        else:
            raise

    games_df = pd.DataFrame(_require_parsed_games(xml_text, season_code, status_code))
    games_df["season_code"] = season_code
    games_df["date"] = games_df["date"].map(_to_iso_datetime)
    games_df = games_df.sort_values(["date", "gamecode_num"], kind="stable").reset_index(drop=True)

    pan_oly_df = games_df[
        games_df["home_team_code"].isin(TEAMS) | games_df["away_team_code"].isin(TEAMS)
    ].copy()
    pan_oly_df = pan_oly_df.sort_values(["date", "gamecode_num"], kind="stable").reset_index(drop=True)

    logger.info("Total games parsed for %s: %s", season_code, len(games_df))
    logger.info("PAN/OLY games found for %s: %s", season_code, len(pan_oly_df))

    games_df.to_csv(CURATED_DIR / "games_all.csv", index=False)
    pan_oly_df.to_csv(CURATED_DIR / "games.csv", index=False)
    return games_df


def main():
    parser = argparse.ArgumentParser(description="Fetch EuroLeague results feed and curate games")
    parser.add_argument("--season", dest="season_code", default=DEFAULT_SEASON_CODE)
    args = parser.parse_args()
    run(season_code=args.season_code)


if __name__ == "__main__":
    main()
