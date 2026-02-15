"""Schedule fetcher with fallback for next Panathinaikos game."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import CURATED_DIR, DEFAULT_SEASON_CODE
from src.utils_http import build_session

SCHEDULE_ENDPOINT = "https://api-live.euroleague.net/v1/schedules"


def _normalize_schedule(payload: dict, season_code: str) -> pd.DataFrame:
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        rows = payload if isinstance(payload, list) else []
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        home = (row.get("homeTeam") or row.get("homeClub") or row.get("homeCode") or row.get("homeTeamCode") or "").upper()
        away = (row.get("awayTeam") or row.get("awayClub") or row.get("awayCode") or row.get("awayTeamCode") or "").upper()
        date = row.get("date") or row.get("utcDate") or row.get("startDate")
        gamecode = row.get("gameCode") or row.get("gamecode")
        out.append(
            {
                "season_code": season_code,
                "gamecode_full": str(gamecode or ""),
                "gamecode_num": pd.to_numeric(str(gamecode).split("_")[-1], errors="coerce"),
                "date": date,
                "home_team_code": home,
                "away_team_code": away,
            }
        )
    return pd.DataFrame(out)


def fetch_schedule(season_code: str = DEFAULT_SEASON_CODE) -> pd.DataFrame:
    session = build_session()
    resp = session.get(SCHEDULE_ENDPOINT, params={"seasonCode": season_code}, timeout=30)
    resp.raise_for_status()
    df = _normalize_schedule(resp.json(), season_code)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df


def next_pan_game(season_code: str = DEFAULT_SEASON_CODE, now_utc: pd.Timestamp | None = None) -> dict | None:
    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    sched = pd.DataFrame()
    try:
        sched = fetch_schedule(season_code)
    except Exception:
        sched = pd.DataFrame()

    if sched.empty:
        games_all = pd.read_csv(CURATED_DIR / "games_all.csv")
        if "season_code" in games_all.columns:
            games_all = games_all[games_all["season_code"] == season_code].copy()
        sched = games_all[["season_code", "gamecode_full", "gamecode_num", "date", "home_team_code", "away_team_code"]].copy()
        sched["date"] = pd.to_datetime(sched["date"], errors="coerce", utc=True)

    scoped = sched[
        ((sched["home_team_code"] == "PAN") | (sched["away_team_code"] == "PAN")) & (sched["date"] > now_utc)
    ].copy()
    if scoped.empty:
        return None
    row = scoped.sort_values(["date", "gamecode_num"], kind="stable").iloc[0]
    return row.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch schedule and print next PAN game")
    parser.add_argument("--season", dest="season_code", default=DEFAULT_SEASON_CODE)
    args = parser.parse_args()
    print(next_pan_game(args.season_code))


if __name__ == "__main__":
    main()
