"""Schedule fetcher with fallback for next Panathinaikos game."""

from __future__ import annotations

import argparse
from datetime import datetime
import xml.etree.ElementTree as ET

import pandas as pd

from src.config import CURATED_DIR, DEFAULT_SEASON_CODE
from src.utils_http import build_session

SCHEDULE_ENDPOINT = "https://api-live.euroleague.net/v1/schedules"


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _first_text(node: ET.Element, *names: str) -> str:
    names_lc = {name.lower() for name in names}
    for child in list(node):
        if _strip_ns(child.tag).lower() in names_lc:
            return (child.text or "").strip()
    return ""


def _parse_game_datetime(date_text: str, startime_text: str) -> datetime | None:
    if not date_text:
        return None
    try:
        parsed_date = datetime.strptime(date_text.strip(), "%b %d, %Y")
    except ValueError:
        return None

    if startime_text:
        try:
            parsed_time = datetime.strptime(startime_text.strip(), "%H:%M")
            return parsed_date.replace(hour=parsed_time.hour, minute=parsed_time.minute)
        except ValueError:
            pass
    return parsed_date.replace(hour=12, minute=0)


def _parse_schedule_xml(xml_text: str, season_code: str) -> pd.DataFrame:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pd.DataFrame()

    items = [node for node in root.iter() if _strip_ns(node.tag).lower() == "item"]
    out: list[dict] = []
    for item in items:
        gamecode = _first_text(item, "gamecode", "gameCode")
        date_text = _first_text(item, "date")
        startime_text = _first_text(item, "startime", "startTime")
        homecode = _first_text(item, "homecode", "homeCode").upper()
        awaycode = _first_text(item, "awaycode", "awayCode").upper()
        out.append(
            {
                "season_code": season_code,
                "gamecode_full": gamecode,
                "gamecode_num": pd.to_numeric(str(gamecode).split("_")[-1], errors="coerce"),
                "date": _parse_game_datetime(date_text, startime_text),
                "date_text": date_text,
                "startime": startime_text,
                "home_team_code": homecode,
                "away_team_code": awaycode,
                "home_team_name": _first_text(item, "hometeam", "homeTeam"),
                "away_team_name": _first_text(item, "awayteam", "awayTeam"),
                "round": _first_text(item, "round", "gameday"),
            }
        )
    return pd.DataFrame(out)


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




def _to_utc_naive(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_localize(None)


def fetch_schedule(season_code: str = DEFAULT_SEASON_CODE) -> pd.DataFrame:
    session = build_session()
    resp = session.get(SCHEDULE_ENDPOINT, params={"seasonCode": season_code}, timeout=30)
    resp.raise_for_status()
    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "xml" in content_type or resp.text.lstrip().startswith("<"):
        df = _parse_schedule_xml(resp.text, season_code)
    else:
        df = _normalize_schedule(resp.json(), season_code)
    if not df.empty:
        df["date"] = _to_utc_naive(df["date"])
    return df


def next_pan_game(season_code: str = DEFAULT_SEASON_CODE, now_utc: datetime | None = None) -> dict | None:
    now_utc = now_utc or datetime.utcnow()
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
        sched["date"] = _to_utc_naive(sched["date"])

    sched["date"] = _to_utc_naive(sched["date"])

    scoped = sched[
        ((sched["home_team_code"] == "PAN") | (sched["away_team_code"] == "PAN"))
        & (sched["date"] > pd.Timestamp(now_utc))
    ].copy()

    next_desc = "None"
    if not scoped.empty:
        sample = scoped.sort_values(["date", "gamecode_num"], kind="stable").iloc[0]
        date_str = sample["date"].strftime("%Y-%m-%d %H:%M") if pd.notna(sample["date"]) else "Unknown"
        next_desc = (
            f"{sample.get('gamecode_full', '')} {date_str} "
            f"{sample.get('home_team_code', '')}-{sample.get('away_team_code', '')}"
        )
    print(
        f"Schedules parsed: {len(sched)} items; PAN future games: {len(scoped)}; next PAN: {next_desc}"
    )

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
