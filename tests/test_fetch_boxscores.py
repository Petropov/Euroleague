import json
import logging
import pathlib
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.fetch_boxscores import parse_player_rows


def test_parse_player_rows_from_stats_playersstats_fixture():
    with open("tests/fixtures/boxscore_sample.json", encoding="utf-8") as f:
        payload = json.load(f)

    game_row = pd.Series(
        {
            "gamecode_full": "E2025_1",
            "gamecode_num": 1,
            "date": "2024-10-03",
            "home_team_code": "BAS",
            "away_team_code": "OLY",
        }
    )

    rows = parse_player_rows(game_row, payload, "E2025", logging.getLogger("test"))

    assert len(rows) > 0
    assert any(row["player_name"] == "HOWARD, MARKUS" for row in rows)
