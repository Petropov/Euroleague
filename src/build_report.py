"""Build self-contained HTML player report for PAN/OLY."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import BASE_DIR, CURATED_DIR, TEAMS

REPORTS_DIR = BASE_DIR / "reports"


def _read_curated(name: str) -> pd.DataFrame:
    """Read curated dataset from CSV, with parquet fallback."""
    csv_path = CURATED_DIR / f"{name}.csv"
    parquet_path = CURATED_DIR / f"{name}.parquet"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"Could not find curated dataset for '{name}'")


def _prepare_latest_by_player(player_gog: pd.DataFrame, season_code: str) -> pd.DataFrame:
    team_scope = sorted(TEAMS)
    scoped = player_gog[
        (player_gog["season_code"] == season_code) & (player_gog["team_code"].isin(team_scope))
    ].copy()
    if scoped.empty:
        return scoped

    scoped["date"] = pd.to_datetime(scoped["date"], errors="coerce", utc=True)
    scoped = scoped.sort_values(["team_code", "player_id", "date", "gamecode_num"])
    return scoped.drop_duplicates(["team_code", "player_id"], keep="last")


def _format_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    if df.empty:
        header_cells = "".join(f"<th>{c}</th>" for c in cols)
        return (
            "<table><thead><tr>"
            f"{header_cells}"
            "</tr></thead><tbody><tr><td colspan='"
            f"{len(cols)}'>No rows match filter.</td></tr></tbody></table>"
        )

    render_df = df.loc[:, cols].copy()
    if "date" in render_df.columns:
        render_df["date"] = pd.to_datetime(render_df["date"], errors="coerce", utc=True).dt.strftime(
            "%Y-%m-%d"
        )
    numeric_cols = render_df.select_dtypes(include="number").columns
    for col in numeric_cols:
        render_df[col] = render_df[col].map(lambda v: f"{v:.3f}".rstrip("0").rstrip("."))

    return render_df.to_html(index=False, escape=False, border=0, classes="report-table")


def _build_team_section(
    latest_df: pd.DataFrame,
    team_code: str,
    title: str,
    note: str,
    filter_expr,
    sort_cols: list[str],
    ascending: list[bool],
    columns: Iterable[str],
) -> str:
    team_df = latest_df[latest_df["team_code"] == team_code]
    filtered = team_df[filter_expr(team_df)]
    if not filtered.empty:
        filtered = filtered.sort_values(sort_cols, ascending=ascending)

    return (
        f"<h4>{team_code}</h4>"
        f"<p class='note'>{note}</p>"
        f"{_format_table(filtered, columns)}"
    )


def build_report(season_code: str) -> tuple[Path, Path]:
    games = _read_curated("games")
    player_game = _read_curated("player_game")
    player_gog = _read_curated("player_gog")

    games_season = games[games["season_code"] == season_code].copy()
    pan_oly_games = games_season[
        games_season["home_team_code"].isin(TEAMS) | games_season["away_team_code"].isin(TEAMS)
    ]
    player_game_season = player_game[player_game["season_code"] == season_code]
    player_gog_season = player_gog[player_gog["season_code"] == season_code]

    latest = _prepare_latest_by_player(player_gog, season_code)
    generated_at = datetime.now(timezone.utc)
    timestamp = generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    warning_banner = ""
    if len(player_gog_season) == 0:
        warning_banner = (
            "<div class='warning'>Warning: player_gog has 0 rows for this season. "
            "Tables are intentionally empty.</div>"
        )

    points_cols = [
        "date",
        "player_name",
        "min",
        "pts",
        "gog_pts",
        "r5_pts",
        "ts",
        "r5_ts",
        "usage_proxy",
        "r5_usage_proxy",
    ]
    eff_cols = ["date", "player_name", "min", "ts", "gog_ts", "r5_ts", "usage_proxy"]
    watch_cols = ["date", "player_name", "min", "usage_proxy", "ts", "pts", "ast", "to"]

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>EuroLeague PAN/OLY Player Report — {season_code}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2, h3, h4 {{ margin: 0.4rem 0; }}
    a {{ color: #0f4c81; }}
    .muted {{ color: #6b7280; }}
    .summary, .section {{ margin-top: 20px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #fafafa; }}
    .value {{ font-size: 1.3rem; font-weight: 700; }}
    .note {{ display: inline-block; background: #eef6ff; border: 1px solid #c9def8; color: #184e87; border-radius: 999px; padding: 4px 10px; font-size: 0.85rem; margin: 8px 0; }}
    .warning {{ background: #fff4e5; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin-top: 16px; font-weight: 600; color: #92400e; }}
    table {{ width: 100%; border-collapse: collapse; margin: 8px 0 20px; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; white-space: nowrap; }}
    thead th {{ position: sticky; top: 0; background: #f8fafc; z-index: 1; }}
    .table-wrap {{ overflow-x: auto; }}
    #definitions ul {{ margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>EuroLeague PAN/OLY Player Report — Season {season_code} — Generated {timestamp}</h1>
  <p class=\"muted\">Jump to <a href=\"#definitions\">definitions</a>.</p>

  {warning_banner}

  <section class=\"summary\">
    <h2>Data freshness</h2>
    <div class=\"summary-grid\">
      <div class=\"card\"><div>Total games parsed</div><div class=\"value\">{len(games_season)}</div></div>
      <div class=\"card\"><div>PAN/OLY games found</div><div class=\"value\">{len(pan_oly_games)}</div></div>
      <div class=\"card\"><div>player_game rows</div><div class=\"value\">{len(player_game_season)}</div></div>
      <div class=\"card\"><div>player_gog rows</div><div class=\"value\">{len(player_gog_season)}</div></div>
    </div>
  </section>

  <section class=\"section\">
    <h2>Top movers — Points (GoG)</h2>
    <div class=\"table-wrap\">
      {_build_team_section(latest, "PAN", "Top movers — Points (GoG)", "Filter: latest game per player, min ≥ 5.", lambda d: d["min"] >= 5, ["gog_pts", "pts"], [False, False], points_cols)}
      {_build_team_section(latest, "OLY", "Top movers — Points (GoG)", "Filter: latest game per player, min ≥ 5.", lambda d: d["min"] >= 5, ["gog_pts", "pts"], [False, False], points_cols)}
    </div>
  </section>

  <section class=\"section\">
    <h2>Top movers — Efficiency (GoG TS)</h2>
    <div class=\"table-wrap\">
      {_build_team_section(latest, "PAN", "Top movers — Efficiency (GoG TS)", "Filter: latest game per player, min ≥ 10 and usage_proxy ≥ 6. Sorted by gog_ts desc.", lambda d: (d["min"] >= 10) & (d["usage_proxy"] >= 6), ["gog_ts", "ts"], [False, False], eff_cols)}
      {_build_team_section(latest, "OLY", "Top movers — Efficiency (GoG TS)", "Filter: latest game per player, min ≥ 10 and usage_proxy ≥ 6. Sorted by gog_ts desc.", lambda d: (d["min"] >= 10) & (d["usage_proxy"] >= 6), ["gog_ts", "ts"], [False, False], eff_cols)}
    </div>
  </section>

  <section class=\"section\">
    <h2>High usage, low efficiency (watchlist)</h2>
    <div class=\"table-wrap\">
      {_build_team_section(latest, "PAN", "High usage, low efficiency (watchlist)", "Filter: latest game per player, usage_proxy ≥ 12, ts < 0.50, min ≥ 10.", lambda d: (d["usage_proxy"] >= 12) & (d["ts"] < 0.50) & (d["min"] >= 10), ["usage_proxy", "ts"], [False, True], watch_cols)}
      {_build_team_section(latest, "OLY", "High usage, low efficiency (watchlist)", "Filter: latest game per player, usage_proxy ≥ 12, ts < 0.50, min ≥ 10.", lambda d: (d["usage_proxy"] >= 12) & (d["ts"] < 0.50) & (d["min"] >= 10), ["usage_proxy", "ts"], [False, True], watch_cols)}
    </div>
  </section>

  <section id=\"definitions\" class=\"section\">
    <h2>Footnotes / Definitions</h2>
    <ul>
      <li><strong>TS (True Shooting):</strong> PTS / (2*(FGA + 0.44*FTA))</li>
      <li><strong>eFG:</strong> (FGM + 0.5*3PM) / FGA</li>
      <li><strong>usage_proxy:</strong> FGA + 0.44*FTA + TO</li>
      <li><strong>GoG (game-over-game):</strong> current stat minus previous game stat for same player_id</li>
      <li><strong>r5_*:</strong> rolling 5-game mean for that stat</li>
      <li><strong>min conversion:</strong> MM:SS converted to minutes as float</li>
    </ul>
  </section>
</body>
</html>
"""

    season_dir = REPORTS_DIR / season_code
    season_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    latest_path = REPORTS_DIR / "latest.html"
    dated_path = season_dir / f"report_{generated_at.strftime('%Y-%m-%d')}.html"
    latest_path.write_text(html, encoding="utf-8")
    dated_path.write_text(html, encoding="utf-8")
    return latest_path, dated_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PAN/OLY HTML report")
    parser.add_argument("--season", required=True, help="Season code, e.g. E2025")
    args = parser.parse_args()

    latest, dated = build_report(args.season)
    print(f"Wrote {latest}")
    print(f"Wrote {dated}")


if __name__ == "__main__":
    main()
