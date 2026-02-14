"""Build self-contained HTML player report for PAN/OLY."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
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


def _safe_date_text(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return value.strftime("%Y-%m-%d")


def _extract_game_date_range(games_df: pd.DataFrame) -> tuple[str, str]:
    if "date" not in games_df.columns or games_df.empty:
        return "N/A", "N/A"

    parsed = pd.to_datetime(games_df["date"], errors="coerce", utc=True).dropna()
    if parsed.empty:
        return "N/A", "N/A"

    return _safe_date_text(parsed.min()), _safe_date_text(parsed.max())


def _compute_momentum_watch(latest_df: pd.DataFrame, team_code: str) -> pd.DataFrame:
    team_df = latest_df[latest_df["team_code"] == team_code].copy()
    if team_df.empty:
        return team_df

    scoped = team_df[(team_df["min"] >= 10) & (team_df["usage_proxy"] >= 6)].copy()
    if scoped.empty:
        return scoped

    eps = 1e-9
    usage_denom = max(scoped["gog_usage_proxy"].abs().max(), eps)
    min_denom = max(scoped["gog_min"].abs().max(), eps)

    scoped["gog_ts_capped"] = scoped["gog_ts"].clip(-0.5, 0.5)
    scoped["gog_usage_proxy_norm"] = scoped["gog_usage_proxy"] / usage_denom
    scoped["gog_min_norm"] = scoped["gog_min"] / min_denom
    scoped["FormScore"] = (
        0.5 * scoped["gog_ts_capped"]
        + 0.3 * scoped["gog_usage_proxy_norm"]
        + 0.2 * scoped["gog_min_norm"]
    )

    return scoped.sort_values(["FormScore", "gog_ts"], ascending=[False, False]).head(10)


def _compute_top_usage(latest_df: pd.DataFrame, team_code: str) -> pd.DataFrame:
    team_df = latest_df[latest_df["team_code"] == team_code].copy()
    if team_df.empty:
        return team_df
    scoped = team_df[team_df["min"] >= 10].copy()
    if scoped.empty:
        return scoped
    return scoped.sort_values(["usage_proxy", "pts"], ascending=[False, False]).head(8)


def _latest_team_summary(latest_df: pd.DataFrame, team_code: str) -> dict[str, str]:
    team_df = latest_df[latest_df["team_code"] == team_code].copy()
    if team_df.empty:
        return {
            "top_scoring": "None",
            "top_eff": "None",
            "highest_usage": "None",
            "watchlist": "None",
        }

    min10 = team_df[team_df["min"] >= 10]
    eff_scope = min10[min10["usage_proxy"] >= 6]
    watch_scope = min10[(min10["usage_proxy"] >= 12) & (min10["ts"] < 0.50)]

    def _pick_line(df: pd.DataFrame, sort_col: str, metric_fmt: str, label: str) -> str:
        if df.empty:
            return "None"
        row = df.sort_values(sort_col, ascending=False).iloc[0]
        return f"{escape(str(row['player_name']))} ({label} {metric_fmt.format(row[sort_col])}, MIN {row['min']:.1f})"

    return {
        "top_scoring": _pick_line(min10, "gog_pts", "{:+.2f}", "GoG PTS"),
        "top_eff": _pick_line(eff_scope, "gog_ts", "{:+.3f}", "GoG TS"),
        "highest_usage": _pick_line(min10, "usage_proxy", "{:.1f}", "USG"),
        "watchlist": _pick_line(watch_scope, "usage_proxy", "{:.1f}", "USG"),
    }


def _render_cell(value: object, col: str) -> str:
    if pd.isna(value):
        return ""

    class_name = ""
    if col == "ts":
        if value >= 0.60:
            class_name = "good"
        elif value < 0.52:
            class_name = "bad"
        else:
            class_name = "ok"
    elif col == "gog_ts":
        if value >= 0.10:
            class_name = "good"
        elif value <= -0.10:
            class_name = "bad"
        else:
            class_name = "ok"
    elif col == "gog_pts":
        if value >= 5:
            class_name = "good"
        elif value <= -5:
            class_name = "bad"
        else:
            class_name = "ok"
    elif col in {"to", "gog_to"}:
        if col == "to":
            if value <= 1:
                class_name = "good"
            elif value >= 3:
                class_name = "bad"
            else:
                class_name = "ok"
        else:
            if value <= -1:
                class_name = "good"
            elif value >= 1:
                class_name = "bad"
            else:
                class_name = "ok"

    if col in {"min", "usage_proxy", "pts", "ast", "reb", "stl", "blk", "to", "pf", "r5_pts", "r5_usage_proxy"}:
        text = f"{value:.1f}"
    elif col in {"ts", "efg", "r5_ts", "r5_efg"}:
        text = f"{value:.3f}"
    elif col.startswith("gog_"):
        text = f"{value:.2f}"
    elif isinstance(value, (int, float)):
        text = f"{value:.2f}"
    else:
        text = escape(str(value))

    if class_name:
        return f"<span class='cell {class_name}'>{text}</span>"
    return text


def _format_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    if df.empty:
        header_cells = "".join(f"<th>{c}</th>" for c in cols)
        return (
            "<table class='report-table'><thead><tr>"
            f"{header_cells}"
            "</tr></thead><tbody><tr><td colspan='"
            f"{len(cols)}'>No rows match filter.</td></tr></tbody></table>"
        )

    render_df = df.loc[:, cols].copy()
    if "date" in render_df.columns:
        render_df["date"] = pd.to_datetime(render_df["date"], errors="coerce", utc=True).dt.strftime(
            "%Y-%m-%d"
        )

    for col in render_df.columns:
        if pd.api.types.is_numeric_dtype(render_df[col]):
            render_df[col] = render_df[col].map(lambda v, c=col: _render_cell(v, c))

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

    return f"<h4>{team_code}</h4>" f"<p class='note'>{note}</p>" f"{_format_table(filtered, columns)}"


def build_report(season_code: str) -> tuple[Path, Path, str]:
    games = _read_curated("games")

    games_all_fallback = False
    try:
        games_all = _read_curated("games_all")
    except FileNotFoundError:
        games_all = games
        games_all_fallback = True

    player_game = _read_curated("player_game")
    player_gog = _read_curated("player_gog")

    games_all_season = games_all[games_all["season_code"] == season_code].copy()
    games_season = games[games["season_code"] == season_code].copy()
    player_game_season = player_game[player_game["season_code"] == season_code]
    player_gog_season = player_gog[player_gog["season_code"] == season_code]

    latest = _prepare_latest_by_player(player_gog, season_code)
    min_game_date, max_game_date = _extract_game_date_range(games_all_season)
    pan_momentum = _compute_momentum_watch(latest, "PAN")
    oly_momentum = _compute_momentum_watch(latest, "OLY")
    pan_top_usage = _compute_top_usage(latest, "PAN")
    oly_top_usage = _compute_top_usage(latest, "OLY")
    pan_summary = _latest_team_summary(latest, "PAN")
    oly_summary = _latest_team_summary(latest, "OLY")

    generated_at = datetime.now(timezone.utc)
    timestamp = generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    warning_messages = []
    if len(player_gog_season) == 0:
        warning_messages.append(
            "Warning: player_gog has 0 rows for this season. Tables are intentionally empty."
        )
    if games_all_fallback:
        warning_messages.append(
            "games_all dataset missing; 'Total games parsed' is currently using PAN/OLY filtered games."
        )

    warning_banner = "".join(f"<div class='warning'>{msg}</div>" for msg in warning_messages)

    points_cols = [
        "date",
        "player_name",
        "min",
        "usage_proxy",
        "pts",
        "ast",
        "reb",
        "stl",
        "blk",
        "to",
        "pf",
        "ts",
        "efg",
        "gog_pts",
        "gog_ts",
        "gog_to",
        "r5_pts",
        "r5_ts",
        "r5_efg",
    ]
    eff_cols = [
        "date",
        "player_name",
        "min",
        "usage_proxy",
        "pts",
        "ts",
        "efg",
        "gog_ts",
        "gog_pts",
        "to",
        "gog_to",
        "r5_ts",
    ]
    watch_cols = ["date", "player_name", "min", "usage_proxy", "pts", "ts", "ast", "to", "r5_ts"]
    usage_cols = ["date", "player_name", "min", "usage_proxy", "pts", "ts", "ast", "to", "r5_ts"]
    momentum_cols = [
        "date",
        "player_name",
        "min",
        "usage_proxy",
        "pts",
        "to",
        "ts",
        "gog_ts",
        "gog_pts",
        "gog_to",
        "gog_usage_proxy",
        "gog_min",
        "FormScore",
        "r5_ts",
        "r5_usage_proxy",
    ]

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
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #fafafa; }}
    .value {{ font-size: 1.3rem; font-weight: 700; }}
    .note {{ display: inline-block; background: #eef6ff; border: 1px solid #c9def8; color: #184e87; border-radius: 999px; padding: 4px 10px; font-size: 0.85rem; margin: 8px 0; }}
    .warning {{ background: #fff4e5; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin-top: 16px; font-weight: 600; color: #92400e; }}
    table {{ width: 100%; border-collapse: collapse; margin: 8px 0 20px; font-size: 0.90rem; table-layout: auto; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; white-space: nowrap; max-width: 170px; overflow: hidden; text-overflow: ellipsis; }}
    thead th {{ position: sticky; top: 0; background: #f8fafc; z-index: 1; }}
    th.sortable {{ cursor: pointer; }}
    .table-wrap {{ overflow-x: auto; }}
    .cell {{ display: inline-block; padding: 2px 6px; border-radius: 4px; }}
    .good {{ background: #e8f7ee; color: #065f46; font-weight: 600; }}
    .bad  {{ background: #fde8e8; color: #991b1b; font-weight: 600; }}
    .ok   {{ background: #f3f4f6; color: #111827; }}
    #definitions ul {{ margin-top: 6px; line-height: 1.6; }}
    #definitions code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>EuroLeague PAN/OLY Player Report — Season {season_code} — Generated {timestamp}</h1>
  <p class=\"muted\">Jump to <a href=\"#definitions\">definitions</a>.</p>

  <section class=\"summary\">
    <h2>Executive Summary</h2>
    <div class=\"summary-grid\">
      <div class=\"card\">
        <h3>PAN</h3>
        <ul>
          <li><strong>Top scoring mover:</strong> {pan_summary['top_scoring']}</li>
          <li><strong>Top efficiency mover:</strong> {pan_summary['top_eff']}</li>
          <li><strong>Highest usage:</strong> {pan_summary['highest_usage']}</li>
          <li><strong>Watchlist (high usage, low TS):</strong> {pan_summary['watchlist']}</li>
        </ul>
      </div>
      <div class=\"card\">
        <h3>OLY</h3>
        <ul>
          <li><strong>Top scoring mover:</strong> {oly_summary['top_scoring']}</li>
          <li><strong>Top efficiency mover:</strong> {oly_summary['top_eff']}</li>
          <li><strong>Highest usage:</strong> {oly_summary['highest_usage']}</li>
          <li><strong>Watchlist (high usage, low TS):</strong> {oly_summary['watchlist']}</li>
        </ul>
      </div>
    </div>
  </section>

  {warning_banner}

  <section class=\"summary\">
    <h2>Data freshness</h2>
    <div class=\"summary-grid\">
      <div class=\"card\"><div>Total games parsed</div><div class=\"value\">{len(games_all_season)}</div></div>
      <div class=\"card\"><div>PAN/OLY games found</div><div class=\"value\">{len(games_season)}</div></div>
      <div class=\"card\"><div>player_game rows</div><div class=\"value\">{len(player_game_season)}</div></div>
      <div class=\"card\"><div>player_gog rows</div><div class=\"value\">{len(player_gog_season)}</div></div>
      <div class=\"card\"><div>Games date range</div><div class=\"value\">{min_game_date} → {max_game_date}</div></div>
      <div class=\"card\"><div>Latest game date</div><div class=\"value\">{max_game_date}</div></div>
      <div class=\"card\"><div>Report generated</div><div class=\"value\">{timestamp}</div></div>
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

  <section class=\"section\">
    <h2>Top Usage (latest game)</h2>
    <p class='note'>usage_proxy approximates possessions used: <code>FGA + 0.44 × FTA + TO</code>.</p>
    <div class=\"table-wrap\">
      <h4>PAN</h4>
      {_format_table(pan_top_usage, usage_cols)}
      <h4>OLY</h4>
      {_format_table(oly_top_usage, usage_cols)}
    </div>
  </section>

  <section class=\"section\">
    <h2>Momentum Watch (latest game per player)</h2>
    <p class='note'>Filter: min ≥ 10 and usage_proxy ≥ 6 in latest game. FormScore is a heuristic.</p>
    <div class=\"table-wrap\">
      <h4>PAN</h4>
      {_format_table(pan_momentum, momentum_cols)}
      <h4>OLY</h4>
      {_format_table(oly_momentum, momentum_cols)}
    </div>
  </section>

  <section id=\"definitions\" class=\"section\">
    <h2>Footnotes / Definitions</h2>
    <ul>
      <li><strong>TS (True Shooting):</strong> <code>PTS / (2 × (FGA + 0.44 × FTA))</code></li>
      <li><strong>eFG:</strong> <code>(FGM + 0.5 × 3PM) / FGA</code></li>
      <li><strong>usage_proxy:</strong> <code>FGA + 0.44 × FTA + TO</code></li>
      <li><strong>GoG (game-over-game):</strong> <code>current_stat - previous_game_stat</code> for the same <code>player_id</code></li>
      <li><strong>r5_*:</strong> <code>rolling_5_game_mean(stat)</code></li>
      <li><strong>min conversion:</strong> <code>MM:SS → minutes_float</code></li>
    </ul>
  </section>

  <script>
    (function () {{
      function parseCell(text) {{
        const cleaned = text.trim().replace(/,/g, '');
        const num = Number(cleaned);
        return Number.isNaN(num) ? cleaned.toLowerCase() : num;
      }}

      document.querySelectorAll('table.report-table').forEach((table) => {{
        const headers = table.querySelectorAll('thead th');
        const tbody = table.querySelector('tbody');
        if (!tbody) return;

        headers.forEach((th, idx) => {{
          th.classList.add('sortable');
          th.dataset.order = 'desc';
          th.addEventListener('click', () => {{
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const asc = th.dataset.order === 'asc';
            rows.sort((a, b) => {{
              const va = parseCell(a.children[idx].innerText);
              const vb = parseCell(b.children[idx].innerText);
              if (va < vb) return asc ? -1 : 1;
              if (va > vb) return asc ? 1 : -1;
              return 0;
            }});
            rows.forEach((r) => tbody.appendChild(r));
            headers.forEach((h) => (h.dataset.order = 'desc'));
            th.dataset.order = asc ? 'desc' : 'asc';
          }});
        }});
      }});
    }})();
  </script>
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

    latest_size = latest_path.stat().st_size
    print(f"Latest game date: {max_game_date}")
    print(f"Wrote reports/latest.html size: {latest_size} bytes")
    return latest_path, dated_path, max_game_date


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PAN/OLY HTML report")
    parser.add_argument("--season", required=True, help="Season code, e.g. E2025")
    args = parser.parse_args()

    latest, dated, _latest_game_date = build_report(args.season)
    print(f"Wrote {latest}")
    print(f"Wrote {dated}")


if __name__ == "__main__":
    main()
