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

HEADER_TOOLTIPS = {
    "ts": "Scoring efficiency; ~0.55 is good. >1.0 often means tiny sample.",
    "usage_proxy": "Possessions used proxy; shows role/load.",
    "gog_ts": "Change vs previous game; big jumps can be matchup/role.",
    "r5_ts": "5-game average; more stable form signal.",
    "to": "Turnovers; end possessions. Rising TO under high usage is risky.",
    "min": "Coach trust / role. Large gog_min indicates rotation change.",
}


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


def _robust_norm(series: pd.Series) -> pd.Series:
    denom = max(series.abs().max(), 1e-6)
    return series / denom


def _add_confidence_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["Confidence"] = pd.Series(dtype="object")
        out["Flags"] = pd.Series(dtype="object")
        return out

    out = df.copy()
    out["Confidence"] = "MED"
    out.loc[out["min"] >= 20, "Confidence"] = "HIGH"
    out.loc[out["min"] < 12, "Confidence"] = "LOW"

    def _row_flags(row: pd.Series) -> str:
        flags: list[str] = []
        if row.get("usage_proxy", 0) < 8:
            flags.append("LOW_USAGE")
        if row.get("ts", 0) > 1.0:
            flags.append("TS_OUTLIER")
        if row.get("to", 0) >= 3:
            flags.append("HIGH_TO")
        if row.get("gog_min", 0) >= 5:
            flags.append("ROLE_SPIKE")
        if row.get("gog_usage_proxy", 0) >= 5:
            flags.append("USG_SPIKE")
        if row.get("gog_ts", 0) <= -0.15:
            flags.append("EFF_DROP")
        if row.get("ts", 0) >= 0.65 and row.get("min", 0) >= 15:
            flags.append("HOT")
        return "|".join(flags) if flags else "-"

    out["Flags"] = out.apply(_row_flags, axis=1)
    return out


def _inject_header_tooltips(table_html: str) -> str:
    for col, tip in HEADER_TOOLTIPS.items():
        header = (
            f"<th>{col} "
            f"<span class='info-icon' title='{escape(tip, quote=True)}'>ℹ︎</span></th>"
        )
        table_html = table_html.replace(f"<th>{col}</th>", header)
    return table_html


def _render_confidence_badge(value: str) -> str:
    css = {"HIGH": "conf-high", "MED": "conf-med", "LOW": "conf-low"}.get(value, "conf-med")
    return f"<span class='badge {css}'>{escape(value)}</span>"


def _render_why_ranked(df: pd.DataFrame, list_type: str, score_col: str) -> str:
    if df.empty:
        return "<p class='muted'>Why these are ranked #1–#3: no qualifying players.</p>"

    top3 = df.head(3)
    items: list[str] = []
    for _, row in top3.iterrows():
        player = escape(str(row["player_name"]))
        confidence = _render_confidence_badge(str(row.get("Confidence", "MED")))
        flags = escape(str(row.get("Flags", "-")))
        if list_type == "impact":
            sentence = (
                f"{player}: {row['pts']:.1f} pts on TS {row['ts']:.3f}; usage {row['usage_proxy']:.1f}; "
                f"TO {row['to']:.1f}."
            )
        elif list_type == "trend":
            sentence = (
                f"{player}: ΔTS {row['gog_ts']:+.2f}, ΔPTS {row['gog_pts']:+.2f}, "
                f"ΔUSG {row['gog_usage_proxy']:+.2f}, ΔMIN {row['gog_min']:+.2f}."
            )
        else:
            sentence = (
                f"{player}: usage {row['usage_proxy']:.1f} with TS {row['ts']:.3f} and TO {row['to']:.1f} "
                f"(risk score {row[score_col]:.3f})."
            )
        items.append(f"<li>{sentence} {confidence} <span class='muted'>Flags: {flags}</span></li>")

    return "<div class='why-ranked'><strong>Why these are ranked #1–#3:</strong><ul>" + "".join(items) + "</ul></div>"


def _compute_stack_rankings(latest_df: pd.DataFrame, team_code: str) -> dict[str, pd.DataFrame]:
    base_df = latest_df[(latest_df["team_code"] == team_code) & (latest_df["min"] >= 10)].copy()
    if base_df.empty:
        return {"impact": base_df, "trend": base_df, "risk": base_df}

    result: dict[str, pd.DataFrame] = {}

    impact_df = base_df[(base_df["usage_proxy"] >= 6) & (base_df["min"] >= 12)].copy()
    if not impact_df.empty:
        impact_ts_for_score = impact_df["ts"].clip(0, 0.80)
        impact_df["ImpactNowScore"] = (
            0.45 * _robust_norm(impact_df["pts"])
            + 0.35 * _robust_norm(impact_ts_for_score - 0.55)
            + 0.20 * _robust_norm(impact_df["usage_proxy"])
        )
        impact_df = impact_df.sort_values(["ImpactNowScore", "pts"], ascending=[False, False]).head(8)
    result["impact"] = impact_df

    trend_df = base_df[(base_df["usage_proxy"] >= 6) | (base_df["min"] >= 20)].copy()
    if not trend_df.empty:
        trend_df["TrendScore"] = (
            0.40 * trend_df["gog_ts"].clip(-0.30, 0.30)
            + 0.30 * _robust_norm(trend_df["gog_pts"])
            + 0.20 * _robust_norm(trend_df["gog_usage_proxy"])
            + 0.10 * _robust_norm(trend_df["r5_ts"] - 0.55)
        )
        trend_df = trend_df.sort_values(["TrendScore", "pts"], ascending=[False, False]).head(8)
    result["trend"] = trend_df

    risk_df = base_df.copy()
    ts_clipped = risk_df["ts"].clip(0, 1)
    usage_risk = (risk_df["usage_proxy"] - 12).clip(lower=0)
    ts_risk = (0.52 - ts_clipped).clip(lower=0)
    to_risk = risk_df["to"].clip(lower=0)
    risk_df["RiskScore"] = (
        0.45 * _robust_norm(usage_risk)
        + 0.35 * _robust_norm(ts_risk)
        + 0.20 * _robust_norm(to_risk)
    )
    result["risk"] = risk_df.sort_values(["RiskScore", "pts"], ascending=[False, False]).head(8)
    return result


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


def _format_table(
    df: pd.DataFrame,
    columns: Iterable[str],
    score_badges: dict[str, str] | None = None,
) -> str:
    cols = list(columns)
    if df.empty:
        header_cells = "".join(f"<th>{c}</th>" for c in cols)
        table_html = (
            "<table class='report-table'><thead><tr>"
            f"{header_cells}"
            "</tr></thead><tbody><tr><td colspan='"
            f"{len(cols)}'>No rows match filter.</td></tr></tbody></table>"
        )
        return _inject_header_tooltips(table_html)

    render_df = df.loc[:, cols].copy()
    if "date" in render_df.columns:
        render_df["date"] = pd.to_datetime(render_df["date"], errors="coerce", utc=True).dt.strftime(
            "%Y-%m-%d"
        )

    score_badges = score_badges or {}
    for col in render_df.columns:
        if col in score_badges:
            badge_class = score_badges[col]
            render_df[col] = render_df[col].map(
                lambda v, b=badge_class: "" if pd.isna(v) else f"<span class='badge {b}'>{v:.3f}</span>"
            )
            continue
        if col == "Confidence":
            render_df[col] = render_df[col].map(lambda v: "" if pd.isna(v) else _render_confidence_badge(str(v)))
            continue
        if pd.api.types.is_numeric_dtype(render_df[col]):
            render_df[col] = render_df[col].map(lambda v, c=col: _render_cell(v, c))

    table_html = render_df.to_html(index=False, escape=False, border=0, classes="report-table")
    return _inject_header_tooltips(table_html)


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
    pan_stack = _compute_stack_rankings(latest, "PAN")
    oly_stack = _compute_stack_rankings(latest, "OLY")
    pan_summary = _latest_team_summary(latest, "PAN")
    oly_summary = _latest_team_summary(latest, "OLY")

    pan_stack = {k: _add_confidence_and_flags(v) for k, v in pan_stack.items()}
    oly_stack = {k: _add_confidence_and_flags(v) for k, v in oly_stack.items()}
    pan_top_usage = _add_confidence_and_flags(pan_top_usage)
    oly_top_usage = _add_confidence_and_flags(oly_top_usage)
    pan_momentum = _add_confidence_and_flags(pan_momentum)
    oly_momentum = _add_confidence_and_flags(oly_momentum)

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
    usage_cols = ["date", "player_name", "min", "usage_proxy", "pts", "ts", "ast", "to", "r5_ts", "Confidence", "Flags"]
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
        "Confidence",
        "Flags",
    ]
    stack_cols = [
        "player_name",
        "min",
        "usage_proxy",
        "pts",
        "ts",
        "to",
        "gog_ts",
        "gog_pts",
        "gog_usage_proxy",
        "gog_min",
        "r5_ts",
        "Confidence",
        "Flags",
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
    .badge {{ padding: 2px 8px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }}
    .badge.good {{ background: #e8f7ee; color: #065f46; }}
    .badge.bad {{ background: #fde8e8; color: #991b1b; }}
    .badge.conf-high {{ background: #e8f7ee; color: #065f46; }}
    .badge.conf-med {{ background: #fff4db; color: #92400e; }}
    .badge.conf-low {{ background: #fde8e8; color: #991b1b; }}
    .good {{ background: #e8f7ee; color: #065f46; font-weight: 600; }}
    .bad  {{ background: #fde8e8; color: #991b1b; font-weight: 600; }}
    .ok   {{ background: #f3f4f6; color: #111827; }}
    .tip {{ border-left: 4px solid #0f4c81; background: #eef6ff; padding: 10px 12px; border-radius: 6px; margin-top: 12px; }}
    .legend {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }}
    .chip {{ padding: 4px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 700; }}
    .chip.green {{ background: #e8f7ee; color: #065f46; }}
    .chip.gray {{ background: #f3f4f6; color: #111827; }}
    .chip.red {{ background: #fde8e8; color: #991b1b; }}
    .info-icon {{ color: #6b7280; font-size: 0.8rem; margin-left: 4px; cursor: help; }}
    .why-ranked ul {{ margin-top: 6px; margin-bottom: 12px; }}
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

  <section class="section">
    <h2>What we measure &amp; why it matters</h2>
    <p>We track more than box-score totals because production quality and role context usually explain future games better than single-game points. Efficiency metrics such as TS and eFG tell us whether scoring came from sustainable shot value (threes, free throws, and clean attempts) rather than volume alone.</p>
    <p>Usage_proxy shows offensive responsibility. A player with high TS on very low usage may simply be finishing easy possessions, while a high-usage player with average TS can still be driving team offense. Reading efficiency together with usage helps separate true creators from low-volume outliers.</p>
    <p>GoG deltas highlight short-term shocks from one game to the next, which is useful for spotting role changes, injury impacts, or matchup swings quickly. Rolling 5-game metrics (r5_*) smooth one-game noise to reveal underlying form and stabilize interpretation.</p>
    <p>Turnovers and minutes complete the picture: turnovers end possessions, so rising TO under heavy usage is a warning sign; minutes reflect coach trust and rotation shifts, and sudden minute spikes often precede box-score jumps.</p>
    <ul>
      <li><strong>Efficiency (TS, eFG):</strong> shot quality converted into points; better sustainability signal than raw points.</li>
      <li><strong>usage_proxy:</strong> offensive load proxy that clarifies role and responsibility.</li>
      <li><strong>GoG deltas:</strong> immediate change detector for form, role, and availability shifts.</li>
      <li><strong>r5_* rolling:</strong> trend lens that reduces game-level randomness.</li>
      <li><strong>Turnovers (TO, gog_to):</strong> possession killers, especially concerning when paired with high usage.</li>
      <li><strong>Minutes (min, gog_min):</strong> rotation confidence indicator; spikes can foreshadow bigger stat lines.</li>
    </ul>
    <div class="tip">
      <strong>How to read this report:</strong> Start with <em>Stack-ranked Insights</em>, then verify context in <em>Top Usage</em>, then confirm direction in <em>Momentum Watch</em>. Treat TS above 1.0 as a likely small-sample/outlier artifact from limited attempts or free-throw weighting.
    </div>
    <div class="legend">
      <span class="chip green">Green: strong / positive</span>
      <span class="chip gray">Gray: neutral</span>
      <span class="chip red">Red: concerning / negative</span>
    </div>
  </section>

  <section class="section">
    <h2>Stack-ranked Insights (per team)</h2>
    <p class='note'>Scope: latest game per player. Top 8 per list for PAN and OLY.</p>
    <div class="table-wrap">
      <h3>PAN</h3>
      <h4>Impact Now</h4>
      <p class='note'>Filter guardrail: usage_proxy ≥ 6 and min ≥ 12; TS contribution capped at 0.80 for scoring.</p>
      {_format_table(pan_stack['impact'], stack_cols + ['ImpactNowScore'], score_badges={'ImpactNowScore': 'good'})}
      {_render_why_ranked(pan_stack['impact'], 'impact', 'ImpactNowScore')}
      <p class='muted'>Prioritizes current production and efficiency: points, TS (centered at 0.55), and offensive load.</p>
      <h4>Trend</h4>
      <p class='note'>Filter guardrail: min ≥ 10 and (usage_proxy ≥ 6 or min ≥ 20); GoG TS contribution capped to ±0.30 for scoring.</p>
      {_format_table(pan_stack['trend'], stack_cols + ['TrendScore'], score_badges={'TrendScore': 'good'})}
      {_render_why_ranked(pan_stack['trend'], 'trend', 'TrendScore')}
      <p class='muted'>Prioritizes direction of change using GoG efficiency/volume plus rolling TS stability.</p>
      <h4>Risk</h4>
      <p class='note'>Filter guardrail: min ≥ 10. Flags emphasize HIGH_TO and LOW_USAGE visibility.</p>
      {_format_table(pan_stack['risk'], stack_cols + ['RiskScore'], score_badges={'RiskScore': 'bad'})}
      {_render_why_ranked(pan_stack['risk'], 'risk', 'RiskScore')}
      <p class='muted'>Higher score is worse: high usage burden, low TS, and turnovers raise risk.</p>

      <h3>OLY</h3>
      <h4>Impact Now</h4>
      <p class='note'>Filter guardrail: usage_proxy ≥ 6 and min ≥ 12; TS contribution capped at 0.80 for scoring.</p>
      {_format_table(oly_stack['impact'], stack_cols + ['ImpactNowScore'], score_badges={'ImpactNowScore': 'good'})}
      {_render_why_ranked(oly_stack['impact'], 'impact', 'ImpactNowScore')}
      <p class='muted'>Prioritizes current production and efficiency: points, TS (centered at 0.55), and offensive load.</p>
      <h4>Trend</h4>
      <p class='note'>Filter guardrail: min ≥ 10 and (usage_proxy ≥ 6 or min ≥ 20); GoG TS contribution capped to ±0.30 for scoring.</p>
      {_format_table(oly_stack['trend'], stack_cols + ['TrendScore'], score_badges={'TrendScore': 'good'})}
      {_render_why_ranked(oly_stack['trend'], 'trend', 'TrendScore')}
      <p class='muted'>Prioritizes direction of change using GoG efficiency/volume plus rolling TS stability.</p>
      <h4>Risk</h4>
      <p class='note'>Filter guardrail: min ≥ 10. Flags emphasize HIGH_TO and LOW_USAGE visibility.</p>
      {_format_table(oly_stack['risk'], stack_cols + ['RiskScore'], score_badges={'RiskScore': 'bad'})}
      {_render_why_ranked(oly_stack['risk'], 'risk', 'RiskScore')}
      <p class='muted'>Higher score is worse: high usage burden, low TS, and turnovers raise risk.</p>
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
      <li><strong>TS (True Shooting):</strong> <code>PTS / (2 × (FGA + 0.44 × FTA))</code> — scoring efficiency adjusted for 3s and free throws; best single-number scorer efficiency proxy.</li>
      <li><strong>eFG:</strong> <code>(FGM + 0.5 × 3PM) / FGA</code> — shooting value per attempt, rewarding 3-point shot value above 2s.</li>
      <li><strong>usage_proxy:</strong> <code>FGA + 0.44 × FTA + TO</code> — approximate possessions a player consumed; proxy for offensive load.</li>
      <li><strong>GoG (game-over-game):</strong> <code>current_stat - previous_game_stat</code> for the same <code>player_id</code> — detects sudden changes game-to-game (role, matchup, injury, form).</li>
      <li><strong>r5_*:</strong> <code>rolling_5_game_mean(stat)</code> — reduces randomness and is usually better for reading true form.</li>
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
