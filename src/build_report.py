"""Build self-contained HTML player report for PAN/OLY."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import BASE_DIR, CURATED_DIR, TEAMS
from src.fetch_schedule import next_pan_game

REPORTS_DIR = BASE_DIR / "reports"

HEADER_TOOLTIPS = {
    "ts": "~0.55 solid, >0.60 strong; >1.0 often small sample.",
    "usage_proxy": "Role/load proxy; interpret TS with this.",
    "gog_ts": "Change vs previous game; big jumps can be matchup/role.",
    "r5_ts": "5-game average; more stable form signal.",
    "to": "Turnovers; end possessions. Rising TO under high usage is risky.",
    "min": "Coach trust / role. Large gog_min indicates rotation change.",
    "RiskScore": "Higher = worse; driven by usage burden + low TS + TO.",
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
    table_class: str = "",
) -> str:
    cols = list(columns)
    class_suffix = f" {table_class}" if table_class else ""

    def _header_cell(col: str) -> str:
        tip = HEADER_TOOLTIPS.get(col)
        if tip:
            return (
                f"<th>{escape(col)} "
                f"<span class='info-icon' title='{escape(tip, quote=True)}'>ℹ︎</span></th>"
            )
        return f"<th>{escape(col)}</th>"

    if df.empty:
        header_cells = "".join(_header_cell(c) for c in cols)
        table_html = (
            f"<table class='report-table{class_suffix}'><thead><tr>"
            f"{header_cells}"
            "</tr></thead><tbody><tr><td colspan='"
            f"{len(cols)}'>No rows match filter.</td></tr></tbody></table>"
        )
        return table_html

    render_df = df.loc[:, cols].copy()
    if "date" in render_df.columns:
        render_df["date"] = pd.to_datetime(render_df["date"], errors="coerce", utc=True).dt.strftime(
            "%Y-%m-%d"
        )

    score_badges = score_badges or {}
    header_cells = "".join(_header_cell(c) for c in cols)
    rows_html: list[str] = []
    for _, row in render_df.iterrows():
        player_name = escape(str(row.get("player_name", ""))).lower()
        row_cells: list[str] = []
        for col in cols:
            value = row[col]
            if col in score_badges:
                badge_class = score_badges[col]
                cell_value = "" if pd.isna(value) else f"<span class='badge {badge_class}'>{value:.3f}</span>"
            elif col == "Confidence":
                cell_value = "" if pd.isna(value) else _render_confidence_badge(str(value))
            elif pd.api.types.is_numeric_dtype(render_df[col]):
                cell_value = _render_cell(value, col)
            elif col == "Flags":
                cell_value = "" if pd.isna(value) else str(value)
            else:
                cell_value = "" if pd.isna(value) else escape(str(value))

            td_class = "num" if col != "player_name" else "text player-col"
            row_cells.append(f"<td data-col='{escape(col)}' class='{td_class}'>{cell_value}</td>")

        rows_html.append(f"<tr data-player-name='{player_name}'>{''.join(row_cells)}</tr>")

    return (
        f"<table class='report-table{class_suffix}'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )


def _build_key_takeaways(team_df: pd.DataFrame, risk_df: pd.DataFrame) -> list[str]:
    if team_df.empty:
        return ["No latest-game rows found for this team."]

    usage_rank = team_df.sort_values("usage_proxy", ascending=False)
    top_usage_row = usage_rank.iloc[0]
    next_usage = usage_rank.iloc[1]["usage_proxy"] if len(usage_rank) > 1 else 0.0
    usage_line = (
        f"Usage concentrated: {escape(str(top_usage_row['player_name']))} {top_usage_row['usage_proxy']:.1f} "
        f"vs next {next_usage:.1f}."
    )

    eff_scope = team_df[(team_df["min"] >= 15) & (team_df["usage_proxy"] >= 6)]
    if eff_scope.empty:
        eff_line = "Efficiency driver: no player met min ≥ 15 and usage ≥ 6 thresholds."
    else:
        best_eff = eff_scope.sort_values("ts", ascending=False).iloc[0]
        eff_line = f"Efficiency led by {escape(str(best_eff['player_name']))} TS {best_eff['ts']:.3f}."

    if risk_df.empty:
        risk_line = "Primary risk: no qualifying risk rows."
    else:
        risk_top = risk_df.sort_values("RiskScore", ascending=False).iloc[0]
        high_to_rows = risk_df[risk_df["Flags"].fillna("").str.contains("HIGH_TO")]
        risk_row = high_to_rows.iloc[0] if not high_to_rows.empty else risk_top
        risk_line = (
            f"Risk: {escape(str(risk_row['player_name']))} TO {risk_row['to']:.1f} "
            f"at usage {risk_row['usage_proxy']:.1f}."
        )

    return [usage_line, eff_line, risk_line]


def _render_table_panel(
    table_id: str,
    title: str,
    slim_df: pd.DataFrame,
    slim_cols: list[str],
    detail_cols: list[str],
    score_badges: dict[str, str] | None = None,
    why_ranked_html: str = "",
) -> str:
    detail_df = slim_df.copy()
    slim_view = slim_df.copy()
    if "Flags" in slim_view.columns and "Confidence" in slim_view.columns and "Confidence" not in slim_cols:
        slim_view["Flags"] = slim_view.apply(
            lambda row: f"{_render_confidence_badge(str(row['Confidence']))} <span class='muted'>{escape(str(row['Flags']))}</span>",
            axis=1,
        )
    details = f"""
    <div class='table-panel'>
      <div class='panel-header'>
        <h4>{escape(title)}</h4>
        <button class='toggle-details' data-target='{table_id}' type='button'>Show details</button>
      </div>
      <div class='table-wrap'>
        {_format_table(slim_view, slim_cols, score_badges=score_badges, table_class='slim-table')}
      </div>
      <div id='{table_id}' class='details-wrap hidden'>
        <div class='table-wrap'>
          {_format_table(detail_df, detail_cols, score_badges=score_badges)}
        </div>
      </div>
      {why_ranked_html}
    </div>
    """
    return details


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


def _prediction_block(season_code: str) -> str:
    model_path = BASE_DIR / "data" / "models" / f"win_model_{season_code}.json"
    team_game_path = CURATED_DIR / "team_game.csv"
    if not model_path.exists() or not team_game_path.exists():
        return "<section><h3>Next game prediction (Panathinaikos)</h3><p>No upcoming game found.</p></section>"

    model = json.loads(model_path.read_text(encoding="utf-8"))
    next_game = next_pan_game(season_code)
    if not next_game:
        return "<section><h3>Next game prediction (Panathinaikos)</h3><p>No upcoming game found.</p></section>"

    team = pd.read_csv(team_game_path)
    if "season_code" in team.columns:
        team = team[team["season_code"] == season_code].copy()
    team["date"] = pd.to_datetime(team["date"], errors="coerce", utc=True)
    game_date = pd.to_datetime(next_game.get("date"), errors="coerce", utc=True)

    pre_cutoff = game_date - pd.Timedelta(days=1)
    hist = team[team["date"] <= pre_cutoff].copy()

    feat_cols = {
        "r5_team_ts_pre": "d_r5_ts_pre",
        "r5_team_efg_pre": "d_r5_efg_pre",
        "r5_team_to_pre": "d_r5_to_pre",
        "r5_margin_pre": "d_r5_margin_pre",
        "r5_usage_conc_top3_pre": "d_r5_usage_conc_pre",
    }

    hist = hist.sort_values(["team_code", "date", "gamecode_num"], kind="stable")
    for col in feat_cols:
        hist[col] = hist.groupby("team_code", sort=False)[col.replace("_pre", "")].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )

    pan_home = next_game.get("home_team_code") == "PAN"
    opp = next_game.get("away_team_code") if pan_home else next_game.get("home_team_code")
    home_team = "PAN" if pan_home else opp
    away_team = opp if pan_home else "PAN"

    home_hist = hist[hist["team_code"] == home_team]
    away_hist = hist[hist["team_code"] == away_team]
    home_last = home_hist.iloc[-1] if not home_hist.empty else pd.Series(dtype=float)
    away_last = away_hist.iloc[-1] if not away_hist.empty else pd.Series(dtype=float)

    feature_names = model["feature_names"]
    values: dict[str, float] = {}
    for raw_col, out_col in feat_cols.items():
        h = float(home_last.get(raw_col, 0.0)) if not home_last.empty else 0.0
        a = float(away_last.get(raw_col, 0.0)) if not away_last.empty else 0.0
        values[out_col] = h - a
    values["home"] = 1.0

    x = [float(values.get(name, 0.0)) for name in feature_names]
    means = model["scaler_mean"]
    scales = [s if s else 1.0 for s in model["scaler_scale"]]
    z = [(xv - m) / s for xv, m, s in zip(x, means, scales)]
    logit = float(model["intercept"] + sum(c * zv for c, zv in zip(model["coef"], z)))
    p_home = 1 / (1 + math.exp(-logit))
    p_pan = p_home if pan_home else (1 - p_home)

    contrib = []
    for name, coef, zv, xv in zip(feature_names, model["coef"], z, x):
        contrib.append((name, coef * zv, xv))
    top = sorted(contrib, key=lambda t: abs(t[1]), reverse=True)[:3]
    drivers = "".join(
        f"<li>{'+' if val >= 0 else '-'}{escape(name)} (value {raw:+.3f})</li>" for name, val, raw in top
    )

    pan_hist_games = len(hist[hist["team_code"] == "PAN"])
    opp_hist_games = len(hist[hist["team_code"] == opp])
    min_hist = min(pan_hist_games, opp_hist_games)
    confidence = "HIGH" if min_hist >= 10 else ("MED" if min_hist >= 5 else "LOW")
    ha = "Home" if pan_home else "Away"
    game_date_text = game_date.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(game_date) else "Unknown"

    return (
        "<section><h3>Next game prediction (Panathinaikos)</h3>"
        f"<p><strong>Game:</strong> {game_date_text} vs {escape(str(opp))} ({ha})</p>"
        f"<p><strong>Predicted PAN win probability:</strong> {p_pan * 100:.1f}%</p>"
        f"<p><strong>Confidence:</strong> {confidence} (PAN prior games: {pan_hist_games}, Opp prior games: {opp_hist_games})</p>"
        f"<p><strong>Top drivers</strong></p><ul>{drivers}</ul>"
        "</section>"
    )


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
    prediction_section = _prediction_block(season_code)

    points_cols = [
        "date",
        "player_name",
        "min",
        "usage_proxy",
        "pts",
        "ts",
        "to",
        "gog_pts",
        "gog_ts",
        "gog_to",
        "r5_pts",
        "r5_ts",
        "Confidence",
        "Flags",
    ]
    stack_slim_cols = ["player_name", "min", "usage_proxy", "pts", "ts", "to", "ImpactNowScore", "Flags"]
    trend_slim_cols = ["player_name", "min", "usage_proxy", "pts", "ts", "to", "TrendScore", "Flags"]
    risk_slim_cols = ["player_name", "min", "usage_proxy", "pts", "ts", "to", "RiskScore", "Flags"]
    usage_slim_cols = ["player_name", "min", "usage_proxy", "pts", "ts", "to", "r5_ts", "Flags"]
    movers_slim_cols = ["player_name", "min", "pts", "gog_pts", "ts", "gog_ts", "to", "Flags"]
    momentum_slim_cols = [
        "player_name",
        "min",
        "usage_proxy",
        "FormScore",
        "ts",
        "gog_ts",
        "gog_usage_proxy",
        "gog_min",
        "Flags",
    ]

    stack_detail_cols = [
        "date",
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
    usage_detail_cols = [
        "date",
        "player_name",
        "min",
        "usage_proxy",
        "pts",
        "ts",
        "ast",
        "to",
        "gog_usage_proxy",
        "gog_pts",
        "gog_ts",
        "r5_ts",
        "r5_usage_proxy",
        "Confidence",
        "Flags",
    ]
    movers_detail_cols = points_cols
    momentum_detail_cols = [
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

    pan_team_df = latest[latest["team_code"] == "PAN"].copy()
    oly_team_df = latest[latest["team_code"] == "OLY"].copy()
    pan_takeaways = _build_key_takeaways(pan_team_df, pan_stack["risk"])
    oly_takeaways = _build_key_takeaways(oly_team_df, oly_stack["risk"])

    pan_movers = _add_confidence_and_flags(pan_team_df[pan_team_df["min"] >= 5].sort_values(["gog_pts", "pts"], ascending=[False, False]).head(8))
    oly_movers = _add_confidence_and_flags(oly_team_df[oly_team_df["min"] >= 5].sort_values(["gog_pts", "pts"], ascending=[False, False]).head(8))

    def _takeaway_html(items: list[str]) -> str:
        return "".join(f"<li>{item}</li>" for item in items)

    def _team_block(team: str, takeaways: list[str], stack: dict[str, pd.DataFrame], usage: pd.DataFrame, movers: pd.DataFrame, momentum: pd.DataFrame) -> str:
        return f"""
    <section class='team-pane' data-team='{team}'>
      <div class='team-header'>
        <h2>{team} Snapshot</h2>
        <input type='search' class='team-search' data-team='{team}' placeholder='Filter by player name...' />
      </div>

      <details open>
        <summary>Key Takeaways</summary>
        <div class='callout'>
          <ul>{_takeaway_html(takeaways)}</ul>
        </div>
      </details>

      <details open>
        <summary>Stack-ranked Insights</summary>
        {_render_table_panel(f'{team.lower()}-impact', 'Impact', stack['impact'], stack_slim_cols, stack_detail_cols + ['ImpactNowScore'], score_badges={'ImpactNowScore': 'good'}, why_ranked_html=_render_why_ranked(stack['impact'], 'impact', 'ImpactNowScore'))}
        {_render_table_panel(f'{team.lower()}-trend', 'Trend', stack['trend'], trend_slim_cols, stack_detail_cols + ['TrendScore'], score_badges={'TrendScore': 'good'}, why_ranked_html=_render_why_ranked(stack['trend'], 'trend', 'TrendScore'))}
        {_render_table_panel(f'{team.lower()}-risk', 'Risk', stack['risk'], risk_slim_cols, stack_detail_cols + ['RiskScore'], score_badges={'RiskScore': 'bad'}, why_ranked_html=_render_why_ranked(stack['risk'], 'risk', 'RiskScore'))}
      </details>

      <details>
        <summary>Top Usage</summary>
        {_render_table_panel(f'{team.lower()}-usage', 'Top Usage (latest game)', usage, usage_slim_cols, usage_detail_cols)}
      </details>

      <details>
        <summary>Top Movers</summary>
        {_render_table_panel(f'{team.lower()}-movers', 'Top Movers — Points (GoG)', movers, movers_slim_cols, movers_detail_cols)}
      </details>

      <details>
        <summary>Momentum Watch</summary>
        {_render_table_panel(f'{team.lower()}-momentum', 'Momentum Watch', momentum, momentum_slim_cols, momentum_detail_cols)}
      </details>
    </section>
        """

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EuroLeague PAN/OLY Player Report — {season_code}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 20px; color: #1f2937; background:#f8fafc; }}
    h1,h2,h3,h4 {{ margin: .4rem 0; }}
    .muted {{ color:#6b7280; }}
    .warning {{ background:#fff4e5; border:1px solid #f59e0b; border-radius:8px; padding:10px; margin:12px 0; }}
    .tabs {{ display:flex; gap:8px; margin:14px 0; }}
    .tab-btn {{ border:1px solid #cbd5e1; background:#fff; border-radius:999px; padding:8px 16px; font-weight:700; cursor:pointer; }}
    .tab-btn.active {{ background:#1d4ed8; color:#fff; border-color:#1d4ed8; }}
    .team-pane {{ display:none; }}
    .team-pane.active {{ display:block; }}
    .team-header {{ display:flex; justify-content:space-between; align-items:center; gap:10px; margin:10px 0; flex-wrap:wrap; }}
    .team-search {{ padding:8px 10px; min-width:260px; border:1px solid #cbd5e1; border-radius:8px; }}
    details {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:8px 12px; margin:10px 0; }}
    summary {{ cursor:pointer; font-weight:700; }}
    .callout {{ background:#eef6ff; border:1px solid #bfdbfe; border-radius:8px; padding:10px; margin-top:8px; }}
    .table-panel {{ margin:10px 0 18px; }}
    .panel-header {{ display:flex; justify-content:space-between; align-items:center; }}
    .toggle-details {{ border:1px solid #94a3b8; background:#fff; border-radius:6px; padding:4px 8px; cursor:pointer; }}
    .details-wrap.hidden {{ display:none; }}
    .table-wrap {{ overflow:auto; border:1px solid #e2e8f0; border-radius:8px; background:#fff; }}
    table {{ width:100%; border-collapse:separate; border-spacing:0; font-size:.88rem; }}
    th,td {{ border-bottom:1px solid #e5e7eb; padding:7px 8px; }}
    th {{ position:sticky; top:0; background:#f1f5f9; z-index:2; text-align:left; white-space:nowrap; }}
    tbody tr:nth-child(even) {{ background:#fafafa; }}
    tbody tr:hover {{ background:#eaf2ff; }}
    td.num {{ white-space:nowrap; }}
    td.player-col, th:nth-child(2), td:nth-child(2) {{ white-space:normal; min-width:180px; max-width:280px; }}
    th:first-child, td:first-child {{ position:sticky; left:0; z-index:1; background:inherit; }}
    th:nth-child(2), td:nth-child(2) {{ position:sticky; left:100px; z-index:1; background:inherit; }}
    th:first-child {{ z-index:3; background:#e2e8f0; min-width:100px; }}
    th:nth-child(2) {{ z-index:3; background:#e2e8f0; }}
    .badge {{ padding:1px 7px; border-radius:999px; font-size:.78rem; font-weight:700; }}
    .badge.good,.badge.conf-high {{ background:#e8f7ee; color:#065f46; }}
    .badge.bad,.badge.conf-low {{ background:#fde8e8; color:#991b1b; }}
    .badge.conf-med {{ background:#fff4db; color:#92400e; }}
    .cell.good {{ background:#e8f7ee; color:#065f46; border-radius:4px; padding:1px 5px; }}
    .cell.bad {{ background:#fde8e8; color:#991b1b; border-radius:4px; padding:1px 5px; }}
    .cell.ok {{ background:#f1f5f9; color:#0f172a; border-radius:4px; padding:1px 5px; }}
    .info-icon {{ color:#64748b; font-size:.75rem; }}
    .why-ranked ul {{ margin-top:6px; }}
  </style>
</head>
<body>
  <h1>EuroLeague PAN/OLY Player Report — Season {season_code}</h1>
  <p class='muted'>Generated {timestamp}. Games date range: {min_game_date} → {max_game_date}.</p>
  {warning_banner}

  <section>
    <h3>Executive Summary</h3>
    <p><strong>PAN:</strong> Top scoring mover {pan_summary['top_scoring']} · Highest usage {pan_summary['highest_usage']}</p>
    <p><strong>OLY:</strong> Top scoring mover {oly_summary['top_scoring']} · Highest usage {oly_summary['highest_usage']}</p>
  </section>

  <div class='tabs'>
    <button class='tab-btn active' data-team='PAN' type='button'>PAN</button>
    <button class='tab-btn' data-team='OLY' type='button'>OLY</button>
  </div>

  {_team_block('PAN', pan_takeaways, pan_stack, pan_top_usage, pan_movers, pan_momentum)}
  {_team_block('OLY', oly_takeaways, oly_stack, oly_top_usage, oly_movers, oly_momentum)}

  {prediction_section}

  <details id='definitions' open>
    <summary>Definitions</summary>
    <p>
      This report uses efficiency + role context to explain performance. Points alone are noisy; TS/eFG tell
      you scoring quality; usage_proxy tells you who carried possessions; GoG and r5 show change vs trend.
    </p>

    <h4>TS (True Shooting)</h4>
    <ul>
      <li><strong>What it means:</strong> Scoring efficiency including 3s + free throws (single-number scorer efficiency).</li>
      <li><strong>Why it matters:</strong> Separates “volume points” from “valuable points”; more predictive than raw PTS.</li>
      <li><strong>How to read it here:</strong> TS ~ 0.55 is solid, &gt;0.60 strong, &lt;0.50 inefficient. TS can exceed 1.0 in tiny samples; treat TS_OUTLIER/LOW_USAGE as low confidence.</li>
    </ul>
    <details>
      <summary>Show formula</summary>
      <code>PTS / (2 × (FGA + 0.44 × FTA))</code>
    </details>

    <h4>eFG (Effective FG%)</h4>
    <ul>
      <li><strong>What it means:</strong> Shot value per attempt, giving 3s extra weight vs 2s.</li>
      <li><strong>Why it matters:</strong> Isolates shooting from free throws; useful to spot hot/cold shooting nights.</li>
      <li><strong>How to read it here:</strong> Compare eFG with TS—if TS is high but eFG is low, free throws likely drove efficiency.</li>
    </ul>
    <details>
      <summary>Show formula</summary>
      <code>(FGM + 0.5 × 3PM) / FGA</code>
    </details>

    <h4>usage_proxy</h4>
    <ul>
      <li><strong>What it means:</strong> Approximate possessions used: FGA + 0.44×FTA + TO.</li>
      <li><strong>Why it matters:</strong> Role indicator; high usage means offense ran through that player.</li>
      <li><strong>How to read it here:</strong> High usage + good TS = star-level impact. High usage + low TS (and/or HIGH_TO) = risk/drag. Low usage + high TS usually profiles as a finisher and can be small-sample sensitive.</li>
    </ul>
    <details>
      <summary>Show formula</summary>
      <code>FGA + 0.44 × FTA + TO</code> <span class='muted'>(mirrors possession-ending events)</span>
    </details>

    <h4>GoG (Game-over-Game)</h4>
    <ul>
      <li><strong>What it means:</strong> Current stat minus previous game stat for the same player.</li>
      <li><strong>Why it matters:</strong> Detects shocks from role change, matchup, foul trouble, or injury/return.</li>
      <li><strong>How to read it here:</strong> Always pair with Confidence + Flags. ROLE_SPIKE (Δmin) suggests rotation change; USG_SPIKE suggests more on-ball responsibility.</li>
    </ul>

    <h4>r5_* (Rolling 5)</h4>
    <ul>
      <li><strong>What it means:</strong> 5-game average.</li>
      <li><strong>Why it matters:</strong> Stabilizes noise and gives a better form read than one game.</li>
      <li><strong>How to read it here:</strong> When GoG and r5 move in the same direction, the trend signal is stronger.</li>
    </ul>

    <h4>Minutes (min)</h4>
    <ul>
      <li><strong>What it means:</strong> Coach trust / rotation role size.</li>
      <li><strong>Why it matters:</strong> Minute spikes often precede stat spikes; minute drops can explain weak boxscores.</li>
      <li><strong>How to read it here:</strong> Trust HIGH confidence (min ≥20) more than LOW.</li>
    </ul>

    <h4>Turnovers (TO)</h4>
    <ul>
      <li><strong>What it means:</strong> Possessions ended with a mistake.</li>
      <li><strong>Why it matters:</strong> High TO is costly, especially when paired with high usage.</li>
      <li><strong>How to read it here:</strong> TO 3+ is a red flag; rising gog_to under high usage signals growing risk.</li>
    </ul>
  </details>

  <div class='callout'>
    <p class='note'><strong>Quick interpretation cheat-sheet</strong></p>
    <ul>
      <li>Start with Impact/Trend top 3 + Why-ranked notes.</li>
      <li>Then check Top Usage to see who carried possessions.</li>
      <li>If a player shows TS_OUTLIER or LOW_USAGE, treat their efficiency as low-confidence.</li>
      <li>If HIGH_TO appears, downgrade performance quality even when points are high.</li>
      <li>Prefer r5_* for “form”, GoG for “shock”.</li>
    </ul>
  </div>

  <script>
    (() => {{
      const tabButtons = document.querySelectorAll('.tab-btn');
      const panes = document.querySelectorAll('.team-pane');
      function setTeam(team) {{
        tabButtons.forEach((btn) => btn.classList.toggle('active', btn.dataset.team === team));
        panes.forEach((pane) => pane.classList.toggle('active', pane.dataset.team === team));
      }}
      tabButtons.forEach((btn) => btn.addEventListener('click', () => setTeam(btn.dataset.team)));
      setTeam('PAN');

      document.querySelectorAll('.toggle-details').forEach((btn) => {{
        btn.addEventListener('click', () => {{
          const target = document.getElementById(btn.dataset.target);
          if (!target) return;
          target.classList.toggle('hidden');
          btn.textContent = target.classList.contains('hidden') ? 'Show details' : 'Hide details';
        }});
      }});

      document.querySelectorAll('.team-search').forEach((input) => {{
        input.addEventListener('input', () => {{
          const team = input.dataset.team;
          const q = input.value.trim().toLowerCase();
          const pane = document.querySelector(`.team-pane[data-team='${{team}}']`);
          if (!pane) return;
          pane.querySelectorAll('tbody tr').forEach((row) => {{
            const player = (row.dataset.playerName || '').toLowerCase();
            row.style.display = !q || player.includes(q) ? '' : 'none';
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
