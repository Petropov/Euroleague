# EuroLeague PAN/OLY Game-over-Game Stats

This repository fetches EuroLeague data for **Panathinaikos (PAN)** and **Olympiakos (OLY)**, curates per-game player stats, and computes game-over-game deltas + rolling metrics.

## What it does

Pipeline steps:
1. Download season results XML and filter games involving PAN/OLY.
2. Download/cache each matching game boxscore JSON.
3. Build `player_game` dataset (one row per player per game) and derived metrics (`ts`, `efg`, `usage_proxy`).
4. Build `player_gog` dataset with:
   - `gog_{stat}` = delta vs previous game per player
   - `r5_{stat}` = rolling 5-game mean per player

Outputs are saved to `data/curated` as both **parquet** and **csv**:
- `games.parquet`, `games.csv`
- `player_game.parquet`, `player_game.csv`
- `player_gog.parquet`, `player_gog.csv`

Raw responses are cached under `data/raw` and ignored by git.

## Data sources

- Results feed (XML):
  - `https://api-live.euroleague.net/v1/results?seasonCode={SEASON_CODE}`
- Boxscore feed (JSON):
  - `https://live.euroleague.net/api/Boxscore?gamecode={GAME_NUMERIC_CODE}&seasoncode={SEASON_CODE}`

> Note: these endpoints are used by EuroLeague's web properties and may change over time.

## Local usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run each step:

```bash
python -m src.fetch_results --season E2025
python -m src.fetch_boxscores --season E2025
python -m src.build_player_gog --season E2025
```

Or run end-to-end:

```bash
python -m src.pipeline --season E2025
```

## Configuration

Edit `src/config.py` to change default season and tracked teams.

## Logging

Execution summary is written to `logs/pipeline.log`, including:
- number of PAN/OLY games found
- number of boxscores fetched vs cached
- player_game rows written
- player_gog rows written

## GitHub Actions

Workflow: `.github/workflows/update.yml`

- Runs daily at **06:00 UTC** and on manual dispatch.
- Installs dependencies, runs pipeline, and commits curated updates when files changed.
