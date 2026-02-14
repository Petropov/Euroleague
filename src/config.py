"""Project configuration constants."""

from pathlib import Path

TEAMS = {"PAN", "OLY"}
DEFAULT_SEASON_CODE = "E2025"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"
LOG_DIR = BASE_DIR / "logs"

RESULTS_ENDPOINT = "https://api-live.euroleague.net/v1/results"
BOXSCORE_ENDPOINT = "https://live.euroleague.net/api/Boxscore"
REQUEST_DELAY_SECONDS = 0.2
MAX_RETRIES = 5
BACKOFF_FACTOR = 0.8
