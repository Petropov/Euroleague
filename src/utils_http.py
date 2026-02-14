"""HTTP helpers with retries and caching."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import BACKOFF_FACTOR, MAX_RETRIES


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.euroleaguebasketball.net/",
    "Origin": "https://www.euroleaguebasketball.net",
}


def build_session() -> requests.Session:
    """Create a requests session with retries for transient API errors."""
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_text(session: requests.Session, url: str, params: dict[str, str] | None = None, timeout: int = 30) -> str:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.text


def get_json(session: requests.Session, url: str, params: dict[str, str] | None = None, timeout: int = 30) -> dict:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def read_json_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text_cache(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def get_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("euroleague_pipeline")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(stream_handler)
    return logger
