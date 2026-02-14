"""Parsing utilities for XML and JSON payloads."""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterable


def safe_int(value, default=None):
    if value in (None, "", "null"):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    if value in (None, "", "null"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_text(value, default=""):
    if value is None:
        return default
    return str(value).strip()


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", safe_text(text)).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return slug or "unknown"


def extract_gamecode_num(gamecode_full: str) -> int | None:
    if not gamecode_full:
        return None
    if "_" in gamecode_full:
        return safe_int(gamecode_full.rsplit("_", maxsplit=1)[-1], default=None)
    return safe_int(gamecode_full, default=None)


def _iter_game_elements(root: ET.Element) -> Iterable[ET.Element]:
    for elem in root.iter():
        tag = elem.tag.lower().split("}")[-1]
        if tag in {"game", "g", "match"}:
            if elem.attrib:
                yield elem


def parse_results_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    games: list[dict] = []

    for elem in _iter_game_elements(root):
        attrs = {k.lower(): v for k, v in elem.attrib.items()}
        gamecode_full = (
            attrs.get("gamecode")
            or attrs.get("code")
            or attrs.get("id")
            or attrs.get("game_code")
        )
        if not gamecode_full:
            continue

        games.append(
            {
                "gamecode_full": gamecode_full,
                "gamecode_num": extract_gamecode_num(gamecode_full),
                "date": attrs.get("date") or attrs.get("datetime") or attrs.get("gamedate") or attrs.get("time"),
                "round": attrs.get("round") or attrs.get("phase") or attrs.get("group") or attrs.get("groupname"),
                "home_team_code": (attrs.get("hometeamcode") or attrs.get("homecode") or attrs.get("home_team") or "").upper(),
                "away_team_code": (attrs.get("awayteamcode") or attrs.get("awaycode") or attrs.get("away_team") or "").upper(),
                "home_score": safe_int(attrs.get("homescore") or attrs.get("hscore"), default=None),
                "away_score": safe_int(attrs.get("awayscore") or attrs.get("ascore"), default=None),
            }
        )
    return games
