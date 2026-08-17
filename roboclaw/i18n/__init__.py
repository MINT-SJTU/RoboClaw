"""Lightweight i18n -- loads shared JSON string tables for CLI and backend."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent
_DEFAULT_LANG = "en"


@lru_cache(maxsize=1)
def _load_strings() -> dict[str, dict[str, str]]:
    """Load and merge all JSON string files."""
    merged: dict[str, dict[str, str]] = {}
    for json_file in sorted(_DIR.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        for key, translations in data.items():
            if isinstance(translations, dict):
                merged[key] = translations
    return merged


def t(key: str, lang: str = "en", **kwargs: object) -> str:
    """Look up a translated string, format with kwargs.

    Falls back to English if lang not found, returns key if key not found.
    """
    strings = _load_strings()
    entry = strings.get(key)
    if entry is None:
        return key
    text = entry.get(lang) or entry.get(_DEFAULT_LANG) or key
    if kwargs:
        text = text.format(**kwargs)
    return text
