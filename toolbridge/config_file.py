"""Persistent JSON configuration file for desktop mode."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

CONFIG_DIR = Path.home() / ".toolbridge"
CONFIG_PATH = CONFIG_DIR / "config.json"


def load_config() -> dict | None:
    """Load configuration from disk. Returns None if file doesn't exist."""
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def save_config(data: dict) -> None:
    """Atomically write configuration to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    # Atomic write: write to temp file, then rename
    fd, tmp = tempfile.mkstemp(dir=str(CONFIG_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(CONFIG_PATH))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
