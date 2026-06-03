"""Entry point for ``python -m toolbridge``."""

from __future__ import annotations

import sys

from .config import Settings
from .server import run_server


def main() -> None:
    if "--desktop" in sys.argv or _should_launch_gui():
        sys.argv = [a for a in sys.argv if a != "--desktop"]
        from .desktop import TrayApp
        settings = _load_settings_for_desktop()
        TrayApp(settings).run()
    else:
        settings = _load_settings_for_desktop()
        run_server(settings)


def _should_launch_gui() -> bool:
    """Detect no-console launch (pythonw.exe or macOS .app bundle)."""
    if sys.platform == "win32":
        return sys.executable.lower().endswith("pythonw.exe")
    return False


def _load_settings_for_desktop() -> Settings:
    """Load from config file first, fall back to environment."""
    from .config_file import load_config

    file_cfg = load_config()
    if file_cfg:
        return Settings.from_dict(file_cfg)
    return Settings.from_environment()


if __name__ == "__main__":
    main()
