"""Auto-start on login — Windows (winreg) / macOS (LaunchAgent) / Linux (.desktop)."""

from __future__ import annotations

import sys

_APP_NAME = "ToolBridge"
_LABEL = "com.toolbridge"


def is_autostart_enabled() -> bool:
    if sys.platform == "win32":
        return _win_is_enabled()
    if sys.platform == "darwin":
        return _mac_is_enabled()
    return _linux_is_enabled()


def enable_autostart() -> None:
    if sys.platform == "win32":
        _win_enable()
    elif sys.platform == "darwin":
        _mac_enable()
    else:
        _linux_enable()


def disable_autostart() -> None:
    if sys.platform == "win32":
        _win_disable()
    elif sys.platform == "darwin":
        _mac_disable()
    else:
        _linux_disable()


# ---------------------------------------------------------------------------
# Windows — HKCU\...\Run
# ---------------------------------------------------------------------------

def _win_key_path() -> str:
    return r"Software\Microsoft\Windows\CurrentVersion\Run"


def _win_is_enabled() -> bool:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _win_key_path(), 0, winreg.KEY_READ) as key:
            winreg.QueryValueEx(key, _APP_NAME)
            return True
    except (FileNotFoundError, OSError, ImportError):
        return False


def _win_enable() -> None:
    import winreg
    exe = sys.executable
    # Use pythonw.exe if available for no-console launch
    if exe.endswith("python.exe"):
        pythonw = exe[:-10] + "pythonw.exe"
        import os
        if os.path.isfile(pythonw):
            exe = pythonw
    if getattr(sys, "frozen", False):
        cmd = f'"{exe}" --desktop'
    else:
        cmd = f'"{exe}" -m toolbridge --desktop'
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _win_key_path(), 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, _APP_NAME, 0, winreg.REG_SZ, cmd)


def _win_disable() -> None:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _win_key_path(), 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, _APP_NAME)
    except (FileNotFoundError, OSError, ImportError):
        pass


# ---------------------------------------------------------------------------
# macOS — LaunchAgent plist
# ---------------------------------------------------------------------------

def _mac_plist_path() -> str:
    from pathlib import Path
    return str(Path.home() / "Library" / "LaunchAgents" / f"{_LABEL}.plist")


def _mac_is_enabled() -> bool:
    import os
    return os.path.isfile(_mac_plist_path())


def _mac_enable() -> None:
    import plistlib
    from pathlib import Path
    if getattr(sys, "frozen", False):
        args = [sys.executable, "--desktop"]
    else:
        args = [sys.executable, "-m", "toolbridge", "--desktop"]
    plist = {
        "Label": _LABEL,
        "ProgramArguments": args,
        "RunAtLoad": True,
        "KeepAlive": False,
    }
    path = Path(_mac_plist_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps(plist))


def _mac_disable() -> None:
    try:
        import os
        os.unlink(_mac_plist_path())
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Linux — XDG autostart .desktop
# ---------------------------------------------------------------------------

def _linux_desktop_path() -> str:
    from pathlib import Path
    return str(Path.home() / ".config" / "autostart" / "toolbridge.desktop")


def _linux_is_enabled() -> bool:
    import os
    return os.path.isfile(_linux_desktop_path())


def _linux_enable() -> None:
    from pathlib import Path
    path = Path(_linux_desktop_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(sys, "frozen", False):
        exec_cmd = f'"{sys.executable}" --desktop'
    else:
        exec_cmd = f'"{sys.executable}" -m toolbridge --desktop'
    content = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=ToolBridge\n"
        f"Exec={exec_cmd}\n"
        "Hidden=false\n"
        "NoDisplay=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )
    path.write_text(content, encoding="utf-8")


def _linux_disable() -> None:
    try:
        import os
        os.unlink(_linux_desktop_path())
    except OSError:
        pass
