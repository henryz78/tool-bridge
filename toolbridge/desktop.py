"""Desktop GUI mode — system tray icon + customtkinter configuration panel."""

from __future__ import annotations

import base64
import io
import json
import sys
from typing import Any

import customtkinter as ctk

from .autostart import is_autostart_enabled, enable_autostart, disable_autostart
from .config import Settings
from .config_file import load_config, save_config
from .icon import load_icon
from .server import start_server_threaded, stop_server, is_server_running, get_server_port

# Customtkinter global settings
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

# Fonts
_FONT_TITLE = ("Microsoft YaHei UI", 14, "bold")
_FONT_LABEL = ("Microsoft YaHei UI", 12)
_FONT_ENTRY = ("Microsoft YaHei UI", 12)
_FONT_CODE = ("Consolas", 12)
_FONT_STATUS = ("Microsoft YaHei UI", 11)
_FONT_SECTION = ("Microsoft YaHei UI", 13, "bold")


class TrayApp:
    """System tray application managing server lifecycle and config panel."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._root: ctk.CTk | None = None
        self._icon: Any = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Create tray icon, start server, and enter main loop."""
        import pystray

        menu = pystray.Menu(
            pystray.MenuItem(lambda _: "运行中" if is_server_running() else "已停止", None, enabled=False),
            pystray.MenuItem("启动服务", self._on_start, visible=lambda _: not is_server_running()),
            pystray.MenuItem("停止服务", self._on_stop, visible=lambda _: is_server_running()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("设置...", self._on_config),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self._on_quit),
        )

        self._icon = pystray.Icon("toolbridge", load_icon(), "Tool Bridge", menu)

        # Start server automatically
        self._start_server()

        # Run tray icon in background thread
        self._icon.run_detached()

        # Create a hidden ctk root to own the main loop
        self._root = ctk.CTk()
        self._root.withdraw()
        self._root.protocol("WM_DELETE_WINDOW", self._on_quit)
        try:
            self._root.mainloop()
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Server management
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        try:
            start_server_threaded(self.settings)
        except Exception as exc:
            self._notify_error(f"启动服务器失败：{exc}")

    def _stop_server(self) -> None:
        stop_server()

    # ------------------------------------------------------------------
    # Tray menu callbacks (called from pystray thread)
    # ------------------------------------------------------------------

    def _on_start(self, icon: Any, item: Any) -> None:
        self._start_server()
        self._refresh_menu()

    def _on_stop(self, icon: Any, item: Any) -> None:
        self._stop_server()
        self._refresh_menu()

    def _on_config(self, icon: Any, item: Any) -> None:
        if self._root is not None:
            self._root.after(0, self._show_config_window)

    def _on_quit(self, icon: Any | None = None, item: Any | None = None) -> None:
        if self._root is not None:
            self._root.after(0, self._do_quit)

    def _do_quit(self) -> None:
        self._cleanup()
        if self._root is not None:
            self._root.destroy()

    def _cleanup(self) -> None:
        self._stop_server()
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass

    def _refresh_menu(self) -> None:
        if self._icon is not None:
            try:
                self._icon.update_menu()
            except Exception:
                pass

    def _notify_error(self, msg: str) -> None:
        if self._root is not None:
            self._root.after(0, lambda: self._show_error(msg))

    def _show_error(self, msg: str) -> None:
        from tkinter import messagebox
        messagebox.showerror("Tool Bridge", msg)

    # ------------------------------------------------------------------
    # Configuration panel (customtkinter)
    # ------------------------------------------------------------------

    def _show_config_window(self) -> None:
        if self._root is None:
            return

        # Single-instance config window
        for w in self._root.winfo_children():
            if isinstance(w, ConfigWindow):
                w.lift()
                return

        win = ConfigWindow(self._root, self.settings, self._on_config_saved)
        win.show()

    def _on_config_saved(self, new_settings: Settings) -> None:
        self._stop_server()
        self.settings = new_settings
        self._start_server()
        self._refresh_menu()


# ---------------------------------------------------------------------------
# Section frame helper
# ---------------------------------------------------------------------------

def _section(parent: ctk.CTkFrame, title: str) -> ctk.CTkFrame:
    """Create a titled section with a header label and inner frame."""
    header = ctk.CTkLabel(parent, text=title, font=_FONT_SECTION, anchor="w")
    header.pack(fill="x", padx=4, pady=(12, 4))
    frame = ctk.CTkFrame(parent, corner_radius=10)
    frame.pack(fill="x", padx=4, pady=(0, 4))
    return frame


def _form_row(parent: ctk.CTkFrame, label_text: str, widget: ctk.CTkBaseClass) -> None:
    """Place a label + widget pair in a horizontal row."""
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=12, pady=4)
    label = ctk.CTkLabel(row, text=label_text, font=_FONT_LABEL, width=100, anchor="e")
    label.pack(side="left", padx=(0, 8))
    widget.pack(side="left", fill="x", expand=True)


# ---------------------------------------------------------------------------
# ConfigWindow
# ---------------------------------------------------------------------------

class ConfigWindow(ctk.CTkToplevel):
    """Modern configuration panel using customtkinter."""

    def __init__(self, parent: ctk.CTk, settings: Settings, on_save: Any):
        super().__init__(parent)
        self._save_callback = on_save
        self.title("Tool Bridge 设置")
        self.geometry("540x600")
        self.resizable(False, False)
        self.grab_set()
        self.transient(parent)

        # Set window icon
        try:
            img = load_icon()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            from tkinter import PhotoImage
            self._photo = PhotoImage(data=base64.b64encode(buf.getvalue()))
            self.iconphoto(True, self._photo)
        except Exception:
            pass

        self._settings = settings
        self._build_ui()
        self._load_from_settings(settings)

    def show(self) -> None:
        # Center on screen
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")

    def _build_ui(self) -> None:
        # Main scrollable container
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        # --- Title ---
        title = ctk.CTkLabel(main, text="Tool Bridge 设置", font=_FONT_TITLE, anchor="w")
        title.pack(fill="x", pady=(0, 12))

        # --- Connection section ---
        conn_frame = _section(main, "连接配置")

        self._upstream_url = ctk.CTkEntry(conn_frame, placeholder_text="http://127.0.0.1:3000", font=_FONT_ENTRY)
        _form_row(conn_frame, "上游地址:", self._upstream_url)

        port_timeout_row = ctk.CTkFrame(conn_frame, fg_color="transparent")
        port_timeout_row.pack(fill="x", padx=12, pady=4)

        ctk.CTkLabel(port_timeout_row, text="监听端口:", font=_FONT_LABEL, width=100, anchor="e").pack(side="left", padx=(0, 8))
        self._port = ctk.CTkEntry(port_timeout_row, width=100, placeholder_text="8080", font=_FONT_ENTRY)
        self._port.pack(side="left", padx=(0, 24))

        ctk.CTkLabel(port_timeout_row, text="超时(秒):", font=_FONT_LABEL, width=80, anchor="e").pack(side="left", padx=(0, 8))
        self._timeout = ctk.CTkEntry(port_timeout_row, width=100, placeholder_text="240", font=_FONT_ENTRY)
        self._timeout.pack(side="left")

        self._auth = ctk.CTkEntry(conn_frame, placeholder_text="Bearer sk-...", font=_FONT_ENTRY, show="•")
        _form_row(conn_frame, "认证头:", self._auth)

        # --- Model config section ---
        model_frame = _section(main, "模型配置")

        map_label = ctk.CTkLabel(model_frame, text="模型映射 (JSON):", font=_FONT_LABEL, anchor="w")
        map_label.pack(fill="x", padx=12, pady=(8, 2))
        self._model_map = ctk.CTkTextbox(model_frame, height=60, font=_FONT_CODE, corner_radius=8)
        self._model_map.pack(fill="x", padx=12, pady=(0, 8))

        native_label = ctk.CTkLabel(model_frame, text="原生工具模型 (JSON):", font=_FONT_LABEL, anchor="w")
        native_label.pack(fill="x", padx=12, pady=(4, 2))
        self._native_models = ctk.CTkTextbox(model_frame, height=40, font=_FONT_CODE, corner_radius=8)
        self._native_models.pack(fill="x", padx=12, pady=(0, 8))

        # --- Options section ---
        opt_frame = _section(main, "选项")

        self._autostart = ctk.CTkCheckBox(opt_frame, text="开机自动启动", font=_FONT_LABEL)
        self._autostart.pack(fill="x", padx=12, pady=6)

        self._retry = ctk.CTkCheckBox(opt_frame, text="解析失败时重试", font=_FONT_LABEL)
        self._retry.pack(fill="x", padx=12, pady=(0, 8))

        # --- Buttons ---
        btn_frame = ctk.CTkFrame(main, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(16, 8))

        save_btn = ctk.CTkButton(
            btn_frame, text="保存并重启", command=self._on_save,
            font=_FONT_LABEL, height=38, corner_radius=10,
            fg_color="#3B82F6", hover_color="#2563EB",
        )
        save_btn.pack(side="left", expand=True, padx=(0, 8))

        cancel_btn = ctk.CTkButton(
            btn_frame, text="取消", command=self.destroy,
            font=_FONT_LABEL, height=38, corner_radius=10,
            fg_color="gray60", hover_color="gray50",
        )
        cancel_btn.pack(side="left", expand=True, padx=(8, 0))

        # --- Status bar ---
        self._status = ctk.CTkLabel(main, text="加载中...", font=_FONT_STATUS, anchor="w")
        self._status.pack(fill="x", pady=(8, 0))

    def _load_from_settings(self, s: Settings) -> None:
        self._upstream_url.insert(0, s.upstream_url)
        self._port.insert(0, str(s.listen_port))
        if s.upstream_auth:
            self._auth.insert(0, s.upstream_auth)
        self._timeout.insert(0, str(s.upstream_timeout))
        self._model_map.insert("1.0", json.dumps(s.name_mapping, indent=2, ensure_ascii=False) if s.name_mapping else "{}")
        self._native_models.insert("1.0", json.dumps(sorted(s.native_tool_model_ids), ensure_ascii=False) if s.native_tool_model_ids else "[]")
        if is_autostart_enabled():
            self._autostart.select()
        if s.retry_on_parse_failure:
            self._retry.select()
        self._update_status()

    def _update_status(self) -> None:
        if is_server_running():
            port = get_server_port()
            self._status.configure(text=f"● 运行中 (端口 {port})", text_color="#22C55E")
        else:
            self._status.configure(text="● 已停止", text_color="#EF4444")

    def _on_save(self) -> None:
        # Parse fields
        try:
            port = int(self._port.get())
        except ValueError:
            from tkinter import messagebox
            messagebox.showerror("错误", "端口必须是整数", parent=self)
            return

        try:
            timeout = int(self._timeout.get())
        except ValueError:
            from tkinter import messagebox
            messagebox.showerror("错误", "超时必须是整数", parent=self)
            return

        model_map_text = self._model_map.get("1.0", "end").strip() or "{}"
        try:
            model_map = json.loads(model_map_text)
            if not isinstance(model_map, dict):
                raise ValueError("必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as exc:
            from tkinter import messagebox
            messagebox.showerror("错误", f"模型映射 JSON 无效：{exc}", parent=self)
            return

        native_text = self._native_models.get("1.0", "end").strip() or "[]"
        try:
            native_models = json.loads(native_text)
            if not isinstance(native_models, list):
                raise ValueError("必须是 JSON 数组")
        except (json.JSONDecodeError, ValueError) as exc:
            from tkinter import messagebox
            messagebox.showerror("错误", f"原生工具模型 JSON 无效：{exc}", parent=self)
            return

        # Build new Settings by replacing fields on existing settings to preserve other advanced fields
        import dataclasses
        new_settings = dataclasses.replace(
            self._settings,
            listen_port=port,
            upstream_url=self._upstream_url.get().strip(),
            upstream_timeout=timeout,
            upstream_auth=self._auth.get().strip(),
            name_mapping=model_map,
            native_tool_model_ids=set(native_models),
            retry_on_parse_failure=bool(self._retry.get()),
        )

        # Save to config file
        save_config(new_settings.to_dict())

        # Handle autostart
        if self._autostart.get():
            enable_autostart()
        else:
            disable_autostart()

        self._update_status()
        self._save_callback(new_settings)
        self.destroy()
