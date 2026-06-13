"""GDF 文件选择（兼容 Pygame + Windows 中文输入法环境）。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pygame


def pick_gdf_file(
    initial_dir: Path,
    screen_size: Optional[Tuple[int, int]] = None,
) -> Optional[Path]:
    """
    弹出文件选择框。
    先最小化 Pygame 窗口，避免对话框被挡住。
    """
    if not initial_dir.exists():
        initial_dir = initial_dir.parent

    minimized = False
    if pygame.display.get_init() and screen_size:
        try:
            pygame.display.iconify()
            pygame.time.wait(250)
            minimized = True
        except Exception:
            pass

    selected: Optional[Path] = None
    try:
        selected = _pick_via_tk(initial_dir)
        if selected is None:
            selected = _pick_via_win32(initial_dir)
    finally:
        if minimized and screen_size:
            try:
                pygame.display.set_mode(screen_size)
                pygame.display.set_caption("BCI Motor Imagery Ball Game")
            except Exception:
                pass

    return selected


def _pick_via_tk(initial_dir: Path) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        root.update_idletasks()
        root.update()
        path = filedialog.askopenfilename(
            parent=root,
            title="Select GDF file",
            initialdir=str(initial_dir.resolve()),
            filetypes=[("GDF files", "*.gdf"), ("All files", "*.*")],
        )
        root.destroy()
        return Path(path) if path else None
    except Exception:
        return None


def _pick_via_win32(initial_dir: Path) -> Optional[Path]:
    """Windows 原生对话框备用方案。"""
    try:
        import ctypes
        import os
        from ctypes import wintypes

        OFN_FILEMUSTEXIST = 0x00001000
        OFN_PATHMUSTEXIST = 0x00000800

        class OPENFILENAMEW(ctypes.Structure):
            _fields_ = [
                ("lStructSize", wintypes.DWORD),
                ("hwndOwner", wintypes.HWND),
                ("hInstance", wintypes.HINSTANCE),
                ("lpstrFilter", wintypes.LPCWSTR),
                ("lpstrCustomFilter", wintypes.LPWSTR),
                ("nMaxCustFilter", wintypes.DWORD),
                ("nFilterIndex", wintypes.DWORD),
                ("lpstrFile", wintypes.LPWSTR),
                ("nMaxFile", wintypes.DWORD),
                ("lpstrFileTitle", wintypes.LPWSTR),
                ("nMaxFileTitle", wintypes.DWORD),
                ("lpstrInitialDir", wintypes.LPCWSTR),
                ("lpstrTitle", wintypes.LPCWSTR),
                ("Flags", wintypes.DWORD),
                ("nFileOffset", wintypes.WORD),
                ("nFileExtension", wintypes.WORD),
                ("lpstrDefExt", wintypes.LPCWSTR),
                ("lCustData", wintypes.LPARAM),
                ("lpfnHook", wintypes.LPVOID),
                ("lpTemplateName", wintypes.LPCWSTR),
                ("pvReserved", wintypes.LPVOID),
                ("dwReserved", wintypes.DWORD),
                ("FlagsEx", wintypes.DWORD),
            ]

        buf = ctypes.create_unicode_buffer(65536)
        title = "Select GDF file"
        filt = "GDF Files (*.gdf)\0*.gdf\0All Files (*.*)\0*.*\0\0"
        ofn = OPENFILENAMEW()
        ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
        ofn.lpstrFilter = filt
        ofn.lpstrFile = buf
        ofn.nMaxFile = 65536
        ofn.lpstrInitialDir = str(initial_dir.resolve())
        ofn.lpstrTitle = title
        ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST

        if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
            return Path(buf.value)
    except Exception:
        return None
    return None

# SDL 物理键 scancode（不受中文输入法影响）
SC_R = 21
SC_F = 9
SC_O = 18
SC_SPACE = 44
SC_RETURN = 40


def is_scancode(event: pygame.event.Event, code: int) -> bool:
    return event.type == pygame.KEYDOWN and getattr(event, "scancode", -1) == code
