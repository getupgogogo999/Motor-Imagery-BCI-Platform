"""Pygame 字体：优先加载 Windows 中文字体，避免方块乱码。"""
from __future__ import annotations

import os
from pathlib import Path

import pygame


def _windows_font_paths() -> list[Path]:
    root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    names = ["msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf"]
    return [root / n for n in names if (root / n).exists()]


def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    pygame.font.init()
    for path in _windows_font_paths():
        try:
            return pygame.font.Font(str(path), size)
        except Exception:
            continue
    for name in ("microsoftyaheui", "microsoftyahei", "simhei", "arial"):
        matched = pygame.font.match_font(name, bold=bold)
        if matched:
            return pygame.font.Font(matched, size)
    return pygame.font.SysFont("arial", size)
