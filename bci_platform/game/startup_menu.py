"""Pygame 启动菜单：选择模型受试者与 GDF 文件。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GDF_DIR, MODELS_DIR
from bci_platform.game.file_picker import SC_F, SC_O, SC_RETURN, SC_SPACE, is_scancode, pick_gdf_file
from bci_platform.game.fonts import get_font
from bci_platform.startup_prompt import list_gdf_files, list_model_subjects

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
HIGHLIGHT = (80, 160, 255)
GREEN = (80, 220, 120)
YELLOW = (240, 220, 60)
BTN_BG = (50, 50, 80)


class StartupMenu:
    def __init__(self, width: int = 820, height: int = 520):
        self.width = width
        self.height = height
        self.subjects = list_model_subjects() or ["A09"]
        self.gdf_files = list_gdf_files() or []
        self.step = 0
        self.sel_model = 0
        self.sel_gdf = 0
        self.model_path: Optional[Path] = None
        self.gdf_path: Optional[Path] = None
        self.message = ""
        self._running = True
        self.result: Optional[Tuple[Path, Path]] = None
        self.btn_browse = pygame.Rect(30, height - 52, 160, 36)
        self.btn_start = pygame.Rect(210, height - 52, 120, 36)

    def _model_path_for(self, subject: str) -> Path:
        if subject == "UNIVERSAL":
            p = MODELS_DIR / "motor_imagery_universal.pkl"
            return p if p.exists() else MODELS_DIR / "motor_imagery_model.pkl"
        p = MODELS_DIR / f"motor_imagery_{subject.lower()}.pkl"
        return p if p.exists() else MODELS_DIR / "motor_imagery_model.pkl"

    def _default_gdf_for(self, subject: str) -> Optional[Path]:
        p = GDF_DIR / f"{subject.upper()}T.gdf"
        return p if p.exists() else None

    def _confirm_model(self) -> None:
        subject = self.subjects[self.sel_model]
        self.model_path = self._model_path_for(subject)
        default_gdf = self._default_gdf_for(subject)
        if default_gdf and default_gdf in self.gdf_files:
            self.sel_gdf = self.gdf_files.index(default_gdf)
        elif self.gdf_files:
            self.sel_gdf = 0
        self.step = 1
        self.message = f"Model: {subject} -> now pick GDF"

    def _confirm_gdf(self) -> None:
        if not self.gdf_files:
            self.message = "No GDF in list - click Browse button"
            return
        self.gdf_path = self.gdf_files[self.sel_gdf]
        if self.model_path and self.gdf_path:
            self.result = (self.model_path, self.gdf_path)
            self._running = False

    def _open_file_dialog(self, screen_size: Tuple[int, int]) -> None:
        picked = pick_gdf_file(GDF_DIR if GDF_DIR.exists() else PROJECT_ROOT, screen_size)
        if picked and picked.exists():
            self.gdf_path = picked
            if picked not in self.gdf_files:
                self.gdf_files.append(picked)
                self.gdf_files.sort(key=lambda p: p.name)
            self.sel_gdf = self.gdf_files.index(picked)
            self.message = f"Loaded: {picked.name}"
        else:
            self.message = "No file selected"

    def _handle_key(self, event: pygame.event.Event, screen_size: Tuple[int, int]) -> None:
        if event.key == pygame.K_ESCAPE:
            self._running = False
            self.result = None
            return

        if self.step == 0:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.sel_model = (self.sel_model - 1) % len(self.subjects)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.sel_model = (self.sel_model + 1) % len(self.subjects)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE) or is_scancode(event, SC_RETURN) or is_scancode(event, SC_SPACE):
                self._confirm_model()
            elif pygame.K_1 <= event.key <= pygame.K_9:
                idx = event.key - pygame.K_1
                if idx < len(self.subjects):
                    self.sel_model = idx
                    self._confirm_model()
        else:
            if event.key in (pygame.K_UP, pygame.K_w):
                if self.gdf_files:
                    self.sel_gdf = (self.sel_gdf - 1) % len(self.gdf_files)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                if self.gdf_files:
                    self.sel_gdf = (self.sel_gdf + 1) % len(self.gdf_files)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE) or is_scancode(event, SC_RETURN) or is_scancode(event, SC_SPACE):
                self._confirm_gdf()
            elif is_scancode(event, SC_F) or is_scancode(event, SC_O):
                self._open_file_dialog(screen_size)
            elif event.key == pygame.K_b:
                self.step = 0
                self.message = ""

    def _draw_button(self, screen, rect: pygame.Rect, label: str, font, hover: bool = False) -> None:
        color = HIGHLIGHT if hover else BTN_BG
        pygame.draw.rect(screen, color, rect, border_radius=6)
        pygame.draw.rect(screen, GRAY, rect, 1, border_radius=6)
        txt = font.render(label, True, WHITE)
        screen.blit(txt, txt.get_rect(center=rect.center))

    def run(self) -> Optional[Tuple[Path, Path]]:
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        size = (self.width, self.height)
        pygame.display.set_caption("BCI Demo - Setup")
        clock = pygame.time.Clock()
        font = get_font(24)
        small = get_font(18)
        mouse_pos = (0, 0)

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.MOUSEMOTION:
                    mouse_pos = event.pos
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.step == 1 and self.btn_browse.collidepoint(event.pos):
                        self._open_file_dialog(size)
                    if self.step == 1 and self.btn_start.collidepoint(event.pos):
                        self._confirm_gdf()
                if event.type == pygame.KEYDOWN:
                    self._handle_key(event, size)

            screen.fill(BLACK)
            if self.step == 0:
                screen.blit(font.render("Step 1/2 - Select Model", True, WHITE), (30, 20))
                if self.message:
                    screen.blit(small.render(self.message, True, YELLOW), (30, 52))
                y = 90
                for i, s in enumerate(self.subjects):
                    color = HIGHLIGHT if i == self.sel_model else GRAY
                    pfx = "> " if i == self.sel_model else "  "
                    screen.blit(small.render(f"{pfx}{s}  ({self._model_path_for(s).name})", True, color), (40, y))
                    y += 28
                screen.blit(small.render("Up/Down or 1-9 | Enter=next | ESC=quit", True, GREEN), (30, self.height - 30))
            else:
                screen.blit(font.render("Step 2/2 - Select GDF File", True, WHITE), (30, 20))
                if self.message:
                    screen.blit(small.render(self.message, True, YELLOW), (30, 52))
                y = 90
                items = [g.name for g in self.gdf_files] or ["(empty - use Browse)"]
                for i, name in enumerate(items):
                    color = HIGHLIGHT if i == self.sel_gdf else GRAY
                    pfx = "> " if i == self.sel_gdf else "  "
                    screen.blit(small.render(f"{pfx}{name}", True, color), (40, y))
                    y += 28
                self._draw_button(screen, self.btn_browse, "Browse GDF...", small, self.btn_browse.collidepoint(mouse_pos))
                self._draw_button(screen, self.btn_start, "Start Game", small, self.btn_start.collidepoint(mouse_pos))
                screen.blit(small.render("Up/Down=list | Enter/Start=play | Click Browse=import file", True, GREEN), (30, self.height - 30))

            pygame.display.flip()
            clock.tick(30)

        return self.result


def run_startup_menu() -> Optional[Tuple[Path, Path]]:
    return StartupMenu().run()
