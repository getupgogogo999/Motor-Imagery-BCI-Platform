"""
Pygame 小球控制游戏 - 空格步进，鼠标按钮 Reset / Open GDF
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bci_platform.game.file_picker import SC_F, SC_O, SC_R, SC_SPACE, is_scancode, pick_gdf_file
from bci_platform.game.fonts import get_font
from bci_platform.inference_engine import InferenceEngine, PredictionResult
from config import GDF_DIR

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (80, 220, 120)
GRAY = (120, 120, 120)
YELLOW = (240, 220, 60)
CYAN = (100, 200, 255)
RED = (220, 80, 80)
BTN_BG = (45, 45, 70)
BTN_HOVER = (70, 90, 140)


class BallGame:
    def __init__(
        self,
        engine: InferenceEngine,
        gdf_path: Path | str | None = None,
        width: int = 820,
        height: int = 560,
        fps: int = 30,
        step: int = 12,
    ):
        self.engine = engine
        self.gdf_path = Path(gdf_path) if gdf_path else None
        self.width = width
        self.height = height
        self.fps = fps
        self.step = step
        self.ball_radius = 20
        self.ball_y = height // 2 + 20
        self._running = False
        self._exhausted = False
        self.status_msg = ""
        self.status_timer = 0
        self.reset_banner = 0
        self.btn_next = pygame.Rect(16, height - 48, 130, 36)
        self.btn_reset = pygame.Rect(156, height - 48, 100, 36)
        self.btn_open = pygame.Rect(266, height - 48, 130, 36)
        self._init_state()

    def _init_state(self) -> None:
        self.ball_x = float(self.width // 2)
        self.last_result: Optional[PredictionResult] = None
        self.trial_index = 0
        self.correct = 0
        self.total = 0
        self.replay = self.engine.create_replay_source(gdf_path=self.gdf_path, shuffle=False)
        self._exhausted = False

    def _flash(self, msg: str, frames: int = 120) -> None:
        self.status_msg = msg
        self.status_timer = frames

    def _apply_prediction(self, result: PredictionResult, true_label: str | None = None) -> None:
        if result.label == "left":
            self.ball_x = max(self.ball_radius, self.ball_x - self.step)
        elif result.label == "right":
            self.ball_x = min(self.width - self.ball_radius, self.ball_x + self.step)
        self.last_result = result
        if true_label is not None:
            self.total += 1
            if result.label == true_label:
                self.correct += 1

    def _next_inference(self) -> bool:
        bx, by = self.replay.next_batch(1)
        if len(by) == 0:
            self._exhausted = True
            self._flash("All trials done - click Reset or press R")
            return False
        epoch = bx[0]
        true_raw = self.engine.label_encoder.inverse_transform([int(by[0])])[0]
        result = self.engine.predict_one(epoch)
        result.true_label = true_raw
        self._apply_prediction(result, true_raw)
        self.trial_index += 1
        self._exhausted = self.replay.position >= len(self.replay)
        return True

    def _do_reset(self) -> None:
        """完全重建 replay，确保 trial 从 0 开始。"""
        gdf = self.gdf_path
        self.ball_x = float(self.width // 2)
        self.last_result = None
        self.trial_index = 0
        self.correct = 0
        self.total = 0
        self._exhausted = False
        self.replay = self.engine.create_replay_source(gdf_path=gdf, shuffle=False)
        self.reset_banner = 45
        self._flash(f"RESET OK | trial 1/{len(self.replay)} | ball centered")

    def _reload_gdf(self, path: Path) -> None:
        self.gdf_path = path
        self._init_state()
        self._flash(f"GDF loaded: {path.name}")

    def _open_gdf_dialog(self, screen_size: tuple[int, int]) -> None:
        self._flash("Opening file dialog...")
        pygame.display.flip()
        picked = pick_gdf_file(GDF_DIR if GDF_DIR.exists() else PROJECT_ROOT, screen_size)
        if picked and picked.exists():
            self._reload_gdf(picked)
        else:
            self._flash("No file selected")

    def _draw_button(self, screen, rect: pygame.Rect, label: str, font, hover: bool) -> None:
        pygame.draw.rect(screen, BTN_HOVER if hover else BTN_BG, rect, border_radius=6)
        pygame.draw.rect(screen, GRAY, rect, 1, border_radius=6)
        t = font.render(label, True, WHITE)
        screen.blit(t, t.get_rect(center=rect.center))

    def _draw(self, screen, font, small, mouse_pos) -> None:
        screen.fill(BLACK)
        if self.reset_banner > 0:
            pygame.draw.rect(screen, RED, (0, 0, self.width, 4))

        pygame.draw.line(screen, GRAY, (self.width // 2, 90), (self.width // 2, self.height - 70), 1)
        pygame.draw.circle(screen, GREEN, (int(self.ball_x), self.ball_y), self.ball_radius)

        subj = self.engine.config.get("subject", "?")
        method = self.engine.bundle.get("model_name", "?")
        gdf_name = self.gdf_path.name if self.gdf_path else f"{subj}T.gdf"
        screen.blit(font.render(f"BCI Ball Demo | Model: {subj} | {method}", True, WHITE), (16, 10))
        screen.blit(small.render(f"GDF: {gdf_name}", True, CYAN), (16, 36))
        screen.blit(small.render(f"Progress: {self.replay.position}/{len(self.replay)}  (next = trial {self.replay.position + 1})", True, GRAY), (16, 56))

        if self.last_result:
            r = self.last_result
            color = YELLOW if r.label in ("left", "right") else GRAY
            screen.blit(font.render(f"Predict: {r.display_name} ({r.label})", True, color), (16, 84))
            screen.blit(small.render(f"Command: {r.command}  |  True: {r.true_label or '-'}", True, WHITE), (16, 112))
            acc = f"Steps: {self.trial_index}  Acc: {self.correct}/{self.total}"
            if self.total:
                acc += f" ({100 * self.correct / self.total:.0f}%)"
            screen.blit(small.render(acc, True, WHITE), (16, 134))
        else:
            screen.blit(small.render("Click [Next] or press SPACE for first trial", True, CYAN), (16, 84))

        if self.status_msg and self.status_timer > 0:
            screen.blit(small.render(self.status_msg, True, GREEN), (16, 160))

        self._draw_button(screen, self.btn_next, "Next (Space)", small, self.btn_next.collidepoint(mouse_pos))
        self._draw_button(screen, self.btn_reset, "Reset", small, self.btn_reset.collidepoint(mouse_pos))
        self._draw_button(screen, self.btn_open, "Open GDF", small, self.btn_open.collidepoint(mouse_pos))
        screen.blit(small.render("Tip: use MOUSE buttons if keyboard fails (IME) | ESC=quit", True, GRAY), (420, self.height - 40))

    def run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        size = (self.width, self.height)
        pygame.display.set_caption("BCI Motor Imagery Ball Game")
        clock = pygame.time.Clock()
        font = get_font(20)
        small = get_font(16)
        self._running = True
        mouse_pos = (0, 0)

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.MOUSEMOTION:
                    mouse_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.btn_next.collidepoint(event.pos):
                        self._next_inference()
                    elif self.btn_reset.collidepoint(event.pos):
                        self._do_reset()
                    elif self.btn_open.collidepoint(event.pos):
                        self._open_gdf_dialog(size)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._running = False
                    elif event.key == pygame.K_SPACE or is_scancode(event, SC_SPACE):
                        self._next_inference()
                    elif event.key == pygame.K_r or is_scancode(event, SC_R):
                        self._do_reset()
                    elif is_scancode(event, SC_F) or is_scancode(event, SC_O):
                        self._open_gdf_dialog(size)

            if self.status_timer > 0:
                self.status_timer -= 1
            if self.reset_banner > 0:
                self.reset_banner -= 1

            self._draw(screen, font, small, mouse_pos)
            pygame.display.flip()
            clock.tick(self.fps)

        pygame.quit()


def run_ball_game(
    model_path: str | None = None,
    gdf_path: str | Path | None = None,
    show_menu: bool = False,
) -> None:
    if show_menu or (model_path is None and gdf_path is None):
        from bci_platform.game.startup_menu import run_startup_menu
        picked = run_startup_menu()
        if not picked:
            return
        model_path, gdf_path = picked

    engine = InferenceEngine(model_path=model_path)
    game = BallGame(engine, gdf_path=gdf_path)
    game.run()
