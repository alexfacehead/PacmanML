"""Renderer for drawing the Pacman game with good-looking graphics."""

import math
from typing import List, Optional, Tuple

from ..utils.constants import (
    GRID_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT,
    BLACK, WHITE, YELLOW, BLUE, DARK_BLUE, RED, GREEN,
    PACMAN_LIVES, FRIGHTENED, EATEN, FRIGHTENED_FLASH_TIME,
    SCATTER, CHASE,
)


class Renderer:
    """Handles all drawing using pygame. Gracefully does nothing if pygame
    is not available (headless training)."""

    def __init__(self):
        self._pygame = None
        self._screen = None
        self._font_small = None
        self._font_medium = None
        self._font_large = None
        self._pellet_pulse = 0
        self._pellet_pulse_dir = 1

    def init_display(self, screen) -> None:
        """Attach a pygame screen surface. Call after pygame.init()."""
        import pygame
        self._pygame = pygame
        self._screen = screen
        self._font_small = pygame.font.Font(None, 20)
        self._font_medium = pygame.font.Font(None, 30)
        self._font_large = pygame.font.Font(None, 48)

    @property
    def available(self) -> bool:
        return self._screen is not None

    # ------------------------------------------------------------------
    # Top-level draw
    # ------------------------------------------------------------------

    def draw(self, level, pacman, ghosts, score: int, lives: int,
             game_over: bool = False, level_complete: bool = False,
             ready: bool = False,
             mode: int = SCATTER, paused: bool = False,
             score_popups: list = None) -> None:
        if not self.available:
            return
        pg = self._pygame
        self._screen.fill(BLACK)
        self._draw_maze(level)
        self._draw_pellets(level)
        if not game_over and not level_complete:
            self._draw_pacman(pacman)
            for ghost in ghosts:
                self._draw_ghost(ghost)
        if score_popups:
            self._draw_score_popups(score_popups)
        self._draw_hud(score, lives, mode)
        if ready:
            self._draw_ready()
        if game_over:
            self._draw_game_over()
        if level_complete:
            self._draw_level_complete(score)
        if paused:
            self._draw_paused()
        pg.display.flip()

    # ------------------------------------------------------------------
    # Maze
    # ------------------------------------------------------------------

    def _draw_maze(self, level) -> None:
        pg = self._pygame
        gs = GRID_SIZE
        for y in range(level.height):
            for x in range(level.width):
                cell = level.grid[y][x]
                px, py = x * gs, y * gs
                if cell == "#":
                    # Dark fill with blue border for classic look
                    pg.draw.rect(self._screen, DARK_BLUE, (px, py, gs, gs))
                    # Draw borders only on sides adjacent to non-wall
                    self._draw_wall_borders(level, x, y, px, py, gs)
                elif cell == "-":
                    # Ghost house door -- horizontal pink line
                    mid_y = py + gs // 2
                    pg.draw.line(self._screen, (255, 184, 255),
                                 (px + 2, mid_y), (px + gs - 2, mid_y), 3)

    def _draw_wall_borders(self, level, gx: int, gy: int,
                           px: int, py: int, gs: int) -> None:
        """Draw blue borders on the inner edges of wall tiles."""
        pg = self._pygame
        thickness = 2
        # Check each neighbor; draw border on side facing non-wall
        # Top
        if gy > 0 and level.grid[gy - 1][gx] != "#":
            pg.draw.line(self._screen, BLUE,
                         (px, py), (px + gs - 1, py), thickness)
        # Bottom
        if gy < level.height - 1 and level.grid[gy + 1][gx] != "#":
            pg.draw.line(self._screen, BLUE,
                         (px, py + gs - 1), (px + gs - 1, py + gs - 1), thickness)
        # Left
        if gx > 0 and level.grid[gy][gx - 1] != "#":
            pg.draw.line(self._screen, BLUE,
                         (px, py), (px, py + gs - 1), thickness)
        # Right
        if gx < level.width - 1 and level.grid[gy][gx + 1] != "#":
            pg.draw.line(self._screen, BLUE,
                         (px + gs - 1, py), (px + gs - 1, py + gs - 1), thickness)

    # ------------------------------------------------------------------
    # Pellets
    # ------------------------------------------------------------------

    def _draw_pellets(self, level) -> None:
        pg = self._pygame
        gs = GRID_SIZE
        half = gs // 2

        # Pulse animation for power pellets
        self._pellet_pulse += self._pellet_pulse_dir * 0.15
        if self._pellet_pulse > 3 or self._pellet_pulse < 0:
            self._pellet_pulse_dir *= -1

        for p in level.pellets:
            if not p.eaten:
                cx = p.position[0] * gs + half
                cy = p.position[1] * gs + half
                pg.draw.circle(self._screen, WHITE, (cx, cy), 2)

        for p in level.power_pellets:
            if not p.eaten:
                cx = p.position[0] * gs + half
                cy = p.position[1] * gs + half
                radius = int(5 + self._pellet_pulse)
                pg.draw.circle(self._screen, WHITE, (cx, cy), radius)

    # ------------------------------------------------------------------
    # Pacman
    # ------------------------------------------------------------------

    def _draw_pacman(self, pacman) -> None:
        pg = self._pygame
        gs = GRID_SIZE
        half = gs // 2
        cx = pacman.position[0] * gs + half
        cy = pacman.position[1] * gs + half
        radius = half - 2

        # Determine rotation based on direction
        dx, dy = pacman.direction
        if dx == 1 and dy == 0:
            start_angle_offset = 0
        elif dx == -1 and dy == 0:
            start_angle_offset = math.pi
        elif dx == 0 and dy == -1:
            start_angle_offset = math.pi / 2
        elif dx == 0 and dy == 1:
            start_angle_offset = -math.pi / 2
        else:
            start_angle_offset = 0

        mouth_rad = math.radians(pacman.mouth_angle)

        # Draw pacman as a filled arc (pie-slice)
        # Build polygon points for the pac shape
        num_points = 30
        points = [(cx, cy)]  # center
        start = start_angle_offset + mouth_rad
        end = start_angle_offset + 2 * math.pi - mouth_rad
        for i in range(num_points + 1):
            angle = start + (end - start) * i / num_points
            px = cx + radius * math.cos(angle)
            py = cy - radius * math.sin(angle)
            points.append((px, py))
        points.append((cx, cy))

        if len(points) >= 3:
            pg.draw.polygon(self._screen, YELLOW, points)

    # ------------------------------------------------------------------
    # Ghosts
    # ------------------------------------------------------------------

    def _draw_ghost(self, ghost) -> None:
        pg = self._pygame
        gs = GRID_SIZE
        half = gs // 2
        cx = ghost.position[0] * gs + half
        cy = ghost.position[1] * gs + half
        radius = half - 2

        if ghost.is_eaten:
            # Just draw eyes
            self._draw_ghost_eyes(cx, cy, radius, ghost.direction)
            return

        color = ghost.color

        # Rounded top half (semicircle)
        pg.draw.circle(self._screen, color, (cx, cy - 1), radius)
        # Body rectangle (lower half)
        body_rect = pg.Rect(cx - radius, cy - 1, radius * 2, radius)
        pg.draw.rect(self._screen, color, body_rect)

        # Wavy bottom edge
        wave_y = cy + radius - 2
        wave_w = radius * 2
        num_waves = 3
        seg_w = wave_w / num_waves
        for i in range(num_waves):
            left_x = cx - radius + i * seg_w
            mid_x = left_x + seg_w / 2
            right_x = left_x + seg_w
            # Triangle pointing down
            points = [
                (left_x, wave_y),
                (mid_x, wave_y + 4),
                (right_x, wave_y),
            ]
            pg.draw.polygon(self._screen, color, points)

        # Eyes
        self._draw_ghost_eyes(cx, cy, radius, ghost.direction)

    def _draw_ghost_eyes(self, cx: int, cy: int, radius: int,
                         direction: Tuple[int, int]) -> None:
        pg = self._pygame
        eye_radius = max(radius // 3, 2)
        pupil_radius = max(eye_radius // 2, 1)
        eye_y = cy - 2
        left_eye_x = cx - radius // 3
        right_eye_x = cx + radius // 3

        # White of eyes
        pg.draw.circle(self._screen, WHITE, (left_eye_x, eye_y), eye_radius)
        pg.draw.circle(self._screen, WHITE, (right_eye_x, eye_y), eye_radius)

        # Pupils (offset based on direction)
        dx, dy = direction if direction != (0, 0) else (0, 0)
        offset_x = dx * (pupil_radius)
        offset_y = dy * (pupil_radius)
        pg.draw.circle(self._screen, (0, 0, 100),
                        (left_eye_x + offset_x, eye_y + offset_y), pupil_radius)
        pg.draw.circle(self._screen, (0, 0, 100),
                        (right_eye_x + offset_x, eye_y + offset_y), pupil_radius)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_score_popups(self, popups: list) -> None:
        """Draw floating score indicators that drift upward and fade."""
        pg = self._pygame
        gs = GRID_SIZE
        half = gs // 2
        for gx, gy, text, frames_left in popups:
            # Float upward based on how long it's been alive
            max_frames = 60
            elapsed = max_frames - frames_left
            drift = int(elapsed * 0.5)
            alpha = min(255, frames_left * 8)
            px = gx * gs + half
            py = gy * gs - drift
            surf = self._font_small.render(text, True, YELLOW)
            alpha_surf = pg.Surface(surf.get_size(), pg.SRCALPHA)
            alpha_surf.fill((255, 255, 255, alpha))
            surf.set_alpha(alpha)
            self._screen.blit(surf, (px - surf.get_width() // 2, py))

    def _draw_hud(self, score: int, lives: int, mode: int) -> None:
        pg = self._pygame
        hud_y = SCREEN_HEIGHT - GRID_SIZE

        # Score
        text = self._font_small.render(f"SCORE: {score}", True, WHITE)
        self._screen.blit(text, (8, hud_y + 4))

        # Lives (small pacman icons)
        for i in range(lives):
            lx = SCREEN_WIDTH - 30 - i * 22
            ly = hud_y + GRID_SIZE // 2
            pg.draw.circle(self._screen, YELLOW, (lx, ly), 8)

        # Mode indicator
        mode_str = {SCATTER: "SCATTER", CHASE: "CHASE"}.get(mode, "")
        if mode_str:
            mt = self._font_small.render(mode_str, True, (100, 100, 100))
            self._screen.blit(mt, (SCREEN_WIDTH // 2 - mt.get_width() // 2, hud_y + 4))

    def _draw_ready(self) -> None:
        text = self._font_large.render("READY!", True, YELLOW)
        x = SCREEN_WIDTH // 2 - text.get_width() // 2
        y = SCREEN_HEIGHT // 2 - text.get_height() // 2
        self._screen.blit(text, (x, y))

    def _draw_game_over(self) -> None:
        # Semi-transparent overlay
        overlay = self._pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill(BLACK)
        self._screen.blit(overlay, (0, 0))

        text = self._font_large.render("GAME OVER", True, RED)
        x = SCREEN_WIDTH // 2 - text.get_width() // 2
        y = SCREEN_HEIGHT // 2 - text.get_height() // 2
        self._screen.blit(text, (x, y))

    def _draw_level_complete(self, score: int) -> None:
        overlay = self._pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill(BLACK)
        self._screen.blit(overlay, (0, 0))

        text = self._font_large.render("LEVEL COMPLETE!", True, GREEN)
        x = SCREEN_WIDTH // 2 - text.get_width() // 2
        y = SCREEN_HEIGHT // 2 - text.get_height() // 2 - 20
        self._screen.blit(text, (x, y))

        score_text = self._font_medium.render(f"Score: {score}", True, WHITE)
        sx = SCREEN_WIDTH // 2 - score_text.get_width() // 2
        self._screen.blit(score_text, (sx, y + text.get_height() + 8))

        hint = self._font_small.render("Press R to restart", True, WHITE)
        hx = SCREEN_WIDTH // 2 - hint.get_width() // 2
        self._screen.blit(hint, (hx, y + text.get_height() + score_text.get_height() + 20))

    def _draw_paused(self) -> None:
        overlay = self._pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill(BLACK)
        self._screen.blit(overlay, (0, 0))

        text = self._font_large.render("PAUSED", True, YELLOW)
        x = SCREEN_WIDTH // 2 - text.get_width() // 2
        y = SCREEN_HEIGHT // 2 - text.get_height() // 2
        self._screen.blit(text, (x, y))

        hint = self._font_small.render("Press ESC to resume", True, WHITE)
        hx = SCREEN_WIDTH // 2 - hint.get_width() // 2
        self._screen.blit(hint, (hx, y + text.get_height() + 8))

    def draw_score(self, score: int) -> None:
        """Standalone score draw (for compatibility)."""
        if not self.available:
            return
        text = self._font_small.render(f"SCORE: {score}", True, WHITE)
        self._screen.blit(text, (8, 8))

    def draw_lives(self, lives: int) -> None:
        """Standalone lives draw (for compatibility)."""
        if not self.available:
            return
        pg = self._pygame
        for i in range(lives):
            pg.draw.circle(self._screen, YELLOW,
                           (SCREEN_WIDTH - 30 - i * 22, 14), 8)
