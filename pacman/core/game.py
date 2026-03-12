"""Core game engine for PacmanML -- supports human play and headless ML training."""

import os
from typing import List, Optional, Tuple

import numpy as np

from ..utils.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, GRID_SIZE, FPS,
    PACMAN_LIVES, PACMAN_SPEED, GHOST_SPEED,
    PELLET_SCORE, POWER_PELLET_SCORE, GHOST_SCORE,
    SCATTER, CHASE, FRIGHTENED, EATEN,
    SCATTER_DURATION, CHASE_DURATION,
    FRIGHTENED_DURATION, FRIGHTENED_FLASH_TIME, POWERUP_DURATION,
    GHOST_NAMES, READY_TIMER,
    NUM_STATE_CHANNELS, CHANNEL_WALLS, CHANNEL_PELLETS,
    CHANNEL_POWER_PELLETS, CHANNEL_PACMAN, CHANNEL_GHOSTS,
    CHANNEL_FRIGHTENED,
    ACTION_TO_DIR,
)
from ..core.level import Level
from ..core.pacman import Pacman
from ..core.ghost import Ghost
from ..core.renderer import Renderer  # noqa: imported lazily for headless


class Game:
    """The Pacman game engine.

    Can run in two modes:
    - Interactive (``headless=False``): opens a pygame window.
    - Headless (``headless=True``): no display, suitable for ML training.

    ``play_as`` selects which entity the human controls:
    - ``"pacman"`` (default)
    - ``"ghost"`` -- human controls ghost[0] (Blinky), Pacman is AI.
    """

    def __init__(self, headless: bool = False, play_as: str = "pacman"):
        self.headless = headless
        self.play_as = play_as

        # Core objects -- populated by load_level
        self.level: Optional[Level] = None
        self.pacman: Optional[Pacman] = None
        self.ghosts: List[Ghost] = []

        # State
        self.score = 0
        self.game_over = False
        self.level_complete = False
        self.running = True

        # Timers
        self.mode_timer = 0
        self.current_mode = SCATTER  # SCATTER or CHASE
        self.ready_timer = READY_TIMER
        self.ghost_eat_combo = 0  # multiplier for consecutive ghost eats
        self.paused = False

        # Reward accumulators (reset each step)
        self._reward_pacman = 0.0
        self._reward_ghost = 0.0

        # Floating score indicators [(x, y, text, frames_remaining)]
        self._score_popups = []

        # Cached wall state for get_state() -- set after load_level
        self._wall_channel: Optional[np.ndarray] = None

        # Spatial lookup for pellets: (x,y) -> pellet object
        self._pellet_at: dict = {}
        self._power_pellet_at: dict = {}

        # Pre-allocated state array for get_state()
        self._state_buf: Optional[np.ndarray] = None

        # Renderer (only created when needed)
        self._renderer: Optional[Renderer] = None
        self._screen = None
        self._clock = None

    @property
    def renderer(self) -> Renderer:
        if self._renderer is None:
            self._renderer = Renderer()
        return self._renderer

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_level(self, level_file: str) -> None:
        """Load a level file and create characters."""
        self.level = Level(level_file)

        # Pacman
        self.pacman = Pacman(self.level.start_position)

        # Ghosts
        self.ghosts = []
        ghost_positions = self.level.ghost_positions
        for i, name in enumerate(GHOST_NAMES):
            pos = ghost_positions[i] if i < len(ghost_positions) else ghost_positions[0]
            ghost = Ghost(
                start_position=pos,
                name=name,
                index=i,
                ghost_house_exit=self.level.ghost_house_door,
            )
            self.ghosts.append(ghost)

        # Wire up Blinky reference for Inky's targeting
        if len(self.ghosts) >= 3:
            self.ghosts[2].set_blinky_ref(self.ghosts[0])

        # Headless optimizations: skip frame-timing overhead
        # Original speeds: Pacman=8, Ghost=9. Setting both to 1 means
        # every step() is a real move. Scale all timers accordingly.
        if self.headless:
            self._timer_scale = PACMAN_SPEED  # divide timers by this
            self.pacman.speed = 1
            self.pacman.powerup_duration = POWERUP_DURATION // self._timer_scale
            for ghost in self.ghosts:
                ghost.speed = 1
                ghost.frightened_duration = FRIGHTENED_DURATION // self._timer_scale
                ghost.frightened_flash_time = FRIGHTENED_FLASH_TIME // self._timer_scale
                ghost.exit_timer = ghost.index * 8  # was index*60
        else:
            self._timer_scale = 1

        # Cache wall channel (walls never change)
        h, w = self.level.height, self.level.width
        wc = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                if self.level.grid[y][x] == "#":
                    wc[y, x] = 1.0
        self._wall_channel = wc

        # Build spatial pellet lookups for O(1) collision checks
        self._pellet_at = {p.position: p for p in self.level.pellets}
        self._power_pellet_at = {p.position: p for p in self.level.power_pellets}

        # Pre-allocate state buffer
        self._state_buf = np.zeros((NUM_STATE_CHANNELS, h, w), dtype=np.float32)

    def enable_test_mode(self, num_pellets: int = 10) -> None:
        """Strip level to only a few pellets near Pacman's spawn for quick testing."""
        from ..utils.pathfinding import manhattan_distance

        # Sort pellets by distance to pacman
        px, py = self.pacman.position
        self.level.pellets.sort(
            key=lambda p: manhattan_distance(p.position, (px, py))
        )
        # Keep only closest pellets, mark rest as eaten
        for i, pellet in enumerate(self.level.pellets):
            if i >= num_pellets:
                pellet.eaten = True
        # Remove all power pellets
        for pp in self.level.power_pellets:
            pp.eaten = True

    # ------------------------------------------------------------------
    # Single step (for ML)
    # ------------------------------------------------------------------

    def step(self, pacman_action: Optional[int] = None,
             ghost_actions: Optional[List[Optional[int]]] = None) -> None:
        """Advance the game by one frame.

        Args:
            pacman_action: Action index for Pacman (None = use internal input).
            ghost_actions: List of action indices for each ghost (None entries
                           use the ghost's built-in AI).
        """
        self._reward_pacman = 0.0
        self._reward_ghost = 0.0

        if self.game_over or self.level_complete:
            return

        # Ready countdown
        if self.ready_timer > 0:
            self.ready_timer -= 1
            return

        # Apply external actions
        if pacman_action is not None:
            self.pacman.set_action(pacman_action)

        if ghost_actions is not None:
            for i, action in enumerate(ghost_actions):
                if action is not None and i < len(self.ghosts):
                    self.ghosts[i].set_action(action)

        # Update
        self._update()

    # ------------------------------------------------------------------
    # Internal update
    # ------------------------------------------------------------------

    def _update(self) -> None:
        """Core update: move characters, check collisions, update timers."""
        # Move Pacman
        self.pacman.update(self.level, headless=self.headless)

        # Move ghosts
        for ghost in self.ghosts:
            ghost.update(self.level, self.pacman)

        # Collisions
        self._check_collisions()

        # Mode timer
        self._update_ghost_mode()

        # Score popups are purely visual — skip in headless
        if not self.headless:
            self._score_popups = [
                (x, y, text, t - 1)
                for x, y, text, t in self._score_popups if t > 1
            ]

    def _check_collisions(self) -> None:
        """Handle pellet eating and ghost collisions."""
        pos = self.pacman.position

        # Pellet — O(1) lookup
        pellet = self._pellet_at.get(pos)
        if pellet is not None and not pellet.eaten:
            pellet.eaten = True
            self.score += PELLET_SCORE
            self._reward_pacman += 1.0
            self._reward_ghost -= 0.1
            if not self.headless:
                self._score_popups.append((*pos, f"+{PELLET_SCORE}", 30))

        # Power pellet — O(1) lookup
        pp = self._power_pellet_at.get(pos)
        if pp is not None and not pp.eaten:
            pp.eaten = True
            self.score += POWER_PELLET_SCORE
            self._reward_pacman += 2.0
            self._reward_ghost -= 0.5
            if not self.headless:
                self._score_popups.append((*pos, f"+{POWER_PELLET_SCORE}", 45))
            self.pacman.activate_power_up()
            self.ghost_eat_combo = 0
            for ghost in self.ghosts:
                if ghost.state != EATEN:
                    ghost.set_frightened()

        # Ghost collisions
        for ghost in self.ghosts:
            if ghost.in_ghost_house:
                continue
            if self.pacman.check_collision(ghost):
                self._handle_ghost_collision(ghost)

        # Check level complete
        if self.level.total_pellets() == 0:
            self.level_complete = True
            self._reward_pacman += 50.0
            self._reward_ghost -= 50.0

    def _handle_ghost_collision(self, ghost: Ghost) -> None:
        """Handle Pacman colliding with a ghost."""
        if ghost.state == FRIGHTENED:
            # Eat the ghost
            gx, gy = ghost.position
            ghost._set_eaten()
            self.ghost_eat_combo += 1
            points = GHOST_SCORE * self.ghost_eat_combo
            self.score += points
            self._reward_pacman += 5.0 * self.ghost_eat_combo
            self._reward_ghost -= 5.0
            if not self.headless:
                self._score_popups.append((gx, gy, f"+{points}", 60))
        elif ghost.state != EATEN:
            # Pacman dies
            self.pacman.lose_life()
            self._reward_pacman -= 10.0
            self._reward_ghost += 10.0
            if self.pacman.lives <= 0:
                self.game_over = True
            else:
                # Reset positions
                self.pacman.reset_position()
                for g in self.ghosts:
                    g.reset_position()
                self.ready_timer = READY_TIMER // self._timer_scale

    def _update_ghost_mode(self) -> None:
        """Timer-based scatter/chase mode switching."""
        self.mode_timer += 1
        cycle_length = (SCATTER_DURATION + CHASE_DURATION) // self._timer_scale
        phase = self.mode_timer % cycle_length

        if phase < SCATTER_DURATION // self._timer_scale:
            new_mode = SCATTER
        else:
            new_mode = CHASE

        if new_mode != self.current_mode:
            self.current_mode = new_mode
            for ghost in self.ghosts:
                ghost.set_mode(new_mode)

    # ------------------------------------------------------------------
    # ML interface
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Return a multi-channel grid state for ML agents.

        Shape: (NUM_STATE_CHANNELS, height, width)
        Channels: walls, pellets, power_pellets, pacman, ghosts, frightened_ghosts
        """
        state = self._state_buf

        # Walls channel stays constant — copy from cache
        state[CHANNEL_WALLS] = self._wall_channel

        # Clear dynamic channels (1-5)
        state[CHANNEL_PELLETS:] = 0.0

        # Pellets
        for p in self.level.pellets:
            if not p.eaten:
                state[CHANNEL_PELLETS, p.position[1], p.position[0]] = 1.0

        # Power pellets
        for p in self.level.power_pellets:
            if not p.eaten:
                state[CHANNEL_POWER_PELLETS, p.position[1], p.position[0]] = 1.0

        # Pacman
        px, py = self.pacman.position
        state[CHANNEL_PACMAN, py, px] = 1.0

        # Ghosts
        for ghost in self.ghosts:
            gx, gy = ghost.position
            if ghost.state == FRIGHTENED:
                state[CHANNEL_FRIGHTENED, gy, gx] = 1.0
            elif ghost.state != EATEN:
                state[CHANNEL_GHOSTS, gy, gx] = 1.0

        # Return a copy — buffer stores references need unique arrays
        return state.copy()

    def get_reward_pacman(self) -> float:
        """Return accumulated reward for Pacman since last step."""
        return self._reward_pacman

    def get_reward_ghost(self) -> float:
        """Return accumulated reward for ghosts since last step."""
        return self._reward_ghost

    def is_done(self) -> bool:
        """True if the game is over or level is complete."""
        return self.game_over or self.level_complete

    def reset(self) -> None:
        """Reset for a new episode. Requires load_level to have been called."""
        if self.level is None:
            return
        self.level.reset_pellets()
        self.pacman.reset()
        self.pacman.start_position = self.level.start_position
        self.pacman.position = self.level.start_position
        for i, ghost in enumerate(self.ghosts):
            pos = (self.level.ghost_positions[i]
                   if i < len(self.level.ghost_positions)
                   else self.level.ghost_positions[0])
            ghost.start_position = pos
            ghost.reset_full()
        self.score = 0
        self.game_over = False
        self.level_complete = False
        self.mode_timer = 0
        self.current_mode = SCATTER
        self.ready_timer = READY_TIMER // self._timer_scale
        self.ghost_eat_combo = 0
        self._reward_pacman = 0.0
        self._reward_ghost = 0.0
        # Re-apply headless speed overrides after reset
        if self.headless:
            self.pacman.speed = 1
            self.pacman.powerup_duration = POWERUP_DURATION // self._timer_scale
            for ghost in self.ghosts:
                ghost.speed = 1
                ghost.frightened_duration = FRIGHTENED_DURATION // self._timer_scale
                ghost.frightened_flash_time = FRIGHTENED_FLASH_TIME // self._timer_scale
                ghost.exit_timer = ghost.index * 8

    # ------------------------------------------------------------------
    # Human-playable loop
    # ------------------------------------------------------------------

    def handle_events(self) -> None:
        """Process pygame events for human play."""
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset()
                else:
                    if self.play_as == "pacman":
                        self.pacman.handle_key_event(event.key)
                    elif self.play_as == "ghost" and len(self.ghosts) > 0:
                        # Human controls Blinky
                        action_map = {
                            pygame.K_UP: 0,
                            pygame.K_DOWN: 1,
                            pygame.K_LEFT: 2,
                            pygame.K_RIGHT: 3,
                        }
                        if event.key in action_map:
                            self.ghosts[0].set_action(action_map[event.key])

    def run(self) -> None:
        """Main game loop with pygame display."""
        import pygame

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("PacmanML")
            self._clock = pygame.time.Clock()
            self.renderer.init_display(self._screen)

        while self.running:
            self.handle_events()
            if not self.paused:
                self.step()

            self.renderer.draw(
                self.level, self.pacman, self.ghosts,
                self.score, self.pacman.lives,
                game_over=self.game_over,
                level_complete=self.level_complete,
                ready=(self.ready_timer > 0),
                mode=self.current_mode,
                paused=self.paused,
                score_popups=self._score_popups,
            )
            self._clock.tick(FPS)

        pygame.quit()
