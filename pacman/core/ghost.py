"""Ghost character classes with personality-based AI."""

from typing import Tuple, List, Optional
from random import choice, random

from ..core.character import Character
from ..utils.constants import (
    GHOST_SPEED, GHOST_COLORS, GHOST_NAMES, GHOST_SCATTER_TARGETS,
    SCATTER, CHASE, FRIGHTENED, EATEN,
    GHOST_HOUSE_TARGET, GHOST_HOUSE_DOOR,
    BLUE, WHITE,
    ACTION_TO_DIR, UP, DOWN, LEFT, RIGHT, STAY,
    FRIGHTENED_DURATION, FRIGHTENED_FLASH_TIME,
)
from ..utils.pathfinding import find_path, get_direction, manhattan_distance


class Ghost(Character):
    """A ghost with personality-driven AI.

    Ghost names and behaviours:
    - BLINKY (red): chases Pacman directly
    - PINKY (pink): targets 4 cells ahead of Pacman
    - INKY (cyan): targets based on Blinky + Pacman offset
    - CLYDE (orange): chases when far, scatters when close (< 8 tiles)
    """

    def __init__(
        self,
        start_position: Tuple[int, int],
        name: str,
        index: int,
        ghost_house_exit: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(start_position, GHOST_SPEED)
        self.name = name
        self.index = index
        self.base_color = GHOST_COLORS.get(name, (255, 0, 0))
        self.scatter_target = GHOST_SCATTER_TARGETS.get(name, (0, 0))
        self.state = SCATTER
        self.previous_state = SCATTER
        self.frightened_timer = 0
        self.frightened_duration = FRIGHTENED_DURATION  # can be overridden for headless
        self.frightened_flash_time = FRIGHTENED_FLASH_TIME
        self.in_ghost_house = True
        self.ghost_house_exit = ghost_house_exit or GHOST_HOUSE_DOOR
        self.exit_timer = index * 60  # stagger ghost exits
        self._blinky_ref: Optional["Ghost"] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def color(self) -> Tuple[int, int, int]:
        """Current display color based on state."""
        if self.state == FRIGHTENED:
            if self.frightened_timer < self.frightened_flash_time:
                # Flash between blue and white
                if (self.frightened_timer // 10) % 2 == 0:
                    return WHITE
                return BLUE
            return BLUE
        if self.state == EATEN:
            return WHITE
        return self.base_color

    @property
    def is_frightened(self) -> bool:
        return self.state == FRIGHTENED

    @property
    def is_eaten(self) -> bool:
        return self.state == EATEN

    # ------------------------------------------------------------------
    # References (for Inky targeting)
    # ------------------------------------------------------------------

    def set_blinky_ref(self, blinky: "Ghost") -> None:
        self._blinky_ref = blinky

    # ------------------------------------------------------------------
    # State changes
    # ------------------------------------------------------------------

    def set_frightened(self) -> None:
        """Enter FRIGHTENED mode (power pellet eaten)."""
        if self.state == EATEN:
            return
        # Only save previous_state if not already frightened,
        # otherwise we'd restore to FRIGHTENED forever
        if self.state != FRIGHTENED:
            self.previous_state = self.state
            # Reverse direction on initial fright
            self.direction = (-self.direction[0], -self.direction[1])
        self.state = FRIGHTENED
        self.frightened_timer = self.frightened_duration

    def set_mode(self, mode: int) -> None:
        """Set ghost mode (SCATTER or CHASE). Does not override FRIGHTENED/EATEN."""
        if self.state not in (FRIGHTENED, EATEN):
            self.state = mode

    def _end_frightened(self) -> None:
        """Return to previous mode after frightened wears off."""
        self.state = self.previous_state

    def _set_eaten(self) -> None:
        """Ghost was eaten by powered-up Pacman."""
        self.state = EATEN

    # ------------------------------------------------------------------
    # AI targeting
    # ------------------------------------------------------------------

    def _get_chase_target(self, pacman_pos: Tuple[int, int],
                          pacman_dir: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate chase target based on ghost personality."""
        if self.name == "blinky":
            return pacman_pos

        if self.name == "pinky":
            tx = pacman_pos[0] + pacman_dir[0] * 4
            ty = pacman_pos[1] + pacman_dir[1] * 4
            return (tx, ty)

        if self.name == "inky":
            # Target is: 2 * (pacman + 2*dir) - blinky_pos
            ahead_x = pacman_pos[0] + pacman_dir[0] * 2
            ahead_y = pacman_pos[1] + pacman_dir[1] * 2
            if self._blinky_ref is not None:
                bx, by = self._blinky_ref.position
                return (2 * ahead_x - bx, 2 * ahead_y - by)
            return pacman_pos

        if self.name == "clyde":
            dist = manhattan_distance(self.position, pacman_pos)
            if dist > 8:
                return pacman_pos
            return self.scatter_target

        return pacman_pos

    def _get_target(self, pacman_pos: Tuple[int, int],
                    pacman_dir: Tuple[int, int]) -> Tuple[int, int]:
        """Get current movement target based on state."""
        if self.state == SCATTER:
            return self.scatter_target
        if self.state == CHASE:
            return self._get_chase_target(pacman_pos, pacman_dir)
        if self.state == EATEN:
            return GHOST_HOUSE_TARGET
        # FRIGHTENED -- handled separately (random)
        return self.position

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def set_action(self, action: int) -> None:
        """Set direction from an ML action index."""
        if action in ACTION_TO_DIR:
            d = ACTION_TO_DIR[action]
            if d != (0, 0):
                self.direction = d

    def update(self, level, pacman=None, blinky=None) -> None:
        """Update ghost position. Uses AI unless an action was set externally."""
        # Ghost house exit logic
        if self.in_ghost_house:
            if self.exit_timer > 0:
                self.exit_timer -= 1
                return
            # Move toward ghost house exit
            if self.position == self.ghost_house_exit:
                self.in_ghost_house = False
                # Move one cell up to exit
                self.position = (self.position[0], self.position[1] - 1)
                return
            path = find_path(level.grid, self.position, self.ghost_house_exit,
                             passable_chars={" ", "-", ".", "o"})
            if path:
                self.position = path[0]
            else:
                # Just move up
                self.position = (self.position[0], self.position[1] - 1)
            return

        # Frightened timer
        if self.state == FRIGHTENED:
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self._end_frightened()

        # Check if eaten ghost reached ghost house — re-enter and exit normally
        if self.state == EATEN:
            if self.position == GHOST_HOUSE_TARGET:
                self.state = self.previous_state
                self.in_ghost_house = True
                self.exit_timer = 60  # brief pause before re-exiting

        if not self.can_move():
            return

        # Get valid moves (exclude reversing direction unless frightened just started)
        valid_moves = level.get_valid_moves(
            self.position[0], self.position[1],
            allow_door=(self.state == EATEN or self.in_ghost_house),
        )

        if not valid_moves:
            return

        # Remove reverse direction (ghosts can't reverse except when frightened)
        reverse = (-self.direction[0], -self.direction[1])
        forward_moves = [m for m in valid_moves if m != reverse]
        if not forward_moves:
            forward_moves = valid_moves

        if self.state == FRIGHTENED:
            # Random movement when frightened
            chosen = choice(forward_moves)
        else:
            # Target-based movement
            pacman_pos = pacman.position if pacman else (0, 0)
            pacman_dir = pacman.direction if pacman else (0, 0)
            target = self._get_target(pacman_pos, pacman_dir)

            # Choose direction that minimizes distance to target
            best_dir = forward_moves[0]
            best_dist = float("inf")
            for d in forward_moves:
                nx = self.position[0] + d[0]
                ny = self.position[1] + d[1]
                nx, ny = level.wrap_position(nx, ny)
                dist = manhattan_distance((nx, ny), target)
                if dist < best_dist:
                    best_dist = dist
                    best_dir = d

            chosen = best_dir

        # Apply movement
        self.direction = chosen
        nx, ny = self.get_next_position(chosen)
        nx, ny = level.wrap_position(nx, ny)
        self.position = (nx, ny)

    def reset_position(self) -> None:
        """Reset ghost to start (inside ghost house)."""
        super().reset_position()
        self.in_ghost_house = True
        self.exit_timer = self.index * 60
        self.state = SCATTER
        self.frightened_timer = 0

    def reset_full(self) -> None:
        """Full reset for new game."""
        self.reset_position()
        self.previous_state = SCATTER
