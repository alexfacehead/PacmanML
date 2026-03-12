"""Level loading and grid management."""

import os
from typing import List, Tuple, Optional

from ..core.pellet import Pellet, PowerPellet


class Level:
    """Represents a Pacman maze loaded from a text file."""

    def __init__(self, level_file: str):
        self.grid: List[List[str]] = self.load_grid(level_file)
        self._height = len(self.grid)
        self._width = len(self.grid[0]) if self._height > 0 else 0
        self.start_position: Tuple[int, int] = self.find_start_position()
        self.ghost_positions: List[Tuple[int, int]] = self.find_ghost_positions()
        self.pellets: List[Pellet] = []
        self.power_pellets: List[PowerPellet] = []
        self._create_pellets()
        self.tunnel_y: Optional[int] = None
        self._find_tunnel_row()
        self.ghost_house_door: Optional[Tuple[int, int]] = self._find_ghost_house_door()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_grid(self, level_file: str) -> List[List[str]]:
        """Load the grid from a text file. Each line is a row."""
        if not os.path.isabs(level_file):
            # Try relative to CWD first, then relative to package dir
            if os.path.exists(level_file):
                level_file = os.path.abspath(level_file)
            else:
                pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                level_file = os.path.join(pkg_dir, level_file)
        with open(level_file, "r") as f:
            grid = [list(line.rstrip("\n")) for line in f.readlines()]
        # Ensure all rows are the same width
        max_w = max(len(row) for row in grid)
        for row in grid:
            while len(row) < max_w:
                row.append(" ")
        return grid

    def find_start_position(self) -> Tuple[int, int]:
        """Find the 'P' cell (Pacman start)."""
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == "P":
                    self.grid[y][x] = " "  # clear the marker
                    return (x, y)
        raise ValueError("No Pacman start position 'P' found in level.")

    def find_ghost_positions(self) -> List[Tuple[int, int]]:
        """Find all 'G' cells (ghost start positions)."""
        positions = []
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == "G":
                    self.grid[y][x] = " "  # clear the marker
                    positions.append((x, y))
        return positions

    def _create_pellets(self) -> None:
        """Create Pellet / PowerPellet objects for '.' and 'o' cells."""
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == ".":
                    self.pellets.append(Pellet((x, y)))
                elif cell == "o":
                    self.power_pellets.append(PowerPellet((x, y)))

    def _find_tunnel_row(self) -> None:
        """Identify the tunnel row (a row with open space at both edges)."""
        for y, row in enumerate(self.grid):
            if row[0] == " " and row[-1] == " ":
                self.tunnel_y = y
                break

    def _find_ghost_house_door(self) -> Optional[Tuple[int, int]]:
        """Find the '-' (ghost house door) position."""
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == "-":
                    return (x, y)
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_wall(self, x: int, y: int) -> bool:
        """Check if (x, y) is a wall. Out of bounds counts as wall,
        except on the tunnel row where wrapping applies."""
        if y < 0 or y >= self._height:
            return True
        if x < 0 or x >= self._width:
            if y == self.tunnel_y:
                return False  # allow wrapping
            return True
        cell = self.grid[y][x]
        return cell == "#"

    def is_ghost_house_door(self, x: int, y: int) -> bool:
        """Check if position is the ghost house door."""
        if 0 <= x < self._width and 0 <= y < self._height:
            return self.grid[y][x] == "-"
        return False

    def get_valid_moves(self, x: int, y: int, allow_door: bool = False) -> List[Tuple[int, int]]:
        """Return list of (dx, dy) directions that don't lead into walls.
        If allow_door is True, the ghost house door '-' is also passable."""
        moves = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            # Handle wrapping
            if ny == self.tunnel_y:
                if nx < 0:
                    nx = self._width - 1
                elif nx >= self._width:
                    nx = 0
            if not self.is_wall(nx, ny):
                if allow_door or not self.is_ghost_house_door(nx, ny):
                    moves.append((dx, dy))
                elif self.is_ghost_house_door(nx, ny) and not allow_door:
                    # Ghosts can exit through the door but not re-enter
                    # (handled by caller via allow_door)
                    pass
        return moves

    def wrap_position(self, x: int, y: int) -> Tuple[int, int]:
        """Wrap position for tunnel traversal."""
        if y == self.tunnel_y:
            if x < 0:
                x = self._width - 1
            elif x >= self._width:
                x = 0
        return (x, y)

    def get_tunnel_positions(self) -> List[Tuple[int, int]]:
        """Return the entry/exit positions of the tunnel."""
        if self.tunnel_y is None:
            return []
        return [(0, self.tunnel_y), (self._width - 1, self.tunnel_y)]

    def total_pellets(self) -> int:
        """Total number of uneaten pellets and power pellets."""
        regular = sum(1 for p in self.pellets if not p.eaten)
        power = sum(1 for p in self.power_pellets if not p.eaten)
        return regular + power

    def reset_pellets(self) -> None:
        """Mark all pellets as uneaten."""
        for p in self.pellets:
            p.eaten = False
        for p in self.power_pellets:
            p.eaten = False
