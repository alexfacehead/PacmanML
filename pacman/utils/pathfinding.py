"""A* pathfinding for the Pacman grid."""

import heapq
from typing import List, Tuple, Optional


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance between two grid positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_direction(start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[int, int]:
    """Get the direction vector from start to end (clamped to -1, 0, 1)."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx != 0:
        dx = dx // abs(dx)
    if dy != 0:
        dy = dy // abs(dy)
    return (dx, dy)


def find_path(
    grid: List[List[str]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    passable_chars: Optional[set] = None,
) -> List[Tuple[int, int]]:
    """
    A* pathfinding on a grid. Returns a list of (x, y) positions from start
    to end (inclusive of end, exclusive of start). Returns empty list if no
    path exists.

    Walls are '#'. Everything else is passable by default, unless
    passable_chars is specified.
    """
    if start == end:
        return []

    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    def is_passable(x: int, y: int) -> bool:
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        cell = grid[y][x]
        if passable_chars is not None:
            return cell in passable_chars
        return cell != '#'

    if not is_passable(end[0], end[1]):
        # Target is a wall; find the closest passable neighbor
        return []

    # A* with (f_score, counter, x, y)
    counter = 0
    open_set = []
    heapq.heappush(open_set, (manhattan_distance(start, end), counter, start[0], start[1]))
    counter += 1

    came_from = {}
    g_score = {start: 0}

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while open_set:
        _, _, cx, cy = heapq.heappop(open_set)
        current = (cx, cy)

        if current == end:
            # Reconstruct path
            path = []
            node = end
            while node != start:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        current_g = g_score.get(current, float('inf'))

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy

            # Handle wrapping for tunnel rows
            if nx < 0:
                nx = width - 1
            elif nx >= width:
                nx = 0

            neighbor = (nx, ny)

            if not is_passable(nx, ny):
                continue

            tentative_g = current_g + 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan_distance(neighbor, end)
                heapq.heappush(open_set, (f, counter, nx, ny))
                counter += 1

    return []  # No path found
