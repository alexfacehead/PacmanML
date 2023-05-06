import pygame
from typing import Tuple, List
from core.pellet import Pellet, PowerPellet
from utils.constants import *

class Level:
    def __init__(self, level_file: str):
        self.grid = self.load_grid(level_file)
        self.start_position = self.find_start_position()
        self.ghost_positions = self.find_ghost_positions()
        self.pellets, self.power_pellets = self.create_pellets()

    def load_grid(self, level_file: str) -> List[List[str]]:
        with open(level_file, 'r') as file:
            grid = [list(line.strip()) for line in file.readlines()]
        return grid

    def find_start_position(self) -> Tuple[int, int]:
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == 'P':
                    return x, y

    def find_ghost_positions(self) -> List[Tuple[int, int]]:
        ghost_positions = []
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == 'G':
                    ghost_positions.append((x, y))
        return ghost_positions

    def create_pellets(self) -> Tuple[List[Pellet], List[PowerPellet]]:
        pellets = [Pellet((x, y)) for y, row in enumerate(self.grid) for x, cell in enumerate(row) if cell == '.']
        power_pellets = [PowerPellet((x, y)) for y, row in enumerate(self.grid) for x, cell in enumerate(row) if cell == 'o']
        return pellets, power_pellets

    def draw(self, screen: pygame.Surface) -> None:
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == '#':
                    pygame.draw.rect(screen, (0, 0, 255), (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                elif cell == '.':
                    pygame.draw.circle(screen, (255, 255, 0), (x * GRID_SIZE + 16, y * GRID_SIZE + 16), 4)
                elif cell == 'o':
                    pygame.draw.circle(screen, (255, 255, 0), (x * GRID_SIZE + 16, y * GRID_SIZE + 16), 8)

    def is_wall(self, x: int, y: int) -> bool:
        return self.grid[y][x] == "#"
