import pygame
from typing import Tuple, List
from core.pellet import Pellet, PowerPellet
from core.pacman import Pacman

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
                    pygame.draw.rect(screen, (0, 0, 255), (x * 32, y * 32, 32, 32))
                elif cell == '.':
                    pygame.draw.circle(screen, (255, 255, 0), (x * 32 + 16, y * 32 + 16), 4)
                elif cell == 'o':
                    pygame.draw.circle(screen, (255, 255, 0), (x * 32 + 16, y * 32 + 16), 8)

    def check_pellet_collision(self, pacman: Pacman) -> None:
        for pellet in self.pellets:
            if pacman.check_collision(pellet):
                self.handle_pellet_collision(pacman, pellet)

        for power_pellet in self.power_pellets:
            if pacman.check_collision(power_pellet):
                self.handle_power_pellet_collision(pacman, power_pellet)

    def handle_pellet_collision(self, pacman: Pacman, pellet: Pellet) -> None:
        self.pellets.remove(pellet)
        pacman.eat_pellet(pellet)

    def handle_power_pellet_collision(self, pacman: Pacman, power_pellet: PowerPellet) -> None:
        self.power_pellets.remove(power_pellet)
        pacman.eat_power_pellet(power_pellet)

    def is_wall(self, x: int, y: int) -> bool:
        return self.grid[y][x] == "#"
