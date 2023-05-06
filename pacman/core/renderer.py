import pygame
from typing import List
from core.level import Level
from core.pacman import Pacman
from core.ghost import Ghost
from utils.constants import *

class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen

    def draw(self, level: Level, pacman: Pacman, ghosts: List[Ghost], score: int, lives: int) -> None:
        self.screen.fill(BLACK)
        level.draw(self.screen)
        pacman.draw(self.screen)
        for ghost in ghosts:
            ghost.draw(self.screen)
        self.draw_score(score)
        self.draw_lives(lives)

    def draw_score(self, score: int) -> None:
        score_text = FONT_SMALL.render(f"Score: {score}", True, WHITE)
        score_rect = score_text.get_rect(topleft=(10, 10))
        self.screen.blit(score_text, score_rect)

    def draw_lives(self, lives: int) -> None:
        lives_text = FONT_SMALL.render(f"Lives: {lives}", True, WHITE)
        lives_rect = lives_text.get_rect(topleft=(10, 40))
        self.screen.blit(lives_text, lives_rect)