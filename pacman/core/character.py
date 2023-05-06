from typing import Tuple
import pygame

class Character:
    def __init__(self, start_position: Tuple[int, int], speed: int):
        self.start_position = start_position
        self.position = start_position
        self.direction = None
        self.next_direction = None
        self.speed = speed

    def check_collision(self, obj) -> bool:
        return self.position == obj.position