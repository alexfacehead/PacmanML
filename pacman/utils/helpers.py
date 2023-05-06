import os
import pygame
from typing import Tuple

def load_image(image_name: str) -> pygame.Surface:
    """
    Load an image file and return a pygame.Surface object.
    """
    image_path = os.path.join("assets", image_name)
    return pygame.image.load(image_path).convert_alpha()

def grid_to_pixel(grid_x: int, grid_y: int) -> Tuple[int, int]:
    """
    Convert grid coordinates to pixel coordinates.
    """
    return grid_x * 32, grid_y * 32

def is_within_distance(point1: Tuple[int, int], point2: Tuple[int, int], distance: int) -> bool:
    """
    Check if point1 is within a certain distance from point2.
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) <= distance and abs(y1 - y2) <= distance

def load_sound(file_name: str) -> pygame.mixer.Sound:
    # Load a sound from assets folder and return a pygame.mixer.Sound
    pass