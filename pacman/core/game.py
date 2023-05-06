import pygame
from typing import List
from core.level import Level
from core.pacman import Pacman
from core.ghost import Ghost
from utils.constants import *
from core.pellet import PowerPellet
from core.renderer import Renderer

class Game:
    def __init__(self, screen: pygame.Surface, clock: pygame.time.Clock, headless: bool = False):
        self.screen = screen
        self.clock = clock
        self.running = True
        self.level = None
        self.pacman = None
        self.ghosts = []
        self.score = 0
        self.mode_timer = 0
        self.headless = headless
        if not headless:
            self.renderer = Renderer(screen)

    def load_level(self, level_file: str) -> None:
        """Loads a level from a text file and creates Pacman and ghosts."""
        self.level = Level(level_file)
        
        # Calculate level dimensions
        level_width = len(self.level.grid[0])
        level_height = len(self.level.grid)
        
        # Calculate and set GRID_SIZE
        global GRID_SIZE
        GRID_SIZE = min(SCREEN_WIDTH // level_width, SCREEN_HEIGHT // level_height)

        # Print GRID_SIZE, level_width, and level_height for debugging
        print(f"GRID_SIZE: {GRID_SIZE}, level_width: {level_width}, level_height: {level_height}")
        
        self.pacman = Pacman(self.level.start_position)
        self.ghosts = self.create_ghosts()
        
    def create_ghosts(self) -> List[Ghost]:
        """Creates a list of ghosts based on the level ghost positions."""
        ghosts = []
        for ghost_start_position in self.level.ghost_positions:
            ghosts.append(Ghost(ghost_start_position))
        return ghosts

    def handle_events(self) -> None:
        print("Enter handling game.handle_events.")
        """Handles user input events such as keyboard or mouse."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.pacman.handle_key_event(event)
        print("Exit game.handling_events.")

    def update(self) -> None:
        print("Enter game.update.")
        """Updates the game logic such as movement and collisions."""
        self.pacman.update(self.level)
        for ghost in self.ghosts:
            ghost.update(self.level, self.pacman)
        self.check_collisions()
        self.check_game_over()
        self.update_mode_timer()
        print("Exit game.update.")

    def check_collisions(self) -> None:
        print("Enter game.check_collisions")
        """Checks if Pacman collides with any pellets or ghosts."""
        for pellet in self.level.pellets:
            if not pellet.eaten and pellet.check_collision(self.pacman.position, self.pacman.radius):
                pellet.eaten = True
                if isinstance(pellet, PowerPellet):
                    self.pacman.powered_up = True
                    self.pacman.power_up_timer = POWERUP_DURATION * 60  # Adjust the timer based on your game's requirements
                    self.score += POWERUP_SCORE
                else:
                    self.score += PELLET_SCORE
        for ghost in self.ghosts:
            if self.pacman.check_collision(ghost):
                self.handle_collision(ghost)
        print("Enter game.check_collisions")

    def handle_collision(self, ghost: Ghost) -> None:
        """Handles the collision between Pacman and a ghost."""
        if self.pacman.powered_up:
            # Handle ghost being eaten
            ghost.reset_position()
            ghost.set_frightened(False)
            self.score += GHOST_SCORE
            # Play sound effect
        else:
            # Handle Pacman losing a life or game over
            if self.pacman.lives > 0:
                self.pacman.lives -= 1  # Decrement Pacman's lives
                # Reset positions of Pacman and ghosts
                self.pacman.reset_position()
                for g in self.ghosts:
                    g.reset_position()
                # Play sound effect
            else:
                print("Game over")
                pass

    def has_pellets(self) -> bool:
        return bool(self.pellets) or bool(self.power_pellets)

    def draw_score(self) -> None:
        """Draws the score on the top left corner of the screen."""
        score_text = FONT_SMALL.render(f"Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(topleft=(10, 10))
        self.screen.blit(score_text, score_rect)

    def update_mode_timer(self) -> None:
        """Updates the timer for switching between the ghost modes."""
        self.mode_timer += 1
        if self.mode_timer == 60 * SCATTER_MODE_DURATION:
            # Switch from scatter to chase mode
            for ghost in self.ghosts:
                ghost.set_scatter(False)
                ghost.set_chase(True)
        elif self.mode_timer == 60 * (SCATTER_MODE_DURATION + CHASE_MODE_DURATION):
            # Switch from chase to scatter mode
            for ghost in self.ghosts:
                ghost.set_chase(False)
                ghost.set_scatter(True)
            # Reset the timer
            self.mode_timer = 0

    def run(self) -> None:
        while self.running:
            self.handle_events()
            self.update()
            if not self.headless:
                self.renderer.draw(self.level, self.pacman, self.ghosts, self.score, self.pacman.lives)
            pygame.display.flip()
            self.clock.tick(60)

    def is_completed(self) -> bool:
        print("LEVEL CLEARED!")
        return len(self.level.pellets) == 0 and len(self.level.power_pellets) == 0
    
    def check_game_over(self) -> None:
        """Checks if the game is over and handles the game over event."""
        if self.pacman.lives <= 0 or self.is_completed():
            # Game over
            self.running = False
            print("GAME OVER!")
            # Play sound effect
            # Show game over screen
            pass