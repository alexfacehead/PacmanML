import pygame
from typing import List
from core.level import Level
from core.pacman import Pacman
from core.ghost import Ghost
from utils.constants import *

class Game:
    def __init__(self, screen: pygame.Surface, clock: pygame.time.Clock):
        self.screen = screen
        self.clock = clock
        self.running = True
        self.level = None
        self.pacman = None
        self.ghosts = []
        self.score = 0
        self.mode_timer = 0

    def load_level(self, level_file: str) -> None:
        """Loads a level from a text file and creates Pacman and ghosts."""
        self.level = Level(level_file)
        self.pacman = Pacman(self.level.start_position)
        self.ghosts = self.create_ghosts()

    def create_ghosts(self) -> List[Ghost]:
        """Creates a list of ghosts based on the level ghost positions."""
        ghosts = []
        for ghost_start_position in self.level.ghost_positions:
            ghosts.append(Ghost(ghost_start_position))
        return ghosts

    def handle_events(self) -> None:
        """Handles user input events such as keyboard or mouse."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.pacman.handle_key_event(event)

    def update(self) -> None:
        """Updates the game logic such as movement and collisions."""
        self.pacman.update(self.level)
        for ghost in self.ghosts:
            ghost.update(self.level, self.pacman)
        self.check_collisions()
        self.check_game_over()
        self.update_mode_timer()

    def draw(self) -> None:
        """Draws the game elements on the screen."""
        self.screen.fill(BLACK)
        self.level.draw(self.screen)
        self.pacman.draw(self.screen)
        for ghost in self.ghosts:
            ghost.draw(self.screen)
        self.draw_score()
        self.draw_lives()

    def check_collisions(self) -> None:
        """Checks if Pacman collides with any pellets or ghosts."""
        for pellet in self.level.pellets:
            if not pellet.eaten and pellet.check_collision(self.pacman.position, self.pacman.radius):
                pellet.eaten = True
                if pellet.is_power_pellet:
                    self.pacman.powered_up = True
                    self.pacman.power_up_timer = POWERUP_DURATION * 60  # Adjust the timer based on your game's requirements
                    self.score += POWERUP_SCORE
                else:
                    self.score += PELLET_SCORE
        for ghost in self.ghosts:
            if self.pacman.check_collision(ghost):
                self.handle_collision(ghost)

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
                # Reset positions of Pacman and ghosts
                self.pacman.reset_position()
                for g in self.ghosts:
                    g.reset_position()
                # Play sound effect
            else:
                # Game over
                # Play sound effect
                # Show game over screen
                pass

    def check_game_over(self) -> None:
        """Checks if the game is over by clearing all the pellets."""
        if not self.level.has_pellets():
            # Game over
            # Play sound effect
            # Show victory screen
            pass

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
        """Runs the main game loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)