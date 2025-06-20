import pygame
import random
from collections import namedtuple

pygame.init()

font = pygame.font.SysFont('arial', 24)

Point = namedtuple('Point', ['x', 'y'])

# COLOURS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED1 = (255, 0, 0)
RED2 = (200, 0, 0)
RED3 = (150, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)
GREEN3 = (0, 150, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 0, 200)
BLUE3 = (0, 0, 150)

SPEED: int = 20
GRAVITY: int = 10
PIPE_FREQUENCY: int = 1
PIPE_GAP: int = 200
PIPE_WIDTH: int = 80
FLAP_STRENGTH: int = 10
BIRD_SIZE = 20

class FlappyGame:

    def __init__(self, w:int=512, h:int=512):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()

        self.score: int = 0
        self.game_over: bool = False

        self.bird = Point(self.w/3, self.h/2)
        self.bird_vel: float = FLAP_STRENGTH

        y = self.h * random.randint(1 // 3,
                                    2 // 3)  # The centre of the pipes is between 1/3 and 2/3 of the screen height
        self.top_pipes = [self._spawn_top_pipe(y)]
        self.bottom_pipes = [self._spawn_bottom_pipe(y)]
        
        self.pipe_vel = 5
        self.pipe_waiter = 0

    def _spawn_top_pipe(self, y) -> Point:
        self.pipe_top = Point(self.w + 100, y - PIPE_GAP//2)
        return self.pipe_top

    def _spawn_bottom_pipe(self, y) -> Point:
        self.pipe_bottom = Point(self.w + 100, y + PIPE_GAP//2)
        return self.pipe_bottom

    def play_step(self) -> tuple[bool, int]:
        # Get input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == (pygame.K_SPACE or pygame.K_UP):
                    self._flap()
        
        # Check if game over
        self.game_over: bool = False
        if self._is_collision():
            self.game_over = True
            return self.game_over, self.score

        # Place a new pipe or wait
        if self.pipe_waiter * PIPE_FREQUENCY > 10:
            y = self.h * random.randint(1 // 3,
                                        2 // 3)
            self.top_pipes.insert(0,self._spawn_top_pipe(y))
            self.bottom_pipes.insert(0,self._spawn_bottom_pipe(y))
            self.pipe_waiter = 0
        else:
            self.pipe_top.x -= self.pipe_vel
            self.pipe_bottom.x -= self.pipe_vel

        # Remove pipes that have disappeared from the screen
        if self.top_pipes[-1].x < -PIPE_WIDTH:
            self.top_pipes.pop()
            self.bottom_pipes.pop()

        # Add score
        if self.pipe_top.x < self.bird.x:
            self.score += 1

        # Update bird position
        self.bird_vel += GRAVITY
        self.bird.y += self.bird_vel

        # Update pipe timer
        self.pipe_waiter += 1
        
        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        return self.game_over, self.score
        
    def _is_collision(self, top_pipe, bottom_pipe) -> bool:
        if top_pipe.y + BIRD_SIZE//2 < self.bird.y < bottom_pipe.y - BIRD_SIZE//2:
            if top_pipe.x - (PIPE_WIDTH + BIRD_SIZE)//2 < self.bird.x < top_pipe.x + (PIPE_WIDTH + BIRD_SIZE)//2:
                return True
        return False

    def _update_ui(self) -> None:
        self.display.fill(BLUE3)
        
        for top_pipe, bottom_pipe in zip(self.top_pipes, self.bottom_pipes):
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(top_pipe.x, top_pipe.y, PIPE_WIDTH, self.h - top_pipe.y))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(bottom_pipe.x, bottom_pipe.y, PIPE_WIDTH, self.h - bottom_pipe.y))

    def _flap(self) -> None:
        # Set velocity to FLAP_STRENGTH
        pass

if __name__ == '__main__':
    game = FlappyGame()