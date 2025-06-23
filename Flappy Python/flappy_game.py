import pygame
import random

pygame.init()

font = pygame.font.SysFont('arial', 24)

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
YELLOW = (255, 255, 0)

SPEED: int = 40
GRAVITY: int = 1
PIPE_FREQUENCY: int = 2
PIPE_GAP: int = 100
PIPE_WIDTH: int = 80
FLAP_STRENGTH: int = 10
BIRD_SIZE = 20

class Point:
    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y

class FlappyGame:

    def __init__(self, w:int=512, h:int=512):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()

        self.score: int = 0
        self.game_over: bool = False

        self.bird = Point(self.w//3, self.h//2)
        self.bird_vel: float = FLAP_STRENGTH

        y = random.randint(self.h // 3,
                           2*self.h // 3)  # The centre of the pipes is between 1/3 and 2/3 of the screen height
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
        for top_pipe, bottom_pipe in zip(self.top_pipes, self.bottom_pipes):
            if self._is_collision(top_pipe, bottom_pipe):
                self.game_over = True
                return self.game_over, self.score
        # for i in range(len(self.top_pipes)//2 + 1):
        #     if self._is_collision(self.top_pipes[-i], self.bottom_pipes[-i]):
        #         self.game_over = True
        #         return self.game_over, self.score

        # Place a new pipe or wait
        if self.pipe_waiter * PIPE_FREQUENCY > max(100 - self.score, 50):
            y = random.randint(self.h // 3,
                                        2*self.h // 3)
            self.top_pipes.insert(0,self._spawn_top_pipe(y))
            self.bottom_pipes.insert(0,self._spawn_bottom_pipe(y))
            self.pipe_waiter = 0
        else:
            for top_pipe in self.top_pipes:
                top_pipe.x -= self.pipe_vel
            for bottom_pipe in self.bottom_pipes:
                bottom_pipe.x -= self.pipe_vel

        # Remove pipes that have disappeared from the screen
        if self.top_pipes[-1].x < -PIPE_WIDTH:
            self.top_pipes.pop()
            self.bottom_pipes.pop()

        # Add score if bird is past frontmost pipe within a pixel gap of the pipe's velocity
        if self.top_pipes[-1].x + PIPE_WIDTH//2 < self.bird.x < self.top_pipes[-1].x + PIPE_WIDTH//2 + self.pipe_vel:
            self.score += 1
            if self.score % 5 == 0:
                self.pipe_vel = min(self.pipe_vel + 1, 20)

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
        # If the bird y isn't between the pipes
        if not top_pipe.y < self.bird.y < bottom_pipe.y - BIRD_SIZE:
            # If the bird x is within a pipe
            if top_pipe.x - BIRD_SIZE < self.bird.x < top_pipe.x + PIPE_WIDTH:
                print(f"hit pipe top: {top_pipe.x, top_pipe.y} bottom: {bottom_pipe.x + PIPE_WIDTH, bottom_pipe.y - BIRD_SIZE} bird: {self.bird.x, self.bird.y}")
                return True
        if self.bird.y > self.h - BIRD_SIZE:
            print(f"hit ground: {self.bird.x, self.bird.y}")
            return True
        return False

    def _update_ui(self) -> None:
        self.display.fill(BLUE3)
        
        for top_pipe, bottom_pipe in zip(self.top_pipes, self.bottom_pipes):
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(top_pipe.x, 0, PIPE_WIDTH, top_pipe.y))
            pygame.draw.rect(self.display, RED2, pygame.Rect(bottom_pipe.x, bottom_pipe.y, PIPE_WIDTH, self.h - bottom_pipe.y))
            #
            # pygame.draw.rect(self.display, WHITE, pygame.Rect(top_pipe.x, top_pipe.y, PIPE_WIDTH, 2))
            # pygame.draw.rect(self.display, WHITE, pygame.Rect(bottom_pipe.x, bottom_pipe.y, PIPE_WIDTH, 2))
            #
            # pygame.draw.rect(self.display, BLACK, pygame.Rect(top_pipe.x, top_pipe.y, 2, 2))
            # pygame.draw.rect(self.display, BLACK, pygame.Rect(bottom_pipe.x, bottom_pipe.y, 2, 2))

        # Jump prediction
        # for i in range(30):
        #     pygame.draw.rect(self.display, GREEN1, pygame.Rect(self.bird.x+5*i + BIRD_SIZE//2, self.bird.y + BIRD_SIZE//2 - FLAP_STRENGTH*i + GRAVITY*i**2 , 2, 2))

        pygame.draw.rect(self.display, YELLOW, pygame.Rect(self.bird.x, self.bird.y, BIRD_SIZE, BIRD_SIZE))
        pygame.draw.rect(self.display, RED1, pygame.Rect(self.bird.x, self.bird.y, 2, 2))

        text = font.render('Score: %d' % self.score, True, WHITE)
        self.display.blit(text, (0, 0))
        pygame.display.flip()

    def _flap(self) -> None:
        self.bird_vel = -FLAP_STRENGTH

if __name__ == '__main__':
    game = FlappyGame()
    game.play_step()
    game._flap()
    pygame.time.delay(2000)
    while True:
        game_over, score = game.play_step()

        if game_over:
            break

    pygame.quit()