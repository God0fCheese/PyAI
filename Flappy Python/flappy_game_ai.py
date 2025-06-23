import pygame
import random
import numpy as np

pygame.init()

font = pygame.font.SysFont('arial', 24)

# COLOURS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)
GREEN3 = (0, 150, 0)
BLUE = (100, 230, 255)
YELLOW = (255, 255, 0)
BROWN = (150, 75, 0)

SPEED: int = 100
GRAVITY: int = 1
PIPE_FREQUENCY: int = 1
PIPE_GAP: int = 200
PIPE_WIDTH: int = 80
FLAP_STRENGTH: int = 10
BIRD_SIZE = 20
PIPE_DECO = 5
MAX_PIPE_VEL = 20

class Point:
    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y

class FlappyGame:

    def __init__(self, w:int=512, h:int=512, hd=True):
        self.w = w
        self.h = h
        self.high_detail = hd

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.score: int = 0
        self.game_over: bool = False

        self.bird = Point(self.w//3, self.h//2)
        self.bird_vel: float = FLAP_STRENGTH

        y = random.randint(self.h // 3,
                           2*self.h // 3)  # The centre of the pipes is between 1/3 and 2/3 of the screen height
        self.top_pipes = [self._spawn_top_pipe(y)]
        self.bottom_pipes = [self._spawn_bottom_pipe(y)]
        self.visual_top_pipes = [self._spawn_top_pipe(y)]
        self.visual_bottom_pipes = [self._spawn_bottom_pipe(y)]
        
        self.pipe_vel = 5
        self.pipe_waiter = 0
        if self.high_detail:
            self.clouds = [Point(random.randint(0, self.w), random.randint(0, self.h//2)) for _ in range(6)]

    def _spawn_top_pipe(self, y) -> Point:
        self.pipe_top = Point(self.w + 100, y - PIPE_GAP//2)
        return self.pipe_top

    def _spawn_bottom_pipe(self, y) -> Point:
        self.pipe_bottom = Point(self.w + 100, y + PIPE_GAP//2)
        return self.pipe_bottom

    def play_step(self, action) -> tuple[int, bool, int]:
        # Move
        self.flap(action)

        reward = 0

        # Check if game over
        self.game_over: bool = False
        for top_pipe, bottom_pipe in zip(self.top_pipes, self.bottom_pipes):
            if self._is_collision(top_pipe, bottom_pipe):
                self.game_over = True
                reward = -10
                return reward, self.game_over, self.score
        # for i in range(len(self.top_pipes)//2 + 1):
        #     if self._is_collision(self.top_pipes[-i], self.bottom_pipes[-i]):
        #         self.game_over = True
        #         return self.game_over, self.score

        if self.bottom_pipes[-1].y + PIPE_GAP//3 > self.bird.y > self.top_pipes[-1].y - PIPE_GAP//3:
            reward = 5

        # Place a new pipe or wait
        if self.pipe_waiter * PIPE_FREQUENCY > max(100 - self.score, 50):
            y = random.randint(self.h // 3,
                                        2*self.h // 3)
            self.top_pipes.insert(0,self._spawn_top_pipe(y))
            self.bottom_pipes.insert(0,self._spawn_bottom_pipe(y))
            self.visual_top_pipes.insert(0,self._spawn_top_pipe(y))
            self.visual_bottom_pipes.insert(0,self._spawn_bottom_pipe(y))
            self.pipe_waiter = 0
        else:
            for top_pipe in self.top_pipes:
                top_pipe.x -= self.pipe_vel
            for bottom_pipe in self.bottom_pipes:
                bottom_pipe.x -= self.pipe_vel
            for top_pipe in self.visual_top_pipes:
                top_pipe.x -= self.pipe_vel
            for bottom_pipe in self.visual_bottom_pipes:
                bottom_pipe.x -= self.pipe_vel

        # Remove pipes that have disappeared from the screen
        if self.visual_top_pipes[-1].x < -PIPE_WIDTH:
            self.visual_top_pipes.pop()
            self.visual_bottom_pipes.pop()

        if self.top_pipes[-1].x + PIPE_WIDTH < self.bird.x:
            self.top_pipes.pop()
            self.bottom_pipes.pop()

        # Add score if bird is past frontmost pipe within a pixel gap of the pipe's velocity
        if self.top_pipes[-1].x + PIPE_WIDTH//2 < self.bird.x < self.top_pipes[-1].x + PIPE_WIDTH//2 + self.pipe_vel:
            self.score += 1
            reward = 10
            if self.score % 5 == 0:
                self.pipe_vel = min(self.pipe_vel + 1, MAX_PIPE_VEL)

        # Update bird position
        self.bird_vel += GRAVITY
        self.bird.y += self.bird_vel

        # Update pipe timer
        self.pipe_waiter += 1
        
        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, self.game_over, self.score
        
    def _is_collision(self, top_pipe, bottom_pipe) -> bool:
        # If the bird y isn't between the pipes
        if not top_pipe.y < self.bird.y < bottom_pipe.y - BIRD_SIZE:
            # If the bird x is within a pipe
            if top_pipe.x - BIRD_SIZE < self.bird.x < top_pipe.x + PIPE_WIDTH:
                return True
        # Hit ground
        if self.bird.y > self.h - BIRD_SIZE:
            return True
        # Hit ceiling
        if self.bird.y < -BIRD_SIZE:
            return True
        # Else:
        return False

    def _update_ui(self) -> None:
        # Sky
        self.display.fill(BLUE)
        # Ground
        pygame.draw.rect(self.display, BROWN, pygame.Rect(0, self.h-20, self.w, 21 ))
        # Clouds
        if self.high_detail:
            for cloud in self.clouds:
                random.seed(cloud.y)
                pygame.draw.rect(self.display, WHITE, pygame.Rect(cloud.x, cloud.y, 40*random.randint(2,3), 40*random.randint(1,2)))
                if cloud.x < -120:
                    cloud.x = self.w + random.randint(0, self.w//2)
                    cloud.y = random.randint(0, self.h//2)
                else:
                    cloud.x -= self.pipe_vel//2

        for top_pipe, bottom_pipe in zip(self.visual_top_pipes, self.visual_bottom_pipes):
            d = pygame.draw.rect
            ds = self.display
            r = pygame.Rect
            pd = PIPE_DECO
            pw = PIPE_WIDTH
            # Top main pipe
            d(ds, GREEN3, r(top_pipe.x, 0, pw, top_pipe.y))
            # Bottom main pipe
            d(ds, GREEN3, r(bottom_pipe.x, bottom_pipe.y, pw, self.h - bottom_pipe.y))
            if self.high_detail:
                # Top main pipe detail
                d(ds, GREEN1, r(top_pipe.x + pd, 0, pw - 2*pd, top_pipe.y - pd))
                d(ds, GREEN2, r(top_pipe.x + pd + pw//3 + 1, 0, 2*pw//3 - 2*pd, top_pipe.y - pd))
                # Top pipe tip
                d(ds, GREEN3, r(top_pipe.x - pd, top_pipe.y - 4*pd, pw + 2*pd, 4*pd))
                d(ds, GREEN1, r(top_pipe.x, top_pipe.y - 3*pd, pw, 2*pd))
                d(ds, GREEN2, r(top_pipe.x + 2*pd + pw//3 + 1, top_pipe.y - 3*pd, 2*pw//3 - 2*pd, 2*pd))
                # Bottom main pipe detail
                d(ds, GREEN1, r(bottom_pipe.x + pd, bottom_pipe.y, pw - 2*pd, self.h - bottom_pipe.y))
                d(ds, GREEN2, r(bottom_pipe.x + pd + pw//3 + 1, bottom_pipe.y, 2*pw//3 - 2*pd, self.h-bottom_pipe.y))
                # Bottom pipe tip
                d(ds, GREEN3, r(bottom_pipe.x - pd, bottom_pipe.y, pw + 2*pd, 4*pd))
                d(ds, GREEN1, r(bottom_pipe.x, bottom_pipe.y + pd, pw, 2*pd))
                d(ds, GREEN2, r(bottom_pipe.x + 2*pd + pw//3 + 1, bottom_pipe.y + pd, 2*pw//3 - 2*pd, 2*pd))

            # pygame.draw.rect(self.display, WHITE, pygame.Rect(top_pipe.x, top_pipe.y, PIPE_WIDTH, 2))
            # pygame.draw.rect(self.display, WHITE, pygame.Rect(bottom_pipe.x, bottom_pipe.y, PIPE_WIDTH, 2))

            # pygame.draw.rect(self.display, BLACK, pygame.Rect(top_pipe.x, top_pipe.y, 2, 2))
            # pygame.draw.rect(self.display, BLACK, pygame.Rect(bottom_pipe.x, bottom_pipe.y, 2, 2))

        # Jump prediction
        # for i in range(30):
        #     pygame.draw.rect(self.display, GREEN1, pygame.Rect(self.bird.x+5*i + BIRD_SIZE//2, self.bird.y +
        #     BIRD_SIZE//2 - FLAP_STRENGTH*i + GRAVITY*i**2 , 2, 2))

        pygame.draw.rect(self.display, YELLOW, pygame.Rect(self.bird.x, self.bird.y, BIRD_SIZE, BIRD_SIZE))
        # pygame.draw.rect(self.display, RED1, pygame.Rect(self.bird.x, self.bird.y, 2, 2))

        text = font.render('Score: %d' % self.score, True, WHITE)
        self.display.blit(text, (0, 0))
        pygame.display.flip()

    def flap(self, action) -> None:
        if np.array_equal(action, [0, 1]):
            self.bird_vel = -FLAP_STRENGTH
