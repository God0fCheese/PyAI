import pygame
import random
import numpy as np
import colorsys

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
RED = (255, 0, 0)

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

class Bird:
    def __init__(self, x, y, color=(255, 255, 0)):
        self.pos = Point(x, y)
        self.vel = FLAP_STRENGTH
        self.alive = True
        self.score = 0
        self.fitness = 0
        self.color = color
        self.passed_pipe = False

    def flap(self, action):
        if np.array_equal(action, [0, 1]):
            self.vel = -FLAP_STRENGTH

    def update(self, gravity):
        self.vel += gravity
        self.pos.y += self.vel

class FlappyGame:

    def __init__(self, w:int=800, h:int=600, hd=True, population_size=50):
        self.w = w
        self.h = h
        self.high_detail = hd
        self.population_size = population_size

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Flappy Bird - Genetic Algorithm')
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.generation = 0
        self.max_score = 0
        self.game_over = False
        self.frame_count = 0

        # Generate birds with different colors
        self.birds = []
        for i in range(self.population_size):
            # Generate a unique color for each bird using HSV color space
            hue = i / self.population_size
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            color = (int(r * 255), int(g * 255), int(b * 255))
            self.birds.append(Bird(self.w//3, self.h//2, color))

        self.alive_birds = self.population_size

        y = random.randint(self.h // 3, 2*self.h // 3)
        self.top_pipes = [self._spawn_top_pipe(y)]
        self.bottom_pipes = [self._spawn_bottom_pipe(y)]
        self.visual_top_pipes = [self._spawn_top_pipe(y)]
        self.visual_bottom_pipes = [self._spawn_bottom_pipe(y)]

        self.pipe_vel = 5
        self.pipe_waiter = 0
        if self.high_detail:
            self.clouds = [Point(random.randint(0, self.w), random.randint(0, self.h//2)) for _ in range(6)]

    def reset_birds(self, new_generation=False):
        """Reset birds for a new generation or new game"""
        if new_generation:
            self.generation += 1
            # Keep the same birds but reset their positions and states
            for bird in self.birds:
                bird.pos = Point(self.w//3, self.h//2)
                bird.vel = FLAP_STRENGTH
                bird.alive = True
                bird.score = 0
                bird.fitness = 0
                bird.passed_pipe = False
        else:
            # Just reset positions and states
            for bird in self.birds:
                bird.pos = Point(self.w//3, self.h//2)
                bird.vel = FLAP_STRENGTH
                bird.alive = True
                bird.score = 0
                bird.passed_pipe = False

        self.alive_birds = len(self.birds)

        y = random.randint(self.h // 3, 2*self.h // 3)
        self.top_pipes = [self._spawn_top_pipe(y)]
        self.bottom_pipes = [self._spawn_bottom_pipe(y)]
        self.visual_top_pipes = [self._spawn_top_pipe(y)]
        self.visual_bottom_pipes = [self._spawn_bottom_pipe(y)]

        self.pipe_vel = 5
        self.pipe_waiter = 0
        self.frame_count = 0
        self.game_over = False

    def _spawn_top_pipe(self, y) -> Point:
        self.pipe_top = Point(self.w + 100, y - PIPE_GAP//2)
        return self.pipe_top

    def _spawn_bottom_pipe(self, y) -> Point:
        self.pipe_bottom = Point(self.w + 100, y + PIPE_GAP//2)
        return self.pipe_bottom

    def play_step(self, actions) -> tuple[list, bool, list]:
        """
        Play one step of the game for all birds

        Args:
            actions: List of actions for each bird

        Returns:
            list of fitness scores, game_over flag, list of scores
        """
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move birds based on actions
        for i, bird in enumerate(self.birds):
            if bird.alive:
                bird.flap(actions[i])

        # Update game state
        self._update_game_state()

        # Check collisions and update scores
        self._check_collisions()

        # Update UI
        self._update_ui()

        # Check if all birds are dead
        if self.alive_birds == 0:
            self.game_over = True

        # Get fitness scores and scores
        fitness_scores = [bird.fitness for bird in self.birds]
        scores = [bird.score for bird in self.birds]

        # Update clock
        self.clock.tick(SPEED)

        return fitness_scores, self.game_over, scores

    def _update_game_state(self):
        """Update game state (pipes, birds, etc.)"""
        self.frame_count += 1

        # Place a new pipe or wait
        if self.pipe_waiter * PIPE_FREQUENCY > max(100 - self.max_score, 50):
            y = random.randint(self.h // 3, 2*self.h // 3)
            self.top_pipes.insert(0, self._spawn_top_pipe(y))
            self.bottom_pipes.insert(0, self._spawn_bottom_pipe(y))
            self.visual_top_pipes.insert(0, self._spawn_top_pipe(y))
            self.visual_bottom_pipes.insert(0, self._spawn_bottom_pipe(y))
            self.pipe_waiter = 0

            # Reset passed_pipe flag for all birds
            for bird in self.birds:
                if bird.alive:
                    bird.passed_pipe = False
        else:
            # Move pipes
            for top_pipe in self.top_pipes:
                top_pipe.x -= self.pipe_vel
            for bottom_pipe in self.bottom_pipes:
                bottom_pipe.x -= self.pipe_vel
            for top_pipe in self.visual_top_pipes:
                top_pipe.x -= self.pipe_vel
            for bottom_pipe in self.visual_bottom_pipes:
                bottom_pipe.x -= self.pipe_vel

            # Update pipe timer
            self.pipe_waiter += 1

        # Remove pipes that have disappeared from the screen
        if self.visual_top_pipes and self.visual_top_pipes[-1].x < -PIPE_WIDTH:
            self.visual_top_pipes.pop()
            self.visual_bottom_pipes.pop()

        if self.top_pipes and self.top_pipes[-1].x + PIPE_WIDTH < self.w//3 - BIRD_SIZE:
            self.top_pipes.pop()
            self.bottom_pipes.pop()

        # Update bird positions
        for bird in self.birds:
            if bird.alive:
                bird.update(GRAVITY)

                # Check if bird passed a pipe
                if self.top_pipes and not bird.passed_pipe and bird.pos.x > self.top_pipes[-1].x + PIPE_WIDTH:
                    bird.score += 1
                    bird.fitness += 10  # Reward for passing a pipe
                    bird.passed_pipe = True

                    # Update max score
                    self.max_score = max(self.max_score, bird.score)

                    # Increase pipe velocity every 5 points
                    if bird.score % 5 == 0:
                        self.pipe_vel = min(self.pipe_vel + 1, MAX_PIPE_VEL)

                # Add fitness for staying alive
                bird.fitness += 0.1

                # Add fitness for staying near the center of the pipe gap
                if self.top_pipes:
                    pipe_center_y = (self.top_pipes[-1].y + self.bottom_pipes[-1].y) / 2
                    distance_to_center = abs(bird.pos.y - pipe_center_y)
                    # Reward birds that stay near the center of the gap
                    if distance_to_center < PIPE_GAP / 4:
                        bird.fitness += 0.1

    def _check_collisions(self):
        """Check collisions for all birds"""
        for bird in self.birds:
            if bird.alive:
                # Check collision with pipes
                for top_pipe, bottom_pipe in zip(self.top_pipes, self.bottom_pipes):
                    if self._is_collision(bird, top_pipe, bottom_pipe):
                        bird.alive = False
                        self.alive_birds -= 1
                        break

                # Check collision with ground or ceiling
                if bird.pos.y > self.h - BIRD_SIZE or bird.pos.y < 0:
                    bird.alive = False
                    self.alive_birds -= 1

    def _is_collision(self, bird, top_pipe, bottom_pipe) -> bool:
        # If the bird y isn't between the pipes
        if not top_pipe.y < bird.pos.y < bottom_pipe.y - BIRD_SIZE:
            # If the bird x is within a pipe
            if top_pipe.x - BIRD_SIZE < bird.pos.x < top_pipe.x + PIPE_WIDTH:
                return True
        return False

    def _update_ui(self) -> None:
        # Sky
        self.display.fill(BLUE)

        # Ground
        pygame.draw.rect(self.display, BROWN, pygame.Rect(0, self.h-20, self.w, 21))

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

        # Draw pipes
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

        # Draw birds
        for bird in self.birds:
            if bird.alive:
                pygame.draw.rect(self.display, bird.color, pygame.Rect(bird.pos.x, bird.pos.y, BIRD_SIZE, BIRD_SIZE))

        # Draw stats
        text_y = 0
        text = font.render(f'Generation: {self.generation}', True, WHITE)
        self.display.blit(text, (0, text_y))
        text_y += 24

        text = font.render(f'Alive: {self.alive_birds}/{len(self.birds)}', True, WHITE)
        self.display.blit(text, (0, text_y))
        text_y += 24

        text = font.render(f'Max Score: {self.max_score}', True, WHITE)
        self.display.blit(text, (0, text_y))

        pygame.display.flip()

    def get_state(self, bird_idx):
        """Get state for a specific bird"""
        bird = self.birds[bird_idx]

        if not bird.alive:
            # Return a default state for dead birds
            return np.zeros(6)

        # Get the nearest pipe
        if self.top_pipes:
            pipe_x = self.top_pipes[-1].x
            top_pipe_y = self.top_pipes[-1].y
            bottom_pipe_y = self.bottom_pipes[-1].y

            # Calculate normalized distances
            # Vertical distance to the center of the pipe gap
            pipe_center_y = (top_pipe_y + bottom_pipe_y) / 2
            vertical_distance = (bird.pos.y - pipe_center_y) / self.h

            # Horizontal distance to the pipe
            horizontal_distance = (pipe_x - bird.pos.x) / self.w

            # Bird's velocity (normalized)
            bird_velocity = bird.vel / 20  # Normalize by max velocity

            # Gap size (normalized)
            gap_size = (bottom_pipe_y - top_pipe_y) / self.h

            # Next pipe information if available
            if len(self.top_pipes) > 1:
                next_pipe_x = self.top_pipes[-2].x
                next_pipe_center_y = (self.top_pipes[-2].y + self.bottom_pipes[-2].y) / 2
                next_horizontal_distance = (next_pipe_x - bird.pos.x) / self.w
                next_vertical_distance = (bird.pos.y - next_pipe_center_y) / self.h
            else:
                # If no next pipe, use default values
                next_horizontal_distance = 1.0  # Far away
                next_vertical_distance = 0.0  # Center

            state = [
                vertical_distance,  # Normalized vertical distance to pipe center
                horizontal_distance,  # Normalized horizontal distance to pipe
                bird_velocity,  # Normalized bird velocity
                gap_size,  # Normalized gap size
                next_horizontal_distance,  # Distance to next pipe
                next_vertical_distance  # Vertical distance to next pipe center
            ]

            return np.array(state, dtype=float)
        else:
            # If no pipes, return a default state
            return np.zeros(6)
