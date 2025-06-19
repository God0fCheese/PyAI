import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 24)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', ['x', 'y'])

# rgb colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)

BLOCK_SIZE = 20
SPEED = 100


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.width = w
        self.height = h

        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.width - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) //
                           BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Check if board is 80% accessible
        if not self.is_move_safe_and_accessible(self.direction):
            reward -= 10

        # 6. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 7. return game over and score
        game_over = False
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.width - BLOCK_SIZE or pt.x < 0 or pt.y > self.height - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for point in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(
                point.x+4, point.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render('Score: %d' % self.score, True, WHITE)
        self.display.blit(text, (0, 0))
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % len(clock_wise)
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % len(clock_wise)
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def get_board_array(self):
        """
        Returns a 2D numpy array representing the board:
        0 = empty, 1 = snake body
        """
        rows = self.height // BLOCK_SIZE
        cols = self.width // BLOCK_SIZE
        board = np.zeros((rows, cols), dtype=np.int8)
        for point in self.snake:
            x, y = int(point.x // BLOCK_SIZE), int(point.y // BLOCK_SIZE)
            if 0 <= y < rows and 0 <= x < cols:
                board[y, x] = 1
        return board

    def _get_next_head(self, direction):
        x, y = self.head.x, self.head.y
        if direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        return Point(x, y)

    def _is_within_bounds(self, pt):
        return 0 <= pt.x < self.width and 0 <= pt.y < self.height

    def _bfs_accessible(self, board, start):
        rows, cols = board.shape
        visited = np.zeros_like(board, dtype=bool)
        queue = [(start[0], start[1])]
        visited[start[0], start[1]] = True
        count = 1
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and board[nr, nc] == 0:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                    count += 1
        return count

    def is_move_safe_and_accessible(self, direction, min_accessible_ratio=0.8):
        """
        Simulate moving in the given direction. Returns True if at least min_accessible_ratio
        of empty cells are accessible from the new head position.
        """
        board = self.get_board_array()
        rows, cols = board.shape
        # Simulate move
        new_head = self._get_next_head(direction)
        if not self._is_within_bounds(new_head):
            return False
        nh_y, nh_x = int(
            new_head.y // BLOCK_SIZE), int(new_head.x // BLOCK_SIZE)
        if board[nh_y, nh_x] == 1:
            return False
        # Simulate snake body after move (remove tail if not eating food)
        temp_board = board.copy()
        temp_board[nh_y, nh_x] = 1
        tail = self.snake[-1]
        tail_y, tail_x = int(tail.y // BLOCK_SIZE), int(tail.x // BLOCK_SIZE)
        if new_head != self.food:
            temp_board[tail_y, tail_x] = 0
        # Count accessible empty cells from new head
        total_empty = np.sum(temp_board == 0)
        if total_empty == 0:
            return True  # No empty cells left
        accessible = self._bfs_accessible(temp_board, (nh_y, nh_x))
        return accessible / total_empty >= min_accessible_ratio
