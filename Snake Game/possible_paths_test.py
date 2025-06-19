# import torch
# import random
# import random
# import numpy as np
from enum import Enum
# from collections import deque
# from collections import namedtuple
# from model import Linear_QNet, QTrainer
# from snake_game_ai import SnakeGameAI, Direction, Point

WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20

for i in range(0, WIDTH//BLOCK_SIZE):
    print()
    for j in range(0, HEIGHT//BLOCK_SIZE):
        print(f"({i:>2}, {j:<2}) ", end='')


class Direction(Enum):
    RIGHT = 1
    LEFT = -1
    UP = -18
    DOWN = 18
