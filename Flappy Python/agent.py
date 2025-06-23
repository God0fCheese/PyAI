import torch
import random
import numpy as np
import math
from collections import deque
from flappy_game_ai import FlappyGame
from model import Linear_QNet, QTrainer, PrioritizedReplayBuffer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 128  # Smaller batch size for more frequent updates
LR = 0.0005  # Lower learning rate for stability
GAMMA = 0.99  # Higher discount factor to value future rewards more


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 2000  # Decay over more games for better exploration
        self.gamma = GAMMA

        # Use prioritized experience replay
        self.memory = PrioritizedReplayBuffer(MAX_MEMORY)

        # Create model with improved architecture
        self.model = Linear_QNet(6, 256, 2)  # 6 state values, 256 hidden neurons, 2 outputs (don't flap, flap)

        # Try to load existing model
        if not self.model.load():
            print("No existing model found, starting fresh")

        # Create trainer with target network
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # Get the nearest pipe
        pipe_x = game.top_pipes[-1].x
        top_pipe_y = game.top_pipes[-1].y
        bottom_pipe_y = game.bottom_pipes[-1].y

        # Calculate normalized distances (better than boolean values)
        # Vertical distance to the center of the pipe gap
        pipe_center_y = (top_pipe_y + bottom_pipe_y) / 2
        vertical_distance = (game.bird.y - pipe_center_y) / game.h

        # Horizontal distance to the pipe
        horizontal_distance = (pipe_x - game.bird.x) / game.w

        # Bird's velocity (normalized)
        bird_velocity = game.bird_vel / 20  # Normalize by max velocity

        # Gap size (normalized)
        gap_size = (bottom_pipe_y - top_pipe_y) / game.h

        # Next pipe information if available
        if len(game.top_pipes) > 1:
            next_pipe_x = game.top_pipes[-2].x
            next_pipe_center_y = (game.top_pipes[-2].y + game.bottom_pipes[-2].y) / 2
            next_horizontal_distance = (next_pipe_x - game.bird.x) / game.w
            next_vertical_distance = (game.bird.y - next_pipe_center_y) / game.h
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

    def remember(self, state, action, reward, next_state, done, priority=None):
        """Store experience in prioritised replay buffer"""
        self.memory.add(state, action, reward, next_state, done, priority)

    def train_long_memory(self):
        """Train on a batch from prioritised experience replay"""
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough samples to train

        # Sample batch with priorities
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(BATCH_SIZE)

        # Train and get TD errors for priority update
        td_errors = self.trainer.train_step(states, actions, rewards, next_states, dones, weights)

        # Update priorities based on TD errors
        self.memory.update_priorities(indices, td_errors)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on a single experience and get its priority for the replay buffer"""
        # Convert to tensors for single sample
        state_t = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0)
        next_state_t = torch.tensor(np.array(next_state), dtype=torch.float).unsqueeze(0)
        action_t = torch.tensor(np.array(action), dtype=torch.long).unsqueeze(0)
        reward_t = torch.tensor(np.array(reward), dtype=torch.float).unsqueeze(0)

        # Train and get TD error for priority
        td_errors = self.trainer.train_step(state_t, action_t, reward_t, next_state_t, (done,))

        # Return the TD error as priority
        return td_errors[0] if td_errors else 1.0

    def get_action(self, state):
        """Get action using epsilon-greedy policy with exponential decay"""
        # Calculate epsilon with exponential decay
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1. * self.n_games / self.epsilon_decay)

        final_move = [0, 0]  # [don't flap, flap]

        # Exploration: random action
        if random.random() < epsilon:
            move = random.randint(0, 1)
            final_move[move] = 1
        # Exploitation: best action from model
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            with torch.no_grad():  # No need to track gradients for inference
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def enhance_reward(reward, state_old, state_new, done, score):
    """Enhance the reward signal to provide more learning guidance"""
    enhanced_reward = reward

    if not done:
        # Reward for maintaining a good position (near the center of the pipe gap)
        vertical_distance_old = abs(state_old[0])  # Distance from pipe center
        vertical_distance_new = abs(state_new[0])  # Distance from pipe center

        # Reward improvement in position
        if vertical_distance_new < vertical_distance_old:
            enhanced_reward += 1  # Reward for getting closer to the center

        # Penalize extreme positions
        if abs(state_new[0]) > 0.4:  # If far from centre
            enhanced_reward -= 0.5

        # Reward for maintaining stable flight
        if abs(state_new[2]) < 0.3:  # If velocity is moderate
            enhanced_reward += 0.5

        # Extra reward for approaching a pipe successfully
        if 0 < state_new[1] < 0.2:  # Close to passing a pipe
            enhanced_reward += 2

    # Larger penalty for dying early
    if done and score < 5:
        enhanced_reward -= 10

    # Larger reward for achieving high scores
    if score > 10:
        enhanced_reward += score * 0.5

    return enhanced_reward


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = FlappyGame()

    # Training stats
    frame_iteration = 0
    max_memory_size = 0

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        # Get reward from game
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Enhance reward for better learning signals
        enhanced_reward = enhance_reward(reward, state_old, state_new, done, score)

        # Train short memory and get TD error for priority
        td_error = agent.train_short_memory(state_old, final_move, enhanced_reward, state_new, done)

        # Remember experience with priority
        agent.remember(state_old, final_move, enhanced_reward, state_new, done, td_error)

        frame_iteration += 1

        # Update max memory size for tracking
        max_memory_size = max(max_memory_size, len(agent.memory))

        if done:
            agent.n_games += 1

            # Train on batch from replay memory
            agent.train_long_memory()

            # Save model if score improves
            if score > record:
                record = score
                agent.model.save()
                print(f"New record! {record} - Saving model...")

            # Print detailed stats
            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}, ' +
                  f'Frames: {frame_iteration}, Memory: {len(agent.memory)}/{MAX_MEMORY}')

            # Reset for next game
            frame_iteration = 0

            # Update plots
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Reset game
            game.reset()


if __name__ == "__main__":
    train()
