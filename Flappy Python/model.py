import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import random
from collections import namedtuple, deque


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Deeper network with two hidden layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, output_size)

        # Initialize weights using He initialization for better gradient flow
        nn.init.kaiming_uniform_(self.linear1.weight)
        nn.init.kaiming_uniform_(self.linear2.weight)
        nn.init.kaiming_uniform_(self.linear3.weight)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            return True
        return False

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer to store and sample experiences based on their importance."""

    Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = no prioritization, 1 = full prioritization)
        self.beta = beta    # Importance sampling weight (0 = no correction, 1 = full correction)
        self.beta_increment = beta_increment  # Beta will be incremented by this amount every time we sample
        self.epsilon = epsilon  # Small constant to avoid zero priority
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self._next_idx = 0

    def add(self, state, action, reward, next_state, done, priority=None):
        """Add a new experience to the buffer with its priority."""
        if priority is None:
            # If no priority given, set it to max priority to encourage exploration of new experiences
            priority = max(self.priorities) if self.priorities else 1.0

        experience = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        """Sample a batch of experiences based on their priorities."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Increment beta for next time
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        states = torch.tensor(np.array([e.state for e in experiences]), dtype=torch.float)
        actions = torch.tensor(np.array([e.action for e in experiences]), dtype=torch.long)
        rewards = torch.tensor(np.array([e.reward for e in experiences]), dtype=torch.float)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]), dtype=torch.float)
        dones = tuple(e.done for e in experiences)
        weights = torch.tensor(weights, dtype=torch.float)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities for experiences at given indices."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self):
        return len(self.buffer)


class QTrainer:
    def __init__(self, model, lr, gamma, tau=0.005):
        self.lr = lr
        self.gamma = gamma
        self.tau = tau  # Target network update rate

        # Online network (the one being trained)
        self.model = model

        # Target network (for stable Q-value estimation)
        self.target_model = Linear_QNet(
            model.linear1.in_features,
            model.linear1.out_features,
            model.linear3.out_features
        )
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()  # Set to evaluation mode

        # Optimizer with weight decay for regularization
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        # Huber loss is more robust to outliers than MSE
        self.criterion = nn.SmoothL1Loss()

    def update_target_network(self):
        """Soft update target network: θ_target = τ*θ_online + (1-τ)*θ_target"""
        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def train_step(self, state, action, reward, next_state, done, weights=None):
        """Train the model on a batch of experiences."""
        # Convert to tensors if not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array(state), dtype=torch.float)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(np.array(action), dtype=torch.long)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(np.array(reward), dtype=torch.float)

        # Add batch dimension if needed
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            if weights is not None:
                weights = torch.unsqueeze(weights, 0)

        # Get current Q values
        pred = self.model(state)

        # Create target Q values
        target = pred.clone()

        # Calculate TD errors for prioritized replay
        td_errors = []

        for idx in range(len(done)):
            # Get action index
            action_idx = torch.argmax(action[idx]).item()

            # Calculate new Q value
            if not done[idx]:
                # Double Q-learning: use online network to select action, target network to evaluate
                with torch.no_grad():
                    best_action = self.model(next_state[idx]).argmax().item()
                    next_q = self.target_model(next_state[idx])[best_action]
                    Q_new = reward[idx] + self.gamma * next_q
            else:
                Q_new = reward[idx]

            # Calculate TD error for prioritized replay
            td_error = abs(Q_new - pred[idx][action_idx]).item()
            td_errors.append(td_error)

            # Update target
            target[idx][action_idx] = Q_new

        # Apply importance sampling weights if provided
        if weights is not None:
            # Element-wise multiplication of loss with weights
            loss = self.criterion(target * weights.unsqueeze(1), pred * weights.unsqueeze(1))
        else:
            loss = self.criterion(target, pred)

        # Optimize
        self.optimiser.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimiser.step()

        # Update target network
        self.update_target_network()

        return td_errors
