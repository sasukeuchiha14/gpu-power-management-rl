import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from dvfs_env import DVFSEnv  # Importing your custom environment
import os

# ==========================================
# 1. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # A simple but powerful Feed-Forward Network
        self.fc1 = nn.Linear(state_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x) # Outputs Q-values for all 5 actions

# ==========================================
# 2. THE RL AGENT
# ==========================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95           # Discount factor (future rewards)
        self.epsilon = 1.0          # Exploration rate (start at 100% random)
        self.epsilon_min = 0.05     # Minimum exploration rate
        self.epsilon_decay = 0.995  # How fast it stops exploring
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Memory Buffer
        self.memory = deque(maxlen=10000)
        
        # The Brain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in memory for replay."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Chooses an action based on Epsilon-Greedy strategy."""
        # Random Exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation (Ask the Neural Network)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()

    def replay(self):
        """Trains the network using a random batch of past experiences."""
        if len(self.memory) < self.batch_size:
            return # Wait until we have enough memories
        
        # Sample a random batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            reward_tensor = torch.FloatTensor([reward]).to(self.device)
            
            # Bellman Equation: Target = Reward + Gamma * Max(Next State Q-Value)
            target = reward_tensor
            if not done:
                target = reward_tensor + self.gamma * torch.max(self.model(next_state_tensor)[0])
            
            # Current Q-Values
            current_q_values = self.model(state_tensor)
            target_q_values = current_q_values.clone()
            target_q_values[0][action] = target
            
            # Optimize Weights
            self.optimizer.zero_grad()
            loss = self.criterion(current_q_values, target_q_values)
            loss.backward()
            self.optimizer.step()
            
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    env = DVFSEnv(dataset_path="preprocessed_dataset.csv")
    state_size = env.observation_space.shape[0] # 23
    action_size = env.action_space.n            # 5
    
    agent = DQNAgent(state_size, action_size)
    episodes = 200 # Total workloads to run through
    
    print(f"Starting Training on {agent.device}...")
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        # Max 100 steps per episode (defined in your env)
        for time_step in range(100): 
            # 1. Agent chooses an action
            action = agent.act(state)
            
            # 2. Environment reacts
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # 3. Agent remembers the outcome
            agent.remember(state, action, reward, next_state, done)
            
            # 4. Agent learns from past memories
            agent.replay()
            
            state = next_state
            if done:
                print(f"Episode: {e+1}/{episodes} | Total Reward: {total_reward:.2f} | Exploration Rate: {agent.epsilon:.2f}")
                break

    print("\\nTraining Complete! The Agent is now ready for evaluation.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
    weights_dir = os.path.join(gpu_root, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    out_weights = os.path.join(weights_dir, "dvfs_dqn_weights.pth")
    torch.save(agent.model.state_dict(), out_weights)
    print(f"Model weights saved to {out_weights}")
