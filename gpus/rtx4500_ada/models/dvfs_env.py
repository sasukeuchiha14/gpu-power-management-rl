import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

import os

def _resolve_path(path):
    if os.path.exists(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
    cand = os.path.join(gpu_root, "data", os.path.basename(path))
    if os.path.exists(cand):
        return cand
    return path

class DVFSEnv(gym.Env):
    """Custom Environment for Dynamic Voltage and Frequency Scaling (DVFS)"""
    
    def __init__(self, dataset_path="preprocessed_dataset.csv"):
        super(DVFSEnv, self).__init__()
        
        # 1. Load the Preprocessed Dataset
        resolved_path = _resolve_path(dataset_path)
        self.df = pd.read_csv(resolved_path)
        
        # Identify state columns (Everything except EDP and Reward)
        self.state_cols = [col for col in self.df.columns if col not in ['EDP', 'Reward']]
        
        # Get unique workloads to simulate episodes
        # We find columns that start with 'Workload_Name_'
        self.workload_cols = [col for col in self.df.columns if col.startswith('Workload_Name_')]
        
        # 2. Define the Action Space
        # 0: Decrease Core Freq, 1: Increase Core Freq
        # 2: Decrease Mem Freq,  3: Increase Mem Freq
        # 4: Do Nothing (Maintain current clocks)
        self.action_space = spaces.Discrete(5)
        
        # 3. Define the Observation (State) Space
        # The state is a 1D array of our features (Core_F, Mem_F, Util, Temp, One-Hots)
        num_features = len(self.state_cols)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_features,), dtype=np.float32)
        
        # Environment variables
        self.current_step = 0
        self.max_steps = 100 # How many actions the agent takes per workload episode
        self.current_workload_df = None
        self.current_state_row = None

    def reset(self, seed=None, options=None):
        """Starts a new episode by picking a random workload from the dataset."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Pick a random workload column to activate
        random_workload = np.random.choice(self.workload_cols)
        
        # Filter the dataset to only include rows for this specific workload
        self.current_workload_df = self.df[self.df[random_workload] == 1.0].copy()
        
        # Start at a random frequency state within this workload
        self.current_state_row = self.current_workload_df.sample(1).iloc[0]
        
        return self._get_observation(), {}

    def step(self, action):
        """The agent takes an action to change frequencies."""
        self.current_step += 1
        
        # Extract current frequencies (scaled 0 to 1)
        curr_core = self.current_state_row['Core_F']
        curr_mem = self.current_state_row['Mem_F']
        
        # Step sizes based on your Min-Max scaling 
        # (40 core steps = ~0.025 per step, 10 mem steps = ~0.111 per step)
        core_step_size = 1.0 / 39.0 
        mem_step_size = 1.0 / 9.0   
        
        # Apply Action
        if action == 0:   curr_core = max(0.0, curr_core - core_step_size)
        elif action == 1: curr_core = min(1.0, curr_core + core_step_size)
        elif action == 2: curr_mem = max(0.0, curr_mem - mem_step_size)
        elif action == 3: curr_mem = min(1.0, curr_mem + mem_step_size)
        # action == 4 does nothing
        
        # Find the row in our dataset that most closely matches the new frequencies
        # We calculate Euclidean distance to find the nearest valid hardware state
        distances = np.sqrt(
            (self.current_workload_df['Core_F'] - curr_core)**2 + 
            (self.current_workload_df['Mem_F'] - curr_mem)**2
        )
        best_match_idx = distances.idxmin()
        self.current_state_row = self.current_workload_df.loc[best_match_idx]
        
        # Get new state and reward
        observation = self._get_observation()
        reward = self.current_state_row['Reward']
        
        # Check if episode is done
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return observation, reward, terminated, truncated, {}

    def _get_observation(self):
        """Returns the current state as a numpy array."""
        return self.current_state_row[self.state_cols].values.astype(np.float32)

# ==========================================
# QUICK TEST TO ENSURE IT WORKS
# ==========================================
if __name__ == "__main__":
    env = DVFSEnv()
    obs, info = env.reset()
    print("Environment Initialized!")
    print(f"Initial State Shape: {obs.shape}")
    
    print("\nTaking 5 random actions:")
    for i in range(5):
        action = env.action_space.sample() # Pick a random action (0-4)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1} | Action: {action} | Reward (1-EDP): {reward:.4f}")
