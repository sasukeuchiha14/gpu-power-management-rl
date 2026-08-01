import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

import os

ACTION_COST = 0.02

def _resolve_path(path):
    if os.path.exists(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
    cand = os.path.join(gpu_root, "data", os.path.basename(path))
    if os.path.exists(cand):
        return cand
    return path

import json

class AdvancedDVFSEnv(gym.Env):
    """Advanced Environment with Frame Stacking, Workload Randomization, and Noise"""
    
    def __init__(self, dataset_path="preprocessed_dataset.csv", stack_size=3):
        super(AdvancedDVFSEnv, self).__init__()
        
        resolved_path = _resolve_path(dataset_path)
        self.df = pd.read_csv(resolved_path)
        
        # Check if scaler_params.json defines the exact feature ordering for trained weights
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
        scaler_json = os.path.join(gpu_root, "data", "scaler_params.json")
        if not os.path.exists(scaler_json) and os.path.exists("scaler_params.json"):
            scaler_json = "scaler_params.json"

        if os.path.exists(scaler_json):
            with open(scaler_json) as f:
                sc = json.load(f)
            if sc.get("state_col_order"):
                self.state_cols = sc["state_col_order"]
                for c in self.state_cols:
                    if c not in self.df.columns:
                        self.df[c] = 0.0
            else:
                self.state_cols = [col for col in self.df.columns if col not in ['EDP', 'Reward']]
        else:
            self.state_cols = [col for col in self.df.columns if col not in ['EDP', 'Reward']]

        self.workload_cols = [col for col in self.df.columns if col.startswith('Workload_Name_')]
        
        self.action_space = spaces.Discrete(5)
        
        # FRAME STACKING: The state is now a 2D matrix (stack_size, num_features)
        self.num_features = len(self.state_cols)
        self.stack_size = stack_size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.stack_size, self.num_features), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 100 
        self.frames = deque(maxlen=self.stack_size) # Holds our short-term memory
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        random_workload = np.random.choice(self.workload_cols)
        self.current_workload_df = self.df[self.df[random_workload] == 1.0].copy()
        self.current_state_row = self.current_workload_df.sample(1).iloc[0]

        # --- Initialize tracking variables ---
        self.current_tracked_core = self.current_state_row['Core_F']
        self.current_tracked_mem = self.current_state_row['Mem_F']
        
        # Fill the initial frame stack with copies of the starting state
        state = self._get_single_observation()
        for _ in range(self.stack_size):
            self.frames.append(state)
            
        return self._get_stacked_observation(), {}

    def step(self, action):
        self.current_step += 1
        
        # --- Use tracked frequencies and LARGER step sizes ---
        curr_core = self.current_tracked_core
        curr_mem = self.current_tracked_mem
        
        # 10% and 15% jumps to guarantee we hit a new row in the CSV
        core_step_size = 0.10 
        mem_step_size = 0.15   
        
        if action == 0:   curr_core = max(0.0, curr_core - core_step_size)
        elif action == 1: curr_core = min(1.0, curr_core + core_step_size)
        elif action == 2: curr_mem = max(0.0, curr_mem - mem_step_size)
        elif action == 3: curr_mem = min(1.0, curr_mem + mem_step_size)
        
        # Save the new tracked position for the next step!
        self.current_tracked_core = curr_core
        self.current_tracked_mem = curr_mem
        
        # Calculate Euclidean distance to snap to reality
        distances = np.sqrt(
            (self.current_workload_df['Core_F'] - curr_core)**2 + 
            (self.current_workload_df['Mem_F'] - curr_mem)**2
        )
        self.current_state_row = self.current_workload_df.loc[distances.idxmin()]
        
        # Add the new physical state to our short-term memory
        self.frames.append(self._get_single_observation())
        
        reward = self.current_state_row['Reward']
        if action != 4:                    # penalize clock changes (action 4 = hold)
            reward -= ACTION_COST

        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return self._get_stacked_observation(), reward, terminated, truncated, {}

    def _get_single_observation(self):
        """Returns current state with injected physical noise (Domain Randomization)"""
        raw_state = self.current_state_row[self.state_cols].values.astype(np.float32)
        
        # DOMAIN RANDOMIZATION: Inject +/- 1% artificial noise to simulate live hardware
        noise = np.random.normal(0, 0.01, size=raw_state.shape)
        noisy_state = np.clip(raw_state + noise, 0.0, 1.0)
        
        return noisy_state.astype(np.float32)

    def _get_stacked_observation(self):
        return np.array(self.frames, dtype=np.float32)

# ==========================================
# QUICK TEST TO ENSURE IT WORKS
# ==========================================
if __name__ == "__main__":
    env = AdvancedDVFSEnv(dataset_path="preprocessed_dataset.csv")
    obs, info = env.reset()
    print("Environment Initialized!")
    print(f"Initial State Shape: {obs.shape}")
    
    print("\nTaking 5 random actions:")
    for i in range(5):
        action = env.action_space.sample() # Pick a random action (0-4)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1} | Action: {action} | Reward (1-EDP): {reward:.4f}")
