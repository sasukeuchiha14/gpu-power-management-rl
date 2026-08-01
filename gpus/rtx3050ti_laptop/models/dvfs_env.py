import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
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

class LaptopDVFSEnv(gym.Env):
    """Gymnasium Environment for Core-only DVFS Scaling on RTX 3050 Ti Laptop"""
    
    def __init__(self, dataset_path="preprocessed_dataset.csv", stack_size=3):
        super(LaptopDVFSEnv, self).__init__()
        
        resolved_path = _resolve_path(dataset_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Processed dataset {dataset_path} not found. Please run preprocess.py first.")
            
        self.df = pd.read_csv(resolved_path)
        self.state_cols = [col for col in self.df.columns if col not in ['EDP', 'Reward']]
        self.workload_cols = [col for col in self.df.columns if col.startswith('Workload_Name_')]
        
        # Action space: Discrete(3) -> 0: Decrease Core, 1: Increase Core, 2: Hold
        self.action_space = spaces.Discrete(3)
        
        self.num_features = len(self.state_cols)
        self.stack_size = stack_size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.stack_size, self.num_features), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 100 
        self.frames = deque(maxlen=self.stack_size)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        random_workload = np.random.choice(self.workload_cols)
        self.current_workload_df = self.df[self.df[random_workload] == 1.0].copy()
        self.current_state_row = self.current_workload_df.sample(1).iloc[0]

        self.current_tracked_core = self.current_state_row['Core_F']
        
        state = self._get_single_observation()
        for _ in range(self.stack_size):
            self.frames.append(state)
            
        return self._get_stacked_observation(), {}

    def step(self, action):
        self.current_step += 1
        
        curr_core = self.current_tracked_core
        core_step_size = 0.10 # 10% frequency jump (coarse steps)
        
        if action == 0:
            curr_core = max(0.0, curr_core - core_step_size)
        elif action == 1:
            curr_core = min(1.0, curr_core + core_step_size)
        # action 2 is hold
        
        self.current_tracked_core = curr_core
        
        # Calculate distance to snap to the nearest profiled Core frequency
        distances = np.abs(self.current_workload_df['Core_F'] - curr_core)
        self.current_state_row = self.current_workload_df.loc[distances.idxmin()]
        
        self.frames.append(self._get_single_observation())
        
        reward = self.current_state_row['Reward']
        if action != 2: # Penalize unnecessary frequency shifts (action 2 = hold)
            reward -= 0.02

        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return self._get_stacked_observation(), reward, terminated, truncated, {}

    def _get_single_observation(self):
        raw_state = self.current_state_row[self.state_cols].values.astype(np.float32)
        
        # Domain Randomization: Inject +/- 1% artificial noise to simulate live sensors
        noise = np.random.normal(0, 0.01, size=raw_state.shape)
        noisy_state = np.clip(raw_state + noise, 0.0, 1.0)
        
        return noisy_state.astype(np.float32)

    def _get_stacked_observation(self):
        return np.array(self.frames, dtype=np.float32)

if __name__ == "__main__":
    env = LaptopDVFSEnv(dataset_path="preprocessed_dataset.csv")
    obs, info = env.reset()
    print("Environment Initialized!")
    print(f"Initial State Shape: {obs.shape}")
    print(f"Action Space Size: {env.action_space.n}")
