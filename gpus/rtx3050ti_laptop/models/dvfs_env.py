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
    """
    Universal 6-Feature DVFS Gym Environment for RTX 3050 Ti Laptop GPU.
    State: 6 device-normalised features x 3 frame stack = (3, 6) matrix.
    Actions: 3 discrete choices (0: Dec Core, 1: Inc Core, 2: Hold).
    """
    def __init__(self, dataset_path="preprocessed_dataset.csv", stack_size=3):
        super(LaptopDVFSEnv, self).__init__()
        
        resolved_path = _resolve_path(dataset_path)
        self.df = pd.read_csv(resolved_path)
        
        # 6 Universal Device-Normalised Features (Fixed across all GPUs)
        self.state_cols = [
            'core_frac',
            'sm_util',
            'mem_util',
            'power_frac',
            'temp_frac',
            'vram_frac'
        ]
        # Ensure any missing feature exists in dataframe
        for col in self.state_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0

        self.workloads = self.df['Workload_Name'].unique().tolist()
        
        # Action space: 0 = Dec Core, 1 = Inc Core, 2 = Hold
        self.action_space = spaces.Discrete(3)
        
        # Frame Stacking: (stack_size, 6 features)
        self.num_features = len(self.state_cols)
        self.stack_size = stack_size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.stack_size, self.num_features), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 50
        self.frames = deque(maxlen=self.stack_size)
        self.current_workload_df = None
        self.current_state_row = None
        self.current_tracked_core = 1.0 # Current core fraction

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.frames.clear()
        
        # Pick a random workload for this training episode
        random_workload = np.random.choice(self.workloads)
        self.current_workload_df = self.df[self.df['Workload_Name'] == random_workload].copy()
        
        # Start at a random frequency state within this workload
        self.current_state_row = self.current_workload_df.sample(1).iloc[0]
        self.current_tracked_core = float(self.current_state_row['core_frac'])
        
        # Pre-fill frame stack
        single_obs = self._get_single_observation()
        for _ in range(self.stack_size):
            self.frames.append(single_obs)
            
        return self._get_stacked_observation(), {}

    def step(self, action):
        self.current_step += 1
        
        # Core clock delta stride (~100 MHz step in normalised scale)
        # 100 MHz / 1400 MHz range ≈ 0.0714
        stride = 100.0 / 1400.0
        
        if action == 0:   # Decrease Core Clock
            self.current_tracked_core = max(0.0, self.current_tracked_core - stride)
        elif action == 1: # Increase Core Clock
            self.current_tracked_core = min(1.0, self.current_tracked_core + stride)
        # action == 2: Hold (no change)

        # Nearest-neighbour lookup for the closest physical state in this workload table
        distances = np.abs(self.current_workload_df['core_frac'] - self.current_tracked_core)
        best_match_idx = distances.idxmin()
        self.current_state_row = self.current_workload_df.loc[best_match_idx]
        
        # Update frame stack with new observation + small telemetry sensor noise
        single_obs = self._get_single_observation()
        noise = np.random.normal(0, 0.01, size=single_obs.shape).astype(np.float32)
        noisy_obs = np.clip(single_obs + noise, 0.0, 1.0)
        self.frames.append(noisy_obs)
        
        reward = float(self.current_state_row['Reward'])
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return self._get_stacked_observation(), reward, terminated, truncated, {}

    def _get_single_observation(self):
        return self.current_state_row[self.state_cols].values.astype(np.float32)

    def _get_stacked_observation(self):
        return np.array(self.frames, dtype=np.float32)
