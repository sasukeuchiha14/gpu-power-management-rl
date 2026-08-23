"""
UnifiedDVFSEnv -- trains across MULTIPLE GPUs' profiled datasets at once.

This is the piece the "one model, any card" claim in NORTHSTAR.md depends on.
Before this file existed, nothing in the repo could train on more than one
GPU's data -- gpus/<card>/models/train_ddqn.py loads exactly one
preprocessed_dataset.csv, always. Rent or borrow ten cards and there was
still no way to combine them into one training run.

Each episode here samples a random (GPU, workload) pair from whatever
datasets are loaded, so the policy has to learn a rule that holds across all
of them, not one card's quirks. Uses the same 6-feature state
(common/features.py) and 3-action core-only control as
gpus/rtx3050ti_laptop/models/dvfs_env.py -- this is deliberately that same
environment's logic, generalised to draw episodes from N datasets instead
of one, with a real leave-one-GPU-out mechanism for testing generalisation.
"""
import glob
import os

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from features import FEATURES, STACK_SIZE, ACTION_DIM


def discover_datasets(repo_root=None):
    """Find every gpus/*/data/preprocessed_dataset.csv that matches the
    6-feature schema. Anything that doesn't (e.g. the RTX 4500 tree's older
    workload-one-hot format) is silently skipped -- it isn't compatible with
    this state design and can't be combined into unified training.

    Returns {gpu_tag: DataFrame}, where gpu_tag is the gpus/<tag>/ folder name.
    """
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pattern = os.path.join(repo_root, "gpus", "*", "data", "preprocessed_dataset.csv")

    found = {}
    for path in sorted(glob.glob(pattern)):
        try:
            header = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        if not set(FEATURES).issubset(header.columns):
            continue
        gpu_tag = os.path.basename(os.path.dirname(os.path.dirname(path)))
        found[gpu_tag] = pd.read_csv(path)
    return found


def _infer_core_stride(workload_df):
    """Same logic as gpus/rtx3050ti_laptop/models/dvfs_env.py -- the median
    gap between adjacent profiled core_frac levels for THIS workload, so the
    step size is automatically correct regardless of which card it came
    from. Not imported from there directly to avoid a cross-tree import;
    kept in sync deliberately, it's four lines."""
    levels = np.sort(workload_df["core_frac"].unique())
    if len(levels) < 2:
        return 0.1
    return float(np.median(np.diff(levels)))


class UnifiedDVFSEnv(gym.Env):
    """Core-only DVFS environment trained across multiple GPUs' data at once."""

    def __init__(self, datasets=None, holdout_gpu=None, stack_size=STACK_SIZE):
        super().__init__()
        if datasets is None:
            datasets = discover_datasets()
        if holdout_gpu:
            datasets = {k: v for k, v in datasets.items() if k != holdout_gpu}
        if not datasets:
            raise ValueError(
                "No 6-feature datasets available to train on. Either none were "
                "discovered under gpus/*/data/preprocessed_dataset.csv, or "
                "--holdout excluded the only one present. Run that tree's "
                "preprocess.py first, or pass `datasets=` explicitly."
            )

        frames = []
        for gpu_tag, df in datasets.items():
            df = df.copy()
            df["_gpu_source"] = gpu_tag  # grouping key only -- never fed to the network
            frames.append(df)
        self.df = pd.concat(frames, ignore_index=True)
        self.gpu_tags = sorted(datasets.keys())
        self.held_out = holdout_gpu

        self.episode_keys = list(
            self.df.groupby(["_gpu_source", "Workload_Name"]).groups.keys()
        )

        # 0 = decrease core, 1 = increase core, 2 = hold (matches common/features.py)
        self.action_space = spaces.Discrete(ACTION_DIM)
        self.num_features = len(FEATURES)
        self.stack_size = stack_size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.stack_size, self.num_features),
            dtype=np.float32,
        )

        self.max_steps = 50
        self.current_step = 0
        self.frames = deque(maxlen=self.stack_size)
        self.current_workload_df = None
        self.current_state_row = None
        self.current_gpu_tag = None
        self.current_tracked_core = 1.0
        self.core_stride = 0.1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.frames.clear()

        idx = np.random.randint(len(self.episode_keys))
        gpu_tag, workload = self.episode_keys[idx]
        self.current_gpu_tag = gpu_tag
        self.current_workload_df = self.df[
            (self.df["_gpu_source"] == gpu_tag) & (self.df["Workload_Name"] == workload)
        ].copy()

        self.current_state_row = self.current_workload_df.sample(1).iloc[0]
        self.current_tracked_core = float(self.current_state_row["core_frac"])
        self.core_stride = _infer_core_stride(self.current_workload_df)

        single_obs = self._get_single_observation()
        for _ in range(self.stack_size):
            self.frames.append(single_obs)

        return self._get_stacked_observation(), {"gpu_source": gpu_tag, "workload": workload}

    def step(self, action):
        self.current_step += 1
        stride = self.core_stride

        if action == 0:
            self.current_tracked_core = max(0.0, self.current_tracked_core - stride)
        elif action == 1:
            self.current_tracked_core = min(1.0, self.current_tracked_core + stride)
        # action == 2: hold

        distances = np.abs(self.current_workload_df["core_frac"] - self.current_tracked_core)
        self.current_state_row = self.current_workload_df.loc[distances.idxmin()]

        single_obs = self._get_single_observation()
        noise = np.random.normal(0, 0.01, size=single_obs.shape).astype(np.float32)
        self.frames.append(np.clip(single_obs + noise, 0.0, 1.0))

        reward = float(self.current_state_row["Reward"])
        terminated = False
        truncated = self.current_step >= self.max_steps
        return self._get_stacked_observation(), reward, terminated, truncated, {}

    def _get_single_observation(self):
        return self.current_state_row[FEATURES].values.astype(np.float32)

    def _get_stacked_observation(self):
        return np.array(self.frames, dtype=np.float32)


if __name__ == "__main__":
    found = discover_datasets()
    print(f"Discovered {len(found)} compatible GPU dataset(s): {sorted(found.keys())}")
    if found:
        env = UnifiedDVFSEnv(datasets=found)
        state, info = env.reset()
        print(f"Env OK. State shape {state.shape}, first episode: {info}")
