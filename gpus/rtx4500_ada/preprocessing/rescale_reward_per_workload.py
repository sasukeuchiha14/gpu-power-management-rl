"""
rescale_reward_per_workload.py
Fixes the flat-signal problem WITHOUT needing the original filtered_dataset.csv.

The current preprocessed_dataset.csv still keeps the raw 'EDP' column and the
Workload_Name_* one-hot columns, so we can recompute the reward correctly:
scale EDP WITHIN each workload (best row -> reward 1.0, worst -> 0.0). This gives
every workload a full 0..1 reward range, so the agent gets a real gradient on
light workloads (lud, myocyte, etc.) instead of a near-flat reward.

Backs up the original as preprocessed_dataset_global.csv, then overwrites
preprocessed_dataset.csv in place. Re-run train_ddqn.py afterwards.
"""
import os
import shutil
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(gpu_root, "data")

SRC = os.path.join(data_dir, "preprocessed_dataset.csv")
BACKUP = os.path.join(data_dir, "preprocessed_dataset_global.csv")

def main():
    if not os.path.exists(SRC):
        if os.path.exists("preprocessed_dataset.csv"):
            SRC_path = "preprocessed_dataset.csv"
        else:
            raise FileNotFoundError(f"{SRC} not found. Please run preprocess.py first.")
    else:
        SRC_path = SRC

    df = pd.read_csv(SRC_path)
    if "EDP" not in df.columns:
        raise SystemExit("EDP column missing; cannot rescale. Re-run preprocess.py.")

    # Guard: never overwrite an existing backup (avoids destroying the true
    # original if this script is run twice).
    if not os.path.exists(BACKUP):
        shutil.copyfile(SRC_path, BACKUP)
        print(f"Backed up original global-scaled reward to {BACKUP}")
    else:
        print(f"Backup {BACKUP} already exists; leaving it untouched.")

    wl_cols = [c for c in df.columns if c.startswith("Workload_Name_")]
    # Recover the workload label for each row from the one-hot columns.
    df["_wl"] = df[wl_cols].idxmax(axis=1)

    def scale_edp(e):
        span = e.max() - e.min()
        if span <= 0:                       # constant EDP -> scaled 0 -> reward 1.0
            return pd.Series(0.0, index=e.index)
        return (e - e.min()) / span

    df["Reward"] = 1.0 - df.groupby("_wl")["EDP"].transform(scale_edp)
    df = df.drop(columns=["_wl"])
    df.to_csv(SRC_path, index=False)
    print(f"Rewrote {SRC_path} with per-workload reward.\n")

    # verify the signal is no longer flat
    df["sEDP"] = 1.0 - df["Reward"]
    print(f"{'workload':<15}{'reward_min':>12}{'reward_max':>12}")
    for c in wl_cols:
        s = df[df[c] == 1.0]["Reward"]
        print(f"{c.replace('Workload_Name_',''):<15}{s.min():>12.3f}{s.max():>12.3f}")

if __name__ == "__main__":
    main()
