"""
aggregate_seed_sweep.py -- aggregates a multi-seed leave-one-GPU-out sweep
into per-GPU mean +/- std, closing part of NORTHSTAR.md row 9 (no seed sweep,
no variance/error bars) for row 4's result specifically.

Expects, per GPU, one holdout_results/<gpu>.csv (the original seed-42 run,
unsuffixed) plus any holdout_results/<gpu>_trainseed<N>.csv files (additional
training seeds, produced with a fixed eval seed so only training stochasticity
varies -- see evaluate_holdout.py's --seed docstring).

    python3 aggregate_seed_sweep.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

_common_dir = os.path.dirname(os.path.abspath(__file__))
_results_dir = os.path.join(_common_dir, "holdout_results")

GPUS = [
    "nvidia_geforce_rtx_3060", "nvidia_geforce_rtx_3070", "nvidia_geforce_rtx_3080",
    "nvidia_geforce_rtx_3090", "nvidia_geforce_rtx_4070", "nvidia_geforce_rtx_4070_ti",
    "nvidia_geforce_rtx_4090", "rtx3050ti_laptop",
]


def _reduction_pct(df):
    mean_gov = df.governor_edp.mean()
    mean_ai = df.ai_edp.mean()
    return 100.0 * (mean_gov - mean_ai) / mean_gov if mean_gov > 0 else float("nan")


def aggregate():
    per_gpu_seed_rows = []
    for gpu in GPUS:
        base = os.path.join(_results_dir, f"{gpu}.csv")
        if os.path.exists(base):
            df = pd.read_csv(base)
            per_gpu_seed_rows.append({"gpu": gpu, "train_seed": 42, "reduction_pct": _reduction_pct(df),
                                       "n_wins": int(df.ai_beats_governor.sum()), "n_workloads": len(df)})
        for path in sorted(glob.glob(os.path.join(_results_dir, f"{gpu}_trainseed*.csv"))):
            m = re.search(r"_trainseed(\d+)\.csv$", path)
            seed = int(m.group(1)) if m else -1
            df = pd.read_csv(path)
            per_gpu_seed_rows.append({"gpu": gpu, "train_seed": seed, "reduction_pct": _reduction_pct(df),
                                       "n_wins": int(df.ai_beats_governor.sum()), "n_workloads": len(df)})

    detail = pd.DataFrame(per_gpu_seed_rows)
    if detail.empty:
        raise SystemExit(f"No result files found in {_results_dir}.")

    print("Per (GPU, seed) results:")
    print(detail.sort_values(["gpu", "train_seed"]).to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    agg = detail.groupby("gpu").agg(
        n_seeds=("train_seed", "count"),
        mean_reduction_pct=("reduction_pct", "mean"),
        std_reduction_pct=("reduction_pct", "std"),
        min_reduction_pct=("reduction_pct", "min"),
        max_reduction_pct=("reduction_pct", "max"),
        seeds_ai_wins=("reduction_pct", lambda s: int((s > 0).sum())),
    ).reset_index().sort_values("gpu")

    print("\nPer-GPU aggregate across seeds (positive reduction_pct = AI beats governor):")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    consistent_wins = agg[(agg.seeds_ai_wins == agg.n_seeds) & (agg.n_seeds > 1)]
    consistent_losses = agg[(agg.seeds_ai_wins == 0) & (agg.n_seeds > 1)]
    flips = agg[(agg.seeds_ai_wins > 0) & (agg.seeds_ai_wins < agg.n_seeds)]
    print(f"\nGPUs where AI won on EVERY seed tested: {list(consistent_wins.gpu)}")
    print(f"GPUs where AI lost on EVERY seed tested: {list(consistent_losses.gpu)}")
    print(f"GPUs where the result FLIPPED depending on seed: {list(flips.gpu)}")

    detail_out = os.path.join(_results_dir, "seed_sweep_detail.csv")
    agg_out = os.path.join(_results_dir, "seed_sweep_summary.csv")
    detail.to_csv(detail_out, index=False)
    agg.to_csv(agg_out, index=False)
    print(f"\nSaved -> {detail_out}\nSaved -> {agg_out}")
    return detail, agg


if __name__ == "__main__":
    aggregate()
