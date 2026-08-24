"""
dataset_view.py -- the core-only view of each GPU's profiled dataset.

WHY THIS EXISTS (found 25 Aug 2026, while diagnosing the leave-one-GPU-out
seed instability from commit 8b68617):

collect_dataset.py profiles a 2D grid -- 8 core levels x 3 memory levels per
workload, 24 rows. The unified controller, by design (see common/features.py),
controls the CORE clock only: memory control is deliberately excluded because
mobile GeForce parts reject it outright, and one unified action space is worth
more than per-tier capability. Two consequences went unnoticed until now:

  1. DEAD DATA. The environment snaps to the nearest profiled row by core_frac
     alone. When 3 rows share a core_frac, the tie-break is stable dataframe
     order, which is memory-major from the collector -- so snapping ALWAYS
     lands on the top memory state and the other 2/3 of every dataset is
     unreachable during both training and evaluation.

  2. A CRUSHED REWARD SCALE, which is the damaging one. preprocess computes
     Reward = 1 - per-workload-minmax(EDP) across all 24 rows. Memory clock
     moves EDP far more than core clock does on these cards, so most of the
     [0,1] reward range is consumed by a dimension the controller cannot
     touch. Measured on RTX 3060 / pt_vector_add: every core level the
     controller can actually reach lies in Reward 0.742-1.000 -- the whole
     decision space is squeezed into the top 26% of the scale, so the
     difference between the best and worst available action is ~4x smaller
     than the reward range implies. That is a weak, noise-prone learning
     signal, and a plausible driver of the observed seed-to-seed instability.

This module returns the view the controller actually operates in: rows at the
single memory state core-only snapping reaches, with Reward re-normalised over
just those rows so the full [0,1] range spans the reachable decision space.

The 2D data is NOT discarded -- the on-disk preprocessed_dataset.csv keeps all
24 rows, and a future 2D (core+memory) controller can use them. This is a view,
not a migration.
"""
import os, sys
import numpy as np
import pandas as pd

_common = os.path.dirname(os.path.abspath(__file__))
if _common not in sys.path:
    sys.path.insert(0, _common)
from unified_env import discover_datasets as _discover_full  # noqa: E402


def core_only_view(df, mem_select="max"):
    """One GPU's dataframe -> the core-only reachable view, Reward re-normalised
    per workload over the reachable rows only."""
    if "Mem_F" not in df.columns:
        return df.copy()
    out = []
    for wl, g in df.groupby("Workload_Name"):
        if mem_select == "max":
            mem = g["Mem_F"].max()
        else:
            raise ValueError(mem_select)
        gg = g[g["Mem_F"] == mem].copy()
        if gg["core_frac"].duplicated().any():
            gg = gg.sort_values("core_frac").drop_duplicates("core_frac", keep="first")
        span = gg["EDP"].max() - gg["EDP"].min()
        gg["Reward"] = 1.0 if span <= 0 else 1.0 - (gg["EDP"] - gg["EDP"].min()) / span
        out.append(gg)
    return pd.concat(out, ignore_index=True)


def discover_core_only(mem_select="max"):
    """{gpu_tag: core-only view df} for every discovered 6-feature dataset."""
    return {g: core_only_view(d, mem_select) for g, d in _discover_full().items()}


if __name__ == "__main__":
    full = _discover_full()
    view = discover_core_only()
    print(f"{'gpu':<28}{'rows(2D)':>10}{'rows(core-only)':>17}{'reward span before':>20}{'after':>8}")
    for g in sorted(full):
        f, v = full[g], view[g]
        # reward span actually reachable by core-only snapping, before the fix
        spans = []
        for wl, gg in f.groupby("Workload_Name"):
            o = gg.sort_values("core_frac", kind="stable").groupby("core_frac", sort=True).head(1)
            spans.append(o["Reward"].max() - o["Reward"].min())
        after = [gg["Reward"].max() - gg["Reward"].min() for _, gg in v.groupby("Workload_Name")]
        print(f"{g:<28}{len(f):>10}{len(v):>17}{np.mean(spans):>20.3f}{np.mean(after):>8.3f}")
