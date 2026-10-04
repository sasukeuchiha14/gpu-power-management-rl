"""
action_cost_fix_validate.py -- full 5-GPU, 10-seed offline validation of the
winning candidate from reward_shape_sweep.py: use_log_reward=True combined
with action_cost=0.05. Found to have the lowest limit-cycle rate of all 8
candidates screened (0.22, even below the original pre-fix baseline's 0.39)
while still settling pt_resnet50_train at the hardware-validated safe clock
level (0.866) -- see reward_shape_sweep.py's docstring and
docs/results/reward_shape_sweep.csv for the screening that selected this.

    python common/analysis/action_cost_fix_validate.py
"""
import os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool

import pandas as pd
import torch
torch.set_num_threads(1)

from _common import REPO  # noqa: F401
from dataset_view import discover_core_only
import eval_policies as P
from train_unified_v2 import train, Cfg, GreedyAgent

FIVE_GPU_FLEET = [
    "nvidia_geforce_rtx_3060", "nvidia_geforce_rtx_3070",
    "nvidia_geforce_rtx_3080", "nvidia_geforce_rtx_3090",
    "nvidia_geforce_rtx_4090",
]
SETTLED = dict(gamma=0.5, double=False, lr=3e-4, hidden=128, huber=True)

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


def _job(t):
    gpu, seed = t
    cfg = Cfg(use_response=True, use_log_reward=True, action_cost=0.05, episodes=800, **SETTLED)
    net, _ = train(gpu, cfg, seed=seed, datasets={g: DS[g] for g in FIVE_GPU_FLEET})
    pol = P.p_agent(GreedyAgent(net))
    tr, ss, _ = P.evaluate_policy(PREP[gpu], pol, use_response=True)
    return dict(policy="log-reward + action_cost=0.05", gpu=gpu, seed=seed, traj=tr, ss=ss)


def main():
    jobs = [(g, s) for g in FIVE_GPU_FLEET for s in range(10)]
    print(f"Training {len(jobs)} jobs...", flush=True)
    t0, rows = time.time(), []
    with Pool(min(len(jobs), os.cpu_count() or 8)) as p:
        for i, r in enumerate(p.imap_unordered(_job, jobs)):
            rows.append(r)
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    out = f"{REPO}/docs/results/action_cost_fix_10seed.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)
    print(d.groupby("gpu").ss.agg(["mean", "std"]).round(4).to_string())
    print(f"\noverall mean: {d.ss.mean():.4f}  std: {d.ss.std():.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
