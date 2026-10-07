"""
combo_validate.py -- full 5-GPU, 10-seed validation of the combo candidate
found while exploring reversal_cost further (context.md, 7 Oct 2026):
reward_power=0.25 (partial log compression, not the full log reward) PLUS
reversal_cost=0.40. A 3-seed screen on the RTX 4090 holdout got ZERO
cycling across all 3 seeds -- better than any single-knob candidate tried
(reversal_cost=0.30 alone: 0.30 cycle rate*, see reversal_cost_validate.py).
Checks cycle rate on every held-out GPU (not just one) and the
pt_resnet50_train/4090 safety settling point, same rigor as every other
candidate this investigation.

    python common/analysis/combo_validate.py
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
from reward_shape_sweep import rollout_trace

FIVE_GPU_FLEET = [
    "nvidia_geforce_rtx_3060", "nvidia_geforce_rtx_3070",
    "nvidia_geforce_rtx_3080", "nvidia_geforce_rtx_3090",
    "nvidia_geforce_rtx_4090",
]
SETTLED = dict(gamma=0.5, double=False, lr=3e-4, hidden=128, huber=True)
REWARD_POWER = 0.25
REVERSAL_COST = 0.40

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


def _job(t):
    gpu, seed = t
    cfg = Cfg(use_response=True, reward_power=REWARD_POWER, reversal_cost=REVERSAL_COST,
              episodes=800, **SETTLED)
    net, _ = train(gpu, cfg, seed=seed, datasets={g: DS[g] for g in FIVE_GPU_FLEET})
    agent = GreedyAgent(net)

    n_cycle, resnet_settle = 0, None
    for wl, W in sorted(PREP[gpu].items()):
        trace = rollout_trace(agent, W)
        if len(set(trace[-20:])) > 1:
            n_cycle += 1
        if wl == "pt_resnet50_train":
            resnet_settle = float(W["core"][trace[-1]])

    pol = P.p_agent(agent)
    tr, ss, _ = P.evaluate_policy(PREP[gpu], pol, use_response=True)
    return dict(gpu=gpu, seed=seed, n_cycling=n_cycle, n_total=len(PREP[gpu]),
                resnet_settle=resnet_settle, traj=tr, ss=ss)


def main():
    jobs = [(g, s) for g in FIVE_GPU_FLEET for s in range(10)]
    print(f"Training {len(jobs)} jobs (reward_power={REWARD_POWER}, "
          f"reversal_cost={REVERSAL_COST})...", flush=True)
    t0, rows = time.time(), []
    with Pool(min(len(jobs), os.cpu_count() or 8)) as p:
        for i, r in enumerate(p.imap_unordered(_job, jobs)):
            rows.append(r)
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    out = f"{REPO}/docs/results/combo_power025_reversal040_10seed.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)

    print("\n=== per-GPU: cycle rate, resnet settle, steady-state scaled EDP ===")
    agg = d.groupby("gpu").agg(
        cycle_rate=("n_cycling", lambda x: (x / d.loc[x.index, "n_total"]).mean()),
        resnet_settle_mean=("resnet_settle", "mean"),
        ss_mean=("ss", "mean"), ss_std=("ss", "std"),
    ).round(4)
    print(agg.to_string())
    print(f"\noverall cycle_rate: {(d.n_cycling / d.n_total).mean():.4f}")
    print(f"overall ss mean: {d.ss.mean():.4f}  std: {d.ss.std():.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
