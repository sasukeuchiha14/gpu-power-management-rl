"""
combo_sweep.py -- quick 3-seed screen (RTX 4090 holdout only) of whether
combining reversal_cost with a partial log compression (reward_power,
instead of the full log reward) recovers some of the steady-state cost
reversal_cost pays on its own, without reopening the limit cycle
(context.md, NORTHSTAR row 11, 7 Oct 2026).

RESULT: the screen's best candidate (reward_power=0.25 + reversal_cost=0.40,
0/12 cycling across all 3 seeds here) did NOT hold up at full 5-GPU/10-seed
validation (common/analysis/combo_validate.py) -- it matched the best
single-knob cycle rate (0.007, same as reversal_cost=0.60 alone) but
introduced a genuinely worse failure mode: pt_resnet50_train settled at
EXACTLY max clock (core_frac=1.0, no downclocking at all) on 3 of 5 held-out
GPUs. Rejected in favour of the simpler reversal_cost=0.30-alone candidate.
Kept here for the record, same reason hillclimb_sweep.py is kept for its
own non-reproduction.

    python common/analysis/combo_sweep.py
"""
import os
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
HOLDOUT = "nvidia_geforce_rtx_4090"
SEEDS = [0, 1, 2]
CANDIDATES = {
    "reversal0.30_fulllog":    dict(use_log_reward=True, reversal_cost=0.30),
    "reversal0.30_power0.5":   dict(reward_power=0.5, reversal_cost=0.30),
    "reversal0.30_power0.25":  dict(reward_power=0.25, reversal_cost=0.30),
    "reversal0.40_power0.25":  dict(reward_power=0.25, reversal_cost=0.40),
}

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


def _job(t):
    name, kw, seed = t
    cfg = Cfg(use_response=True, episodes=800, **SETTLED, **kw)
    train_gpus = [g for g in FIVE_GPU_FLEET if g != HOLDOUT]
    net, _ = train(HOLDOUT, cfg, seed=seed, datasets={g: DS[g] for g in train_gpus})
    agent = GreedyAgent(net)

    n_cycle, resnet_settle = 0, None
    for wl, W in sorted(PREP[HOLDOUT].items()):
        trace = rollout_trace(agent, W)
        if len(set(trace[-20:])) > 1:
            n_cycle += 1
        if wl == "pt_resnet50_train":
            resnet_settle = float(W["core"][trace[-1]])

    pol = P.p_agent(agent)
    _, ss, _ = P.evaluate_policy(PREP[HOLDOUT], pol, use_response=True)
    return dict(config=name, seed=seed, n_cycling=n_cycle, n_total=len(PREP[HOLDOUT]),
                resnet_settle=resnet_settle, ss=ss)


def main():
    jobs = [(n, kw, s) for n, kw in CANDIDATES.items() for s in SEEDS]
    with Pool(min(len(jobs), os.cpu_count() or 4)) as p:
        rows = list(p.imap_unordered(_job, jobs))
    d = pd.DataFrame(rows)
    out = f"{REPO}/docs/results/combo_sweep.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)
    agg = d.groupby("config").agg(
        cycle_rate=("n_cycling", lambda x: (x / d.loc[x.index, "n_total"]).mean()),
        resnet_settle=("resnet_settle", "mean"), ss_mean=("ss", "mean"), ss_std=("ss", "std"),
    ).round(4)
    print(agg.sort_values("cycle_rate").to_string())
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
