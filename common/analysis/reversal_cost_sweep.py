"""
reversal_cost_sweep.py -- screens the reversal-penalty candidate fix for the
limit-cycle problem (context.md, 7 Oct 2026 follow-up -- debounce was tried
and rejected first, see debounce_validate.py).

The limit cycle is a genuine mutual attraction between two adjacent clock
levels: from level A the policy prefers moving to B, and from B it prefers
moving back to A. `action_cost` (penalise every non-HOLD move) was already
tried and didn't fix it (reward_shape_sweep.py) -- it dilutes the one thing
that actually needs discouraging (reversing direction) by also taxing
legitimate one-way corrections equally. `reversal_cost` (train_unified_v2.Cfg)
instead penalises ONLY choosing the opposite direction from the last
non-HOLD move, directly targeting the mutual-attraction mechanism.

Mirrors reward_shape_sweep.py's structure exactly: screens on the held-out
RTX 4090 only (fast), 3 seeds, checks (a) cycle rate via the same
deterministic-rollout check, (b) pt_resnet50_train settling point, (c)
aggregate scaled-EDP sanity check.

    python common/analysis/reversal_cost_sweep.py
"""
import os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from _common import REPO  # noqa: F401
from dataset_view import discover_core_only
import eval_policies as P
from train_unified_v2 import train, Cfg, GreedyAgent
from reward_shape_sweep import rollout_trace  # same cycle-check rollout

FIVE_GPU_FLEET = [
    "nvidia_geforce_rtx_3060", "nvidia_geforce_rtx_3070",
    "nvidia_geforce_rtx_3080", "nvidia_geforce_rtx_3090",
    "nvidia_geforce_rtx_4090",
]
SETTLED = dict(gamma=0.5, double=False, lr=3e-4, hidden=128, huber=True)
HOLDOUT = "nvidia_geforce_rtx_4090"
SEEDS = [0, 1, 2]

CANDIDATES = {
    "pure_log (the problem)":  dict(use_log_reward=True),
    "reversal_cost_0.02":      dict(use_log_reward=True, reversal_cost=0.02),
    "reversal_cost_0.05":      dict(use_log_reward=True, reversal_cost=0.05),
    "reversal_cost_0.10":      dict(use_log_reward=True, reversal_cost=0.10),
    "reversal_cost_0.20":      dict(use_log_reward=True, reversal_cost=0.20),
    "reversal_cost_0.40":      dict(use_log_reward=True, reversal_cost=0.40),
}

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


def _job(t):
    name, cfg_kwargs, seed = t
    cfg = Cfg(use_response=True, episodes=800, **SETTLED, **cfg_kwargs)
    train_gpus = [g for g in FIVE_GPU_FLEET if g != HOLDOUT]
    t0 = time.time()
    net, _ = train(HOLDOUT, cfg, seed=seed, datasets={g: DS[g] for g in train_gpus})
    agent = GreedyAgent(net)

    n_cycle = 0
    resnet_settle = None
    for wl, W in sorted(PREP[HOLDOUT].items()):
        trace = rollout_trace(agent, W)
        distinct = len(set(trace[-20:]))
        if distinct > 1:
            n_cycle += 1
        if wl == "pt_resnet50_train":
            resnet_settle = float(W["core"][trace[-1]])

    pol = P.p_agent(agent)
    _, ss, _ = P.evaluate_policy(PREP[HOLDOUT], pol, use_response=True)

    return dict(config=name, seed=seed, n_cycling=n_cycle, n_total=len(PREP[HOLDOUT]),
                resnet_settle_core_frac=resnet_settle, ss=ss, secs=time.time() - t0)


def main():
    jobs = [(name, kw, seed) for name, kw in CANDIDATES.items() for seed in SEEDS]
    print(f"Screening {len(CANDIDATES)} candidates x {len(SEEDS)} seeds = {len(jobs)} jobs "
          f"(held out {HOLDOUT} only, for fast screening)...", flush=True)
    t0, rows = time.time(), []
    with Pool(min(len(jobs), os.cpu_count() or 4)) as p:
        for i, r in enumerate(p.imap_unordered(_job, jobs)):
            rows.append(r)
            if (i + 1) % 3 == 0:
                print(f"  {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    out = f"{REPO}/docs/results/reversal_cost_sweep.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)

    print("\n=== per-candidate: cycle rate, pt_resnet50_train settling point, scaled EDP ===")
    agg = d.groupby("config").agg(
        cycle_rate=("n_cycling", lambda x: (x / d.loc[x.index, "n_total"]).mean()),
        resnet_settle_mean=("resnet_settle_core_frac", "mean"),
        ss_mean=("ss", "mean"), ss_std=("ss", "std"),
    ).round(4)
    print(agg.sort_values("cycle_rate").to_string())
    print(f"\n(for reference, pt_resnet50_train settles at 0.737 under the ORIGINAL linear "
          f"checkpoint and 0.866 under the pure-log checkpoint -- higher is the hardware-"
          f"validated 'safer' direction)")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
