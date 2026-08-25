"""
sweep_rl.py -- measure which RL knobs and architectures actually help, instead
of arguing about them.

Trains common/train_unified_v2.py under many configurations, leave-one-GPU-out,
across several seeds, and reports mean and seed-variance of the resulting
policy's steady-state scaled EDP on the held-out GPU.

    # quick screen: individual knobs, 4 GPUs, 2 seeds  (~30-70 min on 8 cores)
    python common/analysis/sweep_rl.py --preset screen

    # best combos, all 8 GPUs, 3 seeds  (~60-90 min)
    python common/analysis/sweep_rl.py --preset combos

    # anything specific
    python common/analysis/sweep_rl.py --configs gamma0.50,no_double --seeds 0,1,2

WHAT THE SCREEN FOUND (25 Aug 2026, on the corrected core-only view):
  * The three bug fixes in train_unified_v2 plus tuning roughly halved both the
    error (0.132 -> 0.069 steady-state scaled EDP) and its seed variance.
  * gamma 0.99 -> 0.50 was the single biggest lever, consistent with the task
    being near-bandit rather than long-horizon: the reward depends only on the
    row you land on, never on the path taken there.
  * Dueling heads and an absolute action space both HURT. Plain DQN beat
    Double-DQN.
  * Even so, every RL configuration lost to a fixed 0.6x clock (0.037). Read
    analysis/telemetry_dilution.py before concluding anything from that -- the
    telemetry these policies consume is startup-diluted, so this measures
    RL-on-broken-inputs, not the ceiling of RL on this problem.
"""
import argparse, os, sys, time
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
from train_unified_v2 import train, Cfg, GreedyAgent, p_agent_abs

CONFIGS = {
    # --- individual knobs (screen) ---
    "v2_fixes_only":    dict(),
    "eps3000":          dict(episodes=3000),
    "gamma0.90":        dict(gamma=0.90),
    "gamma0.50":        dict(gamma=0.50),
    "gamma0.00":        dict(gamma=0.0),
    "dueling":          dict(dueling=True),
    "soft_tau0.01":     dict(tau=0.01),
    "lr3e-4":           dict(lr=3e-4),
    "batch128":         dict(batch=128),
    "hidden256":        dict(hidden=256),
    "actioncost0.02":   dict(action_cost=0.02),
    "no_double":        dict(double=False),
    "abs8_gamma0":      dict(abs_actions=8, gamma=0.0),
    "abs8_gamma0.9":    dict(abs_actions=8, gamma=0.9),
    "gamma0_eps3000":   dict(gamma=0.0, episodes=3000),
    # --- combinations of the individually-best knobs ---
    "C_nd_g50":         dict(double=False, gamma=0.50),
    "C_nd_g50_lr":      dict(double=False, gamma=0.50, lr=3e-4),
    "C_nd_g50_lr_h256": dict(double=False, gamma=0.50, lr=3e-4, hidden=256),
    "C_full_ep2500":    dict(double=False, gamma=0.50, lr=3e-4, hidden=256, episodes=2500),
    "C_full_tau":       dict(double=False, gamma=0.50, lr=3e-4, hidden=256, tau=0.005),
}
PRESETS = {
    "screen": (list(CONFIGS)[:15],
               ["nvidia_geforce_rtx_3080", "nvidia_geforce_rtx_3090",
                "nvidia_geforce_rtx_4090", "rtx3050ti_laptop"], [0, 1]),
    "combos": (list(CONFIGS)[15:], None, [0, 1, 2]),
}

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


def _job(t):
    name, gpu, seed = t
    cfg = Cfg(**CONFIGS[name])
    t0 = time.time()
    net, _ = train(gpu, cfg, seed=seed, datasets=DS)
    pol = p_agent_abs(net, cfg.abs_actions) if cfg.abs_actions else P.p_agent(GreedyAgent(net))
    tr, ss, _ = P.evaluate_policy(PREP[gpu], pol)
    return dict(config=name, gpu=gpu, seed=seed, traj=tr, ss=ss, secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--configs", default=None)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfgs, gpus, seeds = PRESETS.get(a.preset, (list(CONFIGS), None, [0, 1]))
    if a.configs: cfgs = a.configs.split(",")
    if a.gpus:    gpus = a.gpus.split(",")
    if a.seeds:   seeds = [int(s) for s in a.seeds.split(",")]
    gpus = gpus or sorted(DS)
    bad = [c for c in cfgs if c not in CONFIGS]
    if bad: sys.exit(f"unknown config(s): {bad}\nknown: {sorted(CONFIGS)}")

    jobs = [(c, g, s) for c in cfgs for g in gpus for s in seeds]
    print(f"{len(jobs)} runs on {a.workers} workers "
          f"({len(cfgs)} configs x {len(gpus)} GPUs x {len(seeds)} seeds)", flush=True)
    t0, res = time.time(), []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(_job, jobs)):
            res.append(r)
            if (i + 1) % a.workers == 0:
                print(f"  {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(res)
    out = a.out or f"{REPO}/docs/results/sweep_{a.preset or 'custom'}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)

    print("\n=== steady-state scaled EDP, mean over GPUs+seeds (lower better) ===")
    agg = (d.groupby("config")
             .agg(ss_mean=("ss", "mean"), ss_std=("ss", "std"),
                  traj_mean=("traj", "mean"), secs=("secs", "mean"))
             .sort_values("ss_mean"))
    print(agg.round(4).to_string())
    print("\n=== per-GPU ===")
    print(d.pivot_table(index="config", columns="gpu", values="ss").round(3).to_string())
    print(f"\nsaved -> {out}")
    print("\nCompare against the baselines in docs/results/baselines.csv -- in "
          "particular the fixed 0.6x clock, which no RL config has beaten.")


if __name__ == "__main__":
    main()
