"""
reward_shaping_eval.py -- does training on the log-scale reward (use_log_reward,
see train_unified_v2.Cfg and dataset_view.core_only_view) fix the over-aggressive
downclocking a Phase 2 hardware run exposed, WITHOUT regressing the already-
validated offline result (response-aware DDQN, 0.0250 mean steady-state scaled
EDP, 10 seeds, response_eval.py)?

Trains ONLY the response-aware variant (use_response=True), with
use_log_reward=True, same settled hyperparameters, same 5-GPU leave-one-GPU-out
setup and seed range as the validated Phase 1 result, so this is a direct,
apples-to-apples comparison against docs/results/response_eval_10seed.csv.
Evaluation uses the SAME unchanged linear scaled-EDP metric as every other
number in this project -- only the training signal differs.

    python common/analysis/reward_shaping_eval.py --seeds 0,1,2,3,4,5,6,7,8,9
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
from train_unified_v2 import train, Cfg, GreedyAgent

FIVE_GPU_FLEET = [
    "nvidia_geforce_rtx_3060", "nvidia_geforce_rtx_3070",
    "nvidia_geforce_rtx_3080", "nvidia_geforce_rtx_3090",
    "nvidia_geforce_rtx_4090",
]
SETTLED = dict(gamma=0.5, double=False, lr=3e-4, hidden=128, huber=True)

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


def _train_job(t):
    gpu, seed, episodes, train_gpus = t
    cfg = Cfg(use_response=True, use_log_reward=True, episodes=episodes, **SETTLED)
    t0 = time.time()
    net, _ = train(gpu, cfg, seed=seed, datasets={g: DS[g] for g in train_gpus})
    pol = P.p_agent(GreedyAgent(net))
    tr, ss, _ = P.evaluate_policy(PREP[gpu], pol, use_response=True)
    return dict(policy="log-reward response-aware DDQN", gpu=gpu, seed=seed,
                traj=tr, ss=ss, secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gpus = a.gpus.split(",") if a.gpus else FIVE_GPU_FLEET
    seeds = [int(s) for s in a.seeds.split(",")]

    jobs = [(g, s, a.episodes, gpus) for g in gpus for s in seeds]
    print(f"Training {len(jobs)} log-reward DDQNs on {a.workers} workers "
          f"({len(gpus)} GPUs x {len(seeds)} seeds)...", flush=True)
    t0, rows = time.time(), []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(_train_job, jobs)):
            rows.append(r)
            if (i + 1) % max(1, a.workers) == 0:
                print(f"  {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    out = a.out or f"{REPO}/docs/results/reward_shaping_eval.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)

    print("\n=== log-reward response-aware DDQN: steady-state scaled EDP ===")
    print(f"mean: {d.ss.mean():.4f}  std: {d.ss.std():.4f}  n: {len(d)}")
    print("\nper-GPU:")
    print(d.groupby("gpu")["ss"].agg(["mean", "std"]).round(4).to_string())
    print(f"\nsaved -> {out}")
    print("\nCompare against docs/results/response_eval_10seed.csv "
          "(response-aware DDQN, linear reward, mean 0.0250, std 0.0127).")


if __name__ == "__main__":
    main()
