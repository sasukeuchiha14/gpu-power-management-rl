"""
dwell_screen.py -- offline screen of reward variants meant to keep the
reversal_cost cycle suppression WITHOUT erasing the compute-bound power saving
(the trade-off on the RESULTS slide: cycle rate 0.182 -> 0.022 at 0.30, but the
batch=32 power saving goes from +8% to -1.5% on real hardware).

For every candidate it trains leave-one-GPU-out (held-out GPU = --holdout) and
rolls the greedy policy out from max clock on every workload, then reports:
  cycle    fraction of workloads whose last 20 clock indices are not constant
  p_save   power saving vs pinned max clock over the last 20 steps (table lookup)
  slow     runtime change vs pinned max clock over the last 20 steps
  edp      scaled EDP (lower is better) over the last 20 steps
split by regime: 'compute' workloads vs everything else, since that is where
the saving is lost.

    python common/analysis/dwell_screen.py --seeds 3
"""
import argparse, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from _common import REPO
from dataset_view import discover_core_only
import eval_policies as P
from train_unified_v2 import train, Cfg, GreedyAgent
from reward_shape_sweep import rollout_trace

FLEET = ["nvidia_geforce_rtx_3060", "nvidia_geforce_rtx_3070", "nvidia_geforce_rtx_3080",
         "nvidia_geforce_rtx_3090", "nvidia_geforce_rtx_4090"]
SETTLED = dict(gamma=0.5, double=False, lr=3e-4, hidden=128, huber=True)
COMPUTE = {"pt_resnet50_train", "pt_matmul", "pt_resnet50", "pt_bert", "pt_mobilenet"}

CANDIDATES = {
    "pure_log":        dict(),
    "rc0.10":          dict(reversal_cost=0.10),
    "rc0.30":          dict(reversal_cost=0.30),
    "rc0.30_win2":     dict(reversal_cost=0.30, reversal_window=2),
    "rc0.30_win5":     dict(reversal_cost=0.30, reversal_window=5),
    "rc0.60_win2":     dict(reversal_cost=0.60, reversal_window=2),
    "rc0.10_hold.02":  dict(reversal_cost=0.10, hold_bonus=0.02),
    "rc0.10_hold.05":  dict(reversal_cost=0.10, hold_bonus=0.05),
    "util_rc0.60":     dict(reversal_cost=0.60, reversal_util_scale=True),
    "util_rc1.00":     dict(reversal_cost=1.00, reversal_util_scale=True),
    "win2_hold.02":    dict(reversal_cost=0.30, reversal_window=2, hold_bonus=0.02),
    "win2_hold.01":    dict(reversal_cost=0.30, reversal_window=2, hold_bonus=0.01),
    "win2_hold.03":    dict(reversal_cost=0.30, reversal_window=2, hold_bonus=0.03),
}

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


def _job(t):
    name, holdout, seed = t
    cfg = Cfg(use_response=True, use_log_reward=True, episodes=800, **SETTLED, **CANDIDATES[name])
    net, _ = train(holdout, cfg, seed=seed, datasets={g: DS[g] for g in FLEET})
    ag = GreedyAgent(net)
    rows = []
    for wl, W in sorted(PREP[holdout].items()):
        tr = rollout_trace(ag, W)
        tail = np.array(tr[-20:])
        top = len(W["core"]) - 1
        p_max, t_max = W["power_w"][top], W["time_s"][top]
        rows.append(dict(
            cand=name, gpu=holdout, seed=seed, wl=wl,
            regime="compute" if wl in COMPUTE else "other",
            cycle=int(len(set(tail)) > 1),
            p_save=1.0 - W["power_w"][tail].mean() / p_max,
            slow=W["time_s"][tail].mean() / t_max - 1.0,
            edp=W["sedp"][tail].mean(),
            settle=float(W["core"][tail[-1]]),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--holdout", default="nvidia_geforce_rtx_4090")
    ap.add_argument("--only", default=None, help="comma list of candidate names")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--tag", default="screen")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(CANDIDATES)
    jobs = [(n, a.holdout, s) for n in names for s in range(a.seeds)]
    print(f"{len(jobs)} jobs", flush=True)
    t0, rows = time.time(), []
    with Pool(a.procs) as p:
        for i, r in enumerate(p.imap_unordered(_job, jobs)):
            rows += r
            print(f"  {i+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows)
    out = f"{REPO}/docs/results/dwell_screen_{a.tag}.csv"
    d.to_csv(out, index=False)
    g = d.groupby(["cand", "regime"]).agg(cycle=("cycle", "mean"), p_save=("p_save", "mean"),
                                           slow=("slow", "mean"), edp=("edp", "mean")).round(3)
    print(g.to_string())
    print("\nresnet50_train only:")
    print(d[d.wl == "pt_resnet50_train"].groupby("cand").agg(
        cycle=("cycle", "mean"), p_save=("p_save", "mean"), slow=("slow", "mean"),
        settle=("settle", "mean")).round(3).to_string())
    print("saved", out)


if __name__ == "__main__":
    main()
