"""
workload_holdout_eval.py -- held-out WORKLOAD generalization (NORTHSTAR row 5),
redone under the current 6-feature `common/` architecture. The row 5 result on
the old RTX 4500 tree doesn't count: that design gives an unseen app a
borrowed identity via nearest-neighbour match against its workload one-hot,
which structurally cannot generalize to a truly novel workload. The 6-feature
design has no such one-hot, so this is a real test.

Protocol: leave-one-WORKLOAD-out cross-validation on the 5-GPU clean fleet
(3060/3070/3080/3090/4090 -- the same fleet row 11's main result uses).
For each of the 12 workloads: train the response-aware DDQN (same settled
hyperparameters as every other validated result: gamma=0.5, plain DQN, Huber,
lr=3e-4, width 128) on all 5 GPUs but with that ONE workload's rows removed
from every GPU's training pool, then evaluate ONLY on that workload, across
all 5 GPUs (the GPUs were seen during training -- only the workload is
novel, isolating this question from the already-answered GPU-generalization
question). Compares against the tuned-fixed-clock baseline on the same
held-out workload, so "generalizes" means "still beats a baseline that needs
no learning at all," not just "produces a number."

Multiple seeds per fold (this investigation has been burned by single-seed
claims repeatedly -- see context.md's reversal_cost section) via --seeds.

    python common/analysis/workload_holdout_eval.py --seeds 0,1,2
"""
import argparse, os, time
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
WORKLOADS = sorted(set().union(*[set(df["Workload_Name"].unique()) for df in DS.values()]))


def _job(t):
    holdout_wl, seed = t
    # Training data: every GPU in the fleet, every workload EXCEPT the
    # held-out one. Pass a holdout_gpu that matches nothing real so
    # train_unified_v2.build_pool() excludes no GPU -- only our pre-filtered
    # per-GPU dataframes (workload rows removed) control what's excluded.
    filtered = {g: df[df["Workload_Name"] != holdout_wl].reset_index(drop=True)
                for g, df in DS.items() if g in FIVE_GPU_FLEET}
    cfg = Cfg(use_response=True, episodes=800, **SETTLED)
    net, _ = train("__none__", cfg, seed=seed, datasets=filtered)
    agent = GreedyAgent(net)
    pol = P.p_agent(agent)

    # Evaluation: the held-out workload's rows, UNFILTERED, across all 5 GPUs.
    ss_vals, traj_vals = [], []
    for g in FIVE_GPU_FLEET:
        df_wl = DS[g][DS[g]["Workload_Name"] == holdout_wl]
        if df_wl.empty:
            continue
        prep = P.prepare(df_wl)
        traj, ss, _ = P.evaluate_policy(prep, pol, use_response=True)
        ss_vals.append(ss)
        traj_vals.append(traj)

    # Tuned-fixed-clock baseline, calibrated the SAME way baselines.py does it
    # (`--loo-tune`): pick ONE global fraction by sweeping on the TRAINING
    # workloads only (never touching the held-out one), then evaluate that
    # single fraction on the held-out workload. Tuning the fraction directly
    # on the held-out workload would let a "fixed clock" cheat its way to
    # near-oracle performance on a single-workload evaluation -- not the
    # calibration this project uses anywhere else.
    train_prep = {g: P.prepare(filtered[g]) for g in FIVE_GPU_FLEET}
    best_frac, best_train_ss = None, None
    for frac in np.arange(0.3, 1.01, 0.05):
        ss_this_frac = [P.evaluate_policy(train_prep[g], P.p_fixed(frac), use_response=False)[1]
                         for g in FIVE_GPU_FLEET]
        m = float(np.mean(ss_this_frac))
        if best_train_ss is None or m < best_train_ss:
            best_train_ss, best_frac = m, frac

    fixed_vals = []
    for g in FIVE_GPU_FLEET:
        df_wl = DS[g][DS[g]["Workload_Name"] == holdout_wl]
        if df_wl.empty:
            continue
        prep = P.prepare(df_wl)
        _, ss, _ = P.evaluate_policy(prep, P.p_fixed(best_frac), use_response=False)
        fixed_vals.append(ss)

    return dict(workload=holdout_wl, seed=seed, tuned_frac=best_frac,
                ss_mean=float(np.mean(ss_vals)), traj_mean=float(np.mean(traj_vals)),
                fixed_best_ss=float(np.mean(fixed_vals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    jobs = [(wl, s) for wl in WORKLOADS for s in seeds]
    print(f"Training {len(jobs)} jobs ({len(WORKLOADS)} workloads x {len(seeds)} seeds)...", flush=True)
    t0, rows = time.time(), []
    with Pool(min(len(jobs), os.cpu_count() or 8)) as p:
        for i, r in enumerate(p.imap_unordered(_job, jobs)):
            rows.append(r)
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    out = f"{REPO}/docs/results/workload_holdout_eval.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)

    print("\n=== per-workload: held-out ss, tuned-fixed-clock ss (lower is better) ===")
    agg = d.groupby("workload").agg(
        ss_mean=("ss_mean", "mean"), ss_std=("ss_mean", "std"),
        fixed_best_ss=("fixed_best_ss", "mean"),
    ).round(4)
    agg["beats_fixed"] = agg.ss_mean < agg.fixed_best_ss
    print(agg.to_string())
    print(f"\noverall: response-aware DDQN ss={d.ss_mean.mean():.4f} (std {d.ss_mean.std():.4f}), "
          f"tuned-fixed-clock ss={d.fixed_best_ss.mean():.4f}")
    print(f"wins on {agg.beats_fixed.sum()}/{len(agg)} held-out workloads")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
