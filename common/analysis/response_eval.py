"""
response_eval.py -- Phase 1 checkpoint (context.md "IMMEDIATE NEXT STEP",
NORTHSTAR row 11): does a DDQN trained on the response-augmented 8-feature
state beat (a) a tuned fixed clock, (b) the old 6-feature snapshot-state
DDQN, and critically (c) the hand-coded response-driven hill-climb?

Leave-one-GPU-out, restricted to the 5 GPUs re-collected with clean,
window-scoped telemetry (context.md, 4 Oct 2026 decision) -- 4070/4070 Ti/
laptop are deferred, not included, because their telemetry is still
startup-diluted (NORTHSTAR row 10).

Policies compared, all evaluated under the identical protocol
(eval_policies.evaluate_policy: 50 steps, fixed per-workload start state):
  * governor            -- pinned max clock, no telemetry, no learning
  * fixed, tuned LOO     -- single constant clock, tuned on the training
                            GPUs only (baselines.tune_fixed_loo)
  * old snapshot DDQN    -- train_unified_v2 with use_response=False: the
                            ablation. Same settled hyperparameters, same
                            clean data, same bugfixes as the new agent --
                            the ONLY difference is whether the state carries
                            the 2 response features. Isolates what they buy.
  * hill-climb (ref)     -- eval_policies.p_hillclimb: hand-coded, measures
                            the same response features directly, no
                            learning. The ceiling-ish reference from
                            context.md's 4 Oct finding (re-implemented --
                            see that function's docstring for the caveat).
  * response DDQN (ours) -- train_unified_v2 with use_response=True: the
                            actual contribution this phase is testing.

Settled hyperparameters (CLAUDE.md Sec 0.2, do not re-sweep): gamma=0.5,
plain DQN (double=False), Huber loss, lr=3e-4, hidden=128.

    python common/analysis/response_eval.py
    python common/analysis/response_eval.py --seeds 0,1,2 --episodes 800
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
from baselines import tune_fixed_loo

FIVE_GPU_FLEET = [
    "nvidia_geforce_rtx_3060", "nvidia_geforce_rtx_3070",
    "nvidia_geforce_rtx_3080", "nvidia_geforce_rtx_3090",
    "nvidia_geforce_rtx_4090",
]

SETTLED = dict(gamma=0.5, double=False, lr=3e-4, hidden=128, huber=True)

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


def _train_job(t):
    name, use_response, gpu, seed, episodes, train_gpus = t
    cfg = Cfg(use_response=use_response, episodes=episodes, **SETTLED)
    t0 = time.time()
    # Train only on the clean 5-GPU fleet (minus whichever is held out) --
    # never mix in the still-diluted-telemetry GPUs (NORTHSTAR row 10).
    net, _ = train(gpu, cfg, seed=seed, datasets={g: DS[g] for g in train_gpus})
    pol = P.p_agent(GreedyAgent(net))
    tr, ss, _ = P.evaluate_policy(PREP[gpu], pol, use_response=use_response)
    return dict(policy=name, gpu=gpu, seed=seed, traj=tr, ss=ss, secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default=None, help="comma-separated, default = the 5-GPU clean fleet")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gpus = a.gpus.split(",") if a.gpus else FIVE_GPU_FLEET
    missing = [g for g in gpus if g not in DS]
    if missing:
        sys.exit(f"not discovered (ran preprocess.py?): {missing}\navailable: {sorted(DS)}")
    seeds = [int(s) for s in a.seeds.split(",")]

    rows = []

    # --- static / tuned baselines (cheap, no training, run directly) ---
    print("Static baselines...", flush=True)
    for g in gpus:
        tr, ss, _ = P.evaluate_policy(PREP[g], P.p_governor)
        rows.append(dict(policy="governor", gpu=g, seed=-1, traj=tr, ss=ss, secs=0.0))

        # Constant policy label across GPUs even though the picked f can
        # differ per fold -- baselines.py already hit a KeyError from
        # embedding f in the label when two folds disagreed (3 Oct 2026).
        f = tune_fixed_loo(PREP, gpus, g)  # tuned on the OTHER clean-fleet GPUs only
        tr, ss, _ = P.evaluate_policy(PREP[g], P.p_fixed(f))
        rows.append(dict(policy="fixed, tuned LOO", gpu=g, seed=-1, traj=tr, ss=ss, secs=0.0, picked=f))

        tr, ss, _ = P.evaluate_policy(PREP[g], P.p_hillclimb(), use_response=True)
        rows.append(dict(policy="hill-climb (hand-coded reference)", gpu=g, seed=-1, traj=tr, ss=ss, secs=0.0))

    picks_df = pd.DataFrame(rows)
    picks = picks_df.loc[picks_df.policy == "fixed, tuned LOO", ["gpu", "picked"]]
    print("tuned fixed clock picked per held-out GPU: "
          + ", ".join(f"{r.gpu}={r.picked:.2f}" for r in picks.itertuples()))

    # --- the two trained DDQNs: same hyperparameters, only the state differs ---
    jobs = []
    for g in gpus:
        for s in seeds:
            jobs.append(("old snapshot DDQN (ablation)", False, g, s, a.episodes, gpus))
            jobs.append(("response-aware DDQN (ours)", True, g, s, a.episodes, gpus))

    print(f"Training {len(jobs)} DDQNs on {a.workers} workers "
          f"({len(gpus)} GPUs x {len(seeds)} seeds x 2 state variants)...", flush=True)
    t0 = time.time()
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(_train_job, jobs)):
            rows.append(r)
            if (i + 1) % max(1, a.workers) == 0:
                print(f"  {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    out = a.out or f"{REPO}/docs/results/response_eval.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)

    print("\n=== steady-state scaled EDP, mean (+- std over seeds) per policy -- LOWER IS BETTER ===")
    agg = (d.groupby("policy")
             .agg(ss_mean=("ss", "mean"), ss_std=("ss", "std"), n=("ss", "size"))
             .sort_values("ss_mean"))
    print(agg.round(4).to_string())

    print("\n=== per-GPU steady-state scaled EDP (mean over seeds where applicable) ===")
    piv = d.pivot_table(index="policy", columns="gpu", values="ss", aggfunc="mean")
    print(piv.round(4).to_string())

    print("\n=== wins: how many of the GPUs does each learned/hand-coded policy beat "
          "'fixed, tuned LOO' on? ===")
    fixed_rows = d[d.policy == "fixed, tuned LOO"]
    fixed_by_gpu = fixed_rows.groupby("gpu")["ss"].mean()
    for name in ["hill-climb (hand-coded reference)", "old snapshot DDQN (ablation)",
                 "response-aware DDQN (ours)"]:
        sub = d[d.policy == name].groupby("gpu")["ss"].mean()
        wins = int((sub < fixed_by_gpu.reindex(sub.index)).sum())
        print(f"  {name:<36} {wins}/{len(sub)}")

    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
