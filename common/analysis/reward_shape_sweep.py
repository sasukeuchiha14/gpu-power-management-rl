"""
reward_shape_sweep.py -- screens candidate fixes for the limit-cycle problem
the pure log-reward fix introduced (context.md, NORTHSTAR row 11, 5 Oct 2026
repeated-trials finding): repeated hardware trials showed the log-reward
checkpoint fixes the compute-bound over-aggressiveness it was built for, but
genuinely OSCILLATES (not just noisily flip-flops -- a real deterministic
2-step limit cycle) on some other workloads.

Confirmed offline (cheap, no GPU needed): rolling the trained policy out
deterministically for 100 steps from max clock, the OLD (linear-reward)
checkpoint settles to ONE clock level on every workload tested; the NEW
(log-reward) checkpoint genuinely cycles between two adjacent levels on 4/6
workloads checked by hand. This is the real mechanism -- not measurement
noise (a noise-injection proxy tried first did NOT predict the known
hardware result and was discarded).

Screens two knobs, both already wired into train_unified_v2.Cfg:
  * action_cost  -- penalise every non-HOLD action. Should directly discourage
                    the back-and-forth of a limit cycle (each flip now costs
                    something), independent of the reward curve's shape.
  * reward_power -- interpolates the training reward between the original
                    linear curve (1.0) and the pure log curve (0.0) that
                    caused the problem. Addresses the root cause directly
                    (less over-sharpening near the optimum) at the cost of
                    weakening the compute-bound fix.

For each candidate: trains ONLY on the held-out RTX 4090 (fast screening,
not the full 5-GPU protocol -- that's reserved for whatever candidate wins
here), checks (a) cycle rate across all 12 workloads via the deterministic
rollout test above, (b) whether pt_resnet50_train specifically still settles
at the higher, hardware-validated-safe clock level (not just "doesn't
cycle"), (c) mean steady-state scaled EDP as a sanity check against outright
regression.

    python common/analysis/reward_shape_sweep.py
"""
import os, sys, time
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
HOLDOUT = "nvidia_geforce_rtx_4090"
SEEDS = [0, 1, 2]

CANDIDATES = {
    "baseline_linear":        dict(),  # reward_power=None, use_log_reward=False -- the ORIGINAL, pre-fix checkpoint
    "pure_log (the problem)": dict(use_log_reward=True),  # reproduces the already-tested, already-problematic fix
    "action_cost_0.01":       dict(use_log_reward=True, action_cost=0.01),
    "action_cost_0.02":       dict(use_log_reward=True, action_cost=0.02),
    "action_cost_0.05":       dict(use_log_reward=True, action_cost=0.05),
    "reward_power_0.5":       dict(reward_power=0.5),
    "reward_power_0.25":      dict(reward_power=0.25),
    "reward_power_0.1":       dict(reward_power=0.1),
}

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


def rollout_trace(agent, W, steps=100):
    core, stride = W["core"], W["stride"]
    i = len(core) - 1
    tracked = float(core[i])
    obs0 = P.augmented_obs(W, i, None, True)
    stack = np.repeat(obs0[None, :], 3, axis=0)
    trace = [i]
    for _ in range(steps):
        a = agent.act(stack)
        tracked = float(min(1.0, max(0.0, tracked + (-stride if a == 0 else (stride if a == 1 else 0.0)))))
        j = int(np.argmin(np.abs(core - tracked)))
        obs = P.augmented_obs(W, j, i, True)
        stack = np.concatenate([stack[1:], obs[None, :]], axis=0)
        i = j
        trace.append(i)
    return trace


def _job(t):
    name, cfg_kwargs, seed = t
    cfg = Cfg(use_response=True, episodes=800, **SETTLED, **cfg_kwargs)
    train_gpus = [g for g in FIVE_GPU_FLEET if g != HOLDOUT]
    t0 = time.time()
    net, _ = train(HOLDOUT, cfg, seed=seed, datasets={g: DS[g] for g in train_gpus})
    agent = GreedyAgent(net)

    # (a) cycle check across all 12 workloads on the held-out GPU
    n_cycle = 0
    resnet_settle = None
    for wl, W in sorted(PREP[HOLDOUT].items()):
        trace = rollout_trace(agent, W)
        distinct = len(set(trace[-20:]))
        if distinct > 1:
            n_cycle += 1
        if wl == "pt_resnet50_train":
            resnet_settle = float(W["core"][trace[-1]])

    # (b) sanity: aggregate scaled-EDP metric (unchanged evaluation path)
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
    out = f"{REPO}/docs/results/reward_shape_sweep.csv"
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
