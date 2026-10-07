"""
reversal_cost_validate.py -- full 5-GPU, 10-seed offline validation of the
reversal_cost candidate screened in reversal_cost_sweep.py (3-seed screen on
the RTX 4090 only suggested 0.05; it did NOT hold up as the best choice at
full scale -- see below).

Unlike action_cost_fix_validate.py (which only checked the aggregate
steady-state EDP metric at full validation and got burned -- a 3-seed
screening cycle-rate of 0.22 turned into 9/12 cycling on the one seed that
was actually checked), this script checks the CYCLE RATE on every held-out
GPU too, not just the aggregate metric and not just the 4090. That's the
whole point of doing this properly: don't let a small screen's favourable
number stand in for the thing that actually matters.

Swept --reversal-cost over {0.05, 0.10, 0.20, 0.30, 0.40, 0.60}, each at
this full 5-GPU/10-seed scale:

| reversal_cost | cycle rate | ss_mean | ss_std |
|---|---|---|---|
| 0.05 | 0.182 | 0.040 | 0.040 |
| 0.10 | 0.103 | 0.040 | 0.034 |
| 0.20 | 0.055 | 0.055 | 0.054 |
| 0.30 | 0.022 | 0.049 | 0.047 |
| 0.40 | 0.017 | 0.053 | 0.056 |
| 0.60 | 0.007 | 0.060 | 0.055 |

Cycle rate drops monotonically with reversal_cost (nearly eliminated by
0.30-0.40, essentially gone by 0.60 -- three of five GPUs hit exactly zero
cycling across all 10 seeds there). Cost plateaus from 0.20 onward rather
than growing steadily, and a paired comparison (same GPU+seed) shows 0.30
is NOT reliably beaten on cost by 0.40 or 0.60, while matching their cycle
suppression closely (lots of ties, few genuine wins for the larger values)
-- so 0.30 is the selected sweet spot, not the largest value tested. See
common/analysis/reversal_cost_checkpoint.py for the actual deployable
checkpoint and the pt_resnet50_train/4090 safety check.

    python common/analysis/reversal_cost_validate.py --reversal-cost 0.30
"""
import argparse, os, time
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

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


def _job(t):
    gpu, seed, reversal_cost = t
    cfg = Cfg(use_response=True, use_log_reward=True, reversal_cost=reversal_cost,
              episodes=800, **SETTLED)
    net, _ = train(gpu, cfg, seed=seed, datasets={g: DS[g] for g in FIVE_GPU_FLEET})
    agent = GreedyAgent(net)

    n_cycle = 0
    for wl, W in sorted(PREP[gpu].items()):
        trace = rollout_trace(agent, W)
        if len(set(trace[-20:])) > 1:
            n_cycle += 1

    pol = P.p_agent(agent)
    tr, ss, _ = P.evaluate_policy(PREP[gpu], pol, use_response=True)
    return dict(policy=f"log-reward + reversal_cost={reversal_cost}", gpu=gpu, seed=seed,
                n_cycling=n_cycle, n_total=len(PREP[gpu]), traj=tr, ss=ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reversal-cost", type=float, required=True)
    a = ap.parse_args()

    jobs = [(g, s, a.reversal_cost) for g in FIVE_GPU_FLEET for s in range(10)]
    print(f"Training {len(jobs)} jobs (reversal_cost={a.reversal_cost})...", flush=True)
    t0, rows = time.time(), []
    with Pool(min(len(jobs), os.cpu_count() or 8)) as p:
        for i, r in enumerate(p.imap_unordered(_job, jobs)):
            rows.append(r)
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    tag = f"{a.reversal_cost:.2f}".replace(".", "")
    out = f"{REPO}/docs/results/reversal_cost_10seed_{tag}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)

    print("\n=== per-GPU: cycle rate, steady-state scaled EDP ===")
    agg = d.groupby("gpu").agg(
        cycle_rate=("n_cycling", lambda x: (x / d.loc[x.index, "n_total"]).mean()),
        ss_mean=("ss", "mean"), ss_std=("ss", "std"),
    ).round(4)
    print(agg.to_string())
    print(f"\noverall cycle_rate: {(d.n_cycling / d.n_total).mean():.4f}")
    print(f"overall ss mean: {d.ss.mean():.4f}  std: {d.ss.std():.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
