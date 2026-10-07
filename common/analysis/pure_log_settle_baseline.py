"""
pure_log_settle_baseline.py -- full 5-GPU/10-seed resnet_settle breakdown for
the PURE log-reward checkpoint (no reversal_cost), needed to tell whether
reversal_cost=0.30's higher per-GPU settle points
(reversal_cost_validate.py) are something it CAUSES, or something already
there under plain log-reward training that just hadn't been measured on
these 4 GPUs before (context.md, NORTHSTAR row 11, 7 Oct 2026 -- the
0.866-safe-settling claim for the log-reward checkpoint had only ever been
measured on a single RTX 4090 seed).

RESULT: the settle point rises on EVERY GPU under reversal_cost=0.30
relative to this baseline (e.g. RTX 3090: 0.797 here vs 1.000 with
reversal_cost) -- a real, systematic effect of the fix, not something
already present. See reversal_cost_validate.py's docstring for the full
comparison table.

    python common/analysis/pure_log_settle_baseline.py
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

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


def _job(t):
    gpu, seed = t
    cfg = Cfg(use_response=True, use_log_reward=True, episodes=800, **SETTLED)
    net, _ = train(gpu, cfg, seed=seed, datasets={g: DS[g] for g in FIVE_GPU_FLEET})
    agent = GreedyAgent(net)

    n_cycle, resnet_settle = 0, None
    for wl, W in sorted(PREP[gpu].items()):
        trace = rollout_trace(agent, W)
        if len(set(trace[-20:])) > 1:
            n_cycle += 1
        if wl == "pt_resnet50_train":
            resnet_settle = float(W["core"][trace[-1]])
    return dict(gpu=gpu, seed=seed, n_cycling=n_cycle, n_total=len(PREP[gpu]),
                resnet_settle=resnet_settle)


def main():
    jobs = [(g, s) for g in FIVE_GPU_FLEET for s in range(10)]
    t0, rows = time.time(), []
    with Pool(min(len(jobs), os.cpu_count() or 8)) as p:
        for i, r in enumerate(p.imap_unordered(_job, jobs)):
            rows.append(r)
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    out = f"{REPO}/docs/results/pure_log_resnet_settle_10seed.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)
    agg = d.groupby("gpu").agg(
        cycle_rate=("n_cycling", lambda x: (x / d.loc[x.index, "n_total"]).mean()),
        resnet_settle_mean=("resnet_settle", "mean"),
    ).round(4)
    print(agg.to_string())
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
