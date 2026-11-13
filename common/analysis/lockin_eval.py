"""
lockin_eval.py -- offline test of a DEPLOYMENT-SIDE cycle lock-in rule, no retraining.

Observation from dwell_hw_trials: on compute-bound work the cycling policies still save
~5% energy at 1-3% slowdown; what hurts is the oscillation itself. Rule: if the last K
chosen clock levels alternate between exactly two distinct levels (a limit cycle), stop
consulting the policy and park at one of the two levels for the rest of the run.
  lock_edp : park at the level with lower live EDP (P*t^2, both measurable in deploy_live)
  lock_low : park at the lower clock of the pair
Applied as a post-process on the deterministic greedy rollout (the policy is not consulted
after the lock, so this is exactly equivalent to applying it online).

    python common/analysis/lockin_eval.py --seeds 3
"""
import argparse, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from _common import REPO
from dataset_view import discover_core_only
import eval_policies as P
from train_unified_v2 import train, Cfg, GreedyAgent
from reward_shape_sweep import rollout_trace
from dwell_screen import CANDIDATES, SETTLED, FLEET, COMPUTE

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}
K = 6


def lock(trace, W, mode):
    out = list(trace)
    for t in range(K, len(trace)):
        win = trace[t - K + 1:t + 1]
        vals = sorted(set(win))
        if len(vals) == 2 and sum(win[i] != win[i + 1] for i in range(K - 1)) >= 3:
            park = min(vals, key=lambda j: W["edp"][j]) if mode == "lock_edp" else vals[0]
            out[t:] = [park] * (len(trace) - t)
            return out
    return out


def metrics(tr, W):
    tail = np.array(tr[-20:]); top = len(W["core"]) - 1
    return dict(cycle=int(len(set(tail)) > 1),
                p_save=1 - W["power_w"][tail].mean() / W["power_w"][top],
                slow=W["time_s"][tail].mean() / W["time_s"][top] - 1,
                edp=W["sedp"][tail].mean())


def _job(t):
    name, holdout, seed = t
    cfg = Cfg(use_response=True, use_log_reward=True, episodes=800, **SETTLED, **CANDIDATES[name])
    net, _ = train(holdout, cfg, seed=seed, datasets={g: DS[g] for g in FLEET})
    ag = GreedyAgent(net); rows = []
    for wl, W in sorted(PREP[holdout].items()):
        raw = rollout_trace(ag, W, steps=100)
        for mode in ("raw", "lock_edp", "lock_low"):
            tr = raw if mode == "raw" else lock(raw, W, mode)
            rows.append(dict(cand=name, mode=mode, gpu=holdout, seed=seed, wl=wl,
                             regime="compute" if wl in COMPUTE else "other", **metrics(tr, W)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", default="pure_log,rc0.10,rc0.30,win2_hold.02")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    jobs = [(n, g, s) for n in a.only.split(",") for g in FLEET for s in range(a.seeds)]
    print(len(jobs), "jobs", flush=True); t0, rows = time.time(), []
    with Pool(a.procs) as p:
        for i, r in enumerate(p.imap_unordered(_job, jobs)):
            rows += r
            if (i + 1) % 6 == 0: print(f"  {i+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows); d.to_csv(f"{REPO}/docs/results/lockin_eval.csv", index=False)
    for rg in ("compute", "other"):
        print(f"\n== {rg} ==")
        print(d[d.regime == rg].groupby(["cand", "mode"]).agg(cycle=("cycle", "mean"), p_save=("p_save", "mean"),
              slow=("slow", "mean"), edp=("edp", "mean")).round(3).to_string())
    print("\n== pt_resnet50_train ==")
    print(d[d.wl == "pt_resnet50_train"].groupby(["cand", "mode"]).agg(cycle=("cycle", "mean"),
          p_save=("p_save", "mean"), slow=("slow", "mean"), edp=("edp", "mean")).round(3).to_string())
