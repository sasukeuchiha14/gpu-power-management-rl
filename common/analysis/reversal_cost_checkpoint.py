"""
reversal_cost_checkpoint.py -- trains and saves the actual deployable
checkpoint for the winning reversal_cost candidate from
reversal_cost_validate.py's sweep (0.05/0.10/0.20/0.30/0.40/0.60, full
5-GPU leave-one-GPU-out, 10 seeds each), same protocol as every other
deployed checkpoint this project has produced (holdout=RTX 4090, seed=0).

Also checks the one thing that matters most beyond the aggregate metrics:
does this candidate still settle pt_resnet50_train/4090 at a safe clock
(the exact hardware-validated case the log-reward fix was built for -- see
context.md's Phase 2 hardware section)? A candidate that fixes cycling by
making the policy too conservative to move at all would silently regress
this.

    python common/analysis/reversal_cost_checkpoint.py --reversal-cost 0.40
"""
import argparse, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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
HOLDOUT = "nvidia_geforce_rtx_4090"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reversal-cost", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    DS = discover_core_only()
    cfg = Cfg(use_response=True, use_log_reward=True, reversal_cost=a.reversal_cost,
              episodes=800, **SETTLED)
    net, _ = train(HOLDOUT, cfg, seed=a.seed, datasets={g: DS[g] for g in FIVE_GPU_FLEET})
    agent = GreedyAgent(net)

    prep = {wl: W for wl, W in P.prepare(DS[HOLDOUT]).items()}
    n_cycle, resnet_settle = 0, None
    for wl, W in sorted(prep.items()):
        trace = rollout_trace(agent, W)
        if len(set(trace[-20:])) > 1:
            n_cycle += 1
        if wl == "pt_resnet50_train":
            resnet_settle = float(W["core"][trace[-1]])

    pol = P.p_agent(agent)
    _, ss, _ = P.evaluate_policy(prep, pol, use_response=True)

    tag = f"reversal_cost_{a.reversal_cost:.2f}".replace(".", "")
    seed_suffix = "" if a.seed == 0 else f"_seed{a.seed}"
    out = f"{REPO}/common/weights/{tag}_fix_holdout_{HOLDOUT}{seed_suffix}.pth"
    torch.save(net.state_dict(), out)
    print(f"cycle_rate={n_cycle}/{len(prep)}  pt_resnet50_train_settle={resnet_settle:.3f}  "
          f"ss={ss:.4f}")
    print(f"(for reference: 0.737 = old linear checkpoint (unsafe), "
          f"0.866 = pure-log checkpoint (safe but cycles))")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
