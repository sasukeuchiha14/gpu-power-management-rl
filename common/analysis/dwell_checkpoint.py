"""
dwell_checkpoint.py -- trains the deployable checkpoint for the dwell-window +
hold-bonus candidate from dwell_screen.py (reversal_cost=0.30, reversal_window=2,
hold_bonus=0.02), holdout=RTX 4090, same protocol as reversal_cost_checkpoint.py.

    python common/analysis/dwell_checkpoint.py --seed 0
"""
import argparse, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)

from _common import REPO
from dataset_view import discover_core_only
import eval_policies as P
from train_unified_v2 import train, Cfg, GreedyAgent
from reward_shape_sweep import rollout_trace

FLEET = ["nvidia_geforce_rtx_3060", "nvidia_geforce_rtx_3070", "nvidia_geforce_rtx_3080",
         "nvidia_geforce_rtx_3090", "nvidia_geforce_rtx_4090"]
HOLDOUT = "nvidia_geforce_rtx_4090"

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
DS = discover_core_only()
cfg = Cfg(use_response=True, use_log_reward=True, reversal_cost=0.30, reversal_window=2,
          hold_bonus=0.02, episodes=800, gamma=0.5, double=False, lr=3e-4, hidden=128, huber=True)
net, _ = train(HOLDOUT, cfg, seed=a.seed, datasets={g: DS[g] for g in FLEET})
ag = GreedyAgent(net)
prep = P.prepare(DS[HOLDOUT])
n_cycle = sum(len(set(rollout_trace(ag, W)[-20:])) > 1 for W in prep.values())
settle = float(prep["pt_resnet50_train"]["core"][rollout_trace(ag, prep["pt_resnet50_train"])[-1]])
suffix = "" if a.seed == 0 else f"_seed{a.seed}"
out = f"{REPO}/common/weights/dwell_win2_hold02_holdout_{HOLDOUT}{suffix}.pth"
torch.save(net.state_dict(), out)
print(f"seed={a.seed} cycle={n_cycle}/{len(prep)} resnet_settle={settle:.3f} saved -> {out}")
