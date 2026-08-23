"""
evaluate_holdout.py -- the actual leave-one-GPU-out RESULT (NORTHSTAR.md row 4).

train_unified.py --holdout <gpu> only trains; it never touches the held-out
card's data. This script takes that trained policy and tests it against the
held-out GPU's OWN profiled operating points -- data the policy never saw
during training -- and compares it to the static max-clock governor
baseline. That comparison is the generalisation claim.

    python3 evaluate_holdout.py --holdout nvidia_geforce_rtx_4090
    python3 evaluate_holdout.py --holdout nvidia_geforce_rtx_4090 --weights <path>

Same evaluation shape as gpus/rtx3050ti_laptop/evaluation/evaluate_baselines.py
(governor = always hold at max clock; AI = greedy trained policy), generalised
to run over EVERY workload on the held-out card rather than one, since the
whole point here is a fleet-wide generalisation number, not a single proof plot.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_common_dir = os.path.dirname(os.path.abspath(__file__))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)
_laptop_models_dir = os.path.abspath(
    os.path.join(_common_dir, "..", "gpus", "rtx3050ti_laptop", "models")
)
if _laptop_models_dir not in sys.path:
    sys.path.insert(0, _laptop_models_dir)

from unified_env import UnifiedDVFSEnv, discover_datasets, _infer_core_stride  # noqa: E402
from ddqn_agent import DDQNAgent  # noqa: E402

STEPS = 50
SEED = 42


def _force_start(env, gpu_tag, workload_df, start_row):
    env.current_step = 0
    env.current_gpu_tag = gpu_tag
    env.current_workload_df = workload_df
    env.current_state_row = start_row
    env.current_tracked_core = float(start_row["core_frac"])
    env.core_stride = _infer_core_stride(workload_df)
    env.frames.clear()
    single = env._get_single_observation()
    for _ in range(env.stack_size):
        env.frames.append(single)
    return env._get_stacked_observation()


def _run_governor(env, gpu_tag, workload_df, start_row, steps):
    _force_start(env, gpu_tag, workload_df, start_row)
    hist = []
    for _ in range(steps):
        env.current_tracked_core = 1.0  # max core clock, held
        _, reward, _, _, _ = env.step(2)  # HOLD
        hist.append(1.0 - reward)  # scaled EDP (lower is better)
    return hist


def _run_agent(env, agent, gpu_tag, workload_df, start_row, steps):
    state = _force_start(env, gpu_tag, workload_df, start_row)
    hist = []
    for _ in range(steps):
        action = agent.act(state)
        state, reward, _, _, _ = env.step(action)
        hist.append(1.0 - reward)
    return hist


def evaluate(holdout_gpu, weights_path=None, steps=STEPS, seed=SEED, out_csv=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_datasets = discover_datasets()
    if holdout_gpu not in all_datasets:
        raise SystemExit(f"'{holdout_gpu}' not among discovered GPUs: {sorted(all_datasets.keys())}")

    # Evaluation env restricted to ONLY the held-out card's own data -- the
    # policy must have never trained on any row that can appear here.
    holdout_df = all_datasets[holdout_gpu]
    env = UnifiedDVFSEnv(datasets={holdout_gpu: holdout_df})

    if weights_path is None:
        weights_path = os.path.join(_common_dir, "weights", f"unified_ddqn_holdout_{holdout_gpu}.pth")
    agent = DDQNAgent(env.observation_space.shape, env.action_space.n)
    agent.policy_net.load_state_dict(torch.load(weights_path, map_location=agent.device, weights_only=True))
    agent.policy_net.eval()
    agent.epsilon = 0.0
    print(f"Loaded holdout-trained policy from {weights_path}")

    workloads = sorted(holdout_df["Workload_Name"].unique())
    print(f"Evaluating '{holdout_gpu}' on its own {len(workloads)} workload(s), "
          f"{steps} steps each, from an identical start state per workload.\n")

    rows = []
    for wl in workloads:
        wdf = holdout_df[holdout_df["Workload_Name"] == wl].reset_index(drop=True)
        start_row = wdf.sample(1, random_state=seed).iloc[0]

        gov_hist = _run_governor(env, holdout_gpu, wdf, start_row, steps)
        ai_hist = _run_agent(env, agent, holdout_gpu, wdf, start_row, steps)

        gov_mean = float(np.mean(gov_hist))
        ai_mean = float(np.mean(ai_hist))
        improvement_pct = 100.0 * (gov_mean - ai_mean) / gov_mean if gov_mean > 0 else float("nan")
        rows.append({
            "workload": wl, "governor_edp": gov_mean, "ai_edp": ai_mean,
            "improvement_pct": improvement_pct, "ai_beats_governor": ai_mean < gov_mean,
        })
        print(f"  {wl:<24} governor={gov_mean:.4f}  AI={ai_mean:.4f}  "
              f"{'better' if ai_mean < gov_mean else 'WORSE':<7} ({improvement_pct:+.1f}%)")

    results = pd.DataFrame(rows)
    n_better = int(results["ai_beats_governor"].sum())
    mean_gov = results.governor_edp.mean()
    mean_ai = results.ai_edp.mean()
    # NOT results.improvement_pct.mean() -- averaging per-workload PERCENTAGES
    # is misleading here: several workloads have a governor EDP already very
    # close to 0 (max clock happens to be near that workload's own optimum),
    # so even a tiny absolute miss there reads as a huge percentage (observed
    # up to -2957% on one workload during real runs, 25 Aug 2026). The
    # decision-relevant number is the reduction on the MEAN of the raw EDP
    # values, which is not distorted by near-zero denominators.
    mean_reduction_pct = 100.0 * (mean_gov - mean_ai) / mean_gov if mean_gov > 0 else float("nan")
    print(f"\n{holdout_gpu}: AI beat the governor on {n_better}/{len(workloads)} workloads.")
    print(f"Mean governor EDP: {mean_gov:.4f}")
    print(f"Mean AI EDP:       {mean_ai:.4f}")
    print(f"EDP reduction (on the means, NOT mean-of-percentages): {mean_reduction_pct:+.1f}%")
    print(f"(per-workload improvement_pct in the saved CSV is informative per-row but NOT meant "
          f"to be averaged -- see comment above)")

    if out_csv is None:
        out_dir = os.path.join(_common_dir, "holdout_results")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{holdout_gpu}.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    results.to_csv(out_csv, index=False)
    print(f"\nSaved per-workload results -> {out_csv}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", required=True, help="GPU folder name, e.g. nvidia_geforce_rtx_4090")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=SEED,
                     help="seed for the fixed evaluation start-state per workload (NOT the "
                          "training seed) -- kept fixed across a training-seed sweep so results "
                          "are only comparing trained-policy quality, not eval start-state luck")
    ap.add_argument("--out-csv", default=None, help="override the output CSV path")
    args = ap.parse_args()
    evaluate(args.holdout, weights_path=args.weights, steps=args.steps, seed=args.seed, out_csv=args.out_csv)


if __name__ == "__main__":
    main()
