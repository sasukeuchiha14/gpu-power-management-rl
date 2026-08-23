"""
train_unified.py -- train ONE DDQN policy across every discovered GPU's data
at once. This is the step NORTHSTAR.md scorecard row 4 (held-out GPU
generalisation) depends on; nothing before this could do it.

    python3 train_unified.py                              # train on everything found
    python3 train_unified.py --holdout rtx3050ti_laptop    # leave-one-GPU-out
    python3 train_unified.py --episodes 1200 --seed 42

After training with --holdout, evaluate the saved policy against the held-out
card's OWN data (it was never trained on it) -- that comparison is the actual
generalisation result. This script only trains; it doesn't run that
evaluation itself.

Reuses DDQNAgent/QNetwork from gpus/rtx3050ti_laptop/models/ddqn_agent.py
unmodified -- the network only depends on input_shape/action_size, nothing
in it is card-specific, so there's no reason to duplicate it here.
"""
import argparse
import os
import sys

import numpy as np
import torch

_common_dir = os.path.dirname(os.path.abspath(__file__))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)

_laptop_models_dir = os.path.abspath(
    os.path.join(_common_dir, "..", "gpus", "rtx3050ti_laptop", "models")
)
if _laptop_models_dir not in sys.path:
    sys.path.insert(0, _laptop_models_dir)

from unified_env import UnifiedDVFSEnv, discover_datasets  # noqa: E402
from ddqn_agent import DDQNAgent  # noqa: E402


def train(episodes=800, holdout_gpu=None, seed=None, out_path=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    datasets = discover_datasets()
    if not datasets:
        raise SystemExit(
            "No 6-feature datasets found under gpus/*/data/preprocessed_dataset.csv. "
            "Run that GPU's preprocess.py first."
        )
    print(f"Discovered {len(datasets)} GPU dataset(s): {sorted(datasets.keys())}")
    if holdout_gpu:
        if holdout_gpu not in datasets:
            raise SystemExit(
                f"--holdout '{holdout_gpu}' is not among the discovered GPUs: "
                f"{sorted(datasets.keys())}"
            )
        print(f"Holding out '{holdout_gpu}' -- training on the rest only, "
              f"for a leave-one-GPU-out test.")

    env = UnifiedDVFSEnv(datasets=datasets, holdout_gpu=holdout_gpu)
    print(f"Training GPUs ({len(env.gpu_tags)}): {env.gpu_tags}")
    print(f"State shape: {env.observation_space.shape} | Actions: {env.action_space.n}")

    agent = DDQNAgent(env.observation_space.shape, env.action_space.n)
    agent.epsilon_decay = 0.99

    reward_history = []
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        for _ in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        if e % 5 == 0:
            agent.update_target_network()
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        reward_history.append(total_reward)
        if (e + 1) % 10 == 0:
            avg = np.mean(reward_history[-10:])
            print(f"Episode {e+1}/{episodes} | Avg reward (last 10): {avg:.2f} "
                  f"| epsilon: {agent.epsilon:.3f}")

    weights_dir = os.path.join(_common_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    if out_path is None:
        suffix = f"_holdout_{holdout_gpu}" if holdout_gpu else ""
        out_path = os.path.join(weights_dir, f"unified_ddqn{suffix}.pth")
    torch.save(agent.policy_net.state_dict(), out_path)

    history_path = out_path.replace(".pth", "_reward_history.npy")
    np.save(history_path, np.array(reward_history))

    print(f"\nSaved policy weights   -> {out_path}")
    print(f"Saved reward history   -> {history_path}")
    return out_path, reward_history


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--holdout", default=None,
                    help="GPU folder name (e.g. rtx3050ti_laptop) to exclude "
                         "from training, for a leave-one-GPU-out test")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="Override output weights path")
    args = ap.parse_args()
    train(episodes=args.episodes, holdout_gpu=args.holdout, seed=args.seed, out_path=args.out)


if __name__ == "__main__":
    main()
