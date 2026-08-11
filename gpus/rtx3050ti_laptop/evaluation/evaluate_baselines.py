import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
models_dir = os.path.join(gpu_root, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from ddqn_agent import DDQNAgent
from dvfs_env import LaptopDVFSEnv

TARGET_WORKLOAD = None
SEED = 42
STEPS = 50
PATIENCE = 3
COOLDOWN = 5

def _resolve(subfolder, filename):
    cand = os.path.join(gpu_root, subfolder, filename)
    if os.path.exists(cand):
        return cand
    if os.path.exists(filename):
        return filename
    return cand

def force_start(env, workload_df, start_row):
    env.current_step = 0
    env.current_workload_df = workload_df
    env.current_state_row = start_row
    env.current_tracked_core = float(start_row['Core_F'])
    env.frames.clear()
    single = env._get_single_observation()
    for _ in range(env.stack_size):
        env.frames.append(single)
    return env._get_stacked_observation()

def run_agent(env, agent, workload_df, start_row, steps):
    state = force_start(env, workload_df, start_row)
    hist = []
    for _ in range(steps):
        action = agent.act(state)
        state, reward, _, _, _ = env.step(action)
        hist.append(1.0 - reward)
    return hist

def run_governor(env, workload_df, start_row, steps):
    force_start(env, workload_df, start_row)
    hist = []
    for _ in range(steps):
        env.current_tracked_core = 1.0  # Max core clock (governor default)
        _, reward, _, _, _ = env.step(2) # Action 2 = Hold
        hist.append(1.0 - reward)
    return hist

def run_arbitrated(env, agent, workload_df, start_row, steps, gov_edp):
    state = force_start(env, workload_df, start_row)
    hist, mode, bad, cd = [], "AI", 0, 0
    reverts = 0
    for _ in range(steps):
        if mode == "AI":
            action = agent.act(state)
            state, reward, _, _, _ = env.step(action)
            edp = 1.0 - reward
            bad = bad + 1 if edp > gov_edp else 0
            if bad >= PATIENCE:
                mode, cd, reverts = "GOV", COOLDOWN, reverts + 1
        else:
            env.current_tracked_core = 1.0
            state, reward, _, _, _ = env.step(2)
            edp = 1.0 - reward
            cd -= 1
            if cd <= 0:
                mode, bad = "AI", 0
        hist.append(edp)
    return hist, reverts

def evaluate():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    prep_path = _resolve("data", "preprocessed_dataset.csv")
    weights_path = _resolve("weights", "ddqn_weights.pth")

    env = LaptopDVFSEnv(dataset_path=prep_path)
    agent = DDQNAgent(env.observation_space.shape, env.action_space.n)
    try:
        agent.policy_net.load_state_dict(
            torch.load(weights_path, map_location=agent.device, weights_only=True))
        agent.policy_net.eval()
        agent.epsilon = 0.0
        print(f"Loaded trained DDQN weights from {weights_path}.")
    except FileNotFoundError:
        print(f"Warning: {weights_path} not found. Comparing an UNTRAINED agent.")

    env.reset()
    if TARGET_WORKLOAD is not None:
        env.current_workload_df = env.df[env.df[TARGET_WORKLOAD] == 1.0].copy()
    workload_df = env.current_workload_df.copy()
    start_row = workload_df.sample(1, random_state=SEED).iloc[0]
    active = [c for c in env.workload_cols if start_row[c] == 1.0]
    wl_name = active[0].replace("Workload_Name_", "") if active else "unknown"
    print(f"Evaluating RTX 3050 Ti Laptop on workload '{wl_name}' from an identical start state.\n")

    gov = run_governor(env, workload_df, start_row, STEPS)
    gov_edp = float(np.mean(gov))
    ai = run_agent(env, agent, workload_df, start_row, STEPS)
    arb, reverts = run_arbitrated(env, agent, workload_df, start_row, STEPS, gov_edp)

    print(f"  Governor mean EDP      : {gov_edp:.4f}")
    print(f"  Raw DDQN mean EDP      : {np.mean(ai):.4f}")
    print(f"  Arbitrated mean EDP    : {np.mean(arb):.4f}  (governor takeovers: {reverts})")
    print(f"  --> Arbiter guarantees the controller never trails the governor for long.\n")

    logs_plots_dir = os.path.join(gpu_root, "logs_and_plots")
    os.makedirs(logs_plots_dir, exist_ok=True)
    out_graph = os.path.join(logs_plots_dir, "rtx3050ti_proof_graph.png")

    plt.figure(figsize=(10, 6))
    plt.plot(gov, label="Static Max-Clock Governor (race-to-idle)",
             color='red', linestyle='--', linewidth=2)
    plt.plot(ai, label="DDQN AI (raw)", color='orange', linewidth=2, alpha=0.9)
    plt.plot(arb, label="DDQN AI + Do-No-Harm Arbiter", color='green', linewidth=3)
    plt.title(f"RTX 3050 Ti: AI Optimization vs. Standard GPU Governor ({wl_name})")
    plt.xlabel("Execution Time Steps")
    plt.ylabel("Per-Workload Energy-Delay Product (Lower is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_graph, dpi=300)
    print(f"Graph saved as {out_graph}.")
    plt.show()

if __name__ == "__main__":
    evaluate()
