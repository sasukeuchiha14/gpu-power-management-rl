import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
models_dir = os.path.join(gpu_root, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from ddqn_agent import DDQNAgent  
from dvfs_env import LaptopDVFSEnv 

def _resolve(subfolder, filename):
    cand = os.path.join(gpu_root, subfolder, filename)
    if os.path.exists(cand):
        return cand
    if os.path.exists(filename):
        return filename
    return cand

def evaluate_and_plot_forced():
    prep_path = _resolve("data", "preprocessed_dataset.csv")
    weights_path = _resolve("weights", "ddqn_weights.pth")

    env = LaptopDVFSEnv(dataset_path=prep_path)
    input_shape = env.observation_space.shape
    action_size = env.action_space.n
    
    agent = DDQNAgent(input_shape, action_size)
    
    try:
        agent.policy_net.load_state_dict(torch.load(weights_path, map_location=agent.device, weights_only=True))
        agent.policy_net.eval()
        agent.epsilon = 0.0
    except FileNotFoundError:
        print(f"Error: {weights_path} not found.")
        return

    # Reset Env and FORCE THE WORST STATE
    state, _ = env.reset()
    worst_state_idx = env.current_workload_df['EDP'].idxmax()
    env.current_state_row = env.current_workload_df.loc[worst_state_idx]
    env.current_tracked_core = env.current_state_row['Core_F']
    
    single_obs = env._get_single_observation()
    for _ in range(env.stack_size):
        env.frames.append(single_obs)
    state = env._get_stacked_observation()
    
    print(f"Forced starting state to WORST possible EDP on RTX 3050 Ti.")
    
    history_core = []
    history_edp = []
    
    print("\nStarting Evaluation Run...")
    for step in range(30): 
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Denormalize core frequency (500 to 1900 MHz)
        history_core.append(env.current_state_row['Core_F'] * 1400 + 500) 
        history_edp.append(1.0 - reward)
        
        state = next_state
        if terminated or truncated:
            break

    print("Run complete. Generating graph...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(history_core, label="Core Frequency (MHz)", color='blue', linewidth=2, marker='o')
    ax1.set_title("RTX 3050 Ti: AI Dynamically Tuning Core Clock for Efficiency")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(history_edp, label="System EDP (Lower is Better)", color='red', linewidth=2, marker='o')
    ax2.set_title("Energy-Delay Product Plummeting to Optimal Point")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Scaled EDP")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    logs_plots_dir = os.path.join(gpu_root, "logs_and_plots")
    os.makedirs(logs_plots_dir, exist_ok=True)
    out_graph = os.path.join(logs_plots_dir, "rtx3050ti_waterfall_graph.png")
    plt.savefig(out_graph)
    print(f"Graph saved as {out_graph}.")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot_forced()
