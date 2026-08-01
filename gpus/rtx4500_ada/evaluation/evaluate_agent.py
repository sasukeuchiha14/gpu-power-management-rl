import os
import sys
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
models_dir = os.path.join(gpu_root, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from dqn_agent import DQN  # Imports the brain structure
from dvfs_env import DVFSEnv # Imports custom world

def _resolve(subfolder, filename):
    cand = os.path.join(gpu_root, subfolder, filename)
    if os.path.exists(cand):
        return cand
    if os.path.exists(filename):
        return filename
    return cand

def evaluate_and_plot():
    prep_path = _resolve("data", "preprocessed_dataset.csv")
    weights_path = _resolve("weights", "dvfs_dqn_weights.pth")

    # 1. Setup the Environment and Load the Brain
    env = DVFSEnv(dataset_path=prep_path)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_size, action_size).to(device)
    
    # Load the trained weights!
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval() # Set to evaluation mode (no more random exploration)
        print(f"Trained weights loaded successfully from {weights_path}.")
    except FileNotFoundError:
        print(f"Error: Could not find {weights_path}. Did you save it?")
        return

    # 2. Run a single evaluation episode
    state, _ = env.reset()
    
    history_core = []
    history_mem = []
    history_edp = []
    
    print("\nStarting Evaluation Run...")
    for step in range(100):
        # The agent acts completely greedily (no exploration, just pure intelligence)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.argmax(model(state_tensor)[0]).item()
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Log the raw metrics (un-scaling them slightly for the graph)
        # Note: state[0] is Core, state[1] is Mem. They are scaled 0-1.
        history_core.append(state[0] * 1950 + 1000) # Roughly mapping back to 1000-2950MHz
        history_mem.append(state[1] * 4500 + 5001)  # Roughly mapping back to 5001-9501MHz
        history_edp.append(1.0 - reward) # Getting the scaled EDP back
        
        state = next_state
        if terminated or truncated:
            break

    # 3. Generate the Presentation Graphs
    print("Run complete. Generating graphs...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Graph 1: Frequency Scaling over time
    ax1.plot(history_core, label="Core Frequency (MHz)", color='blue', linewidth=2)
    ax1.plot(history_mem, label="Memory Frequency (MHz)", color='green', linewidth=2)
    ax1.set_title("DQN Agent: Dynamic Frequency Scaling Decisions")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.legend()
    ax1.grid(True)
    
    # Graph 2: Energy-Delay Product (EDP) over time
    ax2.plot(history_edp, label="System EDP (Lower is Better)", color='red', linewidth=2)
    ax2.set_title("Energy-Delay Product Optimization")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Scaled EDP")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    out_graph = _resolve("logs_and_plots", "dqn_evaluation_results.png")
    plt.savefig(out_graph)
    print(f"Graph saved as {out_graph}.")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()
