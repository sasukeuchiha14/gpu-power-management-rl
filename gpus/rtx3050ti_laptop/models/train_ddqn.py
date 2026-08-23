import numpy as np
import torch
import os
from dvfs_env import LaptopDVFSEnv
from ddqn_agent import DDQNAgent

def _get_weights_path(filename="ddqn_weights.pth"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
    weights_dir = os.path.join(gpu_root, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    return os.path.join(weights_dir, filename)

def train_agent(episodes=800):
    env = LaptopDVFSEnv(dataset_path="preprocessed_dataset.csv")
    input_shape = env.observation_space.shape
    action_size = env.action_space.n
    
    agent = DDQNAgent(input_shape, action_size)
    agent.epsilon_decay = 0.99
    
    print("Starting Laptop DDQN Core Clock Tuning Training Phase...")
    print(f"State Shape: {input_shape} | Action Space: {action_size} (0: Dec, 1: Inc, 2: Hold)")
    
    reward_history = []
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for time_step in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        # Target network sync
        if e % 5 == 0:
            agent.update_target_network()
            
        # Decay Epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        reward_history.append(total_reward)
        
        if (e + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode: {e+1}/{episodes} | Avg Reward (Last 10): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    weights_out = _get_weights_path("ddqn_weights.pth")
    print("\nTraining Complete! Saving policy weights...")
    torch.save(agent.policy_net.state_dict(), weights_out)

    history_out = weights_out.replace(".pth", "_reward_history.npy")
    np.save(history_out, np.array(reward_history))

    print(f"Saved as '{weights_out}'. Ready for deployment!")
    print(f"Saved training curve to '{history_out}'.")

if __name__ == "__main__":
    train_agent()
