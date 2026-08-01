import os
import numpy as np
import torch
from dvfs_env_advanced import AdvancedDVFSEnv
from ddqn_agent import DDQNAgent

def _get_weights_path(filename="ddqn_weights.pth"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
    weights_dir = os.path.join(gpu_root, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    return os.path.join(weights_dir, filename)

def train_agent(episodes=800):
    env = AdvancedDVFSEnv(dataset_path="preprocessed_dataset.csv")
    input_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = DDQNAgent(input_shape, action_size)
    agent.epsilon_decay = 0.99

    print("Starting DDQN Training Phase...")
    print(f"State Shape: {input_shape} | Action Space: {action_size}")
    print(f"Episodes: {episodes} | decay: {agent.epsilon_decay} | eps_min: {agent.epsilon_min}")

    reward_history = []

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
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
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode: {e+1}/{episodes} | Avg Reward (Last 10): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    weights_out = _get_weights_path("ddqn_weights.pth")
    print(f"\nFinal epsilon: {agent.epsilon:.3f}  (target ~0.05)")
    print(f"Training Complete! Saving weights to {weights_out} ...")
    torch.save(agent.policy_net.state_dict(), weights_out)
    print("Saved.")

if __name__ == "__main__":
    train_agent()
