# Target GPU: NVIDIA RTX 4500 Ada Generation (2D DVFS Tuning)

This directory contains the reinforcement learning DVFS controller framework configured for the **NVIDIA RTX 4500 Ada Generation GPU** (24GB VRAM, 210W TGP).

---

## 🎯 Target System Specifications

* **GPU Target:** NVIDIA RTX 4500 Ada Generation Workstation GPU.
* **Environment:** Linux (Ubuntu 20.04/22.04 LTS).
* **Control Dimension:** 2D Co-optimization (Core Frequency & Memory Frequency).
* **Core Frequency Grid:** $1000\text{ MHz} \rightarrow 2950\text{ MHz}$ (40 discrete levels).
* **Memory Frequency Grid:** $5001\text{ MHz} \rightarrow 9501\text{ MHz}$ (10 discrete levels).
* **Search Space Size:** 400 unique frequency pairs per workload.

---

## 📂 Directory Structure

* `benchmarks/`: Rodinia CUDA benchmarks (`gaussian.cu`), custom synthetic AI/Data workloads (`custom_benchmarks/`), and execution scripts.
* `data_collection/`: Profiling harness scripts (`generate_8000_datasets.py`, `generate_300_datasets.py`) for collecting high-resolution telemetry.
* `data/`: CSV datasets (`rtx4500_8000_dataset.csv`, `preprocessed_dataset.csv`) and Min-Max scaler parameters (`scaler_params.json`).
* `preprocessing/`: Feature normalizers and reward scaling functions (`preprocess.py`, `rescale_reward_per_workload.py`).
* `models/`: Gymnasium environment (`dvfs_env_advanced.py`), DDQN architecture (`ddqn_agent.py`), and training loop (`train_ddqn.py`).
* `weights/`: Trained PyTorch network weights (`ddqn_weights.pth`).
* `deployment/`: Real-time userspace control daemon (`sim_to_real_deploy.py`) featuring dual-layer safety overrides.
* `evaluation/`: Hardware evaluation scripts (`measure_edp.py`, `plot_edp.py`, `evaluate_baselines.py`, `evaluate_forced.py`).
* `logs_and_plots/`: Generated plots (`edp_comparison.png`, `review_2_proof_graph.png`, `dqn_waterfall_graph.png`) and execution logs.

---

## 🚀 Execution Workflow

### 1. Data Collection
Run the profiling script to generate the 8,000-sample telemetry dataset across 20 workloads:
```bash
python data_collection/generate_8000_datasets.py
```

### 2. Data Preprocessing
Clean the dataset, calculate per-workload EDP rewards, and export normalization bounds:
```bash
python preprocessing/preprocess.py
python preprocessing/export_scaler.py
```

### 3. Agent Training
Train the Double DQN agent in simulation (800 episodes):
```bash
python models/train_ddqn.py
```

### 4. Real-Time Hardware Deployment
Launch the live control loop daemon (requires `sudo` for `nvidia-smi` clock setting):
```bash
sudo -E python deployment/sim_to_real_deploy.py
```
*(By default `DRY_RUN = False` actuates hardware clocks. Set `DRY_RUN = True` in `sim_to_real_deploy.py` to test in observation-only mode).*

### 5. A/B Hardware EDP Measurement & Evaluation
To run an empirical hardware comparison between the stock NVIDIA governor and the AI daemon:
```bash
# 1. Measure Governor Baseline
sudo -E python evaluation/measure_edp.py governor 20

# 2. Measure AI Daemon (start sim_to_real_deploy.py in another terminal first)
sudo -E python evaluation/measure_edp.py ai 20

# 3. Plot Comparison Graph
python evaluation/plot_edp.py
```
