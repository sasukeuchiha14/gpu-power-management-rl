# GPU DVFS RL Controller: Autonomous GPU Power Management via Deep Reinforcement Learning

> **Proposed Repository Name:** `gpu-dvfs-rl-controller`  
> **Description:** Autonomous, real-time Dynamic Voltage and Frequency Scaling (DVFS) for NVIDIA GPUs using Double Deep Q-Networks (DDQN) to minimize Energy-Delay Product (EDP) across compute- and memory-bound workloads.

---

## 📌 Project Overview

Modern GPUs running AI workloads (ResNet-50, BERT), numerical linear algebra, and data-parallel kernels often consume excessive power under standard static max-clock governors ("race-to-idle"). 

This project implements an end-to-end **Deep Reinforcement Learning (DDQN) framework** that dynamically optimizes GPU Core and Memory clock frequencies in real-time. By continuously monitoring live telemetry via NVIDIA NVML (power, utilization, temperature, frequencies), the RL controller selects clock states that reduce power consumption while maintaining workload throughput, targeting optimal **Energy-Delay Product (EDP)**.

---

## 🌟 Key Architecture & Highlights

1. **Two-Layer Safety Net Architecture:**
   - **Layer 1 (Thermal Safety Override):** Hard temperature limit override (reverts clocks to base speeds if GPU temp exceeds thermal thresholds, e.g., 80°C–85°C).
   - **Layer 2 (Do-No-Harm Performance Arbiter):** A supervisor watching the live EDP. If the AI model degrades performance below the stock governor for $N$ consecutive steps, control temporarily reverts to the stock governor.

2. **Frame Stacking & Domain Randomization:**
   - Observes temporal trends using a 3-frame stack ($S_t = [s_{t-2}, s_{t-1}, s_t]$).
   - Injects Gaussian noise during training to robustly handle live hardware sensor noise (Sim-to-Real transfer).

3. **Multi-Target Hardware Customization:**
   - **NVIDIA RTX 4500 Ada Generation (2D Control):** Co-optimizes Core ($1000\text{ MHz} \rightarrow 2950\text{ MHz}$) and Memory ($5001\text{ MHz} \rightarrow 9501\text{ MHz}$) frequency grids.
   - **NVIDIA RTX 3050 Ti Laptop GPU (1D Control):** Tailored for thermal safety and Core-only frequency tuning ($500\text{ MHz} \rightarrow 1900\text{ MHz}$) with explicit cooling pauses and thermal recovery loops.

---

## 📁 Repository Structure

```
code/
├── .gitignore               # Ignored build outputs, logs, pycache, binaries
├── LICENSE                  # MIT Open Source License
├── README.md                # Root Project Documentation (This File)
├── CONTRIBUTING.md          # Guidelines for contributing and extending
└── gpus/
    ├── rtx4500_ada/         # Linux Research Workstation (RTX 4500 Ada - 2D DVFS)
    │   ├── README.md
    │   ├── benchmarks/      # CUDA & PyTorch benchmarking suite (Rodinia, MatMul, etc.)
    │   ├── data_collection/ # Telemetry profiling harness scripts
    │   ├── data/            # Raw & preprocessed profiling CSV datasets
    │   ├── preprocessing/   # Min-Max normalizer & per-workload reward scaler
    │   ├── models/          # Gymnasium DVFS environments & DDQN agent architecture
    │   ├── weights/         # Trained PyTorch neural network model weights (.pth)
    │   ├── deployment/      # User-space real-time control daemon (sim_to_real_deploy.py)
    │   ├── evaluation/      # A/B testing, EDP measurement & plotting scripts
    │   └── logs_and_plots/  # Evaluation graphs and execution log files
    │
    └── rtx3050ti_laptop/    # Windows Laptop (RTX 3050 Ti - 1D DVFS)
        ├── README.md
        ├── benchmarks/      # Pure-Python PyTorch benchmark workloads
        ├── data_collection/ # Thermal-aware profiling script (generate_dataset.py)
        ├── data/            # Laptop raw/processed datasets & scaler params
        ├── preprocessing/   # Data preprocessor and normalization exporter
        ├── models/          # 1D Core-tuning Gym environment & DDQN agent
        ├── weights/         # Laptop model weights (.pth)
        └── deployment/      # Windows-compatible live deployment daemon
```

---

## 🚀 Quickstart Guide

### 1. Requirements & Dependencies
* **Operating Systems:** Linux (Ubuntu 20.04/22.04) or Windows 11.
* **GPU Drivers:** NVIDIA Driver with NVML support (`pynvml`).
* **Python Environment:** Python 3.10+ with PyTorch (CUDA support), Gymnasium, Pandas, Scikit-learn, Matplotlib.

```bash
pip install torch torchvision pynvml gymnasium pandas scikit-learn matplotlib
```

### 2. Standard Workflow (Data Collection -> Preprocess -> Train -> Deploy)

Select your target hardware under `gpus/`:

#### For RTX 4500 Ada Workstation:
```bash
cd gpus/rtx4500_ada

# 1. Profile Telemetry
python data_collection/generate_8000_datasets.py

# 2. Preprocess & Export Scaler
python preprocessing/preprocess.py
python preprocessing/export_scaler.py

# 3. Train RL Agent
python models/train_ddqn.py

# 4. Deploy Live Controller (Dry Run by default)
sudo python deployment/sim_to_real_deploy.py
```

#### For RTX 3050 Ti Laptop (Windows):
```cmd
cd gpus/rtx3050ti_laptop

:: 1. Profile Telemetry (Run CMD as Administrator)
python data_collection/generate_dataset.py

:: 2. Preprocess & Export Scaler
python preprocessing/preprocess.py
python preprocessing/export_scaler.py

:: 3. Train RL Agent
python models/train_ddqn.py

:: 4. Deploy Live Controller (Dry Run by default)
python deployment/sim_to_real_deploy.py
```

---

## 📊 Results Summary

Under real-world workloads, the DDQN controller achieves:
* **~18% to 26% Energy Savings** compared to standard GPU race-to-idle governors.
* **Reduced Peak Temperatures** by keeping frequencies dynamically scaled to workload intensity.
* **Sub-50ms Decision Latency**, enabling seamless user-space execution without frame drops or throughput stalls.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
