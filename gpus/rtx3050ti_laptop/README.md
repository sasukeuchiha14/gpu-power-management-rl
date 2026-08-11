# Target GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (1D Core DVFS)

This directory contains a customized version of the RL DVFS controller framework tailored specifically for the **NVIDIA GeForce RTX 3050 Ti Laptop GPU** (4GB VRAM, 70W TGP cap) under Windows 11.

---

## 🎯 Target System Specifications

* **GPU Target:** NVIDIA GeForce RTX 3050 Ti Laptop GPU.
* **Operating System:** Windows 11 Home / Pro.
* **Memory Constraints:** Optimized for low free RAM (uses standalone Python subprocesses for memory isolation).
* **Thermal Protection:** Cooldown loop (<60°C check) and a 5-second heat-dissipation pause between iterations.
* **Control Dimension:** 1D Core Clock Scaling ($500\text{ MHz} \rightarrow 1900\text{ MHz}$ in 100 MHz steps, 15 discrete levels). Memory clock is fixed at stock 3D speed ($5871\text{ MHz}$) as consumer laptop VRAM frequency locking is driver-restricted.

---

## 📂 Workspace Structure

* `benchmarks/`: Standalone Python PyTorch benchmark scripts (ResNet-50, BERT, MatMul, 3D FFT, Vector Add, Training step, Crypto proxy).
* `data_collection/`: Thermal-aware data collection harness (`generate_dataset.py`).
* `data/`: Raw dataset (`rtx3050ti_dataset.csv`), processed dataset (`preprocessed_dataset.csv`), and scaler parameters (`scaler_params.json`).
* `preprocessing/`: Preprocessor (`preprocess.py`) and scaler exporter (`export_scaler.py`).
* `models/`: 1D Core-tuning Gym environment (`dvfs_env.py`), DDQN architecture (`ddqn_agent.py`), and training loop (`train_ddqn.py`).
* `weights/`: Trained PyTorch network weights (`ddqn_weights.pth`).
* `deployment/`: Real-time Windows deployment daemon (`sim_to_real_deploy.py`) with Administrator privilege checking and 80°C thermal safety override.
* `evaluation/`: Model performance visualizer (`evaluate_baselines.py`, `evaluate_forced.py`) and live A/B hardware power measurement tools (`measure_edp.py`, `plot_edp.py`).
* `logs_and_plots/`: Generated evaluation graphs (`rtx3050ti_proof_graph.png`, `rtx3050ti_waterfall_graph.png`, `edp_comparison.png`).

---

## 🚀 Execution Workflow

### Step 1: Open Administrator Terminal
Open Command Prompt or PowerShell as **Administrator** (Right-click -> *Run as Administrator*):
```cmd
cd gpus\rtx3050ti_laptop
```

### Step 2: Data Collection (Run on Laptop)
Sweep 15 core clock frequencies across 7 benchmarks:
```cmd
python data_collection\generate_dataset.py
```
*(If you need to stop early, press `Ctrl+C`. The script automatically executes `nvidia-smi -rgc` to restore automatic GPU governor scaling).*

### Step 3: Preprocess & Export Scaler (Run on Research PC or Laptop)
```cmd
python preprocessing\preprocess.py
python preprocessing\export_scaler.py
```

### Step 4: Train RL Agent (Run on Research PC or Laptop)
Train the DDQN model in simulation (800 episodes):
```cmd
python models\train_ddqn.py
```

### Step 5: Evaluate Model & Generate Proof Graphs
Generate comparison graphs and recovery waterfall plots for your reports/presentations:
```cmd
python evaluation\evaluate_baselines.py
python evaluation\evaluate_forced.py
```
*(Outputs saved to `logs_and_plots/rtx3050ti_proof_graph.png` and `logs_and_plots/rtx3050ti_waterfall_graph.png`).*

### Step 6: Test Real-Time Deployment
Run the live user-space daemon to observe state reconstruction and decisions:
```cmd
python deployment\sim_to_real_deploy.py
```
* By default, `DRY_RUN = True` is enabled (observe-only mode).
* To actuate real-time hardware clock locks, set `DRY_RUN = False` inside `deployment/sim_to_real_deploy.py`.

### Step 7: Empirical A/B Hardware Power Measurement
To measure live physical power and energy savings between the stock NVIDIA governor and the AI daemon on your laptop:
```cmd
:: 1. Measure Governor Baseline
python evaluation\measure_edp.py governor 10

:: 2. Measure AI Daemon (Run sim_to_real_deploy.py in another terminal first)
python evaluation\measure_edp.py ai 10

:: 3. Generate Comparison Plot
python evaluation\plot_edp.py
```

---

## 🛡️ Safety & Recovery
* **Automatic Clock Restoration:** All deployment and data collection scripts catch exit signals and run `nvidia-smi -i 0 -rgc` in `finally` blocks.
* **Manual Emergency Reset:** If GPU clocks ever remain locked, execute this in an Administrator terminal:
  ```cmd
  nvidia-smi -i 0 -rgc
  ```
* **Hardware Volatility:** Software clock locks do not persist across reboots. Restarting your laptop immediately clears all clock locks.
