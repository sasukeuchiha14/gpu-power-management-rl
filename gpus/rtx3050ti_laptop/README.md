# Target GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (1D Core DVFS)

This directory contains a customized version of the RL DVFS controller framework tailored specifically for the **NVIDIA GeForce RTX 3050 Ti Laptop GPU** (4GB VRAM, 70W TGP cap) under Windows 11.

---

## 🎯 Target System Specifications

* **GPU Target:** NVIDIA GeForce RTX 3050 Ti Laptop GPU.
* **Operating System:** Windows 11 Home / Pro.
* **Memory Constraints:** Optimized for low free RAM (uses standalone Python subprocesses for memory isolation).
* **Thermal Protection:** Cooldown loop (<70°C check) and a 5-second heat-dissipation pause between iterations.
* **Control Dimension:** 1D Core Clock Scaling ($500\text{ MHz} \rightarrow 1900\text{ MHz}$ in 100 MHz steps, 15 discrete levels). Memory clock is fixed at stock 3D speed ($5871\text{ MHz}$) as consumer laptop VRAM frequency locking is driver-restricted.

---

## 📂 Workspace Structure

* `benchmarks/`: Standalone Python PyTorch benchmark scripts. **10 trained workloads** (ResNet-50, BERT,
  MobileNetV2, MatMul, 3D FFT, Vector Add, Elementwise chain, ResNet-50 training step, Crypto proxy, LSTM)
  plus `zero_shot_workloads/` (ViT, SqueezeNet, Sort+Reduce -- deliberately *not* in `BENCHMARKS`, for
  testing generalization to unseen apps; see that folder's own README).
* `data_collection/`: Thermal-aware data collection harness (`generate_dataset.py`), cooldown gate at
  **70°C**.
* `data/`: Raw dataset (`rtx3050ti_dataset.csv`), processed dataset (`preprocessed_dataset.csv`), and
  scaler parameters (`scaler_params.json`). **Stale as of this update** -- re-run Step 2 below to pick up
  the 3 new trained workloads and the `pt_fft` timing fix (see Known Issues).
* `preprocessing/`: Preprocessor (`preprocess.py`) and scaler exporter (`export_scaler.py`).
* `models/`: 1D Core-tuning Gym environment (`dvfs_env.py`), DDQN architecture (`ddqn_agent.py`), and training loop (`train_ddqn.py`).
* `weights/`: Trained PyTorch network weights (`ddqn_weights.pth`). **Stale** -- retrain (Step 4) once the
  dataset above is regenerated; the state shape hasn't changed (still 6 universal features) so this is a
  drop-in retrain, no code changes needed.
* `deployment/`: Real-time Windows deployment daemon (`sim_to_real_deploy.py`) with Administrator privilege
  checking, in-process NVML clock control, and **90°C** thermal safety override.
* `evaluation/`: Model performance visualizer (`evaluate_baselines.py`, `evaluate_forced.py`), live A/B
  hardware power measurement tools (`measure_edp.py`, `plot_edp.py` -- both now take a workload name as an
  argument, e.g. `measure_edp.py governor 10 pt_lstm`), and `summarize_tests.py` (rebuilds a results
  table + bar chart across every workload you've tested).
* `logs_and_plots/`: Generated evaluation graphs (`rtx3050ti_proof_graph.png`, `rtx3050ti_waterfall_graph.png`,
  `edp_comparison[_<workload>].png`, `test_results_summary.png`).

---

## 🚀 Execution Workflow

### Step 1: Open Administrator Terminal
Open Command Prompt or PowerShell as **Administrator** (Right-click -> *Run as Administrator*):
```cmd
cd gpus\rtx3050ti_laptop
```

### Step 2: Data Collection (Run on Laptop)
Sweep 15 core clock frequencies across **10 benchmarks** (3 new since the last dataset: `pt_mobilenet`,
`pt_elementwise`, `pt_lstm`; `pt_fft`'s iteration count was also increased so it's actually measurable now
-- see Known Issues):
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
* **`DRY_RUN = False` by default -- it actuates real clocks immediately**, not observe-only. Set
  `DRY_RUN = True` inside `deployment/sim_to_real_deploy.py` first if you want to watch its decisions
  before letting it touch hardware.
* Must be run from an Administrator terminal to actually change clocks (falls back to `DRY_RUN` automatically otherwise).

### Step 7: Empirical A/B Hardware Power Measurement
To measure live physical power and energy savings between the stock NVIDIA governor and the AI daemon on
your laptop, for any workload (trained or zero-shot). `measure_edp.py` and `plot_edp.py` both take the
workload name as an argument now -- omit it to default to `pt_matmul` (keeps the original filenames for
backward compatibility):
```cmd
:: 1. Measure Governor Baseline
python evaluation\measure_edp.py governor 10 pt_lstm

:: 2. Measure AI Daemon (Run sim_to_real_deploy.py in another terminal first)
python evaluation\measure_edp.py ai 10 pt_lstm

:: 3. Generate Comparison Plot for this workload
python evaluation\plot_edp.py pt_lstm
```
Repeat for each workload you want to test -- trained ones from the list in Step 2, or the 3 zero-shot ones
under `benchmarks/zero_shot_workloads/` (`pt_vit`, `pt_squeezenet`, `pt_sort_reduce`) to test how the
daemon behaves on an app it never trained on. Once you've tested a few:
```cmd
:: 4. Build a combined results table + bar chart across every workload tested so far
python evaluation\summarize_tests.py
```
Outputs `logs_and_plots/test_results_summary.md` and `.png`.

---

## ⚠️ Known Issues (fixed 20 Aug 2026, needs re-collection)

* **`pt_fft` was a dead workload in the last collected dataset.** Its measured `Duration:` was only
  0.07-0.12s -- under the NVML profiler's 50ms sample interval -- so most rows recorded ~0% GPU
  utilization and the power/EDP curve tracked idle power vs. clock, not real FFT behaviour (same class of
  bug as the RTX 4500 track's `dvfs-triage` A2: a workload that never produces measurable signal gets
  silently recorded as valid data). Fixed by increasing its iteration count (15 -> 120) so it's now in the
  same measurable range (~0.4-0.5s) as the other short benchmarks. **Re-run Step 2** to pick this up --
  the old `rtx3050ti_dataset.csv` still has the bad `pt_fft` rows in it.
* Everything else reviewed (state design, epsilon decay, in-process NVML clock control, thermal
  thresholds) was already correct and needed no changes.

---

## 🛡️ Safety & Recovery
* **Automatic Clock Restoration:** All deployment and data collection scripts catch exit signals and run `nvidia-smi -i 0 -rgc` in `finally` blocks.
* **Manual Emergency Reset:** If GPU clocks ever remain locked, execute this in an Administrator terminal:
  ```cmd
  nvidia-smi -i 0 -rgc
  ```
* **Hardware Volatility:** Software clock locks do not persist across reboots. Restarting your laptop immediately clears all clock locks.
