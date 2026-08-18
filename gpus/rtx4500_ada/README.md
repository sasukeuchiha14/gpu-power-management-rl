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
* **Dataset:** `rtx4500_8000_dataset.csv` holds **14,800 rows** across **19 valid workloads** (see below) x
  400 freq pairs, plus 3 dropped-but-still-present-in-the-raw-file dead workloads. The original 12
  workloads were profiled twice each (`n=2` repeat trials, the collector appends on every run); 7 workloads
  added 18 Aug 2026 to grow the dataset were profiled once (`n=1`) to keep the collection run to a single
  session -- a second pass can be queued later with `DVFS_NEW_ONLY=1 python data_collection/generate_8000_datasets.py`
  for full n=2 parity. `preprocess.py` averages whatever repeat trials exist per (workload, core, mem) cell
  and drops the 3 workloads that never actually ran on the GPU (`hotspot`, `heartwall`, `cupy_fft` — 0-2%
  utilization, see `dvfs-triage` A2), leaving **19 valid workloads x 400 pairs = 7,600 training rows** in
  `preprocessed_dataset.csv`. `data_collection/generate_8000_datasets.py`'s `BENCHMARKS` list is trimmed to
  exactly those 19 (12 original + 7 in `NEW_BENCHMARKS`), so re-running collection from scratch doesn't waste
  time on dead/phantom workloads. The filename is a historical artifact from an earlier 8,000-row target and
  is kept for path compatibility.
* **The 7 workloads added 18 Aug 2026:** `pt_cnn` (VGG16, different compute pattern from resnet50), `pt_lstm`
  (recurrent -- the only non-feedforward workload, stresses a different part of the power curve),
  `cupy_fft_fixed` (the original `cupy_fft` OOM'd trying a 2048^3 array on this 24GB card; this is the same
  FFT at a VRAM-safe 640^3), and 4 more vendored Rodinia CUDA benchmarks: `srad`, `lavamd`,
  `particlefilter_naive`, `particlefilter_float`. All 7 were smoke-tested standalone (util, no crash) before
  being trusted with a full 400-point sweep -- see `dvfs-triage` A2 on why that check matters. Verified
  utilization means: pathfinder 5.3, backprop 8.2, nw 17.3, pt_lstm 19.9, streamcluster 33.0, pt_cnn 33.9,
  crypto_hash 57.9, pt_resnet50 62.0, particlefilter_naive 62.2, lud 66.1, myocyte 67.5, cupy_matmul 67.5,
  render_proxy 73.2, cupy_fft_fixed 74.0, lavamd 83.2, srad 89.3, pt_bert 91.0, particlefilter_float 92.1,
  gaussian 94.8.

---

## 📂 Directory Structure

* `benchmarks/`: Fully self-contained -- no external rodinia checkout needed. Rodinia CUDA benchmark source +
  Makefiles for the 10 vendored workload binaries (`lud/`, `nw/`, `streamcluster/`, `pathfinder/`,
  `backprop/`, `myocyte/`, `srad/`, `lavaMD/`, `particlefilter/` [naive + float]), `gaussian.cu` at the top
  level, custom synthetic AI/Data workloads (`custom_benchmarks/`), and execution scripts. Compiled binaries
  are gitignored (built via each workload's `Makefile`/`make`, same as `gaussian`); this checkout already has
  them built. Shared Rodinia build config lives in `../common/` (referenced by the vendored Makefiles as
  `../../common` from inside each workload dir). `lud/` has its own nested `common/` instead, matching
  upstream Rodinia's layout for that one. `srad/`'s Makefile still references its original 3-levels-deep
  path (`../../../common`) from upstream and wasn't adjusted -- the binary is already built and works, but
  rebuilding `srad` from source here needs that path fixed by hand first.
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
Run the profiling script to generate the telemetry dataset. The `BENCHMARKS` list defines all 19 valid
workloads (all vendored inside `benchmarks/`, no external rodinia checkout needed). Every (workload, core,
mem) cell is profiled once per run of the script, so running it twice on the same output file produces
`n=2` repeat trials, which `preprocess.py` averages. Re-running is resumable -- it skips any
(workload, core, mem) config already in the CSV, so a killed/interrupted run picks back up instead of
duplicating rows (set `DVFS_FORCE_RECOLLECT=1` to disable and always collect a fresh repeat trial):
```bash
python data_collection/generate_8000_datasets.py

# To sweep only the 7 workloads added 18 Aug 2026 (e.g. for a second n=2 pass):
DVFS_NEW_ONLY=1 python data_collection/generate_8000_datasets.py
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
