# CLAUDE.md — Technical Reference (Capstone: GPU Power Management via RL)

> **This is the deep technical reference — architecture, RL formulation, data notes.**
> It changes slowly. For current status, active branch, and what to do next, read
> **[context.md](context.md) first, always** — that file is the one kept up to date
> every session. For the end-product definition and the direction scorecard, see
> **[NORTHSTAR.md](NORTHSTAR.md)**.
>
> This file's own internal facts (dataset sizes, feature counts, file paths) can go
> stale as work lands on `unified-model` — when it disagrees with the code, the code
> wins, and it's due for a refresh. It was last verified in full against commit
> `1dbe108`, then partially updated against `99f943d`; treat sections that read
> `state_col_order` / `Workload_Name_*` one-hots as describing the **old, pre-unified**
> design (kept for the RTX 4500 contrast case), not the target architecture.

---

> Maintained as the technical source of truth for research-paper writing.
> Everything here is derived from the actual repository code/data at commit `1dbe108`
> ("Realtime testing on rtx3050ti"). Nothing here is assumed or invented.
> **Rule: if it is not in the code or the committed data, it is not a result.**

---

## 1. Project Overview

**Objective.** Learn a *Dynamic Voltage and Frequency Scaling* (DVFS) policy for NVIDIA GPUs
with a Double Deep Q-Network (DDQN) that selects GPU core (and, on the workstation GPU,
memory) clock frequencies at runtime so as to minimise the **Energy-Delay Product (EDP)**
of the running workload, relative to the stock NVIDIA "race-to-idle" max-clock governor.

**Problem statement.** The stock NVIDIA driver governor is workload-agnostic: it boosts to the
highest clock the power/thermal budget allows and races to idle. For workloads that are not
compute-bound (memory-bound, launch-bound, or low-occupancy), the extra frequency buys little
or no runtime reduction while power rises superlinearly, so energy and EDP get worse. There is
no per-workload, per-phase frequency selection exposed to userspace.

**Why RL.** The optimal (core, mem) pair is workload-dependent, hardware-dependent and not
analytically known; it must be discovered by interaction with measured hardware response.
The project frames this as a sequential decision problem over a discrete frequency grid with
EDP-derived reward.

**Solution shape.** Offline hardware profiling → a table-backed simulator ("digital twin") of the
GPU's frequency/EDP response → DDQN trained in that simulator → the trained policy is deployed
as a real-time userspace daemon that reads NVML telemetry and issues `nvidia-smi` clock locks,
guarded by a thermal override and a "do-no-harm" performance arbiter.

**Two hardware targets** (separate, near-duplicated code trees under `gpus/`):

| | `gpus/rtx4500_ada/` | `gpus/rtx3050ti_laptop/` |
|---|---|---|
| GPU | RTX 4500 Ada (24 GB, 210 W) | RTX 3050 Ti Laptop (4 GB, 70 W) |
| OS | Linux | Windows 11 |
| Control | 2D: core + memory | 1D: core only (mem fixed 5871 MHz) |
| Core grid | 1000–2950 MHz, 50 MHz steps (40) | 500–1900 MHz, 100 MHz steps (15) |
| Mem grid | 5001–9501 MHz, 500 MHz steps (10) | fixed |
| Actions | 5 | 3 |
| State dim | 23 | 16 |
| Workloads profiled | 15 (Rodinia + PyTorch/CuPy) | 7 (PyTorch) |
| Dataset rows | 12 000 | 105 |

---

## 2. System Architecture (verified against code)

The pipeline the code actually implements is:

```
OFFLINE (per GPU)
  benchmarks/                    workload binaries / PyTorch scripts
      |
  data_collection/*.py           for each (workload, core_f, mem_f):
      |                            nvidia-smi -lgc/-lmc  -> lock clocks
      |                            NVML sampler @20 Hz   -> power, temp, util
      |                            wall-clock timing     -> Avg_Time_S
      |                            EDP = Avg_Pwr_W * Avg_Time_S^2
      v
  data/<raw>.csv                 one row per (workload, core, mem)
      |
  preprocessing/preprocess.py    one-hot(workload, type) + MinMax(numerics)
      |                          + Reward = 1 - per-workload-minmax(EDP)
      v
  data/preprocessed_dataset.csv
  preprocessing/export_scaler.py -> data/scaler_params.json (ranges + feature order)
      |
  models/dvfs_env*.py            Gymnasium env: dataset used as a LOOKUP TABLE
      |                          (nearest-neighbour snap on normalised clocks)
      |
  models/train_ddqn.py           DDQN, 800 episodes x 100 steps
      v
  weights/ddqn_weights.pth

ONLINE (deployment)
  deployment/sim_to_real_deploy.py   50 ms loop:
      NVML read (core, mem, power, temp, util)
        -> Layer 1: temp >= LIMIT ? force base clocks, skip AI
        -> normalise with scaler_params.json
        -> 1-NN against preprocessed dataset to reconstruct the
           non-observable features (Avg_Time_S) and workload one-hot
        -> frame stack (3)
        -> policy_net -> argmax action
        -> Layer 2: do-no-harm arbiter (revert to max clocks if the
           NN-estimated EDP exceeds the workload's governor EDP for
           ARB_PATIENCE ticks)
        -> nvidia-smi -lgc / -lmc
```

**Important structural fact:** the RL environment is *not* a live GPU and *not* a dynamics model.
It is a static table of profiled operating points. `step()` moves a tracked normalised frequency
and snaps to the nearest profiled row of the *same* workload. The reward depends only on the
resulting row, never on the trajectory. This is a stateless contextual-optimisation problem
dressed as an MDP — see §9 (Issues).

---

## 3. RL Formulation

### 3.1 State

`state_cols` = every column of `preprocessed_dataset.csv` except `EDP` and `Reward`
(on RTX 4500 the env prefers `scaler_params.json["state_col_order"]` when present).

RTX 4500 (23 features), RTX 3050 Ti (16 features):

| Feature | Meaning | Source | Processing |
|---|---|---|---|
| `Core_F` | SM core clock | `nvidia-smi -lgc` setpoint (offline) / `NVML_CLOCK_SM` (live) | MinMax to [0,1] |
| `Mem_F` | memory clock | `nvidia-smi -lmc` setpoint / `NVML_CLOCK_MEM` | MinMax |
| `Avg_Time_S` | mean wall-clock time per workload run | `time.time()` around the subprocess | MinMax |
| `Avg_Pwr_W` | mean board power | `nvmlDeviceGetPowerUsage`/1000, 20 Hz mean | MinMax |
| `Peak_T_C` | peak GPU temperature | `nvmlDeviceGetTemperature`, max over run | MinMax |
| `Avg_Util_PCT` | mean SM utilisation | `nvmlDeviceGetUtilizationRates().gpu` | MinMax |
| `Workload_Name_*` | workload identity | one-hot from profiling label | 14 cols (4500) / 7 cols (3050 Ti) |
| `Workload_Type_*` | 0=compute, 1=memory, 2=mixed | one-hot from hand-assigned `category` | 3 cols |

**Frame stacking:** observation is `(3, num_features)` — the last 3 single observations.
**Domain randomisation:** `_get_single_observation()` adds `N(0, 0.01)` noise, clipped to [0,1].

Live reconstruction (`build_state`): only `Core_F, Mem_F, Avg_Pwr_W, Peak_T_C, Avg_Util_PCT`
are observable; a 1-nearest-neighbour lookup over those 5 normalised features against the whole
preprocessed dataset supplies `Avg_Time_S` and the workload/type one-hots (i.e. the 1-NN *is*
the workload classifier), then the 5 observable slots are overwritten with the live values.

### 3.2 Action

RTX 4500 — `spaces.Discrete(5)`, applied to a tracked normalised setpoint:

| a | meaning | normalised stride | MHz equivalent |
|---|---|---|---|
| 0 | decrease core | −0.10 | −195 MHz |
| 1 | increase core | +0.10 | +195 MHz |
| 2 | decrease memory | −0.15 | −675 MHz |
| 3 | increase memory | +0.15 | +675 MHz |
| 4 | hold | 0 | — |

RTX 3050 Ti — `spaces.Discrete(3)`: 0 = core −0.10 (−140 MHz), 1 = core +0.10, 2 = hold.

Actuation: `denorm_to_mhz()` then
`nvidia-smi -lgc <f>,<f>` and (4500 only) `nvidia-smi -lmc <f>,<f>`.
In simulation the setpoint is clipped to [0,1] and snapped to the nearest profiled row by
Euclidean distance on (`Core_F`, `Mem_F`) — 4500 — or absolute distance on `Core_F` — 3050 Ti.

### 3.3 Reward

Two stages. Offline, in `preprocessing/preprocess.py`:

```
EDP_i        = Avg_Pwr_W_i * (Avg_Time_S_i)^2                (data_collection)
sEDP_i       = (EDP_i - min_{j in W(i)} EDP_j) / (max_{j in W(i)} EDP_j - min_{j in W(i)} EDP_j)
Reward_i     = 1 - sEDP_i
```
where `W(i)` is the set of rows of the *same workload*. Per-workload min-max, so every workload
gets the full [0,1] range (best config → 1.0, worst → 0.0). If the span is 0, `Reward = 1.0`.

Online, in `step()`:
```
r_t = Reward(row_t) - c * 1[a_t != hold],    c = ACTION_COST = 0.02
```
So the complete reward the agent optimises is

**R(s,a) = 1 − (EDP(s') − EDP_min^w) / (EDP_max^w − EDP_min^w) − 0.02·1[a ≠ hold]**

Note EDP here is *Energy × Delay* = (P·t)·t, i.e. `P·t²`.

### 3.4 Environment

- `reset()`: pick a uniformly random workload one-hot column, filter the dataframe to that
  workload, sample one random row as the start, initialise the tracked normalised clocks from
  that row, fill the 3-frame stack with copies of the first observation.
- `step(a)`: move tracked clocks → snap to nearest profiled row → append frame → reward as above.
- `terminated` is always `False`. `truncated` when `current_step >= max_steps = 100`.
- Episode = 100 steps within one workload. No thermal/power constraint inside the env.

### 3.5 Agent

`DDQNAgent` (identical in both trees except action count):
- Two networks (`policy_net`, `target_net`), target initialised from policy and kept in `eval()`.
- ε-greedy action selection.
- `step()` pushes to replay and calls `_learn()` on **every** environment step once
  `len(memory) > 64`.
- Double-DQN target: action chosen by `policy_net`, evaluated by `target_net`:
  `y = r + γ · Q_target(s', argmax_a Q_policy(s', a)) · (1 − done)`.
- Loss `MSELoss`, optimiser `Adam`.
- `update_target_network()` = hard copy, called every 5 episodes from the training loop.

---

## 4. Neural Network Architecture

`QNetwork` (`models/ddqn_agent.py`):

```
Flatten()                        # (stack, features) -> (stack*features,)
Linear(stack*features, 128) -> ReLU
Linear(128, 128)            -> ReLU
Linear(128, action_size)         # Q-values, no output activation
```

| | RTX 4500 | RTX 3050 Ti |
|---|---|---|
| Input | 3 × 23 = 69 | 3 × 16 = 48 |
| Hidden | 128, 128 (ReLU) | 128, 128 (ReLU) |
| Output | 5 | 3 |
| Params | ≈ 25.6 k | ≈ 22.9 k |

Verified against the committed `.pth` tensor sizes (35 328 B = 128×69 floats; 24 576 B = 128×48).

Legacy single-network `DQN` (`models/dqn_agent.py`, RTX 4500 only): `Linear(23,128)-ReLU-
Linear(128,128)-ReLU-Linear(128,5)`, no frame stacking; weights `weights/dvfs_dqn_weights.pth`.
Superseded by the DDQN but still importable by `evaluation/evaluate_agent.py`.

---

## 5. Hyperparameters (exact, from code)

| Parameter | Value | Where |
|---|---|---|
| Episodes | 800 | `train_ddqn.py: train_agent(episodes=800)` |
| Steps per episode | 100 | `env.max_steps` |
| Batch size | 64 (hardcoded in `_learn`) | `ddqn_agent.py` |
| Replay buffer | `deque(maxlen=10000)` | `ddqn_agent.py` |
| Learn frequency | every env step (after buffer > 64) | `ddqn_agent.py: step()` |
| γ (discount) | 0.99 | `ddqn_agent.py` |
| Optimiser | Adam | `ddqn_agent.py` |
| Learning rate | 0.001 | `ddqn_agent.py` |
| Loss | MSE | `ddqn_agent.py` |
| ε start / min | 1.0 / 0.05 | `ddqn_agent.py` |
| ε decay | 0.995 in class, **overridden to 0.99** in training | `train_ddqn.py` |
| ε schedule | multiplied once per episode while ε > ε_min | `train_ddqn.py` |
| Target sync | hard copy every 5 episodes | `train_ddqn.py` |
| Frame stack | 3 | `dvfs_env*.py`, `sim_to_real_deploy.py` |
| Observation noise | `N(0, 0.01)`, clipped [0,1] | `_get_single_observation` |
| Action cost | 0.02 | `ACTION_COST` / inline `0.02` |
| Core stride | 0.10 normalised | both envs + deploy |
| Mem stride | 0.15 normalised | 4500 only |
| Legacy DQN | γ=0.95, lr=1e-3, batch 32, 200 episodes | `dqn_agent.py` |

**Deployment config**

| Parameter | RTX 4500 | RTX 3050 Ti |
|---|---|---|
| `DRY_RUN` default | `False` (actuates) | `True` (observe only) |
| `LOOP_MS` | 50 | 50 |
| `TEMP_LIMIT_C` (Layer 1) | 85 | 80 |
| `ARBITER_ENABLE` / `ARB_PATIENCE` / `ARB_COOLDOWN` | True / 3 / 5 | True / 3 / 5 |
| `ONLINE_FINETUNE` | `False` (proxy reward only, unused) | not present |
| ε at deploy | 0.0 (0.01 if fine-tuning) | 0.0 |

---

## 6. GPU Monitoring & Power Management

**Measurement — NVML via `pynvml`** (`GPUProfiler`/`PowerSampler` threads):
- `nvmlDeviceGetPowerUsage()` / 1000 → watts
- `nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU)` → °C
- `nvmlDeviceGetUtilizationRates().gpu` → % SM utilisation
- `nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM / NVML_CLOCK_MEM)` → MHz (deployment/eval only)
- Sampling: 20 Hz (`sleep(0.05)`) during profiling; 10 Hz (4500) / 20 Hz (3050 Ti) in `measure_edp.py`.

**Actuation — `nvidia-smi` subprocess calls only** (no NVML setters, no power-limit API):
- `nvidia-smi -lgc <f>,<f>` — lock graphics/SM clock
- `nvidia-smi -lmc <f>,<f>` — lock memory clock (RTX 4500 only)
- `nvidia-smi -rgc` / `-rmc` — reset, called in `finally`/signal handlers
- **The project controls *frequency*, not the power limit** (`-pl` is never used). "Power
  management" is achieved indirectly via DVFS.

**Permissions:** Linux requires `sudo` (and `nvidia-smi -pm 1` persistence mode per the
deployment docstring); Windows requires an Administrator terminal (`ctypes.windll.shell32.
IsUserAnAdmin()`; falls back to DRY_RUN if not admin).

**Thermal handling:** profiling loops block until temperature < 55 °C (4500) / < 60 °C
(3050 Ti) before each configuration; the 3050 Ti profiler also sleeps 5 s between runs.
Deployment Layer 1 forces normalised-0 clocks when temp ≥ limit.

**Latency:** the loop measures and prints `dt_ms` per tick; no measured latency data is
committed to the repository.

---

## 7. Experimental Setup (what the repo shows)

**Hardware.** RTX 4500 Ada Generation (Linux workstation) and RTX 3050 Ti Laptop GPU
(Windows 11 laptop). No driver version, CUDA version, CPU or RAM is recorded anywhere in the
repo — **must be collected before writing the Experimental Setup section.**

**Software.** Python 3.10+, PyTorch (CUDA), Gymnasium, pandas, scikit-learn, matplotlib,
pynvml; CuPy and HuggingFace `transformers` for some RTX 4500 workloads. There is **no
`requirements.txt`, no environment lockfile, no Dockerfile** — only a `pip install` line in the
root README.

**Workloads.**
- RTX 4500 (15 in the dataset, 20 declared in `generate_8000_datasets.py`; `bfs`, `kmeans`,
  `lavaMD`, `srad`, `cfd` were skipped for missing binaries):
  Rodinia CUDA — `gaussian`, `lud`, `nw`, `hotspot`, `backprop`, `streamcluster`, `pathfinder`,
  `myocyte`, `heartwall`; custom — `pt_resnet50`, `pt_bert` (PyTorch), `cupy_matmul`,
  `cupy_fft`, `crypto_hash`, `render_proxy` (CuPy).
  Only `gaussian.cu` + `Makefile` are in the repo; the other Rodinia binaries are external
  (`../lud/cuda/lud_cuda` etc.) — **not reproducible from this repository alone.**
- RTX 3050 Ti (7, all in `benchmarks/`): `pt_resnet50`, `pt_bert`, `pt_matmul`, `pt_fft`,
  `pt_vector_add`, `pt_resnet50_train`, `pt_crypto_proxy`.

**Evaluation methodology implemented.**
1. *Simulation A/B* (`evaluation/evaluate_baselines.py`): identical start state; three policies —
   static max-clock governor, raw DDQN (greedy), DDQN + do-no-harm arbiter — over 50 steps;
   metric = per-workload scaled EDP (`1 − reward`).
2. *Simulation recovery* (`evaluation/evaluate_forced.py`): start forced to the worst-EDP row,
   30 greedy steps, plot frequency and scaled EDP ("waterfall").
3. *Hardware A/B* (`evaluation/measure_edp.py` + `plot_edp.py`): run one fixed workload N times
   under (a) pinned max clocks and (b) the live AI daemon, sample power at 10/20 Hz, integrate
   trapezoidally for energy, report mean power / energy / `P·t_iter²`.

---

## 8. Current Status

### A. Fully implemented and exercised
- Offline profiling harness for both GPUs (NVML sampling + `nvidia-smi` clock locking + cooldown).
- Preprocessing: one-hot encoding, min-max scaling, per-workload EDP→reward, scaler export.
- Gymnasium environments (2D `AdvancedDVFSEnv`, 1D `LaptopDVFSEnv`) with frame stacking,
  workload randomisation, observation noise, action cost.
- DDQN agent + training loop; trained weights committed for both GPUs.
- Deployment daemons with Layer-1 thermal override and Layer-2 do-no-harm arbiter.
- Simulation evaluation scripts + committed proof/waterfall plots.
- Hardware A/B measurement scripts + committed power traces and comparison plots.

### B. Partially implemented
- **RTX 4500 dataset quality:** 5 of the 20 declared workloads never ran; `heartwall`,
  `cupy_fft` and `hotspot` produced ~0 % utilisation (see §9).
- **`ONLINE_FINETUNE`** (RTX 4500): code path exists, disabled, uses an undesigned proxy reward
  `(util/100)·(1 − norm(power))`; the docstring itself calls it a research task.
- **Workload identification at deploy time:** implemented as 1-NN over 5 features; never
  validated (no classification accuracy measured).
- **RTX 3050 Ti hardware A/B:** measured, but the trace suggests the daemon may not have been
  actuating (see §9).
- **Legacy DQN path** (`dqn_agent.py`, `dvfs_env.py`, `evaluate_agent.py`) kept but superseded.

### C. Planned (mentioned but not implemented)
- Online adaptation / continual learning on live hardware.
- Multi-GPU or cross-GPU transfer (the two trees are copies, not a shared framework).
- Execution log files (`logs_and_plots/` is described as holding logs; it holds only PNGs).

### D. Missing (needed for the paper)
- `requirements.txt` / environment spec; driver, CUDA, CPU, RAM, ambient-temperature records.
- Training curves (reward-vs-episode is printed but never saved or plotted); no convergence
  evidence, no seed sweep, no variance/error bars.
- Held-out evaluation: the agent is trained and evaluated on the *same* profiled dataset.
  No train/test split over workloads, no unseen-workload generalisation test.
- Baselines beyond "pinned max clocks": no comparison against the *actual default* driver
  governor (unpinned), no exhaustive-search oracle, no random policy, no simple heuristic
  (e.g. utilisation-threshold), no published DVFS method.
- Performance/throughput accounting in the hardware A/B (only power and wall time are logged;
  no per-workload correctness or throughput check).
- Temperature comparison data (the root README claims reduced peak temperatures; the power CSVs
  contain no temperature column).
- Measured decision-latency data to support the "sub-50 ms" claim.
- Repeated trials / statistical testing for every reported number.
- Multi-workload and mixed/phase-changing workload deployment tests.

---

## 9. Known Issues & Inconsistencies (do not paper over these)

**Data quality**
1. `heartwall` (800 rows): times 0.5–2 ms, util 0.2 % — the binary never ran. Its EDP is ~0 and
   its rows are noise.
2. `cupy_fft` (800 rows): 2048³ complex64 ≈ 32 GB > 24 GB VRAM; times 0.21–0.29 s, util 1.7 % —
   almost certainly OOM'd immediately.
3. `hotspot` (800 rows): util 0.0 % across the sweep. **This is the workload used in the
   headline RTX 4500 simulation proof graph (`review_2_proof_graph.png`).**
4. `backprop` (72–91 ms) and `pathfinder` (159–181 ms) are dominated by process launch overhead.
5. `render_proxy` executes the *same* CuPy matmul script as `cupy_matmul` but is labelled
   category 2 vs 1 — two "workloads" that are the same computation.
6. RTX 3050 Ti: every workload is timed as a full `python …` subprocess, so ~3 s of interpreter +
   CUDA-context startup is inside `Avg_Time_S`. `pt_fft` shows util 0.0 % and a flat 3.17–3.30 s
   across the whole 500–1900 MHz sweep — its EDP curve is pure idle-power-vs-clock. The
   benchmark scripts print their own GPU-only duration, which the harness discards.
7. Consequence: for most 3050 Ti workloads the max clock is by construction the worst EDP, so
   "the agent beats the governor" is partly a measurement artifact, not a DVFS finding.
8. `rtx4500_8000_dataset.csv` contains **12 000** rows (each configuration profiled twice), not
   8 000 as the filename and both READMEs state.

**Code/config drift**
9. `data/scaler_params.json` is **stale**: `state_col_order` has 23 entries (14 workloads, no
   `heartwall`), but `preprocessed_dataset.csv` now has 15 workload one-hots (24 state features).
   The env prefers the JSON, so training silently used 23 features and `heartwall` episodes were
   trained with an all-zero workload one-hot. The committed weights confirm the 69-dim input.
10. RTX 3050 Ti `Mem_F` mismatch: `preprocess.py` maps the constant 5871 MHz to **0.0** in every
    training row, but `export_scaler.py` forces `Mem_F.min = 0`, so the live daemon computes
    `norm(5871, 0, 5871) = 1.0`. The deployed network sees a feature value it never saw in
    training. (The 1-NN lookup is unaffected — constant offset — but the policy input is not.)
11. `data_collection/generate_dataset.py` (3050 Ti) references `CORE_FREQS` and `FIXED_MEM_FREQ`,
    which are **never defined** → `NameError` if run as committed. The dataset exists, so an
    earlier version had them.
12. `data_collection/fix_short_benchmarks.py` (4500) likewise uses undefined `CORE_FREQS` /
    `MEM_FREQS`.
13. 3050 Ti `safety_override()` docstring says "baseline (405 MHz)"; the code sets
    `denorm_to_mhz(0.0) = 500 MHz`.
14. Root README says the deployment daemon is "Dry Run by default" for both GPUs; the RTX 4500
    daemon ships with `DRY_RUN = False` (it *will* actuate hardware on launch).
15. `evaluation/script_name.py` is an unnamed leftover profiling script producing a
    schema-incompatible CSV (`rtx4500_hardware_dataset.csv`, no workload columns).

**Methodological**
16. The environment is a static lookup table. There are no dynamics: `s'` depends only on the
    chosen frequency, the reward depends only on `s'`, and the frame stack carries no extra
    information (the 3 frames are the same row plus i.i.d. noise unless a move just happened).
    With γ = 0.99 over 100 steps this is effectively a per-workload argmin-EDP search, not a
    sequential control problem. A tabular/oracle lookup would achieve the same optimum.
17. The state contains `Avg_Time_S`, `Avg_Pwr_W`, `Peak_T_C`, `Avg_Util_PCT` of the *resulting*
    row — i.e. measured outcomes that directly determine the reward — plus the workload one-hot.
    The agent is largely told the answer; at deployment `Avg_Time_S` and the one-hot must be
    guessed by 1-NN.
18. Train and test use the same 12 000 / 105 rows. No generalisation claim is supported.
19. Action strides (195 MHz core / 675 MHz memory) are much coarser than the profiled grid
    (50 / 500 MHz), so the agent can only reach a subset of the 400 profiled operating points.
20. The "governor" baseline is *pinned max clocks*, not the real NVIDIA adaptive governor. This
    is a strictly worse baseline than the driver default and should be described precisely.
21. In `evaluate_baselines.py` the plotted AI curve is `1 − reward`, which includes the 0.02
    action cost, while the governor curve (always "hold") does not — a small bias *against* the AI.
22. RTX 3050 Ti hardware A/B: in `power_ai.csv` the sampled core clock reaches **2002 MHz**,
    above the 1900 MHz maximum the daemon can command, and the daemon's default is
    `DRY_RUN = True`. This trace is consistent with the driver, not the agent, controlling the
    clocks during the "ai" run. **Verify before using this number.**
23. In the RTX 4500 hardware A/B the AI run took 32.4 s vs the governor's 31.3 s for the same
    work (~3.5 % slower); the energy saving must be reported together with that slowdown.
24. `nw`, the workload used for the RTX 4500 hardware A/B, has a mean utilisation of 17.3 % — a
    light, memory-bound workload, which is the most favourable possible case for downclocking.

---

## 10. Existing Results (only these; everything else must be generated)

### Simulation (scaled-EDP units, one workload, one seed, no repeats)
| Source | Workload | Governor | Raw DDQN | DDQN + arbiter |
|---|---|---|---|---|
| `rtx4500_ada/logs_and_plots/review_2_proof_graph.png` | `hotspot` | ≈ 0.955 (flat) | converges ≈ 0.065 | converges ≈ 0.04 |
| `rtx3050ti_laptop/logs_and_plots/rtx3050ti_proof_graph.png` | `pt_vector_add` | 1.00 (flat) | converges ≈ 0.03 | ≈ 0.03 |

`*_waterfall_graph.png`: from the worst-EDP start the 3050 Ti agent walks 1800 → 900 MHz in
~6 steps and holds; scaled EDP settles ≈ 0.13.

### Hardware A/B (recomputed here directly from the committed CSVs)
| | RTX 4500 (`nw`, 20 iters) | RTX 3050 Ti (`pt_matmul`, 10 iters) |
|---|---|---|
| Governor mean power | 43.62 W | 36.02 W |
| AI mean power | 31.04 W | 28.93 W |
| Power reduction | 28.8 % | 19.7 % |
| Governor energy (trapezoidal) | 1367.1 J | 1253.6 J |
| AI energy | 1005.4 J | 1179.3 J |
| **Energy saving** | **26.5 %** | **5.9 %** |
| Sampling window | 31.3 s → 32.4 s | 44.2 s → 40.8 s |
| Caveats | AI run 3.5 % longer | see issue #22 — trace may not be agent-controlled |

`power_ai_before.csv` (RTX 4500) is an earlier AI run: mean 35.72 W, 1144.7 J (16.3 % energy
saving) — presumably the pre-tuning version behind `edp_comparison_before.png`.

**README claim vs evidence:** the root README states "~18 % to 26 % energy savings", "reduced
peak temperatures" and "sub-50 ms decision latency". Only the 26 % figure (RTX 4500, `nw`,
n = 1) is supported by committed data. The temperature and latency claims have **no** supporting
data in the repository, and the 3050 Ti measurement gives 5.9 %.

**Everything else — power savings across workloads, generalisation, statistical significance,
baseline comparisons, training curves — is currently unavailable and must be generated through
future experiments.**

---

## 11. Repository Map

```
README.md                      root overview, claims results summary (see §10 caveat)
CONTRIBUTING.md                safety guidelines (DRY_RUN first, thermal override always on)
LICENSE                        MIT

gpus/rtx4500_ada/
  README.md                    2D DVFS target spec + workflow
  benchmarks/
    gaussian.cu, Makefile, run, README.txt   Rodinia Gaussian elimination (only CUDA source in repo)
    custom_benchmarks/ai_workloads.py        ResNet-50 / BERT (auto-generated by the collector)
    custom_benchmarks/data_workloads.py      CuPy matmul / fft / crypto proxy (auto-generated)
    custom_benchmarks/run_workload.py        standalone 7-workload PyTorch runner (unused by collectors)
  data_collection/
    generate_8000_datasets.py  MAIN collector: 20 declared workloads x 40 core x 10 mem;
                               GPUProfiler thread (20 Hz NVML); writes rtx4500_8000_dataset.csv
    generate_300_datasets.py   earlier 3-workload, warmup+3-run collector
    fix_short_benchmarks.py    re-profile lud/nw with loop counts (broken: undefined freq lists)
  data/
    rtx4500_8000_dataset.csv   12 000 raw rows, 15 workloads
    rtx4500_300_dataset.csv    372 rows (earlier sweep)
    rtx4500_fixed_part2.csv    200 rows (lud/nw re-profile)
    rtx4500_hardware_dataset.csv  18 rows, incompatible schema (from script_name.py)
    preprocessed_dataset.csv   12 000 rows, 26 cols (24 state + EDP + Reward)
    preprocessed_dataset_global.csv  backup with the OLD global-EDP reward, 14 workloads
    scaler_params.json         ranges + 23-feature order (STALE, see issue #9)
    power_governor.csv / power_ai.csv / power_ai_before.csv   hardware A/B traces
  preprocessing/
    preprocess.py              one-hot + MinMax + per-workload reward -> preprocessed_dataset.csv
    export_scaler.py           ranges + feature order -> scaler_params.json
    rescale_reward_per_workload.py  retrofits per-workload reward onto an existing file
  models/
    dvfs_env.py                legacy 1D-observation env (no stacking)
    dvfs_env_advanced.py       ACTIVE env: AdvancedDVFSEnv, frame stack, noise, ACTION_COST
    dqn_agent.py               legacy DQN class + its own inline training loop
    ddqn_agent.py              ACTIVE: QNetwork + DDQNAgent (act/step/_learn/update_target_network)
    train_ddqn.py              training entry point (800 episodes) -> weights/ddqn_weights.pth
  weights/
    ddqn_weights.pth           ACTIVE policy (69 -> 128 -> 128 -> 5)
    ddqn_weights_nojitter.pth, ddqn_weights_old.pth   variants, same shape
    dvfs_dqn_weights.pth       legacy DQN (23 -> 128 -> 128 -> 5)
  deployment/
    sim_to_real_deploy.py      LiveController: read_sensors, build_state (1-NN), apply_action,
                               set_clocks, safety_override, arbiter, run loop @50 ms
  evaluation/
    evaluate_baselines.py      governor vs raw DDQN vs arbitrated -> review_2_proof_graph.png
    evaluate_forced.py         worst-state recovery -> dqn_waterfall_graph.png
    evaluate_agent.py          legacy DQN evaluation -> dqn_evaluation_results.png
    measure_edp.py             hardware power/energy/EDP measurement (label = governor | ai)
    plot_edp.py                overlays the two power traces -> edp_comparison.png
    script_name.py             leftover profiler (misnamed, incompatible output)
  logs_and_plots/              5 PNGs; no log files despite the README

gpus/rtx3050ti_laptop/         same layout, 1D core-only variant
  benchmarks/pt_*.py           7 standalone PyTorch workloads (resnet50, bert, matmul, fft,
                               vector_add, resnet50_train, crypto_proxy)
  data_collection/generate_dataset.py   thermal-aware collector (broken: undefined CORE_FREQS)
  data/rtx3050ti_dataset.csv   105 rows = 7 workloads x 15 core freqs
  models/dvfs_env.py           LaptopDVFSEnv (3 actions, core only)
  models/ddqn_agent.py, train_ddqn.py, weights/ddqn_weights.pth (48 -> 128 -> 128 -> 3)
  deployment/sim_to_real_deploy.py      Windows daemon, admin check, 80 C limit, DRY_RUN=True
  evaluation/                  evaluate_baselines.py, evaluate_forced.py, measure_edp.py, plot_edp.py
  logs_and_plots/              3 PNGs
```

---

## 12. Research-Paper Notes

**Terminology to keep consistent** (established from the code):
- "DVFS controller" / "DDQN agent"; "static max-clock governor" for the baseline (never call it
  "the NVIDIA governor" — it is pinned max clocks, not the adaptive driver default).
- State variables by their column names: `Core_F`, `Mem_F`, `Avg_Time_S`, `Avg_Pwr_W`,
  `Peak_T_C`, `Avg_Util_PCT`, workload one-hot, workload-type one-hot.
- "Per-workload scaled EDP" for `1 − reward` in simulation; "measured energy (J)" and "mean
  power (W)" for hardware results. Never mix the two units.
- "Layer 1 thermal safety override" and "Layer 2 do-no-harm arbiter".
- "Offline profiling dataset" / "table-backed simulator" — not "GPU simulator" or "digital twin
  model", since no model is fitted.

**Abstract / Introduction.** Safe to state: EDP-directed DDQN DVFS controller, two NVIDIA GPUs,
23-/16-dimensional telemetry state, 5-/3-action discrete frequency control, offline profiling
+ simulation training + live userspace deployment with two safety layers. Report the RTX 4500
hardware result with n = 1 and its 3.5 % slowdown; do not report the 3050 Ti number until
issue #22 is resolved.

**Related Work.** Nothing in this repository; must be written from literature (Rodinia is cited
only as a benchmark source).

**Methodology.** §2–§5 above are directly usable. Be explicit that the environment is a lookup
table over profiled operating points, and that the reward is a per-workload min-max
normalisation of measured `P·t²`.

**Limitations (write these honestly).** Table-lookup environment with no dynamics; train/test on
the same data; several profiled workloads produced negligible GPU utilisation; wall-clock timing
includes process startup on the laptop target; baseline is pinned max clocks; single seed and
single trial per result; workload identity inferred by 1-NN and never validated; frequency-only
control (no power-limit control); no throughput/QoS verification during deployment.

**Future Work.** Online fine-tuning with a designed live reward; held-out workload
generalisation; continuous action / larger frequency resolution; power-limit co-control;
multi-GPU and multi-tenant settings; comparison against published DVFS controllers.

---

## 13. Working Rules for Paper Assistance

1. Every methodological claim must be traceable to a file and function in this repo.
2. Never invent results, hyperparameters, hardware details, baselines, or citations.
3. Implemented ≠ planned. Anything in §8-C/D is future work.
4. Keep §12 terminology fixed across all sections.
5. Flag contradictions with §9 *before* writing text that depends on them.
