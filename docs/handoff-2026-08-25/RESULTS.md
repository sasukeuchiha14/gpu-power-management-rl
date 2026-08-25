# Results — every number we measured

All results produced 23–25 August 2026 on branch `unified-model`. Every table here is regenerable from the repo with the command shown beneath it; raw CSVs are in [`results/`](results/).

**Read the caveat in §6 before quoting anything that depends on telemetry.**

---

## 1. Data collected

Eight GPUs profiled across two NVIDIA architecture generations, **2 098 profiled operating points** in total.

| GPU | Arch | Rows | Workloads | Core levels | Mem levels | Notes |
|---|---|---|---|---|---|---|
| RTX 3060 | Ampere | 288 | 12 | 8 | 3 | 0 skips |
| RTX 3070 | Ampere | 288 | 12 | 8 | 3 | 0 skips |
| RTX 3080 | Ampere | 288 | 12 | 8 | 3 | 0 skips |
| RTX 3090 | Ampere | 288 | 12 | 8 | 3 | 0 skips |
| RTX 4070 | Ada | 256 | 12 | 8 | 3 | 32 skips at top boost bin |
| RTX 4070 Ti | Ada | 288 | 12 | 8 | 3 | 0 skips |
| RTX 4090 | Ada | 252 | 12 | 7 | 3 | 36 skips at top boost bin |
| RTX 3050 Ti (laptop) | Ampere | 150 | 10 | 15 | 1 | core-only hardware |

**On the skips:** both Ada desktop cards cannot jointly lock their top boost bin (3105 MHz) with any tested memory state — the driver coerces to ~2700–2790 MHz. The collector's achieved-clock guard refuses to record a mislabelled point rather than silently writing the wrong clock. RTX 4090 succeeded at that bin only 4/36 times, i.e. it is a marginal hardware state, not a script failure.

**Per-GPU audit** (all passed): header present, exact row count, zero duplicate `(workload, core, mem)`, EDP recomputes to within 0.09 %, all fields in physical range, per-workload trends physically sensible, quick-vs-full run cross-check.

Raw data: `gpus/<gpu_slug>/data/`. Capability survey: `common/fleet_survey/fleet_survey.csv`.

---

## 2. Baseline comparison — the headline result

Steady-state scaled EDP per held-out GPU. **0 = that workload's best profiled operating point, 1 = its worst. Lower is better.**

| Policy | 3060 | 3070 | 3080 | 3090 | 4070 | 4070Ti | 4090 | laptop | **MEAN** |
|---|---|---|---|---|---|---|---|---|---|
| oracle (per-workload best) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| **fixed clock @ 0.6 × max** | 0.053 | 0.075 | 0.009 | 0.012 | 0.020 | 0.034 | 0.034 | 0.062 | **0.037** |
| fixed, tuned leave-one-out (picks 0.60 every fold) | 0.053 | 0.075 | 0.009 | 0.012 | 0.020 | 0.034 | 0.034 | 0.062 | **0.037** |
| supervised RF (feature ceiling) | 0.047 | 0.087 | 0.025 | 0.013 | 0.013 | 0.032 | 0.099 | 0.099 | 0.052 |
| fixed 0.5 | 0.066 | 0.118 | 0.014 | 0.020 | 0.021 | 0.035 | 0.143 | 0.046 | 0.058 |
| fixed 0.7 | 0.053 | 0.075 | 0.050 | 0.025 | 0.026 | 0.110 | 0.126 | 0.101 | 0.071 |
| fixed 0.4 | 0.097 | 0.104 | 0.049 | 0.044 | 0.021 | 0.029 | 0.213 | 0.153 | 0.089 |
| random walk | 0.298 | 0.136 | 0.237 | 0.323 | 0.152 | 0.241 | 0.525 | 0.099 | 0.251 |
| **governor (pinned max clock)** | 0.378 | 0.447 | 0.146 | 0.134 | 0.123 | 0.359 | 0.564 | 0.401 | **0.319** |
| utilisation heuristic | 0.356 | 0.461 | 0.868 | 0.851 | 0.629 | 0.825 | 0.464 | 0.212 | 0.583 |
| min clock | 0.806 | 0.667 | 0.936 | 0.983 | 0.947 | 0.825 | 0.582 | 0.808 | 0.819 |

```bash
python common/analysis/baselines.py --loo-tune      # -> docs/results/baselines.csv
```

**A fixed clock at 60 % of maximum — no telemetry, no learning — scores 0.037 against the max-clock governor's 0.319 and a per-workload oracle's 0.000, capturing ~88 % of the achievable gain.** It beats every learned policy we tested.

It is honest: tuned on the 7 training GPUs only, it selects 0.60 on **all 8 folds**, so it is not fitted to the card it is evaluated on. And it is robust to §6 — it depends only on timing and EDP, both correctly measured.

The optimum is **never** at max clock on any (GPU, workload) pair; mean optimal core fraction is 0.54.

---

## 3. Leave-one-GPU-out RL — single seed vs. four seeds

### 3.1 Single seed (42) — the result that looked good

| Held-out GPU | Governor EDP | AI EDP | Reduction | Workloads AI won |
|---|---|---|---|---|
| RTX 3060 | 0.192 | 0.095 | **+50.7 %** | 8/12 |
| RTX 3080 | 0.142 | 0.103 | **+27.3 %** | 4/12 |
| RTX 4070 | 0.113 | 0.103 | +8.5 % | 4/12 |
| RTX 4070 Ti | 0.222 | 0.056 | **+74.8 %** | 10/12 |
| RTX 4090 | 0.276 | 0.264 | +4.5 % | 8/12 |
| RTX 3050 Ti laptop | 0.401 | 0.106 | **+73.5 %** | 10/10 |
| RTX 3070 | 0.263 | 0.323 | −22.8 % | 5/12 |
| RTX 3090 | 0.119 | 0.167 | −40.4 % | 3/12 |

6/8 apparent wins. Raw: `results/holdout_single_seed42.csv`.

### 3.2 Four seeds (42, 7, 123, 2024) — the result that survives

| Held-out GPU | Mean reduction | **Std** | Min | Max | Seeds won |
|---|---|---|---|---|---|
| **RTX 4070 Ti** | **+72.1 %** | **±2.7** | +69.3 | +74.8 | **4/4** |
| **RTX 4070** | **+33.6 %** | ±24.7 | +8.5 | +56.2 | **4/4** |
| RTX 3050 Ti laptop | +37.5 % | ±73.3 | −72.5 | +74.5 | 3/4 |
| RTX 3060 | +26.2 % | ±27.7 | −2.0 | +50.7 | 3/4 |
| RTX 3070 | +4.5 % | ±70.8 | −82.6 | +69.3 | 2/4 |
| RTX 4090 | −21.7 % | ±54.7 | −69.9 | +42.8 | 2/4 |
| RTX 3080 | −54.8 % | **±185.9** | −332.9 | +56.4 | 3/4 |
| RTX 3090 | −62.5 % | ±116.8 | −233.5 | +20.4 | 2/4 |

```bash
python common/analysis/sweep_rl.py --preset combos   # or common/aggregate_seed_sweep.py
```

Raw: `results/seed_sweep_summary.csv`, `results/seed_sweep_detail.csv`.

**Only 2 of 8 GPUs generalise consistently.** For the other six the outcome flips between clear win and clear loss on random seed alone, with standard deviations frequently exceeding the mean effect. The laptop showed a perfect 10/10-workload sweep on seed 42 and lost badly (−72.5 %) on seed 123.

> **Methodological lesson worth carrying into the paper:** a single-seed leave-one-GPU-out result in this problem is not evidence. Ours would have reported "6/8 GPUs generalise" and a reviewer running the obvious seed check would have found 2/8.

---

## 4. RL configuration sweep

15 configurations × 4 held-out GPUs × 2 seeds, on the corrected core-only view.

| Config | ss mean | ss std |
|---|---|---|
| **no_double** (plain DQN) | **0.0685** | ±0.059 |
| **gamma 0.50** | 0.0709 | **±0.038** ← lowest variance |
| lr 3e-4 | 0.0745 | ±0.070 |
| hidden 256 | 0.0752 | ±0.067 |
| gamma 0.00 (pure bandit) | 0.0828 | ±0.052 |
| 3000 episodes | 0.0844 | ±0.056 |
| soft target tau 0.01 | 0.0947 | ±0.081 |
| action cost 0.02 | 0.0955 | ±0.079 |
| batch 128 | 0.1110 | ±0.115 |
| absolute actions (K=8), gamma 0 | 0.1128 | ±0.079 |
| absolute actions (K=8), gamma 0.9 | 0.1133 | ±0.078 |
| **v2 fixes only (gamma 0.99)** — starting point | **0.1324** | ±0.120 |
| gamma 0.90 | 0.1350 | ±0.103 |
| dueling head | 0.1357 | ±0.097 |

```bash
python common/analysis/sweep_rl.py --preset screen
```

**Bug fixes + tuning roughly halved both error (0.132 → 0.069) and seed variance.** γ 0.99 → 0.50 was the single biggest lever, consistent with the task being near-bandit: reward depends only on the operating point you land on, never the path taken there. Dueling heads and an absolute action space both *hurt*. Plain DQN beat Double-DQN.

**Even so, the best RL configuration (0.069) still loses to a fixed 0.6× clock (0.037).**

---

## 5. Why adaptation loses — the decomposition

### 5.1 Prediction accuracy vs. the resolution required

| Quantity | Value |
|---|---|
| Predict optimal `core_frac` from single-point telemetry (LOO mean abs err) | 0.165 |
| …adding the observed response to a clock change (probing) | 0.157 |
| …just predicting the global mean optimum (0.534) | 0.199 |
| **One clock grid step** | **0.116** |

The predictor is typically **~1.4 grid steps off** — worse than the resolution it needs — and only modestly better than ignoring telemetry entirely. Probing helps marginally.

### 5.2 The feature ablation — this overturned an earlier conclusion

Task: "is the current clock below this workload's EDP optimum?", leave-one-GPU-out.

| Feature set | LOO accuracy |
|---|---|
| majority-class baseline | 0.570 |
| all 6 features | 0.812 |
| **`core_frac` alone** | **0.781** |
| the other 5 telemetry channels (no `core_frac`) | 0.696 |

An earlier session read the 0.812 as proof the features carried transferable workload signal. They do not: `core_frac` — simply *where you sit on the clock range* — accounts for nearly all of it. **The five real telemetry channels add 3.1 percentage points.** That conclusion is retracted.

### 5.3 Tasks that need genuine workload characterisation

| Task (leave-one-GPU-out) | Accuracy | Chance |
|---|---|---|
| Identify which of 12 workloads is running | **0.216** | 0.083 |
| Identify compute- vs bandwidth-bound regime (binary) | **0.677** | ~0.500 |

```bash
python common/analysis/diagnose_features.py
```

A regime-classifier policy (predict the regime, use that regime's mean optimum from training GPUs) scores 0.093 (2-class) / 0.068 (3-class) against the fixed clock's 0.037 — it loses, because at 68 % accuracy it picks the wrong regime often enough to give back more than adaptation gains.

### 5.4 The optima are clustered, and one cluster is very tight

| Workload | Mean optimal core fraction | Std across 8 GPUs |
|---|---|---|
| pt_bert | 0.759 | **0.046** |
| pt_resnet50 | 0.769 | **0.067** |
| pt_matmul | 0.764 | **0.075** |
| pt_crypto_proxy | 0.288 | 0.098 |
| pt_sparse_gather | 0.474 | 0.115 |
| pt_fft | 0.601 | 0.125 |
| pt_vector_add | 0.300 | 0.135 |
| pt_elementwise | 0.342 | 0.138 |
| pt_lstm | 0.511 | 0.144 |
| pt_transformer_decode | 0.282 | 0.148 |
| pt_mobilenet | 0.536 | 0.160 |
| pt_resnet50_train | 0.737 | 0.161 |

Compute-bound workloads want ~0.76 and agree closely across cards; bandwidth/latency-bound want ~0.28–0.30. The structure adaptation would exploit **is real and physically sensible** — the problem is identifying which regime you are in from telemetry (§5.3).

---

## 6. ⚠️ The measurement bug that gates §3, §4 and §5

`collect_dataset.py` sampled NVML across the entire benchmark subprocess — Python interpreter start, `import torch`, CUDA context creation, typically **2–4 s of idle GPU** — while the actual timed work is often **0.15–0.6 s**.

| Evidence across the fleet | Value |
|---|---|
| corr(GPU-only duration, recorded utilisation) | **+0.708** |
| points with < 0.5 s real work | 494/799 (62 %), mean recorded util **8.3 %** |
| points with > 2 s real work | mean recorded util 40.2 % |
| **highest utilisation recorded anywhere in 8 GPUs** | **64.4 %** |
| `pt_matmul` (dense GEMM — should saturate) | **21.7 %** |

```bash
python common/analysis/telemetry_dilution.py
```

Recorded utilisation tracks *how long a workload ran relative to fixed startup cost*, not how hard it drove the GPU. No measurement anywhere in the fleet shows a busy GPU; a dense GEMM at 21.7 % is not physically credible.

**`Avg_Time_S` and EDP are unaffected** — timing comes from each benchmark's own `Duration:` line, which brackets only its measured region. Only the telemetry channels are diluted — precisely the features a controller needs to characterise an unknown workload.

**Fixed in code:** all 12 benchmarks emit `Window: <start> <end>` (printed *after* the timed region, so zero measurement overhead); `GPUProfiler` timestamps every sample and `get_stats(window)` averages in-window samples only, with a safe fallback. Verified on a synthetic trace — 3 s idle at 2 % util then 1 s of work at 95 %: whole-process reports **25.2 %**, window-scoped reports **95.0 %**. That 25.2 % artifact closely matches the real fleet's `pt_matmul` 21.7 % and `pt_bert` 25.4 %.

**The fix cannot repair already-collected data.**

> **Therefore:** "telemetry-driven adaptation cannot beat a fixed clock" is currently an **artifact of instrumentation, not a demonstrated property of GPU DVFS**, and must not be written up as the latter. §2 is unaffected and stands.

---

## 7. What is safe to claim

| Claim | Status |
|---|---|
| A fixed 0.6× clock captures ~88 % of achievable EDP gain over the max-clock governor, across 8 GPUs and 2 generations | ✅ **Robust** — timing/EDP only |
| Single-seed leave-one-GPU-out RL results are unreliable here (6/8 → 2/8 under a 4-seed sweep) | ✅ **Robust** |
| Three specific RL implementation defects; fixing them halves error and variance | ✅ **Robust** |
| Sampling telemetry across a whole benchmark process is a first-class threat to validity in DVFS studies | ✅ **Robust** — and a useful methodological contribution |
| γ ≈ 0.5, plain DQN, Huber, no dueling is the better configuration | 🟡 Likely robust (optimisation dynamics, not feature content) — re-confirm after re-collection |
| Whether a learned controller can beat a tuned static clock | ❌ **Unanswerable on current data** — needs the fleet re-collected |

---

## Raw files

| File | Contents |
|---|---|
| [`results/baselines.csv`](results/baselines.csv) | §2 — full baseline table |
| [`results/holdout_single_seed42.csv`](results/holdout_single_seed42.csv) | §3.1 — single-seed leave-one-out |
| [`results/seed_sweep_summary.csv`](results/seed_sweep_summary.csv) | §3.2 — per-GPU mean/std across 4 seeds |
| [`results/seed_sweep_detail.csv`](results/seed_sweep_detail.csv) | §3.2 — every (GPU, seed) run |
| `../common/holdout_results/*.csv` | per-workload detail for all 32 holdout runs |
| `../common/weights/*.pth` | trained policies + reward histories |
| `../gpus/<slug>/data/` | raw and preprocessed profiling datasets |
| `../common/fleet_survey/fleet_survey.csv` | per-GPU clock-control capability survey |
