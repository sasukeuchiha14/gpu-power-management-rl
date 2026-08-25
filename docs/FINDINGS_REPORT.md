# Findings Report — GPU DVFS via Reinforcement Learning

**Last updated:** 25 August 2026 · **Branch:** `unified-model` · **Status:** results are honest and reproducible, but one central question is currently *unanswerable* on the committed data (see §4).

This is the document to read if you want to know **what we found, what it means, and what to do next**. For the blow-by-blow investigation log including hypotheses that were disproved, see [`rl_instability_investigation.md`](rl_instability_investigation.md). For the deep technical reference, see [`../CLAUDE.md`](../CLAUDE.md). For the end-goal definition and scorecard, see [`../NORTHSTAR.md`](../NORTHSTAR.md).

---

## 1. What the project is trying to do

Learn **one** DDQN policy that sets GPU core clock at runtime to minimise **Energy-Delay Product** (EDP = Power × Time²), reads only live NVML telemetry, runs unmodified on any RTX 3000/4000 consumer GPU — *including ones it never trained on* — and does so without meaningfully slowing the workload.

The scientific claim that matters is **cross-GPU generalisation**: train on N GPUs, work on the N+1th.

---

## 2. What we built and measured this session

### 2.1 Data collection — 8 GPUs profiled

Rented and profiled 7 desktop GPUs across two architecture generations, added to the existing laptop card:

| GPU | Arch | Points collected | Notes |
|---|---|---|---|
| RTX 3060 | Ampere | 288/288 | from a prior session |
| RTX 3070 | Ampere | 288/288 | 0 skips |
| RTX 3080 | Ampere | 288/288 | 0 skips |
| RTX 3090 | Ampere | 288/288 | 0 skips |
| RTX 4070 | Ada | 256/288 | 32 skips at max boost bin (see below) |
| RTX 4070 Ti | Ada | 288/288 | 0 skips |
| RTX 4090 | Ada | 252/288 | 36 skips at max boost bin |
| RTX 3050 Ti | Ampere (laptop) | 150 | core-only, 1 memory state |

Each is a 12-workload × 8-core-level × 3-memory-level sweep, individually audited on 7 dimensions (header integrity, exact row count, zero duplicates, EDP recompute, physical-range sanity, per-workload trend sanity, quick-vs-full reproducibility).

**Skips are legitimate, not failures.** Both Ada cards cannot jointly lock their top boost bin (3105 MHz) with any tested memory state — the driver coerces to ~2700-2790 MHz — so the collector's achieved-clock guard correctly refuses to record a mislabelled point.

### 2.2 The pipeline that was missing

Raw telemetry alone is not trainable. Added `common/preprocess_desktop.py` to convert every collected dataset into the 6-feature schema the training code discovers. Caught a real bug before running it: `fleet_survey.csv` carries multiple rows per GPU model across rental instances with genuinely different `max_core_mhz` (RTX 3060 alone shows 2160 / 2100 / 2145), so naive name-matching would normalise some cards against the wrong ceiling. Each dataset is now pinned to the exact (GPU, driver) pair that produced it, with an automatic abort if a dataset's observed max clock exceeds its pinned ceiling.

### 2.3 Safety work

The do-no-harm arbiter had been **deleted** from the laptop deployment daemon in an earlier rewrite — only unread config constants remained. Restored as `gpus/rtx3050ti_laptop/deployment/arbiter.py`, but **redesigned**: the old RTX 4500 version reconstructed live EDP via nearest-neighbour lookup against the profiled dataset, which requires workload-identifying features the 6-feature design deliberately excludes. Reusing it would have silently reintroduced the closed-world assumption the redesign removed. The new arbiter watches live SM utilisation instead — high utilisation while running below max clock is the always-observable signature of holding back a compute-bound GPU. Covered by 7 unit tests plus a mocked-NVML integration test that drives the real controller with real trained weights.

`DRY_RUN` also now defaults to `True` (it was `False`, i.e. actuating real hardware on launch, contradicting the project's own `CONTRIBUTING.md`).

---

## 3. Results

All numbers are **steady-state scaled EDP**: 0 = that workload's best profiled operating point, 1 = its worst, averaged over the held-out GPU's workloads. Lower is better. Reproduce with `python common/analysis/baselines.py --loo-tune`.

### 3.1 The headline result

| Policy | Mean scaled EDP | Uses telemetry? | Learned? |
|---|---|---|---|
| oracle (per-workload best) | **0.000** | — | — |
| **fixed clock @ 0.6 × max** | **0.037** | no | no |
| supervised classifier (feature ceiling) | 0.053 | yes | yes |
| fixed 0.5 | 0.058 | no | no |
| best RL configuration found | ~0.069 | yes | yes |
| fixed 0.7 | 0.071 | no | no |
| random walk | 0.278 | no | no |
| **governor (pinned max clock)** | **0.319** | no | no |
| utilisation heuristic | 0.573 | yes | no |
| min clock | 0.819 | no | no |

**A single static clock at 60% of maximum, using no telemetry and no learning, captures ~88% of the gap between the max-clock governor and a per-workload oracle — and beats every learned policy tested.**

This is verified leave-one-GPU-out: tuned only on the 7 training GPUs, it selects 0.60 on all 8 folds. It is not hindsight fitting.

**This finding is robust to everything in §4** — it depends only on EDP and timing, both of which are measured correctly.

### 3.2 Leave-one-GPU-out RL, with variance

First single-seed run looked encouraging (6/8 GPUs beating the governor). A 4-seed sweep destroyed that reading:

| held-out GPU | mean reduction vs governor | std across seeds |
|---|---|---|
| RTX 4070 Ti | +72.1% | ±2.7% |
| RTX 4070 | +33.6% | ±24.7% |
| RTX 3060 | +26.2% | ±27.7% |
| RTX 3050 Ti laptop | +37.5% | ±73.3% |
| RTX 3070 | +4.5% | ±70.8% |
| RTX 4090 | −21.7% | ±54.7% |
| RTX 3080 | −54.8% | **±185.9%** |
| RTX 3090 | −62.5% | ±116.8% |

Only **2 of 8** GPUs (4070, 4070 Ti) generalised consistently. For the rest the result flips between clear win and clear loss on random seed alone, with standard deviations often exceeding the mean effect. The laptop showed a perfect 10/10-workload sweep on seed 42 and lost badly (−72.5%) on seed 123.

**Lesson worth keeping:** single-seed RL results in this problem are not evidence. Any number that goes in the paper needs a seed sweep.

### 3.3 What fixing the RL bought

Three genuine defects were found and fixed in `common/train_unified_v2.py`:

1. **Time-limit truncation treated as termination** — `done = terminated or truncated` zeroed the value bootstrap at an arbitrary episode boundary that does not exist in deployment.
2. **Unscaled MSE at large Q magnitude** — rewards ~[0,1] with γ=0.99 over 50 steps ⇒ Q targets ≈ 100; squared error there produces very large gradients. Now Huber + gradient clipping.
3. **No target smoothing** — hard copy every ~250 gradient steps. Polyak now available.

Plus an environment fix (§4 of the investigation log): the reward was min-max normalised across memory levels the core-only controller cannot set, compressing the controllable decision space into as little as 63% of the reward range on the RTX 4090. `common/dataset_view.py` restores a full range on all 8 GPUs.

Effect of fixes + tuning, on the corrected view:

| configuration | ss mean | ss std |
|---|---|---|
| plain DQN (no Double) | **0.0685** | ±0.059 |
| γ = 0.50 | 0.0709 | **±0.038** |
| lr 3e-4 | 0.0745 | ±0.070 |
| γ = 0.00 (pure bandit) | 0.0828 | ±0.052 |
| absolute action space | 0.113 | ±0.079 |
| **fixes only, γ=0.99 (starting point)** | **0.132** | ±0.120 |
| dueling head | 0.136 | ±0.097 |

Error and variance both roughly halved. **γ from 0.99 → 0.50 was the single biggest lever**, consistent with the task being near-bandit: reward depends only on the row you land on, never the path taken there. Dueling heads and absolute actions both hurt. Plain DQN beat Double-DQN.

---

## 4. The finding that changes how everything above should be read

### The telemetry is measuring mostly idle time

`collect_dataset.py` sampled NVML across the entire `subprocess.run(...)`: Python interpreter start, `import torch`, CUDA context creation — typically **2-4 seconds of idle GPU** — while the actual timed work is often **0.15-0.6 seconds**.

Evidence from the committed 8-GPU fleet (`python common/analysis/telemetry_dilution.py`):

| observation | value |
|---|---|
| corr(GPU-only duration, recorded utilisation) | **+0.708** |
| points with < 0.5 s real work | 494/799 (62%) — mean recorded util **8.3%** |
| points with > 2 s real work | mean recorded util 40.2% |
| **highest utilisation recorded anywhere in the fleet** | **64.4%** |
| `pt_matmul` (dense GEMM, canonically saturating) | **21.7%** |

Recorded utilisation tracks *how long a workload ran relative to fixed startup cost*, not how hard it drove the GPU. No measurement in an entire 8-GPU fleet ever shows a busy GPU; a dense GEMM reporting 21.7% is not physically credible.

**`Avg_Time_S` is unaffected** — it comes from each benchmark's own `Duration:` line, which brackets only the timed region. **Only the telemetry channels are diluted** — precisely the features a controller needs to characterise an unknown workload.

### Consequences, measured

`python common/analysis/diagnose_features.py`:

| task (leave-one-GPU-out) | result | chance |
|---|---|---|
| "is the current clock below this workload's optimum?" — all 6 features | 0.812 | 0.570 |
| …same, **`core_frac` alone** | **0.781** | 0.570 |
| …same, the other 5 telemetry channels only | 0.696 | 0.570 |
| identify which of 12 workloads | 0.216 | 0.083 |
| identify compute- vs bandwidth-bound regime (binary) | 0.677 | ~0.5 |

An earlier session conclusion — "the features are fine, the problem is RL training" — **was wrong and has been retracted.** The 81% figure is almost entirely `core_frac`: *"I am at a low clock, so I am probably below the optimum."* That is positional information requiring no knowledge of the workload. The five real telemetry channels add ~3 points.

### The fix, and what it costs

All 12 benchmarks now emit `Window: <start> <end>` around their measured region (printed *after* the timed section, so zero measurement overhead). `GPUProfiler` timestamps every sample and `get_stats(window)` averages only in-window samples, falling back safely when no window is present. Verified on a synthetic trace — 3 s idle at 2% util then 1 s of work at 95%: whole-process reports **25.2%**, window-scoped reports **95.0%**. That 25.2% artifact closely matches the real fleet's `pt_matmul` 21.7% and `pt_bert` 25.4%.

**But the fix cannot repair already-collected data.** The 8 committed datasets keep their diluted telemetry.

> **Therefore: the result "telemetry-driven adaptation cannot beat a fixed clock" is currently an artifact of instrumentation, not a demonstrated property of GPU DVFS. It must not be written up as the latter.** Whether adaptation pays is *open*, and answering it requires re-collecting the fleet.

---

## 5. What to do next

### Priority 1 — Re-collect the fleet with the fixed sampler

The single highest-value action. Everything about cross-GPU adaptation is blocked on it.

* The collector is already fixed and committed; no code work needed first.
* Follow the per-instance checklist in [`../context.md`](../context.md) (VM-type vast.ai instances only; container templates deny clock control).
* Re-running 6-8 GPUs takes roughly 2 hours each, and they parallelise — the previous session ran 6 simultaneously.
* **Sanity check before trusting any run:** `pt_matmul` at max clock should now report utilisation near saturation. If it still reports ~20%, the window markers are not reaching the collector — stop and debug rather than collecting a full sweep.
* Consider adding a small number of *longer-running* workload variants so startup is a smaller fraction of each measurement even in the worst case.

### Priority 2 — Re-run the analysis on clean data

All scripts are committed and take minutes to hours:

```bash
python common/analysis/telemetry_dilution.py     # confirm dilution is gone
python common/analysis/diagnose_features.py      # do the features now carry workload signal?
python common/analysis/baselines.py --loo-tune   # does the fixed-clock baseline still dominate?
python common/analysis/sweep_rl.py --preset combos
```

The decisive question: does telemetry-only workload/regime identification rise well above the current 0.677 binary? If yes, adaptation has room to beat the fixed clock and the RL story is alive. If it stays near chance even with clean telemetry, then the honest paper claim is that **static per-GPU clock selection is the right engineering answer for this problem**, which is a legitimate and useful result.

### Priority 3 — Things worth experimenting with (in rough order of expected value)

1. **Longer/steadier workloads.** Several benchmarks do 0.15-0.6 s of work. Even with a correct window, that is few samples at 20 Hz. Increasing iteration counts would improve telemetry SNR at no scientific cost.
2. **Higher sampling rate** during the window (20 Hz → 100 Hz) for the same reason.
3. **Probe-based control.** Perturb the clock, observe the response, infer the regime. Tested on current data and helped only marginally (MAE 0.165 → 0.157), but that test was run on diluted telemetry and deserves a rerun.
4. **Finer core grid.** 8 levels gives a stride of ~0.116 in normalised clock; predictors need sub-one-step accuracy to beat a constant. More levels would reduce quantisation loss.
5. **γ and algorithm choice are settled enough to fix**: γ≈0.5, plain DQN, Huber, no dueling. Do not re-litigate without new evidence.
6. **Memory-clock co-control** — currently excluded by design because mobile parts reject it, but 2/3 of every collected dataset is memory-level data that the core-only controller discards. A 2D desktop-only variant is a legitimate supplementary result.
7. **Hardware validation of the arbiter.** It is unit- and integration-tested but has never run against a real GPU.

### Priority 4 — Known gaps that still block the paper

* Latency: measured 70-90 ms vs a 50 ms budget. Code audit found no obvious Python-side cost (in-process NVML, `torch.no_grad()`, <1 ms/tick measured under mock), so the cost is likely the NVML/driver round-trip. Needs a real GPU to confirm.
* No live hardware A/B under the current unified design.
* Held-out *workload* generalisation not redone under the 6-feature design.
* `CLAUDE.md` §9 lists pre-existing data-quality issues in the older RTX 4500 tree that remain unaddressed (that tree is out of scope for the product but is cited in the paper as a contrast case).

---

## 6. Honest summary for the paper

Three defensible claims, in decreasing order of strength:

1. **Static clock selection is a strong, underrated baseline.** On 8 consumer GPUs across two generations, a fixed clock at 0.6× maximum captures ~88% of the achievable EDP improvement over the stock race-to-idle governor. Any adaptive DVFS method should be required to beat this, and DVFS papers that compare only against pinned-max-clock are overstating their contribution. *(Robust; unaffected by §4.)*

2. **Naive single-seed RL results in this domain are unreliable.** A leave-one-GPU-out DDQN showed 6/8 apparent wins on one seed; a 4-seed sweep reduced that to 2/8 consistent, with per-GPU standard deviations up to ±186%. We also identify three concrete implementation defects (truncation-as-termination, unscaled MSE at large Q, no target smoothing) and a reward-normalisation error, and show that fixing them plus tuning halves both error and variance. *(Robust.)*

3. **Telemetry instrumentation is a first-class threat to validity in DVFS studies.** Sampling NVML across a whole benchmark process rather than its measured region makes recorded utilisation encode startup overhead rather than workload behaviour — here yielding a fleet where no GPU ever appears more than 64% busy and dense GEMM appears 22% busy. We quantify the resulting loss of discriminative power and provide the fix. *(Robust, and a genuinely useful methodological contribution.)*

What we **cannot** yet claim either way: whether a learned, telemetry-driven controller can beat static clock selection. That requires the re-collection in §5.
