# Why leave-one-GPU-out generalization was unstable — investigation log

Started from commit `8b68617`: a 4-seed leave-one-GPU-out sweep showed only 2/8
held-out GPUs generalizing reliably, with per-seed standard deviations often
exceeding the mean effect (RTX 3080: −54.8% ± 186%). This file records what was
actually measured, in the order it was measured, including the results that
killed earlier hypotheses.

All numbers below are **steady-state scaled EDP** (0 = that workload's best
profiled operating point, 1 = its worst), averaged over the held-out GPU's
workloads, leave-one-GPU-out. Lower is better.

---

## 1. First read: "the state representation is not the bottleneck" — LATER SHOWN WRONG

Kept here because the correction is the point. See §7: an ablation showed this
conclusion did not survive, and the reason it looked convincing is instructive.

A supervised probe on the same 6 features, leave-one-GPU-out, predicting
"is the current clock below this workload's EDP optimum?":

| held-out GPU | majority-class | logistic reg | random forest |
|---|---|---|---|
| mean over 8 GPUs | 0.561 | 0.810 | **0.838** |

81–84% vs a 56% majority baseline, on every held-out GPU. This *looked* like
proof that the features carry transferable workload signal, and the
investigation proceeded on that basis. It was wrong — see §7.

## 2. Two environment defects, one of them damaging

`collect_dataset.py` profiles 8 core × 3 memory levels. The controller is
core-only by design. The env snaps on `core_frac` alone, and the tie-break is
stable dataframe order, which is memory-major from the collector.

* **Dead data.** Snapping always lands on the top memory state, so 2/3 of every
  dataset is unreachable in training and evaluation.
* **A crushed reward scale — this is the damaging one.** `Reward` was min-max
  normalized across all 24 rows, including the memory dimension the controller
  cannot set. Memory clock moves EDP more than core clock does on these cards,
  so the controllable decision space was compressed into part of the range.
  Measured reachable reward span before the fix: **0.628 (RTX 4090) to 0.908**;
  on RTX 3060 / `pt_vector_add` the entire reachable set spanned Reward
  0.742–1.000 — the top 26% of the scale.

`common/dataset_view.py` restores a full 1.000 span on all 8 GPUs by
re-normalizing over reachable rows only. The 2D data stays on disk for a future
2D controller; this is a view, not a migration.

## 3. Three real defects in the v1 agent/trainer

Fixed in `common/train_unified_v2.py`:

1. **Time-limit truncation treated as termination.** `done = terminated or
   truncated`, and the bootstrap was zeroed with `(1 - done)`. `terminated` is
   always `False` in this env; `truncated` fires at `max_steps`. Every episode's
   final transition taught the agent that value vanishes at an arbitrary
   boundary that does not exist in deployment.
2. **Unscaled MSE at large Q magnitude.** Rewards ~[0,1] with γ=0.99 over 50
   steps ⇒ Q targets approach ~100. Squared error there produces very large
   gradients. Now Huber + gradient-norm clipping.
3. **No target smoothing.** Hard copy every 5 episodes (~250 gradient steps
   apart). Polyak now available.

## 4. What the fixes plus tuning actually bought

Screening sweep, 15 configs × 4 held-out GPUs × 2 seeds, on the corrected view:

| config | ss mean | ss std |
|---|---|---|
| no_double (plain DQN) | **0.0685** | ±0.059 |
| gamma 0.50 | 0.0709 | **±0.038** |
| lr 3e-4 | 0.0745 | ±0.070 |
| hidden 256 | 0.0752 | ±0.067 |
| gamma 0.00 (bandit) | 0.0828 | ±0.052 |
| 3000 episodes | 0.0844 | ±0.056 |
| absolute actions (K=8) | 0.113 | ±0.079 |
| **v2 fixes only (γ=0.99)** | 0.1324 | ±0.120 |
| dueling | 0.1357 | ±0.097 |

Error roughly halved and variance roughly halved. Lowering γ from 0.99 to 0.5
was the single biggest lever, consistent with the task being near-bandit rather
than long-horizon. Dueling and an absolute action space both hurt. Plain DQN
beat Double-DQN.

## 5. The baseline that reframes the result

Baselines were missing entirely (CLAUDE.md §8-D). Added in
`common/eval_policies.py` under one protocol:

| policy | mean ss EDP |
|---|---|
| oracle (per-workload best) | 0.000 |
| **fixed clock at 0.6 × max** | **0.037** |
| supervised RF classifier walk | 0.051 |
| fixed 0.5 | 0.058 |
| best RL config | ~0.069 |
| **governor (pinned max clock)** | **0.319** |
| random walk | 0.278 |
| utilization heuristic | 0.573 |
| min clock | 0.819 |

A **single static clock at 60% of maximum**, using no telemetry and no
learning, captures ~88% of the oracle's gain over the max-clock governor, and
beats every learned policy tested. The tuned-fixed baseline was also verified
leave-one-GPU-out (tuned only on the 7 training GPUs): it selects 0.60 on all
8 folds, so this is not hindsight fitting.

## 6. Why adaptation loses — the decomposition

| quantity | value |
|---|---|
| predict optimal `core_frac` from single-point telemetry (LOO MAE) | 0.165 |
| …adding the observed response to a clock change (probe) | 0.157 |
| …just predicting the global mean optimum (0.534) | 0.199 |
| **one clock grid step** | **0.116** |
| identify which of 12 workloads from telemetry (LOO) | **0.216** (chance 0.083) |
| within-workload optimum std across GPUs | 0.118 |
| between-workload std of mean optimum | 0.196 |

The predictor is typically ~1.4 grid steps off, i.e. worse than the resolution
it needs, and only modestly better than ignoring telemetry. Even a regressor
that jumps straight to its predicted optimum scores 0.045 vs the fixed clock's
0.037. Probing (perturbing the clock and observing the response) helps only
marginally.

**But the optima are strongly clustered by regime**, and the compute-bound
cluster is remarkably tight across GPUs:

| workload | mean opt | std across 8 GPUs |
|---|---|---|
| pt_bert | 0.759 | 0.046 |
| pt_resnet50 | 0.769 | 0.067 |
| pt_matmul | 0.764 | 0.075 |
| … | | |
| pt_transformer_decode | 0.282 | 0.148 |
| pt_crypto_proxy | 0.288 | 0.098 |
| pt_vector_add | 0.300 | 0.135 |

Naming the workload is hopeless; identifying the *regime* is a binary question
and much easier. That is the next thing tested.


---

## 7. CORRECTION, and the actual root cause: startup-diluted telemetry

### 7a. The §1 probe was measuring position, not workload

Ablating the feature set on the same leave-one-GPU-out task
("is the current clock below this workload's optimum?"):

| feature set | LOO accuracy |
|---|---|
| majority-class baseline | 0.570 |
| **all 6 features** | **0.812** |
| `core_frac` only | **0.781** |
| the other 5 (no `core_frac`) | 0.696 |

`core_frac` — where you currently sit on the clock range — alone gets 0.781.
All five real telemetry channels together add **3.1 points**. The §1 result was
almost entirely "low clock ⇒ probably below optimum", which requires no
knowledge of the workload at all. The telemetry carries very little
transferable workload signal, and §1's conclusion is retracted.

Consistent with that, everything requiring actual workload characterisation
fails:

| task (leave-one-GPU-out) | result | chance |
|---|---|---|
| identify which of 12 workloads | 0.216 | 0.083 |
| identify compute- vs bandwidth-bound regime (binary) | **0.653** | ~0.5 |

A regime-classifier policy (predict the regime, then use that regime's mean
optimum measured on training GPUs) scores 0.093 (K=2) / 0.068 (K=3) against
the fixed clock's 0.037 — it loses, because at 65% binary accuracy it picks the
wrong regime often enough to give back more than adaptation gains.

### 7b. Why the telemetry is uninformative — a measurement bug

`collect_dataset.py` ran its 20 Hz NVML sampler across the entire
`subprocess.run(...)`: interpreter startup, `import torch`, CUDA context
creation — typically 2–4 s of idle GPU — while the actual timed work is often
0.15–0.6 s. Reported utilisation and power are therefore averages dominated by
idle time.

Evidence from the committed 8-GPU fleet:

| observation | value |
|---|---|
| corr(GPU-only duration, reported utilisation) | **+0.708** |
| profiled points with < 0.5 s of real GPU work | 494/799 (62%) |
| …their mean reported utilisation | **8.3 %** |
| points with > 2 s of real work, mean reported utilisation | 40.2 % |
| **highest utilisation recorded anywhere in the fleet** | **64.4 %** |
| `pt_matmul` (dense GEMM, canonically compute-saturating) | 21.7 % |

Reported utilisation tracks *how long a workload ran relative to fixed startup
cost*, not how hard it drove the GPU. No measurement in the entire fleet ever
shows a saturated GPU, which for dense GEMM is not physically credible.

Note the timing column is NOT affected: `Avg_Time_S` comes from each
benchmark's own `Duration:` line and correctly excludes startup. Only the
telemetry channels are diluted — which is exactly the set of features the
controller must use to characterise a workload.

### 7c. Fix

* All 12 benchmarks now print `Window: <start_ts> <end_ts>` around their
  measured region, emitted *after* the timed section so it adds no measurement
  overhead.
* `GPUProfiler` timestamps every sample; `get_stats(window)` averages only
  in-window samples, falling back to whole-process averaging when a window is
  absent or too small.
* Verified against a synthetic trace (3 s idle at 2% util, then 1 s of work at
  95%): whole-process average reports 25.2%, window-scoped reports 95.0%. The
  25.2% artifact closely matches the real fleet's `pt_matmul` 21.7% and
  `pt_bert` 25.4%.

### 7d. What this means for the conclusion

The negative result in §5–§6 — telemetry-driven adaptation cannot beat a fixed
clock — is, on the current data, **an artifact of instrumentation and not a
demonstrated property of GPU DVFS**. It should not be written up as the latter.

The fixed-clock finding itself still stands and is independent of this bug: it
depends only on EDP and timing, both of which are correctly measured. A static
clock at 0.6x max still captures ~88% of the oracle's gain over the max-clock
governor.

**Deciding this properly requires re-collecting the fleet with the corrected
sampler.** The datasets committed so far cannot answer whether adaptation pays,
because their telemetry columns do not carry the information adaptation needs.
