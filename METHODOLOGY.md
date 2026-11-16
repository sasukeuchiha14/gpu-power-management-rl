# METHODOLOGY.md — What Was Done, Why, and What Was Rejected

> Every claim in this document is traceable to a file, function, or commit in
> this repository — the working rule this whole project holds itself to
> (`CLAUDE.md` §13). Where an alternative was tried and abandoned, the
> reason is the measured result that killed it, not a guess. See
> [ARCHITECTURE.md](ARCHITECTURE.md) for what the chosen design looks like
> structurally, and [RESULTS.md](RESULTS.md) for the numbers.

---

## 1. Problem framing: why RL, and what else was on the table

**The problem.** The stock NVIDIA driver governor is workload-agnostic: it
boosts to the highest clock the power/thermal budget allows. For workloads
that aren't compute-bound (memory-bound, launch-bound, low-occupancy), the
extra frequency buys little or no runtime reduction while power rises
superlinearly — energy and Energy-Delay Product (EDP = power × time²) get
worse, not better. There's no per-workload, per-phase frequency selection
exposed to userspace.

**Alternatives to RL, and why they weren't the final answer:**

| Alternative | What it is | Why it didn't replace RL |
|---|---|---|
| Fixed clock selection (no learning) | Pick one clock fraction, apply it always | **This is the headline baseline, not a rejected idea** — a tuned fixed clock (0.6× max) captures ~88% of the oracle's gain and beats every *naive* RL configuration this project tried early on (CLAUDE.md §0.3). It's the bar every adaptive method has to clear, and for a long stretch of this project (the pre-response-feature state design) nothing cleared it. Kept as the permanent sanity baseline. |
| Utilisation-threshold heuristic | SM busy → clock up, SM idle → clock down | Implemented (`p_util_heuristic`) as the "obvious thing to try before reaching for RL" — included in every baseline suite run, consistently loses to both the tuned fixed clock and the final learned policy. |
| Hand-coded greedy hill-climb using the measured response signal | Step down, read the measured elasticity the step just produced, stop and reverse if it crosses a threshold | Tried as a *reference ceiling*, not a replacement — see §2. Re-implemented carefully (`p_hillclimb`, `hillclimb_sweep.py`); the learned policy beats it decisively (0.025 vs 0.160 scaled EDP) once the state design that makes both possible was found. |
| Exhaustive/offline search per workload | Try every (core, mem) pair offline, cache the optimum | This is the **oracle baseline** (`p_oracle`), always scores 0 by construction. Useless as a deployable policy (needs the workload identified and profiled in advance — exactly what a 6-feature, workload-agnostic design can't do), but kept as the theoretical ceiling every other number is measured against. |
| Supervised classifier on the same 6 features | "Is the current clock below this workload's optimum?" | Built (`p_supervised`) as a feature-ceiling check, not a deployable policy — it answers a binary question, not a control policy, and was used diagnostically (§3) to establish that the features *do* carry transferable signal before blaming the state design for early RL failures. |
| A different RL algorithm family (PPO, SAC, actor-critic) | Policy-gradient or continuous-control methods | **Never tried.** The action space is a 3-way discrete choice and the problem is near-bandit (γ=0.5 dominates γ=0.99 — see §6), which is exactly DQN's home turf; there was no diagnosed failure mode of DQN/DDQN that would have motivated the added complexity of an actor-critic method. This is a scope decision, not a result — flagged honestly as unexplored rather than "ruled out." |

**Why RL survived all of this.** The optimal (core, mem) pair is
workload-dependent, hardware-dependent, and not analytically known — it has
to be discovered by interaction with measured hardware response. A fixed
clock is close but leaves real headroom (the oracle gap); a hand-coded
heuristic can't generalise its one fixed rule across the genuinely different
response shapes different workloads have. RL is the right tool *once the
state actually carries the signal the policy needs* — which turned out to be
the real, multi-month obstacle (§2, §3).

---

## 2. State design: the central design fork of the whole project

### 2.1 What was tried first: per-workload one-hot identity (REJECTED)

The original two code trees (`gpus/rtx4500_ada/`, `gpus/rtx3050ti_laptop/`,
documented in `CLAUDE.md` §1–§11) used a 23-/16-dimensional state: raw
telemetry (`Core_F`, `Mem_F`, `Avg_Time_S`, `Avg_Pwr_W`, `Peak_T_C`,
`Avg_Util_PCT`) **plus a one-hot workload identity and workload-type
identity**, because the training data included `Avg_Time_S` (not observable
live) and the policy needed to know *which* workload it was looking at to
make sense of the other numbers.

**Why this was rejected, not just superseded:**
1. **It cannot generalise to an unseen workload by construction.** At
   deployment, an unknown app has no one-hot slot — the old daemon handled
   this with a 1-nearest-neighbour lookup against the whole profiled
   dataset to guess a "closest" workload identity, i.e. the deployed policy
   never actually sees a truly novel workload; it sees whatever profiled
   workload resembles it most on 5 observable features. This structurally
   caps what "held-out workload generalisation" can mean for that design —
   see NORTHSTAR row 5's distinction between the 4500's result and the
   current one (RESULTS.md §9).
2. **A network whose input width depends on the number of profiled
   workloads cannot be "one model."** Adding a 13th workload to the
   training set changes the input dimension, which breaks every existing
   checkpoint. This directly conflicts with the "one model" product
   requirement (§8).
3. **Two teammates, two trees, two state designs** — `NORTHSTAR.md`'s
   "Resolving the architecture disagreement" section documents this as a
   genuine design fork that needed a decision, not just a style preference.
   The deciding argument: *"an app that must run on an end-user's GPU with
   an unknown application running has no workload one-hot to fill."* That
   sentence alone rules the one-hot design out for the product this is
   actually meant to be.

### 2.2 What replaced it: 6 device-normalised features, no identity

`common/features.py`'s `FEATURES`: `core_frac`, `sm_util`, `mem_util`,
`power_frac`, `temp_frac`, `vram_frac` — every value a fraction of *that
card's own* NVML-reported limits (probed live by `gpu_probe.py`, never
hardcoded), so `0.7` means the same physical thing on a 70W laptop chip and
a 450W desktop card. No workload one-hot. No `Avg_Time_S` (not observable
live, and its absence is what forces the policy to read behaviour instead
of identity).

**Memory clock (`Mem_F`) was deliberately excluded**, even though it exists
in every profiled dataset and 2/3 of the collected data is memory-sweep
data the core-only controller discards. Reason: mobile GPU parts
(`gpu_probe.py` verified this directly on physical hardware) reject
memory-clock control outright with a driver-level "not supported" error,
while desktop cards support it. A fleet with some cards able to act on a
memory dimension and others structurally unable to breaks having one
unified action space. This is a live, acknowledged trade-off
(`NORTHSTAR.md`'s "Still not a decided design change" section) — not an
oversight, and revisited explicitly rather than silently assumed.

### 2.3 Why the 6-feature design alone still wasn't enough: the diagnosis

Even after the 6-feature redesign and three real RL bug fixes (§6), a
careful leave-one-GPU-out sweep found **RL still lost to the tuned fixed
clock** (`sweep_rl.py --preset combos`: best RL 0.038 vs fixed-clock's
0.025–0.031, winning on only 1 of 5 GPUs). This forced a real diagnostic
question instead of more hyperparameter tuning:

- **A supervised probe on the same 6 features** (`diagnose_features.py`),
  leave-one-GPU-out, predicted "is the current clock below this workload's
  EDP optimum?" at 81–84% accuracy vs a 56% majority baseline — the signal
  **is** present and **does** transfer across GPUs. This ruled out "the
  state representation is just bad" as the explanation.
- **But snapshot telemetry alone can't discriminate workloads across GPUs**:
  binary compute/bandwidth regime ID, leave-one-GPU-out, was only 0.56
  (barely above chance), and `core_frac` alone explained 0.781 of the
  0.812 "below optimum" accuracy — the other five real channels added
  only ~3 points. The agent had almost nothing to discriminate *which
  workload-response-shape* it was dealing with.
- **Unit-to-unit hardware variability makes this worse for absolute
  features specifically**: same-model GPUs measured with genuinely
  different real ceilings (RTX 3060: 2145 vs 2190MHz boost; RTX 3090: 390
  vs 350W power limit, same driver both times) — absolute normalisation is
  fragile against this; a *ratio* measurement is naturally invariant to it.

### 2.4 The fix: response-augmented state (the actual contribution)

**Added 2 features, computed from the measured response to the policy's
own last action**, not from any new sensor: `d_log_runtime_d_log_clock`,
`d_log_power_d_log_clock` (`response_features()`, `common/features.py`).
This is additive — the existing 6 features are unchanged, so the "old"
6-feature agent is exactly the same pipeline with the flag off
(`use_response=False`), not a separate code path, which is what makes the
A/B comparison in RESULTS.md §2 a controlled one and not an apples-to-oranges
comparison.

**Why this specific fix, and not something else:**
- It's a *ratio*, so it's naturally device-invariant — directly addresses
  the unit-variability problem found in §2.3, without needing calibration
  per card.
- It's computed identically in training, offline evaluation, and live
  deployment (`eval_policies.augmented_obs`, the fast training loop, and
  `deploy_live.py`'s `tick()` — three call sites, one function each time,
  not three reimplementations that could silently diverge).
- It required **no new sensor and no architecture change elsewhere** — the
  network input just grows from 18 to 24 dimensions (3-frame stack × 6 or
  8 features).

**Alternative considered and rejected: switch to an absolute action space**
(action *k* directly selects core_frac bucket *k*/(*K*−1), with γ=0 making
the problem the contextual bandit it structurally resembles). Tried in the
15-configuration hyperparameter sweep (§6) — **harmful**. Costs deployment
realism too: the live daemon issues relative steps, so an absolute policy
would need its actuation path changed as well. Kept in `train_unified_v2.py`
(`Cfg.abs_actions`) as a documented, measured-off option, not deleted —
useful for anyone who wants to re-litigate it with new evidence, but not
the shipped design.

---

## 3. Reward design

### 3.1 The metric: Energy-Delay Product, not energy alone

`EDP = Avg_Pwr_W · Avg_Time_S²` — penalises slowdown quadratically, not
linearly, which matters because the whole point of DVFS is *not* to save
power by making everything unacceptably slow. Computed once per profiled
row offline; never approximated live (the live reward signal doesn't exist
at all — deployment runs the fully-trained greedy policy with ε=0, no
online fine-tuning. See §11 for why online fine-tuning stayed out of scope.)

### 3.2 Per-workload min-max normalisation, not a global one

`Reward = 1 − (EDP − EDP_min^workload) / (EDP_max^workload − EDP_min^workload)`.
A **global** min-max across all workloads would let one extreme workload's
EDP range dominate the normalisation and compress everyone else's signal
toward one end — per-workload normalisation gives every workload the full
[0,1] range regardless of its absolute EDP scale. `dataset_view.py`'s
`core_only_view()` additionally renormalises this over *only* the rows a
core-only controller can actually reach, because the naive version (over
every profiled core×memory row, including memory states the controller
can't select) compresses the controllable decision space into as little as
63% of the reward range on a measured RTX 4090 — a bug that was caught by
checking, not assumed away.

### 3.3 Linear vs. log-scale reward — found during hardware validation, not offline

**The problem, found on real hardware, not predicted offline**: a Phase 2
hardware run showed the deployed policy over-aggressively downclocking a
compute-bound workload. The offline reward signal read the *entire*
1890–3135MHz range as near-optimal (Reward 0.994–1.000) for that exact
workload, while real hardware showed a genuine ~14% per-iteration cost
across that same range. Checked how pervasive this was across the whole
fleet (`reward_plateau.py`, no rental needed): 78.3% of all 60 (GPU,
workload) pairs have a Reward ≥0.95 plateau spanning more than 30% of the
clock range — systemic, not a one-off.

**The fix**: `dataset_view.core_only_view()` computes a **second** reward
column, `RewardTrain`, as the same per-workload min-max but over `log(EDP)`
instead of `EDP` — log compresses the single-worst-point domination that
causes the flat near-optimum plateau, while preserving far more *relative*
resolution near the actual optimum. `train_unified_v2.Cfg.use_log_reward`
switches **only the training loop's reward array** to this; evaluation
always scores the unchanged linear `Reward`, so a policy trained on the
log-scale signal is still judged by the same metric as everything else —
this was a deliberate methodological choice to prevent the fix from
silently moving the goalposts.

**Alternative considered: a continuous interpolation knob instead of a
binary switch.** Added as `Cfg.reward_power` — `EDP**reward_power` with
`1.0` identical to the original linear reward and `0.0` identical to
`use_log_reward`. Built specifically so the limit-cycle investigation (§9)
could sweep compression strength without precomputing a new dataset column
per candidate. Swept and **rejected as a limit-cycle fix** — made cycling
*worse* across the board, not better (§9) — but the mechanism (full log
compression trades compute-bound stability for something else) is exactly
what this knob was built to let the project discover cheaply.

### 3.4 Action-shaping reward terms: four tried across the limit-cycle investigation

See §9 for the full case study — summarised here because they're part of
the reward design surface:

| Knob | What it penalises | Verdict |
|---|---|---|
| `action_cost` | Every non-HOLD action, uniformly | Looked promising at 3 seeds, regressed at 10 — the dilution-across-all-moves meant it taxed genuine corrections as much as the oscillation it was meant to suppress. **Rejected.** |
| `reward_power` (partial log) | Indirectly, via a less-sharp optimum | Made cycling **worse**, not better. **Rejected**, and informative — confirmed the mechanism was about decision-boundary sharpness, not reward noise. |
| Deployment-side debounce (not a reward term — a symptom-level deployment guard) | Committing a clock change before N identical consecutive choices | Mechanistically analysed and **rejected**: stretches the cycle's period, doesn't break it, and costs the steady-state metric as N grows. See §9. |
| `reversal_cost` | *Only* choosing the opposite direction from the last non-HOLD move | **Accepted, with an honestly quantified condition-dependent trade-off.** The one candidate that targets the actual mechanism (mutual attraction between two adjacent clock levels) instead of taxing motion in general. See §9 and RESULTS.md §7. |

---

## 4. Environment design: table-backed simulator, not a live dynamics model

`unified_env.py`'s `step()` moves a tracked normalised frequency and snaps
to the nearest profiled row **of the same workload**; reward depends only
on the resulting row, never on the trajectory that reached it.

**Alternative considered: a live GPU-in-the-loop training environment** (run
the actual benchmark under each candidate clock during training, instead of
looking it up in a pre-profiled table). Never seriously pursued — training
runs tens of millions of environment steps across sweeps (`sweep_rl.py`,
the 10-seed response-aware runs, the 300-job `reversal_cost_validate.py`
sweeps); the same training volume against live hardware would cost orders
of magnitude more in both money (GPU rental) and wall-clock time, and would
make every sweep in this repository's `analysis/` folder impractical. The
honest cost of the table-backed choice is that the environment has no
dynamics — it's a stateless contextual-optimisation problem dressed as an
MDP (`CLAUDE.md` §9, issue #16) — which is exactly why real hardware
validation (§10) exists as a *separate*, necessary step rather than being
assumed to follow automatically from simulated results.

**Alternative considered: fit a dynamics/surrogate model** (e.g. a learned
function mapping clock → response, interpolating between profiled points)
instead of nearest-neighbour table lookup. Not pursued — the profiled grid
is already dense (194 core levels on a typical card), nearest-neighbour
snap error is small relative to the grid spacing, and a learned surrogate
would add a second source of approximation error (the surrogate's own fit
error) on top of the policy's, without a diagnosed problem that would
justify it.

---

## 5. Data collection methodology

### 5.1 Profiling protocol

For each (workload, core clock, memory clock) triple: lock clocks via
`nvidia-smi`/`pynvml` (readback-verified, not trusted from the command's
exit text — `gpu_probe.py` found and fixed a real bug here, §5.3), run the
benchmark, sample NVML during the run, record mean power/utilisation/temp
and wall-clock duration, compute EDP. 12 PyTorch/CUDA workloads × up to 194
core levels × up to 5 memory states per GPU (near-full 288-point grids on
most cards; a few Ada cards can't jointly lock their top boost bin with any
tested memory state — the collector's achieved-clock guard correctly
refuses to record a mislabelled point rather than silently keeping a wrong
row).

### 5.2 The telemetry-dilution bug — found, root-caused, and fixed mid-project

**Found**: NVML was sampled across the *whole benchmark subprocess* —
interpreter startup, `import torch`, CUDA context creation, typically 2–4s
of idle GPU — while the timed work itself was often 0.15–0.6s. Recorded
utilisation and power were therefore averages dominated by idle time, not
by how hard the workload actually drove the GPU.

**Measured consequence, not assumed**: across the original fleet,
`corr(duration, utilisation) = +0.708`; 62% of points reported under 9%
utilisation; the highest utilisation *anywhere* in an 8-GPU fleet was
64.4%; dense GEMM (`pt_matmul`) reported 21.7%. This directly explained why
early leave-one-GPU-out RL results looked unstable — `core_frac` alone was
carrying almost all of a supervised probe's accuracy, because the other
five channels were encoding "how long did this process take to start," not
workload behaviour.

**Fixed, not patched around**: every benchmark now emits
`Window: <start> <end>` markers around only the actually-timed work;
`GPUProfiler.get_stats(window)` averages samples inside that window only.
Verified on a synthetic trace (25.2% → 95.0% reported utilisation for the
same underlying load) *and* on real re-collected hardware (`pt_matmul`
peak utilisation jumped from the old 21.7% ceiling to 48–89% across the
re-collected fleet).

**Why this couldn't just be re-weighted or corrected post-hoc**: the bug
contaminates the *input features themselves*, not just a derived statistic
— there's no principled way to recover "what utilisation would have been
measured in-window" from an out-of-window average after the fact. The only
fix was re-collecting the affected GPUs with the corrected sampler.

### 5.3 Clock-lock permission — the one external constraint that shaped the whole rental workflow

Found and verified directly, not assumed: on a cloud GPU marketplace
(vast.ai), **container-type rental instances structurally cannot lock
clocks** ("insufficient permission" on core/mem/power-limit, confirmed
repeatedly, including again this session on an RTX 4070 container tried
for due diligence); **VM-type instances (KVM-based, not a privileged
container) can**, confirmed via `gpu_probe.py`'s readback-verified probe
on multiple cards across two architecture generations (Ampere, Ada).

A second, subtler bug was found and fixed in the probe itself: `-lmc`
(memory clock) and `-pl` (power limit) both report success and print the
requested value **even when persistence mode is off and the change
silently never applies** — confirmed by hand (commanded 5001MHz, driver
still reported 10501MHz, only worked after `nvidia-smi -pm 1`). `-lgc`
(core clock) does not have this problem. `gpu_probe.py` was fixed to enable
persistence mode before testing and read back the *achieved* clock/limit
rather than trusting the command's own text — this is why every "USABLE"
verdict in this project's hardware sessions is described as
"readback-verified," not just "command accepted."

**Practical consequence for methodology**: every real-hardware result in
this project follows the same opening sequence — rent a VM-type instance,
smoke-test fast (bandwidth, a bounded `cuInit()` call, then
`gpu_probe.py`) *before* installing anything or running a long experiment
— because even nominally-VM-type hosts have failed this check roughly 1
in 9 times in this project's rental history, and a CUDA-context-init hang
on one host wasn't caught by NVML-only testing alone.

---

## 6. RL training methodology: bug fixes, then a hyperparameter sweep, in that order

### 6.1 Three real defects fixed before any tuning (`train_unified_v2.py`, vs the original `train_unified.py`)

| # | Defect | Why it mattered | Fix |
|---|---|---|---|
| 1 | Time-limit truncation treated as termination | `done = terminated or truncated`, bootstrap zeroed by `(1−done)`. This env never terminates; `truncated` fires at `max_steps=50`, an arbitrary training-time boundary that doesn't exist in deployment. Every episode's final transition taught the agent value vanishes at a wall that isn't real. | Bootstrap through truncation; only zero on true termination (which never happens here, so the bootstrap is always kept alive). |
| 2 | Unscaled MSE at large Q magnitude | Reward ~[0,1], γ up to 0.99 over 50 steps ⇒ Q targets approach ~100; squared error there produces very large gradients. | Huber (smooth-L1) loss + gradient-norm clipping, both standard DQN stabilisers the original trainer omitted. |
| 3 | No target-network smoothing | Hard-copy every 5 episodes (~250 gradient steps apart) makes the regression target lurch. | Optional Polyak (soft) update added (`Cfg.tau`), available but not required by the settled configuration. |

**Why fix these before tuning, not alongside it**: a hyperparameter sweep
run against a trainer with these bugs would be measuring "which
configuration best compensates for broken bootstrapping," not "which
configuration is actually best" — the sweep in §6.2 was only trustworthy
once these were fixed first.

### 6.2 15-configuration hyperparameter sweep (`sweep_rl.py --preset screen`)

Settled, not re-litigated without new evidence (`CLAUDE.md` §0.2):

| Knob | Finding | Why |
|---|---|---|
| Discount γ | **0.5 ≫ 0.99**, the single biggest lever | The task is near-bandit — reward depends only on the row landed on, never the path taken to reach it (table-backed env, §4) |
| Double-DQN | Slightly **harmful** here | Plain DQN wins; the usual overestimation-bias argument for Double-DQN didn't pay off in this near-bandit regime |
| Dueling head | **Harmful** | Not pursued further once measured worse, no mechanism was chased |
| Absolute action space | **Harmful** | See §2.4 — also costs deployment realism |
| Loss | Huber + grad-clip | Carried over from the bug-fix pass (§6.1), confirmed still best in the sweep |
| Learning rate | 3e-4 marginally better than 1e-3 | — |
| Network width | 256 ≈ 128 | No measurable benefit from doubling width; 128 kept for the smaller, faster-training network |

**Caveat carried forward honestly**: every one of these was measured on
pre-telemetry-fix data (`CLAUDE.md` §0.2's own caveat). The relative
*ordering* is treated as probably robust (it concerns optimisation
dynamics, not feature content), but is flagged, not silently assumed to
still hold, and wasn't blindly re-confirmed after the fix either — it's an
open item, stated as such.

---

## 7. Evaluation methodology

### 7.1 Two metrics, always reported together

- **`traj`** (trajectory mean): mean scaled EDP over the whole rollout,
  including the transient while a relative-action controller walks from
  its start point to wherever it settles. Penalises slow convergence.
- **`ss`** (steady-state mean): mean scaled EDP over the last 40% of the
  trajectory — what a controller actually delivers once converged, which
  is what matters in a deployment that runs for hours, not 50 steps.

Static baselines (governor, fixed clock, oracle) are constant by
construction, so `traj == ss` for them; the distinction only bites for
learned or heuristic controllers, which is exactly where it matters most.

### 7.2 Leave-one-GPU-out, not train/test on the same data

Every generalisation claim in this project trains on *N*−1 GPUs and
evaluates on the held-out one, never the reverse. This was not always true
— `CLAUDE.md` §9 issue #18 documents that the *original* per-card trees
trained and evaluated on the same 12,000/105 rows, which the project
explicitly flags as "no generalisation claim is supported" for that design.
The unified architecture's entire evaluation discipline exists specifically
to fix this.

### 7.3 Leave-one-workload-out — built this session, same discipline applied to a different axis

`common/analysis/workload_holdout_eval.py`: for each workload, the policy
trains on all 5 GPUs but with that workload's rows removed from *every*
GPU's pool, then is evaluated only on it. **The baseline calibration
mistake caught and fixed before trusting the first result**: the tuned
fixed-clock comparison must be tuned on the training workloads only, never
on the held-out one — an earlier draft of this exact script swept the
fixed fraction directly against the held-out workload's own data, which
lets a "fixed clock" baseline cheat its way to near-oracle performance on
a single-workload evaluation (caught on the very first smoke test, before
the full sweep ran — see RESULTS.md §9 for the corrected numbers).

### 7.4 The paired comparison, not the raw mean, is load-bearing

Repeatedly in this project, two policies' raw means look close while their
*same-seed, same-held-out-GPU* paired comparison tells a sharper story —
e.g. the response-aware vs. ablation result: means of 0.0250 vs 0.0375
(not dramatically far apart in isolation), but paired win rate 38/50
(p≈0.00015, one-sided binomial). The ablation's own variance (std 0.0367,
*larger* than its mean) is itself evidence that the raw mean is a noisy
statistic here; the paired comparison, which cancels out per-seed/per-GPU
noise common to both policies, is the number actually trusted.

**Found and fixed along the way**: `evaluate_policy()`'s "fixed" per-
workload evaluation start state used Python's built-in `hash()`, which is
salted per-process (`PYTHONHASHSEED`) unless pinned — so a function
documented as giving a fixed, reproducible start state silently wasn't
reproducible across separate process runs. Replaced with `zlib.crc32`
(deterministic across processes and Python versions). This was caught
while debugging an apparently-stuck hill-climb trajectory that only
reproduced in *some* invocations — a methodology bug surfaced by a
substantive result looking wrong, not found by code review alone.

### 7.5 The full baseline suite, and why each exists

| Baseline (`eval_policies.py`) | Purpose |
|---|---|
| `p_oracle` | Theoretical ceiling (always 0 by construction) |
| `p_governor` | Pinned max clock — the closest honest proxy for "no management at all" this table-backed design can express (see the explicit caveat below) |
| `p_minclock` | The opposite extreme, for sanity-bounding |
| `p_fixed(frac)` | The real bar to clear — tuned leave-one-GPU-out, never on the held-out fold |
| `p_random` | Catches "is the agent doing anything better than noise" |
| `p_util_heuristic` | The non-RL thing a reviewer will ask about first |
| `p_supervised` | A feature-ceiling diagnostic, not a deployable policy (§2.3) |
| `p_hillclimb` | A reference for "does RL beat the best simple rule," not a baseline to simply beat |
| `p_agent` | Wraps any trained network (DDQN or an ablation) into the same evaluation protocol as every static policy |

**Named caveat, carried in every relevant table**: `p_governor` is *pinned
max clocks*, not the NVIDIA driver's actual adaptive default governor —
this project never implemented or measured the real driver governor, and
says so explicitly rather than letting "governor" imply something stronger
than what was tested.

---

## 8. "One model," not a per-card weight library

This was the original plan — detect the GPU model, load a matching
pre-trained profile. Reversed, deliberately, for three reasons
(`NORTHSTAR.md`'s "Why one model, not per-card weights"):

1. The one-hot state design *needed* per-card weights (normalisation ranges
   and workload lists differed per card's profiling run). The 6-feature
   redesign removes that need by construction (§2.2) — the same trained
   network is already speaking a language every card understands.
2. Per-card weights don't generalise by construction — it's a lookup table
   with extra steps, and says nothing about whether the *idea* the network
   learned transfers. A unified model that works on a genuinely held-out
   card is the real scientific result a per-card library sidesteps.
3. Maintenance: a per-card library means testing and shipping a new file
   every time a new GPU model exists. One file doesn't have that problem.

The only per-machine step the deployed system performs is reading *that
card's own* hardware limits from the driver at startup (`gpu_probe.py`) to
normalise live readings — never a different network, never different
weights.

---

## 9. Case study: the limit-cycle investigation — a worked example of this project's methodology

This is included in full because it's the clearest single example of every
methodological principle above applied in sequence, and because it's the
most recent, most rigorously hardware-confirmed result in the project.

**The finding**: the log-scale-reward checkpoint (§3.3's fix for the
reward-plateau problem) genuinely *oscillates* between two adjacent clock
levels on most workloads — confirmed via a 100-step deterministic rollout
from max clock with **zero noise needed**, not a flip-flop induced by
observation noise (a noise-injection proxy was tried first, gave backwards
predictions relative to the known hardware result, and was discarded
rather than trusted because it was convenient).

**Four candidates tried, in order, each with a stated reason it might
work and a measured reason it did or didn't**:

1. **`reward_power`** (partial log compression instead of full log) —
   hypothesis: less sharp decision boundary near the optimum means less
   incentive to flip. **Made cycling worse across the board.** Rejected,
   and the failure is informative: confirms the mechanism isn't about
   boundary sharpness in the way hypothesised.
2. **`action_cost`** (flat penalty on every non-HOLD move) — hypothesis:
   discourage the back-and-forth directly. **A textbook case for why this
   project insists on two-stage validation**: looked like a clear winner in
   a fast 3-seed screen (cycle rate 0.22, best of 8 candidates), but a
   seed actually trained to the full 5-GPU/10-seed protocol cycled on 9/12
   workloads — barely better than the 10/12 it was meant to fix. Rejected
   on the full-scale result, not the screen.
3. **Deployment-side debounce** (require N consecutive identical
   non-HOLD choices before committing a clock change) — hypothesis:
   pure symptom suppression, no retraining needed, can't make the reward
   signal worse since it doesn't touch training at all. **Mechanistically
   analysed and rejected**: inspecting the raw rollout traces directly
   (not just the aggregate cycle count) showed debouncing doesn't break the
   cycle, it stretches its period — a frozen setpoint feeds the policy an
   identical "neutral response" observation every tick, so it keeps
   re-choosing the same action until the streak hits N, commits one step,
   the response sign flips, and the same thing happens in reverse. Checked
   out to N=40 over a 300-step rollout specifically to see whether a long
   enough streak requirement eventually breaks the attractor instead of
   just slowing it — it doesn't; the floor (9/12 cycling) is identical
   from N=3 to N=40, while the aggregate cost metric gets *worse* as N
   grows. No hardware rental was spent confirming this: the offline
   deterministic rollout is exhaustive and noise-free, more rigorous for
   this specific question than a hardware trial would be.
4. **`reversal_cost`** (penalise only choosing the opposite direction from
   the last non-HOLD move) — hypothesis: target the actual mechanism (a
   mutual attraction between two adjacent levels) instead of taxing motion
   generally the way `action_cost` did. **Worked, with a real cost, found
   by the same two-stage discipline**: a 3-seed screen's favourite value
   (0.05) did *not* hold up at full 5-GPU/10-seed scale (the same lesson
   `action_cost` already taught, re-confirmed rather than assumed not to
   apply here) — the real sweet spot (0.30) was found by sweeping the full
   protocol across six values and picking by paired comparison, not by
   trusting the cheap screen.

**Then a real self-correction, documented rather than hidden**: the first
full-protocol pass reported the fix as "free" — settling at the same safe
clock point as the pure-log checkpoint on the one GPU checked. That
single-seed, single-GPU comparison was itself exactly the kind of
unverified trust this methodology exists to catch. Testing a *rejected*
combo candidate surfaced a real failure mode (a checkpoint pinned at exactly
max clock on 3 of 5 GPUs), which forced a proper re-check of the accepted
candidate too — full 5-GPU/10-seed, against a pure-log baseline that had
never actually been measured on 4 of the 5 GPUs before. The honest result:
the settle point rises on *every* GPU under the fix, a real, quantified
cost in downclocking aggressiveness, not the clean win first reported.

**Epilogue (13–14 Nov 2026): the trade-off was a property of reward shaping, not of the problem.**
Four more reward-shaping variants (`reversal_window`, `hold_bonus`, `reversal_util_scale`) again
looked good in a 3-seed offline screen and failed on hardware, the same screen-overstates lesson as
`action_cost`. Reading the hardware logs instead of tuning further gave two findings: the cycle
itself is cheap (~5% energy/iter), and the do-no-harm arbiter drives much of the compute-bound
"cycle" (active ~61% of ticks), so a penalty on policy reversals could never reach it. The fix that
worked sits at deployment: detect a short cycle, probe each visited level for a clean EDP, park at
the best one, release on EDP degradation or work shift (`deploy_live.py --lockin-k`). Each component
was added because a hardware run exposed a specific failure (telemetry lag making max clock look
falsely cheapest, parking at the wrong clock after a phase change, probe cost on slow workloads).
Method lesson: when a fix keeps failing at the layer where you tried it, check which component is
actually producing the symptom before tuning further. Writeup: [oscillation_update.md](oscillation_update.md).

**Then three rounds of real-hardware confirmation**, because offline
cycle-rate and cost metrics — however rigorously multi-seeded — are still a
simulation of a simulation (the table-backed environment, §4) until
confirmed on physical silicon. Round 1 confirmed the fix completely
eliminates cycling on real hardware (flip_rate exactly 0.000, every rep,
both conditions) but costs the entire power-saving benefit on the
compute-bound condition specifically. Round 2 tested two more penalty
strengths and found neither offline-predicted result held up cleanly on
hardware — one catastrophically failed in a way that looked like single-
seed noise. Round 3 tested that hypothesis directly with two more seeds and
found the opposite: the same config, different seeds, gave *opposite*
outcomes on the compute-bound condition (one fixes cycling and loses the
benefit, the other keeps the benefit and doesn't fix cycling) — which is
the signature of a real, mechanistically consistent trade-off, not noise
that more seeds would average away. This is reported as a closed, honestly
characterised trade-off (full numbers in RESULTS.md §7), not chased
further once the pattern repeated identically five independent times.

**Why this case study matters methodologically, not just as a result**:
every piece of this project's stated discipline shows up in it in one
place — two-stage (screen then full-protocol) validation before trusting
a number, checking a result against the *mechanism* rather than only the
aggregate metric, a real self-caught overclaim corrected rather than left
standing, and real hardware as the final arbiter rather than simulation
alone.

---

## 10. Hardware validation and reproducibility practices

- **Every reward-shaping or action-shaping change that reached a
  conclusion in this project was eventually confirmed on real hardware**,
  not trusted from simulation alone — the log-reward fix, the
  reward-plateau fix, and the limit-cycle investigation all followed
  offline validation with a real rented-GPU trial.
- **Raw hardware logs are committed to the repository**, not just
  summarised in prose — `docs/results/*_phase2_hardware/` holds the actual
  per-tick `deploy_live.py` output for every hardware session, force-added
  past the repo's own `*.log` gitignore rule specifically because these
  logs are primary data, not scratch output.
- **Every analysis script is a standalone, re-runnable record** of the
  experiment that produced a reported number — `common/analysis/*.py`,
  one file per experiment, each with a docstring stating what it tests,
  what it found, and the exact command to reproduce it. This is why a
  claim like "0.0051, 4.7× better than fixed-clock" could be flagged as
  **unverifiable and explicitly not cited** when the throwaway script that
  produced it turned out never to have been committed — the absence of a
  reproducible script is itself treated as evidence against trusting the
  number, not an oversight to quietly paper over.
- **Repeats, not single trials, for anything reaching a reported
  conclusion** — this project was burned early by single-trial hardware
  results (`context.md`'s "repeated trials reveal the reward-fix is a
  trade-off, not a clean win" entry) and by small-seed-count offline
  screens that didn't hold at full scale (§9, twice). The standing rule
  (`context.md`'s working agreements): *"Single-seed results are not
  evidence here."*
- **Every rented instance is destroyed after use**, clocks reset first on
  every exit path — confirmed via normal exit, `SIGINT`, and `SIGKILL`
  across every hardware session this project has run, and verified with a
  zero-instance marketplace check after each one.

---

## 11. Explicitly out of scope, and why

| Item | Why it's out of scope |
|---|---|
| Online fine-tuning on live hardware | The deployed daemon runs the fully-trained greedy policy (ε=0); a live reward signal and online update loop were never designed, because the offline table-backed training + hardware-validated deployment split (§4) was judged more tractable and more measurable within this project's scope. Flagged as future work, not attempted and abandoned. |
| RTX 4500 architecture unification | A professional workstation card, not a 3000/4000 consumer part — explicitly outside the product definition (`NORTHSTAR.md`). Kept only as a supplementary contrast case for memory-clock control, which consumer cards can't do. |
| Continuous action space / finer clock resolution | Would need a denser profiled grid than currently collected to be meaningful (action strides are already coarser than the grid, `CLAUDE.md` §9 issue #19) — a data-collection cost, not a modelling one, and not pursued without a diagnosed need for finer control. |
| Power-limit co-control | This project controls frequency only (`-pl` never used in the deployed path) — a deliberate scope boundary stated from the start, not a gap found later. |
| Comparison against a published DVFS method from the literature | A literature check (done, not skipped) found real prior art for both closed-loop DVFS under a performance constraint and cross-GPU power-model transfer — acknowledged as reducing the novelty claim to "the specific response-aware mechanism and its cross-GPU validation," not "closed-loop learned DVFS" in general. No direct quantitative comparison against a specific published baseline was implemented. |
