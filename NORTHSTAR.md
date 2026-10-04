# North Star — one model, any RTX 3000/4000 GPU

This is the single definition of done, and the scorecard used to check whether a change
moved us toward it or not. Check work against this document, not against intuition.

Published version (nicer to read, same content): the artifact kept in sync with this file.
Ask Claude for the current link if you don't have it, or regenerate from this file.

---

## The end goal, stated so it can be tested

**One** DDQN policy, trained on several RTX 3000/4000 consumer GPUs, that runs on a
3000/4000 GPU it has **never seen**, reads **only live NVML telemetry**, and **reduces
energy without meaningfully slowing the workload** — proven on a held-out GPU and on
held-out workloads, on Windows and Linux.

Every criterion below follows from that sentence. If a piece of work does not move one
of these rows, it is not on the critical path.

**What "one model" means concretely:** a single trained weights file, same network shape,
loaded unmodified on any card. Not a library of per-GPU weight files selected by detecting
the card model — that was the original plan, and it's been superseded (see
[why below](#why-one-model-not-per-card-weights)). The only per-machine step allowed is
reading *that card's own* hardware limits from the driver at startup (its real max clock,
power ceiling, temperature threshold) to normalize live readings — never a different
neural network, never a different set of learned weights.

---

## Scorecard — are we going in the right direction?

Re-check these eleven rows regularly. This **is** the results section of the paper.

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | One architecture across every GPU | 🟡 Half | 3050 Ti runs the 6-feature device-normalized state. RTX 4500 still runs 28 features with a workload one-hot. Must converge on `common/features.py`. |
| 2 | Runtime measured correctly everywhere | 🟡 Half | 3050 Ti now parses GPU-only duration — fixed the physics, produced a real compute-vs-bandwidth gradient. The 4500 collector still times the whole subprocess. |
| 3 | Four or more in-scope GPUs profiled | 🟢 8 of 4+ done | 3050 Ti (laptop) has real data. **7 desktop GPUs now do too**: RTX 3060, 3070, 3080, 3090 (Ampere) and RTX 4070, 4070 Ti, 4090 (Ada) — each a full 288-point (or near-full, where a card's max boost-clock bin isn't jointly lockable with any tested memory state — 4090: 252/288, 4070: 256/288, both cleanly explained) 2D core+memory sweep, 12 workloads, individually audited across 7 dimensions (header integrity, exact row count, zero duplicates, EDP recompute, physical-range sanity, per-workload trend sanity, quick-vs-full cross-check). Target was 3070/3080/3090 + 4060/4090; landed 4070/4070 Ti instead of 4060 (both valid Ada-tier substitutes), so target exceeded. Row unblocked. |
| 4 | Held-out GPU generalization (leave-one-GPU-out) | 🔴 Currently unanswerable — telemetry is contaminated | A 4-seed leave-one-GPU-out sweep (32 train+eval runs) found only **2/8** GPUs generalising consistently (RTX 4070 +33.6%±24.7%, 4070 Ti +72.1%±2.7%); the other 6 flip between clear win and clear loss on random seed alone, with per-GPU std up to **±186%** (RTX 3080). Fixing three real RL bugs plus a reward-normalisation error and tuning (γ 0.99→0.50 biggest lever; dueling and absolute actions hurt; plain DQN beat Double-DQN) roughly halved both error (0.132→0.069) and variance — but every configuration still lost to a fixed 0.6× clock (0.037). **Then the root cause turned up and it invalidates the reading:** the telemetry features these policies consume were sampled across whole benchmark processes including 2–4 s of startup, so they encode startup overhead rather than workload behaviour (see row 10). This row cannot be resolved until the fleet is re-collected. Evidence: `common/holdout_results/`, `common/analysis/sweep_rl.py`. |
| 5 | Held-out workload generalization | 🟡 Redo | Done on the 4500, but under the old one-hot architecture (unseen app gets a borrowed identity via nearest-neighbor match). Must be redone under the 6-feature design. 3050 Ti zero-shot workloads are written but untested. |
| 6 | No unacceptable slowdown | 🟡 Guard implemented and tested, not hardware-validated | `gpus/rtx3050ti_laptop/deployment/arbiter.py` (`DoNoHarmArbiter`) forces max clock for `ARB_COOLDOWN` ticks after `ARB_PATIENCE` consecutive ticks of (high SM utilisation + below-max clock) — the always-observable signature of holding back a compute-bound GPU, which is the 14–15%-slower-for-12%-less-power failure mode. 7 unit tests on the decision logic (trip timing, cooldown length, non-consecutive reset, disabled passthrough) plus a mocked-NVML integration test driving the real `LiveController` with the committed trained weights. **Never exercised on real hardware.** Note its trigger signal is SM utilisation, which row 10 shows is currently mis-measured offline — live NVML reads are instantaneous and unaffected, but the threshold (85%) was chosen against intuition, not against clean profiled data, and should be revisited after re-collection. |
| 7 | Safety net actually present and working | 🟡 Restored, not hardware-validated | The deleted 3050 Ti arbiter is back (see row 6), redesigned rather than ported: the old RTX 4500 version reconstructed live EDP by nearest-neighbour lookup against the profiled dataset, which needs workload-identifying features `common/features.py` deliberately excludes — reusing it would have silently reintroduced the closed-world assumption the redesign removed. `DRY_RUN` also now defaults to `True` (it shipped as `False`, actuating real hardware on launch, contradicting `CONTRIBUTING.md`). The 4500's own arbiter is untouched. Real-hardware validation still outstanding. |
| 8 | Latency and overhead within budget | 🔴 Failing, unchanged (needs real hardware) | 70–90 ms mean vs a 50 ms budget still stands; could not be re-measured (no rented GPU available). Code audit found no obvious Python-side cost — `set_clocks()`/`_read_telemetry()` already use in-process `pynvml` with no subprocess spawn on the normal path, `agent.act()` already uses `torch.no_grad()`, and a mocked-NVML timing check put Python+torch overhead under 1 ms/tick including the new arbiter. If 70–90 ms holds on real hardware the cost is most likely the NVML/driver round-trip itself, which is a different fix than 'remove subprocess calls'. |
| 9 | Reproducible training evidence | 🟡 Partial | Reward-history curves are now saved by both `train_ddqn.py` and `train_unified.py`, and `--seed` exists on the unified trainer. Still missing: multi-seed runs and error bars — nobody's run more than one seed yet. |
| 10 | **Telemetry measured over the right window** | 🟡 5 of 8 GPUs re-collected (3 Oct 2026) | `collect_dataset.py` sampled NVML across the whole benchmark subprocess — interpreter start, `import torch`, CUDA context creation, typically 2–4 s of idle GPU — while the timed work is often 0.15–0.6 s. Measured across the original fleet: `corr(GPU-only duration, recorded utilisation) = +0.708`; 62% of points (494/799) report under 9% utilisation; the highest utilisation *anywhere* is 64.4%; dense GEMM reports 21.7%. Consequence, measured on that original fleet: `core_frac` alone explains 0.781 of the 0.812 leave-one-GPU-out accuracy on 'is the clock below optimum?', so the five real telemetry channels add ~3 points; 12-way workload ID is 0.216 (chance 0.083) and binary regime ID 0.677 — **all three numbers are stale now that the fleet is partially re-collected and must be re-measured once it's complete.** **Fixed in code** (all 12 benchmarks emit `Window: <start> <end>`; `get_stats(window)` averages in-window only; verified on a synthetic trace: 25.2% → 95.0%). **Now also verified on real re-collected hardware**: RTX 3060/3070/3080/3090/4090 re-profiled with the fixed sampler and audited clean (exact row counts, zero duplicates, EDP recompute, physical range) — highest utilisation anywhere jumped to 100%, `pt_matmul` peak to 48–89% vs the old 21.7% ceiling. RTX 4070, RTX 4070 Ti, and the RTX 3050 Ti laptop still carry pre-fix diluted data. `Avg_Time_S` and EDP were unaffected either way. Reproduce: `python common/analysis/telemetry_dilution.py`. |
| 11 | **Beat a tuned static clock, not just max clock** | 🟢 Response-aware DDQN validated leave-one-GPU-out, 10 seeds across two independent batches | **4 Oct 2026, late night -- Phase 1 checkpoint result, decided: proceed to Phase 2.** Implemented the 2 response features (`common/features.py`), retrained the actual DDQN (not a hand-coded policy) with and without them, same settled hyperparameters, leave-one-GPU-out, **10 seeds run as two independent 5-seed batches** (to check the first batch wasn't a lucky draw -- it wasn't: response-aware mean identical both times, 0.0250/0.0250), on the 5-GPU clean fleet: response-aware DDQN scores **0.0250** mean steady-state scaled EDP (std 0.0127), beating tuned-fixed-clock's **0.0363** on **5/5 GPUs both batches**; the old 6-feature ablation scores **0.0375** (std 0.0367 -- variance comparable to its own mean), beating fixed-clock on only **3/5**. The load-bearing number is the PAIRED comparison (response vs. ablation, same seed, same held-out GPU): response wins **38/50** (one-sided binomial p≈0.00015). **Sharper reading than "modest improvement": without the response features, the DDQN's edge over a tuned fixed clock is close to a coin flip (one bad RTX 4090 seed was enough to flip the ablation's apparent 4/5-GPU win down to 3/5); with them, the result doesn't move.** This also explains why NORTHSTAR row 4's and this row's prior readings called plain RL a loss -- that margin over fixed-clock really is seed-dependent for the snapshot-only state. **The critical question this phase was for -- does the learned agent beat a simple hill-climb, not just the fixed clock -- is a clear yes** (0.025 vs 0.160), against the hill-climb this session could reconstruct (see asterisk). Reproduce: `python common/analysis/response_eval.py --seeds 0,1,2,3,4` and `--seeds 5,6,7,8,9`. Raw: `docs/results/response_eval_5seed.csv`, `response_eval_seeds5-9.csv`, combined in `response_eval_10seed.csv`. Full writeup: `context.md`'s "IMMEDIATE NEXT STEP". **Asterisk: the earlier-that-day hand-coded hill-climb result (0.0051, "4.7x better than fixed-0.6, 5/5 GPUs") could NOT be reproduced.** That script was a throwaway, never committed, and no longer exists. A careful re-implementation of the described mechanism (swept its one free parameter + trip direction, `common/analysis/hillclimb_sweep.py`) tops out at mean 0.160 -- it LOSES to fixed-clock, badly. Either the lost script did something materially different from what's described, or the original finding doesn't survive careful re-implementation -- **do not cite 0.0051 / 4.7x as a validated number**. Also found and fixed a real pre-existing bug while debugging this: `eval_policies.evaluate_policy()`'s "fixed" per-workload eval start-state used Python's salted `hash()`, so it was silently NOT fixed across separate process runs (fixed with `zlib.crc32`) -- affects the exact reproducibility of every number `baselines.py`/`sweep_rl.py` ever produced, though static/converged policies are fairly insensitive to it. Still only 5 of 8 target GPUs (4070/4070 Ti/laptop deferred). |

**Legend:** 🟢 Done and verified · 🟡 Partial / needs rework · 🔴 Blocked or failing

**Rows 4 (held-out GPU) and 6 (no unacceptable slowdown) carry the scientific claim,
but row 10 (telemetry window) now gates row 4 entirely, and row 11 (beat a tuned static
clock) sets the bar any adaptive method must clear to have contributed anything.** Everything else is in service of those two. If a decision
doesn't help one of them, it can wait until after submission.

---

## Why one model, not per-card weights?

This was the original plan (detect the GPU → load a matching pre-trained profile for it).
It doesn't reach the stated goal:

- The old per-workload-one-hot state design *needed* per-card weights, because
  normalization ranges and workload lists differed per card's profiling run. The
  6-feature redesign removes that need — every value is a fraction of *that card's own*
  limits (0.5 power_frac means "half of whatever this card's power limit is," whether
  that's 35 W or 450 W), so the same trained network is already speaking a language every
  card understands.
- Per-card weights don't generalize by construction — you're really doing a lookup table
  with extra steps, and it says nothing about whether the *idea* the network learned
  transfers. A unified model that works on a held-out card is the actual scientific
  result; per-card weights sidestep the question the paper needs to answer.
- Maintenance-wise, a per-card library means testing and updating a new file every time
  a new GPU ships. One file doesn't have that problem.

---

## Resolving the architecture disagreement

Two teammates, two trees, two state designs. The tie isn't a matter of opinion — it's
decided by the product requirement itself: an app that must run on an end-user's GPU with
an unknown application running has no workload one-hot to fill, and a network whose input
width depends on the number of profiled workloads cannot be "one model." The 6-feature
design reads behavior instead of identity, so it's the only one of the two that can reach
the stated goal.

**Structural fix — stop redefining the state twice:**

| File | Owns | Rule |
|---|---|---|
| `common/features.py` | The 6 features, action set, stack size, raw CSV schema | Single source of truth. If a tree disagrees with it, the tree is wrong. |
| `common/gpu_probe.py` | Capability detection + hardware limits | Run first on every machine, before profiling anything. |
| `gpus/<card>/` | Datasets, weights, card-specific runner scripts | May not redefine features or normalization. |

**Note on the RTX 4500's role.** It's a professional workstation card, not a 3000/4000
consumer part — strictly outside the product scope. Don't spend time re-profiling it to
match; its value in the paper is being the one card that supports memory-clock control,
which turns "consumer GPUs expose no usable memory-frequency range" into a contrast
rather than an assertion. Report it as a supplementary professional-GPU case, not part of
the unified-model claim.

---

## The one thing everything else depends on — ANSWERED, 23 Aug 2026

The open question was whether a rented instance is even allowed to change GPU clocks at
all. It's been tested for real now, on two vast.ai instances, same GPU model
(RTX 3060), and the answer is **it depends entirely on container vs VM, not on the GPU
or the marketplace**:

| Instance type | Core lock | Mem lock | Power limit | Verdict |
|---|---|---|---|---|
| Docker container (PyTorch template) | ✕ | ✕ | ✕ | UNUSABLE — "insufficient permission" on all three |
| Full VM (Ubuntu 22.04 CLI template) | ✓ | ✓ | ✓ | USABLE — all three worked |

Full data in `common/fleet_survey/fleet_survey.csv`.

**Practical conclusion: always rent VM-type instances, not container-type ones, for
profiling.** vast.ai (and presumably similar marketplaces) offer both; the template
description usually says outright whether it's a container ("Docker container," lists
"Container Limitations") or a VM ("full virtual machine," "no container limitations").
Check that before renting for a real session — a container-type instance will very
likely reproduce the same UNUSABLE result regardless of which GPU it has.

**Bonus finding — but a real bug in HOW it was first measured, caught and fixed same day.**
Both working VM instances appeared to lock *memory* clock and set the *power limit* too,
not just core. Digging into that directly with the user turned up an important gotcha:
`nvidia-smi -lmc` and `-pl` both **report success and print the requested value even when
persistence mode is off and the change silently never applies** — confirmed by hand on the
RTX 4070 (locked to 5001 MHz, command said "done," `clocks.current.memory` still read
10501 MHz; only worked after `nvidia-smi -pm 1`). `-lgc` (core) does **not** have this
problem — it applies immediately regardless of persistence mode, confirmed independently.

`gpu_probe.py` originally only checked whether the command was *rejected*, never whether
the value actually *changed* — so it couldn't have caught this. Fixed same day: it now
enables persistence mode before testing and reads back the achieved clock/limit instead of
trusting the command's own text, restoring the original persistence-mode state afterward.
Re-ran the fixed probe live on the same RTX 4070: core, mem, and power all **independently
confirmed** this time (`verified_readback=True` in the survey CSV).

**Resolved, 25 Aug 2026, same day:** the original RTX 3060 instance was already destroyed
by the time the bug was found, so it couldn't be re-checked directly — but rented a fresh
RTX 3060 (same model) and re-ran the fixed, readback-verifying probe on it. Result:
core/mem/power all **genuinely confirmed** (e.g. memory locked to 7501 MHz, requested and
achieved matched exactly). The original result turned out to be real all along, not a false
positive from the bug — but that's only knowable in hindsight, because the fixed test could
have just as easily shown a mismatch. **Two desktop cards across two GPU generations are
now both rigorously confirmed**: RTX 3060 (Ampere) and RTX 4070 (Ada), both `verified_readback=True`
in the survey CSV.

The two physical mobile/laptop cards (GTX 1650 Max-Q, RTX 3050 Ti Laptop) rejected
memory-clock locking outright with a driver-level "not supported" error — that kind of
hard rejection doesn't depend on persistence mode, so those results stood correct the whole
time.

**Current read (n=4, all methodologically sound now): 2 mobile GeForce cards hard-reject
memory-clock control at the driver level; 2 desktop GeForce cards across 2 generations
(Ampere, Ada) both genuinely support core, memory, AND power-limit control on an
unrestricted VM.** This is real evidence now, not a hopeful pattern resting on an unverified
data point.

**Still not a decided design change** — `common/features.py` still deliberately excludes
Mem_F. Even with solid evidence now, a fleet where some cards support memory control and
others don't breaks having one unified action space (the mobile cards genuinely can't do
it, and laptop cards are still part of the target product). That's a real design
conversation to have with the user now that the evidence backing it is trustworthy — a
decision made deliberately, not a change slipped in because the data happened to look good.

Next: rent 4–6 **VM-type** instances spanning both generations, profile each with the
unified collector, get a genuine leave-one-GPU-out result. Bonus: different hosts have
different cooling, so thermal variation across rentals becomes real domain
randomization — stronger than the synthetic Gaussian noise currently injected during
training.

Laptop GPUs aren't rentable at all — nobody puts a mobile chip in a datacenter. If laptop
cards are to appear in the results table, they come from physical machines either way.

---

## Risks that would cost the paper

| Risk | Signal it's happening | Response |
|---|---|---|
| Rented GPUs can't be controlled | Probe returns `UNUSABLE` on every host | Switch to borrowed physical cards same day; rescope to cross-architecture transfer |
| Wins only appear on easy workloads | New A/B tests on high-timing-response workloads (gaussian, particlefilter) show losses | Report it as the method's boundary — a characterized limit is publishable, a hidden one is not |
| Slowdown makes energy wins hollow | Runtime regression stays above the bound after the arbiter is restored | Make the action space more conservative; add hysteresis against action-flipping |
| Trees diverge again | A feature or normalization constant defined outside `common/` | Treat as a build error, not a style preference |
| Results are single-run | Any table without repeats or spread | Three repeats minimum on anything that reaches the paper |

---

## Weekly ritual

Walk the eleven scorecard rows. Two questions:

- **Did any row change color?** If not, the week went into work off the critical path.
- **Is any red row older than a week?** A stale red row means it's being avoided, not
  solved.

*Last updated: 3 Oct 2026, repo @ `8e90f12`, branch `unified-model`.*
