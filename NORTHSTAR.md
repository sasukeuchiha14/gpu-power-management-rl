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

Re-check these nine rows regularly. This **is** the results section of the paper.

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | One architecture across every GPU | 🟡 Half | 3050 Ti runs the 6-feature device-normalized state. RTX 4500 still runs 28 features with a workload one-hot. Must converge on `common/features.py`. |
| 2 | Runtime measured correctly everywhere | 🟡 Half | 3050 Ti now parses GPU-only duration — fixed the physics, produced a real compute-vs-bandwidth gradient. The 4500 collector still times the whole subprocess. |
| 3 | Four or more in-scope GPUs profiled | 🟢 8 of 4+ done | 3050 Ti (laptop) has real data. **7 desktop GPUs now do too**: RTX 3060, 3070, 3080, 3090 (Ampere) and RTX 4070, 4070 Ti, 4090 (Ada) — each a full 288-point (or near-full, where a card's max boost-clock bin isn't jointly lockable with any tested memory state — 4090: 252/288, 4070: 256/288, both cleanly explained) 2D core+memory sweep, 12 workloads, individually audited across 7 dimensions (header integrity, exact row count, zero duplicates, EDP recompute, physical-range sanity, per-workload trend sanity, quick-vs-full cross-check). Target was 3070/3080/3090 + 4060/4090; landed 4070/4070 Ti instead of 4060 (both valid Ada-tier substitutes), so target exceeded. Row unblocked. |
| 4 | Held-out GPU generalization (leave-one-GPU-out) | 🟡 First real result: mixed, not a clean win | Trained 8 separate policies (800 episodes each, seed 42), one per GPU held out, then evaluated each against ONLY the held-out card's own data (`common/evaluate_holdout.py`, results in `common/holdout_results/`). **AI has lower mean EDP than the pinned-max-clock governor on 6/8 held-out GPUs** — RTX 3060 (−50.7%), 3080 (−27.3%), 4070 (−8.5%), 4070 Ti (−74.8%), 4090 (−4.5%), and the RTX 3050 Ti laptop (−73.5%, 10/10 workloads won). **Loses on 2/8**: RTX 3070 (+22.8% worse) and RTX 3090 (+40.4% worse). Per-workload win rate ranges from 3/12 (3090) to 10/10 (laptop). This is a single seed, single run, 800 episodes on a small (~288-row) per-GPU lookup table — not enough to call generalization solved or failed; needs multiple seeds and more episodes before drawing a real conclusion (ties into row 9). Caught and fixed a real bug in the evaluation script itself before trusting these numbers: averaging per-workload *percentage* improvements is misleading when a workload's governor EDP is already near 0 (a tiny absolute miss reads as a huge %, one case hit −2957%) — fixed to compare the mean of raw EDP values instead. |
| 5 | Held-out workload generalization | 🟡 Redo | Done on the 4500, but under the old one-hot architecture (unseen app gets a borrowed identity via nearest-neighbor match). Must be redone under the 6-feature design. 3050 Ti zero-shot workloads are written but untested. |
| 6 | No unacceptable slowdown | 🟡 Guard implemented, not hardware-validated | `gpus/rtx3050ti_laptop/deployment/arbiter.py` (`DoNoHarmArbiter`) now watches live SM utilisation and forces max clock for `ARB_COOLDOWN` ticks after `ARB_PATIENCE` consecutive ticks of (high utilisation + below-max clock) — directly targets the 14–15%-slower-for-12%-less-power failure mode: high utilisation while downclocked is the live, workload-agnostic signature of causing real slowdown. Verified with 7 unit tests on the pure decision logic (trip timing, cooldown length, non-consecutive-bad-ticks reset, disabled-mode passthrough) plus a mocked-NVML integration test driving the real `LiveController`/trained weights end-to-end — all pass. **Not yet exercised on real hardware** — needs a live rerun of the workloads that showed the original regression to confirm it prevents it in practice, not just in simulation. |
| 7 | Safety net actually present and working | 🟡 Restored, not hardware-validated | The deleted 3050 Ti arbiter is restored (see row 6), but re-designed for the 6-feature architecture rather than ported as-is: it watches utilisation instead of reconstructing EDP via 1-NN dataset lookup, because that lookup needs workload-identifying features this design deliberately excludes (see `arbiter.py`'s docstring for the full reasoning — reusing the old approach would silently reintroduce the closed-world assumption the redesign removed). Unit- and integration-tested (see row 6); real-hardware validation still outstanding. The 4500's own arbiter was already working and untouched. |
| 8 | Latency and overhead within budget | 🔴 Failing, unchanged (needs real hardware) | Measured 70–90 ms mean vs a 50 ms budget still stands — could not be re-measured this session, no rented GPU available. Audited the code path instead (25 Aug 2026): `set_clocks()` and `_read_telemetry()` in `sim_to_real_deploy.py` already call `pynvml` in-process (no subprocess) on the normal path, and `agent.act()` already uses `torch.no_grad()` — no obvious Python-side inefficiency found. A mocked-NVML timing check (Python/torch overhead only, NVML calls excluded) measured well under 1 ms per tick, including the newly-added arbiter's own logic. If 70–90 ms still holds on real hardware, the cost is most likely in the NVML/driver round-trip itself (e.g. `nvmlDeviceSetGpuLockedClocks` blocking on driver-side settling), not Python overhead — needs a real GPU session to confirm and, if so, is a different fix than "remove subprocess calls." |
| 9 | Reproducible training evidence | 🟡 Partial | Reward-history curves are now saved by both `train_ddqn.py` and `train_unified.py`, and `--seed` exists on the unified trainer. Still missing: multi-seed runs and error bars — nobody's run more than one seed yet. |

**Legend:** 🟢 Done and verified · 🟡 Partial / needs rework · 🔴 Blocked or failing

**Only two rows carry the scientific claim: row 4 (held-out GPU) and row 6 (no
unacceptable slowdown).** Everything else is in service of those two. If a decision
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

Walk the nine scorecard rows. Two questions:

- **Did any row change color?** If not, the week went into work off the critical path.
- **Is any red row older than a week?** A stale red row means it's being avoided, not
  solved.

*Last updated: 23 Aug 2026, repo @ `99f943d`, branch `unified-model`.*
