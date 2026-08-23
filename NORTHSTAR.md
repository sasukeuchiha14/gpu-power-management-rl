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
| 3 | Four or more in-scope GPUs profiled | 🔴 Blocked | Product targets 3000/4000 consumer cards. Exactly **one** is profiled (3050 Ti). The 4500 is a professional card, out of scope for the product. Depends on renting cards. |
| 4 | Held-out GPU generalization (leave-one-GPU-out) | 🔴 Not started | The core scientific claim. Cannot start until row 3 is done. Nothing has been run. |
| 5 | Held-out workload generalization | 🟡 Redo | Done on the 4500, but under the old one-hot architecture (unseen app gets a borrowed identity via nearest-neighbor match). Must be redone under the 6-feature design. 3050 Ti zero-shot workloads are written but untested. |
| 6 | No unacceptable slowdown | 🔴 Failing | 2 of 6 live workloads lost energy outright because runtime grew 14–15% while power fell only 12%. Users notice stutter, not watts. Needs an explicit bound and a guard that enforces it. |
| 7 | Safety net actually present and working | 🔴 Regressed | The 3050 Ti rewrite **deleted** the do-no-harm arbiter — only the config constants remain, unread by any code. On the 4500 the arbiter exists but didn't trip on either regression. README still advertises it. |
| 8 | Latency and overhead within budget | 🔴 Failing | Measured 70–90 ms mean vs a 50 ms budget; ~80% of a CPU core and ~750 MB RSS vs <1% / <50 MB targets. Root cause: subprocess spawns + CPU-only inference. 3050 Ti already has the in-process NVML fix. |
| 9 | Reproducible training evidence | 🟡 Gap | Reward history is still discarded at the end of training. No saved curves, no seed control, no multi-seed runs. |

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

## The one thing everything else depends on

The plan to source 3000/4000 cards from vast.ai (or similar) carries an assumption that
hasn't been tested end to end on a real rental yet: that a rented container is allowed to
change GPU clocks at all. Rented instances are Docker containers, and clock control needs
privileges the NVIDIA container runtime doesn't grant by default.

Settle it for the price of a coffee before renting anything for a full profiling run: rent
the cheapest 3000-series instance, run the probe.

```bash
pip install nvidia-ml-py
python3 common/gpu_probe.py --survey fleet_survey.csv
```

It prints `USABLE` or `UNUSABLE` and writes a survey row. Do this on **two or three
different hosts**, not one — permissions vary per host, so a single failure isn't proof
the whole marketplace is closed.

- **If USABLE:** rent 4–6 cards spanning both generations, profile each with the unified
  collector, get a genuine leave-one-GPU-out result. Bonus: different hosts have
  different cooling, so thermal variation across rentals becomes real domain
  randomization — stronger than the synthetic Gaussian noise currently injected during
  training.
- **If UNUSABLE:** fall back to physical cards immediately. Borrow 3000/4000-series
  laptops from classmates — a full profiling run is 2–3 hours per card. Scope the paper
  to cross-*architecture* transfer (Turing → Ampere → Ada) on the cards physically held.

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
