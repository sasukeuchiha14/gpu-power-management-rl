# context.md — read this first

If you're a new Claude Code session picking this project up, read this file before
anything else. It's the only file in this repo that's meant to be rewritten every
session — everything else changes slowly.

**Read next, in order:** [NORTHSTAR.md](NORTHSTAR.md) (what we're building, and the
scorecard) → [CLAUDE.md](CLAUDE.md) (deep technical reference, only as needed).

---

## Where we are right now

**Branch:** `unified-model` (all work happens here; `main` is teammates' committed work,
left untouched unless explicitly asked to merge).

**The rented-GPU question is answered.** Container-type vast.ai instances deny clock
control entirely; VM-type instances (e.g. "Ubuntu 22.04 CLI (VM)" template) allow it.
**Always rent VM-type for profiling, never container-type** — the template listing says
which it is. Tested: RTX 3060 container (UNUSABLE), RTX 3060 VM (USABLE, but see caveat
below), RTX 4070 VM (USABLE, rigorously confirmed). Full data in
`common/fleet_survey/fleet_survey.csv`.

**Found and fixed a real bug in `gpu_probe.py` on 25 Aug, mid-session, prompted by the user
pushing back on an overclaim.** `nvidia-smi -lmc`/`-pl` report success and print the
requested value even when persistence mode is off and the change silently never applies —
`-lgc` (core) doesn't have this problem, applies immediately regardless. The probe
originally only checked whether the command was rejected, never whether the value actually
changed, so it couldn't catch this. Fixed: it now enables persistence mode before testing
and reads back the achieved value instead of trusting the command text, restoring original
persistence-mode state after. Verified the fix by re-running it live on the RTX 4070 — all
three genuinely confirmed this time. **The earlier RTX 3060 VM "success" on mem/power was
measured with the old buggy probe and that instance is now destroyed, so it can never be
re-verified** — marked `verified_readback=False` in the survey rather than assumed either
way. Right now there is exactly **one** rigorously-confirmed desktop card for memory-clock
control (the RTX 4070), not two. Get a second properly-verified one before calling this
settled. **Still explicitly not changing the core-only action-space design** over this —
that's a real call for the user, not something to slip in unilaterally.

**Currently running:** RTX 4070 VM instance, `ssh -p 25282 root@191.223.212.127` — still
rented, nothing profiled on it yet beyond the probe. Ask the user what's next (rent a
second desktop card to properly re-verify memory-clock control, or move to actually
profiling this one).

**Last session activity:** 23–25 Aug 2026. Set up this doc structure, the shared
`common/features.py` + `common/gpu_probe.py` modules, connected to and probed three real
rented GPU instances, found and fixed the persistence-mode measurement bug above.
**before** doing anything further on that box (explicit agreement — first touch on any
rented machine is probe-then-check-in, never probe-then-immediately-profile).

**The 4 items below (found in the 23 Aug full-repo review) are now FIXED — safe to profile
a new card for real.** Kept here so a fresh session doesn't waste time re-finding them:
1. ✅ `generate_dataset.py` now derives its core-clock sweep from the connected card's real
   NVML-reported range (reuses `gpu_probe.py`'s enumeration), falls back to the original
   literal list only if that fails, and skips (doesn't `sys.exit`) a single bad lock.
   Output filename is GPU-name-derived so different cards can't collide. Verified live on
   the GTX 1650 (correctly got its real 300-1740 MHz range) and confirmed the exact
   original filename/behavior is preserved when it detects the actual 3050 Ti.
2. ✅ `dvfs_env.py`'s action stride is now inferred from the loaded data's own frequency
   spacing instead of a hardcoded `100/1400`. Bonus finding: the old hardcoded value was
   ~50% too large even on the original 3050 Ti, because the real scaler normalizes
   against `max_core_mhz=2100` (this card's true boost clock) not 1400 — so the shipped
   laptop policy was trained with a bigger action step than intended, independent of any
   portability concern.
3. ✅ **Multi-GPU training now exists.** `common/unified_env.py` (discovers every
   `gpus/*/data/preprocessed_dataset.csv` matching the 6-feature schema, skips
   incompatible ones like the RTX 4500's, samples across all of them per episode) +
   `common/train_unified.py --holdout <gpu>` (leave-one-GPU-out). Verified end-to-end:
   discovery, the holdout-excludes-only-dataset edge case (raises cleanly), and a real
   15-episode training run producing correctly-shaped weights + a saved reward curve.
   **Still can't produce an actual holdout RESULT** — there's only one compatible dataset
   in the repo, so there's nothing to hold anything out from yet. That's still gated on
   getting more GPUs profiled (scorecard row 3).
4. ✅ `requirements.txt` added at repo root (torch, torchvision, gymnasium, pandas, numpy,
   matplotlib, transformers, nvidia-ml-py).

Also fixed in passing: `train_ddqn.py` (single-GPU trainer) now saves its reward-history
curve too, closing part of NORTHSTAR.md scorecard row 9.

**Renting alternatives were checked, not assumed:** TensorDock and RunPod are real
SSH/VM alternatives to vast.ai at similar or slightly higher prices; Salad was ruled out
despite its low price — it's a container-orchestration platform with no real SSH access,
wrong model for our interactive workflow. Staying with vast.ai.

## Working agreements from the user (don't relitigate these)

- **Never push to `main`.** All commits go to `unified-model` unless told otherwise.
- **On a rented GPU box:** run `gpu_probe.py`, report back, wait for go-ahead before a
  full profiling run. Don't chain straight into hours of paid GPU time on a box that
  hasn't been checked in on yet.
- **The end product is one unified model** (Option B from the discussion), not per-card
  weight selection. See NORTHSTAR.md's "why one model" section if this gets questioned —
  it's been decided, don't re-open it without new information.
- **The RTX 4500 is explicitly out of scope for the product** (professional card, not
  3000/4000 consumer series). It stays in the repo as a supplementary case, not something
  to migrate to the unified design under time pressure.
- Push access to GitHub already works from this sandbox via `gh` — no token needed from
  the user.

## Doc structure (so it doesn't drift back into one file)

| File | Job | Update cadence |
|---|---|---|
| `context.md` | This file. Session handoff: where we are, what's next. | Every session |
| `NORTHSTAR.md` | End-product definition + 9-row scorecard. Source for the published Artifact. | After any change that moves a scorecard row |
| `CLAUDE.md` | Deep technical reference: architecture, RL formulation, data notes, file map. | Only when the underlying facts change |
| `common/features.py` | The 6 canonical state features, action set, CSV schema. Single source of truth for both GPU trees. | Rarely — this is meant to be stable |
| `common/gpu_probe.py` | Capability probe, run first on any new machine. | Rarely |

**Do not let a status block creep back into CLAUDE.md.** If you're tempted to write
"current status" there, it belongs in this file instead.

## Open questions nobody has answered yet

- Does clock control actually work inside a rented vast.ai container? (This is *the*
  question the probe answers — see NORTHSTAR.md's "one thing everything depends on.")
- How many GPUs, and which models, will actually get rented/borrowed? Determines whether
  the paper claims cross-*generation* transfer (3000→4000) or has to fall back to
  cross-*architecture* transfer (Turing→Ampere→Ada) on whatever's physically available.
- Who fixes the RTX 3050 Ti's deleted do-no-harm arbiter, and when — it's a regression
  from before this session started, flagged but not yet assigned.

## Session log

- **23 Aug 2026** — Reviewed 6 new commits from teammates (A1 timing fix confirmed real
  and produced a physically coherent EDP gradient; 3 dead workloads dropped; state
  redesign done on 3050 Ti only). Found the deleted-arbiter regression. Built
  `common/features.py` + `common/gpu_probe.py`, found and fixed two bugs in the probe
  itself (nvidia-smi exits 0 on unsupported operations; NVML power-limit call fails on
  some cards where the CLI still works). Set up this three-file doc structure. Created
  `unified-model` branch. Verified remote-GPU-access plan is technically sound before the
  user spends money renting anything.
- **23 Aug 2026 (same day, follow-up)** — Full re-read of every file before the first
  rental, at the user's request. Found the 4 items now listed as FIXED above. Also found
  + fixed 2 more bugs in `gpu_probe.py` itself: it would crash uncaught if a container
  has no GPU visible at all (real failure mode on misconfigured rentals), and the
  USABLE/MARGINAL verdict could be wrongly downgraded by a clock-enumeration call failing
  independently of whether locking actually worked. Both fixed, re-verified live on the
  GTX 1650 (still correctly reports UNUSABLE without root), committed and pushed
  (`0e3adff`). Confirmed the RTX 4500 tree's issues (uncaught `sudo` subprocess crashes,
  hardcoded personal-machine path with a working fallback) don't affect the rental plan
  since that tree is out of scope for it.
- **23 Aug 2026 (same day, second follow-up)** — Gave rental instructions (vast.ai
  step-by-step) and checked alternatives via live web search rather than memory (see
  above). Then fixed all 4 "before renting for real" items: generalized the collector's
  clock sweep and made the fatal exit non-fatal, fixed the env's hardcoded action stride
  (and found it was wrong even on the original hardware — see above), built
  `common/unified_env.py` + `common/train_unified.py` for multi-GPU / leave-one-GPU-out
  training, added `requirements.txt`, and added reward-history saving to the existing
  single-GPU trainer. Everything was actually run and verified (used a throwaway venv on
  this sandbox with torch/gymnasium/nvidia-ml-py installed), not just read for
  plausibility — including a real 15-episode training run through the full cross-tree
  import path. Committed as `727deef` and `ac2da98`, pushed. Updated NORTHSTAR.md rows 4
  and 9 and republished the artifact to match.
