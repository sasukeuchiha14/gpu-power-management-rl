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
which it is. Full data in `common/fleet_survey/fleet_survey.csv`.

**Found and fixed a real bug in `gpu_probe.py` on 25 Aug, mid-session, prompted by the user
pushing back on an overclaim, then resolved the same day.** `nvidia-smi -lmc`/`-pl` report
success and print the requested value even when persistence mode is off and the change
silently never applies — `-lgc` (core) doesn't have this problem, applies immediately
regardless. The probe originally only checked whether the command was rejected, never
whether the value actually changed, so it couldn't catch this. Fixed: it now enables
persistence mode before testing and reads back the achieved value instead of trusting the
command text, restoring original persistence-mode state after.

The original RTX 3060 VM result (mem/power success) was measured with the old buggy probe
and that instance was already destroyed, so it looked like it could never be re-verified —
but rented a fresh RTX 3060 (same model) and re-ran the fixed probe on it. **Result: it was
genuine all along.** Core/mem/power all matched exactly (e.g. memory locked to 7501 MHz,
requested and achieved identical). Combined with the RTX 4070 (also rigorously confirmed):
**two desktop cards across two GPU generations (Ampere, Ada) now both properly confirm
memory-clock and power-limit control.** The evidence is solid now, not just promising.

**Still explicitly not changing the core-only action-space design** over this — the mobile
cards (part of the target product) genuinely can't do memory control regardless, so adding
it for desktop-only would break having one unified action space. That's a real design
conversation to have with the user, not something to slip in unilaterally just because the
data now supports it.

**The full 288-point RTX 3060 dataset is DONE and fully audited (25 Aug).** Both the quick
test (18 points) and the full run (288 points, 12 workloads x 8 core x 3 memory, zero
skipped) completed successfully on the On-Demand instance. This is the first complete
desktop-GPU dataset the project has ever produced, and the first real proof
`collect_dataset.py` works end-to-end on live hardware, not just in local logic tests.

**Found one more real bug during the full-run audit, root-caused and fixed:** the output
file was missing its header row. Cause: an early *buffered* launch (killed and replaced
with an unbuffered one, before the quick test) had already created a 0-byte stub file via
`open(mode="a")`. The real run's `file_exists = os.path.isfile(...)` check saw that stub
"existed" and skipped writing the header, so all 288 rows got appended under no header at
all. **No data was lost** -- confirmed by matching the "malformed header" row's exact
values against the live log from that data point. Fixed by prepending the correct header,
and fixed the root cause in `collect_dataset.py` (file_exists now also requires non-empty).

Ran a full 6-part audit across all 288 rows (not a sample): exact coverage (288 rows, 0
duplicates, all 12 workloads x 24 rows each), EDP formula verified to 0.06% across every
row, every field in sane physical range everywhere, per-workload trend sanity (one minor,
explained non-monotonicity in pt_lstm from subprocess-overhead noise on its very short
runtime, not a defect), cross-checked against the quicktest for reproducibility (timing
matched to <0.5% in 11/12 comparisons), and both brand-new workloads behaved sensibly on
their first-ever real-hardware run. Dataset and the header-bug fix are committed and pushed.

**Next: rent 3+ more desktop GPUs** (target list agreed with the user: 3070/3080/3090 for
Ampere high-tier, 4060 for Ada low-tier, 4090 for Ada high-tier) and run the same
`collect_dataset.py` full sweep on each, to get enough GPUs for a real leave-one-GPU-out
test. Consider running some of these in parallel once the single-GPU pipeline is this
proven -- that was always the plan after one clean full run succeeded.

**Currently running:** the RTX 3060 instance (`ssh -p 61448 root@142.170.96.180`) is still
up with a complete, audited dataset on it (already pulled locally and committed) -- fine to
destroy whenever, nothing more needed from it right now.

**Built `common/collect_dataset.py` (25 Aug), the 2D core+memory data collector, plus 2 new
workloads** (`common/benchmarks/pt_sparse_gather.py`, `pt_transformer_decode.py`, chosen
from a literature check, not guessed). User explicitly asked for a re-check before handing
over SSH access -- found and fixed 6 real bugs on that pass, most importantly: the script
never verified an achieved clock actually matched what it locked, same class of mistake as
the persistence-mode bug. See git log on `common/collect_dataset.py` for full detail.

**Then found and fixed a SECOND, subtler bug the same day, live on a rented RTX 3060:** the
readback-verification fix above had a race condition -- reading a clock immediately after
locking it can catch a stale pre-transition value (measured under 0.5s to settle on one run,
over 1s on the very next run of the identical script on the same host). A single fixed delay
can't reliably fix variable settling time. Replaced with `_wait_for_clock()` in
`gpu_probe.py` -- polls every 0.5s for up to 4s instead of checking once -- used by all three
gpu_probe.py tests AND imported into collect_dataset.py so both share one fix. Verified with
4 repeated runs on the same host: 4/4 consistent after the fix (previously flaky).

**Then hit two more real, unresolved things in sequence, on that same RTX 3060 rental:**
1. The default `pip install torch` grabbed a CUDA-13-built wheel, but the host's driver only
   supports up to CUDA 12.9 -- `torch.cuda.is_available()` was False, so `--quick` correctly
   failed all 18 points (the FATAL-on-100%-skip check worked exactly as designed, no silent
   bad data). Fix identified but not yet completed: install from
   `https://download.pytorch.org/whl/cu121` instead of the default index.
2. **The instance got outbid and reclaimed mid-fix** (confirmed directly by the user, not
   inferred) -- "Connection refused," not a dropped session. This was an Interruptible
   instance. The actual `--quick` short test run has **still never succeeded end to end** --
   every attempt so far has been interrupted by something else (bugs found, or the rental
   dying) before completing cleanly.

**Next step:** rent again. Consider On-Demand this time given the reclaim just happened for
real, not hypothetically -- ask the user. Once connected: install torch from the cu121 index
FROM THE START (don't repeat the default-index mismatch), then finally get one clean
`--quick` run to complete.

**Last session activity:** 23–25 Aug 2026. Set up this doc structure, the shared
`common/features.py` + `common/gpu_probe.py` modules, connected to and probed five real
rented GPU instances across the session, found and fully resolved three separate real bugs
(persistence-mode silent no-op, missing readback verification, and the readback race
condition above) -- each found by the user pushing back or asking for a recheck, not by
proactive review alone.

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
