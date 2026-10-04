# context.md — read this first

If you're a new Claude Code session (or a new person) picking this project up, read this
file before anything else. It's the only file in this repo meant to be rewritten every
session — everything else changes slowly.

**Read next, in order:** [docs/handoff-2026-08-25/FINDINGS_REPORT.md](docs/handoff-2026-08-25/FINDINGS_REPORT.md) (what we found,
what to do next — the main document) → [NORTHSTAR.md](NORTHSTAR.md) (end goal + scorecard) →
[docs/handoff-2026-08-25/ONBOARDING.md](docs/handoff-2026-08-25/ONBOARDING.md) (setup, layout, working agreements) →
[CLAUDE.md](CLAUDE.md) (deep technical reference, as needed).

**Branch:** `unified-model`. Never push to `main`.

---

## IMMEDIATE NEXT STEP (as of 4 Oct 2026, night): Phase 1 checkpoint passed -- modestly, with real caveats

**Phase 1 (response-aware state augmentation) is done and the critical question has an
answer: yes, but modestly, not the dramatic 4.7x the 4 Oct hand-coded hill-climb found.**

### What was built
1. `common/features.py`: added `RESPONSE_FEATURES` (`d_log_runtime_d_log_clock`,
   `d_log_power_d_log_clock`) and `response_features()` -- elasticities measured between the
   row before an action and the row after it, clipped to `±RESPONSE_CLIP=3.0` (chosen from
   the real measured distribution, `common/analysis/response_distribution.py`) and mapped to
   [0,1]. NOT added to `FEATURES` itself (that's the static per-row CSV schema,
   `preprocess_desktop.py` zips it against `build_features()`'s 6 return values -- growing it
   would have silently broken that). The 2 new dims are computed on the fly, trajectory-side,
   by whichever caller opts in via `use_response=True` (default `False` everywhere, so every
   existing script -- `baselines.py`, `sweep_rl.py` -- is unaffected unless it asks).
2. `common/unified_env.py`, `common/eval_policies.py` (`augmented_obs`), `common/train_unified_v2.py`
   (`Cfg.use_response`): threaded the flag through the reference env, the evaluator, and the
   fast training loop so the SAME semantics apply everywhere; the ablation (old 6-feature
   agent) is just the identical pipeline with the flag off, not a separate code path.
3. `eval_policies.p_hillclimb()`: a RE-implementation of the 4 Oct hand-coded hill-climb --
   **the original script was a throwaway that was never committed and no longer exists.**
   Swept its one free parameter + trip direction (`common/analysis/hillclimb_sweep.py`); best
   found (threshold=0.05, step back up on trip) scores **mean ss 0.160 across 8 GPUs, which
   LOSES to fixed-0.6 (~0.03) badly.** This does NOT reproduce the 0.0051 / "beats fixed-0.6
   by 4.7x" finding. Either the lost script did something materially different, or that
   finding doesn't survive careful re-implementation. Either way: do not cite 0.0051 or 4.7x
   as a validated number going forward -- it rests on code that no longer exists.
4. `common/analysis/response_eval.py`: the actual Phase 1 checkpoint. Leave-one-GPU-out,
   5-seed, on the 5-GPU clean fleet (3060/3070/3080/3090/4090), settled hyperparameters
   (γ=0.5, plain DQN, Huber, lr=3e-4, width 128), comparing governor / tuned-fixed-LOO /
   old-ablation-DDQN / hill-climb-reference / response-aware-DDQN. Raw: `docs/results/response_eval_5seed.csv`.
5. **Found and fixed a real pre-existing bug while debugging the hill-climb**:
   `eval_policies.evaluate_policy()`'s "fixed per-workload eval start state" used Python's
   built-in `hash()` on the workload name, which is salted per-process (`PYTHONHASHSEED`)
   unless set -- so the claimed-fixed start state was NOT actually fixed across separate runs
   (confirmed: `hash('pt_fft')` differs between two bare `python3 -c` invocations). Replaced
   with `zlib.crc32`, which is deterministic. This affects every number ever produced by
   `baselines.py` / `sweep_rl.py` to some degree -- re-run those before citing old numbers
   that are sensitive to start-state (static policies and converged-enough learned ones are
   fairly insensitive; a greedy hand-coded policy, as the hill-climb debugging showed, is not).

### The result (steady-state scaled EDP, leave-one-GPU-out, 5 seeds, lower is better)

| policy | mean | std | GPU-mean wins vs tuned-fixed |
|---|---|---|---|
| **response-aware DDQN (ours)** | **0.0250** | 0.0133 | **5/5** |
| old snapshot DDQN (ablation) | 0.0286 | 0.0125 | 4/5 |
| fixed clock, tuned LOO | 0.0363 | 0.0072 | -- |
| hill-climb (reconstructed reference) | 0.1598 | 0.1375 | 0/5 |
| governor (pinned max clock) | 0.2668 | 0.1047 | 0/5 |

**The real evidence is the paired comparison** (response vs. ablation, same seed, same held-out
GPU, only the state differs): response-aware wins **19/25** matched pairs (one-sided binomial
p≈0.007), mean reduction ≈0.0036 (~13% relative). Not uniform -- it's a clear, consistent win
on 3060/3080/4090 (4-5 of 5 seeds each) and a closer call on 3070/3090 (2/5 and 3/5). This is
the number that isolates what the 2 response features bought, independent of fixed-clock noise.

**Answering the question this phase was for**: does the learned agent beat the simple
hill-climb, not just the fixed clock? **Yes, decisively** (0.025 vs 0.160) -- but only against
the hill-climb THIS session could reconstruct, which, per point 3 above, is NOT validated as
equivalent to the original 4 Oct finding. Read this as "RL beats a reasonable hand-coded
response-driven heuristic," not "RL beats the best possible one."

### Caveats before this goes in the paper
- Still only 5 of 8 target GPUs (4070/4070 Ti/laptop deferred, diluted telemetry, unchanged
  from before).
- The headline "4.7x" number from 4 Oct is not reproduced and should not be cited without a
  "could not be independently verified" caveat.
- 5 seeds, leave-one-GPU-out -- better than the single-seed era, thinner than ideal given the
  historical variance on this exact task (the pre-existing `C_nd_g50_lr` sweep had std LARGER
  than its mean: 0.038 ± 0.043). The paired response-vs-ablation comparison is the most
  trustworthy number here because pairing cancels seed-level noise that a mean-vs-baseline
  comparison does not.
- `old snapshot DDQN (ablation)` beating the tuned fixed clock here (4/5) is itself a
  reversal of NORTHSTAR row 11's prior reading (every RL config lost) -- most likely because
  that prior reading's variance was enormous (see above) and this run's particular 5 seeds
  landed favourably, not because anything about the ablation changed. Do not read the
  ablation's result as "RL now beats fixed clock even without response features" -- read it
  as "this task is noisy enough that 5 seeds isn't fully conclusive for the ablation, which is
  exactly why the PAIRED response-vs-ablation number, not either one's margin over fixed-clock,
  is the load-bearing result."

### Next: Phase 2 (hardware validation) -- gate: user sign-off on this checkpoint
Per the original plan, VMs come back now that Phase 1 has a checkpoint. Given the modest (not
dramatic) effect size, worth deciding with the user whether to proceed to hardware validation
now or spend more offline time first (e.g. more seeds, or trying to recover a stronger
hand-coded reference). Not decided yet -- ask before proceeding to Phase 2's VM rental.

---

## Prior next step (4 Oct 2026, evening, superseded by the Phase 1 result above): build response-aware state augmentation

**This supersedes the "rent 4070/4070 Ti" plan below for now.** User decision: proceed
with the 5-GPU clean fleet (3060/3070/3080/3090/4090); 4070/4070 Ti and the laptop are
deferred, not abandoned -- revisit once rentable. **No VM rental needed for the next
phase** -- it's pure offline work against the CSVs already committed.

### The finding that changed the plan

The "fixed 0.6x is unbeatable" result (NORTHSTAR row 11) was real but **metric-blind**:
scaled EDP is min-max normalised per workload and the EDP curve is flat near its optimum,
so a policy at 12% slowdown and one at 0.8% slowdown can score almost the same. Checked
in absolute terms (energy saved % / slowdown % vs max clock, 5 GPUs x 12 workloads,
leave-one-GPU-out): a **closed-loop policy that measures its own clock changes** (step
down, watch what actually happens, stop when the measured response says to) gets the
same energy savings as fixed-0.6 for a **5x smaller slowdown**, and in the project's own
scaled-EDP metric scores **0.0051 vs fixed-0.6's 0.0240** -- a 4.7x improvement, beating
it on **5 of 5 GPUs**, on data already committed. (This was a hand-coded greedy hill-climb,
not yet a learned policy -- see Phase 1 below.)

**Why the DQN was losing isn't RL -- it's the state.** Snapshot telemetry (power, temp,
util at one instant) cannot discriminate workloads across GPUs: verified binary
compute/bandwidth regime ID leave-one-GPU-out is 0.56 (barely above chance), and
`core_frac` alone explains 0.78 of the 0.81 "is clock below optimum" accuracy -- the other
5 channels add ~3 points (`common/analysis/diagnose_features.py`). The agent has nothing
to discriminate on. **The response to its own action is exactly the missing signal**, and
critically it's a *ratio* (e.g. Δlog(runtime)/Δlog(clock)), so it's naturally
device-invariant -- which matters a lot given the unit-to-unit hardware variability
already found this session (same-model GPUs with different real ceilings: RTX 3060
2145 vs 2190 MHz, RTX 3090 390 vs 350 W power limit -- absolute normalisation is fragile,
ratios aren't).

### Decision: keep RL, DDQN, and EDP. Do not replace them.

User is explicit about this and it's the right call for the paper framing. The change is
additive: **augment the state with 2 response features**, not a new objective or a new
algorithm. Novelty claim for the paper: *"snapshot telemetry doesn't transfer across
consumer GPUs because of unit-level hardware variability; response-aware state
augmentation fixes it, validated leave-one-GPU-out."* That's a full diagnose-explain-fix-
validate arc backed by data already in this repo.

Literature check (done this session, see git log / prior conversation if lost -- searched
and found real prior art, so set expectations accordingly): closed-loop DVFS under a
performance constraint already exists (not a novel mechanism by itself); cross-GPU
transfer for DVFS also exists (WattGPU, MetaDVFS, cross-domain power models, DRLCap for
RL+EDP specifically). **Top-tier venues (ISCA/MICRO/ASPLOS/HPCA/SC) are not realistic.**
Workshops, mid-tier journals (Sustainable Computing, J. Systems Architecture, FGCS),
student tracks, and arXiv are realistic and the right target.

### Phase 1 -- offline, no GPU rental needed (~2 days)

1. Add 2 features to `common/features.py` (6 -> 8): response ratios computed from the
   *last* action taken -- e.g. `d_log_runtime_d_log_clock` and `d_log_power_d_log_clock`
   between the pre- and post-action measurements. Keep the existing 6 features; this is
   additive, matching "don't stray from EDP/RL/DDQN."
2. `unified_env.py`: track the previous clock setpoint and its resulting row so `step()`
   can expose the measured response as the new state dims. The env is still a table
   lookup (same honest limitation as before -- say so in the paper).
3. Retrain via `train_unified_v2.py`, same settled hyperparameters (gamma=0.5, plain DQN
   not Double, Huber loss, lr 3e-4, width 128 -- do not re-litigate these, already swept).
4. **Leave-one-GPU-out evaluation, the key checkpoint**, comparing:
   - governor (max clock) and tuned fixed clocks -- existing baselines, unchanged
   - **snapshot-state DDQN** (the old 6-feature agent) -- the ablation that shows *why*
     the old approach lost
   - **greedy hill-climb** (the hand-coded response-driven policy, 0.0051 scaled EDP) --
     the ceiling-ish reference; if the learned agent can't approach this, something's
     wrong in the RL setup, not the idea
   - **response-aware DDQN** (new, 8-feature) -- the actual contribution
5. **Critical ablation for defending "why RL" to reviewers**: does the learned
   response-aware DDQN meaningfully beat the simple hand-coded greedy hill-climb? If RL
   just matches a dumb rule, the paper's RL framing is weaker -- quantify this honestly
   either way. (Hill-climb gets stuck at local minima the oracle doesn't -- 0.0051 vs
   0.0000 -- so there's headroom for RL to add value by looking past local bumps; whether
   it actually does is an empirical question Phase 1 answers.)

### Phase 2 -- hardware validation (VMs needed, only after Phase 1 confirms the offline result)

Deploy the trained response-aware policy live on a rented GPU. Use a workload with a
natural live work-rate signal -- graphics/frame-time workload as the user suggested, or
any iterative benchmark reporting steps/sec -- so the "measured response" is real, not
simulated. Three-way comparison: driver default (no management) / tuned fixed clock /
response-aware DDQN. Measure real energy via NVML power integration. Headline graphs:
power-vs-time traces overlaid, plus a bar chart of total energy and runtime across the
three conditions. **This is the first point VMs are needed again** -- do not rent before
Phase 1's checkpoint (step 5 above) is done and looks right.

### Phase 3 -- paper (~1 week)

Write-up targeting a workshop or mid-tier journal (see literature check above), arXiv
regardless. Structure: the telemetry-dilution finding, the unit-variability finding, the
snapshot-state-can't-discriminate finding (diagnose_features.py), then the fix
(response-aware state) and its validation (Phase 1 leave-one-GPU-out + Phase 2 hardware).

---

## Prior next step (4 Oct 2026 AM, superseded by response-aware augmentation above): rent 4070/4070 Ti when available; redo laptop

**5 of 7 desktop GPUs re-collected and committed** (RTX 3060, 3070, 3080, 3090, 4090;
commit `8e90f12`), each audited clean (exact row counts, zero duplicates, zero EDP
mismatches, zero physically-invalid rows). `pt_matmul` peak utilisation now reaches
48-89% on these cards vs the old fleet's 21.7% ceiling -- the fix is confirmed working
on real hardware, not just the synthetic trace in `telemetry_dilution.py`.

**The headline question has a first, partial answer: RL still does not beat a fixed
clock, even with clean telemetry.** Run on just the 5 clean GPUs (RTX 4070/4070 Ti were
unavailable to rent, laptop deferred -- see NORTHSTAR row 11 for the full numbers): best
RL config scores 0.038 mean scaled EDP vs fixed-clock's 0.025-0.031 on the identical 5
GPUs, and wins on only 1 of 5 individual cards. **This is evidence, not yet a verdict** --
still only 5 of 8 target GPUs, and the fixed-clock tuning itself was less stable on this
smaller set (picked 0.55-0.70 across folds, not the consistent 0.60 the 7-GPU fleet gave).
Re-run `sweep_rl.py --preset combos` once 4070/4070 Ti/laptop are added back.

**Still needed:** RTX 4070, RTX 4070 Ti (rentable, unavailable as of 4 Oct 2026 -- retry
later), and the RTX 3050 Ti laptop (physical machine, not rentable -- re-collect locally
with `python common/collect_dataset.py`). Until all 8 are done, `diagnose_features.py` and
`baselines.py --loo-tune` are reading a **mixed fleet** (5 fixed + 2-3 still diluted) and
their fleet-wide numbers are not yet trustworthy either way -- a first mixed-fleet read
gave binary regime ID 0.595 (vs the old, now-suspect 0.677), which is not a verdict, just
an artifact of mixing two different data-generating processes across the leave-one-GPU-out
folds. (The 5-GPU-only RL sweep above sidesteps this by excluding the stale GPUs entirely,
which is why it's trustworthy as far as it goes.)

**Two new driver-pin bugs found and fixed during this re-collection** (both the exact
class `preprocess_desktop.py`'s docstring already warned about, now actually hit): a
rental can report the *same* driver string as a previous one but have different real
hardware limits (RTX 3060: 2145 -> 2190 MHz core; RTX 3090: 390 -> 350 W power limit,
same driver both times). Disambiguated with synthetic driver-key suffixes in
`fleet_survey.csv`. Added a power-limit sanity check to `preprocess_one()` mirroring the
existing core-clock one, since the 3090 case would **not** have been caught by any
automated check before this -- only by manually diffing a fresh probe survey against the
committed row. Do this diff-check explicitly for every future re-collection, not just
when a driver string changes.

---

## Prior next step (25 Aug 2026, superseded above): re-collect the fleet with the fixed sampler

**Why this and nothing else:** the telemetry columns in all 8 committed datasets are
contaminated. `collect_dataset.py` sampled NVML across the whole benchmark subprocess —
interpreter startup, `import torch`, CUDA context creation, typically 2–4 s of idle GPU —
while the actual timed work is often 0.15–0.6 s. Recorded utilisation/power therefore
encode how long a workload ran relative to fixed startup cost, not how hard it drove the
GPU. Measured across the fleet: `corr(GPU-only duration, recorded utilisation) = +0.708`,
62% of points report under 9% utilisation, the highest utilisation *anywhere* is 64.4%,
and dense GEMM (`pt_matmul`) reports 21.7%.

Those are exactly the features a controller needs to characterise an unknown workload, so
**every conclusion about telemetry-driven adaptation is currently unanswerable**, not
negative. `Avg_Time_S` and EDP are unaffected and remain trustworthy.

**The collector is already fixed and committed** (all 12 benchmarks emit
`Window: <start> <end>`; `GPUProfiler.get_stats(window)` averages only in-window samples).
No code work is needed before renting.

### Per-instance checklist (proven across 7 rentals)

1. Rent on vast.ai: **VM-type template** ("Ubuntu 22.04 CLI (VM)" or equivalent), **never**
   container-type — containers deny clock control entirely. Prefer **On-Demand**;
   interruptible instances have been reclaimed mid-run for real.
2. SSH in, confirm it's a genuine VM (`hostname` = `ubuntu`, real kernel from `uname -a`).
3. Mirror this exact structure (path logic depends on it):
   ```
   mkdir -p /workspace/repo/common/benchmarks /workspace/repo/gpus/rtx3050ti_laptop/benchmarks
   ```
   then scp `common/gpu_probe.py`, `common/collect_dataset.py`, `common/benchmarks/*.py`,
   and `gpus/rtx3050ti_laptop/benchmarks/*.py` to the matching remote paths.
4. `pip install --quiet nvidia-ml-py`, then `python3 gpu_probe.py --survey fleet_survey.csv`.
   **Confirm `USABLE` before anything else.**
5. **Check the driver's CUDA version before installing torch** (`nvidia-smi | head -5`).
   CUDA 13.x → plain `pip install torch` is fine. CUDA 12.x → install from
   `https://download.pytorch.org/whl/cu121`, and note that a later
   `pip install torchvision transformers` can silently upgrade torch back to a cu130 build
   (hit twice) — pin torch+torchvision together and re-verify.
6. Verify: `python3 -c "import torch, torchvision, transformers; from torchvision.models
   import resnet50; from transformers import BertModel; print(torch.cuda.is_available())"`
   must print `True`.
7. `python3 collect_dataset.py --quick` as a sanity check.
8. **NEW AND CRITICAL — verify the telemetry fix actually took effect.** In the `--quick`
   output, `pt_matmul` at high core clock should now report utilisation near saturation
   (previously ~20%). If it still reports ~20%, the `Window:` markers are not reaching the
   collector — **stop and debug before burning hours on a full sweep.**
9. Full run: `nohup python3 -u collect_dataset.py > /workspace/full_run.log 2>&1 & disown`
   (~2–3.5 h; several cards finished in under 2 h). Check in every ~20–25 min.
10. On completion, audit before trusting: header present, exact row count
    (12 workloads × core levels × usable mem levels), zero duplicate (workload, core, mem),
    EDP recomputes, all fields in physical range, per-workload trends sane, quick-vs-full
    cross-check. Then pull the CSV to `gpus/<gpu_slug>/data/` and commit per GPU.

Skips at the top boost bin are expected on Ada cards (4070/4090 could not jointly lock
3105 MHz with any memory state) — the collector's achieved-clock guard correctly refuses to
record a mislabelled point. That is not a failure.

### After re-collection

```bash
python common/preprocess_desktop.py              # add new (slug, driver) pins first
python common/analysis/telemetry_dilution.py     # confirm the dilution is gone
python common/analysis/diagnose_features.py      # do features now carry workload signal?
python common/analysis/baselines.py --loo-tune   # does fixed-0.6 still dominate?
python common/analysis/sweep_rl.py --preset combos
```

The decisive question: does telemetry-only regime identification rise well above the current
0.677 binary accuracy? If yes, adaptation has room to beat the fixed clock. If it stays near
chance even with clean telemetry, the honest paper claim is that static per-GPU clock
selection is the right engineering answer — which is itself a useful result.

---

## Where we are right now

**8 GPUs profiled** (7 desktop across Ampere + Ada, plus the RTX 3050 Ti laptop), each fully
audited and committed as its own commit. Scorecard row 3 is done and exceeded.

**The headline result, robust to everything above:** a fixed clock at **0.6 × max**, using no
telemetry and no learning, scores **0.037** scaled EDP against the pinned-max-clock
governor's **0.319** and a per-workload oracle's **0.000** — capturing ~88% of the achievable
gain and beating every learned policy tested. Verified leave-one-GPU-out (tuned on the 7
training GPUs only; selects 0.60 on all 8 folds). This rests only on timing and EDP, both
measured correctly.

**Leave-one-GPU-out RL is unstable.** A single seed suggested 6/8 wins; a 4-seed sweep cut
that to **2/8 consistent** (RTX 4070, 4070 Ti), with per-GPU std up to ±186%. Three real RL
bugs were found and fixed (truncation-treated-as-termination, unscaled MSE at Q≈100, no
target smoothing) plus a reward-normalisation error (`dataset_view.py`); fixes + tuning
roughly halved both error and variance. γ 0.99→0.50 was the biggest lever; dueling and
absolute actions hurt; plain DQN beat Double-DQN.

**Safety work landed:** the deleted do-no-harm arbiter is restored (redesigned to watch SM
utilisation rather than reconstruct EDP via dataset lookup, which the 6-feature design
cannot support), with 7 unit tests + a mocked-NVML integration test. `DRY_RUN` now defaults
to `True`.

**Retracted:** an earlier session conclusion that "the 6-feature state is fine, the problem is
purely RL training." An ablation showed `core_frac` alone explains nearly all of the 81%
probe accuracy — the telemetry adds ~3 points. See FINDINGS_REPORT §4.

---

## Working agreements (don't relitigate)

- **Never push to `main`.** All work goes to `unified-model`.
- **Never fabricate results**, hyperparameters, hardware specs, baselines, or citations.
  Flag anything unverified.
- **Single-seed results are not evidence here.** Multi-seed with variance before reporting.
- **Sanity-check computed metrics against raw values by hand at least once** — a real bug was
  caught this way (an averaged-percentage metric hitting −2957% on a near-zero-baseline
  workload).
- **The end product is one unified model** (not per-card weight selection). Decided; see
  NORTHSTAR.md's "why one model".
- **The RTX 4500 is out of scope** for the product — professional card, kept as a
  supplementary contrast case only.
- **On a rented box:** probe first, confirm `USABLE`, report back before hours of paid time.
- Push access via `gh` already works from this sandbox; no token needed.

---

## Doc structure (so it doesn't drift back into one file)

| File | Job | Update cadence |
|---|---|---|
| `context.md` | This file. Session handoff: where we are, what's next. | Every session |
| `docs/handoff-2026-08-25/FINDINGS_REPORT.md` | What we found, what it means, what to do/experiment next. | When results change |
| `docs/handoff-2026-08-25/rl_instability_investigation.md` | Full investigation log, including disproved hypotheses. | Append-only |
| `docs/handoff-2026-08-25/ONBOARDING.md` | New-contributor setup, repo layout, working agreements. | Rarely |
| `docs/handoff-2026-08-25/CLAUDE_HANDOFF_PROMPT.md` | Copy-paste prompt to start a fresh Claude session. | Rarely |
| `NORTHSTAR.md` | End-product definition + 9-row scorecard. | After any scorecard move |
| `CLAUDE.md` | Deep technical reference: architecture, RL formulation, file map. | When facts change |
| `common/features.py` | The 6 state features, action set, CSV schema. Source of truth. | Rarely |

**Do not let a status block creep back into CLAUDE.md.** Status belongs here.

---

## Open questions nobody has answered yet

- **Does telemetry-driven adaptation beat a tuned static clock, once telemetry is measured
  correctly?** The whole RL premise rides on this. Blocked on re-collection.
- Is the ~70–90 ms control-loop latency (vs a 50 ms budget) driver round-trip cost or
  something fixable in our code? Code audit found no obvious Python-side cost; needs a real GPU.
- Should desktop cards use memory-clock co-control? Currently excluded so laptop and desktop
  share one action space, but 2/3 of every collected dataset is memory data the core-only
  controller discards.
- Who fixes the RTX 3050 Ti tree's remaining pre-existing issues, and when?

---

## Session log

- **23 Aug 2026** — Built `common/features.py` + `gpu_probe.py`; found/fixed two probe bugs;
  created `unified-model`; verified the remote-GPU plan before spending money.
- **23 Aug 2026 (follow-up)** — Full pre-rental re-read; fixed 4 blocking items (clock-sweep
  generalisation, hardcoded action stride, multi-GPU training, `requirements.txt`); 2 more
  probe bugs found and fixed live on a GTX 1650.
- **25 Aug 2026** — Rented and profiled 6 GPUs in parallel (3070, 3090, 3080, 4070 Ti, 4070,
  4090), each audited and committed separately. Built `preprocess_desktop.py` (catching a
  driver-pin bug that would have normalised some cards against the wrong clock ceiling).
  Ran the first real leave-one-GPU-out result, then a 4-seed sweep that overturned it.
  Restored the deleted do-no-harm arbiter. Then diagnosed the instability end-to-end: found
  a crushed reward scale, three RL bugs, a missing baseline suite (fixed-0.6 beats
  everything), and finally the startup-diluted telemetry root cause. Fixed the collector and
  all 12 benchmarks. Retracted the earlier "features are fine" conclusion. Wrote
  `docs/handoff-2026-08-25/FINDINGS_REPORT.md`, `docs/handoff-2026-08-25/ONBOARDING.md`, `docs/handoff-2026-08-25/CLAUDE_HANDOFF_PROMPT.md`, and
  committed the analysis scripts under `common/analysis/` so every reported number is
  reproducible from the repo alone.
- **3 Oct 2026** — Rented and re-collected 5 of 7 desktop GPUs (3060, 3070, 3080, 3090,
  4090) in parallel with the fixed window-scoped sampler, each audited (header, exact row
  count, zero duplicates, EDP recompute, physical-range, `pt_matmul` peak utilisation) and
  committed (`8e90f12`). Confirmed the fix live on real hardware before trusting any full
  sweep: instrumented a direct window-vs-whole-process comparison on the RTX 3090 showing
  the `Window:` marker correctly reaching the collector (span matched benchmark duration
  exactly) -- the one near-0% reading that looked alarming at first (`pt_matmul` at max
  clock, 3 samples in a 145ms window) turned out to be the already-documented
  short-workload sampling-noise limitation, not dilution recurring. Found and fixed two new
  driver-pin bugs (RTX 4090's new rental on a different driver with a different core
  ceiling; RTX 3060 and RTX 3090's new rentals reporting the *same* driver string as the
  original units but different real hardware limits -- core ceiling and power limit
  respectively). Added a power-limit sanity check to `preprocess_desktop.py` mirroring the
  existing core-clock one. Still need: RTX 4070, RTX 4070 Ti (rentable), RTX 3050 Ti laptop
  (physical, local re-collection). Fleet-wide `diagnose_features.py` read on the mixed
  5-fixed/3-old fleet is not yet conclusive either way -- do not cite it until the fleet is
  complete.
- **4 Oct 2026** — RTX 4070/4070 Ti unavailable to rent and the laptop deferred, so ran the
  decisive comparison on just the 5 clean GPUs instead of waiting: does RL beat a fixed
  clock now that telemetry is real? **No.** `sweep_rl.py --preset combos` (5 configs x 3
  seeds x 5 GPUs, ~77 min) gave a best RL mean of 0.038 scaled EDP, still worse than fixed
  0.5/0.6/0.7 (0.031/0.025/0.027) on the identical 5-GPU set, and RL only won on 1 of 5
  individual cards (3090) -- checked per-GPU specifically because this project has been
  burned before by a good-looking aggregate mean hiding an inconsistent per-GPU pattern.
  **Conclusion: telemetry dilution was not masking a hidden RL win.** Updated NORTHSTAR row
  11 with this. Also found and fixed a real crash in `baselines.py --loo-tune` (embedded the
  picked fixed-clock value into a dict key, which breaks the moment two leave-one-out folds
  pick different values -- happened on this 5-GPU set: 0.55-0.70 across folds, vs. a
  consistent 0.60 on the original 7-GPU fleet, itself a real finding about fleet-size
  sensitivity, not noise). Sweep output committed at
  `docs/results/sweep_combos_5gpu_partial_2026-10-04.csv`. Confirmed with the user that
  RTX 4070, RTX 4070 Ti, and the laptop are being set aside for now; this 5-GPU reading is
  the working number until/unless those become available again.
