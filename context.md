# context.md — read this first

If you're a new Claude Code session (or a new person) picking this project up, read this
file before anything else. It's the only file in this repo meant to be rewritten every
session — everything else changes slowly.

**Read next, in order:** [docs/FINDINGS_REPORT.md](docs/FINDINGS_REPORT.md) (what we found,
what to do next — the main document) → [NORTHSTAR.md](NORTHSTAR.md) (end goal + scorecard) →
[docs/ONBOARDING.md](docs/ONBOARDING.md) (setup, layout, working agreements) →
[CLAUDE.md](CLAUDE.md) (deep technical reference, as needed).

**Branch:** `unified-model`. Never push to `main`.

---

## IMMEDIATE NEXT STEP (as of 25 Aug 2026): re-collect the fleet with the fixed sampler

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
| `docs/FINDINGS_REPORT.md` | What we found, what it means, what to do/experiment next. | When results change |
| `docs/rl_instability_investigation.md` | Full investigation log, including disproved hypotheses. | Append-only |
| `docs/ONBOARDING.md` | New-contributor setup, repo layout, working agreements. | Rarely |
| `docs/CLAUDE_HANDOFF_PROMPT.md` | Copy-paste prompt to start a fresh Claude session. | Rarely |
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
  `docs/FINDINGS_REPORT.md`, `docs/ONBOARDING.md`, `docs/CLAUDE_HANDOFF_PROMPT.md`, and
  committed the analysis scripts under `common/analysis/` so every reported number is
  reproducible from the repo alone.
