# Onboarding — start here if you just cloned this repo

You are picking up an in-progress research project whose deliverable is a **journal paper**. Rigor and honest reporting matter more than speed here: a plausible-sounding but unverified number is worse than no number.

---

## 1. Get the code

```bash
# clone directly into the right branch (-b unified-model)
git clone -b unified-model https://github.com/sasukeuchiha14/gpu-power-management-rl.git
cd gpu-power-management-rl

git branch --show-current      # must print: unified-model
```

**If you already cloned and landed on `main`:**

```bash
git fetch origin
git switch unified-model       # older git: git checkout unified-model
git branch --show-current
```

> **Work on `unified-model`, never `main`.** `main` holds older teammate work under a superseded architecture — different state design, different results, none of the work described in these docs. If `git branch --show-current` prints anything other than `unified-model`, stop and fix that before reading further, or nothing below will match what you see.

## 2. Environment

Python 3.10+. No GPU is needed for analysis, training, or evaluation — only for collecting new data.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes `torch` (CPU build is fine for everything except data collection), `gymnasium`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `nvidia-ml-py`.

## 3. Verify your setup in about 3 minutes

```bash
python common/analysis/telemetry_dilution.py      # seconds
python common/analysis/diagnose_features.py       # ~2 min
```

The first should report `corr(GPU-only duration, recorded utilisation) = +0.708`. The second should report `core_frac ONLY  0.781`. If you see those, your environment matches ours and every number in the docs is reproducible on your machine.

## 4. Read, in this order

| # | File | Why |
|---|---|---|
| 1 | [`../context.md`](../../context.md) | Where we are right now and what to do next. Rewritten every session. |
| 2 | [`FINDINGS_REPORT.md`](FINDINGS_REPORT.md) | What we found, what it means, what to change and experiment with. **The main document.** |
| 3 | [`../NORTHSTAR.md`](../../NORTHSTAR.md) | The end-goal definition and the 9-row scorecard that decides whether work is on the critical path. |
| 4 | [`rl_instability_investigation.md`](rl_instability_investigation.md) | The full investigation log, including hypotheses the data disproved. Read before re-deriving anything. |
| 5 | [`../CLAUDE.md`](../../CLAUDE.md) | Deep technical reference: RL formulation, architecture, file map, known issues. Consult as needed. |

## 5. The two things most likely to trip you up

1. **The committed telemetry data is known-bad.** NVML was sampled across whole benchmark processes including several seconds of startup, so recorded utilisation/power mostly reflect startup overhead. The collector is fixed but the existing datasets are not. Do not draw conclusions about telemetry-driven adaptation from them. See FINDINGS_REPORT §4.
2. **`Avg_Time_S` and EDP are fine.** Timing comes from each benchmark's own measurement of its own work. Anything resting only on timing/energy — including the headline fixed-clock result — is trustworthy.

## 6. Repo layout

```
common/                     shared, GPU-agnostic code (the current architecture)
  features.py               the 6 canonical state features + action set. Single source of truth.
  gpu_probe.py              capability probe — run FIRST on any new machine
  collect_dataset.py        2D core x memory profiler (telemetry window fix applied)
  preprocess_desktop.py     raw CSV -> 6-feature preprocessed_dataset.csv
  dataset_view.py           core-only view with correctly re-normalised reward
  unified_env.py            multi-GPU Gymnasium env + leave-one-GPU-out support
  train_unified.py          original trainer (kept for reference)
  train_unified_v2.py       configurable trainer with the three RL bug fixes
  eval_policies.py          policy-agnostic evaluation + the baseline suite
  evaluate_holdout.py       leave-one-GPU-out evaluation
  aggregate_seed_sweep.py   multi-seed variance aggregation
  analysis/                 the scripts that produced every reported number
  benchmarks/               workloads added this project (sparse gather, transformer decode)
  fleet_survey/             per-GPU capability survey (which cards allow clock control)
  weights/                  trained policies + reward histories
  holdout_results/          per-GPU leave-one-out results

gpus/<gpu_slug>/data/       one folder per profiled GPU (8 of them)
gpus/rtx3050ti_laptop/      laptop tree: benchmarks, deployment daemon, arbiter
gpus/rtx4500_ada/           older professional-card tree — OUT OF SCOPE for the product,
                            kept as a supplementary contrast case
docs/                       reports and investigation logs (you are here)
docs/results/               CSV outputs from the analysis scripts
```

## 7. Working agreements

* **Never push to `main`.** All work goes to `unified-model`.
* **Never fabricate a result.** If something is unmeasured, say so. The paper depends on this.
* **Single-seed results are not evidence** in this project — see FINDINGS_REPORT §3.2 for why. Multi-seed with variance before anything is reported.
* **Sanity-check computed metrics against raw values by hand at least once.** A real bug was caught this way (an averaged-percentage metric that blew up to −2957% on one near-zero-baseline workload).
* **On a rented GPU box:** run `gpu_probe.py` first, confirm `USABLE`, report back before committing to hours of paid profiling. Rent **VM-type** vast.ai instances only — container templates deny clock control entirely.
