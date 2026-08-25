# GPU Power Management via Reinforcement Learning

Learning a **single** DVFS (Dynamic Voltage and Frequency Scaling) policy that sets GPU core clock at runtime to minimise **Energy-Delay Product** (EDP = Power × Time²), reads only live NVML telemetry, and runs unmodified on any RTX 3000/4000 consumer GPU — *including cards it never trained on*.

## ⚠️ Clone the `unified-model` branch, not `main`

`main` holds older teammate work under a superseded architecture. **All current work is on `unified-model`.**

```bash
# clone straight into the right branch
git clone -b unified-model https://github.com/sasukeuchiha14/gpu-power-management-rl.git
cd gpu-power-management-rl
```

<details><summary>Already cloned and got <code>main</code>?</summary>

```bash
git fetch origin
git checkout unified-model      # or: git switch unified-model
git branch --show-current       # must print: unified-model
```
</details>

**New here? Start with the handoff bundle:** **[docs/handoff-2026-08-25/](docs/handoff-2026-08-25/)** — it has the reading order, the results, and a copy-paste prompt for starting a Claude Code session.

---

## Status at a glance

| | |
|---|---|
| **GPUs profiled** | 8 — RTX 3060 / 3070 / 3080 / 3090 (Ampere), RTX 4070 / 4070 Ti / 4090 (Ada), RTX 3050 Ti laptop |
| **Profiled operating points** | ~2 100 across 12 workloads, each dataset individually audited |
| **Headline result** | A fixed clock at **0.6 × max** beats every learned policy tested |
| **Open blocker** | Telemetry in all committed datasets is startup-contaminated; fleet needs re-collection |
| **Deliverable** | A journal paper — rigor and honest reporting take priority over speed |

---

## The two results worth knowing

### 1. A static clock is a very strong baseline

Steady-state scaled EDP, leave-one-GPU-out (0 = that workload's best profiled point, 1 = its worst; lower is better):

| Policy | Mean | Uses telemetry? | Learned? |
|---|---|---|---|
| oracle (per-workload best) | 0.000 | — | — |
| **fixed clock @ 0.6 × max** | **0.037** | no | no |
| supervised classifier (feature ceiling) | 0.052 | yes | yes |
| best RL configuration found | ~0.069 | yes | yes |
| random walk | 0.251 | no | no |
| **governor (pinned max clock)** | **0.319** | no | no |
| utilisation heuristic | 0.583 | yes | no |
| min clock | 0.819 | no | no |

**A single static clock at 60% of maximum — no telemetry, no learning — captures ~88% of the gap between the stock max-clock governor and a per-workload oracle, and beats every learned policy we tested.** Tuned leave-one-GPU-out (on the 7 training GPUs only), it selects 0.60 on all 8 folds, so this is not hindsight fitting.

Reproduce: `python common/analysis/baselines.py --loo-tune`

### 2. The telemetry we trained on was measuring mostly idle time

`collect_dataset.py` sampled NVML across the entire benchmark subprocess — interpreter startup, `import torch`, CUDA context creation, typically 2–4 s of idle GPU — while the actual timed work is often 0.15–0.6 s.

| Evidence across the 8-GPU fleet | Value |
|---|---|
| corr(GPU-only duration, recorded utilisation) | **+0.708** |
| points with < 0.5 s real work | 494/799 (62%), mean recorded util **8.3%** |
| **highest utilisation recorded anywhere** | **64.4%** |
| `pt_matmul` (dense GEMM — should saturate) | **21.7%** |

Recorded utilisation tracks *how long a workload ran relative to fixed startup cost*, not how hard it drove the GPU. **`Avg_Time_S` and EDP are unaffected** — timing comes from each benchmark's own measurement of its own work — so result #1 above stands. But the telemetry channels are exactly what a controller needs to characterise an unknown workload, which makes the question *"can a learned controller beat a static clock?"* currently **unanswerable rather than answered**.

Fixed in code (window markers + window-scoped averaging, verified 25.2% → 95.0% on a synthetic trace). The fix cannot repair existing data — the fleet needs re-collecting.

Reproduce: `python common/analysis/telemetry_dilution.py`

---

## Quickstart

```bash
git clone -b unified-model https://github.com/sasukeuchiha14/gpu-power-management-rl.git
cd gpu-power-management-rl
git branch --show-current          # sanity check: must print "unified-model"

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No GPU is required for analysis, training or evaluation — only for collecting new data.

**Verify your environment reproduces our numbers (~3 min):**

```bash
python common/analysis/telemetry_dilution.py    # expect corr = +0.708
python common/analysis/diagnose_features.py     # expect "core_frac ONLY  0.781"
```

**Train and evaluate a leave-one-GPU-out policy:**

```bash
python common/train_unified.py --holdout nvidia_geforce_rtx_4090 --episodes 800 --seed 42
python common/evaluate_holdout.py --holdout nvidia_geforce_rtx_4090
python common/analysis/sweep_rl.py --preset screen     # which knobs actually help
```

**Collect data on a rented GPU** — see the per-instance checklist in [context.md](context.md). Rent **VM-type** instances only; container templates deny clock control entirely.

---

## Documentation

**📦 [docs/handoff-2026-08-25/](docs/handoff-2026-08-25/) — the handoff bundle.** Start there; it has the reading order and links everything below.

| Read | For |
|---|---|
| **[context.md](context.md)** | Where the project is right now and what to do next. Rewritten every session. |
| **[docs/handoff-2026-08-25/FINDINGS_REPORT.md](docs/handoff-2026-08-25/FINDINGS_REPORT.md)** | What we found, what it means, what to change and experiment with. **The main document.** |
| **[NORTHSTAR.md](NORTHSTAR.md)** | End-goal definition + the 11-row scorecard that decides what's on the critical path. |
| [docs/handoff-2026-08-25/rl_instability_investigation.md](docs/handoff-2026-08-25/rl_instability_investigation.md) | Full investigation log, including hypotheses the data disproved. |
| [docs/handoff-2026-08-25/ONBOARDING.md](docs/handoff-2026-08-25/ONBOARDING.md) | New-contributor setup, repo layout, working agreements. |
| [docs/handoff-2026-08-25/CLAUDE_HANDOFF_PROMPT.md](docs/handoff-2026-08-25/CLAUDE_HANDOFF_PROMPT.md) | Copy-paste prompt to start a fresh Claude Code session on this project. |
| [CLAUDE.md](CLAUDE.md) | Deep technical reference: architecture, RL formulation, file map, known issues. |

---

## How it works

**Offline** — profile each GPU across a grid of (workload, core clock, memory clock), recording power, temperature, utilisation, VRAM and GPU-only duration; compute EDP per point.

**State** — 6 features, each normalised against *that card's own* NVML-reported limits, so `0.7` means the same physical thing on a 70 W laptop chip and a 450 W RTX 4090: `core_frac`, `sm_util`, `mem_util`, `power_frac`, `temp_frac`, `vram_frac`. No workload one-hot (it cannot exist for an unseen app) and no memory clock (mobile parts reject memory-clock control, and one unified action space is worth more than per-tier capability).

**Action** — 3 discrete: decrease core / increase core / hold.

**Reward** — `1 − per-workload-minmax(EDP)`, re-normalised over the operating points a core-only controller can actually reach.

**Training** — a DDQN over a table-backed environment built from the profiled points, with leave-one-GPU-out support for the generalisation test.

**Deployment** — a userspace daemon reading NVML at 50 ms, guarded by a thermal override (Layer 1) and a do-no-harm arbiter (Layer 2). `DRY_RUN = True` by default.

---

## Repository layout

```
common/                    shared, GPU-agnostic code — the current architecture
  features.py              the 6 state features + action set (single source of truth)
  gpu_probe.py             capability probe — run FIRST on any new machine
  collect_dataset.py       2D core x memory profiler (window-scoped telemetry)
  preprocess_desktop.py    raw CSV -> 6-feature preprocessed_dataset.csv
  dataset_view.py          core-only view with correctly re-normalised reward
  unified_env.py           multi-GPU env + leave-one-GPU-out
  train_unified.py         original trainer      train_unified_v2.py  configurable, bug-fixed
  eval_policies.py         policy-agnostic evaluation + baseline suite
  evaluate_holdout.py      leave-one-GPU-out evaluation
  analysis/                the scripts behind every reported number
  benchmarks/              workloads added by this project
  fleet_survey/            per-GPU capability survey
  weights/  holdout_results/    trained policies and results

gpus/<gpu_slug>/data/      one folder per profiled GPU (8 of them)
gpus/rtx3050ti_laptop/     laptop tree: benchmarks, deployment daemon, arbiter
gpus/rtx4500_ada/          older professional-card tree — out of scope, kept as contrast case
docs/                      reports, investigation logs, onboarding
docs/handoff-2026-08-25/results/              CSV outputs from the analysis scripts
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for hardware-safety rules (always `DRY_RUN` first; the thermal override stays on) and [docs/handoff-2026-08-25/ONBOARDING.md](docs/handoff-2026-08-25/ONBOARDING.md) for working agreements. In short: work on `unified-model`, never push to `main`, never fabricate a result, and treat single-seed RL numbers as not-yet-evidence.

## License

[MIT](LICENSE).
