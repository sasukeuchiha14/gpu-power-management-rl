# Prompt to paste into a fresh Claude Code session

Copy everything in the box below into a new Claude Code session, started from the repo root.

> Clone with `git clone -b unified-model https://github.com/sasukeuchiha14/gpu-power-management-rl.git` — a plain `git clone` lands on `main`, which is superseded work. It orients Claude on the project, the current state, and the known traps, without it having to re-derive anything.

---

```
I'm picking up an in-progress research project. The deliverable is a journal paper,
so rigor and honest reporting matter more than speed -- never fabricate or
smooth over a result, and flag anything unverified as unverified.

First confirm you're on the right branch -- run `git branch --show-current`.
It must print `unified-model`. If it prints `main`, run `git switch unified-model`
first: `main` is superseded work and none of the below will match it.

Then read these, in this order, before doing anything else. They all live in
docs/handoff-2026-08-25/ except where noted:
  1. context.md  (repo root)               <- where we are, what's next
  2. docs/handoff-2026-08-25/FINDINGS_REPORT.md   <- what we found + what to do (main doc)
  3. docs/handoff-2026-08-25/RESULTS.md           <- every measured number, with repro commands
  4. NORTHSTAR.md  (repo root)             <- end goal + the 11-row scorecard
  5. docs/handoff-2026-08-25/ONBOARDING.md        <- setup, repo layout, working agreements
  6. docs/handoff-2026-08-25/rl_instability_investigation.md
                                           <- full investigation log, incl. disproved hypotheses
  7. CLAUDE.md  (repo root)                <- deep technical reference, as needed

Then verify your environment reproduces our numbers:
  python common/analysis/telemetry_dilution.py   # expect corr = +0.708
  python common/analysis/diagnose_features.py    # expect "core_frac ONLY  0.781"

Two things that will mislead you if you don't know them:
  - The committed telemetry data is known-bad: NVML was sampled across whole
    benchmark processes including seconds of startup, so recorded utilisation and
    power mostly reflect startup overhead, not workload behaviour. The collector
    is fixed; the existing 8 datasets are not. Don't conclude anything about
    telemetry-driven adaptation from them.
  - Timing and EDP ARE correct. Anything resting only on those -- including our
    headline result that a fixed clock at 0.6x max beats every learned policy --
    is trustworthy.

Working agreements: work on branch `unified-model`, never push to `main`.
Single-seed RL results are not evidence in this project; use multi-seed with
variance. Sanity-check any computed metric against raw values by hand at least
once before trusting it.

Tell me what you think the highest-value next step is and why, before starting work.
```

---

## Shorter version, if they just want to get going

```
Check `git branch --show-current` says `unified-model` (if not, `git switch
unified-model`). Then read context.md, then docs/handoff-2026-08-25/FINDINGS_REPORT.md,
then docs/handoff-2026-08-25/ONBOARDING.md. It's a GPU DVFS reinforcement-learning research
project and the deliverable is a journal paper, so don't fabricate results and
flag anything unverified. Two traps: the committed telemetry data is known-bad
(sampled across process startup -- see FINDINGS_REPORT section 4), but timing and
EDP are correct. Then tell me what you think the best next step is.
```
