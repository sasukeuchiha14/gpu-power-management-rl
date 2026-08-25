# Prompt to paste into a fresh Claude Code session

Copy everything in the box below into a new Claude Code session, started from the repo root on the `unified-model` branch. It orients Claude on the project, the current state, and the known traps, without it having to re-derive anything.

---

```
I'm picking up an in-progress research project. The deliverable is a journal paper,
so rigor and honest reporting matter more than speed -- never fabricate or
smooth over a result, and flag anything unverified as unverified.

Read these files first, in this order, before doing anything else:
  1. context.md                            <- where we are, what's next
  2. docs/FINDINGS_REPORT.md               <- what we found and what to do (main doc)
  3. NORTHSTAR.md                          <- end goal + the 9-row scorecard
  4. docs/rl_instability_investigation.md  <- full investigation log, incl. disproved hypotheses
  5. docs/ONBOARDING.md                    <- setup, repo layout, working agreements
  6. CLAUDE.md                             <- deep technical reference, as needed

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
Read context.md, then docs/FINDINGS_REPORT.md, then docs/ONBOARDING.md in this
repo (branch: unified-model). It's a GPU DVFS reinforcement-learning research
project and the deliverable is a journal paper, so don't fabricate results and
flag anything unverified. Two traps: the committed telemetry data is known-bad
(sampled across process startup -- see FINDINGS_REPORT section 4), but timing and
EDP are correct. Then tell me what you think the best next step is.
```
