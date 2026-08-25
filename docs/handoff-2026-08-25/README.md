# Handoff bundle — 25 August 2026

**Everything a new person (or a new Claude Code session) needs to pick this project up.** Self-contained snapshot of where the project stood on this date.

> **Repo:** https://github.com/sasukeuchiha14/gpu-power-management-rl
> **Branch: `unified-model`** — not `main`. See [ONBOARDING.md](ONBOARDING.md#1-get-the-code).

---

## Read in this order

| # | Document | What it gives you | Time |
|---|---|---|---|
| 1 | [**../../context.md**](../../context.md) | Where the project is *right now* and the single next action. Lives at repo root because it's rewritten every session. | 5 min |
| 2 | [**FINDINGS_REPORT.md**](FINDINGS_REPORT.md) | **The main document.** What we did, what we found, what it means, what to change, what to experiment with next (prioritised), and what is safe to claim in the paper. | 20 min |
| 3 | [**RESULTS.md**](RESULTS.md) | Every number we measured, in tables, each with the command that regenerates it. Raw CSVs in [`results/`](results/). | 15 min |
| 4 | [**../../NORTHSTAR.md**](../../NORTHSTAR.md) | The end-goal definition and the 11-row scorecard that decides whether a piece of work is on the critical path. | 10 min |
| 5 | [**ONBOARDING.md**](ONBOARDING.md) | Clone, branch, environment setup, a 3-minute check that reproduces our headline numbers, repo layout, working agreements. | 10 min |
| 6 | [**rl_instability_investigation.md**](rl_instability_investigation.md) | The full investigation log in the order things were measured, *including hypotheses the data disproved and one conclusion we retracted*. Read before re-deriving anything. | 15 min |
| 7 | [**../../CLAUDE.md**](../../CLAUDE.md) | Deep technical reference: architecture, RL formulation, hyperparameters, file map, known issues. Consult as needed rather than front-to-back. | reference |

**Starting a Claude Code session?** [CLAUDE_HANDOFF_PROMPT.md](CLAUDE_HANDOFF_PROMPT.md) has a copy-paste prompt that orients Claude on all of the above, including the two traps below.

---

## The 60-second version

**What works:** 8 GPUs profiled across two architecture generations (2 098 operating points, each dataset independently audited). A full analysis, baseline and evaluation stack that runs on CPU in minutes.

**The headline finding:** a **fixed clock at 60 % of maximum** — no telemetry, no learning — captures **~88 %** of all achievable energy-delay improvement over the stock max-clock governor, and beats every learned policy we tested. It is verified leave-one-GPU-out (tuned on 7 GPUs, evaluated on the 8th; picks 0.60 on all 8 folds).

**The blocker:** the telemetry in every committed dataset was sampled across whole benchmark processes including 2–4 s of startup, while the real work was often 0.15–0.6 s. So recorded utilisation/power encode startup overhead rather than workload behaviour. The collector is fixed; the data is not. **The fleet needs re-collecting** before anything about learned adaptation can be concluded.

## Two things that will mislead you if you don't know them

1. **The committed telemetry data is known-bad.** Do not draw conclusions about telemetry-driven adaptation from it. See [RESULTS.md §6](RESULTS.md#6-️-the-measurement-bug-that-gates-3-4-and-5).
2. **Timing and EDP are correct.** Anything resting only on those — including the headline fixed-clock result — is trustworthy. See [RESULTS.md §7](RESULTS.md#7-what-is-safe-to-claim) for the claim-by-claim breakdown.

## Verify your setup reproduces our numbers (~3 min)

```bash
python common/analysis/telemetry_dilution.py    # expect corr = +0.708
python common/analysis/diagnose_features.py     # expect "core_frac ONLY  0.781"
```

If those match, every number in this bundle is reproducible on your machine.
