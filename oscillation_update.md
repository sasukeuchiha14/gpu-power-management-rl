# Oscillation update (13–14 Nov 2026): did we fix the clock limit-cycle?

**Short answer: yes, on real hardware, without giving up the power saving, using a deployment-side
"cycle lock-in" (no retraining).** Reward shaping, which was the obvious route, did *not* work.
The lock-in has been tested on several rented GPU hosts (RTX 4090, 3090 and an unseen 4060 Ti), but not yet on long runs, shared
GPUs or very short phase changes (see [Open items](#open-items-and-honest-limits)).

Branch: `unified-model` (never `main`). Detail and raw numbers: [context.md](context.md),
[docs/results/](docs/results/). Earlier history of the problem: [RESULTS.md §7](RESULTS.md).

---

## 1. The problem we started from

The DDQN picks DEC / INC / HOLD for the GPU core clock. On some workloads it flips between two
adjacent clock levels forever (a *limit cycle*). The earlier fix, `reversal_cost=0.30`, suppressed the
cycle (cycle rate 0.182 → 0.022 offline, `flip_rate` 0.000 on hardware) but on **compute-bound**
work (ResNet-50 training, batch=32) every checkpoint that stopped the cycle also pinned the clock at
maximum and **erased the power saving** (net −0.7% to −1.5%). Every checkpoint that kept the saving
kept cycling. Five hardware tests, no value or seed did both. That was the open trade-off.

## 2. What we tried

| # | Approach | Result | Verdict |
|---|---|---|---|
| 1 | Reward shaping: `reversal_window` (penalise a reversal only within K steps of the last move), `hold_bonus`, `reversal_util_scale`; 11 variants screened offline (`dwell_screen.py`) | Best variant looked good offline (compute-bound saving 12.4% → 16.0%, EDP 0.020 → 0.014). On a real 4090, 3 seeds: batch=32 kept the saving but still cycled (flip_rate 0.50–0.78); seed 0 pinned at max clock (−31% at batch=4) | **Rejected.** A 3-seed offline screen overstated the effect again, the same lesson as `action_cost` |
| 2 | Understand *why* the cycle costs what it costs (logs from step 1) | Two findings below | Reframed the problem |
| 3 | **Deploy-time cycle lock-in** (`deploy_live.py --lockin-k 8`) | Works | **Accepted** |
| 4 | Refinements found by testing on harder cases: slowdown cap, work-shift release, probe-before-park, time-aware probe | Each fixed a real failure (below) | Kept |

**Two findings that reframed the problem.**
1. The cycle itself is cheap: cycling policies on batch=32 net about 5% energy/iter at 1–3% slowdown.
   The real loss was the *fix* (pinning at max clock), not the cycle.
2. On compute-bound work the **do-no-harm arbiter drives much of the "cycle"**. It was active on ~61%
   of ticks: the policy DECs, the arbiter (sees >85% SM util) jumps back to max, repeat. That is a
   4+ level pattern, not a 2-level dither, so a reward penalty on direction reversals could never
   have fixed it.

## 3. The fix: cycle lock-in

Implemented in `common/deploy_live.py` (off unless `--lockin-k` is given; no retraining, works with
any checkpoint):

1. **Detect** a short cycle: the last K=8 committed clock targets use 2–4 distinct levels with at
   least 2 direction reversals.
2. **Probe** each visited level: hold it, let telemetry settle, measure a clean EDP (`P · ms²`).
   Settle/measure windows are sized in seconds from the observed tick period (clamped to 1–4 ticks).
3. **Park** at the best level (optionally with a slowdown cap, `--lockin-max-slowdown`), and stop
   consulting policy and arbiter.
4. **Release** if live EDP degrades >15% (mean of 4 settled samples), or if settled ms/iter moves
   >25% from its value at parking (the workload changed phase).

Why each piece exists (each was a bug we hit on hardware, not a guess):
- *Settle ticks after parking*: the first measurement after a clock change still mixes the old clock.
  Without them the first version released too eagerly and held only 7–76% of ticks.
- *Probe-before-park*: NVML power and ms/iter lag a clock change, so a level visited for one tick
  records the previous level's power. Max clock, held longest by the arbiter, looked falsely cheapest
  and the lock-in re-parked there on a memory-bound phase (saving 1.4% vs 21.6% steady).
- *Work-shift release*: without it the lock-in stayed parked at the wrong clock after a phase change.
- *Time-aware probe*: a fixed 16-tick probe was 45% of a 100 s run on a slow workload.

## 4. Results (real hardware, all against the default driver)

**Headline: RTX 4090, ResNet-50 training, batch=32, 3 interleaved reps** (the exact case that failed):

| policy | power saving | slowdown | EDP gain | energy/iter | clock changes/tick |
|---|---|---|---|---|---|
| `reversal_cost=0.30` (prior fix) | −0.7% | 1.1% | −3.0% | +1.8% | 0.04 |
| old pure-log checkpoint | 7.6% | 2.8% | +2.3% | −5.0% | 0.51 |
| **old + lock-in** | **33.6%** | **14.4%** | **+13.1%** | **−24.1%** | **0.05** |
| old + lock-in, 8% slowdown cap (other host) | 23.2% | **5.9%** | **+13.8%** | n/a | 0.05 |

Clock changes fell ~10x (the oscillation is gone) and the saving went from erased to 23–34%. The
14% slowdown at no cap is the cost: it is the EDP-optimal level, but report it beside the saving.

**Generalisation:**

| case | result |
|---|---|
| Memory-bound (1 GiB elementwise), 4090 | 6.0% → **21.6%** saving, 1.8% slowdown, EDP +18.7% |
| Pure GEMM (6144², ~100% util), 4090 | no policy saves anything (default already near-optimal); lock-in costs nothing and removes the cycling |
| Phase-changing workload (GEMM ↔ memory every 30 s), 4090 | memory phase 1.4% → **20.2%** with probe; compute phase correctly parks at max (0.7%, probe cost) |
| RTX 3090, memory-bound | 10.0% → **30.8%**, 0% slower; phased memory phase 28.3% |
| RTX 3090, ResNet b=32 | modest (5–8%, EDP +3.7–4.0%): the 8% cap excluded the best-EDP level, so the cap is a dial, not a universal optimum |
| RTX 4060 Ti (**not in the training fleet**), memory-bound | 9.2% → **17.2%**, 0% slower |
| RTX 4060 Ti, ResNet b=32, 7 min | **29.1%** saving, 5.3% slower, EDP +21.4% |
| RTX 4060 Ti, ResNet b=32, 100 s | time-aware probe: probing share 45.5% → 19.7%, saving 16.1% → **22.0%**, EDP +9.7% → +14.5% |

The 4090 ResNet result reproduced across hosts (33.6%, 30.4%; one host showed a lower no-probe number
because of the telemetry-lag problem the probe later fixed). Reps agreed to within 0.2 points on the
headline host.

## 5. So, did we fix it?

- **Oscillation: fixed** where it mattered. Clock changes/tick 0.50 → 0.05; locked on ~90% of ticks
  with no spurious releases on the headline runs.
- **Power saving on compute-bound work: recovered and increased.** From −0.7% (prior fix) to 23–34%,
  with a slowdown that the cap can bound (5.9% at 23.2%).
- **The earlier "mutually exclusive" verdict no longer holds as a general statement.** It was a
  property of reward-shaped checkpoints; a deployment-side lock-in breaks it. RESULTS.md §7.3 and
  NORTHSTAR row 11 are updated accordingly.
- **Not a reward-shaping success.** Be precise in the paper: the contribution is a deployment-time
  mechanism that sits on top of the policy and arbiter, and the checkpoint is unchanged.

## 6. Open items and honest limits

- **Launch-bound batch=4 is unresolved.** Lock-in cut clock changes 0.81 → 0.23/tick but still did
  not beat the default driver on that host; every policy varied by host there. Not a lock-in
  success or failure.
- **Run length:** only 90–150 s runs on most cases, plus 7-minute runs on the 3090 and 4060 Ti.
  Hour-long runs are untested.
- **Phases shorter than ~20 s:** the probe costs ~16 ticks per detected cycle (less with the
  time-aware probe); untested.
- **Shared or multi-tenant GPUs, non-NVIDIA hardware:** untested.
- **Fleet coverage:** no VM-capable RTX 4070 / 4070 Ti / 4080 / 3080 hosts exist on vast.ai, so
  NORTHSTAR rows 2 and 10 are blocked externally, unchanged.
- **Statistics:** hardware rows are 2–3 reps per host, not a statistical study. The slowdown cap
  (8%) was not tuned per GPU.
- **Slowdown trade-off:** uncapped lock-in on ResNet costs ~14% runtime for 30%+ power. Always
  report both numbers.
- Time-aware probe was not re-run on the fast-tick workloads (the 2+2 path is arithmetically
  identical, risk judged low).

## 7. How to use and reproduce

```bash
# live control with lock-in (off by default); cap slowdown at 8%
python common/deploy_live.py ... --lockin-k 8 --lockin-max-slowdown 0.08
# optional: --lockin-ms-shift 0.25   --no-lockin-probe (old direct-park behaviour)
```

| what | where |
|---|---|
| Implementation | `common/deploy_live.py` (`LiveController._maybe_lock`, `_park`, probing/parked branches in `tick`) |
| Offline screen + lock-in eval | `common/analysis/dwell_screen.py`, `dwell_checkpoint.py`, `lockin_eval.py`, `xfer_summary.py` |
| Synthetic workloads | `common/benchmarks/live_synthetic.py` (`matmul`, `membound`, `phased`) |
| Trial scripts | `common/dwell_hw_trials.sh`, `lockin*_hw_trials.sh`, `phased_hw_trials.sh`, `probe_hw_trials.sh`, `gpu_xfer_trials.sh`, `xfer_probe_check.sh` |
| Raw logs and summaries | `docs/results/{dwell_hw_trials,lockin_hw_trials,lockin2_hw_trials,lockin3_hw_trials,phased_hw_trials,probe_hw_trials,probe_time_trials,xfer3090,xfer4060ti}/`, `dwell_screen_*.csv`, `lockin_eval.csv` |
| New trainer knobs (default off, did not help) | `reversal_window`, `hold_bonus`, `reversal_util_scale` in `common/train_unified_v2.py` |
| Checkpoints from the failed reward-shaping variant | `common/weights/dwell_win2_hold02_holdout_nvidia_geforce_rtx_4090{,_seed1,_seed2}.pth` |

Hardware spend for this round: about $2.8 of vast.ai credit across five rentals (RTX 4090 x4, 3090,
4060 Ti), all destroyed, clocks reset after each use.
