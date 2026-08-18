# Live A/B Test Report — 18 Aug 2026

Follow-on to `dvfs-triage` and the 3 July 2026 `Capstone Testing & Validation
Report`'s validation ladder: this covers rungs **"Live A/B measurement"** and
**"Zero-shot generalization"**, plus ad-hoc **latency/overhead** measurement.

## Method

For each workload: `measure_edp.py governor 20 <workload>` (pins clocks to
max, runs the workload 20x, samples NVML power at 10Hz), then
`sim_to_real_deploy.py` is started fresh, given ~1.5s to initialize, and
`measure_edp.py ai 20 <workload>` runs the identical 20x workload while the
daemon controls clocks live. The daemon is stopped with `SIGINT` immediately
after (clean shutdown resets clocks to default) — no process was left running
between or after tests. `summarize_tests.py` then rebuilds the table/graph
below directly from the saved `power_governor_*.csv` / `power_ai_*.csv`
pairs, not from hand-copied numbers.

Two workload sets, per the requested plan:
- **in-dataset** (`pathfinder`, `nw`, `streamcluster`): profiled and trained
  on — this is the "does it actually help on what it knows" check. `nw`'s
  result is the original validated run from before today; `pathfinder` and
  `streamcluster` are new.
- **zero-shot** (`resnet18`, `transformer`, `elementwise`): never profiled,
  never trained on, not even similar-in-name to a trained workload's exact
  config. This is the honest test of `dvfs-triage` B1's core concern — the
  model is handed a one-hot workload identity as input, so on a genuinely
  unseen app the live daemon can only nearest-neighbour-match live sensor
  readings to whichever trained workload looks closest, then act on that
  borrowed identity. `resnet18` is a different depth/width CNN than the
  trained `pt_resnet50`/`pt_cnn`; `transformer` is a plain
  `nn.TransformerEncoder` stack (attention-based, unlike the trained set's
  only NLP entry, encoder-only BERT); `elementwise` is a memory-bandwidth-bound
  CuPy op chain, structurally unlike the trained `cupy_matmul`/`cupy_fft_fixed`.

## Results

| Workload | Category | Governor (W / J / s) | AI (W / J / s) | Energy saved | Power saved | Runtime delta |
|---|---|---|---|---|---|---|
| pathfinder | in-dataset | 40.8 / 142.0 / 3.48 | 28.0 / 94.2 / 3.36 | **+33.6%** | +31.4% | -3.4% (faster) |
| nw | in-dataset | 43.6 / 1367.1 / 31.34 | 31.0 / 1005.4 / 32.37 | **+26.5%** | +28.8% | +3.3% |
| streamcluster | in-dataset | 51.8 / 2378.3 / 45.95 | 45.6 / 2395.0 / 52.54 | **-0.7%** | +12.0% | +14.3% (slower) |
| resnet18 | zero-shot | 81.9 / 3366.2 / 40.89 | 59.4 / 2683.7 / 44.85 | **+20.3%** | +27.5% | +9.7% |
| transformer | zero-shot | 81.0 / 1695.3 / 20.90 | 67.3 / 1466.9 / 21.79 | **+13.5%** | +16.9% | +4.3% |
| elementwise | zero-shot | 110.8 / 2621.7 / 23.64 | 97.8 / 2666.7 / 27.22 | **-1.7%** | +11.7% | +15.2% (slower) |

See `logs_and_plots/test_results_summary.png` for the graph and
`logs_and_plots/edp_comparison_<workload>.png` for each individual power-trace
comparison, plus `logs_and_plots/test_results_summary.md` (regenerable via
`python3 evaluation/summarize_tests.py`).

## Honest reading of the results

**4 of 6 workloads show a real energy win (13.5% to 33.6%), including 2 of
the 3 zero-shot ones.** This is a genuinely encouraging generalization
result — the agent is not just memorizing the training table, it's reacting
to live telemetry in a way that transfers. `pathfinder` and `resnet18` won on
*both* energy and (for pathfinder) runtime; none of the zero-shot workloads
regressed catastrophically.

**But 2 workloads regressed slightly** (`streamcluster` -0.7%, `elementwise`
-1.7%) — energy went up, not down, because mean power dropped (11-12%) but
wall-clock time grew more (14-15%), netting a small loss. Notably this is
*not* a clean in-dataset-wins/zero-shot-loses split: `streamcluster` is
in-dataset and regressed; `resnet18`/`transformer` are zero-shot and won.
The pattern looks more like: workloads with a genuinely low-clock optimum in
the training data (`pathfinder`, `nw`) transfer best; workloads whose live
utilization signal is noisier/more variable (`streamcluster`'s util swung
44-66% during its run, causing frequent action-flipping — see the raw daemon
log) transfer worse, independent of whether they were trained on.

**The do-no-harm arbiter did not trip on any of these 6 runs** (`grep -c ARB`
on every daemon log returns 0). For `streamcluster`/`elementwise` that means
the Layer-2 safety net's nearest-neighbour EDP proxy didn't detect the
regression in time to revert to the governor — the *real* outcome (measured
after the fact from wall-clock energy) was worse than the proxy the arbiter
uses tick-to-tick. This is worth flagging as a real limitation, not
something to paper over: the arbiter is not a guarantee against every kind
of regression, only the ones its live proxy can see quickly enough.

## Latency and overhead (validation report's "Latency budget" / "Overhead" tests)

| Workload | Mean loop latency | Max loop latency | Daemon CPU | Daemon RSS |
|---|---|---|---|---|
| pathfinder | 76.5 ms | 108.1 ms | ~80% | ~750 MB |
| streamcluster | 69.4 ms | 93.3 ms | ~80% | ~750 MB |
| resnet18 | 78.4 ms | 197.5 ms | ~81% | ~750 MB |
| transformer | 88.1 ms | 195.8 ms | ~81% | ~750 MB |
| elementwise | 73.8 ms | 149.0 ms | ~79% | ~750 MB |

**Both numbers miss their targets, consistently across every workload** — the
50ms latency budget (actual: 70-90ms mean, up to ~200ms peak) and the
"<1% CPU, <50MB RAM" overhead target (actual: ~80% of one core, ~750MB RSS).
This is `dvfs-triage` B6 (shelling out to `nvidia-smi` twice per tick costs
10-30ms per spawn) plus the daemon's PyTorch inference running on CPU, not
GPU — the base Python environment's CUDA is unavailable to torch here for
the same driver-version-mismatch reason documented in `CLAUDE.md` (this
daemon still launches via bare `python3`, unlike the data-collection scripts
which were already repointed at the working `llm_gpu` env earlier today).
**Not fixed in this pass** — B6's proper fix (in-process
`nvmlDeviceSetGpuLockedClocks` instead of shelling out) needs the whole
daemon process to run with elevated privileges, and this machine's
passwordless-sudo rule is scoped to exactly `/usr/bin/nvidia-smi`, not
arbitrary processes — extending that scope is a sudoers change outside what
should be done without explicit sign-off. Flagging as follow-up, not doing
it silently.

## What this does and doesn't prove

Confirms: downward-control transfers to genuinely unseen workloads at least
some of the time (this was previously untested — the 3 July report's
"Zero-shot generalization" rung was still open). Real energy savings exist
on both in-dataset and zero-shot workloads.

Does not prove: that the model reliably generalizes — 2 of 6 regressed, and
the failure mode (noisy live utilization -> jittery actions -> worse
wall-clock time despite lower mean power) is exactly the class of problem
`dvfs-triage` B1 predicts will get worse, not better, as more truly novel
applications are thrown at it, because the state still doesn't tell the
agent *how confident* its identity guess is.
