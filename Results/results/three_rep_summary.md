# Three-rep summary — meets this project's own citability minimum

Three independent full runs (fresh workload + fresh controller process each time), same rented
RTX 4090, same checkpoint (`response_aware_holdout_nvidia_geforce_rtx_4090.pth`), same 120s/phase
duration, spread over ~2 hours (8 Oct 2026). `glmark2` excluded throughout — confirmed
reproducibly broken (`glmark2-drm` hangs on the same scene every time), not included below.

## Power saving (%) — the headline metric

| Workload | Rep 1 | Rep 2 | Rep 3 | Mean | Std dev |
|---|---|---|---|---|---|
| resnet50 (batch=4) | +21.9 | +20.5 | +22.0 | **+21.5%** | ±0.8 |
| sparse_gather | -2.3 | -2.8 | -2.8 | **-2.6%** | ±0.3 |
| transformer_decode | +29.7 | +31.9 | +32.8 | **+31.5%** | ±1.6 |

**This is solid.** All three workloads land within 1-2 percentage points of their own mean across
three independent runs. This is the number to cite.

## Energy saving (%) — tracks power saving closely, same solidity

| Workload | Rep 1 | Rep 2 | Rep 3 | Mean | Std dev |
|---|---|---|---|---|---|
| resnet50 | +22.1 | +21.0 | +22.0 | **+21.7%** | ±0.6 |
| sparse_gather | -2.3 | -2.8 | -2.8 | **-2.6%** | ±0.3 |
| transformer_decode | +29.8 | +32.1 | +33.0 | **+31.6%** | ±1.7 |

## Slowdown (%) — the weak link, read this carefully

| Workload | Rep 1 | Rep 2 | Rep 3 | Mean | Std dev |
|---|---|---|---|---|---|
| resnet50 | -15.4 | -18.2 | -13.6 | -15.7% | ±2.3 |
| sparse_gather | +0.1 | +0.0 | +0.0 | +0.03% | ±0.06 |
| transformer_decode | **-12.6** | **+0.3** | **+6.0** | -2.1% | ±9.5 |

- **`sparse_gather`: reliably ~0%, three times.** Makes physical sense — the controller barely
  touches the clock on this workload (see power table), so there's nothing to slow down. This is
  a genuinely solid null result.
- **`resnet50`: consistently *negative* across all three reps** (i.e., it appears to run faster
  under the controller's lower clock than at pinned max). The direction is repeatable
  (std ±2.3pp around a mean of -15.7%, never crosses zero) — **not just noise**, but also not
  yet explained. Likely candidates worth checking before trusting the number: a warmup/thermal
  difference between the two phases (pinned-max may run hotter over 120s and throttle), or a
  measurement-window artifact from `ab_benchmark_run.py`'s own sampling. **Do not cite this
  number as "the controller speeds things up" without first explaining the mechanism** — report
  it as an open, repeatable anomaly if it goes in the paper at all.
- **`transformer_decode`: the sign itself isn't stable** (-12.6% → +0.3% → +6.0%, std ±9.5pp is
  larger than the mean). **This number is not trustworthy from this data and should not be
  cited.** Treat it as "no measurable slowdown effect detected" rather than picking any one of
  the three values.

## EDP change (%) — inherits the slowdown column's reliability (or lack of it)

| Workload | Rep 1 | Rep 2 | Rep 3 | Mean | Std dev |
|---|---|---|---|---|---|
| resnet50 | +33.9 | +35.0 | +32.6 | **+33.8%** | ±1.2 |
| sparse_gather | -2.4 | -2.8 | -2.8 | **-2.7%** | ±0.2 |
| transformer_decode | +38.5 | +31.7 | +28.8 | +33.0% | ±5.0 |

resnet50 and sparse_gather's EDP numbers are solid (tight std, because their slowdown numbers
were either consistent or reliably ~0). transformer_decode's EDP is noisier (±5pp) because it's
built on top of the unreliable slowdown number above — report the power/energy saving for this
workload, not the EDP figure, until slowdown is understood.

## Bottom line — what's actually citable now

- **Power saving and energy saving: citable for all three workloads**, 3-rep standard met,
  tight agreement.
- **EDP: citable for resnet50 and sparse_gather**, not yet for transformer_decode.
- **Slowdown: citable as "~0%" for sparse_gather only.** resnet50's negative slowdown is a real,
  repeatable pattern that needs a mechanism before it belongs in the paper. transformer_decode's
  slowdown is not citable in any form from this data.
- **`glmark2` contributes nothing yet** — needs the hang fixed first.
