# Verification pass — raw results

Independent A/B hardware verification run by Gagan, 8 Oct 2026, on a rented vast.ai RTX 4090
(single-GPU, 24GB VRAM, matching the project's existing `response_aware_holdout_nvidia_geforce_rtx_4090.pth`
checkpoint). Data collected with `verification/analysis/ab_benchmark_run.py`, plotted with
`verification/analysis/ab_benchmark_plot.py` — both written independently of Govind's
`common/demo_power_trace.sh` / `docs/demo_power_trace.html`, per the team's request to not just
re-trust the existing graphs.

**Single-trial results. Not yet repeated. Do not cite in the paper without 2-3 more reps**,
consistent with this project's own standard (see METHODOLOGY.md / RESULTS.md).

## Workloads tested

| Workload | Shape | Status |
|---|---|---|
| `resnet50` (batch=4) | compute-bound, launch-bound condition | **Valid** — real controller decisions confirmed in log |
| `sparse_gather` | memory-bandwidth-bound (random-index gather) | **Valid** |
| `transformer_decode` | autoregressive decode (LLM-inference-shaped) | **Valid** |
| `glmark2` | real OpenGL rendering | **Excluded — see below** |

## `glmark2` exclusion

`glmark2-drm` hung on the same scene (`build use-vbo=true`) in both the pinned and controlled
phases of this run (`util_pct` frozen near 1%, `ms_per_iter` frozen at exactly 20.0 for the full
120s, and `deploy_live.py`'s own staleness guard fired: `workload status stale/missing -- forcing
max clock (safety)`). This matches a hang seen independently during an earlier smoke test of the
same binary on the same host. Raw (broken) data kept in `glmark2/` for the record, but it is
**not** included in `summary_valid.csv` — treat as a reproducible `glmark2-drm` compatibility
issue to debug separately, not a result.

## Summary

See `summary_valid.csv` and each workload's `*_power_trace.png`. The `slowdown`/EDP columns are
noisier than the power columns — see the session's own discussion for why (brief per-iteration
timing jitter present in both phases, not a clock effect) before trusting those specific numbers.
