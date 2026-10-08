# Rep 1 vs Rep 2 — run-to-run spread

Two independent full runs (fresh workload process + fresh controller process each time) on the
same rented RTX 4090, same checkpoint, same 120s/phase duration, ~15 minutes apart.

| Workload | Power saving (rep1 → rep2) | Slowdown (rep1 → rep2) | EDP change (rep1 → rep2) |
|---|---|---|---|
| resnet50 | +21.9% → +20.5% | -15.4% → -18.2% | +33.9% → +35.0% |
| sparse_gather | -2.3% → -2.8% | +0.1% → +0.0% | -2.4% → -2.8% |
| transformer_decode | +29.7% → +31.9% | -12.6% → +0.3% | +38.5% → +31.7% |

## Reading this honestly

- **Power saving numbers are tight across both reps** (within ~1-2 percentage points for all
  three workloads) — this is a good sign of a real, repeatable effect, not a lucky single draw.
  Still only 2 reps, not the project's 3-rep minimum, but the agreement so far is encouraging.
- **`sparse_gather`'s near-zero/slightly-negative result reproduced almost exactly** (-2.3% then
  -2.8%) — reinforces the earlier read that this is a real, physically-sensible null result
  (memory-bandwidth-bound workload, no clock lever to pull), not noise.
- **The "slowdown" column is where rep-to-rep disagreement actually shows up**:
  `resnet50`'s negative slowdown (apparent speedup) repeated in both reps (-15.4%, -18.2%) —
  consistent enough that it might not be pure sampling noise, worth a closer look at *why*
  rather than dismissing it outright. `transformer_decode`'s slowdown flipped sign entirely
  (-12.6% → +0.3%) — that one specific number is genuinely noisy and shouldn't be trusted from
  either single rep.
- `glmark2` hung identically in both reps (frozen at workload step 12, `util_pct` ~1%, same
  `ms_per_iter=20.0` the whole 120s) — confirms this is a reliably reproducible `glmark2-drm`
  compatibility bug on this host/driver, not a one-off. Still excluded; needs separate debugging
  (try a different glmark2 scene list, or a different rendering benchmark entirely) before it can
  contribute real data.

## Still short of citable
This project's own standard is 3+ reps minimum before a hardware number is citable in the paper.
Two reps in, the power-saving numbers look solid; the delay/slowdown numbers need at least one
more rep (and ideally a root-cause look at why resnet50's slowdown keeps coming back negative)
before trusting them.
