# analysis/ — the scripts that produced the reported findings

Every number in `docs/FINDINGS_REPORT.md` and
`docs/rl_instability_investigation.md` comes from one of these. They are
committed (rather than left as throwaway scratch work) because the paper needs
its results to be reproducible by someone who only has this repo.

All of them are read-only with respect to the datasets, need no GPU, and run on
CPU in seconds to minutes.

| script | answers | runtime |
|---|---|---|
| `baselines.py` | How do governor / fixed-clock / oracle / random / heuristic / supervised policies compare? | ~3 min |
| `diagnose_features.py` | Do the 6 state features actually carry transferable workload signal? | ~2 min |
| `telemetry_dilution.py` | Is the recorded telemetry contaminated by process startup? | seconds |
| `sweep_rl.py` | Which RL hyperparameters/architectures actually help? | 20-90 min |

Run them from anywhere:

```bash
python common/analysis/baselines.py
python common/analysis/diagnose_features.py
python common/analysis/telemetry_dilution.py
python common/analysis/sweep_rl.py --help
```
