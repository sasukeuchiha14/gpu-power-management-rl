"""
summarize_holdout_results.py -- aggregates common/holdout_results/*.csv (each
produced by evaluate_holdout.py) into one fleet-wide table: the actual
NORTHSTAR.md row 4 result. Uses EDP reduction computed on the MEAN of raw
EDP values per GPU, not an average of per-workload percentages -- see
evaluate_holdout.py's comment on why that average is misleading (a few
workloads have a governor EDP near 0, so a tiny absolute miss reads as a
huge percentage there).

    python3 summarize_holdout_results.py
"""
import glob
import os

import pandas as pd

_common_dir = os.path.dirname(os.path.abspath(__file__))
_results_dir = os.path.join(_common_dir, "holdout_results")


def summarize():
    paths = sorted(glob.glob(os.path.join(_results_dir, "*.csv")))
    if not paths:
        raise SystemExit(f"No results found in {_results_dir}. Run evaluate_holdout.py first.")

    rows = []
    for p in paths:
        gpu = os.path.splitext(os.path.basename(p))[0]
        df = pd.read_csv(p)
        mean_gov = df.governor_edp.mean()
        mean_ai = df.ai_edp.mean()
        reduction_pct = 100.0 * (mean_gov - mean_ai) / mean_gov if mean_gov > 0 else float("nan")
        rows.append({
            "gpu": gpu,
            "n_workloads": len(df),
            "workloads_ai_wins": int(df.ai_beats_governor.sum()),
            "mean_governor_edp": mean_gov,
            "mean_ai_edp": mean_ai,
            "edp_reduction_pct": reduction_pct,
        })

    summary = pd.DataFrame(rows).sort_values("gpu").reset_index(drop=True)
    fleet_wins = int((summary.mean_ai_edp < summary.mean_governor_edp).sum())

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nAI has a lower mean EDP than the governor on {fleet_wins}/{len(summary)} held-out GPUs.")
    print(f"Mean EDP reduction across GPUs (simple average of each GPU's own reduction): "
          f"{summary.edp_reduction_pct.mean():+.1f}%")

    out_path = os.path.join(_results_dir, "summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    return summary


if __name__ == "__main__":
    summarize()
