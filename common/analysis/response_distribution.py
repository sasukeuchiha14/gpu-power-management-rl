"""
response_distribution.py -- measures the actual distribution of the two
response ratios (d_log_runtime/d_log_clock, d_log_power/d_log_clock) across
the 5-GPU clean fleet, over every pair of adjacent profiled core levels
within a workload. This is what RESPONSE_CLIP in common/features.py is
chosen from -- see that file's comment.

    python common/analysis/response_distribution.py
"""
import numpy as np
from _common import REPO  # noqa: F401
from dataset_view import discover_core_only


def main():
    DS = discover_core_only()
    ratios_t, ratios_p = [], []
    for gpu, df in DS.items():
        for wl, g in df.groupby("Workload_Name"):
            g = g.sort_values("core_frac").reset_index(drop=True)
            core = g["core_frac"].to_numpy()
            time = g["Avg_Time_S"].to_numpy()
            power = g["Avg_Pwr_W"].to_numpy()
            for i in range(len(core) - 1):
                dlc = np.log(core[i + 1]) - np.log(core[i])
                if abs(dlc) < 1e-9:
                    continue
                ratios_t.append((np.log(time[i + 1]) - np.log(time[i])) / dlc)
                ratios_p.append((np.log(power[i + 1]) - np.log(power[i])) / dlc)

    for name, r in [("d_log_runtime/d_log_clock", np.array(ratios_t)),
                     ("d_log_power/d_log_clock", np.array(ratios_p))]:
        print(f"{name}  n={len(r)}  min={r.min():.3f}  max={r.max():.3f}")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"  p{p:<3d} {np.percentile(r, p):.3f}")


if __name__ == "__main__":
    main()
