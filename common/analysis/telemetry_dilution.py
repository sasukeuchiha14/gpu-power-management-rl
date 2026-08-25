"""
telemetry_dilution.py -- evidence that the recorded telemetry columns are
dominated by process startup rather than by the workload.

THE BUG (found 25 Aug 2026, fixed the same day in common/collect_dataset.py):
the 20 Hz NVML sampler ran for the whole `subprocess.run(...)` -- Python
interpreter start, `import torch`, CUDA context creation, typically 2-4 s of
idle GPU -- while the actual timed work is often 0.15-0.6 s. So the recorded
Avg_GPU_Util / Avg_Pwr_W are averages dominated by idle time, and they mostly
encode how long a workload ran relative to fixed startup cost rather than how
hard it drove the GPU.

Avg_Time_S is NOT affected: it comes from each benchmark's own "Duration:"
line, which brackets only the timed region. Only the telemetry channels are
diluted -- which is exactly the set of features a controller needs to
characterise an unknown workload.

Datasets collected BEFORE the fix still carry the dilution; this script
quantifies it on whatever is currently committed.
"""
import numpy as np
import pandas as pd
from _common import REPO  # noqa: F401
from dataset_view import discover_core_only


def main():
    DS = discover_core_only()
    d = pd.concat([df.assign(gpu=g) for g, df in DS.items()], ignore_index=True)
    d["util_pct"] = d["sm_util"] * 100.0

    print(__doc__)
    print("=" * 78)
    print("Per-workload: real GPU work vs the utilisation that got recorded\n")
    w = (d.groupby("Workload_Name")
           .agg(gpu_only_time_s=("Avg_Time_S", "mean"),
                recorded_util_pct=("util_pct", "mean"))
           .sort_values("gpu_only_time_s"))
    print(w.round(2).to_string())

    c = float(np.corrcoef(d["Avg_Time_S"], d["util_pct"])[0, 1])
    short = d[d["Avg_Time_S"] < 0.5]
    long_ = d[d["Avg_Time_S"] > 2.0]
    print("\n" + "=" * 78)
    print(f"corr(GPU-only duration, recorded utilisation) = {c:+.3f}")
    print("  A strong positive correlation means recorded utilisation mainly reflects")
    print("  how long the workload ran -- i.e. how much it outweighed fixed startup --")
    print("  not how hard it drove the GPU. That is the signature of startup dilution.")
    print(f"\npoints with < 0.5s of real GPU work : {len(short):>4}/{len(d)} "
          f"({100*len(short)/len(d):.0f}%), mean recorded util {short['util_pct'].mean():.1f}%")
    print(f"points with > 2.0s of real GPU work : {len(long_):>4}/{len(d)}"
          f",        mean recorded util {long_['util_pct'].mean():.1f}%")
    print(f"\nhighest utilisation recorded anywhere in the fleet: {d['util_pct'].max():.1f}%")
    mm = d[d["Workload_Name"] == "pt_matmul"]["util_pct"].mean()
    print(f"pt_matmul (dense GEMM, canonically compute-saturating): {mm:.1f}%")
    print("\nA saturating GEMM kernel reporting well under 100% utilisation, and no")
    print("measurement anywhere in an 8-GPU fleet ever showing a busy GPU, is not")
    print("physically credible -- it is the averaging window, not the hardware.")


if __name__ == "__main__":
    main()
