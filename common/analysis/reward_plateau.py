"""
reward_plateau.py -- how wide is the "near-optimal" plateau in the offline
per-workload scaled-EDP reward, across the whole fleet?

Motivated by a Phase 2 hardware finding (context.md, 4 Oct 2026 late night):
on a real rented RTX 4090 running pt_resnet50_train continuously, the trained
response-aware policy repeatedly tried to downclock from 3135 to ~1965 MHz,
triggering the do-no-harm arbiter 62% of the time. The offline profiled table
for that exact (GPU, workload) reads Reward=0.994-1.000 across that ENTIRE
range -- i.e. the training signal said it was free. Real hardware measured a
genuine ~14% per-iteration slowdown across the same range. The policy did
exactly what its reward signal taught it.

This script checks whether that's a one-off quirk of this workload or a
systemic property of the reward design. Answer (4 Oct 2026, 5-GPU clean
fleet): systemic -- mean plateau width 41% of the full clock range, 78.3% of
all (GPU, workload) pairs have a Reward>=0.95 plateau wider than 30% of the
range. This is the same "scaled EDP is flat near the optimum" issue that
originally motivated the response-feature project (NORTHSTAR row 11), now
with a concrete real-hardware consequence rather than just a metric critique.

    python common/analysis/reward_plateau.py
"""
import numpy as np
from _common import REPO  # noqa: F401
from dataset_view import discover_core_only

FIVE_GPU_FLEET = [
    "nvidia_geforce_rtx_3060", "nvidia_geforce_rtx_3070",
    "nvidia_geforce_rtx_3080", "nvidia_geforce_rtx_3090",
    "nvidia_geforce_rtx_4090",
]


def main(reward_threshold=0.95):
    DS = discover_core_only()
    rows = []
    for g in FIVE_GPU_FLEET:
        df = DS[g]
        for wl, grp in df.groupby("Workload_Name"):
            grp = grp.sort_values("core_frac")
            near_optimal = grp[grp.Reward >= reward_threshold]
            span = 0.0 if len(near_optimal) < 2 else (
                near_optimal.core_frac.max() - near_optimal.core_frac.min())
            rows.append((g, wl, span, len(near_optimal), len(grp)))

    spans = np.array([r[2] for r in rows])
    print(f"Across {len(rows)} (GPU, workload) pairs on the 5-GPU clean fleet "
          f"(Reward >= {reward_threshold} plateau width, as a fraction of the full clock range):")
    print(f"  mean:   {spans.mean():.3f}")
    print(f"  median: {np.median(spans):.3f}")
    print(f"  fraction with plateau width > 0.3: {np.mean(spans > 0.3):.3f}")
    print()
    print("widest 10 plateaus:")
    for g, wl, span, n, tot in sorted(rows, key=lambda r: -r[2])[:10]:
        print(f"  {g:<28} {wl:<24} plateau_width={span:.3f}  ({n}/{tot} rows >= {reward_threshold})")


if __name__ == "__main__":
    main()
