"""
hillclimb_sweep.py -- sweeps eval_policies.p_hillclimb()'s one free parameter
(elasticity_threshold) and its trip direction (step back up vs. hold) across
every discovered GPU, to find the best this policy family can do and check
it against the 4 Oct 2026 context.md finding (0.0051 scaled EDP, 4.7x better
than fixed-0.6, 5/5 GPUs) from a hill-climb script that was never committed
and no longer exists.

Result (4 Oct 2026): the best point found (threshold=0.05, step up on trip)
gets mean steady-state scaled EDP ~0.13 across all 8 discovered GPUs, which
LOSES to fixed-0.6 (~0.03) -- it does not reproduce the original finding.
"Hold" instead of "step up" on trip is uniformly much worse (~0.52+).

    python common/analysis/hillclimb_sweep.py
"""
import numpy as np
from _common import REPO  # noqa: F401
from dataset_view import discover_core_only
from features import FEATURES, RESPONSE_CLIP
import eval_policies as P

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}
GPUS = sorted(DS)
CORE_IDX = FEATURES.index("core_frac")
RT_IDX = len(FEATURES)


def make_policy(threshold, on_trip):
    def f(stack, tracked, stride, W):
        current_core = float(stack[-1][CORE_IDX])
        if current_core <= W["core"].min() + 1e-9:
            return current_core + stride
        raw = (float(stack[-1][RT_IDX]) - 0.5) * 2.0 * RESPONSE_CLIP
        if abs(raw) > threshold:
            return current_core + stride if on_trip == "up" else current_core
        return current_core - stride
    return f


def main():
    for on_trip in ["up", "hold"]:
        for thr in [0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
            vals = [P.evaluate_policy(PREP[g], make_policy(thr, on_trip), use_response=True)[1]
                    for g in GPUS]
            print(f"on_trip={on_trip:<4} thr={thr:<5} mean_ss={np.mean(vals):.4f}  "
                  f"per_gpu={[round(v, 3) for v in vals]}")


if __name__ == "__main__":
    main()
