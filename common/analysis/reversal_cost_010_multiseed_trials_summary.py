"""
reversal_cost_010_multiseed_trials_summary.py -- parses the raw hardware
logs from common/reversal_cost_010_multiseed_trials.sh. Tests whether
reversal_cost=0.10's seed=0 failure to suppress cycling on batch=32 was a
seed-specific artifact: compares old vs reversal_cost=0.10 seed=1 and
seed=2 (context.md, NORTHSTAR row 11, 7 Oct 2026).

    python common/analysis/reversal_cost_010_multiseed_trials_summary.py
"""
import os
import re

import numpy as np
import pandas as pd

from _common import REPO  # noqa: F401

D = f"{REPO}/docs/results/reversal_cost_phase2_hardware/reversal_cost_010_multiseed_trials"
WARMUP_TICKS = 3
VARIANTS = ["old", "r010s1", "r010s2"]


def parse_ai_log(path):
    targets, powers, ms_iters, actions = [], [], [], []
    for line in open(path):
        m = re.search(r'a=(\w+).*?target=\s*(\d+)MHz.*?(\d+\.\d+)W.*?ms/iter=\s*([\d.]+)', line)
        if m:
            actions.append(m.group(1))
            targets.append(float(m.group(2)))
            powers.append(float(m.group(3)))
            ms_iters.append(float(m.group(4)))
    return np.array(targets), np.array(powers), np.array(ms_iters), actions


def flip_rate(targets):
    deltas = np.diff(targets)
    deltas = deltas[deltas != 0]
    if len(deltas) < 2:
        return 0.0
    signs = np.sign(deltas)
    reversals = (signs[1:] != signs[:-1]).sum()
    return float(reversals / (len(signs) - 1))


def parse_baseline(path):
    powers = []
    for line in open(path):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                powers.append(float(parts[1].replace(' W', '')))
            except ValueError:
                pass
    return np.array(powers)


def main():
    rows = []
    for batch in [4, 32]:
        base_pre = parse_baseline(f"{D}/baseline_b{batch}_pre.csv")
        base_post = parse_baseline(f"{D}/baseline_b{batch}_post.csv")
        base_power = np.concatenate([base_pre, base_post])
        for variant in VARIANTS:
            for rep in [1, 2]:
                targets, p, ms, actions = parse_ai_log(f"{D}/ai_b{batch}_{variant}_rep{rep}.log")
                if len(p) == 0:
                    print(f"WARNING: no ticks parsed for b{batch}_{variant}_rep{rep}")
                    continue
                t_ss, p_ss, ms_ss = targets[WARMUP_TICKS:], p[WARMUP_TICKS:], ms[WARMUP_TICKS:]
                rows.append(dict(
                    batch=batch, variant=variant, rep=rep, n=len(p_ss),
                    power_mean=p_ss.mean(), ms_mean=ms_ss.mean(),
                    flip_rate=flip_rate(t_ss),
                    dec=actions.count("DEC"), inc=actions.count("INC"), hold=actions.count("HOLD"),
                    base_power_mean=base_power.mean(),
                ))

    df = pd.DataFrame(rows)
    df["power_saving_pct"] = 100 * (df.base_power_mean - df.power_mean) / df.base_power_mean
    out = f"{REPO}/docs/results/reversal_cost_phase2_hardware/reversal_cost_010_multiseed_trials_summary.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(df[["batch", "variant", "rep", "n", "power_mean", "ms_mean", "flip_rate",
              "dec", "inc", "hold", "power_saving_pct"]].round(3).to_string(index=False))

    print("\n=== mean +- std across reps, old vs 0.10 seed1 vs seed2 ===\n")
    for batch in [4, 32]:
        sub = df[df.batch == batch]
        for variant in VARIANTS:
            v = sub[sub.variant == variant]
            if len(v) == 0:
                continue
            print(f"batch={batch} {variant:<7} power_saving={v.power_saving_pct.mean():.1f}%"
                  f"+-{v.power_saving_pct.std():.1f}  flip_rate={v.flip_rate.mean():.3f}"
                  f"+-{v.flip_rate.std():.3f}  ms={v.ms_mean.mean():.2f}")

    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
