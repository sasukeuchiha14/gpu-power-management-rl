"""
repeat_trials_summary.py -- parses the raw repeated-hardware-trial logs
(docs/results/phase2_hardware/repeat_trials/) into the summary CSV and
prints the same mean+-std comparison reported in context.md / NORTHSTAR.md
row 11 (5 Oct 2026, early morning entry).

Context: a single-trial hardware re-validation of the reward-plateau fix
(context.md's prior entry) looked like a clean win. Repeating it 3x per
condition revealed it's a real trade-off, not an unqualified improvement --
see those docs for the full story. This script is how those numbers were
produced; it does not re-run anything on hardware, just parses already-
collected logs.

    python common/analysis/repeat_trials_summary.py
"""
import re
import numpy as np
import pandas as pd
from scipy import stats

from _common import REPO  # noqa: F401

D = f"{REPO}/docs/results/phase2_hardware/repeat_trials"
WARMUP_TICKS = 3  # dropped as transient before computing steady-state means


def parse_ai_log(path):
    powers, ms_iters, actions, arb = [], [], [], []
    for line in open(path):
        m = re.search(r'a=(\w+).*?(\d+\.\d+)W.*?ms/iter=\s*([\d.]+)', line)
        if m:
            actions.append(m.group(1))
            powers.append(float(m.group(2)))
            ms_iters.append(float(m.group(3)))
            arb.append('[ARBITER]' in line)
    return np.array(powers), np.array(ms_iters), np.array(arb), actions


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
    for batch in [32, 4]:
        base_pre = parse_baseline(f"{D}/baseline_b{batch}_pre.csv")
        base_post = parse_baseline(f"{D}/baseline_b{batch}_post.csv")
        base_power = np.concatenate([base_pre, base_post])
        for variant in ["old", "new"]:
            for rep in [1, 2, 3]:
                p, ms, arb, actions = parse_ai_log(f"{D}/ai_b{batch}_{variant}_rep{rep}.log")
                if len(p) == 0:
                    print(f"WARNING: no ticks parsed for b{batch}_{variant}_rep{rep}")
                    continue
                p_ss, ms_ss = p[WARMUP_TICKS:], ms[WARMUP_TICKS:]
                rows.append(dict(
                    batch=batch, variant=variant, rep=rep, n=len(p_ss),
                    power_mean=p_ss.mean(), ms_mean=ms_ss.mean(), arb_frac=arb.mean(),
                    dec=actions.count("DEC"), inc=actions.count("INC"), hold=actions.count("HOLD"),
                    base_power_mean=base_power.mean(),
                ))

    df = pd.DataFrame(rows)
    df["power_saving_pct"] = 100 * (df.base_power_mean - df.power_mean) / df.base_power_mean
    out = f"{REPO}/docs/results/phase2_hardware/repeat_trials_summary.csv"
    df.to_csv(out, index=False)
    print(df[["batch", "variant", "rep", "n", "power_mean", "ms_mean", "arb_frac",
              "dec", "inc", "hold", "power_saving_pct"]].round(3).to_string(index=False))

    print("\n=== mean +- std across 3 reps, old vs new ===\n")
    for batch in [32, 4]:
        base = df[df.batch == batch].base_power_mean.iloc[0]
        print(f"--- batch={batch} (baseline power = {base:.1f}W) ---")
        for variant in ["old", "new"]:
            sub = df[(df.batch == batch) & (df.variant == variant)]
            print(f"  {variant}: power={sub.power_mean.mean():.1f}+-{sub.power_mean.std():.1f}W  "
                  f"ms/iter={sub.ms_mean.mean():.2f}+-{sub.ms_mean.std():.2f}  "
                  f"power_saving={sub.power_saving_pct.mean():+.1f}%+-{sub.power_saving_pct.std():.1f}  "
                  f"arb={sub.arb_frac.mean():.2f}  INC_mean={sub.inc.mean():.1f}  "
                  f"DEC_mean={sub.dec.mean():.1f}  HOLD_mean={sub.hold.mean():.1f}")
        old = df[(df.batch == batch) & (df.variant == "old")]
        new = df[(df.batch == batch) & (df.variant == "new")]
        t, p = stats.ttest_ind(old.power_saving_pct, new.power_saving_pct)
        print(f"  power_saving t-test (old vs new): t={t:.2f} p={p:.3f}")
        print()
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
