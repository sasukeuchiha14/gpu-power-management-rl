"""
baselines.py -- the baseline suite (CLAUDE.md sec 8-D: "no baselines beyond
pinned max clocks" was a listed gap; this closes it).

Produces the headline comparison table. Every policy is evaluated under the
same protocol on each held-out GPU: per workload, start from a fixed start row,
run 50 control steps in the table-backed environment, report scaled EDP
(0 = that workload's best profiled operating point, 1 = its worst).

Tunable baselines are tuned LEAVE-ONE-GPU-OUT (on the 7 training GPUs only,
never on the held-out one) so the comparison is honest -- `--loo-tune` does
this for the fixed-clock baseline and reports which fraction each fold picked.

Headline result: a fixed clock at 0.6x max, using no telemetry and no learning,
beats every learned policy tested and captures most of the gap between the
max-clock governor and a per-workload oracle. That finding rests only on EDP
and timing, both of which are measured correctly -- unlike the telemetry
columns (see analysis/telemetry_dilution.py).
"""
import argparse
import numpy as np
import pandas as pd
from _common import REPO  # noqa: F401
from dataset_view import discover_core_only
from features import FEATURES
import eval_policies as P
from sklearn.ensemble import RandomForestClassifier

GRID = np.round(np.arange(0.20, 1.001, 0.05), 2)


def supervised_clf(DS, holdout, seed=0):
    """'is the current clock below the optimum?' classifier, trained on the
    other GPUs only. Used as a reference ceiling for the feature set."""
    X, y = [], []
    for g, df in DS.items():
        if g == holdout:
            continue
        for wl, grp in df.groupby("Workload_Name"):
            copt = float(grp.loc[grp["EDP"].idxmin(), "core_frac"])
            for _, r in grp.iterrows():
                X.append([float(r[f]) for f in FEATURES])
                y.append(int(r["core_frac"] < copt))
    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(
        np.array(X), np.array(y))


def tune_fixed_loo(PREP, ALL, holdout):
    """Best single fixed core_frac chosen using ONLY the training GPUs."""
    best, bf = float("inf"), None
    for f in GRID:
        v = [P.evaluate_policy(PREP[g], P.p_fixed(f))[1] for g in ALL if g != holdout]
        m = float(np.mean(v))
        if m < best:
            best, bf = m, float(f)
    return bf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo-tune", action="store_true",
                    help="also tune the fixed-clock baseline leave-one-GPU-out (slower, honest)")
    ap.add_argument("--no-supervised", action="store_true", help="skip the supervised ceiling (faster)")
    a = ap.parse_args()

    DS = discover_core_only()
    PREP = {g: P.prepare(d) for g, d in DS.items()}
    ALL = sorted(DS)
    rng = np.random.default_rng(0)

    policies = {
        "oracle (per-workload best)": P.p_oracle,
        "governor (pinned max clock)": P.p_governor,
        "min clock": P.p_minclock,
        "random walk": P.p_random(rng),
        "fixed 0.4": P.p_fixed(0.4),
        "fixed 0.5": P.p_fixed(0.5),
        "fixed 0.6": P.p_fixed(0.6),
        "fixed 0.7": P.p_fixed(0.7),
        "utilisation heuristic": P.p_util_heuristic(),
    }

    res = {g: {} for g in ALL}
    for g in ALL:
        for name, pol in policies.items():
            res[g][name] = P.evaluate_policy(PREP[g], pol)[1]
        if not a.no_supervised:
            res[g]["supervised RF (ceiling)"] = P.evaluate_policy(
                PREP[g], P.p_supervised(supervised_clf(DS, g)))[1]
        if a.loo_tune:
            f = tune_fixed_loo(PREP, ALL, g)
            res[g][f"fixed, tuned LOO (picked {f:.2f})"] = P.evaluate_policy(PREP[g], P.p_fixed(f))[1]

    names = list(next(iter(res.values())).keys())
    short = lambda g: g.replace("nvidia_geforce_rtx_", "").replace("rtx3050ti_laptop", "lap")
    hdr = f"{'policy':<32}" + "".join(f"{short(g):>8}" for g in ALL) + f"{'MEAN':>9}"
    print("\nSteady-state scaled EDP  (0 = that workload's best point, 1 = worst). LOWER IS BETTER.\n")
    print(hdr); print("-" * len(hdr))
    rows = []
    for n in names:
        v = [res[g][n] for g in ALL]
        rows.append(dict(policy=n, **{g: res[g][n] for g in ALL}, mean=float(np.mean(v))))
        print(f"{n:<32}" + "".join(f"{x:>8.3f}" for x in v) + f"{np.mean(v):>9.3f}")

    out = f"{REPO}/docs/results/baselines.csv"
    import os; os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
