"""
diagnose_features.py -- do the 6 state features actually carry transferable
workload signal, or does the controller only know where it sits on the dial?

This is the script that overturned an earlier conclusion. A first probe showed
81-84% leave-one-GPU-out accuracy on "is the current clock below this
workload's EDP optimum?" against a 56% majority baseline, which looked like
proof of transferable signal. The ablation below shows `core_frac` alone --
simply where you currently sit on the clock range -- accounts for nearly all of
it, and the five real telemetry channels add ~3 points. "I am at a low clock so
I am probably below the optimum" requires no knowledge of the workload at all.

The two tasks that DO require characterising the workload (naming it, and
identifying its compute- vs bandwidth-bound regime) are then measured directly,
and both are weak. See docs/rl_instability_investigation.md sec 7 and
common/analysis/telemetry_dilution.py for why.
"""
import numpy as np
from _common import REPO  # noqa: F401
from dataset_view import discover_core_only
from features import FEATURES
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

SEED = 0


def _rows(DS):
    """One row per profiled operating point, tagged with its GPU and the
    labels the three diagnostics need."""
    X, below, wl_lab, gpu = [], [], [], []
    for g, df in sorted(DS.items()):
        for wl, grp in df.groupby("Workload_Name"):
            copt = float(grp.loc[grp["EDP"].idxmin(), "core_frac"])
            for _, r in grp.iterrows():
                X.append([float(r[f]) for f in FEATURES])
                below.append(int(r["core_frac"] < copt))
                wl_lab.append(wl); gpu.append(g)
    return (np.array(X), np.array(below), np.array(wl_lab), np.array(gpu))


def _loo(X, y, G, cols=None, model="rf"):
    accs = []
    for held in sorted(set(G)):
        tr, te = G != held, G == held
        Xtr = X[tr][:, cols] if cols else X[tr]
        Xte = X[te][:, cols] if cols else X[te]
        m = {"rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
             "lr": LogisticRegression(max_iter=2000),
             "dummy": DummyClassifier(strategy="most_frequent")}[model]
        accs.append(m.fit(Xtr, y[tr]).score(Xte, y[te]))
    return float(np.mean(accs)), accs


def main():
    DS = discover_core_only()
    X, below, wl, G = _rows(DS)
    ci = FEATURES.index("core_frac")
    others = [i for i in range(len(FEATURES)) if i != ci]

    print(__doc__)
    print("=" * 78)
    print("TASK 1 -- 'is the current clock below this workload's optimum?'")
    print("         (leave-one-GPU-out; the probe that looked convincing)\n")
    print(f"{'feature set':<34}{'LOO acc':>9}")
    print("-" * 43)
    for name, cols, model in [
        ("majority-class baseline", None, "dummy"),
        ("all 6 features", None, "rf"),
        ("all 6 features (logistic reg)", None, "lr"),
        ("core_frac ONLY", [ci], "rf"),
        ("the other 5 (no core_frac)", others, "rf"),
    ]:
        m, _ = _loo(X, below, G, cols, model)
        print(f"{name:<34}{m:>9.3f}")
    print("\n-> core_frac alone gets most of it. The telemetry adds ~3 points.")
    print("   This is positional information, not workload characterisation.")

    print("\n" + "=" * 78)
    print("TASK 2 -- identify WHICH workload is running (needs real characterisation)\n")
    m, per = _loo(X, wl, G, None, "rf")
    print(f"12-way workload ID, leave-one-GPU-out : {m:.3f}   (chance {1/len(set(wl)):.3f})")
    print(f"  per held-out GPU: {np.round(per, 3)}")

    print("\n" + "=" * 78)
    print("TASK 3 -- identify the compute- vs bandwidth-bound REGIME (binary,")
    print("          strictly easier than naming the workload)\n")
    opt = {(g, w): float(grp.loc[grp["EDP"].idxmin(), "core_frac"])
           for g, df in DS.items() for w, grp in df.groupby("Workload_Name")}
    thr = float(np.median(list(opt.values())))
    regime = np.array([int(opt[(G[i], wl[i])] > thr) for i in range(len(G))])
    m, per = _loo(X, regime, G, None, "rf")
    print(f"binary regime ID, leave-one-GPU-out   : {m:.3f}   (chance ~0.500)")
    print(f"  per held-out GPU: {np.round(per, 3)}")
    print("\n-> Even the binary question is barely better than a coin flip on an")
    print("   unseen GPU. A policy that must pick a regime will pick wrong often")
    print("   enough to give back more than adaptation gains.")


if __name__ == "__main__":
    main()
