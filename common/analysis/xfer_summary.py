"""
xfer_summary.py -- per-workload summary of a gpu_xfer_trials.sh results dir (any GPU model).
Power saving / slowdown / EDP gain are against the 'base' (observe-only, default driver) runs of the
same workload. The phased workload's two phases are separated by clustering ms/iter in the base run
(geometric mean of its 5th and 95th percentiles).

    python common/analysis/xfer_summary.py docs/results/xfer3090
"""
import glob, re, sys
import numpy as np, pandas as pd


def parse(f):
    P, T, tg, lk, pr, rel = [], [], [], [], [], 0
    for l in open(f):
        rel += "releasing" in l
        m = re.search(r'target=\s*(\d+)MHz.*?(\d+\.\d+)W.*?ms/iter=\s*([\d.]+)', l)
        if not m: continue
        tg.append(float(m.group(1))); P.append(float(m.group(2))); T.append(float(m.group(3)))
        lk.append("LOCKED" in l); pr.append("PROBING" in l)
    return np.array(tg), np.array(P), np.array(T), np.array(lk), np.array(pr), rel


def main(D):
    runs = {}
    for f in sorted(glob.glob(f"{D}/ai_*_rep*.log")):
        w, c, rep = re.match(r".*ai_([a-z]+(?:_long)?)_(.+)_rep(\d)\.log", f).groups()
        runs[(w, c, rep)] = parse(f)
    rows = []
    for w in sorted({k[0] for k in runs}):
        base = [v for k, v in runs.items() if k[0] == w and k[1] == "base"]
        if not base: continue
        bP = np.concatenate([b[1] for b in base]); bT = np.concatenate([b[2] for b in base])
        thr = np.sqrt(np.percentile(bT, 5) * np.percentile(bT, 95)) if w == "phased" else None
        def split(T):
            return [("all", np.ones(len(T), bool))] if thr is None else [("hi", T > thr), ("lo", T <= thr)]
        bref = {ph: (bP[m].mean(), bT[m].mean()) for ph, m in split(bT)}
        for (ww, c, rep), (tg, P, T, lk, pr, rel) in runs.items():
            if ww != w: continue
            for ph, m in split(T):
                if not m.any() or ph not in bref: continue
                p0, t0 = bref[ph]; p, t = P[m].mean(), T[m].mean()
                rows.append(dict(workload=w, phase=ph, policy=c, rep=rep, n=len(P), locked=lk.mean(),
                                 probing=pr.mean(), changes=int((np.diff(tg) != 0).sum()) / len(P),
                                 power_save=1 - p / p0, slow=t / t0 - 1, edp_gain=1 - (p * t * t) / (p0 * t0 * t0)))
    d = pd.DataFrame(rows); d.to_csv(f"{D}/summary.csv", index=False)
    print(d.groupby(["workload", "phase", "policy"]).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
