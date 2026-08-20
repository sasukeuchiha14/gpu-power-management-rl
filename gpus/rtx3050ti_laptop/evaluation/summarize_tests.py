#!/usr/bin/env python3
"""
summarize_tests.py -- consolidate every power_governor[_<w>].csv /
power_ai[_<w>].csv pair in data/ into one results table + one aggregate bar
chart, split into IN_DATASET (trained on) vs ZERO_SHOT (never trained on --
see measure_edp.py's ZERO_SHOT_WORKLOADS). Mirrors
gpus/rtx4500_ada/evaluation/summarize_tests.py.

Run after collecting the individual A/B pairs with measure_edp.py:
    python summarize_tests.py
"""
import csv
import glob
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measure_edp import IN_DATASET_WORKLOADS, ZERO_SHOT_WORKLOADS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(GPU_ROOT, "data")
PLOTS_DIR = os.path.join(GPU_ROOT, "logs_and_plots")


def load_stats(path):
    t, p = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_s"]))
            p.append(float(row["power_W"]))
    mean_p = sum(p) / len(p)
    energy = sum(0.5 * (p1 + p2) * (t2 - t1) for (t1, p1), (t2, p2) in zip(zip(t, p), zip(t[1:], p[1:])))
    wall = t[-1] - t[0]
    return mean_p, energy, wall


def find_workloads():
    names = set()
    for f in glob.glob(os.path.join(DATA_DIR, "power_governor*.csv")):
        base = os.path.basename(f)
        m = re.match(r"power_governor(?:_(.+))?\.csv", base)
        names.add(m.group(1) or "pt_matmul")  # bare filename = the default pt_matmul run
    return sorted(names)


def main():
    rows = []
    for name in find_workloads():
        suffix = "" if name == "pt_matmul" else f"_{name}"
        gov_path = os.path.join(DATA_DIR, f"power_governor{suffix}.csv")
        ai_path = os.path.join(DATA_DIR, f"power_ai{suffix}.csv")
        if not (os.path.isfile(gov_path) and os.path.isfile(ai_path)):
            continue
        gp, ge, gw = load_stats(gov_path)
        ap, ae, aw = load_stats(ai_path)
        e_save = (ge - ae) / ge * 100 if ge else 0
        p_save = (gp - ap) / gp * 100 if gp else 0
        t_delta = (aw - gw) / gw * 100 if gw else 0
        category = ("in-dataset" if name in IN_DATASET_WORKLOADS
                    else "zero-shot" if name in ZERO_SHOT_WORKLOADS
                    else "unclassified")
        rows.append(dict(name=name, category=category, gov_w=gp, gov_j=ge, gov_s=gw,
                          ai_w=ap, ai_j=ae, ai_s=aw, e_save=e_save, p_save=p_save, t_delta=t_delta))

    if not rows:
        print("No governor/ai CSV pairs found in data/. Run measure_edp.py first.")
        return

    # ---- text table ----
    lines = [
        "| Workload | Category | Governor (W / J / s) | AI (W / J / s) | Energy saved | Power saved | Runtime delta |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['category']} | {r['gov_w']:.1f} / {r['gov_j']:.1f} / {r['gov_s']:.2f} "
            f"| {r['ai_w']:.1f} / {r['ai_j']:.1f} / {r['ai_s']:.2f} "
            f"| {r['e_save']:+.1f}% | {r['p_save']:+.1f}% | {r['t_delta']:+.1f}% |"
        )
    table_md = "\n".join(lines)
    print(table_md)
    out_md = os.path.join(PLOTS_DIR, "test_results_summary.md")
    with open(out_md, "w") as f:
        f.write("# Live A/B test results (governor vs AI daemon), RTX 3050 Ti Laptop\n\n")
        f.write("Energy saved = (governor_J - ai_J) / governor_J. Positive = AI used less energy.\n")
        f.write("Runtime delta = (ai_s - governor_s) / governor_s. Negative = AI was also faster.\n\n")
        f.write(table_md + "\n")
    print(f"\nWrote {out_md}")

    # ---- aggregate bar chart ----
    rows.sort(key=lambda r: (r["category"], -r["e_save"]))
    names = [r["name"] for r in rows]
    savings = [r["e_save"] for r in rows]
    colors = ["#2a9d8f" if r["category"] == "in-dataset" else "#e76f51" for r in rows]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, savings, color=colors)
    plt.axhline(0, color="black", lw=0.8)
    plt.ylabel("Energy saved vs governor (%)  -- higher is better")
    plt.title("RTX 3050 Ti: DDQN AI daemon vs max-clock governor, by workload")
    plt.xticks(rotation=25, ha="right")
    ymin, ymax = min(savings + [0]), max(savings + [0])
    span = max(ymax - ymin, 1e-6)
    plt.ylim(ymin - 0.22 * span, ymax + 0.12 * span)
    for b, v in zip(bars, savings):
        offset = 0.02 * span if v >= 0 else -0.02 * span
        plt.text(b.get_x() + b.get_width() / 2, v + offset, f"{v:+.1f}%",
                  ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color="#2a9d8f", label="in-dataset (trained on)"),
                         Patch(color="#e76f51", label="zero-shot (never trained on)")])
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "test_results_summary.png")
    plt.savefig(out_png, dpi=200)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
