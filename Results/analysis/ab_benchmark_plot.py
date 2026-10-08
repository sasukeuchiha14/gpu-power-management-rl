"""
ab_benchmark_plot.py -- runs LOCALLY on CSVs pulled back from
ab_benchmark_run.py. Computes summary stats (power saving, energy saving,
EDP change, measured slowdown) and renders matplotlib charts -- a from-
scratch, independent plot, not a reuse of docs/demo_power_trace.html's
JS chart or common/demo_power_trace.sh's data.

Energy is integrated via the trapezoidal rule over the sampled power
trace (Wh). EDP uses mean ms_per_iter as the "delay" term, read from the
workload's own status-file throughput reports captured during sampling --
the same measured-not-assumed philosophy METHODOLOGY.md documents for
every other EDP number in this project.

Lives under Results/ (not common/) deliberately -- Gagan's independent
verification pass, kept separate from the model code Govind owns.

Usage:
    python Results/analysis/ab_benchmark_plot.py --results-dir Results/results/resnet50_batch4 --workload resnet50
    python Results/analysis/ab_benchmark_plot.py --results-dir Results/results --all --out-summary Results/results/summary.csv
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def trapezoidal_energy_wh(rows):
    """Integrate power (W) over elapsed time (s) -> watt-seconds -> Wh."""
    pts = [(to_float(r["elapsed_s"]), to_float(r["power_w"])) for r in rows]
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    pts.sort(key=lambda p: p[0])
    ws = 0.0
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        ws += 0.5 * (p0 + p1) * (t1 - t0)
    return ws / 3600.0


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize_phase(rows):
    powers = [to_float(r["power_w"]) for r in rows]
    delays = [to_float(r["ms_per_iter"]) for r in rows]
    mean_power = mean(powers)
    mean_delay = mean(delays)
    energy_wh = trapezoidal_energy_wh(rows)
    edp = (mean_power * mean_delay) if (mean_power is not None and mean_delay is not None) else None
    return {
        "mean_power_w": mean_power,
        "mean_delay_ms": mean_delay,
        "energy_wh": energy_wh,
        "edp": edp,
    }


def plot_workload(results_dir, workload, out_png):
    without_rows = load_csv(os.path.join(results_dir, "without.csv"))
    with_rows = load_csv(os.path.join(results_dir, "with.csv"))

    fig, ax = plt.subplots(figsize=(9, 5))
    xw = [to_float(r["elapsed_s"]) for r in without_rows]
    yw = [to_float(r["power_w"]) for r in without_rows]
    xc = [to_float(r["elapsed_s"]) for r in with_rows]
    yc = [to_float(r["power_w"]) for r in with_rows]
    ax.plot(xw, yw, label="WITHOUT (pinned max clock)", color="#d94f4f", marker="o", markersize=3)
    ax.plot(xc, yc, label="WITH (DDQN controller, live)", color="#3fae5c", marker="o", markersize=3)
    ax.set_xlabel("elapsed time (s)")
    ax.set_ylabel("power draw (W)")
    ax.set_title(f"{workload} -- power draw, pinned max vs. trained controller")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    s_without = summarize_phase(without_rows)
    s_with = summarize_phase(with_rows)
    return s_without, s_with


def pct_change(before, after):
    if before in (None, 0) or after is None:
        return None
    return (before - after) / before * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                     help="a single workload's results dir, or the parent of several if --all")
    ap.add_argument("--workload", default=None, help="label for a single-workload run")
    ap.add_argument("--all", action="store_true",
                     help="treat --results-dir as a parent containing one subdir per workload")
    ap.add_argument("--out-summary", default=None)
    a = ap.parse_args()

    summary_rows = []

    if a.all:
        for name in sorted(os.listdir(a.results_dir)):
            sub = os.path.join(a.results_dir, name)
            if not os.path.isdir(sub) or not os.path.exists(os.path.join(sub, "without.csv")):
                continue
            out_png = os.path.join(sub, f"{name}_power_trace.png")
            s_without, s_with = plot_workload(sub, name, out_png)
            summary_rows.append((name, s_without, s_with))
    else:
        workload = a.workload or os.path.basename(a.results_dir.rstrip("/\\"))
        out_png = os.path.join(a.results_dir, f"{workload}_power_trace.png")
        s_without, s_with = plot_workload(a.results_dir, workload, out_png)
        summary_rows.append((workload, s_without, s_with))

    print(f"{'workload':<22}{'power_W (w/o -> w/)':<26}{'power save %':<14}"
          f"{'delay_ms (w/o -> w/)':<26}{'slowdown %':<12}{'EDP change %':<14}{'energy save %'}")
    out_lines = [["workload", "power_without_w", "power_with_w", "power_saving_pct",
                  "delay_without_ms", "delay_with_ms", "slowdown_pct",
                  "energy_without_wh", "energy_with_wh", "energy_saving_pct",
                  "edp_without", "edp_with", "edp_change_pct"]]
    for name, wo, wi in summary_rows:
        power_save = pct_change(wo["mean_power_w"], wi["mean_power_w"])
        slowdown = -pct_change(wo["mean_delay_ms"], wi["mean_delay_ms"]) if wo["mean_delay_ms"] else None
        energy_save = pct_change(wo["energy_wh"], wi["energy_wh"])
        edp_change = pct_change(wo["edp"], wi["edp"])

        power_str = "{:.1f} -> {:.1f}".format(wo["mean_power_w"], wi["mean_power_w"])
        power_save_str = "{:+.1f}%".format(power_save) if power_save is not None else "n/a"
        if wo["mean_delay_ms"] and wi["mean_delay_ms"]:
            delay_str = "{:.1f} -> {:.1f}".format(wo["mean_delay_ms"], wi["mean_delay_ms"])
        else:
            delay_str = "n/a"
        slowdown_str = "{:+.1f}%".format(slowdown) if slowdown is not None else "n/a"
        edp_str = "{:+.1f}%".format(edp_change) if edp_change is not None else "n/a"
        energy_str = "{:+.1f}%".format(energy_save) if energy_save is not None else "n/a"

        print(f"{name:<22}{power_str:<26}{power_save_str:<14}{delay_str:<26}"
              f"{slowdown_str:<12}{edp_str:<14}{energy_str}")
        out_lines.append([name, wo["mean_power_w"], wi["mean_power_w"], power_save,
                           wo["mean_delay_ms"], wi["mean_delay_ms"], slowdown,
                           wo["energy_wh"], wi["energy_wh"], energy_save,
                           wo["edp"], wi["edp"], edp_change])

    if a.out_summary:
        with open(a.out_summary, "w", newline="") as f:
            csv.writer(f).writerows(out_lines)
        print(f"\nsummary written to {a.out_summary}")


if __name__ == "__main__":
    main()
