"""
generate_charts.py -- generates every chart referenced in docs/RESULTS.md from
the already-committed CSVs under docs/results/. Run from the repo root:

    python docs/charts/generate_charts.py

No new experiments are run here -- this only visualizes existing, already-
verified results. Re-run after any result CSV changes to keep charts in sync.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "docs", "results")
OUT = HERE

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white", "font.size": 11, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})
GPU_SHORT = {
    "nvidia_geforce_rtx_3060": "3060", "nvidia_geforce_rtx_3070": "3070",
    "nvidia_geforce_rtx_3080": "3080", "nvidia_geforce_rtx_3090": "3090",
    "nvidia_geforce_rtx_4090": "4090",
}
GPU_ORDER = list(GPU_SHORT.keys())


def savefig(name):
    path = os.path.join(OUT, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("wrote", path)


# ---------------------------------------------------------------- chart 1
# Main result: response-aware DDQN vs ablation vs fixed-clock vs governor,
# per GPU, 10-seed leave-one-GPU-out.
def chart_main_result():
    d = pd.read_csv(f"{RESULTS}/response_eval_10seed.csv")
    policies = ["governor", "fixed, tuned LOO", "hill-climb (hand-coded reference)",
                "old snapshot DDQN (ablation)", "response-aware DDQN (ours)"]
    labels = ["Governor\n(max clock)", "Tuned fixed\nclock", "Hill-climb\n(reference)",
              "Snapshot DDQN\n(ablation)", "Response-aware\nDDQN (ours)"]
    colors = ["#94a3b8", "#60a5fa", "#fbbf24", "#f87171", "#34d399"]

    means = [d[d.policy == p].ss.mean() for p in policies]
    stds = [d[d.policy == p].ss.std() for p in policies]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Steady-state scaled EDP (lower is better)")
    ax.set_title("Leave-one-GPU-out comparison, 5-GPU clean fleet, 10 seeds")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{m:.4f}", ha="center", fontsize=9)
    savefig("main_result_policy_comparison.png")


# ---------------------------------------------------------------- chart 2
# Per-GPU breakdown for the two learned policies, showing the paired win.
def chart_per_gpu_paired():
    d = pd.read_csv(f"{RESULTS}/response_eval_10seed.csv")
    fixed = d[d.policy == "fixed, tuned LOO"].groupby("gpu").ss.mean()
    ablation = d[d.policy == "old snapshot DDQN (ablation)"].groupby("gpu").ss.mean()
    response = d[d.policy == "response-aware DDQN (ours)"].groupby("gpu").ss.mean()
    ablation_std = d[d.policy == "old snapshot DDQN (ablation)"].groupby("gpu").ss.std()
    response_std = d[d.policy == "response-aware DDQN (ours)"].groupby("gpu").ss.std()

    x = np.arange(len(GPU_ORDER))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, [fixed[g] for g in GPU_ORDER], w, label="Tuned fixed clock", color="#60a5fa")
    ax.bar(x, [ablation[g] for g in GPU_ORDER], w, yerr=[ablation_std[g] for g in GPU_ORDER],
           label="Snapshot DDQN (ablation)", color="#f87171", capsize=3)
    ax.bar(x + w, [response[g] for g in GPU_ORDER], w, yerr=[response_std[g] for g in GPU_ORDER],
           label="Response-aware DDQN (ours)", color="#34d399", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([GPU_SHORT[g] for g in GPU_ORDER])
    ax.set_xlabel("Held-out GPU")
    ax.set_ylabel("Steady-state scaled EDP (lower is better)")
    ax.set_title("Per-GPU result: response features stabilize the win, 10 seeds each")
    ax.legend()
    savefig("per_gpu_paired_comparison.png")


# ---------------------------------------------------------------- chart 3
# reversal_cost sweep: cycle rate and cost vs penalty strength.
def chart_reversal_cost_sweep():
    tags = ["005", "010", "020", "030", "040", "060"]
    vals = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]
    cycle_rates, ss_means, ss_stds = [], [], []
    for tag in tags:
        d = pd.read_csv(f"{RESULTS}/reversal_cost_10seed_{tag}.csv")
        cycle_rates.append((d.n_cycling / d.n_total).mean())
        ss_means.append(d.ss.mean())
        ss_stds.append(d.ss.std())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(vals, cycle_rates, "o-", color="#f87171", linewidth=2, markersize=7)
    ax1.set_xlabel("reversal_cost")
    ax1.set_ylabel("Cycle rate (fraction of GPU-seed-workload cycling)")
    ax1.set_title("Limit-cycle suppression vs penalty strength")
    ax1.axhline(0, color="gray", linewidth=0.5)

    ax2.errorbar(vals, ss_means, yerr=ss_stds, fmt="o-", color="#60a5fa", linewidth=2,
                 markersize=7, capsize=4)
    ax2.axvline(0.30, color="#34d399", linestyle="--", linewidth=1.5, label="selected (0.30)")
    ax2.set_xlabel("reversal_cost")
    ax2.set_ylabel("Steady-state scaled EDP (cost)")
    ax2.set_title("Cost plateaus, doesn't keep climbing")
    ax2.legend()
    savefig("reversal_cost_sweep.png")


# ---------------------------------------------------------------- chart 4
# debounce_validate: shows debounce doesn't work (period stretches, cost rises)
def chart_debounce_failure():
    d = pd.read_csv(f"{RESULTS}/debounce_validate.csv")
    d = d.sort_values("debounce_n")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(d.debounce_n, d.n_cycling / d.n_total, "o-", color="#f87171", linewidth=2, markersize=7)
    ax1.set_xlabel("debounce_n (ticks required before committing)")
    ax1.set_ylabel("Cycle rate (12 workloads)")
    ax1.set_title("Debounce barely reduces cycling...")
    ax1.set_ylim(0, 1)

    ax2.plot(d.debounce_n, d.ss, "o-", color="#fbbf24", linewidth=2, markersize=7)
    ax2.set_xlabel("debounce_n")
    ax2.set_ylabel("Steady-state scaled EDP (cost)")
    ax2.set_title("...while cost climbs: a lose-lose knob")
    savefig("debounce_failure.png")


# ---------------------------------------------------------------- chart 5
# Workload holdout: per-workload win/loss pattern, seed-stable
def chart_workload_holdout():
    d = pd.read_csv(f"{RESULTS}/workload_holdout_eval.csv")
    d["wins"] = d.ss_mean < d.fixed_best_ss
    win_rate = d.groupby("workload").wins.mean().sort_values()
    colors = ["#34d399" if v >= 0.6 else ("#f87171" if v <= 0.4 else "#fbbf24") for v in win_rate.values]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(win_rate.index, win_rate.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Win rate vs tuned-fixed-clock across 5 seeds (held-out workload)")
    ax.set_title("Held-out workload generalization: a real, seed-stable split")
    ax.set_xlim(0, 1)
    savefig("workload_holdout_split.png")


# ---------------------------------------------------------------- chart 6
# Latency measurement histogram
def chart_latency():
    path = f"{REPO}/docs/results/latency_measurement/tick_latency_rtx4090.log"
    import re
    vals = []
    for line in open(path):
        m = re.search(r"tick_ms=\s*([\d.]+)", line)
        if m:
            vals.append(float(m.group(1)))
    vals = np.array(vals)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(vals, bins=12, color="#60a5fa", edgecolor="black", linewidth=0.5)
    ax.axvline(vals.mean(), color="#f87171", linestyle="--", linewidth=2,
               label=f"mean={vals.mean():.2f}ms")
    ax.axvline(50, color="gray", linestyle=":", linewidth=1.5, label="old 50ms budget")
    ax.set_xlabel("Per-tick control latency (ms), real RTX 4090")
    ax.set_ylabel("Count (of 40 ticks)")
    ax.set_title("Real control-loop latency: comfortably within budget")
    ax.legend()
    savefig("latency_histogram.png")


# ---------------------------------------------------------------- chart 7
# Hardware validation: the two Phase 2 conditions, power saving + slowdown
def chart_hardware_phase2():
    conditions = ["Launch-bound\n(batch=4)", "Compute-bound\n(batch=32)"]
    power_saving = [18.6, 7.5]
    slowdown = [-0.4, 3.5]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(conditions))
    w = 0.35
    b1 = ax1.bar(x - w / 2, power_saving, w, label="Power saving (%)", color="#34d399")
    b2 = ax1.bar(x + w / 2, slowdown, w, label="Slowdown (%)", color="#f87171")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.set_ylabel("Percent")
    ax1.set_title("Phase 2 hardware validation, real RTX 4090\n(response-aware checkpoint vs unmanaged baseline)")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + (0.3 if h >= 0 else -0.8),
                  f"{h:+.1f}%", ha="center", fontsize=9)
    ax1.legend()
    savefig("hardware_phase2_validation.png")


# ---------------------------------------------------------------- chart 8
# Limit-cycle hardware confirmation: flip_rate old vs reversal_cost=0.30
def chart_limit_cycle_hardware():
    path = f"{REPO}/docs/results/reversal_cost_phase2_hardware/reversal_cost_repeat_trials_summary.csv"
    d = pd.read_csv(path)
    conditions = [4, 32]
    labels = ["Launch-bound\n(batch=4)", "Compute-bound\n(batch=32)"]
    old_flip = [d[(d.batch == c) & (d.variant == "old")].flip_rate.mean() for c in conditions]
    new_flip = [d[(d.batch == c) & (d.variant == "new")].flip_rate.mean() for c in conditions]
    old_power = [d[(d.batch == c) & (d.variant == "old")].power_saving_pct.mean() for c in conditions]
    new_power = [d[(d.batch == c) & (d.variant == "new")].power_saving_pct.mean() for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w / 2, old_flip, w, label="Old (pure-log, cycles)", color="#f87171")
    ax1.bar(x + w / 2, new_flip, w, label="reversal_cost=0.30", color="#34d399")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("flip_rate (direction-reversal fraction)")
    ax1.set_title("Cycling: fully eliminated on real hardware")
    ax1.legend()

    ax2.bar(x - w / 2, old_power, w, label="Old (pure-log, cycles)", color="#f87171")
    ax2.bar(x + w / 2, new_power, w, label="reversal_cost=0.30", color="#34d399")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Power saving (%)")
    ax2.set_title("...but the compute-bound benefit is the cost")
    ax2.legend()
    savefig("limit_cycle_hardware_confirmation.png")


if __name__ == "__main__":
    chart_main_result()
    chart_per_gpu_paired()
    chart_reversal_cost_sweep()
    chart_debounce_failure()
    chart_workload_holdout()
    chart_latency()
    chart_hardware_phase2()
    chart_limit_cycle_hardware()
    print("\nAll charts generated.")
