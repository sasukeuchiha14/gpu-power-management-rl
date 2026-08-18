#!/usr/bin/env python3
"""
measure_edp.py  -- measure real energy/EDP of a workload under ONE clock policy.

Run it TWICE to build an A/B comparison. No sudo needed on the outer command --
the clock-setting calls invoke `sudo nvidia-smi` themselves (this machine's
NOPASSWD sudoers rule is scoped to exactly /usr/bin/nvidia-smi):

  # 1) Governor baseline (race-to-idle = pinned max clocks)
  python3 measure_edp.py governor

  # 2) AI daemon: start sim_to_real_deploy.py with DRY_RUN=False in ANOTHER
  #    terminal FIRST, then run this. It does NOT touch clocks itself; it just
  #    generates the workload's load (which the daemon reacts to) and measures power.
  python3 measure_edp.py ai

IMPORTANT for the 'ai' run: do NOT also run a separate workload-generating loop
yourself. Let THIS script be the only workload generator so both runs do
identical work.

Each run:
  * samples NVML power/clocks at 10 Hz in the background,
  * runs the target workload N times (default 20),
  * writes power_<label>.csv and prints a summary (mean power, energy, EDP).

Then:  python3 plot_edp.py   -> overlays the two runs into a comparison graph.

18 Aug 2026: switched the default target from `nw` to `pathfinder` (dvfs-triage
next-steps item -- the original nw headline result was already valid, but the
*graph* that shipped alongside it used `hotspot`, which never actually ran on
the GPU; this redoes it on a workload confirmed to be real). `nw` is still a
valid choice -- pass it as WORKLOAD_NAME if you want to reproduce that run.
"""
import sys
import time
import csv
import threading
import subprocess
import pynvml
import os

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
N_ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
WORKLOAD_NAME = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("WORKLOAD_NAME", "pathfinder")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCH_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "benchmarks")

# Low-optimum workloads (real, verified running -- see dvfs-triage / CLAUDE.md
# section 7). Edit/extend this table to target a different one via argv[3].
_WORKLOAD_TABLE = {
    "nw": [os.path.join(_BENCH_DIR, "nw", "needle"), "16384", "10"],
    "pathfinder": [os.path.join(_BENCH_DIR, "pathfinder", "pathfinder"), "100000", "100", "20"],
    "backprop": [os.path.join(_BENCH_DIR, "backprop", "backprop"), "65536"],
    "streamcluster": [os.path.join(_BENCH_DIR, "streamcluster", "sc_gpu"),
                       "10", "20", "256", "65536", "65536", "1000", "none", "output.txt", "1"],
}
if WORKLOAD_NAME not in _WORKLOAD_TABLE:
    raise SystemExit(f"Unknown WORKLOAD_NAME '{WORKLOAD_NAME}'. Choices: {list(_WORKLOAD_TABLE)}")
WORKLOAD = _WORKLOAD_TABLE[WORKLOAD_NAME]

# Governor baseline = static max clocks (race-to-idle). Pinned only when label == governor.
MAX_CORE, MAX_MEM = 2950, 9501
GPU_INDEX = 0
SAMPLE_HZ = 10


class PowerSampler(threading.Thread):
    def __init__(self, handle):
        super().__init__()
        self.h = handle
        self.running = True
        self.rows = []            # (t_s, power_W, core_MHz, mem_MHz)

    def run(self):
        t0 = time.perf_counter()
        while self.running:
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
                c = pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)
                m = pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_MEM)
                self.rows.append((time.perf_counter() - t0, p, c, m))
            except pynvml.NVMLError:
                pass
            time.sleep(1.0 / SAMPLE_HZ)

    def stop(self):
        self.running = False


def set_max_clocks():
    # "sudo" per-call (not the whole script under sudo) -- this machine's
    # NOPASSWD sudoers rule is scoped to exactly /usr/bin/nvidia-smi.
    subprocess.run(["sudo", "nvidia-smi", "-lgc", f"{MAX_CORE},{MAX_CORE}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["sudo", "nvidia-smi", "-lmc", f"{MAX_MEM},{MAX_MEM}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def reset_clocks():
    subprocess.run(["sudo", "nvidia-smi", "-rgc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["sudo", "nvidia-smi", "-rmc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)

    if LABEL == "governor":
        print("Governor baseline: pinning clocks to max (race-to-idle).")
        set_max_clocks()
        time.sleep(1.0)
    else:
        print(f"'{LABEL}' run: NOT touching clocks (the AI daemon should control them).")

    sampler = PowerSampler(h)
    sampler.start()

    print(f"Running {' '.join(WORKLOAD)}  x{N_ITERS} ...")
    wall0 = time.perf_counter()
    for _ in range(N_ITERS):
        subprocess.run(WORKLOAD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wall = time.perf_counter() - wall0

    sampler.stop()
    sampler.join()

    if LABEL == "governor":
        reset_clocks()

    rows = sampler.rows
    if len(rows) < 2:
        print("Not enough samples collected!")
        pynvml.nvmlShutdown()
        return

    # Trapezoidal energy integral over the sampled window (Joules)
    energy = 0.0
    for (t1, p1, *_), (t2, p2, *_) in zip(rows, rows[1:]):
        energy += 0.5 * (p1 + p2) * (t2 - t1)

    powers = [r[1] for r in rows]
    mean_p = sum(powers) / len(powers)
    t_per_iter = wall / N_ITERS
    edp_iter = mean_p * t_per_iter * t_per_iter      # power * time^2 (project's EDP)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(gpu_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    # nw keeps the original bare filename (power_governor.csv / power_ai.csv)
    # so the already-validated nw result and its plot aren't clobbered by a
    # run against a different workload.
    suffix = "" if WORKLOAD_NAME == "nw" else f"_{WORKLOAD_NAME}"
    out_csv = os.path.join(data_dir, f"power_{LABEL}{suffix}.csv")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "power_W", "core_MHz", "mem_MHz"])
        w.writerows(rows)

    print("\n================  RESULT  ================")
    print(f" label            : {LABEL}")
    print(f" iterations       : {N_ITERS}")
    print(f" wall time        : {wall:.2f} s   ({t_per_iter:.3f} s/iter)")
    print(f" mean power       : {mean_p:.2f} W")
    print(f" total energy     : {energy:.1f} J")
    print(f" EDP (P*t_iter^2) : {edp_iter:.4f}")
    print(f" wrote {out_csv}")
    print("=========================================")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
