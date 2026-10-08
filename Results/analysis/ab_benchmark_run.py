"""
ab_benchmark_run.py -- independent data-collection script for the
Gagan-led verification pass (see context.md / team chat, Oct 2026): runs
ON the rented GPU, samples power/clock/throughput for a given workload
under two conditions (clocks pinned at max vs. the trained controller
live), and writes raw CSVs. Deliberately NOT built from
common/demo_power_trace.sh -- a separate implementation, generalized to
any of the four "live_" workloads, and sampling the workload's own status
file in the same loop as nvidia-smi so throughput (for EDP) is captured at
the same instants as power, not read separately.

Lives under Results/ (not common/) deliberately -- this is Gagan's
independent verification pass, kept out of the model code Govind owns.
Run this script from the repo root so the relative paths below resolve.

This script only COLLECTS. Use ab_benchmark_plot.py locally on the pulled-
back CSVs to compute summary stats and render charts -- keeps "did we
measure it right" and "did we plot it right" independently checkable.

Usage (on the rented VM, repo already cloned):
    python3 Results/analysis/ab_benchmark_run.py \\
        --workload resnet50 --weights common/weights/response_aware_holdout_nvidia_geforce_rtx_4090.pth \\
        --duration 120 --results-dir Results/results/resnet50_batch4 --batch-size 4

    python3 Results/analysis/ab_benchmark_run.py --workload sparse_gather --weights <ckpt> --duration 120 --results-dir Results/results/sparse_gather
    python3 Results/analysis/ab_benchmark_run.py --workload transformer_decode --weights <ckpt> --duration 120 --results-dir Results/results/transformer_decode
    python3 Results/analysis/ab_benchmark_run.py --workload glmark2 --weights <ckpt> --duration 120 --results-dir Results/results/glmark2

Then pull the --results-dir folder back and run ab_benchmark_plot.py on it.
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time

WORKLOAD_SCRIPTS = {
    "resnet50": "common/benchmarks/live_resnet50_train.py",
    "glmark2": "common/benchmarks/live_glmark2_render.py",
    "sparse_gather": "Results/benchmarks/live_sparse_gather.py",
    "transformer_decode": "Results/benchmarks/live_transformer_decode.py",
}


def run_cmd(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def query_nvidia_smi(gpu_index):
    out = run_cmd(
        f"nvidia-smi -i {gpu_index} "
        f"--query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu "
        f"--format=csv,noheader,nounits"
    ).stdout.strip()
    power, clock, temp, util = [float(x) for x in out.split(",")]
    return power, clock, temp, util


def read_status(status_file):
    try:
        with open(status_file) as f:
            d = json.load(f)
        return d.get("ms_per_iter"), d.get("iters_per_sec"), d.get("step")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None, None, None


def get_max_core_clock(gpu_index):
    out = run_cmd(
        f"nvidia-smi -i {gpu_index} --query-gpu=clocks.max.sm --format=csv,noheader,nounits"
    ).stdout.strip()
    return out


def sample_loop(outfile, gpu_index, status_file, duration, sample_interval):
    n = int(duration / sample_interval)
    t0 = time.perf_counter()
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_s", "power_w", "clock_mhz", "temp_c", "util_pct",
                    "ms_per_iter", "iters_per_sec", "workload_step"])
        for _ in range(n):
            tick_start = time.perf_counter()
            power, clock, temp, util = query_nvidia_smi(gpu_index)
            ms_per_iter, iters_per_sec, step = read_status(status_file)
            elapsed = time.perf_counter() - t0
            w.writerow([f"{elapsed:.3f}", power, clock, temp, util,
                        ms_per_iter, iters_per_sec, step])
            f.flush()
            sleep_left = sample_interval - (time.perf_counter() - tick_start)
            if sleep_left > 0:
                time.sleep(sleep_left)


def start_workload(workload, status_file, duration, batch_size):
    script = WORKLOAD_SCRIPTS[workload]
    minutes = (duration + 30) / 60.0  # run a bit longer than the sampling window so it doesn't exit mid-sample
    cmd = [sys.executable, script, "--minutes", str(minutes), "--status-file", status_file]
    if workload == "resnet50":
        cmd += ["--batch-size", str(batch_size)]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True, choices=list(WORKLOAD_SCRIPTS))
    ap.add_argument("--weights", required=True, help="DDQN checkpoint path for the controlled phase")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds sampled per phase")
    ap.add_argument("--sample-interval", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=4, help="resnet50 only")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--status-file", default="/tmp/dvfs_ab_status.json")
    a = ap.parse_args()

    os.makedirs(a.results_dir, exist_ok=True)
    if os.path.exists(a.status_file):
        os.remove(a.status_file)

    # --- Phase 1: WITHOUT the controller (clocks pinned at max) ---
    print("=== phase 1: WITHOUT (clocks pinned at max) ===", flush=True)
    wl_proc = start_workload(a.workload, a.status_file, a.duration, a.batch_size)
    time.sleep(8)  # let the workload warm up and start writing its status file
    max_clock = get_max_core_clock(a.gpu_index)
    run_cmd(f"nvidia-smi -i {a.gpu_index} -lgc {max_clock},{max_clock}")
    sample_loop(os.path.join(a.results_dir, "without.csv"), a.gpu_index, a.status_file,
                a.duration, a.sample_interval)
    run_cmd(f"nvidia-smi -i {a.gpu_index} -rgc")
    wl_proc.send_signal(signal.SIGINT)
    try:
        wl_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        wl_proc.kill()
    time.sleep(3)

    # --- Phase 2: WITH the controller (trained DDQN, live) ---
    print("=== phase 2: WITH (trained DDQN, live) ===", flush=True)
    if os.path.exists(a.status_file):
        os.remove(a.status_file)
    wl_proc = start_workload(a.workload, a.status_file, a.duration, a.batch_size)
    time.sleep(8)
    ctrl_cmd = [
        sys.executable, "common/deploy_live.py",
        "--weights", a.weights, "--status-file", a.status_file,
        "--gpu-index", str(a.gpu_index), "--live",
    ]
    ctrl_log = open(os.path.join(a.results_dir, "deploy_live.log"), "w")
    ctrl_proc = subprocess.Popen(ctrl_cmd, stdout=ctrl_log, stderr=subprocess.STDOUT)
    sample_loop(os.path.join(a.results_dir, "with.csv"), a.gpu_index, a.status_file,
                a.duration, a.sample_interval)
    ctrl_proc.send_signal(signal.SIGINT)
    try:
        ctrl_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        ctrl_proc.kill()
    ctrl_log.close()
    run_cmd(f"nvidia-smi -i {a.gpu_index} -rgc")
    wl_proc.send_signal(signal.SIGINT)
    try:
        wl_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        wl_proc.kill()

    with open(os.path.join(a.results_dir, "DONE.flag"), "w") as f:
        f.write(f"DONE {time.ctime()} workload={a.workload} weights={a.weights}\n")
    print(f"=== done, results in {a.results_dir} ===", flush=True)


if __name__ == "__main__":
    main()
