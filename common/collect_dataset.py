#!/usr/bin/env python3
"""
collect_dataset.py -- universal 2D (core x memory) DVFS data collector.

Runs on ANY NVIDIA GPU that passed gpu_probe.py -- run that first. This
sweeps both core AND memory clock together (unlike the laptop-specific
gpus/rtx3050ti_laptop/data_collection/generate_dataset.py, which is core-only
because the laptop hardware genuinely can't do memory control). For each
(workload, core level, memory level), locks clocks, runs the workload a few
times, records power/temp/utilisation, writes one row.

    python3 collect_dataset.py --quick       # fast sanity check, ~5-10 min
    python3 collect_dataset.py               # full sweep, sized for ~1-2 hr

Writes to gpus/<gpu_slug>/data/<gpu_slug>_dataset.csv -- same directory
convention as every other GPU tree, so common/unified_env.py's
discover_datasets() and the existing preprocess.py pattern both find it
without needing a new discovery path.

SAFETY: enables persistence mode once at the start (nvidia-smi -lmc/-pl
silently no-op without it -- see NORTHSTAR.md, found 25 Aug 2026), restores
whatever it was set to afterward. Resets clocks on exit, exception, or
Ctrl+C. A single failed lock skips that one data point and continues --
never aborts the whole run over one bad config.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time

import pynvml

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from gpu_probe import (  # noqa: E402
    hardware_limits, supported_clocks, _smi, _get_persistence_mode, _set_persistence_mode,
    _read_gpu_field, _wait_for_clock,
)

_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))

# =============================================================================
# Workloads. The first 10 are the same PyTorch benchmarks already validated
# on the laptop tree (gpus/rtx3050ti_laptop/benchmarks/) -- reused as-is, not
# duplicated. The last 2 are new (common/benchmarks/), added 25 Aug 2026 to
# fill gaps identified from DVFS literature: nothing in the existing 10 has
# an irregular/sparse memory-access pattern (the class SpMV/graph workloads
# represent), and pt_bert is single-shot-encoder-only, not the sequential
# small-step pattern real autoregressive LLM inference actually has.
#
# category: 0 = compute-bound, 1 = memory-bound, 2 = mixed.
# Note: pt_matmul is filed under compute-bound here, not memory-bound as the
# laptop tree's own generate_dataset.py has it -- DVFS literature treats
# dense GEMM as the canonical *most* core-frequency-sensitive workload, the
# opposite of memory-bound. Not fixed in the laptop tree itself (that's a
# label on an already-collected, working dataset; not worth disturbing).
# =============================================================================
_LAPTOP_BENCH_DIR = os.path.join(_repo_root, "gpus", "rtx3050ti_laptop", "benchmarks")
_COMMON_BENCH_DIR = os.path.join(_script_dir, "benchmarks")

WORKLOADS = [
    {"name": "pt_resnet50",         "script": os.path.join(_LAPTOP_BENCH_DIR, "pt_resnet50.py"),         "category": 0},
    {"name": "pt_bert",             "script": os.path.join(_LAPTOP_BENCH_DIR, "pt_bert.py"),             "category": 0},
    {"name": "pt_mobilenet",        "script": os.path.join(_LAPTOP_BENCH_DIR, "pt_mobilenet.py"),        "category": 0},
    {"name": "pt_matmul",           "script": os.path.join(_LAPTOP_BENCH_DIR, "pt_matmul.py"),           "category": 0},
    {"name": "pt_fft",              "script": os.path.join(_LAPTOP_BENCH_DIR, "pt_fft.py"),              "category": 1},
    {"name": "pt_vector_add",       "script": os.path.join(_LAPTOP_BENCH_DIR, "pt_vector_add.py"),       "category": 1},
    {"name": "pt_elementwise",      "script": os.path.join(_LAPTOP_BENCH_DIR, "pt_elementwise.py"),      "category": 1},
    {"name": "pt_resnet50_train",   "script": os.path.join(_LAPTOP_BENCH_DIR, "pt_resnet50_train.py"),   "category": 2},
    {"name": "pt_crypto_proxy",     "script": os.path.join(_LAPTOP_BENCH_DIR, "pt_crypto_proxy.py"),     "category": 2},
    {"name": "pt_lstm",             "script": os.path.join(_LAPTOP_BENCH_DIR, "pt_lstm.py"),             "category": 2},
    {"name": "pt_sparse_gather",    "script": os.path.join(_COMMON_BENCH_DIR, "pt_sparse_gather.py"),    "category": 1},
    {"name": "pt_transformer_decode", "script": os.path.join(_COMMON_BENCH_DIR, "pt_transformer_decode.py"), "category": 2},
]

# Workloads used by --quick: fast-running, one from each rough category, just
# enough to prove the whole pipeline (lock -> run -> measure -> record) works
# before committing to a real multi-hour sweep.
QUICK_WORKLOADS = {"pt_matmul", "pt_vector_add"}


class GPUProfiler(threading.Thread):
    """20 Hz background telemetry sampler -- same approach as every other
    collector in this repo."""

    def __init__(self, handle):
        super().__init__()
        self.h = handle
        self.running = True
        self.samples = {"pwr": [], "temp": [], "gpu_util": [], "mem_util": [], "vram_used": []}

    def run(self):
        while self.running:
            try:
                self.samples["pwr"].append(pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0)
                self.samples["temp"].append(
                    pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU))
                util = pynvml.nvmlDeviceGetUtilizationRates(self.h)
                self.samples["gpu_util"].append(util.gpu)
                self.samples["mem_util"].append(util.memory)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.h)
                self.samples["vram_used"].append(mem_info.used / (1024.0 * 1024.0))
            except pynvml.NVMLError:
                pass
            time.sleep(0.05)

    def stop(self):
        self.running = False

    def get_stats(self):
        if not self.samples["pwr"]:
            return 0.0, 0, 0.0, 0.0, 0.0
        s = self.samples
        avg = lambda k: sum(s[k]) / len(s[k])
        return avg("pwr"), max(s["temp"]), avg("gpu_util"), avg("mem_util"), avg("vram_used")


def _slugify(name):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return s or "gpu"


def _usable_mem_levels(mems, max_mem_mhz):
    """Drop video-decode/idle memory P-states (e.g. 810/405 MHz) -- keep only
    levels that are real 3D-capable operating points. Both cards probed so
    far show a clean gap between the two groups, but the gap sits in
    different places per card: RTX 3060's states are [100%, 97%, 67%, 11%,
    5%], RTX 4070's are [100%, 98%, 48%, 8%, 4%] -- the compute-capable
    "5001 MHz" state (already confirmed working live on the 4070) is only
    48% of that card's max, which a naive 50% cutoff would wrongly exclude.
    25% sits cleanly in the gap on both cards observed so far (well below
    the lowest real state, 48%, well above the highest idle one, 11%)."""
    return sorted([m for m in mems if m >= 0.25 * max_mem_mhz], reverse=True)


def _build_core_sweep(cores, n_levels):
    if len(cores) <= n_levels:
        return sorted(cores)
    cores = sorted(set(cores))
    lo, hi = cores[0], cores[-1]
    targets = [lo + (hi - lo) * i / (n_levels - 1) for i in range(n_levels)]
    return sorted(set(min(cores, key=lambda c: abs(c - t)) for t in targets))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="fast sanity check: 2 workloads, 3 core levels, 1 measured run")
    ap.add_argument("--core-levels", type=int, default=8)
    ap.add_argument("--runs-per-point", type=int, default=4, help="1 warmup + (N-1) measured")
    ap.add_argument("--workloads", default=None,
                    help="comma-separated workload names (default: all, or the quick set with --quick)")
    ap.add_argument("--cooldown-temp", type=int, default=70)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.quick:
        args.core_levels = min(args.core_levels, 3)
        args.runs_per_point = min(args.runs_per_point, 2)

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()
    slug = _slugify(name)

    limits = hardware_limits(h)
    mems, cores = supported_clocks(h)
    if not cores:
        sys.exit(f"No supported core clocks enumerated for {name} -- run gpu_probe.py first.")

    mem_levels = _usable_mem_levels(mems, limits["max_mem_mhz"]) or [int(limits["max_mem_mhz"])]
    core_levels = _build_core_sweep(cores, args.core_levels)

    if args.workloads:
        names = set(w.strip() for w in args.workloads.split(","))
    elif args.quick:
        names = QUICK_WORKLOADS
    else:
        names = {w["name"] for w in WORKLOADS}
    workloads = [w for w in WORKLOADS if w["name"] in names]

    unmatched = names - {w["name"] for w in WORKLOADS}
    if unmatched:
        print(f"WARNING: --workloads named {sorted(unmatched)}, which don't match any "
              f"known workload. Check for typos. Known names: {[w['name'] for w in WORKLOADS]}")
    if not workloads:
        sys.exit("No workloads selected -- nothing to do. Check --workloads for typos.")

    out_dir = os.path.join(_repo_root, "gpus", slug, "data")
    os.makedirs(out_dir, exist_ok=True)
    # --quick writes to a clearly separate file so a sanity-check run can
    # never silently mix its rows into the real dataset -- especially since
    # both use the same _build_core_sweep() over the same range, some core
    # levels will likely coincide between the two, which would otherwise
    # produce literal duplicate rows if they shared one file.
    default_name = f"{slug}_quicktest.csv" if args.quick else f"{slug}_dataset.csv"
    output_csv = args.output or os.path.join(out_dir, default_name)

    total_points = len(workloads) * len(core_levels) * len(mem_levels)
    print("=" * 70)
    print(f"  {name}  (slug: {slug})")
    print("=" * 70)
    print(f"  Workloads    : {len(workloads)}  {[w['name'] for w in workloads]}")
    print(f"  Core levels  : {len(core_levels)}  {core_levels}")
    print(f"  Memory levels: {len(mem_levels)}  {mem_levels}  (idle/video-decode states excluded)")
    print(f"  Runs/point   : {args.runs_per_point} (1 warmup + {args.runs_per_point - 1} measured)")
    print(f"  Total points : {total_points}")
    print(f"  Output       : {output_csv}")
    print("=" * 70)

    orig_persistence = _get_persistence_mode()
    _set_persistence_mode(True)  # -lmc/-pl silently no-op without this

    skipped = []
    # os.path.isfile() alone isn't enough: a killed/crashed prior run can
    # leave a 0-byte stub file behind (opened in append mode, never got as
    # far as flushing a header) -- happened for real on 25 Aug 2026 when an
    # early buffered run was killed and restarted. The restart saw the stub
    # file "exists", skipped writing the header, and appended 288 real data
    # rows directly under no header at all. Checking size too catches this.
    file_exists = os.path.isfile(output_csv) and os.path.getsize(output_csv) > 0
    try:
        with open(output_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Workload_Name", "Workload_Type", "Core_F", "Mem_F",
                                 "Avg_Time_S", "Avg_Pwr_W", "Peak_T_C",
                                 "Avg_GPU_Util", "Avg_Mem_Util", "VRAM_Used_MB", "EDP"])

            for bench in workloads:
                if not os.path.exists(bench["script"]):
                    print(f"  [SKIP] {bench['script']} not found.")
                    continue
                print(f"\n>>> {bench['name']}")

                for mf in mem_levels:
                    for cf in core_levels:
                        # Cooldown wait is capped -- on a hot host or with a
                        # threshold that's slightly wrong for this card, an
                        # uncapped wait would hang forever on a paid rental
                        # with no indication anything's wrong.
                        cooldown_attempts = 0
                        while cooldown_attempts < 60:  # up to 2 minutes
                            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                            if temp < args.cooldown_temp:
                                break
                            print(f"    cooling ({temp}C > {args.cooldown_temp}C)...", end="\r")
                            time.sleep(2)
                            cooldown_attempts += 1
                        else:
                            print(f"    [WARN] still above {args.cooldown_temp}C after 2 min "
                                  f"-- proceeding anyway rather than hanging.")

                        core_ok, _ = _smi("-i", "0", "-lgc", f"{cf},{cf}")
                        mem_ok, _ = _smi("-i", "0", "-lmc", f"{mf},{mf}")
                        if not (core_ok and mem_ok):
                            skipped.append((bench["name"], cf, mf))
                            print(f"    [SKIP] core={cf} mem={mf}: lock command rejected")
                            continue

                        # Verify the ACHIEVED clocks actually match what we
                        # asked for. supported_clocks() only enumerates core
                        # clocks valid at the TOP memory state (see its
                        # docstring), so a core level we picked isn't
                        # guaranteed jointly valid with every memory level we
                        # also try it at -- the driver can silently coerce to
                        # a different, jointly-valid combination while still
                        # reporting success. Same lesson as the persistence
                        # mode bug: trust the reading, not the message.
                        #
                        # Uses gpu_probe's polling helper rather than a
                        # single fixed-delay check: empirically on a rented
                        # RTX 3060 (25 Aug 2026), the identical lock command
                        # took under 0.5s to settle on one run and over 1s on
                        # the very next run -- no fixed delay was reliable,
                        # and a false rejection here would wrongly skip a
                        # perfectly good data point.
                        core_tol = max(50.0, cf * 0.1)
                        mem_tol = max(100.0, mf * 0.05)
                        achieved_core, core_matched = _wait_for_clock(
                            "clocks.current.sm", cf, core_tol)
                        achieved_mem, mem_matched = _wait_for_clock(
                            "clocks.current.memory", mf, mem_tol)
                        if not (core_matched and mem_matched):
                            skipped.append((bench["name"], cf, mf))
                            print(f"    [SKIP] core={cf} mem={mf}: achieved "
                                  f"core={achieved_core} mem={achieved_mem} after retrying "
                                  f"-- not a jointly valid combination, would have recorded "
                                  f"the wrong label")
                            continue

                        run_data = {"times": [], "pwrs": [], "gpu_utils": [],
                                    "mem_utils": [], "temps": [], "vram": []}
                        for run_idx in range(args.runs_per_point):
                            monitor = GPUProfiler(h)
                            monitor.start()
                            try:
                                # A hung workload (real risk for the 2 new,
                                # never-yet-run-on-hardware scripts) must not
                                # be able to freeze the whole paid rental
                                # indefinitely.
                                res = subprocess.run([sys.executable, bench["script"]],
                                                     capture_output=True, text=True, timeout=120)
                            except subprocess.TimeoutExpired:
                                monitor.stop()
                                monitor.join()
                                print(f"    [SKIP] {bench['name']} timed out after 120s")
                                continue
                            monitor.stop()
                            monitor.join()

                            if res.returncode != 0:
                                continue
                            dur = None
                            for line in res.stdout.splitlines():
                                if "Duration:" in line:
                                    try:
                                        dur = float(line.split("Duration:")[1].strip())
                                        break
                                    except ValueError:
                                        pass
                            if dur is None or dur <= 0:
                                continue
                            if run_idx > 0:  # skip warmup
                                pwr, tpeak, gu, mu, vram = monitor.get_stats()
                                run_data["times"].append(dur)
                                run_data["pwrs"].append(pwr)
                                run_data["gpu_utils"].append(gu)
                                run_data["mem_utils"].append(mu)
                                run_data["temps"].append(tpeak)
                                run_data["vram"].append(vram)

                        if run_data["times"]:
                            avg_time = sum(run_data["times"]) / len(run_data["times"])
                            avg_pwr = sum(run_data["pwrs"]) / len(run_data["pwrs"])
                            avg_gu = sum(run_data["gpu_utils"]) / len(run_data["gpu_utils"])
                            avg_mu = sum(run_data["mem_utils"]) / len(run_data["mem_utils"])
                            peak_t = max(run_data["temps"])
                            avg_vram = sum(run_data["vram"]) / len(run_data["vram"])
                            edp = (avg_pwr * avg_time) * avg_time
                            writer.writerow([bench["name"], bench["category"], cf, mf,
                                             f"{avg_time:.4f}", f"{avg_pwr:.2f}", peak_t,
                                             f"{avg_gu:.2f}", f"{avg_mu:.2f}",
                                             f"{avg_vram:.2f}", f"{edp:.6f}"])
                            f.flush()
                            print(f"    core={cf:5d} mem={mf:5d} | t={avg_time:.3f}s "
                                  f"pwr={avg_pwr:.1f}W util={avg_gu:.0f}%/{avg_mu:.0f}% "
                                  f"edp={edp:.4f}")
                        else:
                            skipped.append((bench["name"], cf, mf))

        if skipped:
            print(f"\n[SUMMARY] {len(skipped)}/{total_points} points skipped.")
            if len(skipped) == total_points:
                print("[FATAL] every point was skipped -- check clock-lock permission "
                      "(run gpu_probe.py) or that the workload scripts actually run here.")
    finally:
        _smi("-i", "0", "-rgc")
        _smi("-i", "0", "-rmc")
        _set_persistence_mode(orig_persistence.strip().lower() == "enabled")
        pynvml.nvmlShutdown()
        print(f"\nDone. Clocks reset, persistence mode restored to '{orig_persistence}'.")
        print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
