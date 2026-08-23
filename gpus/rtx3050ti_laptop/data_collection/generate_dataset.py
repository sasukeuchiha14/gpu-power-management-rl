import time
import csv
import re
import subprocess
import threading
import pynvml
import os
import sys

# Reuse the clock-enumeration logic from common/gpu_probe.py instead of
# duplicating it -- see NORTHSTAR.md: common/ is the single source of truth
# for anything that needs to work identically across cards.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_common_dir = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", "common"))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)
try:
    from gpu_probe import supported_clocks as _supported_clocks
except ImportError:
    _supported_clocks = None

# =================================================================
# 1. THE WORKLOADS (Pure PyTorch equivalents running as separate processes)
# =================================================================
BENCHMARKS = [
    # --- Category 0: Compute-Bound ---
    {"name": "pt_resnet50", "script": "benchmarks/pt_resnet50.py", "category": 0},
    {"name": "pt_bert", "script": "benchmarks/pt_bert.py", "category": 0},
    {"name": "pt_mobilenet", "script": "benchmarks/pt_mobilenet.py", "category": 0},

    # --- Category 1: Memory-Bound ---
    {"name": "pt_matmul", "script": "benchmarks/pt_matmul.py", "category": 1},
    {"name": "pt_fft", "script": "benchmarks/pt_fft.py", "category": 1},
    {"name": "pt_vector_add", "script": "benchmarks/pt_vector_add.py", "category": 1},
    {"name": "pt_elementwise", "script": "benchmarks/pt_elementwise.py", "category": 1},

    # --- Category 2: Mixed ---
    {"name": "pt_resnet50_train", "script": "benchmarks/pt_resnet50_train.py", "category": 2},
    {"name": "pt_crypto_proxy", "script": "benchmarks/pt_crypto_proxy.py", "category": 2},
    {"name": "pt_lstm", "script": "benchmarks/pt_lstm.py", "category": 2},
]
# 18 Aug 2026 additions: pt_mobilenet (lighter CNN family than resnet50,
# depthwise-separable convs), pt_elementwise (memory-bandwidth-bound,
# distinct from matmul/fft/vector_add), pt_lstm (the only recurrent /
# non-feedforward workload in the set -- stresses a different part of the
# power curve than everything else here). 7 -> 10 trained workloads.

# =================================================================
# 2. CORE FREQUENCY GRID -- read from THIS card's real supported range at
#    runtime (fixes dvfs-triage: a hardcoded 500-1900 MHz list would either
#    silently miss most of a different card's range, or hit a value outside
#    it and abort the whole sweep -- see _build_core_sweep / the removed
#    sys.exit(1) below). The literal list below is used ONLY as a fallback
#    if NVML enumeration is unavailable -- it is what this script always
#    used, so behaviour on the exact machine it was written for is unchanged
#    in the worst case.
# =================================================================
_FALLBACK_CORE_FREQS = list(range(500, 1901, 100))  # 15 levels, 500-1900 MHz
FIXED_MEM_FREQ = 5871                     # Stock 3D memory clock (MHz) -- laptop-specific fallback label only

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(gpu_root, "data")
os.makedirs(data_dir, exist_ok=True)

COOLDOWN_TEMP_C = 70  # Cooldown threshold to avoid thermal throttling (set to 70C)

def _resolve_benchmark_script(rel_script):
    cand1 = os.path.join(gpu_root, rel_script)
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(script_dir, rel_script)
    if os.path.exists(cand2):
        return cand2
    return os.path.basename(rel_script)

def _build_core_sweep(handle, n_levels=15):
    """Evenly-spaced core clock levels across this card's REAL supported
    range, each snapped to a value the driver will actually accept. Falls
    back to the original literal laptop range if NVML enumeration fails or
    returns too little to work with."""
    if _supported_clocks is None:
        return _FALLBACK_CORE_FREQS
    try:
        _mems, cores = _supported_clocks(handle)
    except Exception:
        return _FALLBACK_CORE_FREQS
    cores = sorted(set(cores))
    if len(cores) < 2:
        return _FALLBACK_CORE_FREQS
    lo, hi = cores[0], cores[-1]
    targets = [lo + (hi - lo) * i / (n_levels - 1) for i in range(n_levels)]
    picked = sorted(set(min(cores, key=lambda c: abs(c - t)) for t in targets))
    return picked

def _slugify(name):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return s or "gpu"

def _resolve_output_csv(handle):
    """Name the output file after the connected card so runs on different
    rented/borrowed GPUs never collide or silently overwrite each other."""
    try:
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
    except Exception:
        name = ""
    if "3050" in name and "Ti" in name:
        # Exact match for the card this script was written for -- keep the
        # filename every other script in this tree already expects.
        return os.path.join(data_dir, "rtx3050ti_dataset.csv")
    if name:
        return os.path.join(data_dir, f"{_slugify(name)}_dataset.csv")
    return os.path.join(data_dir, "rtx3050ti_dataset.csv")

# =================================================================
# 3. PROFILING TELEMETRY ENGINE (Runs at 20Hz)
# =================================================================
class GPUProfiler(threading.Thread):
    def __init__(self, index=0):
        super().__init__()
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.running = True
        self.samples = {
            "pwr": [],
            "temp": [],
            "gpu_util": [],
            "mem_util": [],
            "vram_used": []
        }

    def run(self):
        while self.running:
            try:
                pwr = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                tmp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                utl_rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                
                self.samples["pwr"].append(pwr)
                self.samples["temp"].append(tmp)
                self.samples["gpu_util"].append(utl_rates.gpu)
                self.samples["mem_util"].append(utl_rates.memory)
                self.samples["vram_used"].append(mem_info.used / (1024.0 * 1024.0)) # in MB
            except pynvml.NVMLError:
                pass
            time.sleep(0.05)

    def stop(self):
        self.running = False

    def get_stats(self):
        if not self.samples["pwr"]:
            return 0.0, 0, 0.0, 0.0, 0.0
        avg_p = sum(self.samples["pwr"]) / len(self.samples["pwr"])
        max_t = max(self.samples["temp"])
        avg_gu = sum(self.samples["gpu_util"]) / len(self.samples["gpu_util"])
        avg_mu = sum(self.samples["mem_util"]) / len(self.samples["mem_util"])
        avg_vram = sum(self.samples["vram_used"]) / len(self.samples["vram_used"])
        return avg_p, max_t, avg_gu, avg_mu, avg_vram

# =================================================================
# 4. MAIN PROFILER LOOP
# =================================================================
def main():
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    core_freqs = _build_core_sweep(nvml_handle)
    output_csv = _resolve_output_csv(nvml_handle)
    gpu_name = pynvml.nvmlDeviceGetName(nvml_handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    print("=================================================================")
    print(f" GPU DVFS Dataset Profiler -- {gpu_name}")
    print("=================================================================")
    print(f"Total Workloads: {len(BENCHMARKS)}")
    print(f"Core Frequencies Grid ({len(core_freqs)} levels): {core_freqs}")
    print(f"Total Sweep Runs: {len(BENCHMARKS) * len(core_freqs)}")
    print(f"Target Cooldown Temp: {COOLDOWN_TEMP_C}C")
    print(f"Output File: {output_csv}")
    print("-----------------------------------------------------------------")
    print("WARNING: Requires root (Linux) or an Administrator terminal (Windows)")
    print("         to actually lock clocks -- run common/gpu_probe.py first")
    print("         if you're not sure this host allows it.")
    print("=================================================================\n")

    # Overwrite cleanly for reproducible dataset
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Workload_Name", "Workload_Type", "Core_F", "Mem_F", 
            "Avg_Time_S", "Avg_Pwr_W", "Peak_T_C", 
            "Avg_GPU_Util", "Avg_Mem_Util", "VRAM_Used_MB", "EDP"
        ])

        skipped_freqs = []
        try:
            for bench in BENCHMARKS:
                resolved_script = _resolve_benchmark_script(bench['script'])
                print(f"\n>>> Running Workload: {bench['name']} ({resolved_script})")

                if not os.path.exists(resolved_script):
                    print(f"Error: Script {resolved_script} not found. Skipping...")
                    continue

                for cf in core_freqs:
                    # 1. COOLDOWN LOOP
                    print(f"  Checking GPU Temperature...")
                    while True:
                        temp = pynvml.nvmlDeviceGetTemperature(nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                        if temp < COOLDOWN_TEMP_C:
                            break
                        print(f"    Current Temp: {temp}C (Target: <{COOLDOWN_TEMP_C}C). Cooling down...", end="\r")
                        time.sleep(2)

                    # 2. LOCK FREQUENCY
                    print(f"  Locking Core Clock to {cf} MHz...")
                    lock_cmd = ["nvidia-smi", "-i", "0", "-lgc", f"{cf},{cf}"]
                    res = subprocess.run(lock_cmd, capture_output=True, text=True)
                    if res.returncode != 0:
                        # One bad frequency shouldn't kill a multi-hour paid
                        # rental's whole sweep (dvfs-triage: this used to be
                        # sys.exit(1)). Skip just this config and keep going;
                        # if EVERY lock is failing (e.g. no root on this host)
                        # that becomes obvious fast from the skip count below
                        # without silently producing an empty dataset.
                        print(f"  [SKIP] Could not lock {cf} MHz: {res.stderr.strip()}")
                        skipped_freqs.append((bench['name'], cf))
                        continue

                    # 3. WARMUP + 3 BENCHMARK RUNS
                    run_data = {
                        "times": [], 
                        "pwrs": [], 
                        "gpu_utils": [], 
                        "mem_utils": [], 
                        "temps": [], 
                        "vram": []
                    }
                    
                    for run_idx in range(4): # 0 = Warmup, 1-3 = Benchmark
                        if run_idx > 0:
                            print(f"    Cooling pause for 5 seconds...", end="\r")
                            time.sleep(5)
                            
                        mode_str = "Warmup" if run_idx == 0 else f"Run {run_idx}/3"
                        print(f"    Executing {mode_str}...")
                        
                        monitor = GPUProfiler(index=0)
                        monitor.start()
                        
                        bench_res = subprocess.run(
                            [sys.executable, resolved_script], 
                            capture_output=True, 
                            text=True
                        )
                        
                        monitor.stop()
                        monitor.join()
                        
                        if bench_res.returncode != 0:
                            print(f"    [WARNING] Workload exited with error: {bench_res.stderr.strip()}")
                            continue

                        # Parse GPU kernel execution duration directly from stdout (Fix for Issue A1)
                        dur = None
                        for line in bench_res.stdout.splitlines():
                            if "Duration:" in line:
                                try:
                                    dur = float(line.split("Duration:")[1].strip())
                                    break
                                except ValueError:
                                    pass

                        if dur is None or dur <= 0:
                            print(f"    [WARNING] Failed to parse duration! Output: {bench_res.stdout.strip()}")
                            continue

                        if run_idx > 0: # Record only benchmarking runs
                            pwr, temp_peak, gpu_u, mem_u, vram_mb = monitor.get_stats()
                            run_data["times"].append(dur)
                            run_data["pwrs"].append(pwr)
                            run_data["gpu_utils"].append(gpu_u)
                            run_data["mem_utils"].append(mem_u)
                            run_data["temps"].append(temp_peak)
                            run_data["vram"].append(vram_mb)

                    # 4. AVERAGE DATA AND RECORD
                    if len(run_data["times"]) > 0:
                        avg_time = sum(run_data["times"]) / len(run_data["times"])
                        avg_pwr = sum(run_data["pwrs"]) / len(run_data["pwrs"])
                        avg_gpu_u = sum(run_data["gpu_utils"]) / len(run_data["gpu_utils"])
                        avg_mem_u = sum(run_data["mem_utils"]) / len(run_data["mem_utils"])
                        peak_temp = max(run_data["temps"])
                        avg_vram = sum(run_data["vram"]) / len(run_data["vram"])
                        edp = (avg_pwr * avg_time) * avg_time

                        writer.writerow([
                            bench['name'], 
                            bench['category'], 
                            cf, 
                            FIXED_MEM_FREQ, 
                            f"{avg_time:.4f}", 
                            f"{avg_pwr:.2f}", 
                            peak_temp, 
                            f"{avg_gpu_u:.2f}", 
                            f"{avg_mem_u:.2f}",
                            f"{avg_vram:.2f}",
                            f"{edp:.6f}"
                        ])
                        f.flush()
                        
                        print(f"  [RESULT] Core: {cf}MHz | GPU Dur: {avg_time:.4f}s | Pwr: {avg_pwr:.1f}W | GPU-Util: {avg_gpu_u:.1f}% | Mem-Util: {avg_mem_u:.1f}% | EDP: {edp:.4f}")

            total_attempted = len(BENCHMARKS) * len(core_freqs)
            if skipped_freqs:
                print(f"\n[SUMMARY] {len(skipped_freqs)}/{total_attempted} configs skipped "
                      f"(frequency lock failed): {skipped_freqs[:10]}"
                      + (" ..." if len(skipped_freqs) > 10 else ""))
                if len(skipped_freqs) == total_attempted:
                    print("[FATAL] EVERY frequency lock failed -- the output CSV is empty.")
                    print("        This almost always means the process doesn't have")
                    print("        permission to change clocks on this host (no root /")
                    print("        Administrator, or a rented container without GPU")
                    print("        clock-control privileges). Run common/gpu_probe.py")
                    print("        to confirm before spending more time or money here.")

        finally:
            print("\nCleaning up: resetting GPU clocks to factory defaults...")
            subprocess.run(["nvidia-smi", "-i", "0", "-rgc"], stdout=subprocess.DEVNULL)
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            print("GPU auto-scaling restored. Cleanup complete.")

if __name__ == "__main__":
    main()
