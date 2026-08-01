import time
import csv
import subprocess
import threading
import pynvml
import os
import sys

# =================================================================
# 1. THE WORKLOADS (Pure Python equivalents running as separate processes)
# =================================================================
BENCHMARKS = [
    # --- Category 0: Compute-Bound ---
    {"name": "pt_resnet50", "script": "benchmarks/pt_resnet50.py", "category": 0},
    {"name": "pt_bert", "script": "benchmarks/pt_bert.py", "category": 0},
    
    # --- Category 1: Memory-Bound ---
    {"name": "pt_matmul", "script": "benchmarks/pt_matmul.py", "category": 1},
    {"name": "pt_fft", "script": "benchmarks/pt_fft.py", "category": 1},
    {"name": "pt_vector_add", "script": "benchmarks/pt_vector_add.py", "category": 1},
    
    # --- Category 2: Mixed ---
    {"name": "pt_resnet50_train", "script": "benchmarks/pt_resnet50_train.py", "category": 2},
    {"name": "pt_crypto_proxy", "script": "benchmarks/pt_crypto_proxy.py", "category": 2},
]

# =================================================================
# 2. LAPTOP CORE FREQUENCY GRID (Mem Clock remains stock 5871 MHz)
# =================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(gpu_root, "data")
os.makedirs(data_dir, exist_ok=True)

OUTPUT_CSV = os.path.join(data_dir, "rtx3050ti_dataset.csv")
COOLDOWN_TEMP_C = 60  # Set to 60C for thermal safety and faster iteration on cleaned laptop

def _resolve_benchmark_script(rel_script):
    cand1 = os.path.join(gpu_root, rel_script)
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(script_dir, rel_script)
    if os.path.exists(cand2):
        return cand2
    return os.path.basename(rel_script)

# =================================================================
# 3. PROFILING TELEMETRY ENGINE (Runs at 20Hz)
# =================================================================
class GPUProfiler(threading.Thread):
    def __init__(self, index=0):
        super().__init__()
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.running = True
        self.samples = {"pwr": [], "temp": [], "util": []}

    def run(self):
        while self.running:
            try:
                pwr = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                tmp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                utl = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                self.samples["pwr"].append(pwr)
                self.samples["temp"].append(tmp)
                self.samples["util"].append(utl)
            except pynvml.NVMLError:
                pass
            time.sleep(0.05)

    def stop(self):
        self.running = False

    def get_stats(self):
        if not self.samples["pwr"]:
            return 0.0, 0, 0.0
        avg_p = sum(self.samples["pwr"]) / len(self.samples["pwr"])
        max_t = max(self.samples["temp"])
        avg_u = sum(self.samples["util"]) / len(self.samples["util"])
        return avg_p, max_t, avg_u

# =================================================================
# 4. MAIN PROFILER LOOP
# =================================================================
def main():
    print("=================================================================")
    print(" RTX 3050 Ti Laptop DVFS Dataset Profiler (Windows)")
    print("=================================================================")
    print(f"Total Workloads: {len(BENCHMARKS)}")
    print(f"Core Frequencies Grid: {CORE_FREQS}")
    print(f"Total Sweep Runs: {len(BENCHMARKS) * len(CORE_FREQS)}")
    print(f"Target Cooldown Temp: {COOLDOWN_TEMP_C}C")
    print(f"Output File: {OUTPUT_CSV}")
    print("-----------------------------------------------------------------")
    print("WARNING: Make sure you ran this terminal as Administrator!")
    print("=================================================================\n")

    # Initialize NVML to query temperature for cooldown checking
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    file_exists = os.path.isfile(OUTPUT_CSV)
    
    try:
        with open(OUTPUT_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Workload_Name", "Workload_Type", "Core_F", "Mem_F", "Avg_Time_S", "Avg_Pwr_W", "Peak_T_C", "Avg_Util_PCT", "EDP"])

            for bench in BENCHMARKS:
                resolved_script = _resolve_benchmark_script(bench['script'])
                print(f"\n>>> Running Workload: {bench['name']} ({resolved_script})")
                
                if not os.path.exists(resolved_script):
                    print(f"Error: Script {resolved_script} not found. Skipping...")
                    continue

                for cf in CORE_FREQS:
                    # 1. COOLDOWN LOOP (Prevents thermal throttling noise)
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
                        print("  [ERROR] Failed to lock frequency! Are you running as Administrator?")
                        print(f"  nvidia-smi output: {res.stderr.strip()}")
                        sys.exit(1)

                    # 3. WARMUP + 3 BENCHMARK RUNS
                    run_data = {"times": [], "pwrs": [], "utils": [], "temps": []}
                    
                    for run_idx in range(4): # 0 = Warmup, 1-3 = Benchmark
                        # Explicit cooling pause before runs to prevent heat build-up
                        if run_idx > 0:
                            print(f"    Cooling pause for 5 seconds...", end="\r")
                            time.sleep(5)
                            
                        mode_str = "Warmup" if run_idx == 0 else f"Run {run_idx}/3"
                        print(f"    Executing {mode_str}...")
                        
                        monitor = GPUProfiler(index=0)
                        monitor.start()
                        
                        start_time = time.time()
                        # Run workload as a separate process to guarantee memory reclamation
                        bench_res = subprocess.run([sys.executable, resolved_script], capture_output=True, text=True)
                        dur = time.time() - start_time
                        
                        monitor.stop()
                        monitor.join()
                        
                        if bench_res.returncode != 0:
                            print(f"    [WARNING] Workload exited with error: {bench_res.stderr.strip()}")
                            continue

                        if run_idx > 0: # Record only the benchmarking runs
                            pwr, temp_peak, util = monitor.get_stats()
                            run_data["times"].append(dur)
                            run_data["pwrs"].append(pwr)
                            run_data["temps"].append(temp_peak)
                            run_data["utils"].append(util)

                    # 4. AVERAGE DATA AND RECORD
                    if len(run_data["times"]) > 0:
                        avg_time = sum(run_data["times"]) / len(run_data["times"])
                        avg_pwr = sum(run_data["pwrs"]) / len(run_data["pwrs"])
                        avg_util = sum(run_data["utils"]) / len(run_data["utils"])
                        peak_temp = max(run_data["temps"])
                        edp = (avg_pwr * avg_time) * avg_time

                        writer.writerow([
                            bench['name'], 
                            bench['category'], 
                            cf, 
                            FIXED_MEM_FREQ, 
                            f"{avg_time:.4f}", 
                            f"{avg_pwr:.2f}", 
                            peak_temp, 
                            f"{avg_util:.2f}", 
                            f"{edp:.6f}"
                        ])
                        f.flush()
                        
                        print(f"  [RESULT] Core: {cf} MHz | Avg Time: {avg_time:.3f}s | Avg Pwr: {avg_pwr:.2f}W | Peak Temp: {peak_temp}C | EDP: {edp:.4f}")

    finally:
        # ALWAYS RESET CLOCKS TO AUTOMATIC
        print("\nCleaning up: resetting GPU clocks to factory defaults...")
        subprocess.run(["nvidia-smi", "-i", "0", "-rgc"], stdout=subprocess.DEVNULL)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        print("GPU auto-scaling restored. Cleanup complete.")

if __name__ == "__main__":
    main()
