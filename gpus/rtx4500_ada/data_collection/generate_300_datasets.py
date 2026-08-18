import time
import csv
import subprocess
import threading
import pynvml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
benchmarks_dir = os.path.join(gpu_root, "benchmarks")
data_dir = os.path.join(gpu_root, "data")
os.makedirs(data_dir, exist_ok=True)

def _bin(name):
    return os.path.relpath(os.path.join(benchmarks_dir, name), script_dir)

gaussian_bin = _bin("gaussian")

# =================================================================
# 1. EXPANDED WORKLOADS (3 BENCHMARKS)
# =================================================================
BENCHMARKS = [
    # 1. GAUSSIAN (Compute Heavy)
    {"name": "gaussian", "path": gaussian_bin, "args": ["-s", "4096"], "category": 0},

    # 2. LU DECOMPOSITION (Compute Heavy)
    {"name": "lud", "path": _bin("lud/lud_cuda"), "args": ["-s", "4096"], "category": 0},

    # 3. NEEDLEMAN-WUNSCH (Integer/Memory Heavy)
    # Args: <dim> <penalty>
    {"name": "nw", "path": _bin("nw/needle"), "args": ["4096", "10"], "category": 1},
]

# =================================================================
# 2. HIGH-RESOLUTION FREQUENCY GRID (TARGET: 300 ROWS)
# =================================================================
# Core: 1200 to 3100 MHz in steps of 100 (20 values)
CORE_FREQS = list(range(1200, 3105, 100))

# Memory: 5001 to 9001 MHz in steps of 1000 (5 values)
MEM_FREQS  = list(range(5001, 9002, 1000))

OUTPUT_CSV = os.path.join(data_dir, "rtx4500_300_dataset.csv")

# =================================================================
# 3. PROFILING ENGINE
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
                time.sleep(0.05) # 20Hz Sampling
            except: break

    def stop(self): self.running = False

    def get_stats(self):
        if not self.samples["pwr"]: return 0, 0, 0
        return (sum(self.samples["pwr"])/len(self.samples["pwr"]),
                max(self.samples["temp"]),
                sum(self.samples["util"])/len(self.samples["util"]))

# =================================================================
# 4. EXECUTION LOOP
# =================================================================
def main():
    print(f"Plan: {len(BENCHMARKS)} Benchmarks x {len(CORE_FREQS)} Cores x {len(MEM_FREQS)} Mems = {len(BENCHMARKS)*len(CORE_FREQS)*len(MEM_FREQS)} Datasets")
    
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Workload_Name", "Workload_Type", "Core_F", "Mem_F", "Avg_Time_S", "Avg_Pwr_W", "Peak_T_C", "Avg_Util_PCT", "EDP"])

        for bench in BENCHMARKS:
            print(f"\n==========================================")
            print(f" BENCHMARK: {bench['name']}")
            print(f"==========================================")
            
            # Verify file exists before running
            if not os.path.exists(bench['path']):
                print(f"ERROR: Could not find {bench['path']}")
                print(f"Did you compile it? Skipping...")
                continue

            for cf in CORE_FREQS:
                for mf in MEM_FREQS:
                    # COOLDOWN CHECK
                    while True:
                        pynvml.nvmlInit()
                        h = pynvml.nvmlDeviceGetHandleByIndex(0)
                        # Increased threshold to avoid infinite waiting
                        if pynvml.nvmlDeviceGetTemperature(h, 0) < 55: break
                        print(f"Cooling down... (Temp > 45C)", end="\r")
                        time.sleep(1)

                    # SET CLOCKS
                    subprocess.run(["sudo", "nvidia-smi", "-lgc", f"{cf},{cf}"], stdout=subprocess.DEVNULL)
                    subprocess.run(["sudo", "nvidia-smi", "-lmc", f"{mf},{mf}"], stderr=subprocess.DEVNULL)

                    # WARMUP + 3 RUNS
                    run_data = {"times": [], "pwrs": [], "utils": [], "temps": []}
                    
                    for r in range(4): # 0=Warmup, 1-3=Record
                        monitor = GPUProfiler()
                        monitor.start()
                        
                        start = time.time()
                        subprocess.run([bench['path']] + bench['args'], stdout=subprocess.DEVNULL)
                        dur = time.time() - start
                        
                        monitor.stop()
                        monitor.join()

                        if r > 0: # Skip warmup
                            p, t, u = monitor.get_stats()
                            run_data["times"].append(dur)
                            run_data["pwrs"].append(p)
                            run_data["temps"].append(t)
                            run_data["utils"].append(u)
                        
                        time.sleep(0.5)

                    # AVERAGE
                    if len(run_data["times"]) > 0:
                        avg_time = sum(run_data["times"]) / 3.0
                        avg_pwr = sum(run_data["pwrs"]) / 3.0
                        avg_util = sum(run_data["utils"]) / 3.0
                        peak_t = max(run_data["temps"])
                        edp = (avg_pwr * avg_time) * avg_time

                        writer.writerow([bench['name'], bench['category'], cf, mf,
                                         f"{avg_time:.4f}", f"{avg_pwr:.2f}", peak_t, f"{avg_util:.2f}", f"{edp:.6f}"])
                        f.flush()
                        
                        print(f" {bench['name']} @ {cf}/{mf} | Time:{avg_time:.3f}s | Pwr:{avg_pwr:.1f}W")

    subprocess.run(["sudo", "nvidia-smi", "-rgc"], stdout=subprocess.DEVNULL)
    print("\nGeneration Complete.")

if __name__ == "__main__":
    main()
