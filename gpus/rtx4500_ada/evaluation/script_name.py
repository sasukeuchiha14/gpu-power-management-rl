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

# ==========================================
# 1. HARDWARE & WORKLOAD CONFIG
# ==========================================
BENCHMARK_PATH = os.path.relpath(os.path.join(benchmarks_dir, "gaussian"), script_dir)
BENCHMARK_ARGS = ["-s", "1024"]  # Increased size for better power profiling
OUTPUT_CSV = os.path.join(data_dir, "rtx4500_hardware_dataset.csv")

# RTX 4500 Clock Ranges (MHz)
# Check 'nvidia-smi -q -d SUPPORTED_CLOCKS' to verify these for your card
CORE_FREQS = [1200, 1500, 1800, 2100, 2400, 2580] 
MEM_FREQS  = [5001, 7501, 9501]

# ==========================================
# 2. PROFILER ENGINE (NVML)
# ==========================================
class GPUProfiler(threading.Thread):
    def __init__(self, index=0):
        super().__init__()
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.running = True
        self.samples = {"pwr": [], "temp": [], "util": []}

    def run(self):
        while self.running:
            pwr = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            tmp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            utl = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            self.samples["pwr"].append(pwr)
            self.samples["temp"].append(tmp)
            self.samples["util"].append(utl)
            time.sleep(0.05) # 20Hz Sampling

    def stop(self):
        self.running = False

    def get_stats(self):
        if not self.samples["pwr"]: return 0, 0, 0
        return (sum(self.samples["pwr"])/len(self.samples["pwr"]), 
                max(self.samples["temp"]), 
                sum(self.samples["util"])/len(self.samples["util"]))

# ==========================================
# 3. MAIN DATA COLLECTION LOOP
# ==========================================
def main():
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Core_F", "Mem_F", "Time_S", "Avg_Pwr_W", "Peak_T_C", "Util_PCT", "EDP"])

        for cf in CORE_FREQS:
            for mf in MEM_FREQS:
                # A. COOLDOWN: Reset and wait for GPU to chill
                subprocess.run(["sudo", "nvidia-smi", "-rgc"], stdout=subprocess.DEVNULL)
                while True:
                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    curr_temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    if curr_temp < 45: break
                    print(f"Cooling down... Current Temp: {curr_temp}°C", end="\r")
                    time.sleep(2)

                # B. SET CLOCKS (DVFS)
                print(f"\nTesting Core:{cf} Mem:{mf} ...")
                subprocess.run(["sudo", "nvidia-smi", "-lgc", f"{cf},{cf}"], check=True)
                # Note: Memory locking depends on driver support; ignoring failure
                subprocess.run(["sudo", "nvidia-smi", "-lmc", f"{mf},{mf}"], stderr=subprocess.DEVNULL)

                # C. RUN & PROFILE
                monitor = GPUProfiler()
                monitor.start()
                start = time.time()
                subprocess.run([BENCHMARK_PATH] + BENCHMARK_ARGS, stdout=subprocess.DEVNULL)
                dur = time.time() - start
                monitor.stop()
                monitor.join()

                # D. LOG DATA
                pwr, tmp, utl = monitor.get_stats()
                edp = (pwr * dur) * dur
                writer.writerow([cf, mf, f"{dur:.4f}", f"{pwr:.2f}", tmp, utl, f"{edp:.6f}"])
                f.flush()

    subprocess.run(["sudo", "nvidia-smi", "-rgc"], stdout=subprocess.DEVNULL)
    print("\nDataset Generation Complete.")

if __name__ == "__main__":
    main()
