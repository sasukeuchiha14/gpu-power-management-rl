import time
import csv
import subprocess
import threading
import pynvml
import os

# ONLY Run the broken benchmarks
BENCHMARKS = [
    # Running them in a loop of 20 iterations to force 2+ seconds duration
    {"name": "lud", "path": "../lud/cuda/lud_cuda", "args": ["-s", "8192"], "category": 0, "loops": 10}, 
    {"name": "nw", "path": "../nw/needle", "args": ["16384", "10"], "category": 1, "loops": 10},
]

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(gpu_root, "data")
os.makedirs(data_dir, exist_ok=True)

OUTPUT_CSV = os.path.join(data_dir, "rtx4500_fixed_part2.csv") # Save to NEW file in data dir

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
                time.sleep(0.05)
            except: break

    def stop(self): self.running = False

    def get_stats(self):
        if not self.samples["pwr"]: return 0, 0, 0
        return (sum(self.samples["pwr"])/len(self.samples["pwr"]),
                max(self.samples["temp"]),
                sum(self.samples["util"])/len(self.samples["util"]))

def main():
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Workload_Name", "Workload_Type", "Core_F", "Mem_F", "Avg_Time_S", "Avg_Pwr_W", "Peak_T_C", "Avg_Util_PCT", "EDP"])

        for bench in BENCHMARKS:
            print(f"Fixing {bench['name']}...")
            for cf in CORE_FREQS:
                for mf in MEM_FREQS:
                    # COOL DOWN Check
                    while True:
                        pynvml.nvmlInit()
                        h = pynvml.nvmlDeviceGetHandleByIndex(0)
                        if pynvml.nvmlDeviceGetTemperature(h, 0) < 55: break
                        time.sleep(1)

                    subprocess.run(["sudo", "nvidia-smi", "-lgc", f"{cf},{cf}"], stdout=subprocess.DEVNULL)
                    subprocess.run(["sudo", "nvidia-smi", "-lmc", f"{mf},{mf}"], stderr=subprocess.DEVNULL)

                    monitor = GPUProfiler()
                    monitor.start()
                    
                    start = time.time()
                    # === THE FIX: RUN MULTIPLE TIMES ===
                    for _ in range(bench['loops']):
                        subprocess.run([bench['path']] + bench['args'], stdout=subprocess.DEVNULL)
                    total_duration = time.time() - start
                    
                    monitor.stop()
                    monitor.join()

                    # Average time per single run
                    avg_time_per_run = total_duration / bench['loops'] 
                    
                    p, t, u = monitor.get_stats()
                    edp = (p * avg_time_per_run) * avg_time_per_run

                    writer.writerow([bench['name'], bench['category'], cf, mf,
                                     f"{avg_time_per_run:.4f}", f"{p:.2f}", t, f"{u:.2f}", f"{edp:.6f}"])
                    f.flush()
                    print(f"  {bench['name']} @ {cf}/{mf}: Util={u:.1f}% Time={avg_time_per_run:.4f}s")

    subprocess.run(["sudo", "nvidia-smi", "-rgc"], stdout=subprocess.DEVNULL)

if __name__ == "__main__":
    main()
