import time
import csv
import subprocess
import threading
import pynvml
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
benchmarks_dir = os.path.join(gpu_root, "benchmarks")
custom_benchmarks_dir = os.path.join(benchmarks_dir, "custom_benchmarks")
os.makedirs(custom_benchmarks_dir, exist_ok=True)

ai_script = """import torch
import time
import sys
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel, BertConfig

device = torch.device('cuda')
workload = sys.argv[1]

if workload == 'resnet':
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    data = torch.randn(64, 3, 224, 224, device=device)
    for _ in range(50): model(data)
    
elif workload == 'bert':
    config = BertConfig()
    model = BertModel(config).to(device)
    data = torch.randint(0, 2000, (32, 512), device=device)
    for _ in range(100): model(data)
"""
ai_workload_path = os.path.join(custom_benchmarks_dir, "ai_workloads.py")
with open(ai_workload_path, "w") as f: f.write(ai_script)

data_script = """import cupy as cp
import time
import sys

workload = sys.argv[1]
if workload == 'matmul':
    A = cp.random.rand(8000, 8000, dtype=cp.float32)
    B = cp.random.rand(8000, 8000, dtype=cp.float32)
    for _ in range(20): cp.matmul(A, B)
    
elif workload == 'fft':
    A = cp.random.rand(2048, 2048, 2048, dtype=cp.complex64)
    for _ in range(5): cp.fft.fftn(A)
    
elif workload == 'crypto_proxy':
    A = cp.random.randint(0, 2**31, size=(20000, 20000), dtype=cp.uint32)
    for _ in range(30): cp.bitwise_xor(A, cp.left_shift(A, 3))
"""
data_workload_path = os.path.join(custom_benchmarks_dir, "data_workloads.py")
with open(data_workload_path, "w") as f: f.write(data_script)

# =================================================================
# 2. THE 20 WORKLOADS (Tailored to rodinia_3.1/cuda/gaussian structure)
# =================================================================
# We resolve custom script paths dynamically
ai_script_rel = os.path.relpath(ai_workload_path, script_dir)
data_script_rel = os.path.relpath(data_workload_path, script_dir)
gaussian_bin = os.path.relpath(os.path.join(benchmarks_dir, "gaussian"), script_dir)

BENCHMARKS = [
    # --- Custom AI & Data Workloads ---
    {"name": "pt_resnet50", "path": "python3", "args": [ai_script_rel, "resnet"], "category": 0, "loops": 1},
    {"name": "pt_bert", "path": "python3", "args": [ai_script_rel, "bert"], "category": 0, "loops": 1},
    {"name": "cupy_matmul", "path": "python3", "args": [data_script_rel, "matmul"], "category": 1, "loops": 1},
    {"name": "cupy_fft", "path": "python3", "args": [data_script_rel, "fft"], "category": 1, "loops": 1},
    {"name": "crypto_hash", "path": "python3", "args": [data_script_rel, "crypto_proxy"], "category": 0, "loops": 1},
    {"name": "render_proxy", "path": "python3", "args": [data_script_rel, "matmul"], "category": 2, "loops": 2}, 
    
    # --- Existing Rodinia Binaries ---
    {"name": "gaussian", "path": gaussian_bin, "args": ["-s", "4096"], "category": 0, "loops": 3},
    {"name": "lud", "path": "../lud/cuda/lud_cuda", "args": ["-s", "8192"], "category": 0, "loops": 5},
    {"name": "nw", "path": "../nw/needle", "args": ["16384", "10"], "category": 1, "loops": 5},
    
    # --- Compiled Rodinia Binaries (Requires 'make' in rodinia_3.1/cuda) ---
    {"name": "hotspot", "path": "../hotspot/hotspot", "args": ["512", "2", "2", "../../data/hotspot/temp_512", "../../data/hotspot/power_512", "output.out"], "category": 1, "loops": 10},
    {"name": "backprop", "path": "../backprop/backprop", "args": ["65536"], "category": 0, "loops": 10},
    {"name": "bfs", "path": "../bfs/bfs", "args": ["../../data/bfs/graph1MW_6.txt"], "category": 1, "loops": 15},
    {"name": "kmeans", "path": "../kmeans/kmeans", "args": ["-o", "-i", "../../data/kmeans/kdd_cup"], "category": 2, "loops": 5},
    {"name": "lavamd", "path": "../lavaMD/lavaMD", "args": ["-boxes1d", "15"], "category": 0, "loops": 2},
    {"name": "srad", "path": "../srad/srad_v1/srad", "args": ["100", "0.5", "502", "458"], "category": 1, "loops": 5},
    {"name": "streamcluster", "path": "../streamcluster/sc_gpu", "args": ["10", "20", "256", "65536", "65536", "1000", "none", "output.txt", "1"], "category": 2, "loops": 2},
    {"name": "pathfinder", "path": "../pathfinder/pathfinder", "args": ["100000", "100", "20"], "category": 1, "loops": 10},
    {"name": "myocyte", "path": "../myocyte/myocyte.out", "args": ["100", "1", "0"], "category": 0, "loops": 5},
    {"name": "cfd", "path": "../cfd/euler3d", "args": ["../../data/cfd/fvcorr.domn.097K"], "category": 0, "loops": 5},
    {"name": "heartwall", "path": "../heartwall/heartwall", "args": ["../../data/heartwall/test.avi", "20"], "category": 2, "loops": 2}
]

# =================================================================
# 3. HIGH-RESOLUTION FREQUENCY GRID (400 COMBOs -> 8000 Rows)
# =================================================================
CORE_FREQS = list(range(1000, 2951, 50))  # 40 values
MEM_FREQS  = list(range(5001, 9502, 500)) # 10 values

data_dir = os.path.join(gpu_root, "data")
os.makedirs(data_dir, exist_ok=True)
OUTPUT_CSV = os.path.join(data_dir, "rtx4500_8000_dataset.csv")

# =================================================================
# 4. PROFILER ENGINE
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
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                self.samples["pwr"].append(pwr)
                self.samples["temp"].append(tmp)
                self.samples["util"].append(util)
            except pynvml.NVMLError:
                pass
            time.sleep(0.05)

    def stop(self):
        self.running = False

    def get_stats(self):
        if not self.samples["pwr"]: return 0, 0, 0
        avg_p = sum(self.samples["pwr"]) / len(self.samples["pwr"])
        max_t = max(self.samples["temp"])
        avg_u = sum(self.samples["util"]) / len(self.samples["util"])
        return avg_p, max_t, avg_u

# =================================================================
# 5. MAIN EXECUTION LOOP
# =================================================================
if __name__ == "__main__":
    print(f"Starting execution for {len(BENCHMARKS)} benchmarks across 400 frequencies.")
    print("Target Dataset Size: 8000 Rows.")
    
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Workload_Name", "Workload_Type", "Core_F", "Mem_F", "Avg_Time_S", "Avg_Pwr_W", "Peak_T_C", "Avg_Util_PCT", "EDP"])

        for bench in BENCHMARKS:
            print(f"\n--- Running Workload: {bench['name']} ---")
            
            # Skip safely if the Rodinia binary hasn't been compiled or data is missing
            if not os.path.exists(bench['path']) and bench['path'] != "python3":
                print(f"WARNING: Executable {bench['path']} not found. Skipping to prevent crash.")
                continue
                
            # Quick check to ensure the ../../data/ file exists for Rodinia workloads
            if len(bench['args']) > 0 and "../../data/" in bench['args'][-1]:
                if not os.path.exists(bench['args'][-1]):
                    print(f"WARNING: Data file {bench['args'][-1]} not found. Skipping {bench['name']}.")
                    continue

            for cf in CORE_FREQS:
                for mf in MEM_FREQS:
                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    while pynvml.nvmlDeviceGetTemperature(h, 0) > 55:
                        print(f"Cooling down {bench['name']}...", end="\r")
                        time.sleep(1)

                    subprocess.run(["sudo", "nvidia-smi", "-lgc", f"{cf},{cf}"], stdout=subprocess.DEVNULL)
                    subprocess.run(["sudo", "nvidia-smi", "-lmc", f"{mf},{mf}"], stderr=subprocess.DEVNULL)

                    monitor = GPUProfiler()
                    monitor.start()
                    start = time.time()
                    
                    for _ in range(bench['loops']):
                        try:
                            subprocess.run([bench['path']] + bench['args'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except Exception:
                            pass 
                            
                    total_dur = time.time() - start
                    monitor.stop()
                    monitor.join()

                    avg_time_per_run = total_dur / bench['loops']
                    p, t, u = monitor.get_stats()
                    edp = (p * avg_time_per_run) * avg_time_per_run

                    writer.writerow([bench['name'], bench['category'], cf, mf, f"{avg_time_per_run:.4f}", f"{p:.2f}", t, f"{u:.2f}", f"{edp:.6f}"])
                    f.flush()
                    print(f"  {bench['name']} @ {cf}/{mf} | Time: {avg_time_per_run:.3f}s | Util: {u:.1f}%")

    subprocess.run(["sudo", "nvidia-smi", "-rgc"], stdout=subprocess.DEVNULL)
    subprocess.run(["sudo", "nvidia-smi", "-rmc"], stderr=subprocess.DEVNULL)
    print("\nDataset Generation Complete: rtx4500_8000_dataset.csv created.")
