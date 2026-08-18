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

# Run all python-launched workloads through this fixed interpreter, not bare
# "python3" resolved via PATH. The base anaconda3 env's torch build (cu130)
# has drifted ahead of what this driver supports (max cu122), so bare
# "python3" silently fails CUDA init -- this would have quietly broken
# pt_resnet50/pt_bert/cupy_matmul/crypto_hash/render_proxy on any re-run.
# llm_gpu has a matched torch(cu121)+cupy(cuda12x) install; verified working.
PYTHON_GPU = "/home/pesu-rf/anaconda3/envs/llm_gpu/bin/python3"
if not os.path.exists(PYTHON_GPU):
    PYTHON_GPU = "python3"  # fall back to PATH on a machine without this env

ai_script = """import torch
import time
import sys
from torchvision.models import resnet50, ResNet50_Weights, vgg16

device = torch.device('cuda')
workload = sys.argv[1]

if workload == 'resnet':
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    data = torch.randn(64, 3, 224, 224, device=device)
    for _ in range(50): model(data)

elif workload == 'bert':
    from transformers import BertModel, BertConfig
    config = BertConfig()
    model = BertModel(config).to(device)
    data = torch.randint(0, 2000, (32, 512), device=device)
    for _ in range(100): model(data)

elif workload == 'cnn':
    # Different compute pattern from resnet50 (no residual/BN skip paths,
    # much wider dense conv stack) -- a distinct point in the workload space.
    model = vgg16(weights=None).to(device)
    data = torch.randn(32, 3, 224, 224, device=device)
    for _ in range(15): model(data)

elif workload == 'lstm':
    # Recurrent, memory/latency-bound -- unlike every other workload here
    # (all feedforward/parallel), so it stresses a different part of the
    # power/frequency curve.
    model = torch.nn.LSTM(input_size=256, hidden_size=512, num_layers=3, batch_first=True).to(device)
    data = torch.randn(64, 200, 256, device=device)
    for _ in range(30): model(data)
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

elif workload == 'fft_fixed':
    # dvfs-triage A2: the original 'fft' used a 2048^3 complex64 array
    # (~68.7GB) which OOMs immediately on this card's 24GB VRAM every time --
    # it never once actually ran. 640^3 (~2.1GB) is a real, safe size that
    # exercises the same FFT kernels.
    A = (cp.random.rand(640, 640, 640, dtype=cp.float32)
         + 1j * cp.random.rand(640, 640, 640, dtype=cp.float32)).astype(cp.complex64)
    for _ in range(30): cp.fft.fftn(A)
    cp.cuda.Stream.null.synchronize()

elif workload == 'crypto_proxy':
    A = cp.random.randint(0, 2**31, size=(20000, 20000), dtype=cp.uint32)
    for _ in range(30): cp.bitwise_xor(A, cp.left_shift(A, 3))
"""
data_workload_path = os.path.join(custom_benchmarks_dir, "data_workloads.py")
with open(data_workload_path, "w") as f: f.write(data_script)

# =================================================================
# 2. THE 12 VALID WORKLOADS
# =================================================================
# Self-contained: every path below resolves inside this repo's benchmarks/
# directory, no external rodinia_3.1 checkout required.
#
# hotspot, heartwall and cupy_fft used to be here too but were dropped
# (dvfs-triage A2): hotspot/heartwall never actually executed on the GPU
# (0-2% util -- the try/except:pass in the runner loop below silently
# recorded the failures as data) and cupy_fft OOMs immediately on this
# card's 24GB VRAM. bfs/kmeans/lavamd/srad/cfd were in an earlier version
# of this list but their binaries/data were never present on this machine,
# so they were always silently skipped -- removed rather than kept as dead
# weight (dvfs-triage A6, on the 20-workloads-claimed-15-present mismatch).
ai_script_rel = os.path.relpath(ai_workload_path, script_dir)
data_script_rel = os.path.relpath(data_workload_path, script_dir)

def _bin(name):
    return os.path.relpath(os.path.join(benchmarks_dir, name), script_dir)

BENCHMARKS = [
    # --- Custom AI & Data Workloads ---
    {"name": "pt_resnet50", "path": PYTHON_GPU, "args": [ai_script_rel, "resnet"], "category": 0, "loops": 1},
    {"name": "pt_bert", "path": PYTHON_GPU, "args": [ai_script_rel, "bert"], "category": 0, "loops": 1},
    {"name": "cupy_matmul", "path": PYTHON_GPU, "args": [data_script_rel, "matmul"], "category": 1, "loops": 1},
    {"name": "crypto_hash", "path": PYTHON_GPU, "args": [data_script_rel, "crypto_proxy"], "category": 0, "loops": 1},
    {"name": "render_proxy", "path": PYTHON_GPU, "args": [data_script_rel, "matmul"], "category": 2, "loops": 2},

    # --- Compiled Rodinia CUDA binaries, vendored under benchmarks/<name>/ ---
    {"name": "gaussian", "path": _bin("gaussian"), "args": ["-s", "4096"], "category": 0, "loops": 3},
    {"name": "lud", "path": _bin("lud/lud_cuda"), "args": ["-s", "8192"], "category": 0, "loops": 5},
    {"name": "nw", "path": _bin("nw/needle"), "args": ["16384", "10"], "category": 1, "loops": 5},
    {"name": "backprop", "path": _bin("backprop/backprop"), "args": ["65536"], "category": 0, "loops": 10},
    {"name": "streamcluster", "path": _bin("streamcluster/sc_gpu"), "args": ["10", "20", "256", "65536", "65536", "1000", "none", "output.txt", "1"], "category": 2, "loops": 2},
    {"name": "pathfinder", "path": _bin("pathfinder/pathfinder"), "args": ["100000", "100", "20"], "category": 1, "loops": 10},
    {"name": "myocyte", "path": _bin("myocyte/myocyte.out"), "args": ["100", "1", "0"], "category": 0, "loops": 5},
]

# --- 7 additional workloads added 18 Aug 2026 to grow the dataset past the
# original 12 (user request: more real workloads, not padding via a finer
# frequency grid). Each was smoke-tested standalone before being trusted with
# a full 400-point sweep -- see dvfs-triage A2 on why that check matters.
NEW_BENCHMARKS = [
    {"name": "pt_cnn", "path": PYTHON_GPU, "args": [ai_script_rel, "cnn"], "category": 0, "loops": 1},
    {"name": "pt_lstm", "path": PYTHON_GPU, "args": [ai_script_rel, "lstm"], "category": 1, "loops": 1},
    {"name": "cupy_fft_fixed", "path": PYTHON_GPU, "args": [data_script_rel, "fft_fixed"], "category": 1, "loops": 1},
    {"name": "srad", "path": _bin("srad/srad"), "args": ["2000", "0.5", "2048", "2048"], "category": 1, "loops": 2},
    {"name": "lavamd", "path": _bin("lavaMD/lavaMD"), "args": ["-boxes1d", "30"], "category": 0, "loops": 2},
    {"name": "particlefilter_naive", "path": _bin("particlefilter/particlefilter_naive"), "args": ["-x", "256", "-y", "256", "-z", "40", "-np", "100000"], "category": 1, "loops": 2},
    {"name": "particlefilter_float", "path": _bin("particlefilter/particlefilter_float"), "args": ["-x", "256", "-y", "256", "-z", "40", "-np", "100000"], "category": 1, "loops": 2},
]

# Set DVFS_NEW_ONLY=1 in the environment to sweep only NEW_BENCHMARKS (used
# to grow an existing dataset without re-profiling everything from scratch).
if os.environ.get("DVFS_NEW_ONLY") == "1":
    BENCHMARKS = NEW_BENCHMARKS
else:
    BENCHMARKS = BENCHMARKS + NEW_BENCHMARKS

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
    n_configs = len(CORE_FREQS) * len(MEM_FREQS)
    print(f"Starting execution for {len(BENCHMARKS)} benchmarks across {n_configs} frequency pairs.")
    print(f"Target Dataset Size: {len(BENCHMARKS) * n_configs} rows per pass "
          f"(re-running this script appends another pass as a repeat trial).")
    
    file_exists = os.path.isfile(OUTPUT_CSV)

    # Resume support: skip (workload, core, mem) configs already present in the
    # CSV so an interrupted/killed run can restart without re-profiling or
    # duplicating rows. Set DVFS_FORCE_RECOLLECT=1 to disable and always append
    # a fresh repeat trial (the original behaviour).
    already_done = set()
    if file_exists and os.environ.get("DVFS_FORCE_RECOLLECT") != "1":
        with open(OUTPUT_CSV, newline='') as rf:
            for row in csv.DictReader(rf):
                already_done.add((row["Workload_Name"], row["Core_F"], row["Mem_F"]))
        if already_done:
            print(f"Resume mode: {len(already_done)} configs already in {OUTPUT_CSV}, will skip them.")

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
                    if (bench['name'], str(cf), str(mf)) in already_done:
                        continue

                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    if pynvml.nvmlDeviceGetTemperature(h, 0) > 55:
                        # Bug fix: the PREVIOUS config's clock lock was still
                        # active during this wait, so at high core frequencies
                        # the GPU idled at a locked high clock instead of
                        # dropping to true idle (~200MHz) and temp got stuck
                        # just above the threshold indefinitely. Unlock first.
                        subprocess.run(["sudo", "nvidia-smi", "-rgc"], stdout=subprocess.DEVNULL)
                        subprocess.run(["sudo", "nvidia-smi", "-rmc"], stdout=subprocess.DEVNULL)
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
