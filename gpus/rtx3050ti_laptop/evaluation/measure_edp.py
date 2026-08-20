import os
import sys
import time
import csv
import threading
import subprocess
import pynvml

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
N_ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
WORKLOAD_NAME = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("WORKLOAD_NAME", "pt_matmul")

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
benchmarks_dir = os.path.join(gpu_root, "benchmarks")
zs_dir = os.path.join(benchmarks_dir, "zero_shot_workloads")
data_dir = os.path.join(gpu_root, "data")
os.makedirs(data_dir, exist_ok=True)

# Trained workloads (profiled and in the training set) + zero-shot ones
# (never profiled or trained on -- see benchmarks/zero_shot_workloads/).
# Edit/extend this table to target a different one via argv[3].
IN_DATASET_WORKLOADS = {
    "pt_resnet50", "pt_bert", "pt_mobilenet", "pt_matmul", "pt_fft",
    "pt_vector_add", "pt_elementwise", "pt_resnet50_train",
    "pt_crypto_proxy", "pt_lstm",
}
ZERO_SHOT_WORKLOADS = {"pt_vit", "pt_squeezenet", "pt_sort_reduce"}

if WORKLOAD_NAME in ZERO_SHOT_WORKLOADS:
    _script = os.path.join(zs_dir, f"{WORKLOAD_NAME}.py")
elif WORKLOAD_NAME in IN_DATASET_WORKLOADS:
    _script = os.path.join(benchmarks_dir, f"{WORKLOAD_NAME}.py")
else:
    raise SystemExit(
        f"Unknown WORKLOAD_NAME '{WORKLOAD_NAME}'. Choices: "
        f"{sorted(IN_DATASET_WORKLOADS | ZERO_SHOT_WORKLOADS)}"
    )
WORKLOAD = [sys.executable, _script]

MAX_CORE = 1900
GPU_INDEX = 0
SAMPLE_HZ = 20

class PowerSampler(threading.Thread):
    def __init__(self, handle):
        super().__init__()
        self.h = handle
        self.running = True
        self.rows = []

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
    subprocess.run(["nvidia-smi", "-i", "0", "-lgc", f"{MAX_CORE},{MAX_CORE}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def reset_clocks():
    subprocess.run(["nvidia-smi", "-i", "0", "-rgc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)

    if LABEL == "governor":
        print("Governor baseline: pinning Core clock to max (race-to-idle).")
        set_max_clocks()
        time.sleep(1.0)
    else:
        print(f"'{LABEL}' run: NOT touching clocks directly (AI daemon should control them).")

    sampler = PowerSampler(h)
    sampler.start()

    print(f"Running {' '.join(WORKLOAD)} x{N_ITERS} ...")
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
        print("Not enough telemetry samples collected!")
        pynvml.nvmlShutdown()
        return

    energy = 0.0
    for (t1, p1, *_), (t2, p2, *_) in zip(rows, rows[1:]):
        energy += 0.5 * (p1 + p2) * (t2 - t1)

    powers = [r[1] for r in rows]
    mean_p = sum(powers) / len(powers)
    t_per_iter = wall / N_ITERS
    edp_iter = mean_p * t_per_iter * t_per_iter

    # pt_matmul keeps the original bare filename (power_governor.csv /
    # power_ai.csv) so the existing default result isn't clobbered.
    suffix = "" if WORKLOAD_NAME == "pt_matmul" else f"_{WORKLOAD_NAME}"
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
