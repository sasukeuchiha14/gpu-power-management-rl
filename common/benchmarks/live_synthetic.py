"""
live_synthetic.py -- stationary live workloads with a clean, known regime, for testing whether
deploy_live.py's behaviour (cycle lock-in in particular) generalises beyond ResNet-50 training.
Same status-file contract as live_resnet50_train.py.

  --mode matmul    large fp32 GEMM loop            -> compute-bound
  --mode membound  large elementwise / copy loop   -> memory-bandwidth-bound

    python3 common/benchmarks/live_synthetic.py --mode matmul --minutes 15
"""
import argparse, gc, json, os, signal, time
import torch

DEFAULT_STATUS_FILE = "/tmp/dvfs_workload_status.json"


def write_status(path, step, ms_per_iter, iters_per_sec):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"timestamp": time.time(), "step": step,
                   "ms_per_iter": ms_per_iter, "iters_per_sec": iters_per_sec}, f)
    os.replace(tmp, path)


def make_step(mode, device):
    if mode == "matmul":
        a = torch.randn(6144, 6144, device=device)
        b = torch.randn(6144, 6144, device=device)
        torch.backends.cuda.matmul.allow_tf32 = False
        def step():
            return a @ b
        return step, 80
    if mode == "membound":
        x = torch.randn(256 * 1024 * 1024 // 4 * 4, device=device)   # ~1 GiB fp32
        y = torch.empty_like(x)
        def step():
            torch.mul(x, 1.0001, out=y)
            torch.add(y, 0.5, out=x)
            return x
        return step, 160
    raise SystemExit(f"unknown mode {mode}")


def run(mode, minutes, status_file):
    device = torch.device("cuda")
    step_fn, window = make_step(mode, device)
    stop = {"flag": False}
    def handle(signum, frame): stop["flag"] = True
    signal.signal(signal.SIGINT, handle); signal.signal(signal.SIGTERM, handle)
    for _ in range(5): step_fn()
    torch.cuda.synchronize()
    print(f"[live_synthetic] mode={mode} running for {minutes} min, window={window} -> {status_file}", flush=True)
    deadline = time.time() + minutes * 60.0
    n, t0 = 0, time.perf_counter()
    while time.time() < deadline and not stop["flag"]:
        step_fn(); n += 1
        if n % window == 0:
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            write_status(status_file, n, 1000.0 * el / window, window / el)
            t0 = time.perf_counter()
    gc.collect(); torch.cuda.empty_cache()
    print(f"[live_synthetic] stopped at step {n}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["matmul", "membound"])
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--status-file", default=DEFAULT_STATUS_FILE)
    a = ap.parse_args()
    run(a.mode, a.minutes, a.status_file)
