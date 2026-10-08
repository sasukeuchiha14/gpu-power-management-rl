"""
live_sparse_gather.py -- continuous version of pt_sparse_gather.py for live
A/B testing against common/deploy_live.py, same status-file contract as
live_resnet50_train.py / live_glmark2_render.py.

Why this workload specifically: every other "live_" benchmark in this repo
is either compute-bound (live_resnet50_train.py, dense GEMM-heavy) or a
real rendering pipeline (live_glmark2_render.py). Neither stresses the
memory subsystem the way irregular-access workloads do. This repeats
pt_sparse_gather.py's random-index gather (SpMV/graph-traversal-style
access, distinct from vector_add's sequential access) in a loop, reporting
throughput every WINDOW gathers -- giving the live controller a genuinely
different GPU load shape (memory-bandwidth-bound, not compute-bound) to be
tested against, same as the offline profiling suite already treats it as
its own category.

    python common/benchmarks/live_sparse_gather.py --minutes 15
    python common/benchmarks/live_sparse_gather.py --minutes 15 --status-file /tmp/dvfs_workload_status.json
"""
import argparse
import gc
import json
import os
import signal
import time

import torch

DEFAULT_STATUS_FILE = "/tmp/dvfs_workload_status.json"
WINDOW = 200  # gathers averaged per throughput report -- each gather is fast, so this window is larger than the training one


def write_status(path, step, ms_per_iter, iters_per_sec):
    """Atomic write (temp file + rename), same contract as
    live_resnet50_train.py.write_status -- deploy_live.py reads this file
    regardless of which workload produced it."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "step": step,
            "ms_per_iter": ms_per_iter,
            "iters_per_sec": iters_per_sec,
        }, f)
    os.replace(tmp, path)


def run(minutes, status_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available!")

    n = 16_000_000
    table = torch.randn(n, device=device)
    idx_pool = [torch.randint(0, n, (n,), device=device) for _ in range(5)]

    stop = {"flag": False}

    def handle_exit(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Warmup -- not timed, same reasoning as every offline benchmark.
    for i in range(5):
        _ = table[idx_pool[i % len(idx_pool)]]
    torch.cuda.synchronize()

    print(f"[live_sparse_gather] running for {minutes} min, "
          f"reporting every {WINDOW} gathers -> {status_file}", flush=True)

    deadline = time.time() + minutes * 60.0
    step = 0
    window_start = time.perf_counter()

    while time.time() < deadline and not stop["flag"]:
        _ = table[idx_pool[step % len(idx_pool)]]
        step += 1

        if step % WINDOW == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - window_start
            ms_per_iter = 1000.0 * elapsed / WINDOW
            iters_per_sec = WINDOW / elapsed
            write_status(status_file, step, ms_per_iter, iters_per_sec)
            window_start = time.perf_counter()

    del table, idx_pool
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[live_sparse_gather] stopped at step {step}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--status-file", default=DEFAULT_STATUS_FILE)
    a = ap.parse_args()
    run(a.minutes, a.status_file)


if __name__ == "__main__":
    main()
