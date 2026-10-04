"""
live_resnet50_train.py -- Phase 2 hardware-validation workload (context.md's
"IMMEDIATE NEXT STEP" / NORTHSTAR row 11: "Deploy ... Use a workload with a
natural live work-rate signal").

Every other benchmark in this repo (gpus/*/benchmarks/pt_*.py) is one-shot: run
N iterations, print a duration, exit. That's fine for OFFLINE profiling (one row
per (workload, core, mem) config) but useless for a LIVE controller, which needs
to watch a workload that's still running and react to it in real time.

This runs ResNet-50 training continuously and, every WINDOW iterations, writes
the measured throughput to a small JSON status file. The live deployment daemon
(common/deploy_live.py) tails that file each control tick to get a live analog
of "runtime" for the response features (common/features.py:response_features) --
ms/iteration here plays the role Avg_Time_S played in the offline profiled
tables, just measured continuously instead of once per config.

    python common/benchmarks/live_resnet50_train.py --minutes 15
    python common/benchmarks/live_resnet50_train.py --minutes 15 --status-file /tmp/dvfs_workload_status.json
"""
import argparse
import gc
import json
import os
import signal
import sys
import time

import torch
import torch.nn as nn
from torchvision.models import resnet50

DEFAULT_STATUS_FILE = "/tmp/dvfs_workload_status.json"
WINDOW = 20  # iterations averaged per throughput report


def write_status(path, step, ms_per_iter, iters_per_sec):
    """Atomic write (temp file + rename) so the daemon never reads a partial file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "step": step,
            "ms_per_iter": ms_per_iter,
            "iters_per_sec": iters_per_sec,
        }, f)
    os.replace(tmp, path)


def run(minutes, batch_size, status_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available!")

    model = resnet50().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    data = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, 1000, (batch_size,), device=device)

    stop = {"flag": False}

    def handle_exit(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Warmup -- not timed, lets CUDA context / cuDNN autotune settle before
    # the first throughput window, same reasoning as every offline benchmark.
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(model(data), targets)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    print(f"[live_resnet50_train] running for {minutes} min, batch={batch_size}, "
          f"reporting every {WINDOW} iters -> {status_file}", flush=True)

    deadline = time.time() + minutes * 60.0
    step = 0
    window_start = time.perf_counter()

    while time.time() < deadline and not stop["flag"]:
        optimizer.zero_grad()
        loss = criterion(model(data), targets)
        loss.backward()
        optimizer.step()
        step += 1

        if step % WINDOW == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - window_start
            ms_per_iter = 1000.0 * elapsed / WINDOW
            iters_per_sec = WINDOW / elapsed
            write_status(status_file, step, ms_per_iter, iters_per_sec)
            window_start = time.perf_counter()

    del model, data, targets
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[live_resnet50_train] stopped at step {step}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="32 is safe on an 8GB+ desktop card; the laptop profiling runs used 16 for 4GB VRAM")
    ap.add_argument("--status-file", default=DEFAULT_STATUS_FILE)
    a = ap.parse_args()
    run(a.minutes, a.batch_size, a.status_file)


if __name__ == "__main__":
    main()
