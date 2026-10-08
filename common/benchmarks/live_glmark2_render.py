"""
live_glmark2_render.py -- a real GPU-RENDERING workload for the live demo
(common/live_demo.py), not an ML training workload. Same status-file
contract as live_resnet50_train.py (common/deploy_live.py doesn't care
which workload is feeding it, only that the file matches the contract), so
deploy_live.py / live_demo.py work completely unmodified against this.

Why glmark2-drm specifically: it's a real OpenGL rendering benchmark --
vertex/fragment shaders, textures, draw calls -- a genuinely different GPU
load shape than training a neural net, closer to what a game's GPU load
actually looks like. The `-drm` variant renders via DRM/GBM directly, no
X11 or Wayland session needed, which is what makes this work headless on a
rented cloud GPU with no display attached (the plain X11 `glmark2` binary
fails outright on a headless box -- confirmed directly: "Error: main:
Could not initialize canvas" -- `glmark2-drm` renders successfully against
the real GPU, confirmed `GL_RENDERER: NVIDIA GeForce RTX 4090`).

Runs the default glmark2 benchmark suite (varied rendering scenes: build,
texture, shading, bump, effect2d, pulsar, desktop, buffer, jellyfish,
terrain, shadow, refract, ...) on a loop via --run-forever, parses each
scene's "FPS: N FrameTime: M ms" completion line as it streams, and writes
it to the status file as one "step" -- the same role an iteration plays in
the ML-training workload, just reported once per finished scene (every few
seconds) instead of every N training iterations.

Install first (Ubuntu/Debian):
    apt-get install -y glmark2-drm

Usage:
    python3 common/benchmarks/live_glmark2_render.py --minutes 15
    python3 common/benchmarks/live_glmark2_render.py --minutes 15 --status-file /tmp/dvfs_workload_status.json
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

DEFAULT_STATUS_FILE = "/tmp/dvfs_workload_status.json"
FPS_LINE = re.compile(r"FPS:\s*(\d+)\s*FrameTime:\s*([\d.]+)\s*ms")


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
    if shutil.which("glmark2-drm") is None:
        raise RuntimeError("glmark2-drm not found -- install it first: apt-get install -y glmark2-drm")

    deadline = time.time() + minutes * 60
    step = 0
    stop = {"flag": False}

    def handle_exit(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    print(f"[live_glmark2_render] running for {minutes:.1f} min, "
          f"reporting every finished scene -> {status_file}", flush=True)

    while time.time() < deadline and not stop["flag"]:
        proc = subprocess.Popen(
            ["glmark2-drm", "--run-forever"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        try:
            for line in proc.stdout:
                if stop["flag"] or time.time() >= deadline:
                    break
                m = FPS_LINE.search(line)
                if not m:
                    continue
                fps, frame_ms = float(m.group(1)), float(m.group(2))
                step += 1
                write_status(status_file, step, frame_ms, fps)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    print(f"[live_glmark2_render] stopped at step {step}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--status-file", default=DEFAULT_STATUS_FILE)
    a = ap.parse_args()
    run(a.minutes, a.status_file)


if __name__ == "__main__":
    main()
