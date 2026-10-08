"""
live_demo.py -- a LIVE, watchable terminal dashboard for showing the DVFS
controller's energy savings to an audience in real time. Different from
demo_power_trace.sh (which records two fixed 90s phases and plots a static
chart afterward, docs/demo_power_trace.html): this redraws a live chart in
the terminal every tick, continuously, so it can be screen-shared during a
live demo and watched update as it happens -- no post-processing step.

Runs on whatever machine has the GPU and clock-lock permission (almost
always the RENTED GPU, not a laptop -- see the permission note below).
Alternates automatically between two phases, forever, until Ctrl+C:

  WITHOUT  -- clocks pinned at max (no management, the "before" case)
  WITH     -- the trained response-aware DDQN controlling clocks live
              (reuses deploy_live.py's LiveController directly, not a
              reimplementation, so the demo is showing the real controller)

Each tick samples real NVML power, redraws a live terminal chart (pure
ANSI/Unicode block characters, zero external dependencies -- deliberately
NOT using a terminal-charting library: this needs to work unmodified on
whatever rented box is available for a live demo, without debugging an
unfamiliar package's API under audience-watching pressure), and keeps a
running "energy saved so far" statistic computed from the actual measured
phase means, updated live.

Usage (on the GPU machine, same checklist as every other live run in this
project -- gpu_probe.py USABLE check first):

    # terminal 1
    python3 common/benchmarks/live_resnet50_train.py --minutes 60 --batch-size 4

    # terminal 2 (the live demo -- this is what you screen-share)
    python3 common/live_demo.py --weights common/weights/reversal_cost_030_fix_holdout_nvidia_geforce_rtx_4090.pth --live

Permission note: needs the same clock-lock permission every other script in
this project needs (gpu_probe.py USABLE) -- a laptop's own integrated/
consumer GPU usually doesn't have this on a managed OS, and a rented
container-type cloud instance usually doesn't either (see METHODOLOGY.md
§5.3). A rented VM-type instance, or a physical Linux desktop/workstation
the presenter administers directly, both work. --live is required to
actually lock clocks; without it this runs in a safe dry-run preview mode
(still live-charts, just never touches hardware).

Tested live on a rented RTX 4090: short phases are too short for the policy
to settle and give noisy, sometimes backwards-looking numbers -- a 15s-phase
test showed -7% ("savings" in the wrong direction), pure settling noise.
The default 45s phases settle to a stable, correctly-signed +10% on the
same checkpoint and workload condition this project's own hardware-
validated launch-bound result used (that result, over longer 90s phases,
measured +18.6% -- same direction, same order of magnitude). Don't shorten
--without-seconds/--with-seconds much below 45 for a demo anyone will
actually trust the number from.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

import pynvml

_common = os.path.dirname(os.path.abspath(__file__))
if _common not in sys.path:
    sys.path.insert(0, _common)
from gpu_probe import hardware_limits  # noqa: E402

WITHOUT_SECONDS = 45   # phase duration -- short enough to see several
WITH_SECONDS = 45      # alternations in a normal demo window
SAMPLE_HZ = 2           # power samples per second
HISTORY_SECONDS = 240   # how much of the trace stays on screen at once

RED, GREEN, GRAY, BOLD, RESET = "\033[31m", "\033[32m", "\033[90m", "\033[1m", "\033[0m"
BLOCKS = " ▁▂▃▄▅▆▇█"  # 8 levels + blank, for a vertical-bar chart


class PhaseStats:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def add(self, power_w):
        self.sum += power_w
        self.n += 1

    @property
    def mean(self):
        return self.sum / self.n if self.n else 0.0


def clear_screen():
    print("\033[H\033[2J", end="")  # move cursor home + clear, avoids the
                                      # full terminal-reset flicker \033c causes


def draw_chart(powers, phases):
    """Dependency-free vertical-bar chart: one column per sample, height by
    power, coloured red (WITHOUT) or green (WITH). Renders cleanly over SSH,
    no GUI, no extra package to install on a rented box mid-demo."""
    if not powers:
        return
    width = max(10, min(shutil.get_terminal_size((100, 24)).columns - 4, len(powers)))
    sample_p = powers[-width:]
    sample_ph = phases[-width:]
    lo, hi = min(sample_p), max(sample_p)
    span = max(hi - lo, 1e-6)
    rows = 14
    grid = [[" " for _ in sample_p] for _ in range(rows)]
    for col, (p, ph) in enumerate(zip(sample_p, sample_ph)):
        filled_rows = (p - lo) / span * rows
        full = int(filled_rows)
        frac = filled_rows - full
        color = RED if ph == "WITHOUT" else GREEN
        for r in range(full):
            grid[rows - 1 - r][col] = color + "█" + RESET
        if full < rows:
            idx = max(1, min(8, int(round(frac * 8))))
            grid[rows - 1 - full][col] = color + BLOCKS[idx] + RESET

    print(f"  {hi:6.0f}W ┤")
    for row in grid:
        print("          │" + "".join(row))
    print(f"  {lo:6.0f}W ┴" + "─" * width)
    print(f"  {RED}■{RESET} WITHOUT (pinned max)    {GREEN}■{RESET} WITH (DDQN controller)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--live", action="store_true", help="actually lock clocks (default: dry-run preview)")
    ap.add_argument("--without-seconds", type=int, default=WITHOUT_SECONDS)
    ap.add_argument("--with-seconds", type=int, default=WITH_SECONDS)
    a = ap.parse_args()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(a.gpu_index)
    hw = hardware_limits(handle)
    name = pynvml.nvmlDeviceGetName(handle)

    times, powers, phases = [], [], []
    t0 = time.time()
    stats = {"WITHOUT": PhaseStats(), "WITH": PhaseStats()}
    deploy_proc = None
    current_phase = None
    phase_deadline = 0.0
    cleaned_up = False

    def cleanup(*_):
        nonlocal deploy_proc, cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        print(f"\n{GRAY}[live_demo] resetting clocks and exiting...{RESET}")
        if deploy_proc is not None:
            deploy_proc.send_signal(signal.SIGINT)
            try:
                deploy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                deploy_proc.kill()
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        except (AttributeError, pynvml.NVMLError):
            subprocess.run(["nvidia-smi", "-rgc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    def start_without():
        nonlocal deploy_proc
        if deploy_proc is not None:
            deploy_proc.send_signal(signal.SIGINT)
            deploy_proc.wait(timeout=5)
            deploy_proc = None
        if a.live:
            try:
                pynvml.nvmlDeviceSetGpuLockedClocks(handle, int(hw["max_core_mhz"]), int(hw["max_core_mhz"]))
            except (AttributeError, pynvml.NVMLError):
                subprocess.run(["nvidia-smi", "-lgc",
                                f"{int(hw['max_core_mhz'])},{int(hw['max_core_mhz'])}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def start_with():
        nonlocal deploy_proc
        cmd = [sys.executable, "-u", os.path.join(_common, "deploy_live.py"),
               "--weights", a.weights, "--gpu-index", str(a.gpu_index)]
        if a.live:
            cmd.append("--live")
        deploy_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"[live_demo] {name} | max_core={hw['max_core_mhz']:.0f}MHz | "
          f"{'LIVE (will lock real clocks)' if a.live else 'DRY RUN (preview only)'}")
    print("[live_demo] starting in 2s -- Ctrl+C any time to stop and reset clocks cleanly.")
    time.sleep(2)

    try:
        while True:
            now = time.time() - t0
            if current_phase is None or now >= phase_deadline:
                current_phase = "WITHOUT" if current_phase != "WITHOUT" else "WITH"
                phase_deadline = now + (a.without_seconds if current_phase == "WITHOUT" else a.with_seconds)
                if current_phase == "WITHOUT":
                    start_without()
                else:
                    start_with()

            power = float(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            times.append(now)
            powers.append(power)
            phases.append(current_phase)
            stats[current_phase].add(power)
            cutoff = now - HISTORY_SECONDS
            while times and times[0] < cutoff:
                times.pop(0); powers.pop(0); phases.pop(0)

            clear_screen()
            phase_color = RED if current_phase == "WITHOUT" else GREEN
            print(f"{BOLD}{'=' * 70}{RESET}")
            print(f"{BOLD}  LIVE DVFS CONTROLLER DEMO  --  {name}{RESET}")
            print(f"  phase: {phase_color}{BOLD}{current_phase}{RESET}   "
                  f"({phase_deadline - now:4.0f}s left this phase)   "
                  f"{'[LIVE]' if a.live else '[DRY RUN]'}")
            print(f"{BOLD}{'=' * 70}{RESET}\n")
            draw_chart(powers, phases)

            w_mean, a_mean = stats["WITHOUT"].mean, stats["WITH"].mean
            print(f"\n  current power:        {power:6.1f} W")
            print(f"  WITHOUT mean so far:  {w_mean:6.1f} W   (n={stats['WITHOUT'].n})")
            print(f"  WITH mean so far:     {a_mean:6.1f} W   (n={stats['WITH'].n})")
            if w_mean > 0 and stats["WITH"].n > 0:
                saved = 100.0 * (w_mean - a_mean) / w_mean
                color = GREEN if saved > 0 else RED
                print(f"  {color}{BOLD}>>> ENERGY SAVED SO FAR: {saved:+.1f}% <<<{RESET}")
            print(f"\n  {GRAY}Ctrl+C to stop (clocks reset automatically){RESET}")

            time.sleep(1.0 / SAMPLE_HZ)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
