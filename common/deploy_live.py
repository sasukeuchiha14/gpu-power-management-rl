"""
deploy_live.py -- Phase 2 hardware validation: run the trained response-aware
DDQN as a real control loop against a live NVML GPU, driving
common/benchmarks/live_resnet50_train.py (or any workload that writes the same
status-file schema).

This is the piece context.md's Phase 2 plan calls for: "deploy the trained
response-aware policy live on a rented GPU... so the 'measured response' is
real, not simulated." Everywhere else in this repo, `response_features()` is
computed from two rows of a profiled table (common/eval_policies.augmented_obs,
common/unified_env.py). Here there is no table -- `_read_telemetry()` below
and the workload's live throughput file ARE the two measurements being
compared, taken for real, seconds apart, on physical hardware.

Usage (on the rented machine, as root/with CUDA + NVIDIA driver):

    # terminal 1
    python common/benchmarks/live_resnet50_train.py --minutes 20

    # terminal 2 (dry run first -- ALWAYS watch a dry run's logs before --live)
    python common/deploy_live.py --weights common/weights/response_aware_holdout_nvidia_geforce_rtx_4090.pth

    # only after the dry run looks sane:
    python common/deploy_live.py --weights ... --live

SAFETY: clocks are always reset to automatic on exit (Ctrl+C, SIGTERM, or any
uncaught exception) -- see `finally` in main(). DRY_RUN is the default; --live
is required to actually lock clocks.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque

import numpy as np
import torch
import pynvml

_common = os.path.dirname(os.path.abspath(__file__))
if _common not in sys.path:
    sys.path.insert(0, _common)
_arbiter_dir = os.path.abspath(os.path.join(_common, "..", "gpus", "rtx3050ti_laptop", "deployment"))
if _arbiter_dir not in sys.path:
    sys.path.insert(0, _arbiter_dir)

from features import FEATURES, STATE_DIM, build_features, response_features  # noqa: E402
from train_unified_v2 import QNet, GreedyAgent  # noqa: E402
from gpu_probe import hardware_limits  # noqa: E402
from arbiter import DoNoHarmArbiter  # noqa: E402

LOOP_MS = 200           # control cadence; the workload's own throughput window
                        # (common/benchmarks/live_resnet50_train.py, WINDOW=20
                        # iterations) is typically 1-3s on a desktop card, so
                        # most ticks will see no fresh measurement yet -- see
                        # _wait_for_fresh_status().
STATUS_STALE_S = 10.0   # if the workload hasn't reported in this long, assume
                        # it died and force max clock (safety, not a policy decision)
TEMP_LIMIT_C_MARGIN = 3.0   # force-cool this many degrees under the probed slowdown threshold
# Normalised core stride. Training inferred this per-workload from the
# profiled grid's own level spacing (~0.13-0.14 across the fleet, see
# common/dataset_view.py); live deployment has no profiled grid to infer
# from, so this is a fixed approximation of that same spacing, not a
# re-derivation. Flag this if results look coarser/finer than expected.
CORE_STRIDE_FRAC = 0.125
ARB_PATIENCE = 3
ARB_COOLDOWN = 5
ARB_UTIL_THRESHOLD = 85.0
# Debounce (context.md "IMMEDIATE NEXT STEP", 5-7 Oct 2026): a deployment-side
# attempt at the limit-cycle problem (a trained policy genuinely oscillating
# between two adjacent clock levels, confirmed via deterministic rollout --
# see common/analysis/reward_shape_sweep.py). Two reward-shaping fixes were
# tried first and didn't hold up (reward_power made it worse; action_cost
# looked good at 3 seeds but regressed at 10). This targets the SYMPTOM
# instead of the reward cause: require the policy to choose the same
# non-HOLD action N ticks in a row before a clock change actually commits.
#
# VALIDATED OFFLINE, DOES NOT WORK (common/analysis/debounce_validate.py):
# debouncing does not break the cycle, it only STRETCHES ITS PERIOD. At a
# frozen setpoint the policy sees an identical ("neutral response") input
# every tick, so it keeps re-choosing the same action until the streak hits
# N, commits one step, flips the response sign, and the same thing happens
# in the other direction -- confirmed directly by inspecting the rollout
# traces at N=1/3/10 (period 2, then ~6, then ~18-20). Checked out to N=40
# over a 300-step rollout: still 9/12 workloads cycling, same count as
# N=3-10, with the aggregate steady-state metric getting WORSE as N grows
# (0.079 at N=1 to ~0.12 at N=10) because the forced delay also slows
# genuinely beneficial moves. There is no N that fixes this; larger N only
# trades oscillation frequency for responsiveness. Left here, default OFF
# (N=1), as a documented negative result and because a long streak
# requirement might still be worth trying deliberately in a follow-up
# (e.g. combined with a smaller action stride) -- not because it's expected
# to work as-is.
DEBOUNCE_N = 1


def _wait_for_fresh_status(status_file, last_step, timeout_s):
    """Block until the workload's status file reports a step newer than
    `last_step`, or timeout. Ensures the ms_per_iter we read actually reflects
    at least one full measurement window run under the CURRENT clock setting
    -- reading on a fixed timer instead would mix pre- and post-change
    samples into one 'response' measurement."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with open(status_file) as f:
                s = json.load(f)
            if s["step"] > last_step and (time.time() - s["timestamp"]) < STATUS_STALE_S:
                return s
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        time.sleep(LOOP_MS / 1000.0)
    return None


class LiveController:
    def __init__(self, weights_path, status_file, gpu_index=0, live=False, debounce_n=DEBOUNCE_N):
        self.live = live
        self.status_file = status_file

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.hw = hardware_limits(self.handle)  # ALWAYS probed live -- never hardcoded,
                                                 # see features.py's unit-to-unit variability note
        name = pynvml.nvmlDeviceGetName(self.handle)
        print(f"[deploy_live] {name} | hw limits: {self.hw}")

        self.core_idx = FEATURES.index("core_frac")
        self.net = QNet(3 * STATE_DIM, 3, hidden=128, dueling=False)
        self.net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.net.eval()
        self.agent = GreedyAgent(self.net)

        self.arbiter = DoNoHarmArbiter(
            util_threshold=ARB_UTIL_THRESHOLD, patience=ARB_PATIENCE, cooldown=ARB_COOLDOWN,
        )

        self.core_stride_mhz = CORE_STRIDE_FRAC * self.hw["max_core_mhz"]
        self.current_core_mhz = self.hw["max_core_mhz"]  # start at max, like every offline episode start option
        self.debounce_n = debounce_n
        self.pending_action = None
        self.pending_count = 0
        self.last_step = -1
        self.prev_core_frac = None
        self.prev_ms_per_iter = None
        self.prev_power_w = None
        self.stack = deque(maxlen=3)

    def _read_telemetry(self):
        core_clk = float(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM))
        pwr = float(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
        temp = float(pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU))
        utl = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return dict(core_mhz=core_clk, power_w=pwr, temp_c=temp,
                    gpu_util=utl.gpu, mem_util=utl.memory,
                    vram_mb=mem.used / (1024.0 * 1024.0))

    def set_clocks(self, target_mhz):
        self.current_core_mhz = float(np.clip(target_mhz, 1, self.hw["max_core_mhz"]))
        if not self.live:
            return
        try:
            pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, int(self.current_core_mhz), int(self.current_core_mhz))
        except (AttributeError, pynvml.NVMLError):
            subprocess.run(["nvidia-smi", "-lgc", f"{int(self.current_core_mhz)},{int(self.current_core_mhz)}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def debounce(self, action):
        """Returns the action to actually act on: `action` once it has been
        chosen `self.debounce_n` ticks in a row, HOLD otherwise. A policy that
        alternates between two actions every tick never reaches the streak
        length, so the clock stays put instead of cycling."""
        if action == self.pending_action:
            self.pending_count += 1
        else:
            self.pending_action = action
            self.pending_count = 1
        if action == 2 or self.pending_count >= self.debounce_n:
            return action
        return 2  # HOLD -- streak not long enough yet to commit

    def reset_clocks(self):
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)
        except (AttributeError, pynvml.NVMLError):
            subprocess.run(["nvidia-smi", "-rgc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def tick(self):
        fresh = _wait_for_fresh_status(self.status_file, self.last_step, timeout_s=STATUS_STALE_S)
        if fresh is None:
            print("[deploy_live] workload status stale/missing -- forcing max clock (safety)")
            self.set_clocks(self.hw["max_core_mhz"])
            return
        self.last_step = fresh["step"]

        raw = self._read_telemetry()
        base = build_features(raw["core_mhz"], raw["gpu_util"], raw["mem_util"],
                               raw["power_w"], raw["temp_c"], raw["vram_mb"], self.hw)
        core_frac_now = base[self.core_idx]
        resp = response_features(self.prev_core_frac, core_frac_now,
                                  self.prev_ms_per_iter, fresh["ms_per_iter"],
                                  self.prev_power_w, raw["power_w"])
        self.prev_core_frac, self.prev_ms_per_iter, self.prev_power_w = (
            core_frac_now, fresh["ms_per_iter"], raw["power_w"])

        obs = np.array(base + list(resp), dtype=np.float32)
        if not self.stack:
            for _ in range(3):
                self.stack.append(obs)
        else:
            self.stack.append(obs)
        state = np.array(self.stack, dtype=np.float32)

        # Layer 1: hard thermal override
        if raw["temp_c"] >= self.hw["slowdown_temp_c"] - TEMP_LIMIT_C_MARGIN:
            print(f"[THERMAL] {raw['temp_c']:.1f}C -- forcing min-ish clock")
            self.set_clocks(self.hw["max_core_mhz"] * 0.3)
            return

        raw_action = self.agent.act(state)
        action = self.debounce(raw_action)
        if action == 0:
            requested = self.current_core_mhz - self.core_stride_mhz
        elif action == 1:
            requested = self.current_core_mhz + self.core_stride_mhz
        else:
            requested = self.current_core_mhz

        # Layer 2: do-no-harm arbiter
        target, overridden, tripped = self.arbiter.decide(
            sm_util_pct=raw["gpu_util"], tracked_core_mhz=self.current_core_mhz,
            core_max_mhz=self.hw["max_core_mhz"], requested_core_mhz=requested,
        )
        self.set_clocks(target)

        tag = "[LIVE]" if self.live else "[DRY_RUN]"
        arb = " [ARBITER]" if tripped or overridden else ""
        deb = f" [DEBOUNCED {['DEC','INC','HOLD'][raw_action]}->HOLD]" if action != raw_action else ""
        print(f"{tag} a={['DEC','INC','HOLD'][action]:<4} target={self.current_core_mhz:6.0f}MHz "
              f"observed={raw['core_mhz']:6.0f}MHz {raw['power_w']:5.1f}W {raw['temp_c']:4.1f}C "
              f"util={raw['gpu_util']:3.0f}% ms/iter={fresh['ms_per_iter']:6.2f} "
              f"resp=({resp[0]:.2f},{resp[1]:.2f}){arb}{deb}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--status-file", default="/tmp/dvfs_workload_status.json")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--live", action="store_true", help="actually lock clocks (default: observe only)")
    ap.add_argument("--debounce-n", type=int, default=DEBOUNCE_N,
                    help="consecutive identical non-HOLD actions required before a clock "
                         "change commits (limit-cycle mitigation; 1 = no debounce)")
    a = ap.parse_args()

    controller = LiveController(a.weights, a.status_file, a.gpu_index, live=a.live,
                                 debounce_n=a.debounce_n)

    def handle_exit(signum, frame):
        print("\n[deploy_live] resetting clocks and exiting...")
        controller.reset_clocks()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        while True:
            controller.tick()
            time.sleep(LOOP_MS / 1000.0)
    finally:
        controller.reset_clocks()


if __name__ == "__main__":
    main()
