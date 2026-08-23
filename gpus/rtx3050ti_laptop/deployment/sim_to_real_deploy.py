import json
import signal
import subprocess
import sys
import time
import ctypes
import os
from collections import deque

import numpy as np
import torch
import pynvml

# Ensure models/ is on sys.path for DDQNAgent import
script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
models_dir = os.path.join(gpu_root, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from ddqn_agent import DDQNAgent
from arbiter import DoNoHarmArbiter

def _resolve(subfolder, filename):
    cand = os.path.join(gpu_root, subfolder, filename)
    if os.path.exists(cand):
        return cand
    if os.path.exists(filename):
        return filename
    return cand

# ============================== CONFIG =======================================
DRY_RUN = True              # True = observe only, no real clock changes. Set False only after watching a
                             # DRY_RUN session's logs look sane -- see CONTRIBUTING.md. The Layer 2 arbiter
                             # added 25 Aug 2026 has only been tested against a mocked GPU, never live.
GPU_INDEX = 0
LOOP_MS = 50               # control loop cadence (50ms budget)
TEMP_LIMIT_C = 90          # hard thermal safety override for laptop (Layer 1, set to 90C)
STACK_SIZE = 3             # matching environment stack size
WEIGHTS = _resolve("weights", "ddqn_weights.pth")

# Layer 2: do-no-harm performance arbiter. See arbiter.py's module docstring
# for why this watches utilisation rather than reconstructing EDP the way
# the old RTX 4500 arbiter did -- that approach needs workload-identifying
# lookups this design deliberately removed.
ARBITER_ENABLE = True
ARB_PATIENCE = 3
ARB_COOLDOWN = 5
ARB_UTIL_THRESHOLD = 85.0   # % SM utilisation considered "compute-bound, at risk of real slowdown"

CORE_MIN_MHZ = 500
CORE_MAX_MHZ = 1900
CORE_STRIDE_MHZ = 100
# =============================================================================

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False

class LiveController:
    def __init__(self):
        if not DRY_RUN and not is_admin():
            print("\n" + "="*80)
            print("WARNING: You must run this command from an ADMINISTRATOR terminal to change GPU clocks!")
            print("DRY_RUN mode will be enforced automatically.")
            print("="*80 + "\n")
            globals()["DRY_RUN"] = True

        # Load hardware limits
        scaler_path = _resolve("data", "scaler_params.json")
        with open(scaler_path) as f:
            sc = json.load(f)
        self.hw = sc.get("hardware_limits", {
            "max_core_mhz": 1900.0,
            "power_limit_w": 70.0,
            "slowdown_temp_c": 87.0,
            "total_vram_mb": 4096.0
        })

        # Initialize NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)

        # 6 Universal Features (core_frac, sm_util, mem_util, power_frac, temp_frac, vram_frac)
        self.n_features = 6
        self.stack = deque(maxlen=STACK_SIZE)

        # Load Trained DDQN Model
        self.agent = DDQNAgent((STACK_SIZE, self.n_features), action_size=3)
        try:
            self.agent.policy_net.load_state_dict(
                torch.load(WEIGHTS, map_location=self.agent.device, weights_only=True)
            )
            self.agent.policy_net.eval()
            self.agent.epsilon = 0.0
            print(f"[OK] Loaded trained weights from {WEIGHTS}")
        except FileNotFoundError:
            print(f"[WARNING] Weights {WEIGHTS} not found. Running with UNTRAINED model.")

        self.current_core_mhz = CORE_MAX_MHZ
        self.arbiter = DoNoHarmArbiter(
            util_threshold=ARB_UTIL_THRESHOLD,
            patience=ARB_PATIENCE,
            cooldown=ARB_COOLDOWN,
            enabled=ARBITER_ENABLE,
        )

        # Prime the frame stack with live sensor readings
        init_obs = self._read_telemetry()
        for _ in range(STACK_SIZE):
            self.stack.append(init_obs)

    def _read_telemetry(self):
        """Reads live NVML sensors and normalizes into the 6 universal features."""
        try:
            core_clk = float(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM))
            pwr = float(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
            temp = float(pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU))
            utl = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            core_frac = np.clip(core_clk / self.hw["max_core_mhz"], 0.0, 1.0)
            sm_util = np.clip(utl.gpu / 100.0, 0.0, 1.0)
            mem_util = np.clip(utl.memory / 100.0, 0.0, 1.0)
            pwr_frac = np.clip(pwr / self.hw["power_limit_w"], 0.0, 1.0)
            temp_frac = np.clip(temp / self.hw["slowdown_temp_c"], 0.0, 1.0)
            vram_frac = np.clip((mem.used / (1024.0 * 1024.0)) / self.hw["total_vram_mb"], 0.0, 1.0)

            self.last_raw = {
                "core_mhz": core_clk,
                "power_w": pwr,
                "temp_c": temp,
                "gpu_util": utl.gpu,
                "mem_util": utl.memory,
                "vram_mb": mem.used / (1024.0 * 1024.0)
            }
            return np.array([core_frac, sm_util, mem_util, pwr_frac, temp_frac, vram_frac], dtype=np.float32)
        except Exception as e:
            return np.zeros(self.n_features, dtype=np.float32)

    def set_clocks(self, target_core_mhz):
        """Set locked core frequency via NVML or nvidia-smi."""
        self.current_core_mhz = int(np.clip(target_core_mhz, CORE_MIN_MHZ, CORE_MAX_MHZ))
        if DRY_RUN:
            return

        try:
            # Fast in-process NVML call
            pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, self.current_core_mhz, self.current_core_mhz)
        except (AttributeError, pynvml.NVMLError):
            # Fallback to subprocess
            subprocess.run(
                ["nvidia-smi", "-i", "0", "-lgc", f"{self.current_core_mhz},{self.current_core_mhz}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    def reset_clocks(self):
        """Restore automatic GPU factory clock governor."""
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)
        except (AttributeError, pynvml.NVMLError):
            subprocess.run(["nvidia-smi", "-i", "0", "-rgc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def step(self):
        t0 = time.perf_counter()

        # 1. Ingest live telemetry
        obs = self._read_telemetry()
        self.stack.append(obs)
        state = np.array(self.stack, dtype=np.float32)

        # 2. Hard Safety Net (Layer 1)
        if self.last_raw["temp_c"] >= TEMP_LIMIT_C:
            print(f"[THERMAL OVERRIDE] Temp {self.last_raw['temp_c']:.1f}C >= {TEMP_LIMIT_C}C. Cooling down!")
            self.set_clocks(CORE_MIN_MHZ)
            return

        # 3. Choose action from universal DDQN policy
        action = self.agent.act(state)
        action_names = {0: "DEC_CORE", 1: "INC_CORE", 2: "HOLD"}

        # 4. Apply Action
        if action == 0:
            requested_core = self.current_core_mhz - CORE_STRIDE_MHZ
        elif action == 1:
            requested_core = self.current_core_mhz + CORE_STRIDE_MHZ
        else:
            requested_core = self.current_core_mhz

        # Layer 2: do-no-harm arbiter can override the policy's requested
        # clock (see arbiter.py). Evaluated against the clock that was
        # actually in effect going into this tick, before the policy's move.
        target_core, arb_overridden, arb_tripped = self.arbiter.decide(
            sm_util_pct=self.last_raw["gpu_util"],
            tracked_core_mhz=self.current_core_mhz,
            core_max_mhz=CORE_MAX_MHZ,
            requested_core_mhz=requested_core,
        )

        self.set_clocks(target_core)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Log status
        mode_tag = "[DRY_RUN]" if DRY_RUN else "[LIVE]"
        arb_tag = " [ARBITER TRIPPED]" if arb_tripped else (" [arbiter hold]" if arb_overridden else "")
        print(
            f"{mode_tag} Action: {action_names[action]:<8} | "
            f"Target Core: {self.current_core_mhz:4d} MHz | "
            f"Observed: {self.last_raw['core_mhz']:4.0f} MHz, {self.last_raw['power_w']:5.1f} W, "
            f"{self.last_raw['temp_c']:4.1f} C, GPU: {self.last_raw['gpu_util']:3.0f}%, "
            f"Mem-Busy: {self.last_raw['mem_util']:3.0f}% | "
            f"Tick: {elapsed_ms:4.1f} ms{arb_tag}"
        )

def main():
    print("="*80)
    print(" NVIDIA RTX 3050 Ti Live DVFS RL Controller (Universal 6-Feature Engine)")
    print("="*80)
    print(f" Cadence: {LOOP_MS} ms | Dry Run: {DRY_RUN} | Thermal Limit: {TEMP_LIMIT_C} C")
    print(" Press Ctrl+C to safely exit and restore automatic GPU governor.")
    print("="*80 + "\n")

    controller = LiveController()

    def handle_exit(signum, frame):
        print("\n\nStopping controller and resetting GPU clocks to automatic defaults...")
        controller.reset_clocks()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        print("Done. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    while True:
        controller.step()
        time.sleep(LOOP_MS / 1000.0)

if __name__ == "__main__":
    main()
