import json
import signal
import subprocess
import sys
import time
import ctypes
from collections import deque

import numpy as np
import pandas as pd
import torch
import pynvml

import os
import sys

# Ensure models/ is on sys.path for DDQNAgent import
script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
models_dir = os.path.join(gpu_root, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from ddqn_agent import DDQNAgent

def _resolve(subfolder, filename):
    cand = os.path.join(gpu_root, subfolder, filename)
    if os.path.exists(cand):
        return cand
    if os.path.exists(filename):
        return filename
    return cand

# ============================== CONFIG =======================================
DRY_RUN = True             # True = observe only, never change clocks. FLIP TO FALSE TO ACTUATE.
GPU_INDEX = 0
LOOP_MS = 50               # control loop cadence (50ms budget)
TEMP_LIMIT_C = 80          # lowered hard thermal safety override for laptop (Layer 1)
STACK_SIZE = 3             # matching environment stack size
WEIGHTS = _resolve("weights", "ddqn_weights.pth")
ONLINE_FINETUNE = False

# Layer 2: do-no-harm performance arbiter. If estimated EDP proxy exceeds governor,
# revert to standard automatic governor for COOLDOWN ticks.
ARBITER_ENABLE = True
ARB_PATIENCE = 3
ARB_COOLDOWN = 5

CORE_STRIDE = 0.10

# Telemetry features
OBSERVABLE = ["Core_F", "Mem_F", "Avg_Pwr_W", "Peak_T_C", "Avg_Util_PCT"]
# =============================================================================

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def norm(value, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

def denorm_to_mhz(norm_val, lo, hi):
    return int(round(lo + norm_val * (hi - lo)))

class LiveController:
    def __init__(self):
        # --- Check Administrator privileges ---
        if not DRY_RUN and not is_admin():
            print("\n" + "="*80)
            print("WARNING: You must run this command from an ADMINISTRATOR terminal to change GPU clocks!")
            print("DRY_RUN mode will be enforced automatically.")
            print("="*80 + "\n")
            globals()["DRY_RUN"] = True

        # --- Load scaler parameters ---
        scaler_path = _resolve("data", "scaler_params.json")
        with open(scaler_path) as f:
            sc = json.load(f)
        self.ranges = sc["ranges"]
        self.state_cols = sc["state_col_order"]
        self.n_features = len(self.state_cols)

        # --- Load preprocessed dataset for Nearest-Neighbor state reconstruction ---
        dataset_path = _resolve("data", "preprocessed_dataset.csv")
        df = pd.read_csv(dataset_path)
        self.ref_states = df[self.state_cols].to_numpy(dtype=np.float32)
        self.obs_idx = [self.state_cols.index(c) for c in OBSERVABLE]
        self.ref_obs = self.ref_states[:, self.obs_idx]
        self.ref_edp = df["EDP"].to_numpy(dtype=np.float32)

        # --- Setup Arbiter thresholds ---
        self.wl_cols = [c for c in self.state_cols if c.startswith("Workload_Name_")]
        self.wl_idx = [self.state_cols.index(c) for c in self.wl_cols]
        self.gov_edp = {}
        for c in self.wl_cols:
            s = df[df[c] == 1.0]
            # Max Core (Core_F = 1.0) is the governor baseline
            g = s[s["Core_F"] == 1.0]
            self.gov_edp[c] = float(g["EDP"].mean()) if len(g) else float(s["EDP"].max())
        
        self.arb_mode, self.arb_bad, self.arb_cd = "AI", 0, 0

        # --- Initialize NVML ---
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)

        # --- Load Agent ---
        self.agent = DDQNAgent((STACK_SIZE, self.n_features), 3) # 3 actions (Dec, Inc, Hold)
        self.agent.policy_net.load_state_dict(
            torch.load(WEIGHTS, map_location=self.agent.device, weights_only=True)
        )
        self.agent.policy_net.eval()
        self.agent.epsilon = 0.0

        self.frames = deque(maxlen=STACK_SIZE)
        
        core_clk, mem_clk, _, _, _ = self.read_sensors()
        self.track_core = norm(core_clk, *self._r("Core_F"))
        self._prev = None

    def _r(self, col):
        return self.ranges[col]["min"], self.ranges[col]["max"]

    def read_sensors(self):
        core = pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)
        mem = pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_MEM)
        pwr = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
        temp = pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.h).gpu
        return core, mem, pwr, temp, util

    def build_state(self, core, mem, pwr, temp, util):
        # Normalize observable readings
        live_obs = np.array([
            norm(core, *self._r("Core_F")),
            norm(mem, *self._r("Mem_F")),
            norm(pwr, *self._r("Avg_Pwr_W")),
            norm(temp, *self._r("Peak_T_C")),
            norm(util, *self._r("Avg_Util_PCT")),
        ], dtype=np.float32)

        # Nearest neighbor lookup for hidden features (one-hot classification and timing metrics)
        d = np.sum((self.ref_obs - live_obs) ** 2, axis=1)
        nn = int(np.argmin(d))
        base = self.ref_states[nn].copy()
        
        # Overwrite observable features with actual live values
        for k, idx in enumerate(self.obs_idx):
            base[idx] = live_obs[k]
            
        wl_col = self.wl_cols[int(np.argmax(base[self.wl_idx]))] if self.wl_cols else None
        return base, wl_col, float(self.ref_edp[nn])

    def apply_action(self, action):
        """Update tracked target clocks and set them on the GPU."""
        if action == 0:
            self.track_core = max(0.0, self.track_core - CORE_STRIDE)
        elif action == 1:
            self.track_core = min(1.0, self.track_core + CORE_STRIDE)
        # action 2 is hold

        core_mhz = denorm_to_mhz(self.track_core, *self._r("Core_F"))
        self.set_clocks(core_mhz)
        return core_mhz

    def set_clocks(self, core_mhz):
        if DRY_RUN:
            return
        # Set core frequency locks on GPU 0
        subprocess.run(["nvidia-smi", "-i", "0", "-lgc", f"{core_mhz},{core_mhz}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def safety_override(self):
        """Force clocks to baseline (405 MHz) on severe temperature override."""
        base_core = denorm_to_mhz(0.0, *self._r("Core_F"))
        self.track_core = 0.0
        self.set_clocks(base_core)

    def reset_clocks(self):
        if DRY_RUN:
            return
        subprocess.run(["nvidia-smi", "-i", "0", "-rgc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run(self):
        period = LOOP_MS / 1000.0
        print(f"Live loop started | DRY_RUN={DRY_RUN} | cadence={LOOP_MS}ms | safety={TEMP_LIMIT_C}C")
        
        while True:
            t0 = time.perf_counter()
            core, mem, pwr, temp, util = self.read_sensors()

            # 1) Layer 1: HARD SAFETY OVERRIDE
            if temp >= TEMP_LIMIT_C:
                self.safety_override()
                print(f"[SAFETY OVERRIDE] Temp: {temp}C >= Limit: {TEMP_LIMIT_C}C. Forcing core to base speed.")
                self._sleep(period, t0)
                continue

            # 2) Build state vector & pad stack
            state_vec, wl_col, nn_edp = self.build_state(core, mem, pwr, temp, util)
            self.frames.append(state_vec)
            while len(self.frames) < STACK_SIZE:
                self.frames.append(state_vec)
            stacked = np.array(self.frames, dtype=np.float32)

            gov_floor = self.gov_edp.get(wl_col, float("inf"))
            tag = ""

            # 3) Layer 2: Performance Arbiter vs. AI Decision
            if ARBITER_ENABLE and self.arb_mode == "GOV":
                core_mhz = denorm_to_mhz(1.0, *self._r("Core_F"))
                self.track_core = 1.0
                self.set_clocks(core_mhz)
                action = -1 # Governor mode
                tag = f" [ARB->GOV {self.arb_cd}]"
                self.arb_cd -= 1
                if self.arb_cd <= 0:
                    self.arb_mode, self.arb_bad = "AI", 0
            else:
                action = self.agent.act(stacked)
                core_mhz = self.apply_action(action)
                if ARBITER_ENABLE:
                    self.arb_bad = self.arb_bad + 1 if nn_edp > gov_floor else 0
                    if self.arb_bad >= ARB_PATIENCE:
                        self.arb_mode, self.arb_cd, tag = "GOV", ARB_COOLDOWN, " [ARB TRIP]"

            dt_ms = (time.perf_counter() - t0) * 1000.0
            act_str = "GOV" if action < 0 else str(action)
            print(f"T={temp:>2}C U={util:>3}% P={pwr:5.1f}W | act={act_str} "
                  f"-> core={core_mhz} MHz | {dt_ms:4.1f}ms{tag}"
                  + ("  [DRY]" if DRY_RUN else ""))
                  
            self._sleep(period, t0)

    @staticmethod
    def _sleep(period, t0):
        remaining = period - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)

def main():
    ctrl = LiveController()

    def shutdown(*_):
        print("\nShutting down: restoring default GPU clocks.")
        try:
            ctrl.reset_clocks()
        finally:
            pynvml.nvmlShutdown()
            sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    ctrl.run()

if __name__ == "__main__":
    main()
