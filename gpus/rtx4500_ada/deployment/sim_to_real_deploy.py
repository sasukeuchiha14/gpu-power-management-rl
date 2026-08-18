"""
sim_to_real_deploy.py  —  Phase 3 live controller for the RTX 4500.

Takes the DDQN trained in simulation (ddqn_weights.pth) and runs it as a real-time
DVFS control loop on the physical GPU: every ~50ms it reads live NVML telemetry,
rebuilds the state the network expects (feature count/order comes from
scaler_params.json, never hardcoded here), picks an action, and steps the
core/memory clocks via nvidia-smi. A hard 85C safety net overrides the AI.

SAFETY FIRST
------------
  * Check the DRY_RUN flag below before running -- it does NOT default to
    observe-only. When True, it reads sensors and PRINTS the clocks it
    *would* set without touching hardware; when False it actuates real clocks
    immediately. Watch a DRY_RUN=True run for a few minutes before flipping it.
  * Real clock control needs `sudo nvidia-smi -pm 1` (persistence mode)
    enabled once beforehand. No sudo needed on the outer `python3` command --
    the clock-setting calls invoke `sudo nvidia-smi` themselves (this
    machine's NOPASSWD sudoers rule is scoped to exactly /usr/bin/nvidia-smi).

Prereqs on the RTX 4500 machine:
    sudo nvidia-smi -pm 1
    python3 export_scaler.py           # once, to create scaler_params.json
Run:
    python3 sim_to_real_deploy.py
Stop with Ctrl+C (clocks are automatically reset to default on exit).
"""
import json
import signal
import subprocess
import sys
import time
import os
from collections import deque
import numpy as np
import pandas as pd
import torch
import pynvml

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
DRY_RUN = False            # True = observe only, never change clocks. FLIP LAST.
GPU_INDEX = 0
LOOP_MS = 50              # decision cadence (the 50ms latency budget)
TEMP_LIMIT_C = 85         # hard safety override temperature (Layer 1)
STACK_SIZE = 3            # must match training (frame stacking)
WEIGHTS = _resolve("weights", "ddqn_weights.pth")
ONLINE_FINETUNE = False   # optional online adaptation (see note at bottom)

# Layer 2: do-no-harm arbiter. If the agent's live EDP proxy stays worse than the
# governor for ARB_PATIENCE ticks, revert to max clocks for ARB_COOLDOWN ticks.
ARBITER_ENABLE = True
ARB_PATIENCE = 3
ARB_COOLDOWN = 5

# Normalized stride per action, identical to dvfs_env_advanced.py
CORE_STRIDE = 0.10
MEM_STRIDE = 0.15

# Observable-live columns (everything else in the state is reconstructed)
OBSERVABLE = ["Core_F", "Mem_F", "Avg_Pwr_W", "Peak_T_C", "Avg_Util_PCT"]
# =============================================================================


def norm(value, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def denorm_to_mhz(norm_val, lo, hi):
    return int(round(lo + norm_val * (hi - lo)))


class LiveController:
    def __init__(self):
        # --- Load normalization ranges + exact feature order ---
        scaler_path = _resolve("data", "scaler_params.json")
        with open(scaler_path) as f:
            sc = json.load(f)
        self.ranges = sc["ranges"]
        self.state_cols = sc["state_col_order"]        # feature count/order comes from the scaler, not hardcoded
        self.n_features = len(self.state_cols)

        # --- Dataset used to reconstruct the non-observable features ---
        dataset_path = _resolve("data", "preprocessed_dataset.csv")
        df = pd.read_csv(dataset_path)
        self.ref_states = df[self.state_cols].to_numpy(dtype=np.float32)
        self.obs_idx = [self.state_cols.index(c) for c in OBSERVABLE]
        self.ref_obs = self.ref_states[:, self.obs_idx]
        self.ref_edp = df["EDP"].to_numpy(dtype=np.float32)     # raw EDP per ref row

        # --- Arbiter references: per-workload governor (max-clock) EDP ---
        self.wl_cols = [c for c in self.state_cols if c.startswith("Workload_Name_")]
        self.wl_idx = [self.state_cols.index(c) for c in self.wl_cols]
        self.gov_edp = {}
        for c in self.wl_cols:
            s = df[df[c] == 1.0]
            g = s[(s["Core_F"] == 1.0) & (s["Mem_F"] == 1.0)]
            self.gov_edp[c] = float(g["EDP"].mean()) if len(g) else float(s["EDP"].max())
        self.arb_mode, self.arb_bad, self.arb_cd = "AI", 0, 0

        # --- NVML ---
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)

        # --- Agent ---
        self.agent = DDQNAgent((STACK_SIZE, self.n_features), 5)
        self.agent.policy_net.load_state_dict(
            torch.load(WEIGHTS, map_location=self.agent.device, weights_only=True))
        self.agent.policy_net.eval()
        self.agent.epsilon = 0.01 if ONLINE_FINETUNE else 0.0

        self.frames = deque(maxlen=STACK_SIZE)
        # tracked normalized target clocks (like current_tracked_* in the sim)
        c, m, _, _, _ = self.read_sensors()
        self.track_core = norm(c, *self._r("Core_F"))
        self.track_mem = norm(m, *self._r("Mem_F"))
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
        """Return the 23-feature vector: live values for observable columns,
        nearest-neighbour reconstruction for the rest."""
        live_obs = np.array([
            norm(core, *self._r("Core_F")),
            norm(mem, *self._r("Mem_F")),
            norm(pwr, *self._r("Avg_Pwr_W")),
            norm(temp, *self._r("Peak_T_C")),
            norm(util, *self._r("Avg_Util_PCT")),
        ], dtype=np.float32)

        # nearest known operating point by observable features
        d = np.sum((self.ref_obs - live_obs) ** 2, axis=1)
        nn = int(np.argmin(d))
        base = self.ref_states[nn].copy()
        # overwrite observable slots with the real live readings
        for k, idx in enumerate(self.obs_idx):
            base[idx] = live_obs[k]
        # identify workload of the nearest row (for the arbiter) + its raw EDP proxy
        wl_col = self.wl_cols[int(np.argmax(base[self.wl_idx]))] if self.wl_cols else None
        return base, wl_col, float(self.ref_edp[nn])

    def apply_action(self, action):
        """Update tracked target clocks and (unless DRY_RUN) set them."""
        if action == 0:
            self.track_core = max(0.0, self.track_core - CORE_STRIDE)
        elif action == 1:
            self.track_core = min(1.0, self.track_core + CORE_STRIDE)
        elif action == 2:
            self.track_mem = max(0.0, self.track_mem - MEM_STRIDE)
        elif action == 3:
            self.track_mem = min(1.0, self.track_mem + MEM_STRIDE)
        # action 4 = hold

        core_mhz = denorm_to_mhz(self.track_core, *self._r("Core_F"))
        mem_mhz = denorm_to_mhz(self.track_mem, *self._r("Mem_F"))
        self.set_clocks(core_mhz, mem_mhz)
        return core_mhz, mem_mhz

    def set_clocks(self, core_mhz, mem_mhz):
        if DRY_RUN:
            return
        subprocess.run(["sudo", "nvidia-smi", "-lgc", f"{core_mhz},{core_mhz}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["sudo", "nvidia-smi", "-lmc", f"{mem_mhz},{mem_mhz}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def safety_override(self):
        """Force clocks to base when too hot. Returns True if it fired."""
        base_core = denorm_to_mhz(0.0, *self._r("Core_F"))
        base_mem = denorm_to_mhz(0.0, *self._r("Mem_F"))
        self.track_core, self.track_mem = 0.0, 0.0
        self.set_clocks(base_core, base_mem)
        return True

    def reset_clocks(self):
        if DRY_RUN:
            return
        subprocess.run(["sudo", "nvidia-smi", "-rgc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["sudo", "nvidia-smi", "-rmc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run(self):
        period = LOOP_MS / 1000.0
        print(f"Live loop started | DRY_RUN={DRY_RUN} | cadence={LOOP_MS}ms | "
              f"safety={TEMP_LIMIT_C}C | online_finetune={ONLINE_FINETUNE}")
        while True:
            t0 = time.perf_counter()
            core, mem, pwr, temp, util = self.read_sensors()

            # 1) SAFETY NET (always wins over the AI)
            if temp >= TEMP_LIMIT_C:
                self.safety_override()
                print(f"[SAFETY] {temp}C >= {TEMP_LIMIT_C}C -> clocks forced to base.")
                self._sleep(period, t0)
                continue

            # 2) build state + frame stack
            state_vec, wl_col, nn_edp = self.build_state(core, mem, pwr, temp, util)
            self.frames.append(state_vec)
            while len(self.frames) < STACK_SIZE:      # pad on the very first tick
                self.frames.append(state_vec)
            stacked = np.array(self.frames, dtype=np.float32)

            gov_floor = self.gov_edp.get(wl_col, float("inf"))
            tag = ""

            # 3) LAYER 2: do-no-harm arbiter (else the AI decides)
            if ARBITER_ENABLE and self.arb_mode == "GOV":
                core_mhz = denorm_to_mhz(1.0, *self._r("Core_F"))
                mem_mhz = denorm_to_mhz(1.0, *self._r("Mem_F"))
                self.track_core, self.track_mem = 1.0, 1.0
                self.set_clocks(core_mhz, mem_mhz)
                action = -1                            # governor override
                tag = f" [ARB->GOV {self.arb_cd}]"
                self.arb_cd -= 1
                if self.arb_cd <= 0:
                    self.arb_mode, self.arb_bad = "AI", 0
            else:
                action = self.agent.act(stacked)
                core_mhz, mem_mhz = self.apply_action(action)
                if ARBITER_ENABLE:
                    self.arb_bad = self.arb_bad + 1 if nn_edp > gov_floor else 0
                    if self.arb_bad >= ARB_PATIENCE:
                        self.arb_mode, self.arb_cd, tag = "GOV", ARB_COOLDOWN, " [ARB TRIP]"

            # 4) optional online fine-tuning (see note below)
            if ONLINE_FINETUNE and self._prev is not None and action >= 0:
                p_state, p_action = self._prev
                reward = self._live_reward(pwr, util)      # proxy reward
                self.agent.step(p_state, p_action, reward, stacked, False)
            if action >= 0:
                self._prev = (stacked.copy(), action)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            act_str = "GOV" if action < 0 else str(action)
            print(f"T={temp:>2}C U={util:>3}% P={pwr:5.1f}W | act={act_str} "
                  f"-> core={core_mhz} mem={mem_mhz} | {dt_ms:4.1f}ms{tag}"
                  + ("  [DRY]" if DRY_RUN else ""))
            self._sleep(period, t0)

    def _live_reward(self, pwr, util):
        # Proxy: reward low power while work is happening. Real online reward
        # design is a research task; refine before enabling ONLINE_FINETUNE.
        p_lo, p_hi = self._r("Avg_Pwr_W")
        return (util / 100.0) * (1.0 - norm(pwr, p_lo, p_hi))

    @staticmethod
    def _sleep(period, t0):
        remaining = period - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)


def main():
    ctrl = LiveController()

    def shutdown(*_):
        print("\nShutting down: resetting GPU clocks to default.")
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
