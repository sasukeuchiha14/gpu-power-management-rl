"""
Canonical state definition — SINGLE SOURCE OF TRUTH for every GPU tree.

Both the rtx3050ti_laptop and rtx4500_ada trees (and every rented-GPU dataset)
must import from here. Do not redefine these features locally; if this file and
a local copy disagree, this file wins.

Why these six: they are (a) readable live from NVML on any NVIDIA GPU, and
(b) normalised by limits queried from the device itself, so a value of 0.7
means the same physical thing on a 35W GTX 1650 and a 450W RTX 4090.

Deliberately EXCLUDED:
  * workload one-hot  -> cannot exist for an app the model was not trained on
  * Mem_F             -> kept core-only for now, but see NOTE below: this is a
                         design choice, not a settled hardware fact
  * Avg_Time_S        -> not observable live for an unknown workload

NOTE on Mem_F (23 Aug 2026, corrected 25 Aug, resolved 25 Aug): the original
reasoning here was "consumer cards expose no usable memory-clock range,"
based on two MOBILE GeForce cards (GTX 1650 Max-Q, RTX 3050 Ti Laptop) where
`-lmc` was outright rejected by the driver -- that result is solid, always
was. Two rented DESKTOP cards were tested and both genuinely support
memory-clock + power-limit control on an unrestricted VM: RTX 3060 (Ampere)
and RTX 4070 (Ada) -- two different generations, both confirmed with a
readback-verifying probe (gpu_probe.py originally had a bug where
`-lmc`/`-pl` report success even when the change silently never applies;
fixed, then both cards re-checked properly; see NORTHSTAR.md for the full
story). So the evidence is now solid, not just promising: mobile GeForce
chips hard-reject memory-clock control, desktop GeForce chips across two
generations support it fully. Core-only is STILL kept here -- not because
the capability doesn't exist on desktop cards, but because a fleet where
laptop cards (part of the target product) can't do memory control and
desktop cards can breaks having one unified action space. That's now a real
design decision worth discussing with the user, not a question of whether
the data supports it.
"""

FEATURES = [
    "core_frac",   # SM clock / max SM clock
    "sm_util",     # SM utilisation %/100
    "mem_util",    # memory-controller busy %/100  <- compute vs bandwidth bound
    "power_frac",  # board power / power limit
    "temp_frac",   # temperature / slowdown threshold
    "vram_frac",   # VRAM used / VRAM total
]
FEATURE_DIM = len(FEATURES)

STACK_SIZE = 3          # frames of history fed to the network
ACTIONS = ["DEC_CORE", "INC_CORE", "HOLD"]
ACTION_DIM = len(ACTIONS)

# Raw telemetry columns every collector must write.
RAW_COLUMNS = [
    "Workload_Name", "Workload_Type", "Core_F", "Mem_F",
    "Avg_Time_S", "Avg_Pwr_W", "Peak_T_C",
    "Avg_GPU_Util", "Avg_Mem_Util", "VRAM_Used_MB", "EDP",
]


def build_features(core_mhz, gpu_util, mem_util, power_w, temp_c,
                   vram_used_mb, hw):
    """Map raw telemetry -> the 6 normalised features, in FEATURES order.

    `hw` is the hardware_limits dict produced by common/gpu_probe.py.
    Used by BOTH the offline preprocessor and the live daemon, so training and
    deployment can never drift apart.
    """
    def clip01(x):
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

    return [
        clip01(core_mhz / hw["max_core_mhz"]),
        clip01(gpu_util / 100.0),
        clip01(mem_util / 100.0),
        clip01(power_w / hw["power_limit_w"]),
        clip01(temp_c / hw["slowdown_temp_c"]),
        clip01(vram_used_mb / hw["total_vram_mb"]),
    ]


def edp(avg_power_w, avg_time_s):
    """Energy-Delay Product = (P * t) * t. Lower is better."""
    return (avg_power_w * avg_time_s) * avg_time_s
