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

NOTE on Mem_F (23 Aug 2026): the original reasoning here was "consumer cards
expose no usable memory-clock range," based on two MOBILE GeForce cards (GTX
1650 Max-Q, RTX 3050 Ti Laptop) where `-lmc` was rejected by the driver
itself. A rented desktop RTX 3060 on an unrestricted VM successfully locked
memory clock, core clock, AND power limit -- all three -- while an identical
RTX 3060 model on a Docker-container instance failed all three with
"insufficient permission." See common/fleet_survey/fleet_survey.csv. So the
real pattern (so far, n=3) looks like: mobile/laptop GeForce chips may
genuinely lack memory-clock control at the driver level, but desktop GeForce
chips might have it -- gated by container vs VM privilege, not by "consumer
vs professional." Core-only is kept here because a fleet with inconsistent
per-card action spaces breaks the "one unified model" goal, not because the
capability doesn't exist. Revisit if more desktop-card data confirms the
pattern -- see NORTHSTAR.md.
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
