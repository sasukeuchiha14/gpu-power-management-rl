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
import math

FEATURES = [
    "core_frac",   # SM clock / max SM clock
    "sm_util",     # SM utilisation %/100
    "mem_util",    # memory-controller busy %/100  <- compute vs bandwidth bound
    "power_frac",  # board power / power limit
    "temp_frac",   # temperature / slowdown threshold
    "vram_frac",   # VRAM used / VRAM total
]
FEATURE_DIM = len(FEATURES)

# --- response-aware augmentation (4 Oct 2026, context.md "IMMEDIATE NEXT STEP") ---
# NOT raw-CSV columns -- unlike FEATURES above, these can't be precomputed into
# preprocessed_dataset.csv because a profiling sweep has no "previous action":
# each row is an independent (workload, core, mem) operating point, not a step
# in a trajectory. These are computed on the fly, per control step, from the
# PREVIOUS visited row and the CURRENT one -- see response_features() below.
# Env/trainer code appends them to the per-step observation only when response
# augmentation is enabled (the "new" 8-dim state); the ablation that reproduces
# the old DDQN trains with just FEATURES (6-dim) for a clean A/B.
RESPONSE_FEATURES = ["d_log_runtime_d_log_clock", "d_log_power_d_log_clock"]
STATE_FEATURES = FEATURES + RESPONSE_FEATURES
STATE_DIM = len(STATE_FEATURES)

# Clip bound for the raw log-ratio before mapping to [0,1], chosen from the
# measured distribution across the 5-GPU clean fleet (705 adjacent-core-level
# pairs, all 12 workloads): runtime ratio p1/p99 = [-1.03, 0.29], power ratio
# p1/p99 = [-0.48, 2.67]; a few true outliers reach ~-5 / +4. +-3 keeps the
# bulk of real transitions linear and only saturates the rare large jump.
# Reproduce: common/analysis/response_distribution.py.
RESPONSE_CLIP = 3.0

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


def response_features(core_prev, core_curr, time_prev, time_curr, power_prev, power_curr):
    """Measured response to the LAST clock change -> the 2 RESPONSE_FEATURES,
    each normalised to [0, 1].

    d_log_runtime_d_log_clock = Δlog(time)  / Δlog(core_frac)
    d_log_power_d_log_clock   = Δlog(power) / Δlog(core_frac)

    This is an elasticity: how much a metric moved, in log space, per unit of
    log clock change. A compute-bound workload has runtime elasticity near
    -1 (runtime scales inversely with clock); a memory/launch-bound one is
    near 0 (lowering clock is nearly free). It's a ratio of two deltas, so
    it's naturally device-invariant -- the same physical regime reads the
    same way on a 3060 and a 4090, unlike an absolute telemetry snapshot.

    `core_prev` is None on the very first observation of an episode (no prior
    action exists yet to measure a response to).

    Degenerate cases (no previous action; HOLD / snap-back with zero clock
    delta) both map to NEUTRAL = 0.5 -- the same value a genuinely-measured
    zero elasticity would normalise to. Deliberate: "no information yet" and
    "measured a flat response" are both "nothing says don't go lower" to a
    consumer of this feature (the hill-climb policy and, implicitly, whatever
    the DDQN learns), so collapsing them is the conservative choice, not a
    shortcut -- the alternative (a 9th "validity" bit) would contradict
    "2 response features" above.
    """
    NEUTRAL = 0.5

    def _norm(raw):
        clipped = max(-RESPONSE_CLIP, min(RESPONSE_CLIP, raw))
        return (clipped + RESPONSE_CLIP) / (2.0 * RESPONSE_CLIP)

    if core_prev is None:
        return (NEUTRAL, NEUTRAL)

    d_log_clock = math.log(core_curr) - math.log(core_prev)
    if abs(d_log_clock) < 1e-9:
        return (NEUTRAL, NEUTRAL)

    d_log_runtime = math.log(time_curr) - math.log(time_prev)
    d_log_power = math.log(power_curr) - math.log(power_prev)
    return (_norm(d_log_runtime / d_log_clock), _norm(d_log_power / d_log_clock))


def edp(avg_power_w, avg_time_s):
    """Energy-Delay Product = (P * t) * t. Lower is better."""
    return (avg_power_w * avg_time_s) * avg_time_s
