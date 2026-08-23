"""
Layer 2 do-no-harm arbiter -- watches the live control loop and forces full
clock if the policy looks like it is actively hurting a compute-bound
workload's throughput.

WHY THIS DIFFERS FROM THE OLD (RTX 4500) ARBITER DESIGN: the original
implementation estimated live EDP by 1-nearest-neighbour lookup against the
profiled dataset (reconstructing the unobservable Avg_Time_S and workload
identity from 5 observable features), then compared that estimate against
the workload's own governor EDP. That machinery requires exactly the
per-workload dataset lookup that common/features.py deliberately removed
from the state design (see its docstring: workload identity and Avg_Time_S
"cannot exist for an app the model was not trained on"). Reusing it here
would silently reintroduce the same closed-world assumption the redesign
was meant to eliminate -- an EDP arbiter for the new architecture would only
work on workloads present in the profiled dataset, not on arbitrary live
apps, which defeats the point of a live daemon at all.

Instead this arbiter uses a signal that is ALWAYS observable live, on any
workload, with no lookup: SM utilisation. A GPU sitting at high utilisation
is, by definition, compute-bound right now -- it is the bottleneck, so
running it below max clock is directly and immediately extending wall-clock
time. That is precisely the "do-no-harm" failure mode found in CLAUDE.md's
real hardware measurements (14-15% slower for only ~12% less power): the
agent kept a busy GPU downclocked. Watching utilisation catches exactly
that pattern without needing to know what the workload is or estimate its
EDP.

Trip condition: SM utilisation stays at or above ARB_UTIL_THRESHOLD for
ARB_PATIENCE consecutive ticks WHILE the tracked clock is below max. On
trip, clocks are forced to max for ARB_COOLDOWN ticks, then control returns
to the policy. This is deliberately conservative (a false trip just means a
few ticks at full clock -- the safe direction to be wrong in) rather than
trying to precisely estimate the size of the slowdown.
"""


class DoNoHarmArbiter:
    def __init__(self, util_threshold=85.0, patience=3, cooldown=5, enabled=True):
        self.util_threshold = util_threshold
        self.patience = patience
        self.cooldown = cooldown
        self.enabled = enabled
        self.consecutive_bad = 0
        self.cooldown_ticks = 0

    def decide(self, sm_util_pct, tracked_core_mhz, core_max_mhz, requested_core_mhz):
        """Given this tick's live utilisation and what the policy wants to
        do, return (final_target_core_mhz, overridden: bool, tripped_now: bool).

        `tracked_core_mhz` is the clock actually in effect *before* this
        tick's action. `requested_core_mhz` is what the policy's action
        would set it to, absent any override.
        """
        if not self.enabled:
            return requested_core_mhz, False, False

        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1
            self.consecutive_bad = 0  # don't let bad ticks accumulate while already overridden
            return core_max_mhz, True, False

        is_bad_tick = (sm_util_pct >= self.util_threshold) and (tracked_core_mhz < core_max_mhz)
        self.consecutive_bad = self.consecutive_bad + 1 if is_bad_tick else 0

        if self.consecutive_bad >= self.patience:
            self.consecutive_bad = 0
            self.cooldown_ticks = self.cooldown - 1  # this tick counts as the first cooldown tick
            return core_max_mhz, True, True

        return requested_core_mhz, False, False
