"""
eval_policies.py -- policy-agnostic evaluation on a held-out GPU, plus the
baseline suite the paper needs (NORTHSTAR row 4; CLAUDE.md 8-D "no baselines
beyond pinned max clocks").

Every policy is evaluated under the SAME protocol: for each workload on the
held-out GPU, start from a fixed per-workload start row, run `steps` control
steps in the table-backed environment, and record the scaled EDP
(1 - Reward, lower is better) at each step.

Two metrics are reported, because they answer different questions:
  * traj -- mean scaled EDP over the whole trajectory. Includes the transient
            while a relative-action controller walks from its start point to
            wherever it settles, so it penalises slow convergence. This is the
            metric the original evaluate_holdout.py used.
  * ss   -- mean scaled EDP over the last `ss_frac` of the trajectory
            ("steady state"): what a converged controller actually delivers in
            production, where an episode is not 50 steps but hours. Reported
            ALONGSIDE traj, never instead of it.

Static baselines (governor/min/fixed/oracle) are constant, so traj == ss for
them by construction; the distinction only bites for learned/heuristic
controllers.

Workload groups are pre-extracted to plain numpy once (`prepare`), because the
RL sweeps run this inner loop millions of times and pandas indexing dominated
the runtime otherwise.
"""
import os, sys, zlib
import numpy as np
import pandas as pd

_common = os.path.dirname(os.path.abspath(__file__))
if _common not in sys.path:
    sys.path.insert(0, _common)
from features import FEATURES, STACK_SIZE, RESPONSE_FEATURES, RESPONSE_CLIP, response_features  # noqa: E402


def infer_stride(core_levels):
    """Median gap between adjacent profiled core_frac levels -- same rule as
    unified_env._infer_core_stride, kept numpy-native here."""
    u = np.unique(core_levels)
    return float(np.median(np.diff(u))) if len(u) > 1 else 0.1


def prepare(df):
    """{workload: dict of numpy arrays} for fast repeated rollouts."""
    out = {}
    for wl, g in df.groupby("Workload_Name"):
        g = g.sort_values("core_frac").reset_index(drop=True)
        core = g["core_frac"].to_numpy(dtype=np.float64)
        out[wl] = dict(
            feats=g[FEATURES].to_numpy(dtype=np.float32),
            core=core,
            sedp=(1.0 - g["Reward"].to_numpy(dtype=np.float64)),
            stride=infer_stride(core),
            opt_core=float(g.loc[g["EDP"].idxmin(), "core_frac"]),
            # raw (unnormalised) time/power, needed only to compute the
            # response features on the fly -- not part of the base state.
            time_s=g["Avg_Time_S"].to_numpy(dtype=np.float64),
            power_w=g["Avg_Pwr_W"].to_numpy(dtype=np.float64),
        )
    return out


def augmented_obs(W, idx, prev_idx, use_response=False):
    """Per-step feature vector for row `idx`, response-augmented with the
    measured effect of the action that landed here FROM `prev_idx` (None if
    no action has been taken yet this episode). Shared by rollout() here and
    train_unified_v2.py's fast training loop so both see identical semantics."""
    base = W["feats"][idx]
    if not use_response:
        return base
    if prev_idx is None:
        resp = response_features(None, float(W["core"][idx]), None, None, None, None)
    else:
        resp = response_features(
            float(W["core"][prev_idx]), float(W["core"][idx]),
            float(W["time_s"][prev_idx]), float(W["time_s"][idx]),
            float(W["power_w"][prev_idx]), float(W["power_w"][idx]),
        )
    return np.concatenate([base, np.asarray(resp, dtype=np.float32)]).astype(np.float32)


def rollout(W, policy, steps=50, start_idx=0, use_response=False):
    """One workload episode. `policy(stack, tracked, stride, W) -> new tracked
    core_frac`. Returns per-step scaled EDP."""
    core, sedp, stride = W["core"], W["sedp"], W["stride"]
    i = int(start_idx)
    tracked = float(core[i])
    obs0 = augmented_obs(W, i, None, use_response)
    stack = np.repeat(obs0[None, :], STACK_SIZE, axis=0)
    out = np.empty(steps)
    for t in range(steps):
        tracked = float(min(1.0, max(0.0, policy(stack, tracked, stride, W))))
        j = int(np.argmin(np.abs(core - tracked)))
        obs = augmented_obs(W, j, i, use_response)
        stack = np.concatenate([stack[1:], obs[None, :]], axis=0)
        out[t] = sedp[j]
        i = j
    return out


# ---------------------------------------------------------------- baselines
def p_governor(stack, tracked, stride, W):  return 1.0     # pinned max clock
def p_minclock(stack, tracked, stride, W):  return 0.0     # pinned min clock
def p_oracle(stack, tracked, stride, W):    return W["opt_core"]

def p_fixed(frac):
    def f(stack, tracked, stride, W): return frac
    return f

def p_random(rng):
    def f(stack, tracked, stride, W):
        return tracked + float(rng.choice([-stride, stride, 0.0]))
    return f

def p_util_heuristic(hi=0.5, lo=0.2):
    """Simple explainable non-RL controller: SM busy -> clock up, SM idle ->
    clock down. The obvious thing to try before reaching for RL, and exactly
    the baseline a reviewer will ask about."""
    j = FEATURES.index("sm_util")
    def f(stack, tracked, stride, W):
        sm = float(stack[-1][j])
        if sm >= hi: return tracked + stride
        if sm <= lo: return tracked - stride
        return tracked
    return f

def p_supervised(clf):
    """Greedy controller driven by a classifier trained on the OTHER GPUs to
    answer 'is the current clock below this workload's EDP optimum?'. This is a
    reference CEILING for what the 6 features can support -- not a deployable
    proposal."""
    def f(stack, tracked, stride, W):
        return tracked + (stride if clf.predict(stack[-1].reshape(1, -1))[0] == 1 else -stride)
    return f

def p_hillclimb(elasticity_threshold=0.05):
    """Hand-coded reference policy (context.md 4 Oct 2026 finding, 0.0051
    scaled EDP / 4.7x over fixed-0.6, 5/5 GPUs -- that run used a throwaway
    script that was never committed, so this is a RE-implementation of the
    described behaviour, not a byte-for-byte reproduction.

    IMPORTANT, measured 4 Oct 2026: this reconstruction does NOT reproduce
    that result. Swept elasticity_threshold in [0, 0.8] and the trip
    direction (step back up vs. hold) on all 8 discovered GPUs; the best
    point found (threshold=0.05, step up on trip -- the default here) gets
    mean steady-state scaled EDP ~0.13, which LOSES to fixed-0.6 (~0.03), not
    beats it. Treat this policy as "a reasonable-effort reconstruction of
    the described mechanism," not as a validated reproduction of the 4.7x
    finding -- that finding rests entirely on a script that no longer
    exists and cannot currently be verified. Reproduce the sweep:
    common/analysis/hillclimb_sweep.py.

    Greedy hill-climb: default to stepping the clock DOWN. After each move,
    read the measured runtime-response elasticity this same step just
    produced (stack[-1]'s d_log_runtime_d_log_clock, un-normalised back from
    [0,1]) -- if its magnitude crosses `elasticity_threshold`, downclocking
    just started costing real runtime, so stop and step back up one level;
    otherwise the workload wasn't sensitive at this point, keep going down.
    Deliberately reuses the SAME response feature the learned agent's state
    carries, rather than a separate calculation, so the comparison is about
    what each policy DOES with the measured signal, not about who has a
    better sensor.

    Floor guard, two bugs found by testing this (neither in the original
    description -- that script no longer exists, see above):

    1. Once the tracked setpoint hits the grid floor, a requested decrease
       stops actually changing the landed row, so Δlog(clock) == 0 and
       response_features() correctly reports the NEUTRAL value -- which this
       policy reads as "flat response, safe to keep going down." Nothing
       then pulls it back up, so it sits pinned at the single worst
       operating point for any compute-bound workload (measured:
       pt_lstm/pt_resnet50 both landed at scaled EDP 1.000, the worst
       possible, before this guard). Fixed by checking the actuation limit
       directly instead of waiting for the response signal to notice it.

    2. The grid floor is NOT core_frac == 0.0 -- it's whatever the lowest
       PROFILED level normalises to (e.g. 0.185, not 0). Bouncing by adding
       `stride` to the abstract `tracked` tracker after it had drifted below
       the grid (clamped to 0.0 by rollout(), which clips but does not know
       about the profiled grid) landed one stride above 0.0, which is still
       closer to the SAME bottom row than to the next one up -- so the
       first "bounce" was a no-op and it stayed pinned regardless. Fixed by
       bouncing relative to the actual last-landed row's core_frac (read
       from the stack) rather than the abstract, possibly-drifted `tracked`
       scalar, and by comparing against this workload's actual minimum
       profiled level (`W["core"].min()`), not a hardcoded 0.0.
    """
    core_idx = FEATURES.index("core_frac")
    rt_idx = len(FEATURES)  # d_log_runtime_d_log_clock is the first response dim
    def f(stack, tracked, stride, W):
        current_core = float(stack[-1][core_idx])
        if current_core <= W["core"].min() + 1e-9:
            return current_core + stride
        raw = (float(stack[-1][rt_idx]) - 0.5) * 2.0 * RESPONSE_CLIP
        if abs(raw) > elasticity_threshold:
            return current_core + stride
        return current_core - stride
    return f


def p_agent(agent):
    """Wraps a DDQN-style agent exposing .act(stack) -> {0 dec, 1 inc, 2 hold}."""
    def f(stack, tracked, stride, W):
        a = agent.act(stack)
        return tracked + (-stride if a == 0 else (stride if a == 1 else 0.0))
    return f


def evaluate_policy(prepared, policy, steps=50, ss_frac=0.4, start_seed=42, use_response=False):
    """(traj_mean, ss_mean, per_workload_df) over every workload on one GPU."""
    rows = []
    for wl, W in sorted(prepared.items()):
        # NOT Python's builtin hash() -- it's salted per-process (PYTHONHASHSEED)
        # unless explicitly fixed, so the "fixed eval start-state" this function
        # promises silently was NOT fixed across separate runs/workers (found
        # 4 Oct 2026 while debugging an apparently-stuck hill-climb trajectory
        # that only reproduced in some process invocations, not others).
        # zlib.crc32 is deterministic across processes and Python versions.
        rs = np.random.RandomState(start_seed + (zlib.crc32(wl.encode()) % 10000))
        start_idx = rs.randint(len(W["core"]))
        h = rollout(W, policy, steps=steps, start_idx=start_idx, use_response=use_response)
        k = max(1, int(len(h) * ss_frac))
        rows.append(dict(workload=wl, traj=float(h.mean()), ss=float(h[-k:].mean())))
    d = pd.DataFrame(rows)
    return float(d.traj.mean()), float(d.ss.mean()), d
