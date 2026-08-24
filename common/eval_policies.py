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
import os, sys
import numpy as np
import pandas as pd

_common = os.path.dirname(os.path.abspath(__file__))
if _common not in sys.path:
    sys.path.insert(0, _common)
from features import FEATURES, STACK_SIZE  # noqa: E402


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
        )
    return out


def rollout(W, policy, steps=50, start_idx=0):
    """One workload episode. `policy(stack, tracked, stride, W) -> new tracked
    core_frac`. Returns per-step scaled EDP."""
    core, feats, sedp, stride = W["core"], W["feats"], W["sedp"], W["stride"]
    i = int(start_idx)
    tracked = float(core[i])
    stack = np.repeat(feats[i][None, :], STACK_SIZE, axis=0)
    out = np.empty(steps)
    for t in range(steps):
        tracked = float(min(1.0, max(0.0, policy(stack, tracked, stride, W))))
        i = int(np.argmin(np.abs(core - tracked)))
        stack = np.concatenate([stack[1:], feats[i][None, :]], axis=0)
        out[t] = sedp[i]
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

def p_agent(agent):
    """Wraps a DDQN-style agent exposing .act(stack) -> {0 dec, 1 inc, 2 hold}."""
    def f(stack, tracked, stride, W):
        a = agent.act(stack)
        return tracked + (-stride if a == 0 else (stride if a == 1 else 0.0))
    return f


def evaluate_policy(prepared, policy, steps=50, ss_frac=0.4, start_seed=42):
    """(traj_mean, ss_mean, per_workload_df) over every workload on one GPU."""
    rows = []
    for wl, W in sorted(prepared.items()):
        rs = np.random.RandomState(start_seed + (abs(hash(wl)) % 10000))
        start_idx = rs.randint(len(W["core"]))
        h = rollout(W, policy, steps=steps, start_idx=start_idx)
        k = max(1, int(len(h) * ss_frac))
        rows.append(dict(workload=wl, traj=float(h.mean()), ss=float(h[-k:].mean())))
    d = pd.DataFrame(rows)
    return float(d.traj.mean()), float(d.ss.mean()), d
