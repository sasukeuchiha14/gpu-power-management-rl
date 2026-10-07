"""
debounce_validate.py -- offline validation of the deployment-side debounce fix
for the limit-cycle problem (context.md "IMMEDIATE NEXT STEP", 5-7 Oct 2026).

RESULT: DOES NOT WORK. Debouncing doesn't break the cycle, it only stretches
its period. At a frozen setpoint the policy sees an identical ("neutral
response") observation every tick, so it keeps re-choosing the same action
until the streak reaches debounce_n, commits one step, the response sign
flips, and the same thing happens in the other direction -- confirmed by
inspecting the raw rollout traces at n=1/3/10 (period 2, then ~6, then
~18-20). Swept n up to 40 over a 300-step rollout: still 9/12 workloads
cycling (same floor as n=3-10), while the aggregate steady-state metric gets
WORSE as n grows (0.079 at n=1 to ~0.12 at n=10) because the same forced
delay also slows genuinely beneficial moves. There is no n that fixes this --
see `common/deploy_live.py`'s `DEBOUNCE_N` comment, left OFF (n=1) by
default. This is the third candidate fix tried and rejected this
investigation (after reward_power and action_cost), all three offline/
mechanistic, no hardware rental needed to reach this verdict.

Two reward-shaping fixes were tried first and didn't hold up (reward_power
made cycling worse across the board; action_cost looked like a clear winner
in a 3-seed screen but regressed at full 10-seed validation -- see
reward_shape_sweep.py and action_cost_fix_validate.py). This script tests the
remaining, not-yet-tried lever: a DEPLOYMENT-side fix that needs no
retraining at all. `common/deploy_live.py`'s `LiveController.debounce()`
requires the policy to choose the same non-HOLD action `debounce_n` ticks in
a row before a clock change actually commits -- a policy alternating
DEC/INC every tick never reaches that streak, so the clock just sits still
instead of cycling.

Reuses the EXACT deterministic-rollout cycle check reward_shape_sweep.py
built (roll the trained policy out for 100 steps from max clock, count how
many of the last 20 steps land on more than one distinct row), the same
checkpoint already confirmed to cycle on 10/12 workloads
(`common/weights/log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth`,
NORTHSTAR row 11), replicating `LiveController.debounce()`'s streak logic
exactly (not importing it directly -- that class also pulls in pynvml /
real-hardware setup this offline check doesn't need) so the thing validated
here offline is the thing that will actually run on hardware. Also checks
the aggregate steady-state scaled-EDP metric at each debounce_n to catch an
N large enough to make the fix itself cost something (a debounce that never
lets the policy move loses the ability to track a workload phase change).

    python common/analysis/debounce_validate.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from _common import REPO  # noqa: F401
from dataset_view import discover_core_only
import eval_policies as P
from train_unified_v2 import QNet, GreedyAgent
from features import STATE_DIM

HOLDOUT = "nvidia_geforce_rtx_4090"
CHECKPOINT = f"{REPO}/common/weights/log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth"
DEBOUNCE_NS = [1, 2, 3, 4, 5, 7, 10]  # 1 = no debounce, reproduces the known 10/12 baseline

DS = discover_core_only()
PREP = {g: P.prepare(d) for g, d in DS.items()}


class DebouncedAgent:
    """Mirrors deploy_live.LiveController.debounce() exactly: wraps a greedy
    agent so act() returns HOLD until the underlying agent has chosen the
    same non-HOLD action `n` ticks in a row."""
    def __init__(self, agent, n):
        self.agent, self.n = agent, n
        self.pending_action, self.pending_count = None, 0

    def act(self, stack):
        a = self.agent.act(stack)
        if a == self.pending_action:
            self.pending_count += 1
        else:
            self.pending_action, self.pending_count = a, 1
        if a == 2 or self.pending_count >= self.n:
            return a
        return 2


def rollout_trace(agent, W, steps=100):
    """Identical to reward_shape_sweep.py's rollout_trace."""
    core, stride = W["core"], W["stride"]
    i = len(core) - 1
    tracked = float(core[i])
    obs0 = P.augmented_obs(W, i, None, True)
    stack = np.repeat(obs0[None, :], 3, axis=0)
    trace = [i]
    for _ in range(steps):
        a = agent.act(stack)
        tracked = float(min(1.0, max(0.0, tracked + (-stride if a == 0 else (stride if a == 1 else 0.0)))))
        j = int(np.argmin(np.abs(core - tracked)))
        obs = P.augmented_obs(W, j, i, True)
        stack = np.concatenate([stack[1:], obs[None, :]], axis=0)
        i = j
        trace.append(i)
    return trace


def load_checkpoint():
    net = QNet(3 * STATE_DIM, 3, hidden=128, dueling=False)  # STATE_DIM already includes the 2 response features
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    net.eval()
    return net


def main():
    net = load_checkpoint()
    rows = []
    for n in DEBOUNCE_NS:
        base_agent = GreedyAgent(net)
        n_cycle, resnet_settle = 0, None
        per_workload = {}
        for wl, W in sorted(PREP[HOLDOUT].items()):
            # fresh debounce state per workload episode, same as a fresh
            # deploy -- a streak shouldn't survive across unrelated episodes
            agent = DebouncedAgent(base_agent, n) if n > 1 else base_agent
            trace = rollout_trace(agent, W)
            distinct = len(set(trace[-20:]))
            per_workload[wl] = distinct
            if distinct > 1:
                n_cycle += 1
            if wl == "pt_resnet50_train":
                resnet_settle = float(W["core"][trace[-1]])

        # aggregate steady-state scaled-EDP metric. evaluate_policy() shares
        # one policy closure across all workloads, which would leak debounce
        # state between unrelated episodes (unlike a real deployment, which
        # is one long-running process per workload) -- so call it once per
        # workload with a fresh agent, matching rollout_trace's per-workload
        # reset above, and average by hand the same way evaluate_policy does.
        ss_vals = []
        for wl, W in PREP[HOLDOUT].items():
            ag = DebouncedAgent(GreedyAgent(net), n) if n > 1 else GreedyAgent(net)
            _, ss_wl, _ = P.evaluate_policy({wl: W}, P.p_agent(ag), use_response=True)
            ss_vals.append(ss_wl)
        ss = float(np.mean(ss_vals))

        rows.append(dict(debounce_n=n, n_cycling=n_cycle, n_total=len(PREP[HOLDOUT]),
                          cycle_rate=n_cycle / len(PREP[HOLDOUT]),
                          resnet_settle_core_frac=resnet_settle, ss=ss))
        print(f"debounce_n={n:<3} cycle_rate={n_cycle}/{len(PREP[HOLDOUT])} "
              f"ss={ss:.4f} resnet_settle={resnet_settle:.3f}", flush=True)

    d = pd.DataFrame(rows)
    out = f"{REPO}/docs/results/debounce_validate.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    print(f"\n(for reference, no-debounce (n=1) reproduces the known 10/12 cycling result "
          f"from reward_shape_sweep.py / context.md's 5 Oct limit-cycle finding)")


if __name__ == "__main__":
    main()
