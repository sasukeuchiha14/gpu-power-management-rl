"""
train_unified_v2.py -- configurable DDQN trainer for the unified (multi-GPU)
DVFS policy, written to diagnose and fix the seed instability found in the
4-seed sweep (commit 8b68617: only 2/8 held-out GPUs generalised reliably;
several had a per-seed std larger than the mean effect).

Diagnosis that motivated this file (25 Aug 2026):
  * A supervised probe on the SAME 6 features, leave-one-GPU-out, predicts
    "is the current clock below this workload's EDP optimum?" at 81-84%
    accuracy vs a 56% majority baseline. The signal is present and it DOES
    transfer across GPUs -- so the state representation is NOT the problem
    and the instability is in the RL training itself.

Three concrete defects in the v1 trainer/agent this fixes:
  1. TIME-LIMIT TRUNCATION TREATED AS TERMINATION. v1 computes
     `done = terminated or truncated` and multiplies the bootstrap by
     (1 - done). `terminated` is always False in this env; `truncated` fires
     at max_steps. So every episode's final transition had its bootstrap
     zeroed, teaching the agent that value vanishes at an arbitrary wall-clock
     boundary that does not exist in deployment. Fixed: bootstrap through
     truncation, and only zero it on true termination.
  2. UNSCALED MSE LOSS AT LARGE Q MAGNITUDE. Reward is ~[0,1] per step and
     gamma=0.99 over 50 steps, so Q targets approach ~1/(1-gamma) = 100.
     Squared error on targets of that magnitude produces very large
     gradients. Fixed: Huber (smooth-L1) loss plus gradient-norm clipping,
     both standard DQN stabilisers that v1 omitted.
  3. NO TARGET SMOOTHING. v1 hard-copies the target net every 5 episodes
     (~250 gradient steps apart), which makes the regression target lurch.
     Optional Polyak (soft) updates added.

Everything else is exposed as a knob so the alternatives can actually be
measured rather than argued about: gamma, learning rate, batch size, network
width, dueling head, Double-DQN on/off, epsilon schedule, episode count,
action-cost shaping, and reward scaling.

The environment is re-implemented here in numpy (identical snapping
semantics to unified_env.UnifiedDVFSEnv: move a tracked core_frac by the
workload's own median level spacing, snap to the nearest profiled row of the
same workload, reward is that row's Reward) purely for speed -- the sweeps
run this loop tens of millions of times and the pandas path dominated
runtime. unified_env.py remains the reference implementation.
"""
import argparse, os, random, sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn

_common = os.path.dirname(os.path.abspath(__file__))
if _common not in sys.path:
    sys.path.insert(0, _common)
from features import FEATURES, RESPONSE_FEATURES, STACK_SIZE  # noqa: E402
from dataset_view import discover_core_only  # noqa: E402
from eval_policies import prepare, augmented_obs  # noqa: E402

UTIL_IDX = FEATURES.index("sm_util") if "sm_util" in FEATURES else 1


class QNet(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=128, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        if dueling:
            self.v = nn.Linear(hidden, 1)
            self.a = nn.Linear(hidden, n_actions)
        else:
            self.out = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.trunk(x)
        if not self.dueling:
            return self.out(h)
        v, a = self.v(h), self.a(h)
        return v + a - a.mean(dim=1, keepdim=True)


class Cfg:
    """Defaults reproduce the v1 setup EXCEPT the three bug fixes above, so a
    v1-vs-v2 comparison isolates 'fixes + knobs', not an accidental rewrite."""
    episodes=800; steps=50; gamma=0.99; lr=1e-3; batch=64; buffer=10000
    abs_actions=0   # 0 = relative {dec,inc,hold} (deployed semantics).
                    # K>0 = ABSOLUTE: action k directly selects core_frac
                    # k/(K-1). With gamma=0 this makes the problem the
                    # contextual bandit it structurally is (reward depends only
                    # on the chosen row, never on the path taken there), which
                    # removes both the multi-step credit assignment and the
                    # walk-to-target transient. Costs deployment realism: the
                    # live daemon issues relative steps, so an absolute policy
                    # would need its actuation path changed too.
    hidden=128; dueling=False; double=True; huber=True; grad_clip=10.0
    target_hard_every=5; tau=0.0          # tau>0 -> Polyak instead of hard copy
    eps_start=1.0; eps_min=0.05; eps_decay=0.99
    action_cost=0.0; reward_scale=1.0
    learn_every=1; warmup=64
    use_response=False   # False = old 6-feature state (the ablation).
                          # True = +2 response features (context.md Phase 1).
    use_log_reward=False # False = train on the linear scaled-EDP reward (same
                          # metric as evaluation, original behaviour).
                          # True = train on the log-scale reward (sedp_train,
                          # see dataset_view.core_only_view) -- fixes the flat
                          # near-optimum plateau a Phase 2 hardware run showed
                          # causes real over-aggressive downclocking. Evaluation
                          # always uses the unchanged linear metric regardless.
    reward_power=None    # None = use use_log_reward above (unchanged). A float
                          # OVERRIDES use_log_reward with a tunable per-workload
                          # min-max over EDP**reward_power (1.0 = identical to
                          # the original linear reward, 0.0 = identical to
                          # use_log_reward's log(EDP), values in between
                          # interpolate the compression strength). Computed on
                          # the fly from W["edp"] so it can be swept without
                          # precomputing a new dataset column per candidate --
                          # see common/analysis/reward_shape_sweep.py, built
                          # after repeated hardware trials found the pure-log
                          # fix (reward_power=0) trades compute-bound stability
                          # for launch-bound oscillation (context.md, 5 Oct).
    reversal_cost=0.0    # Penalise choosing the OPPOSITE direction from the
                          # last non-HOLD move (INC right after DEC, or vice
                          # versa). Unlike action_cost (penalises every
                          # non-HOLD move uniformly, tried and found not to
                          # fix the limit cycle -- see
                          # common/analysis/reward_shape_sweep.py), this
                          # targets the specific mechanism behind the cycle:
                          # two adjacent clock levels each looking like the
                          # better move FROM the other, a mutual attraction a
                          # flat per-move cost doesn't discourage any more
                          # than it discourages genuine one-way corrections.
                          # See common/analysis/reversal_cost_sweep.py.
    reversal_window=None # None = reversal_cost applies however long ago the last
                          # non-HOLD move was (original behaviour). int K = only
                          # penalise a reversal within K steps of that move, so a
                          # genuine correction after a dwell is free.
    hold_bonus=0.0       # Added to the reward on every HOLD, to make "stay put"
                          # a stable attractor at a plateau.
    reversal_util_scale=False  # True = scale reversal_cost by the landing row's
                          # sm_util, so cycling is taxed more on busy
                          # (compute-bound) rows than on idle ones.

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(Cfg, k): raise ValueError(f"unknown cfg key {k}")
            setattr(self, k, v)

    def __repr__(self):
        base = Cfg()
        d = {k: getattr(self, k) for k in vars(Cfg)
             if not k.startswith("_") and getattr(self, k) != getattr(base, k)}
        return "default" if not d else ",".join(f"{k}={v}" for k, v in sorted(d.items()))


def build_pool(holdout_gpu, datasets=None):
    """All (gpu, workload) numpy groups EXCEPT the held-out GPU's."""
    ds = datasets if datasets is not None else discover_core_only()
    pool = []
    for gpu, df in sorted(ds.items()):
        if gpu == holdout_gpu: continue
        for wl, W in sorted(prepare(df).items()):
            pool.append(W)
    if not pool:
        raise SystemExit(f"no training data left after holding out {holdout_gpu}")
    return pool


def train(holdout_gpu, cfg=None, seed=0, datasets=None, verbose=False):
    cfg = cfg or Cfg()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pool = build_pool(holdout_gpu, datasets)

    in_dim = STACK_SIZE * (len(FEATURES) + (len(RESPONSE_FEATURES) if cfg.use_response else 0))
    n_act = cfg.abs_actions if cfg.abs_actions else 3
    policy = QNet(in_dim, n_act, cfg.hidden, cfg.dueling)
    target = QNet(in_dim, n_act, cfg.hidden, cfg.dueling)
    target.load_state_dict(policy.state_dict()); target.eval()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    lossfn = nn.SmoothL1Loss() if cfg.huber else nn.MSELoss()
    mem = deque(maxlen=cfg.buffer)
    eps = cfg.eps_start
    hist = []
    gstep = 0

    for ep in range(cfg.episodes):
        W = pool[np.random.randint(len(pool))]
        core, stride = W["core"], W["stride"]
        if cfg.reward_power is not None:
            edp = W["edp"]
            x = np.log(edp) if cfg.reward_power == 0 else edp ** cfg.reward_power
            span = x.max() - x.min()
            rew_arr = np.ones_like(edp) if span <= 0 else 1.0 - (x - x.min()) / span
        else:
            sedp_for_training = W["sedp_train"] if cfg.use_log_reward else W["sedp"]
            rew_arr = 1.0 - sedp_for_training
        i = np.random.randint(len(core))
        tracked = float(core[i])
        obs0 = augmented_obs(W, i, None, cfg.use_response)
        stack = np.repeat(obs0[None, :], STACK_SIZE, axis=0)
        total = 0.0
        last_dir = None  # last non-HOLD action direction this episode (0=dec, 1=inc)
        since_move = 10**6

        for t in range(cfg.steps):
            if np.random.rand() <= eps:
                a = np.random.randint(n_act)
            else:
                with torch.no_grad():
                    a = int(policy(torch.from_numpy(stack).unsqueeze(0)).argmax(1).item())
            if cfg.abs_actions:
                tracked = a / (n_act - 1)
            else:
                tracked = float(min(1.0, max(0.0, tracked + (-stride if a == 0 else (stride if a == 1 else 0.0)))))
            j = int(np.argmin(np.abs(core - tracked)))
            obs = augmented_obs(W, j, i, cfg.use_response)
            nxt = np.concatenate([stack[1:], obs[None, :]], axis=0)
            moved = (a != 2) if not cfg.abs_actions else (abs(tracked - float(core[i])) > 1e-9)
            reversed_dir = (not cfg.abs_actions) and a in (0, 1) and last_dir is not None and a != last_dir
            if reversed_dir and cfg.reversal_window is not None and since_move > cfg.reversal_window:
                reversed_dir = False
            since_move = 0 if a in (0, 1) else since_move + 1
            if a in (0, 1):
                last_dir = a
            rc = cfg.reversal_cost
            if cfg.reversal_util_scale:
                rc = rc * float(W["feats"][j][UTIL_IDX])
            r = (rew_arr[j] - (cfg.action_cost if moved else 0.0)
                 - (rc if reversed_dir else 0.0)
                 + (cfg.hold_bonus if a == 2 else 0.0)) * cfg.reward_scale
            # Fix (1): time-limit truncation is NOT termination. This env never
            # terminates, so the bootstrap is always kept alive.
            mem.append((stack, a, r, nxt, 0.0))
            stack = nxt; i = j; total += r; gstep += 1

            if len(mem) > max(cfg.warmup, cfg.batch) and gstep % cfg.learn_every == 0:
                bs = random.sample(mem, cfg.batch)
                S = torch.from_numpy(np.array([b[0] for b in bs], dtype=np.float32))
                A = torch.tensor([b[1] for b in bs]).unsqueeze(1)
                R = torch.tensor([b[2] for b in bs], dtype=torch.float32).unsqueeze(1)
                S2 = torch.from_numpy(np.array([b[3] for b in bs], dtype=np.float32))
                D = torch.tensor([b[4] for b in bs], dtype=torch.float32).unsqueeze(1)
                with torch.no_grad():
                    if cfg.double:
                        a2 = policy(S2).argmax(1, keepdim=True)
                        q2 = target(S2).gather(1, a2)
                    else:
                        q2 = target(S2).max(1, keepdim=True)[0]
                    y = R + cfg.gamma * q2 * (1.0 - D)
                loss = lossfn(policy(S).gather(1, A), y)
                opt.zero_grad(); loss.backward()
                if cfg.grad_clip:      # Fix (2b): gradient-norm clipping
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                opt.step()
                if cfg.tau > 0:        # Fix (3): optional Polyak smoothing
                    with torch.no_grad():
                        for tp, pp in zip(target.parameters(), policy.parameters()):
                            tp.mul_(1 - cfg.tau).add_(cfg.tau * pp)

        if cfg.tau == 0 and ep % cfg.target_hard_every == 0:
            target.load_state_dict(policy.state_dict())
        if eps > cfg.eps_min:
            eps *= cfg.eps_decay
        hist.append(total)
        if verbose and (ep + 1) % 200 == 0:
            print(f"  ep {ep+1}/{cfg.episodes} avg_r(last100)={np.mean(hist[-100:]):.2f} eps={eps:.3f}")
    return policy, hist


class GreedyAgent:
    """.act(stack) -> action, matching eval_policies.p_agent's expectation."""
    def __init__(self, net): self.net = net; self.net.eval()
    def act(self, stack):
        with torch.no_grad():
            return int(self.net(torch.from_numpy(np.asarray(stack, dtype=np.float32)).unsqueeze(0)).argmax(1).item())


def p_agent_abs(net, n_act):
    """eval_policies-compatible policy for an ABSOLUTE-action network: the
    chosen action IS the target core_frac bucket, no walking."""
    ag = GreedyAgent(net)
    def f(stack, tracked, stride, W):
        return ag.act(stack) / (n_act - 1)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    net, hist = train(a.holdout, Cfg(episodes=a.episodes, gamma=a.gamma), seed=a.seed, verbose=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.save(net.state_dict(), a.out)
        print("saved", a.out)


if __name__ == "__main__":
    main()
