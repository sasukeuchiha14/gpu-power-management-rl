# context.md — read this first

If you're a new Claude Code session picking this project up, read this file before
anything else. It's the only file in this repo that's meant to be rewritten every
session — everything else changes slowly.

**Read next, in order:** [NORTHSTAR.md](NORTHSTAR.md) (what we're building, and the
scorecard) → [CLAUDE.md](CLAUDE.md) (deep technical reference, only as needed).

---

## Where we are right now

**Branch:** `unified-model` (all work happens here; `main` is teammates' committed work,
left untouched unless explicitly asked to merge).

**Last session ended:** 23 Aug 2026. Set up this doc structure, the shared
`common/features.py` + `common/gpu_probe.py` modules, and verified remote-GPU access is
viable (network egress works from this sandbox, SSH client present, GitHub push already
authenticated via `gh`).

**Immediate next step:** waiting on the user to rent a vast.ai GPU instance and share the
SSH connection line (`ssh -p <port> root@<ip>`). Once given: connect, clone this branch,
`cd common && python3 gpu_probe.py`, report USABLE/UNUSABLE + specs back to the user
**before** doing anything further on that box (explicit agreement — first touch on any
rented machine is probe-then-check-in, never probe-then-immediately-profile).

## Working agreements from the user (don't relitigate these)

- **Never push to `main`.** All commits go to `unified-model` unless told otherwise.
- **On a rented GPU box:** run `gpu_probe.py`, report back, wait for go-ahead before a
  full profiling run. Don't chain straight into hours of paid GPU time on a box that
  hasn't been checked in on yet.
- **The end product is one unified model** (Option B from the discussion), not per-card
  weight selection. See NORTHSTAR.md's "why one model" section if this gets questioned —
  it's been decided, don't re-open it without new information.
- **The RTX 4500 is explicitly out of scope for the product** (professional card, not
  3000/4000 consumer series). It stays in the repo as a supplementary case, not something
  to migrate to the unified design under time pressure.
- Push access to GitHub already works from this sandbox via `gh` — no token needed from
  the user.

## Doc structure (so it doesn't drift back into one file)

| File | Job | Update cadence |
|---|---|---|
| `context.md` | This file. Session handoff: where we are, what's next. | Every session |
| `NORTHSTAR.md` | End-product definition + 9-row scorecard. Source for the published Artifact. | After any change that moves a scorecard row |
| `CLAUDE.md` | Deep technical reference: architecture, RL formulation, data notes, file map. | Only when the underlying facts change |
| `common/features.py` | The 6 canonical state features, action set, CSV schema. Single source of truth for both GPU trees. | Rarely — this is meant to be stable |
| `common/gpu_probe.py` | Capability probe, run first on any new machine. | Rarely |

**Do not let a status block creep back into CLAUDE.md.** If you're tempted to write
"current status" there, it belongs in this file instead.

## Open questions nobody has answered yet

- Does clock control actually work inside a rented vast.ai container? (This is *the*
  question the probe answers — see NORTHSTAR.md's "one thing everything depends on.")
- How many GPUs, and which models, will actually get rented/borrowed? Determines whether
  the paper claims cross-*generation* transfer (3000→4000) or has to fall back to
  cross-*architecture* transfer (Turing→Ampere→Ada) on whatever's physically available.
- Who fixes the RTX 3050 Ti's deleted do-no-harm arbiter, and when — it's a regression
  from before this session started, flagged but not yet assigned.

## Session log

- **23 Aug 2026** — Reviewed 6 new commits from teammates (A1 timing fix confirmed real
  and produced a physically coherent EDP gradient; 3 dead workloads dropped; state
  redesign done on 3050 Ti only). Found the deleted-arbiter regression. Built
  `common/features.py` + `common/gpu_probe.py`, found and fixed two bugs in the probe
  itself (nvidia-smi exits 0 on unsupported operations; NVML power-limit call fails on
  some cards where the CLI still works). Set up this three-file doc structure. Created
  `unified-model` branch. Verified remote-GPU-access plan is technically sound before the
  user spends money renting anything.
