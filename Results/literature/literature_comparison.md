# Literature comparison — verified against primary sources

Previous version of this table (now superseded) pulled numbers secondhand from `paper_unified.tex`'s
own related-work prose. This version is built from actually reading each paper Gagan was able to
obtain, correcting several numbers and adding detail the secondhand summary didn't have. Source
PDFs: `C:\capstone\all\` and `C:\capstone\ma'am docs\` (8 Oct 2026).

**Status: 8 of 11 bibliography entries verified against the primary source. 3 remain unverified
(not available — see "Not verified" below).**

## Verified entries

### Zhao et al. 2023 (SoCC) — GPU power capping at HPC scale
Numbers are far more specific than the draft's "not quantified":
- **200W cap** (the practical operating point): BERT ~15% energy reduction, DimeNet ~15-20%,
  ResNet50 ~10% — all with **negligible (<5%) training-speed impact**.
- **100W cap** (aggressive): 40-60% energy reduction, but **30-40% speed degradation** — a much
  worse tradeoff, showing the method doesn't scale gracefully to stricter caps.
- **LLaMA 65B inference @175W cap**: 22-24% energy savings, 5-8% speed reduction.
- Datacenter-scale fleet average: ~10W power reduction, ~4.8°C temperature reduction per GPU;
  for high-utilization jobs (>70%) specifically: ~13.3W, ~7.5°C.
- Method: static power capping (coarse-grained, one ceiling for the whole job), not continuous
  frequency control — this project's clock-level, per-tick control is a finer-grained mechanism
  than anything tested here.

### Wang et al. 2024 — DRLCap (IEEE Trans. Sustainable Computing)
Headline numbers (22% energy, <3% perf loss, 20% EDP) match the draft, but with far more
detail now available:
- Per-GPU breakdown: **V100S** 27% energy / 2.4% loss / 23% EDP (best case); **RTX 3080 Ti** 19%/3%/17%;
  **RTX 2080 Ti** 22%/4.5%/19%; **AMD RX 6700 XT** 10%/4%/7.8% (worst case, different vendor).
- System-level (including CPU/memory overhead, not just GPU): 17% energy / 2.4% loss / 12% EDP.
- Beats a static heuristic baseline (COORD: 25%/5%/21%) by 8 points of EDP.
- Beats a naive RL baseline (double Q-learning ported from prior CPU work: only 5%/1%/4%) by a
  wide margin — evidence that the DRL formulation specifically matters, not just "some RL."
- Sampling/control overhead: 0.2% time, 0.3% energy (negligible).
- **Critical limitation, not in the draft's summary: DRLCap's own paper states it must be
  retrained when moving to a new GPU architecture.** This is the single sharpest point of
  contrast with this project's headline claim (one trained policy, zero-shot on a held-out GPU
  architecture it has never seen) — worth stating explicitly in the paper.

### Ye et al. 2025 — AGFT (arXiv:2508.01744)
- **Steady-state** (post-convergence): 44.3% energy saving, 40.3% EDP reduction — but at a
  **9.3% TTFT (time-to-first-token) and 7.1% TPOT (time-per-output-token) latency cost**, not
  the vaguer "under 10%" framing. This is a real, non-trivial latency tradeoff for an
  inference-serving system.
- **12-hour long-term average** (includes the method's own online learning/exploration phase):
  only 30.9% energy / 26.1% EDP — meaningfully lower than the steady-state number, because the
  bandit's exploration cost is a real, amortized overhead the steady-state figure hides.
- Method: **contextual multi-armed bandit (LinUCB)**, not deep RL — a lighter-weight class of
  method than DRLCap or this project's DDQN.
- Scope: single GPU (A6000), LLM-inference workloads only, optimal frequency varies 1200-1395MHz
  across different inference phases (prefill vs. decode) — tightly coupled to that domain, as
  the draft already said, now with the specific mechanism (phase-dependent optimal frequency).

### Li et al. 2024 (arXiv:2409.19434) — DDQN DVFS for edge/multi-task systems
- **Scope correction**: this targets an **NVIDIA Jetson TX2 embedded edge device**, not a
  discrete desktop/server GPU. Joint CPU+GPU frequency scaling via DDQN, deadline-constrained
  multi-task scheduling — a different product category from this project's RTX 3000/4000
  consumer-desktop/laptop fleet.
- ~3-10% power/energy savings while meeting task deadlines (direction matches the draft).

### Subashree et al. 2024/2025 (IGI Global book chapter)
Only a 2-page preview is accessible (paywalled). Confirms the draft's framing: a broad survey of
AI-for-HPC-energy approaches, no specific GPU DVFS mechanism proposed or evaluated. Nothing to
correct here — just unable to extract more without the full chapter.

### Divyaprabha & Sudarshan — three papers, all verified (full PDFs + the author's own PhD thesis
defense deck for cross-checking consistency)
**Important scope correction not in the draft**: all three of these are **CloudSim/WorkflowSim
simulation studies of CPU-GPU task-scheduling and partitioning** within a cloud/VM environment —
the "DVFS" here is a simplified analytical cubic power model (`P = β·f³`) used inside a scheduling
optimization, evaluated entirely in simulation. **None of these three papers does real hardware
GPU clock control.** This is a fundamentally different category of contribution from Zhao/DRLCap/
AGFT/Li (all four of which control real silicon) and from this project. The draft's one-line
summary didn't make this distinction clear — worth stating explicitly in the paper so the
comparison table doesn't imply a false apples-to-apples.

- **J1 — Divyaprabha & Sudarshan, 2024 (Springer LNNS, "Energy Aware Workload Scheduling
  Metrics...")**: EOS scheduler vs. an energy-minimized baseline (EMS), tested only on the
  Montage workflow (I/O-intensive). **23.22% energy reduction, 85.06% overall power reduction,
  78.2% makespan reduction.**
- **J2 — Divyaprabha & Sudarshan, 2025 (IJECE, Q3, "Optimal task partitioning...")**: TFM-OTP
  scheduler vs. EMS, tested on SIPHT and CyberShake workflows. **39.35% energy reduction** (both
  workflows), makespan reduction 44.23% (SIPHT) / 97.33% (CyberShake), task-failure energy
  overhead reduction 69.14% (SIPHT) / 98.26% (CyberShake).
- **J3 — Divyaprabha & Sudarshan, 2025 (Computers and Electrical Engineering, Q1,
  "Energy-deadline optimization...")** — the most complete of the three, two proposed schedulers
  (EDATSO, TFMOS) against two baselines (EMS, and MOPWSDRL — a deep-RL scheduler): on Inspiral,
  energy -41.4%/-33.1%, makespan -83.5%/-65.0%, task-failure overhead -92.6%/-76.7% (EDATSO/TFMOS
  vs. EMS/MOPWSDRL respectively); on CyberShake, energy -39.3%/-59.7%, makespan -97.3%/-42.9%,
  task-failure overhead -98.5%/-98.6%. Notably, **MOPWSDRL (the deep-RL baseline they beat) is
  itself a scheduling-level RL method, not a DVFS controller** — another reason not to place it
  in the same table row as DRLCap/AGFT/this project without the caveat above.

## Not verified — papers not shared, numbers remain secondhand from the draft

These three are likely behind ACM/IEEE paywalls; worth asking your institution's library access
or checking if a free preprint exists, but not independently confirmed:

- **Desai et al. 2025 (HPDC)** — adaptive GPU power capping. Draft claims: 12.9% energy savings,
  11.4% temperature reduction, tree-based (XGBoost/CatBoost/RF) cap prediction.
- **Wang et al. 2024 — DSO (IWQoS)** — fuses dynamic+static program features via an MLP. Draft
  claims: ~19% efficiency improvement within a 5% performance-loss bound, memory frequency held
  fixed during evaluation.
- **Wang et al. 2021 — GPOEO (IEEE TPDS)** — hardware-performance-counter profiling + gradient
  boosting + local search. Draft claims: 16.2% mean energy saving at 5.1% execution-time cost
  across 71 workloads, single GPU platform, requires an offline training stage.

## What this means for the paper's related-work section

1. **Split the table into two tiers**, not one flat list: (a) real-hardware GPU frequency/power
   controllers (Zhao, DRLCap, AGFT, Li, this work) and (b) simulated cloud task-scheduling with
   an analytical DVFS power model (all three Divyaprabha papers, and MOPWSDRL as their RL
   baseline). Conflating them overstates how directly comparable the numbers are.
2. **DRLCap's per-GPU retraining requirement is the sharpest, most citable point of contrast**
   for this project's cross-GPU generalization claim — stronger than anything already in the
   draft.
3. **AGFT's split between steady-state and 12-hour-average numbers** is worth citing as
   precedent for why this project should also report both single-trial and (once available)
   multi-rep numbers rather than only the best case.
4. Pending: Gagan's own multi-workload verification results (`verification/results/`) still need
   2-3 repeats before they're citable alongside any of the above.
