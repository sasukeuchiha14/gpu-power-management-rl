# Zero-shot / unseen workloads

**Do not add these to `data_collection/generate_8000_datasets.py`'s `BENCHMARKS`
list.** Their entire purpose is to stay outside the training data, so the
generalization test in `evaluation/measure_edp.py` (see `_WORKLOAD_TABLE`)
measures how the DDQN behaves on an application it never profiled -- exactly
the scenario dvfs-triage B1 warns about (the deployed daemon can't tell it's
looking at a genuinely new app; it just nearest-neighbour-matches to whichever
trained workload looks closest on the 5 live-observable features).

- `ai_workloads.py`: `resnet18` (different depth/width from the trained
  `pt_resnet50`/`pt_cnn`), `transformer` (attention-based -- the trained set's
  only NLP workload, `pt_bert`, is encoder-only BERT; this is a plain
  `nn.TransformerEncoder` stack, a different architecture family).
- `data_workloads.py`: `elementwise` -- a memory-bandwidth-bound chain of CuPy
  elementwise ops (sqrt/multiply/sin), unlike the trained set's compute-bound
  `cupy_matmul` or the also-memory-bound-but-structurally-different
  `cupy_fft_fixed`.
