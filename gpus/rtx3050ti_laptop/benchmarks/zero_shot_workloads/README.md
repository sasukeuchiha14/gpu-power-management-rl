# Zero-shot / unseen workloads

**Do not add these to `data_collection/generate_dataset.py`'s `BENCHMARKS`
list.** Their entire purpose is to stay outside the training data, so a
live A/B test against them measures how the DDQN behaves on an application
it never profiled -- exactly the scenario the state redesign (6
device-normalised features, no workload one-hot) was meant to make
possible. Mirrors `gpus/rtx4500_ada/benchmarks/zero_shot_workloads/` --
same idea, laptop-appropriate sizes (4GB VRAM budget).

- `pt_vit.py` -- Vision Transformer (`vit_b_16`). Combines the image domain
  with attention-based compute, unlike anything in the trained set (which
  has CNNs and one NLP transformer, but no vision transformer).
- `pt_squeezenet.py` -- a CNN built from Fire modules, structurally
  different from the trained `pt_resnet50`/`pt_mobilenet` (neither
  residual connections nor depthwise-separable convs).
- `pt_sort_reduce.py` -- sort + cumulative-sum chain. Unlike every trained
  memory-bound workload (`pt_vector_add`, `pt_elementwise`, `pt_fft`), this
  one has data-dependent memory access patterns (the sort's memory
  accesses depend on the data values, not just its shape).

Test with (from `evaluation/`):
```cmd
python measure_edp.py governor 10 pt_vit
python measure_edp.py ai 10 pt_vit
python plot_edp.py pt_vit
```
