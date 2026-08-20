import torch
import time
import gc

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA is not available!")

    # Clear memory cache
    gc.collect()
    torch.cuda.empty_cache()

    size = 256
    A = torch.randn(size, size, size, dtype=torch.complex64, device=device)

    # Warmup
    for _ in range(2):
        torch.fft.fftn(A)

    # Benchmark. NOTE: at 15 iterations this workload's own measured Duration
    # was only ~0.07-0.12s -- under 2-3 samples of the 20Hz (50ms) NVML
    # profiler in generate_dataset.py, so the collected dataset showed ~0%
    # GPU utilization (dead-workload pattern, same class of bug as
    # dvfs-triage A2). 120 iterations brings it to the same measurable range
    # as the other short benchmarks (pt_vector_add, pt_crypto_proxy, both
    # ~0.4s) without changing the array size or VRAM footprint.
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(120):
        torch.fft.fftn(A)
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Duration: {duration:.4f}")

    # Explicit cleanup
    del A
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()
