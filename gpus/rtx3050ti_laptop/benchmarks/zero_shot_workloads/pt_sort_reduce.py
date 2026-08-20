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

    # Sort + cumulative-sum chain -- unlike every trained memory-bound
    # workload, this has data-dependent memory access patterns.
    # ~80MB per tensor, safe for 4GB VRAM.
    num_elements = 20 * 1024 * 1024
    A = torch.randn(num_elements, device=device)

    # Warmup
    for _ in range(2):
        sorted_vals, _ = torch.sort(A)
        _ = torch.cumsum(sorted_vals, dim=0)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(15):
        sorted_vals, _ = torch.sort(A)
        _ = torch.cumsum(sorted_vals, dim=0)
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Duration: {duration:.4f}")

    # Explicit cleanup
    del A, sorted_vals
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()
