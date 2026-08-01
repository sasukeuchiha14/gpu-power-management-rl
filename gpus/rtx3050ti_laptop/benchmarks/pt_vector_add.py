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

    # ~256MB per tensor (64M float32 elements)
    num_elements = 64 * 1024 * 1024
    A = torch.randn(num_elements, device=device)
    B = torch.randn(num_elements, device=device)

    # Warmup
    for _ in range(3):
        C = A + B

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(80):
        C = A + B
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Duration: {duration:.4f}")

    # Explicit cleanup
    del A, B, C
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()
