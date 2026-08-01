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

    size = 4096
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(3):
        torch.matmul(A, B)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(25):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Duration: {duration:.4f}")

    # Explicit cleanup
    del A, B
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()
