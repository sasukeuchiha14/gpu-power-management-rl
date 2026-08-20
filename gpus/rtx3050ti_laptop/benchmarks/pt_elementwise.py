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

    # Memory-bandwidth-bound chain of elementwise ops, structurally unlike
    # the trained compute-bound pt_matmul or the FFT/vector_add pair.
    # 5000x5000 float32 ~= 100MB per tensor -- safe for 4GB VRAM.
    size = 5000
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(3):
        C = torch.sqrt(A * B + A) - torch.sin(B)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(60):
        C = torch.sqrt(A * B + A) - torch.sin(B)
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Duration: {duration:.4f}")

    # Explicit cleanup
    del A, B, C
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()
