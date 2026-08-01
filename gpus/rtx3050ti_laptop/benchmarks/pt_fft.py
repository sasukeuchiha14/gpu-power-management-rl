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

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(15):
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
