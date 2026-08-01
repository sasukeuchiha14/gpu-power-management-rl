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

    # ~240MB VRAM
    num_elements = 30 * 1024 * 1024
    A = torch.randint(0, 2**30, (num_elements,), dtype=torch.int64, device=device)

    # Warmup
    for _ in range(3):
        A = (A ^ (A << 3)) % 1000003

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(40):
        A = (A ^ (A << 3)) % 1000003
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Duration: {duration:.4f}")

    # Explicit cleanup
    del A
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()
