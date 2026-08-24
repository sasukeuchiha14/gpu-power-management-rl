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
    # Window markers: the collector samples NVML across this whole process,
    # which includes interpreter start, `import torch` and CUDA context
    # creation -- seconds of idle GPU. Averaging telemetry over all of it
    # made reported utilisation track how long the workload ran rather than
    # how hard it drove the GPU (measured corr +0.708; 62% of profiled
    # points reported <9% util). Emitting the measured window lets the
    # collector average only samples taken inside it. Printed AFTER the
    # timed region so it adds no overhead to the measurement itself.
    print(f"Window: {start:.6f} {start + duration:.6f}")

    # Explicit cleanup
    del A
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()
