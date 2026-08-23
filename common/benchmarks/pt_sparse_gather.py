import torch
import time
import gc


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA is not available!")

    gc.collect()
    torch.cuda.empty_cache()

    # Sparse/irregular memory access via random-index gathers -- unlike
    # vector_add's sequential access, this stresses the memory subsystem the
    # way SpMV / graph-traversal workloads do. DVFS literature treats
    # irregular-access workloads as a distinct category from dense
    # elementwise ops (see e.g. GPGPU DVFS classification work covering
    # Gunrock/Pannotia-style graph benchmarks) -- nothing else in this suite
    # has this access pattern.
    n = 16_000_000
    table = torch.randn(n, device=device)
    idx_pool = [torch.randint(0, n, (n,), device=device) for _ in range(5)]

    # Warmup
    for i in range(3):
        _ = table[idx_pool[i % len(idx_pool)]]

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for i in range(40):
        _ = table[idx_pool[i % len(idx_pool)]]
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Duration: {duration:.4f}")

    del table, idx_pool
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    run()
