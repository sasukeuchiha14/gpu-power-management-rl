import cupy as cp
import sys

workload = sys.argv[1]
if workload == 'elementwise':
    A = cp.random.rand(20000, 20000, dtype=cp.float32)
    B = cp.random.rand(20000, 20000, dtype=cp.float32)
    for _ in range(15):
        C = cp.sqrt(A * B + A) - cp.sin(B)
    cp.cuda.Stream.null.synchronize()
