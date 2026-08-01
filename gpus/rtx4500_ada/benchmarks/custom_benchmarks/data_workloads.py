import cupy as cp
import time
import sys

workload = sys.argv[1]
if workload == 'matmul':
    A = cp.random.rand(8000, 8000, dtype=cp.float32)
    B = cp.random.rand(8000, 8000, dtype=cp.float32)
    for _ in range(20): cp.matmul(A, B)
    
elif workload == 'fft':
    A = cp.random.rand(2048, 2048, 2048, dtype=cp.complex64)
    for _ in range(5): cp.fft.fftn(A)
    
elif workload == 'crypto_proxy':
    A = cp.random.randint(0, 2**31, size=(20000, 20000), dtype=cp.uint32)
    for _ in range(30): cp.bitwise_xor(A, cp.left_shift(A, 3))
