import cupy as cp
import time
import sys

workload = sys.argv[1]
if workload == 'matmul':
    A = cp.random.rand(8000, 8000, dtype=cp.float32)
    B = cp.random.rand(8000, 8000, dtype=cp.float32)
    for _ in range(20): cp.matmul(A, B)

elif workload == 'fft_fixed':
    # dvfs-triage A2: the original 'fft' used a 2048^3 complex64 array
    # (~68.7GB) which OOMs immediately on this card's 24GB VRAM every time --
    # it never once actually ran. 640^3 (~2.1GB) is a real, safe size that
    # exercises the same FFT kernels.
    A = (cp.random.rand(640, 640, 640, dtype=cp.float32)
         + 1j * cp.random.rand(640, 640, 640, dtype=cp.float32)).astype(cp.complex64)
    for _ in range(30): cp.fft.fftn(A)
    cp.cuda.Stream.null.synchronize()

elif workload == 'crypto_proxy':
    A = cp.random.randint(0, 2**31, size=(20000, 20000), dtype=cp.uint32)
    for _ in range(30): cp.bitwise_xor(A, cp.left_shift(A, 3))
