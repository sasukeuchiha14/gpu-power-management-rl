import torch
import time
import gc
from torchvision.models import vit_b_16

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA is not available!")

    # Clear memory cache
    gc.collect()
    torch.cuda.empty_cache()

    # Vision Transformer -- attention-based image processing, never
    # trained on. Scaled down for 4GB VRAM (small batch; ~86M params).
    model = vit_b_16().to(device)
    model.eval()
    data = torch.randn(8, 3, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(data)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(20):
            model(data)
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Duration: {duration:.4f}")

    # Explicit cleanup
    del model, data
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()
