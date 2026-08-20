import torch
import time
import gc
from torchvision.models import squeezenet1_1

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA is not available!")

    # Clear memory cache
    gc.collect()
    torch.cuda.empty_cache()

    # Fire-module CNN -- structurally different from the trained
    # pt_resnet50 (residual) and pt_mobilenet (depthwise-separable).
    model = squeezenet1_1().to(device)
    model.eval()
    data = torch.randn(16, 3, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(data)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(50):
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
