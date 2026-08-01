import torch
import torch.nn as nn
import time
import gc
from torchvision.models import resnet50

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA is not available!")

    # Clear memory cache
    gc.collect()
    torch.cuda.empty_cache()

    model = resnet50().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Scaled down batch size for 4GB VRAM
    data = torch.randn(16, 3, 224, 224, device=device)
    targets = torch.randint(0, 1000, (16,), device=device)

    # Warmup
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(15):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Duration: {duration:.4f}")

    # Explicit cleanup
    del model, data, targets
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()
