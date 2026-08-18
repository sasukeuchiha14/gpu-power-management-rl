import torch
import sys

device = torch.device('cuda')
workload = sys.argv[1]

if workload == 'resnet18':
    from torchvision.models import resnet18
    model = resnet18(weights=None).to(device)
    data = torch.randn(64, 3, 224, 224, device=device)
    for _ in range(40): model(data)

elif workload == 'transformer':
    layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True).to(device)
    model = torch.nn.TransformerEncoder(layer, num_layers=6).to(device)
    data = torch.randn(32, 128, 512, device=device)
    for _ in range(20): model(data)
