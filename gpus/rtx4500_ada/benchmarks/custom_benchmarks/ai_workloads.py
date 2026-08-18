import torch
import time
import sys
from torchvision.models import resnet50, ResNet50_Weights, vgg16

device = torch.device('cuda')
workload = sys.argv[1]

if workload == 'resnet':
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    data = torch.randn(64, 3, 224, 224, device=device)
    for _ in range(50): model(data)

elif workload == 'bert':
    from transformers import BertModel, BertConfig
    config = BertConfig()
    model = BertModel(config).to(device)
    data = torch.randint(0, 2000, (32, 512), device=device)
    for _ in range(100): model(data)

elif workload == 'cnn':
    # Different compute pattern from resnet50 (no residual/BN skip paths,
    # much wider dense conv stack) -- a distinct point in the workload space.
    model = vgg16(weights=None).to(device)
    data = torch.randn(32, 3, 224, 224, device=device)
    for _ in range(15): model(data)

elif workload == 'lstm':
    # Recurrent, memory/latency-bound -- unlike every other workload here
    # (all feedforward/parallel), so it stresses a different part of the
    # power/frequency curve.
    model = torch.nn.LSTM(input_size=256, hidden_size=512, num_layers=3, batch_first=True).to(device)
    data = torch.randn(64, 200, 256, device=device)
    for _ in range(30): model(data)
