import torch
import torch.nn as nn
import torch.nn.functional as F

class FFT2DCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        # Input. (batch, 1, time=100, freq=64)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5,5), padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5,5), padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)  # (time, freq) -> 1x1
        self.fc = nn.Linear(32, n_classes)   # output. 2 classes

    def forward(self, x):
        # x: (batch, time, freq) -> (batch, 1, time, freq)
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

if __name__ == "__main__":
    x = torch.randn(32, 100, 64)  # batch=32, time=100, freq=64
    model = FFT2DCNN(n_classes=2)
    out = model(x)
    print(out.shape)  # torch.Size([32, 2])
