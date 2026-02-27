import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InputEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels, base_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(base_channels, base_channels * 2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            SeparableConv2d(base_channels * 2, base_channels * 4),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            SeparableConv2d(base_channels * 4, base_channels * 8),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            SeparableConv2d(base_channels * 8, base_channels * 16),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            SeparableConv2d(base_channels * 16, base_channels * 32),
            nn.BatchNorm2d(base_channels * 32),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(base_channels * 32 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
