import torch
import torch.nn as nn
from .encoder import InputEncoder


class SeparableConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.depthwise = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pointwise = nn.ConvTranspose2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        output_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()

        self.fc = nn.Linear(latent_dim, base_channels * 32 * 4 * 4)

        self.decoder = nn.Sequential(
            SeparableConvTranspose2d(
                base_channels * 32,
                base_channels * 16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            SeparableConvTranspose2d(
                base_channels * 16,
                base_channels * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            SeparableConvTranspose2d(
                base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            SeparableConvTranspose2d(
                base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            SeparableConvTranspose2d(
                base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels, output_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.decoder(x)
        return x


class PCBAutoencoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 64,
    ):
        super().__init__()

        self.encoder = InputEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_channels=input_channels,
            base_channels=base_channels,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed
