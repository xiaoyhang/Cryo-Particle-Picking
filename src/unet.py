"""A compact U-Net for particle-center heatmap prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.bottleneck = ConvBlock(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)
        self.dec4 = ConvBlock(b * 16 + b * 8, b * 8)
        self.dec3 = ConvBlock(b * 8 + b * 4, b * 4)
        self.dec2 = ConvBlock(b * 4 + b * 2, b * 2)
        self.dec1 = ConvBlock(b * 2 + b, b)
        self.output = nn.Conv2d(b, 1, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.dec4(torch.cat([self._up(x, e4), e4], dim=1))
        x = self.dec3(torch.cat([self._up(x, e3), e3], dim=1))
        x = self.dec2(torch.cat([self._up(x, e2), e2], dim=1))
        x = self.dec1(torch.cat([self._up(x, e1), e1], dim=1))
        return self.output(x)


if __name__ == "__main__":
    model = UNet()
    sample = torch.randn(1, 1, 512, 512)
    print("Output shape:", tuple(model(sample).shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

