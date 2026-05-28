import torch
from torch import nn

from src.model_unet import DoubleConv


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = self.relu(self.gate_proj(gate) + self.skip_proj(skip))
        attention = self.psi(attention)
        return skip * attention


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.down1 = DoubleConv(in_channels, c)
        self.down2 = DoubleConv(c, c * 2)
        self.down3 = DoubleConv(c * 2, c * 4)
        self.down4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(gate_channels=c * 8, skip_channels=c * 8, inter_channels=c * 4)
        self.conv4 = DoubleConv(c * 16, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(gate_channels=c * 4, skip_channels=c * 4, inter_channels=c * 2)
        self.conv3 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(gate_channels=c * 2, skip_channels=c * 2, inter_channels=c)
        self.conv2 = DoubleConv(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.att1 = AttentionGate(gate_channels=c, skip_channels=c, inter_channels=max(c // 2, 1))
        self.conv1 = DoubleConv(c * 2, c)
        self.out = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        x = self.bottleneck(self.pool(d4))

        x = self.up4(x)
        d4 = self.att4(x, d4)
        x = self.conv4(torch.cat([x, d4], dim=1))

        x = self.up3(x)
        d3 = self.att3(x, d3)
        x = self.conv3(torch.cat([x, d3], dim=1))

        x = self.up2(x)
        d2 = self.att2(x, d2)
        x = self.conv2(torch.cat([x, d2], dim=1))

        x = self.up1(x)
        d1 = self.att1(x, d1)
        x = self.conv1(torch.cat([x, d1], dim=1))
        return self.out(x)
