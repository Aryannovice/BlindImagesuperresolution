from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class UNetEncoder(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 64):
        super().__init__()
        self.down1 = nn.Sequential(ConvBlock(in_ch, base_ch), ConvBlock(base_ch, base_ch))
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(ConvBlock(base_ch, base_ch * 2), ConvBlock(base_ch * 2, base_ch * 2))
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = nn.Sequential(ConvBlock(base_ch * 2, base_ch * 4), ConvBlock(base_ch * 4, base_ch * 4))
        self.pool3 = nn.MaxPool2d(2)
        self.bottom = nn.Sequential(ConvBlock(base_ch * 4, base_ch * 8), ConvBlock(base_ch * 8, base_ch * 8))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        xb = self.bottom(self.pool3(x3))
        return x1, x2, x3, xb


class UNetDecoder(nn.Module):
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock(base_ch * 8, base_ch * 4), ConvBlock(base_ch * 4, base_ch * 4))
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(base_ch * 4, base_ch * 2), ConvBlock(base_ch * 2, base_ch * 2))
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock(base_ch * 2, base_ch), ConvBlock(base_ch, base_ch))
        self.final_conv = nn.Conv2d(base_ch, 3, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        x = self.up3(xb)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.final_conv(x)


class UNetSR(nn.Module):
    """
    U-Net backbone for dual-image SR (input: 6 channels -> output: 3 channels).
    """

    def __init__(self, in_ch: int = 6, base_ch: int = 64):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, base_ch)
        self.decoder = UNetDecoder(base_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, xb = self.encoder(x)
        out = self.decoder(x1, x2, x3, xb)
        return torch.sigmoid(out)


class CustomSRNet(nn.Module):
    """
    Lightweight custom CNN for SR, suitable for speed / ablation baselines.
    """

    def __init__(self, in_ch: int = 6, base_ch: int = 64, num_blocks: int = 4):
        super().__init__()
        layers = [ConvBlock(in_ch, base_ch)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBlock(base_ch, base_ch))
        self.backbone = nn.Sequential(*layers)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        out = self.upsample(feat)
        return torch.sigmoid(out)


class VGGSRSimple(nn.Module):
    """
    VGG16-based SR model: use pretrained VGG backbone as feature extractor,
    then upsample to HR resolution.
    """

    def __init__(self, in_ch: int = 6, freeze_backbone: bool = True):
        super().__init__()
        # VGG expects 3-channel input; we project 6->3 first, then re-project features
        self.input_proj = nn.Conv2d(in_ch, 3, kernel_size=1)
        vgg = vgg16_bn(weights="IMAGENET1K_V1")
        self.features = vgg.features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.sr_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        feat = self.features(x)
        out = self.sr_head(feat)
        return torch.sigmoid(out)


# === Simple SRGAN-style generator & discriminator ===


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SRGANGenerator(nn.Module):
    """
    Simplified SRGAN generator for 4x super-resolution.
    Input: dual-image fused tensor (6 channels).
    """

    def __init__(self, in_ch: int = 6, num_residual_blocks: int = 8, upscale_factor: int = 4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        upsample_layers = []
        num_upsamples = int(upscale_factor // 2)
        for _ in range(num_upsamples):
            upsample_layers += [
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]
        self.upsample = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.res_blocks(x1)
        x3 = self.conv2(x2)
        x = x1 + x3
        x = self.upsample(x)
        x = self.conv3(x)
        return torch.sigmoid(x)


class SRGANDiscriminator(nn.Module):
    """
    Patch-based discriminator for SRGAN, operating on HR images.
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        layers = []
        channels = [in_ch, 64, 64, 128, 128, 256, 256, 512, 512]
        strides = [1, 2, 1, 2, 1, 2, 1, 2]

        for i in range(1, len(channels)):
            layers.append(
                nn.Conv2d(
                    channels[i - 1],
                    channels[i],
                    kernel_size=3,
                    stride=strides[i - 1],
                    padding=1,
                )
            )
            if i > 1:
                layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


def build_fused_tensor(lr1: torch.Tensor, lr2: torch.Tensor) -> torch.Tensor:
    """
    Concatenate two LR images along channel dim: [B, 3, H, W] + [B, 3, H, W] -> [B, 6, H, W]
    """
    return torch.cat([lr1, lr2], dim=1)

