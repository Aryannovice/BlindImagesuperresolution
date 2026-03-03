from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn


class PerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss (content loss on feature maps).
    """

    def __init__(self, layer: str = "features_16"):
        super().__init__()
        vgg = vgg16_bn(weights="IMAGENET1K_V1").features
        # freeze
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()
        self.layer_name = layer

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        for name, module in self.vgg._modules.items():
            feat = module(feat)
            if f"features_{name}" == self.layer_name:
                break
        return feat

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = self._extract_features(pred)
        target_f = self._extract_features(target)
        return F.l1_loss(pred_f, target_f)


class EdgeLoss(nn.Module):
    """
    Edge-preserving loss using Sobel gradients.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradients(self, x: torch.Tensor) -> torch.Tensor:
        # convert to luminance
        if x.shape[1] == 3:
            x_gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        else:
            x_gray = x
        gx = F.conv2d(x_gray, self.sobel_x, padding=1)
        gy = F.conv2d(x_gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        g_pred = self._gradients(pred)
        g_target = self._gradients(target)
        return F.l1_loss(g_pred, g_target)


class CompositeLoss(nn.Module):
    """
    L_total = α * L_mse + β * L_perceptual + γ * L_edge
    """

    def __init__(self, mse_weight: float, perceptual_weight: float, edge_weight: float):
        super().__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight

        self.mse = nn.MSELoss()
        self.perceptual = PerceptualLoss()
        self.edge = EdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse_loss = self.mse(pred, target)
        perc_loss = self.perceptual(pred, target)
        edge_loss = self.edge(pred, target)
        total = (
            self.mse_weight * mse_loss
            + self.perceptual_weight * perc_loss
            + self.edge_weight * edge_loss
        )
        return {
            "total": total,
            "mse": mse_loss,
            "perceptual": perc_loss,
            "edge": edge_loss,
        }

