from __future__ import annotations

import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import DualImageSample
from models import build_fused_tensor


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str | None = None,
) -> None:
    """
    Plot loss / PSNR / SSIM curves from a training history dict.

    Expected keys (optional):
      - "train_loss"
      - "val_psnr"
      - "val_ssim"
    """
    plt.figure(figsize=(10, 4))

    if "train_loss" in history:
        plt.plot(history["train_loss"], label="Train loss")
    if "val_psnr" in history:
        plt.plot(history["val_psnr"], label="Val PSNR (dB)")
    if "val_ssim" in history:
        plt.plot(history["val_ssim"], label="Val SSIM")

    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

    plt.close()


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """
    [C, H, W] in [0,1] -> uint8 HxWx3.
    """
    t = t.clamp(0.0, 1.0).detach().cpu().numpy()
    t = np.transpose(t, (1, 2, 0))
    img = (t * 255.0).astype(np.uint8)
    return img


def visualize_triplet(
    sample: DualImageSample,
    pred: torch.Tensor,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """
    Show LR1, LR2, and predicted HR side by side.
    """
    lr1_img = tensor_to_image(sample.lr1)
    lr2_img = tensor_to_image(sample.lr2)
    pred_img = tensor_to_image(pred)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(lr1_img)
    axs[0].set_title("LR view 1")
    axs[0].axis("off")

    axs[1].imshow(lr2_img)
    axs[1].set_title("LR view 2")
    axs[1].axis("off")

    axs[2].imshow(pred_img)
    axs[2].set_title("Agentic SR output")
    axs[2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

    plt.close()


def visualize_with_hr(
    sample: DualImageSample,
    pred: torch.Tensor,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """
    Show LR1, LR2, predicted HR, and reference HR for qualitative comparison.
    """
    lr1_img = tensor_to_image(sample.lr1)
    lr2_img = tensor_to_image(sample.lr2)
    pred_img = tensor_to_image(pred)
    hr_img = tensor_to_image(sample.hr)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(lr1_img)
    axs[0].set_title("LR view 1")
    axs[0].axis("off")

    axs[1].imshow(lr2_img)
    axs[1].set_title("LR view 2")
    axs[1].axis("off")

    axs[2].imshow(pred_img)
    axs[2].set_title("Agentic SR output")
    axs[2].axis("off")

    axs[3].imshow(hr_img)
    axs[3].set_title("HR reference")
    axs[3].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=220)
    else:
        plt.show()

    plt.close()

