from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from piq import niqe

try:
    # third-party PIQE implementation
    from piqe import piqe as _piqe_fn  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _piqe_fn = None

from skimage.metrics import structural_similarity as sk_ssim
import numpy as np


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0, device=pred.device)
    return 20 * torch.log10(max_val) - 10 * torch.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute SSIM using skimage on CPU (per-image average).
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ssim_vals = []
    for i in range(pred_np.shape[0]):
        p = np.transpose(pred_np[i], (1, 2, 0))
        t = np.transpose(target_np[i], (1, 2, 0))
        ssim_vals.append(sk_ssim(t, p, channel_axis=-1, data_range=1.0))
    return float(np.mean(ssim_vals))


def niqe_score(img: torch.Tensor) -> float:
    """
    NIQE expects BCHW tensor in [0,1].
    Returns scalar float.
    """
    with torch.no_grad():
        score = niqe(img)
    return float(score.item())


def piqe_score(img: torch.Tensor) -> float:
    """
    PIQE from optional `piqe` library, expects HWC uint8.
    Uses first image of the batch.
    """
    if _piqe_fn is None:
        raise RuntimeError(
            "piqe package is not installed. Install with `pip install piqe` "
            "or remove PIQE usage from the agentic loop."
        )
    img_np = img[0].detach().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    return float(_piqe_fn(img_np))


def full_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Convenience wrapper to compute PSNR, SSIM for a batch.
    Uses the first image for NIQE/PIQE to keep runtime reasonable.
    """
    p = pred.clamp(0.0, 1.0)
    t = target.clamp(0.0, 1.0)

    psnr_val = float(psnr(p, t).item())
    ssim_val = ssim(p, t)

    with torch.no_grad():
        niqe_val = niqe_score(p[:1])
        try:
            piqe_val = piqe_score(p[:1])
        except RuntimeError:
            piqe_val = float("nan")

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "niqe": niqe_val,
        "piqe": piqe_val,
    }

