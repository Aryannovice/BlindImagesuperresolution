from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from config import ProjectConfig
from losses import CompositeLoss
from metrics import niqe_score, piqe_score
from datasets import DualImageSample
from models import build_fused_tensor


def run_unlearning_step(
    cfg: ProjectConfig,
    model: nn.Module,
    sample: DualImageSample,
    device: torch.device,
    composite_loss: CompositeLoss,
    model_name: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Generic unlearning controller:
      - freezes all parameters except a small "correction head"
      - fine-tunes that head with a small LR for a few steps
      - stops early when NIQE/PIQE fall below thresholds
    """
    hr = sample.hr.unsqueeze(0).to(device)

    # freeze all but last logical head
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "decoder"):
        target_params = model.decoder.parameters()
    elif hasattr(model, "sr_head"):
        target_params = model.sr_head.parameters()
    else:
        # fallback: last 2 parameter tensors
        target_params = list(model.parameters())[-2:]

    for p in target_params:
        p.requires_grad = True

    unlearn_opt = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.unlearning_lr,
        betas=(0.9, 0.999),
    )

    best_pred: torch.Tensor | None = None
    best_niqe = float("inf")
    best_piqe = float("inf")

    for _ in range(cfg.agentic.max_unlearning_steps):
        fused = build_fused_tensor(
            sample.lr1.unsqueeze(0).to(device),
            sample.lr2.unsqueeze(0).to(device),
        )
        unlearn_opt.zero_grad()
        pred_u = model(fused)
        losses_u = composite_loss(pred_u, hr)
        losses_u["total"].backward()
        unlearn_opt.step()

        with torch.no_grad():
            niqe_new = niqe_score(pred_u.clamp(0.0, 1.0))
            try:
                piqe_new = piqe_score(pred_u.clamp(0.0, 1.0))
            except RuntimeError:
                piqe_new = float("nan")

        if niqe_new < best_niqe:
            best_niqe = niqe_new
            best_piqe = piqe_new
            best_pred = pred_u.detach()

        if (
            niqe_new <= cfg.agentic.niqe_threshold
            and (torch.isnan(torch.tensor(piqe_new)) or piqe_new <= cfg.agentic.piqe_threshold)
        ):
            best_pred = pred_u.detach()
            best_niqe, best_piqe = niqe_new, piqe_new
            break

    # unfreeze all
    for p in model.parameters():
        p.requires_grad = True

    if best_pred is None:
        fused = build_fused_tensor(
            sample.lr1.unsqueeze(0).to(device),
            sample.lr2.unsqueeze(0).to(device),
        )
        with torch.no_grad():
            best_pred = model(fused)
            best_niqe = niqe_score(best_pred.clamp(0.0, 1.0))
            try:
                best_piqe = piqe_score(best_pred.clamp(0.0, 1.0))
            except RuntimeError:
                best_piqe = float("nan")

    return best_pred, {"niqe": best_niqe, "piqe": best_piqe}

