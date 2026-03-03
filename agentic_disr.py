from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import ProjectConfig
from datasets import DualImageSample
from losses import CompositeLoss
from metrics import niqe_score, piqe_score, full_metrics
from models import (
    UNetSR,
    VGGSRSimple,
    CustomSRNet,
    SRGANGenerator,
    SRGANDiscriminator,
    build_fused_tensor,
)
from unlearning import run_unlearning_step


ModelName = Literal["custom_cnn", "unet", "vgg16", "srgan"]


@dataclass
class AgenticState:
    config: ProjectConfig
    device: torch.device
    models: Dict[ModelName, nn.Module]
    discriminator: nn.Module | None
    optimizers: Dict[ModelName, optim.Optimizer]
    composite_loss: CompositeLoss


def build_models(cfg: ProjectConfig) -> AgenticState:
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    models: Dict[ModelName, nn.Module] = {
        "custom_cnn": CustomSRNet(in_ch=6),
        "unet": UNetSR(in_ch=6),
        "vgg16": VGGSRSimple(in_ch=6, freeze_backbone=True),
        "srgan": SRGANGenerator(in_ch=6),
    }
    for m in models.values():
        m.to(device)

    discriminator: nn.Module | None = SRGANDiscriminator(in_ch=3).to(device)

    optimizers: Dict[ModelName, optim.Optimizer] = {}
    for name, m in models.items():
        # GAN generator & others share base LR
        optimizers[name] = optim.Adam(
            m.parameters(),
            lr=cfg.train.base_lr,
            betas=(0.9, 0.999),
        )

    composite_loss = CompositeLoss(
        mse_weight=cfg.loss.mse_weight,
        perceptual_weight=cfg.loss.perceptual_weight,
        edge_weight=cfg.loss.edge_weight,
    ).to(device)

    return AgenticState(
        config=cfg,
        device=device,
        models=models,
        discriminator=discriminator,
        optimizers=optimizers,
        composite_loss=composite_loss,
    )


def scene_statistics(sample: DualImageSample) -> Dict[str, float]:
    """
    Compute simple stats from the LR pair for agentic model selection.
    """
    lr1 = sample.lr1
    lr2 = sample.lr2
    pair = torch.stack([lr1, lr2], dim=0)  # [2, 3, H, W]
    mean_brightness = float(pair.mean().item())
    entropy = float(-(pair * (pair + 1e-8).log()).sum().item())

    # crude "cloud" ratio: fraction of very bright pixels
    bright = (pair > 0.9).float().mean().item()

    return {"brightness": mean_brightness, "entropy": entropy, "cloud_ratio": bright}


def select_model_name(stats: Dict[str, float]) -> ModelName:
    """
    Heuristic agentic selector based on scene statistics.
    """
    brightness = stats["brightness"]
    entropy = stats["entropy"]
    cloud = stats["cloud_ratio"]

    if cloud > 0.2:
        # more clouds -> U-Net tends to denoise better
        return "unet"
    if entropy > 0.5 and brightness > 0.4:
        # sharp, high-contrast urban scenes -> VGG16
        return "vgg16"
    if entropy < 0.2:
        # smooth areas -> custom lightweight CNN
        return "custom_cnn"
    # otherwise, demonstrate GAN usage
    return "srgan"


def forward_model(state: AgenticState, name: ModelName, sample: DualImageSample) -> torch.Tensor:
    fused = build_fused_tensor(
        sample.lr1.unsqueeze(0).to(state.device),
        sample.lr2.unsqueeze(0).to(state.device),
    )
    return state.models[name](fused)


def agentic_step(
    state: AgenticState,
    sample: DualImageSample,
    unlearning: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Single agentic step:
      1. Select model based on LR stats.
      2. Forward, compute composite loss, update.
      3. Compute NIQE/PIQE on output.
      4. If poor quality and unlearning=True, run selective fine-tuning.
    """
    stats = scene_statistics(sample)
    model_name = select_model_name(stats)
    model = state.models[model_name]
    model.train()

    hr = sample.hr.unsqueeze(0).to(state.device)

    if model_name == "srgan" and state.discriminator is not None:
        # simple SRGAN step
        g_opt = state.optimizers["srgan"]
        d_opt = optim.Adam(state.discriminator.parameters(), lr=state.config.train.base_lr, betas=(0.9, 0.999))

        pred = forward_model(state, "srgan", sample)

        # train discriminator
        state.discriminator.train()
        d_opt.zero_grad()
        real_logits = state.discriminator(hr)
        fake_logits = state.discriminator(pred.detach())
        d_loss_real = torch.mean((real_logits - 1) ** 2)
        d_loss_fake = torch.mean(fake_logits ** 2)
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        d_opt.step()

        # train generator
        g_opt.zero_grad()
        fake_logits = state.discriminator(pred)
        adv_loss = torch.mean((fake_logits - 1) ** 2)
        comp_losses = state.composite_loss(pred, hr)
        total_loss = comp_losses["total"] + 1e-3 * adv_loss
        total_loss.backward()
        g_opt.step()

        with torch.no_grad():
            niqe_val = niqe_score(pred.clamp(0.0, 1.0))
            try:
                piqe_val = piqe_score(pred.clamp(0.0, 1.0))
            except RuntimeError:
                piqe_val = float("nan")

        metrics = {"niqe": niqe_val, "piqe": piqe_val}
        return pred.detach(), metrics

    # Non-GAN models: standard composite loss step
    opt = state.optimizers[model_name]
    opt.zero_grad()
    pred = forward_model(state, model_name, sample)
    comp_losses = state.composite_loss(pred, hr)
    comp_losses["total"].backward()
    opt.step()

    with torch.no_grad():
        niqe_val = niqe_score(pred.clamp(0.0, 1.0))
        try:
            piqe_val = piqe_score(pred.clamp(0.0, 1.0))
        except RuntimeError:
            piqe_val = float("nan")

    # Unlearning controller: selective fine-tuning if quality poor
    if unlearning and (
        niqe_val > state.config.agentic.niqe_threshold
        or (not torch.isnan(torch.tensor(piqe_val)) and piqe_val > state.config.agentic.piqe_threshold)
    ):
        model = state.models[model_name]
        pred_u, metrics_u = run_unlearning_step(
            cfg=state.config,
            model=model,
            sample=sample,
            device=state.device,
            composite_loss=state.composite_loss,
            model_name=model_name,
        )
        pred = pred_u.detach()
        niqe_val, piqe_val = metrics_u["niqe"], metrics_u["piqe"]

    metrics = {"niqe": niqe_val, "piqe": piqe_val}
    return pred.detach(), metrics


def evaluate_batch(state: AgenticState, batch: DualImageSample, model_name: ModelName) -> Dict[str, float]:
    with torch.no_grad():
        fused = build_fused_tensor(
            batch.lr1.unsqueeze(0).to(state.device),
            batch.lr2.unsqueeze(0).to(state.device),
        )
        hr = batch.hr.unsqueeze(0).to(state.device)
        pred = state.models[model_name](fused)
        return full_metrics(pred, hr)

