from __future__ import annotations

import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ProjectConfig, ensure_dirs
from datasets import create_splits, DualImageSample
from agentic_disr import build_models, agentic_step
from metrics import psnr as psnr_fn
from visualization import plot_training_curves


def collate_fn(samples):
    # keep as simple struct of stacked tensors
    lr1 = torch.stack([s.lr1 for s in samples], dim=0)
    lr2 = torch.stack([s.lr2 for s in samples], dim=0)
    hr = torch.stack([s.hr for s in samples], dim=0)
    return DualImageSample(lr1=lr1, lr2=lr2, hr=hr)


def main():
    cfg = ProjectConfig()
    ensure_dirs(cfg)

    train_ds, val_ds, _ = create_splits(cfg.data)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,  # agentic_step works on single DualImageSample
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=lambda x: x[0],  # DataLoader already returns list of size 1
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=lambda x: x[0],
    )

    state = build_models(cfg)

    history = {
        "train_loss": [],
        "val_psnr": [],
    }
    best_psnr = 0.0
    for epoch in range(cfg.train.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.train.num_epochs}")
        state.models["custom_cnn"].train()
        state.models["unet"].train()
        state.models["vgg16"].train()
        state.models["srgan"].train()

        epoch_loss = 0.0
        for sample in tqdm(train_loader, desc="Train"):
            # sample is DualImageSample
            pred, _metrics = agentic_step(state, sample, unlearning=True)
            # we can't get loss directly from agentic_step without refactor;
            # approximate using MSE between pred and HR for logging only.
            hr = sample.hr.unsqueeze(0).to(state.device)
            with torch.no_grad():
                mse = torch.mean((pred.to(state.device) - hr) ** 2).item()
            epoch_loss += mse

        history["train_loss"].append(epoch_loss / max(len(train_loader), 1))

        # validation (use U-Net as reference for PSNR/SSIM)
        state.models["unet"].eval()
        psnr_vals = []
        for sample in tqdm(val_loader, desc="Val"):
            fused = torch.cat(
                [
                    sample.lr1.unsqueeze(0).to(state.device),
                    sample.lr2.unsqueeze(0).to(state.device),
                ],
                dim=1,
            )
            hr = sample.hr.unsqueeze(0).to(state.device)
            with torch.no_grad():
                pred = state.models["unet"](fused)
            psnr_vals.append(float(psnr_fn(pred, hr).item()))

        mean_psnr = sum(psnr_vals) / max(len(psnr_vals), 1)
        print(f"Validation PSNR (U-Net): {mean_psnr:.3f} dB")
        history["val_psnr"].append(mean_psnr)

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            ckpt_path = os.path.join(cfg.checkpoints_dir, "agentic_disr_best.pth")
            torch.save(
                {
                    "config": cfg.to_dict(),
                    "custom_cnn": state.models["custom_cnn"].state_dict(),
                    "unet": state.models["unet"].state_dict(),
                    "vgg16": state.models["vgg16"].state_dict(),
                    "srgan": state.models["srgan"].state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")

    # save training curves
    plot_training_curves(history, save_path=os.path.join(cfg.results_dir, "training_curves.png"))


if __name__ == "__main__":
    main()

