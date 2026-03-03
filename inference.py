from __future__ import annotations

import os
from typing import List

import numpy as np
from PIL import Image
import torch

from config import ProjectConfig, ensure_dirs
from agentic_disr import build_models, scene_statistics, select_model_name, forward_model
from datasets import DualImageSample
from models import build_fused_tensor
from metrics import full_metrics


def save_image(t: torch.Tensor, path: str) -> None:
    t = t.clamp(0.0, 1.0).detach().cpu().numpy()
    t = np.transpose(t, (1, 2, 0))
    img = (t * 255.0).astype(np.uint8)
    Image.fromarray(img).save(path)


def run_inference_on_folder(
    lr1_paths: List[str],
    lr2_paths: List[str],
    out_dir: str,
    state_dict_path: str,
) -> None:
    cfg = ProjectConfig()
    ensure_dirs(cfg)
    state = build_models(cfg)

    checkpoint = torch.load(state_dict_path, map_location=state.device)
    for name in ["custom_cnn", "unet", "vgg16", "srgan"]:
        state.models[name].load_state_dict(checkpoint[name])
        state.models[name].eval()

    os.makedirs(out_dir, exist_ok=True)

    for p1, p2 in zip(lr1_paths, lr2_paths):
        name = os.path.basename(p1)
        img1 = Image.open(p1).convert("RGB").resize((cfg.data.img_size, cfg.data.img_size))
        img2 = Image.open(p2).convert("RGB").resize((cfg.data.img_size, cfg.data.img_size))
        arr1 = np.transpose(np.asarray(img1).astype(np.float32) / 255.0, (2, 0, 1))
        arr2 = np.transpose(np.asarray(img2).astype(np.float32) / 255.0, (2, 0, 1))
        lr1 = torch.from_numpy(arr1)
        lr2 = torch.from_numpy(arr2)

        sample = DualImageSample(lr1=lr1, lr2=lr2, hr=lr1)  # hr dummy for stats
        stats = scene_statistics(sample)
        model_name = select_model_name(stats)
        fused = build_fused_tensor(lr1.unsqueeze(0).to(state.device), lr2.unsqueeze(0).to(state.device))

        with torch.no_grad():
            pred = state.models[model_name](fused)[0]

        save_image(pred, os.path.join(out_dir, name))
        print(f"{name}: used model={model_name}")


if __name__ == "__main__":
    print(
        "This script is a helper for running inference.\n"
        "Import `run_inference_on_folder` from `inference.py` and call it with your LR paths "
        "and the path to `agentic_disr_best.pth`."
    )

