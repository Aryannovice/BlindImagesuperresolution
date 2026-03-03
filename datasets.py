from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split

from config import DataConfig


def _list_sorted_images(directory: str) -> List[str]:
    files = [f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    files.sort()
    return files


@dataclass
class DualImageSample:
    lr1: torch.Tensor
    lr2: torch.Tensor
    hr: torch.Tensor


class WorldStratDualImageDataset(Dataset):
    """
    Dataset for Dual Image Super-Resolution (WorldStrat-like layout).

    Expected folder layout:
        root/
          lr_1/
            xxx.png
          lr_2/
            xxx.png
          hr/
            xxx.png
    """

    def __init__(self, cfg: DataConfig, split: str = "train"):
        self.cfg = cfg
        self.root = cfg.worldstrat_root
        self.lr1_dir = os.path.join(self.root, cfg.lr_dir_1)
        self.lr2_dir = os.path.join(self.root, cfg.lr_dir_2)
        self.hr_dir = os.path.join(self.root, cfg.hr_dir)

        assert os.path.isdir(self.lr1_dir), f"LR1 directory not found: {self.lr1_dir}"
        assert os.path.isdir(self.lr2_dir), f"LR2 directory not found: {self.lr2_dir}"
        assert os.path.isdir(self.hr_dir), f"HR directory not found: {self.hr_dir}"

        lr1_files = _list_sorted_images(self.lr1_dir)
        lr2_files = _list_sorted_images(self.lr2_dir)
        hr_files = _list_sorted_images(self.hr_dir)

        assert lr1_files == lr2_files == hr_files, "LR1, LR2, HR file lists must match and be aligned."

        self.files = lr1_files
        self.split = split

    def __len__(self) -> int:
        return len(self.files)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.cfg.img_size, self.cfg.img_size), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        # [H, W, C] -> [C, H, W]
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> DualImageSample:
        fname = self.files[idx]
        lr1_path = os.path.join(self.lr1_dir, fname)
        lr2_path = os.path.join(self.lr2_dir, fname)
        hr_path = os.path.join(self.hr_dir, fname)

        lr1 = self._load_image(lr1_path)
        lr2 = self._load_image(lr2_path)
        hr = self._load_image(hr_path)

        return DualImageSample(lr1=lr1, lr2=lr2, hr=hr)


def create_splits(cfg: DataConfig) -> Tuple[Dataset, Dataset, Dataset]:
    full_dataset = WorldStratDualImageDataset(cfg, split="full")
    n_total = len(full_dataset)
    n_train = int(cfg.train_split * n_total)
    n_val = int(cfg.val_split * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(cfg.seed)
    return random_split(full_dataset, [n_train, n_val, n_test], generator=generator)

