import os
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any


@dataclass
class DataConfig:
    worldstrat_root: str = "./worldstrat"  # root folder with LR/HR patches
    lr_dir_1: str = "lr_1"  # first LR view subdir
    lr_dir_2: str = "lr_2"  # second LR view subdir
    hr_dir: str = "hr"  # HR target subdir
    img_size: int = 256
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42


@dataclass
class TrainConfig:
    batch_size: int = 16
    num_workers: int = 4
    num_epochs: int = 100
    base_lr: float = 1e-4
    unlearning_lr: float = 1e-5
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    device: str = "cuda"


@dataclass
class LossConfig:
    mse_weight: float = 0.6   # α
    perceptual_weight: float = 0.3  # β
    edge_weight: float = 0.1  # γ


@dataclass
class AgenticConfig:
    niqe_threshold: float = 4.0
    piqe_threshold: float = 30.0
    max_unlearning_steps: int = 3
    # simple heuristic weights for selector decision
    w_brightness: float = 0.3
    w_entropy: float = 0.3
    w_cloud_ratio: float = 0.4


@dataclass
class ProjectConfig:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    loss: LossConfig = LossConfig()
    agentic: AgenticConfig = AgenticConfig()
    checkpoints_dir: str = "./checkpoints"
    results_dir: str = "./results"

    def to_dict(self) -> Dict[str, Any]:
        cfg = asdict(self)
        # flatten dataclasses
        cfg["data"] = asdict(self.data)
        cfg["train"] = asdict(self.train)
        cfg["loss"] = asdict(self.loss)
        cfg["agentic"] = asdict(self.agentic)
        return cfg


def ensure_dirs(cfg: ProjectConfig) -> None:
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

