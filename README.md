## Dual Image Super-Resolution with Agentic AI (DISR)

This folder contains an end-to-end implementation of the framework described in your project report:

- **Dual-image fusion** for satellite image super-resolution on WorldStrat.
- **Multiple SR models**: custom CNN, U-Net, VGG16-based CNN, and a **GAN-based SR generator**.
- **Agentic selector** that chooses the best model per scene.
- **Blind evaluation** using NIQE and PIQE.
- **Unlearning controller** that selectively fine-tunes layers when blind quality is poor.

### Folder layout

- `config.py` – central configuration (data paths, training and loss hyperparameters, agentic thresholds).
- `datasets.py` – dual-image WorldStrat-style dataset and train/val/test split logic.
- `models.py` – all core model definitions:
  - `UNetSR`, `VGGSRSimple`, `CustomSRNet`, `SRGANGenerator`, `SRGANDiscriminator`, and the `build_fused_tensor` helper.
- `losses.py` – composite loss combining **MSE + perceptual (VGG16) + edge loss**.
- `metrics.py` – PSNR, SSIM, NIQE, PIQE and a helper for full metric reports.
- `agentic_disr.py` – agentic SR loop:
  - scene statistics, model selection, blind evaluation, and **unlearning** controller.
  - optional SRGAN adversarial training step.
- `train.py` – end-to-end training script that:
  - builds all models, runs the agentic loop, and saves `checkpoints/agentic_disr_best.pth`.
- `inference.py` – helper to run inference with the trained agentic ensemble on pairs of LR images.
- `requirements.txt` – Python dependencies.

### Quick start

1. **Install dependencies** (ideally in a fresh virtualenv or conda env):

```bash
pip install -r requirements.txt
```

2. **Prepare data** in a WorldStrat-like folder structure:

```text
worldstrat/
  lr_1/
    00001.png
    ...
  lr_2/
    00001.png
    ...
  hr/
    00001.png
    ...
```

Update `worldstrat_root` in `config.py` if needed.

3. **Train the agentic DISR system**:

```bash
python train.py
```

This will:

- Train custom CNN, U-Net, VGG16-based, and GAN SR models together.
- Run the **agentic loop** with blind evaluation and unlearning.
- Save the best checkpoint to `checkpoints/agentic_disr_best.pth`.

4. **Run inference** on new LR image pairs:

Create two lists of LR image paths (`lr1_paths`, `lr2_paths`) and call:

```python
from inference import run_inference_on_folder

run_inference_on_folder(
    lr1_paths,
    lr2_paths,
    out_dir="./results/inference",
    state_dict_path="./checkpoints/agentic_disr_best.pth",
)
```

Generated HR images and logs will be written to `results/`.

### Git setup and push to an existing repository

Use these commands from this project folder (not from your home directory):

```bash
cd "C:/Users/AYUSH/Desktop/College/ProjectPhase/agentic_ai_final code"
git rev-parse --show-toplevel
```

If needed, initialize Git and make the first commit:

```bash
git init
git branch -M main
git add .
git commit -m "Initial project import with documentation"
```

Connect your existing remote repository:

```bash
git remote add origin <REPO_URL>
git remote -v
```

If the remote already has commits, merge histories first:

```bash
git pull origin main --allow-unrelated-histories
```

Then push:

```bash
git push -u origin main
```

Useful checks before pushing:

```bash
git status
git ls-files | Select-String -Pattern "\.env|\.venv|node_modules|__pycache__|\.cursor"
```

