<div align="left">
  <h1>AnyPPG: An ECG-Guided PPG Foundation Model Trained on Over 100,000 Hours of Recordings for Holistic Health Profiling</h1>
  <p>
    <a href="https://arxiv.org/abs/2511.01747">
      <img src="https://img.shields.io/badge/arXiv-2511.01747-b31b1b.svg" alt="arXiv">
    </a>
  </p>
  <p><em>🚧 This work is ongoing - improvements, new releases, and further code optimization will follow.</em></p>
</div>


---

## 🩺 Overview

**AnyPPG** is a **photoplethysmography (PPG) foundation model** pretrained on **over 100,000 hours** of synchronized **PPG-ECG recordings** from **58,796 subjects**, using a CLIP-style contrastive alignment framework to learn physiologically meaningful representations.

AnyPPG demonstrates strong and versatile performance across a wide range of downstream tasks, including:
- **Conventional physiological analyses** across eight clinical and wearable datasets covering 15 downstream tasks, it achieves SOTA performance with best performance in 13 tasks. 
- **Large-scale phenome-wide disease detection** on the MC-MED dataset, achieving meaningful discriminative capability (AUC ≥ 0.70) for 307 phenotypes across 16 distinct phecode chapters, including 230 non-circulatory conditions such as dementia, chronic kidney disease, hyperkalemia, and glaucoma.

**Note on MC-MED:** The MC-MED disease diagnosis checkpoints, corresponding disease names, and phecode mappings will be updated in the future. The codebase is also undergoing further organization and optimization.

---

## 📂 Repository Structure

The repository is organized into the following main directories:

- **`preprocessing/`**: Scripts for data extraction and processing for various datasets (CFS, HSP, MC-MED, MESA, PulseDB).
- **`pretraining/`**: Implementations of pretraining frameworks including AnyPPG (CLIP-style), SimCLR, and BYOL.
- **`downstream_evaluation/`**: Tools and scripts for evaluating the pretrained models on downstream tasks using linear probing.
- **`load_anyppg/`**: A self-contained module for easily loading the pretrained AnyPPG model and weights.

---

## ⚙️ Getting Started

### 🧩 Installation

Clone the repository:
```bash
git clone https://github.com/Ngk03/AnyPPG.git
cd AnyPPG
```

### 🧠 Using the AnyPPG Encoder

The easiest way to use the pretrained model is via the `load_anyppg` directory. The pretrained checkpoint is located at `load_anyppg/anyppg_ckpt.pth`.

**Input requirements**:
- Sampling rate: 125 Hz
- Normalization: z-score normalization along the time axis
- Input shape: `(Batch Size, 1, Length)`

You can refer to `load_anyppg/demo_load_anyppg.py` for a working example.

```python
import torch
import sys
sys.path.append('load_anyppg')
from resnet1d import Net1D

# AnyPPG encoder configuration
anyppg_cfg = {
    "in_channels": 1,
    "base_filters": 64,
    "ratio": 1.0,
    "filter_list": [64, 160, 160, 400, 400, 512],
    "m_blocks_list": [2, 2, 2, 3, 3, 1],
    "kernel_size": 3,
    "stride": 2,
    "groups_width": 16,
    "use_bn": True,
    "use_do": True,
    "verbose": False,
}

# Initialize encoder
anyppg = Net1D(**anyppg_cfg)

# Load pretrained model weights
ckpt_path = "load_anyppg/anyppg_ckpt.pth"
state_dict = torch.load(ckpt_path, map_location="cpu")
anyppg.load_state_dict(state_dict)

# Example forward pass
x = torch.randn(1, 1, 1250) # 10 seconds at 125 Hz
output = anyppg(x)
print(f"Output shape: {output.shape}") # Expected: (1, 512)
```

### 📊 Downstream Usage

To evaluate on a downstream task (e.g., heart rate regression or disease classification), freeze the encoder and attach a simple linear head:

```python
import torch.nn as nn

# Freeze encoder
for param in anyppg.parameters():
    param.requires_grad = False

# Linear probing model
num_classes = 2 # Example for binary classification
linear_head = nn.Linear(512, num_classes)
model = nn.Sequential(anyppg, linear_head)
```

For full fine-tuning, simply unfreeze the encoder:
```python
for param in anyppg.parameters():
    param.requires_grad = True
```

---

## 📘 Citation

If you find AnyPPG useful in your research, please consider citing:
```bibtex
@article{nie2025anyppg,
  title   = {AnyPPG: An ECG-Guided PPG Foundation Model Trained on Over 100,000 Hours of Recordings for Holistic Health Profiling},
  author  = {Nie, Guangkun and Tang, Gongzheng and Xiao, Yujie and Li, Jun and Huang, Shun and Zhang, Deyun and Zhao, Qinghao and Hong, Shenda},
  journal = {arXiv preprint arXiv:2511.01747},
  year    = {2025}
}
```

### Acknowledgement
This work builds upon the open-source implementation from <https://github.com/hsd1503/resnet1d>.
