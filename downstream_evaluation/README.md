# Downstream Evaluation Module

This module is designed for evaluating pretrained PPG models on various downstream tasks using Linear Probing (LP).

## Directory Structure

- `eval_models/`: Store your models and checkpoints here (e.g., `.pt`, `.pth`, `.pkl`).
- `preprocessed_datasets/`: Root directory for downstream datasets.
- `results/`: Evaluation results (Excel reports).
- `eval_embeddings/`: Cached embeddings extracted from models.

## Dataset Preparation

Datasets should be placed in the `preprocessed_datasets/` directory. Each dataset should have its own folder containing `train.npz` and `test.npz` (or subdirectories for split data).

### Data Format (NPZ)
Each `.npz` file must contain:
- `signals` (or specified signal key): A numpy array of shape `(N, L)` or `(N, 1, L)`.
- `labels` (or specified label key): A numpy array of labels corresponding to the signals.

Example structure:
```text
preprocessed_datasets/
├── butppg/
│   ├── DBPData/
│   │   ├── train.npz
│   │   └── test.npz
│   └── ...
└── dalia/
    └── HRData/
        ├── train.npz
        └── test.npz
```

## Running Evaluation

### 1. Batch Evaluation
To run all models across all datasets and tasks defined in the configuration:
```bash
python run_all.py
```
This script will:
1. Extract embeddings for each model/dataset/task combination across available GPUs.
2. Train a linear probe (Logistic Regression for classification, Ridge for regression).
3. Calculate metrics (AUROC, ACC for classification; MAE, R2 for regression) with 95% Bootstrap Confidence Intervals.
4. Log results into an Excel report in the `results/` folder.

### 2. Manual Embedding Extraction
If you only want to extract embeddings:
```bash
python extract_emb.py
```

## Configuration

- **Models**: Edit `MODELS` list in `run_all.py` or `extract_emb.py`.
- **Datasets**: Edit `DATASETS` list in `run_all.py`.
- **GPUs**: Set `AVAILABLE_GPUS` in `run_all.py`.
