# Wi-Fi CSI Activity Recognition: Cross-Environment Generalisation Study

## Abstract

This project investigates the **deployment cost of Wi-Fi Channel State Information (CSI) based human activity recognition** in previously unseen environments. Using the [ReWiS](https://github.com/niloobah/ReWiS) public dataset, which contains CSI recordings of four activity classes (walking, standing, jumping, and empty room) collected in three distinct indoor environments, we:

1. **Train a baseline CNN** on environment A1 and evaluate within that environment
2. **Quantify the accuracy lost** when deploying directly to unseen environments A2, A3
3. **Measure how much labelled target data** is needed to recover baseline performance via fine-tuning
4. **Compare fine-tuning vs training from scratch** to isolate the benefit of pre-training

The key deliverable is a **characteristic curve** relating recognition accuracy in a new environment to the volume of labelled data available, with statistical confidence intervals over multiple random seeds and sample draws.

## Project Structure

```
Final_year_project/
├── configs/
│   ├── __init__.py
│   └── config.py              # All experiment configurations (dataclasses)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading, CSIDataset, k-shot sampling
│   ├── model.py               # CSIActivityCNN architecture
│   ├── trainer.py             # Training loop, evaluation, metrics
│   ├── visualisation.py       # All plotting functions
│   ├── utils.py               # Utilities, ResultsAggregator, JSON encoding
│   └── experiments.py         # Experiment orchestration (all 4 experiments)
├── few_shot_datasets/         # Place dataset here (see below)
│   └── m1c4_PCA_80_300_extracted_3x4/
│       ├── train_A1/
│       │   ├── walk/, empty/, jump/, stand/
│       ├── test_A2/
│       │   ├── walk/, empty/, jump/, stand/
│       └── test_A3/
│           ├── walk/, empty/, jump/, stand/
├── models/                    # Saved model checkpoints
├── results/                   # Experiment results
│   ├── plots/                 # Generated plots
│   └── tables/                # Summary CSV tables
├── ReWiS-main/                # Original ReWiS repository (reference)
├── run_experiments.py          # Main entry point
├── download_data.py           # Dataset download helper
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

### 1. Install Dependencies

All dependencies should already be installed. If not:

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

The dataset must be downloaded from Google Drive:

**Option A: Automatic download**
```bash
pip install gdown
python download_data.py
```

**Option B: Manual download**
1. Go to: https://drive.google.com/drive/folders/1H-0GFOIHmHpHfdV-T5qv_diov_dCQxMS
2. Download the `Formatted_data_frames` folder
3. Find the folder `m1c4_PCA_80_300_extracted_3x4` inside it
4. Place it under `few_shot_datasets/` so the structure matches:
   ```
   few_shot_datasets/m1c4_PCA_80_300_extracted_3x4/train_A1/walk/*.mat
   ```

### 3. Verify Dataset
```bash
python download_data.py  # Will verify even if download fails
```

## Running Experiments

### Full Pipeline (Recommended)
Runs all 4 experiments sequentially:
```bash
python run_experiments.py
```

### Quick Mode (For Testing)
Fewer seeds, draws, and shot counts — good for verifying the pipeline works:
```bash
python run_experiments.py --quick
```

### Individual Experiments
```bash
# Experiment 1: Baseline (train on A1)
python run_experiments.py --baseline

# Experiment 2: Cross-environment (test on A2, A3)
python run_experiments.py --cross-env

# Experiment 3: Fine-tune on specific environment
python run_experiments.py --finetune A2

# Experiment 4: Train from scratch (control)
python run_experiments.py --scratch A2
```

### Additional Options
```bash
python run_experiments.py --seed 123          # Different random seed
python run_experiments.py --epochs 100        # Override epoch count
python run_experiments.py --no-cuda           # Force CPU
```

## Experiments

### Experiment 1: Baseline Training
- **Goal**: Establish upper-bound accuracy within the source environment
- **Method**: Train CNN on A1, evaluate on A1 validation set (85/15 split)
- **Output**: Baseline accuracy, training curves, confusion matrix, saved model

### Experiment 2: Cross-Environment Evaluation
- **Goal**: Quantify accuracy loss from environmental change
- **Method**: Evaluate A1-trained model directly on A2 and A3 (zero-shot)
- **Output**: Accuracy on each target environment, accuracy drop, confusion matrices

### Experiment 3: Fine-Tuning with Limited Data
- **Goal**: Measure how much new data is needed to recover performance
- **Method**: For each k ∈ {1, 2, 3, 5, 10, 15, 20, 30, 50} shots per class:
  - Sample k labelled examples per class from target environment
  - Fine-tune pre-trained model with low learning rate
  - Evaluate on remaining target data
  - Repeat across 5 seeds × 5 draws = 25 runs per k
- **Output**: Accuracy vs. shots curve with confidence intervals

### Experiment 4: From-Scratch Training (Control)
- **Goal**: Isolate the benefit contributed by pre-training
- **Method**: Same as Experiment 3, but model starts from random weights
- **Output**: Comparison curve showing fine-tuning advantage

## Model Architecture

**CSIActivityCNN** — A compact 4-block convolutional neural network:

| Layer | Output Shape | Parameters |
|-------|-------------|-----------|
| Conv2d(1, 64, 3) + BN + ReLU + MaxPool | (64, H/2, W/2) | ~640 |
| Conv2d(64, 64, 3) + BN + ReLU + MaxPool | (64, H/4, W/4) | ~36,992 |
| Conv2d(64, 128, 3) + BN + ReLU + MaxPool | (128, H/8, W/8) | ~74,112 |
| Conv2d(128, 128, 3) + BN + ReLU + AdaptiveAvgPool(4) | (128, 4, 4) | ~147,712 |
| Flatten → Linear(2048, 256) + ReLU + Dropout | (256) | ~524,544 |
| Linear(256, 4) | (4) | ~1,028 |

**Total: ~785K parameters** (compact enough for the low-data regime)

## Key Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Baseline epochs | 200 | With early stopping (patience=20) |
| Batch size | 16 | |
| Learning rate | 1e-3 | Adam optimiser |
| Fine-tune LR | 1e-4 | 10× smaller than baseline |
| Fine-tune epochs | 100 | With early stopping (patience=15) |
| Seeds | 5 | Random seeds for reproducibility |
| Sample draws | 5 | Random selections per seed |
| Shots per class | [1,2,3,5,10,15,20,30,50] | |

## Expected Outputs

After running the full pipeline, you will find:

### Plots (in `results/plots/`)
- `baseline/training_curves.png` — Loss and accuracy during training
- `baseline/confusion_matrix.png` — Confusion matrix on A1 validation
- `cross_env/confusion_matrix_A2.png` — Zero-shot on A2
- `cross_env/confusion_matrix_A3.png` — Zero-shot on A3
- `summary/cross_env_comparison.png` — Bar chart of all environments
- `summary/finetuning_curve_A2.png` — **THE key figure**
- `summary/finetuning_curve_A3.png` — Same for A3
- `summary/recovery_curve_A2.png` — Fraction of baseline recovered

### Tables (in `results/tables/`)
- `summary_results.csv` — Complete results table

### Data (in `results/`)
- `all_results.json` — Machine-readable results
- `baseline_results.json`, `cross_env_results.json`, etc.

## References

- Bahadori, N., et al. "ReWiS: Reliable Wi-Fi Sensing Through Few-Shot Multi-Antenna Multi-Receiver CSI Learning"
- Dataset: https://github.com/niloobah/ReWiS
