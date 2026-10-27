"""
Wi-Fi CSI Human Activity & Movement Detector.

Run real-time or offline inference on single Wi-Fi CSI files (.mat).
Detects whether human movement is present (Walking, Jumping, Standing)
or if the room is empty (No Movement).

Usage:
    # Detect movement in a single file:
    python predict.py --file path/to/sample.mat

    # Run quick demo across sample activities:
    python predict.py --demo

    # Save a visual plot of the CSI heatmap:
    python predict.py --file path/to/sample.mat --plot
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Ensure writable matplotlib cache in sandbox environments
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import read_mat
from src.model import load_model
from configs.config import ExperimentConfig


CLASS_NAMES = ['walk', 'empty', 'jump', 'stand']
CLASS_DISPLAY = {
    'walk': 'Walking',
    'empty': 'Empty Room (No Movement)',
    'jump': 'Jumping',
    'stand': 'Standing',
}


def draw_bar(prob, length=24):
    filled = int(round(prob * length))
    return '#' * filled + '-' * (length - filled)


def predict_file(file_path: str, model, device: str = 'cpu', plot_heatmap: bool = False):
    """Predict activity and movement for a single CSI .mat file."""
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None

    # Load CSI data matrix
    data = read_mat(file_path)
    if data is None:
        print(f"[ERROR] Could not read valid CSI data from {file_path}")
        return None

    # Handle shape
    if data.ndim != 2:
        print(f"[ERROR] Expected 2D matrix (subcarriers, time), got shape {data.shape}")
        return None

    # Shape tensor into (1, 1, 80, 300)
    tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Movement determination
    if pred_class == 'empty':
        movement_status = "NO MOVEMENT (Room Empty)"
        movement_flag = False
    else:
        movement_status = f"MOVEMENT DETECTED ({CLASS_DISPLAY[pred_class]})"
        movement_flag = True

    # Print results
    print("\n" + "=" * 60)
    print("  WI-FI CSI MOVEMENT & ACTIVITY DETECTION")
    print("=" * 60)
    print(f"  Input File   : {os.path.basename(file_path)}")
    print(f"  Signal Shape : {data.shape[0]} subcarriers x {data.shape[1]} time packets")
    print("-" * 60)
    print(f"  STATUS       : >>> {movement_status} <<<")
    print(f"  Prediction   : {CLASS_DISPLAY[pred_class]} ({confidence * 100:.2f}% confidence)")
    print("-" * 60)
    print("  Class Probabilities:")
    for i, cls in enumerate(CLASS_NAMES):
        bar = draw_bar(probs[i])
        marker = " <--" if i == pred_idx else ""
        print(f"    {CLASS_DISPLAY[cls]:<26} [{bar}] {probs[i] * 100:5.1f}%{marker}")
    print("=" * 60)

    # Optional plot
    if plot_heatmap:
        try:
            import matplotlib.pyplot as plt
            save_name = Path(file_path).stem + "_csi_plot.png"
            plt.figure(figsize=(9, 4))
            plt.imshow(data, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(label='CSI Amplitude (PCA Denoised)')
            plt.title(f"Wi-Fi CSI: {movement_status} ({confidence*100:.1f}%)")
            plt.xlabel("Time Samples (300 packets)")
            plt.ylabel("Subcarrier Channels (80 PCA)")
            plt.tight_layout()
            plt.savefig(save_name, dpi=200)
            plt.close()
            print(f"  Heatmap saved: {save_name}\n")
        except Exception as e:
            print(f"  [Warning] Could not generate plot: {e}")

    return {
        'file': file_path,
        'prediction': pred_class,
        'display_name': CLASS_DISPLAY[pred_class],
        'movement_detected': movement_flag,
        'confidence': confidence,
        'probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }


def run_demo(model, device):
    """Run detection across a sample of each activity class."""
    base_dir = PROJECT_ROOT / "few_shot_datasets" / "m1c4_PCA_80_300_extracted_3x4" / "train_A1"

    if not base_dir.exists():
        print(f"[ERROR] Sample dataset directory not found at: {base_dir}")
        print("Run test_pipeline.py or download_data.py first.")
        return

    print("\n" + "=" * 70)
    print("  RUNNING LIVE DEMO: Detecting Movement Across All 4 Activities")
    print("=" * 70)

    for act in CLASS_NAMES:
        act_dir = base_dir / act
        if not act_dir.exists():
            continue
        files = list(act_dir.glob("*.mat"))
        if not files:
            continue
        sample_file = files[0]
        predict_file(str(sample_file), model, device)


def main():
    parser = argparse.ArgumentParser(description="Wi-Fi CSI Activity and Movement Detection")
    parser.add_argument("--file", type=str, help="Path to a CSI .mat file to analyze")
    parser.add_argument("--demo", action="store_true", help="Run automated demo on available sample files")
    parser.add_argument("--model", type=str, default=None, help="Path to custom model checkpoint (.pt)")
    parser.add_argument("--plot", action="store_true", help="Generate and save CSI spectrogram plot")
    args = parser.parse_args()

    config = ExperimentConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve model path
    model_path = args.model or str(config.models_dir / "baseline_A1.pt")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model checkpoint not found at: {model_path}")
        print("Please train the model first by running:")
        print("    python test_pipeline.py   (for fast synthetic training)")
        print("    python run_experiments.py --baseline  (for full dataset)")
        sys.exit(1)

    print(f"Loading trained CSI model from: {model_path} on {device}...")
    model = load_model(model_path, num_classes=4, device=device)

    if args.demo:
        run_demo(model, device)
    elif args.file:
        predict_file(args.file, model, device, plot_heatmap=args.plot)
    else:
        # Default: if no arg provided, run demo
        print("\nNo specific file provided. Running demonstration on available sample files...")
        run_demo(model, device)


if __name__ == "__main__":
    main()
